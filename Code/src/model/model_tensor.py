import sys
sys.path.append('/home/Code')
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger
from src.config.config import Playconfig,WorkerConfig,ResourceConfig

logger = getLogger(__name__)
    
class AlphaTensorNet(nn.Module):
    def __init__(self,config):
        super(AlphaTensorNet, self).__init__()
        self.config = config
        self.model = None  
        self.digest = None
        self.api = None

    def train(self, x, s, g_action, g_value,N_logits, c=512):
        """
        Model training function.

        :param x: (T, S, S, S)  # Input tensor
        :param s: (s,)  # Scalar features
        :param g_action: (N_steps,)  # Target actions
        :param g_value: (n,)  # Target values
        :param c: int  # Feature dimension
        :return: (float, float)  # Policy loss and value loss
        """
        e = self.torso(x,s,c)
        o,z1 = self.policyhead_training(e,g_action,N_logits)
        l_policy = torch.sum(nn.CrossEntropyLoss(o,g_action))
        q = self.valueHead(z1)
        l_value = Quantile_loss(q,g_value)
        return l_policy,l_value

    def inference(self, x, s, c, N_samples, N_steps, N_logits):    
        """
        Model inference function.

        :param x: (T, S, S, S)  # Input tensor
        :param s: (s,)  # Scalar features
        :param c: int  # Feature dimension
        :param N_samples: int  # Number of samples
        :param N_steps: int  # Number of steps
        :param N_logits: int  # Number of logits
        :return: tuple  # (Sampled actions, Probabilities, Value estimate)
        """
        e = self.torso(x,s,c)         # e : R(3*S*S,c)
        a,p,z1 = self.policyhead_inference(e,N_steps,N_logits,N_samples)        # a : R(N_logits,N_samples,N_steps)   p : [0,1]^R(N_samples)   z : R(2048)
        q = self.valueHead(z1)
        q = self.valueRiskManagement(q)
        return a,p,q


    def policyhead_training(self, e, g, N_logits):
        """
        Policy head behaviour at training time. It returns the logits (and the embeddings of the first step).

        :param e: (m, c)  # Encoded features
        :param g: (N_steps,)  # Target action sequence
        :return o: (N_steps, N_logits)  # Action logits
        :return z: (N_steps, N_features * N_heads)  # Intermediate features
        """
        shifted_g = self.shifted(g) 
        onehot_shifted_g = torch.nn.functional.one_hot(shifted_g, num_classes=N_logits).float()
        # Training by teacher-forcing
        o, z = self.predict_action_logits(onehot_shifted_g, e)
        return o,z


    def policyhead_inference(self, e, N_steps, N_logits, N_samples=32):
        """
        Policy head behaviour at inference time. It returns the sampled action, and its estimated probability (and the embeddings of the first step)

        e: (m, c)  # Encoded features
        Returns:
        :a: (N_samples, N_steps)  # Sampled action sequence
        :p: (N_samples,)  # Sequence probability
        :z: (N_steps, N_features * N_heads)  # Intermediate features
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        a = torch.zeros(N_samples, N_steps, dtype=torch.long, device=device)
        p = torch.ones(N_samples, device=device)
        
        for s in range(N_samples):
            for i in range(N_steps):
                one_hot_a = torch.nn.functional.one_hot(a[s], num_classes=N_logits).float()
                logits, z = self.predict_action_logits(one_hot_a, e, is_training=False)
                probs = torch.softmax(logits, dim=-1)
                action, policy = self.sample_from_logits(probs[i])
                a[s,i] = action
                p[s] *= policy.item()
        
        return a, p, z
    
    def attention(self, x, y, causal_mask=False, Num_Head=16, d=32, w=4):
        """
        Attention module.

        :param x: (batch_size, N_x, c_1)  # Query tensor
        :param y: (batch_size, N_y, c_2)  # Key-value tensor
        :param causal_mask: bool  # Whether to use causal masking
        :param Num_Head: int  # Number of attention heads
        :param d: int  # Attention dimension
        :param w: int  # Feed-forward expansion factor
        :return: (batch_size, N_x, c_1)  # Attention output
        """
        c_1 = x.shape[-1]
        c_2 = y.shape[-1]
        N_x = x.shape[-2]
        N_y = y.shape[-2]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'  
    
        LayerNorm_x = nn.LayerNorm(c_1).to(device)
        LayerNorm_y = nn.LayerNorm(c_2).to(device)
        q_linear = nn.Linear(c_1, Num_Head * d).to(device)
        k_linear = nn.Linear(c_2, Num_Head * d).to(device)
        v_linear = nn.Linear(c_2, Num_Head * d).to(device)
        output_linear = nn.Linear(d * Num_Head, c_1).to(device)
        
        x_norm = LayerNorm_x(x.to(device))
        # print(f"x_norm.shape : {x_norm.shape}")
        y_norm = LayerNorm_y(y.to(device))
        
        q = q_linear(x_norm).view(-1, Num_Head, N_x, d).squeeze(0)
        k = k_linear(y_norm).view(-1, Num_Head, N_y, d).squeeze(0)
        v = v_linear(y_norm).view(-1, Num_Head, N_y, d).squeeze(0)
        a = torch.matmul(q, k.transpose(-2, -1)) / (d ** 0.5)
        # print(f"q.shape : {q.shape},k.shape : {k.shape},v.shape : {v.shape}")
        # print(f"a.shape : {a.shape}")
        if causal_mask:
            mask = torch.triu(torch.ones(N_x, N_y), diagonal=1).bool().to(device)
            #print(f"mask.shape : {mask.shape}")
            mask = mask.unsqueeze(0).unsqueeze(0)
            a_masked = a * mask           
        else:   
            a_masked = a
        
        o = torch.matmul(a_masked, v)
        # print(f"o.shape : {o.shape}")
        o = o.reshape(x.shape[0], -1, d * Num_Head).to(device)
        o = torch.squeeze(o, 1)
        # print(f"o.shape : {o.shape}")
        x = x.to(device) + output_linear(o)

        # Dense Block
        dense_norm = nn.LayerNorm(c_1).to(device)
        dense1 = nn.Linear(c_1, c_1 * w).to(device)
        dense2 = nn.Linear(c_1 * w, c_1).to(device)
        gelu = nn.GELU()
        
        x = x + dense2(gelu(dense1(dense_norm(x))))
        
        return x.clone().detach()

    def sample_from_logits(self, logit):
        sample = torch.multinomial(logit, num_samples=1)
        prob = logit[sample]
        return sample, prob


    def attentivemodes(self, x_1, x_2, x_3):
        """
        Inter-modal attention computation.

        :param x_1: (S, S, c)  # First modality
        :param x_2: (S, S, c)  # Second modality
        :param x_3: (S, S, c)  # Third modality
        :return: list  # Updated three modalities
        """
        S = x_1.shape[0]
        # c = x_1.shape[-1] 
        g = [x_1, x_2, x_3]

        for m, n in [(0,1), (2,0), (1,2)]:
            g_m = g[m] 
            g_n = g[n]
            
            for i in range(S):
                current_m = g_m[i:i+1]  # Shape: (1, S, c)  
                current_n = g_n[i:i+1]  # Shape: (1, S, c)
                
                a = torch.cat([current_m, current_n], dim=1)  # Shape: (1, 2S, c)
                a = a.unsqueeze(0)  # Shape: (1, 1, 2S, c)
                
                c_out = self.attention(a, a)  # Shape: (1, 1, 2S, c)
                # print(f"c_out.shape : {c_out.shape}")
                c_out = c_out.squeeze(0).squeeze(0)  # Shape: (2S, c)
                g[m][i] = c_out[:S]  
                g[n][i] = c_out[S:]  
        [x1, x2, x3] = g  
        return x1, x2, x3


    def torso(self, x, s, c=512):
        """
        Backbone network.

        :param x: (T, S, S, S)  # Input tensor
        :param s: (s,)  # Scalar features
        :param c: int  # Output feature dimension
        :return: (3*S*S, c)  # Encoded features
        """
        T = x.shape[0]  
        S = x.shape[1] 
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        x_1 = x.permute(1, 2, 3, 0).reshape(S, S, S*T)
        x_2 = x.permute(3, 1, 2, 0).reshape(S, S, S*T)
        x_3 = x.permute(2, 3, 1, 0).reshape(S, S, S*T)

        g = [x_1, x_2, x_3]

        for i in range(3):
            p = torch.zeros(S, S, 1, device=device)
            if s is not None:
                linear = nn.Linear(s.shape[0], S*S).to(device)
                p = linear(s.to(device)).reshape(S, S, 1)
            g[i] = torch.cat([g[i].to(device), p], dim=-1)
            linear = nn.Linear(g[i].shape[-1], c).to(device)
            g[i] = linear(g[i])

        [x_1, x_2, x_3] = g

        for _ in range(8):
            x_1, x_2, x_3 = self.attentivemodes(x_1, x_2, x_3)
        
        e = torch.cat([x_1.reshape(-1, c), x_2.reshape(-1, c), x_3.reshape(-1, c)], dim=0)

        return e

    def predict_action_logits(self, a, e, N_features=64, N_heads=32, N_layers=2, is_training=True):
        """
        Predict action logits.

        :param a: (batch_size, seq_length)  # Action sequence
        :param e: (m, c)  # Encoded features
        :param N_features: int  # Number of features
        :param N_heads: int  # Number of attention heads
        :param N_layers: int  # Number of layers
        :param is_training: bool  # Whether in training mode
        :return: tuple  # (Action logits, Intermediate features)
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        N_logits = a.shape[-1]
        # print(f"N_logits : {N_logits}") 
        input_linear = nn.Linear(a.shape[-1], N_features * N_heads).to(device)
        output_linear = nn.Linear(N_features * N_heads, N_logits).to(device)  

        x = input_linear(a.to(device))  
        # print(f"x.shape : {x.shape}")
        pos_enc = LearnablePositionalEncoding(x.shape[1], N_features * N_heads).to(device)
        x = pos_enc(x)
        # print(f"x.shape : {x.shape}")
        for i in range(N_layers):
            # Layer normalization
            layer_norm = nn.LayerNorm(N_features * N_heads).to(device)
            x = layer_norm(x)
            # print(f"x.shape : {x.shape}")
            # Causal self attention
            c = self.attention(x, x, causal_mask=True, Num_Head=N_heads)
            # print(f"c.shape : {c.shape}")
            if is_training:
                dropout = nn.Dropout(p=0.1).to(device)
                c = dropout(c)
            x = x + c
            
            # Layer normalization
            layer_norm = nn.LayerNorm(N_features * N_heads).to(device)
            x = layer_norm(x)
            # Cross attention with encoded features
            # if e.dim() == 2:
            #     e = e.unsqueeze(0)
            c = self.attention(x, e, causal_mask=True, Num_Head=N_heads)
            if is_training:
                c = dropout(c)
            x = x + c
        
        o = output_linear(nn.functional.relu(x))  
        return o, x[0]

    def shifted(self, g):
        shifted_g = torch.roll(g, shifts=1)
        shifted_g[0] = 0 
        return shifted_g

    def valueHead(self, x, n=8):
        """
        Value head network.

        :param x: (batch_size, dim)  # Input features
        :param n: int  # Output dimension
        :return: (batch_size, n)  # Value predictions
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        layers = []
        input_dim = x.shape[-1]
        
        for _ in range(3):
            layers.extend([
                nn.Linear(input_dim, 512),
                nn.ReLU()
            ])
            input_dim = 512
        
        layers.append(nn.Linear(512, n))
        value_net = nn.Sequential(*layers).to(device)
        
        return value_net(x.to(device))

    def valueRiskManagement(self, q,u_q = 0.75):
        """
        Value risk management.

        :param q: (n,)  # Quantile predictions
        :param u_q: float  # Quantile threshold
        :return: float  # Risk-adjusted value
        """
        n = q.shape[0]
        j = [u_q*n]             # j : [1,...,n]
        return torch.mean(q[j:])

    def save(self, config_path, weight_path):
        """
        Save model configuration and weights.

        :param config_path: str  # Configuration file path
        :param weight_path: str  # Weight file path
        """
        torch.save(self.state_dict(), weight_path)
        with open(config_path, 'w') as f:
            json.dump({
                'c': self.c,
                'N_features': self.N_features,
                'N_heads': self.N_heads,
                'N_layers': self.N_layers
            }, f)

def Quantile_loss(q,g,delta = 1):
    """
    Calculate the quantile loss used in value head.

    :param q: (n,)  # Predicted quantile values at equally spaced intervals
    :param g: (n,)  # Ground truth values
    :param delta: float  # Delta parameter for Huber loss
    :return: float  # Quantile loss value
    """
    n = q.shape[0]

    def huber_loss(input, delta):
        abs_input = torch.abs(input)
        quadratic = torch.min(abs_input, torch.tensor(delta))
        linear = abs_input - quadratic
        loss = 0.5 * quadratic**2 + delta * linear
        return loss
    
    # Getting the n quantiles,Tau
    tau = [(0.5 + i)/n for i in range(n)]
    d = g - q
    h = huber_loss(d,delta)
    k = torch.abs(tau - (d < 0).float())
    return torch.mean(torch.mul(k,h))

def KL_divergence_loss(pred_policies, target_policies, is_prob=True):
    """
    Calculate KL divergence loss.

    :param pred_policies: (batch_size, n)  # Predicted policy distribution
    :param target_policies: (batch_size, n)  # Target policy distribution
    :param is_prob: bool  # Whether input is probability distribution
    :return: float  # KL divergence value
    """
    #ensure the input is probability
    if  is_prob:
        pred_probs = pred_policies
        target_probs = target_policies
    else:
        pred_probs = F.softmax(pred_policies, dim=-1)
        target_probs = F.softmax(target_policies, dim=-1)
        
    kl_div = F.kl_div(pred_probs.log(), target_probs, reduction='batchmean', log_target=False  )
    return kl_div

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Initialize learnable positional encoding.

        :param d_model: int  # Model dimension
        :param max_len: int  # Maximum sequence length
        """
        super(LearnablePositionalEncoding, self).__init__()
        # Embedding layer (max_len, d_model)
        self.position_embeddings = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(0)
        # position index (seq_len, 1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        #print(self.position_embeddings(position_ids).shape)
        # Convert position indices to positional encoding through Embedding layer and add to input tensor
        return x + self.position_embeddings(position_ids)
