import sys
sys.path.append('/home/Code')
from  src.model.model_tensor import AlphaTensorNet
from src.genetation_demo import generate_demo
from src.config.config import Playconfig
import numpy as np
import torch
import time

demo = generate_demo(S=(2,2,2), R_limit=7, p_entry=[0.1, 0.8, 0.1], N=1, random_seed=0, value_set=[-1, 0, 1])
model = AlphaTensorNet(config=Playconfig)

def attention_test():
    print("Testing attention")
    input = torch.tensor(np.random.rand(2,4,512),dtype=torch.float)
    x = input[0].squeeze(0)
    y = input[1].squeeze(0)
    print(f"Input of Attention : x.shape:{x.shape},y.shape:{y.shape}")
    o = model.attention(x,y)  
    print(f"output.shape:{o.shape}")


def attentive_modes_test():
    print("Testing attentive_modes")
    input = torch.tensor(np.random.rand(3,4,4,512),dtype=torch.float)
    x1,x2,x3 = model.attentivemodes(input[0],input[1],input[2])
    print(f"x1.shape:{x1.shape}")


def torso_test():
    print("Testing torso")
    input = torch.tensor(np.random.rand(7,4,4,4),dtype=torch.float)
    s = torch.tensor([7,6,5,4,3,2,1],dtype=torch.float)
    e = model.torso(input,s)
    # print(f"e.shape:{e.shape}")

    return e

def policy_head_test():
    print("Testing policy_head")
    e = torso_test()
    a = torch.tensor(np.random.rand(6,4),dtype=torch.float)
    print(f"a.shape:{a.shape}")
    o,z = model.predict_action_logits(a, e)
    print(f"o.shape:{o.shape},z.shape:{z.shape}")

    return o,z

def value_head_test():
    print("Testing value_head")
    _, z = policy_head_test()
    input_valuehead = z
    output_valuehead = model.valueHead(input_valuehead)
    print(f"output_valuehead.shape:{output_valuehead.shape}")

    return output_valuehead

def sample_from_logits_test():  
    print("Testing sample_from_logits")
    a = torch.tensor(np.random.rand(3,2))
    print(a)
    b = model.sample_from_logits(a)
    print(b)

def policy_head_training_test():
    print("Testing policy_head_training")
    e = torso_test()
    N_logits = 7
    g = torch.randint(1,6,(1,6)).to(torch.int64).squeeze(0)
    o,z = model.policyhead_training(e,g,N_logits)
    print(o.shape,z.shape)

def policy_head_inference_test():
    print("Testing policy_head_inference")
    e = torso_test()
    N_steps = 1
    N_logits = 32
    a,p,z = model.policyhead_inference(e,N_steps,N_logits)
    print(a.shape,p.shape,z.shape)
    print(a)
    print(p)

if __name__ == "__main__":
    time1 = time.time()
    # attention_test()
    # attentive_modes_test() 
    # torso_test()
    # policy_head_test()
    # value_head_test()
    policy_head_training_test()
    # policy_head_inference_test()
    time2 = time.time()
    print(f"Time for test:{time2-time1}")


