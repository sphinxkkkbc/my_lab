import numpy as np
import torch

def generate_synthetic_demonstrations(S, R_limit, p_entry, N, random_seed,value_set = [-2, -1, 0, 1, 2]):
    """
    Tensor size S, maximum rank R_limit, factor entry probability distribution pentry, desired number of demonstrations N, random see
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    np.random.seed(random_seed)
    torch.set_default_tensor_type(torch.FloatTensor)
    results = []
    for n in range(N):
        factors = []
        U = torch.zeros((S[0]*S[1], R_limit), device=device)
        V = torch.zeros((S[1]*S[2], R_limit), device=device)
        W = torch.zeros((S[0]*S[2], R_limit), device=device)
        for r in range(R_limit):
            u = torch.tensor(np.random.choice(value_set, size=S[0]*S[1], p=p_entry)).float().to(device=device)
            v = torch.tensor(np.random.choice(value_set, size=S[1]*S[2], p=p_entry)).float().to(device=device)
            w = torch.tensor(np.random.choice(value_set, size=S[0]*S[2], p=p_entry)).float().to(device=device)
            # Check duplicates for u in U
            while any(torch.allclose(u, U[:, prev_r]) for prev_r in range(r)) or torch.all(u == 0) :
                u = torch.tensor(np.random.choice(value_set, size=S[0]*S[1], p=p_entry)).float().to(device=device)

            # Check duplicates for v in V
            while any(torch.allclose(v, V[:, prev_r]) for prev_r in range(r)) or torch.all(v == 0):
                v = torch.tensor(np.random.choice(value_set, size=S[1]*S[2], p=p_entry)).float().to(device=device)

            # Check duplicates for w in W
            while any(torch.allclose(w, W[:, prev_r]) for prev_r in range(r)) or torch.all(w == 0):
                w = torch.tensor(np.random.choice(value_set, size=S[0]*S[2], p=p_entry)).float().to(device=device)

            # Convert to torch tensors
            u = u.clone().detach().float().to(device=device)
            v = v.clone().detach().float().to(device=device)
            w = w.clone().detach().float().to(device=device)
            U[:, r] = u
            V[:, r] = v
            W[:, r] = w
        factors.append((U, V, W))
        J = torch.einsum('ir,jr,kr->ijk', U, V, W)
        results.append((J, factors))
        if (n + 1) % 200 == 0:
            print(f"Generated {n + 1}/{N} demonstrations")
    return results

def generate_demo(S, R_limit, p_entry, N, random_seed, value_set=[-1, 0, 1], len_chain=7):
    demo = generate_synthetic_demonstrations(S, R_limit, p_entry, N, random_seed, value_set)
    state_plane_tensor = []
    state_plane_index = []
    for i in range(len(demo)):
        original_tensor, factors = demo[i]
        factors_U, factors_V, factors_W = factors[0]
        tensor_chain = []
        index = []
        action = []
        reward = float('-inf')
        rank_one_tensor = [torch.einsum('i,j,k->ijk', factors_U[:,i], factors_V[:,i], factors_W[:,i]) for i in range(R_limit)]
        if R_limit == len_chain:
            #R_limit为7时,记录前6次action,tensor_chain长度为7
            #只能生成一组数据
            tensor_chain = torch.stack(reversed(rank_one_tensor))
            action = (factors_U[:,-1], factors_V[:,-1], factors_W[:,-1])
            index = torch.tensor([6,5,4,3,2,1])
            reward = 1.0
            state_plane_tensor.append((tensor_chain, index, action, reward))
        else:
            #R_limit大于7时，记录前7次action,tensor_chain长度为8
            #可以生成R_limit-len_chain-1组数据
            for i in range(R_limit - len_chain - 1): 
                if i == 0:
                    new_tensor = sum(rank_one_tensor[j] for j in range(i, i + len_chain))  
                    current_tensor = original_tensor - new_tensor       
                tensor_chain.append(current_tensor)
                tensor_chain.append(rank_one_tensor[i:i+len_chain][::-1]) 
                reversed_tensor_chain = []
                for item in tensor_chain:
                    if isinstance(item, list):  
                        reversed_tensor_chain.extend(item)  
                    else:
                        reversed_tensor_chain.append(item)  
                tensor_chain = torch.stack(reversed_tensor_chain, dim=0)
                action.append((factors_U[:,i+len_chain], factors_V[:,i+len_chain], factors_W[:,i+len_chain]))
                index.append(torch.tensor(range(i+len_chain,i,-1)))
                index = torch.cat(index, dim=0)
                reward = 1.0 if (i == R_limit - len_chain - 1) else 0.5
                print(current_tensor)
                current_tensor -= rank_one_tensor[i+len_chain]
                state_plane_tensor.append(tensor_chain)
                state_plane_index.append(index)
    return state_plane_tensor, state_plane_index, action, reward

def generate_basis_change_matrices(S, p_cob_entry, N_cob, random_seed):
    np.random.seed(random_seed)
    results = []
    for n in range(N_cob):
        P = np.zeros((S, S))
        L = np.zeros((S, S))
        for i in range(S):
            P[i, i] = np.random.choice([-1, 1])
            L[i, i] = np.random.choice([-1, 1])
        for i in range(S):
            for j in range(i):
                L[i, j] = np.random.choice([0 , -1 , 1],p=p_cob_entry)
            for j in range(i + 1, S):
                P[i, j] = np.random.choice([0 , -1 , 1],p=p_cob_entry)
        result_matrix = np.dot(P, L)
        results.append(result_matrix)
    return results
    