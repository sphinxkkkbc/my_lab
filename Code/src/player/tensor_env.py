import torch
import numpy as np
import copy
import sys
import itertools
sys.path.append('..')
# from player.player_tensor import get_legal_moves

class TensorGame:
    def __init__(self,play_config=None):
        self.board = None
        self.tensor = 0
        self.result = None
        self.step_num = 0
        self.limit = play_config.R_limit
        self.size_tensor = play_config.mul_size
        self.set = play_config.set_values
        self.len_chain = play_config.len_chain
        self.chain = []
        self.element = {
            'u': [],
            'v': [],
            'w': []
        }

    def reset(self):
        """
        Resets to begin a new game
        :return TensorGame: self
        """
        #! self.board = TensorGame() in __init__ is wrong !#
        self.board = TensorGame()
        self.result = None
        return self
    
    def legal_moves(self):
        # Generate all possible combinations
        u_combinations = list(itertools.product(self.set, repeat=self.size_tensor[0]*self.size_tensor[1]))
        v_combinations = list(itertools.product(self.set, repeat=self.size_tensor[1]*self.size_tensor[2]))
        w_combinations = list(itertools.product(self.set, repeat=self.size_tensor[2]*self.size_tensor[0]))
        print(len(u_combinations),len(v_combinations),len(w_combinations))
        # Filter out combinations that were already used
        for u in u_combinations:
            while any(u == e for e in self.element['u']) or np.all(u == 0):
                u_combinations.remove(u)
        for v in v_combinations:
            while any(v == e for e in self.element['v']) or np.all(v == 0):
                v_combinations.remove(v)
        for w in w_combinations:
            while any(w == e for e in self.element['w']) or np.all(w == 0):
                w_combinations.remove(w)        
        moves = []
        for u in u_combinations:
            for v in v_combinations:
                for w in w_combinations:
                    moves.append([u,v,w])

        return moves
        
    
    def update(self, tensor):
        """
        Like reset, but resets the position to whatever was supplied for board
        :param tensor: tensor to reset to
        :return TensorGame: self
        """
        #! wrong defined !#
        self.board = TensorGame(tensor)
        self.result = None
        return self

    def is_game_over(self):
        if self.step_num > self.limit:
            self.result = -1
        elif torch.all(self.tensor == 0):
            self.result = 1
            if self.step_num < self.limit:
                self.result = 2
        else:
            self.result = None
        return self.result


    def get_state(self):
        return self.tensor

    def set_tensor(self, tensor):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(tensor, np.ndarray):
            self.tensor = torch.from_numpy(tensor).float().to(device)
        elif isinstance(tensor, torch.Tensor):
            self.tensor = tensor.to(device)
        else:
            raise TypeError("Tensor must be either numpy array or torch.Tensor")
    
    def step(self, action: tuple, check_over = True):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        """

        Takes an action and updates the game state

        :param tuple action: action to take
        :param boolean check_over: whether to check if game is over
        """
        if check_over and action is None:
            self.result = -1
            return 

        if self.is_game_over():
            return None
        
        if isinstance(action, tuple):
            u, v, w = action
        
        #update tensor
        if self.tensor is not None:
            if isinstance(self.tensor, np.ndarray):
                self.tensor = torch.from_numpy(self.tensor).float().to(self.device)
            
            u_tensor = torch.tensor(u)
            v_tensor = torch.tensor(v)
            w_tensor = torch.tensor(w)
            rank_one_tensor = torch.einsum('i,j,k->ijk', u_tensor, v_tensor, w_tensor)
            self.tensor = self.tensor.to(device) - rank_one_tensor.to(device)
            self.element['u'].append(u)
            self.element['v'].append(v)
            self.element['w'].append(w)

        self.step_num += 1

        if not torch.all(self.tensor == 0):
            if len(self.chain) >= self.len_chain:
                self.chain.pop(0)  
            self.chain.append((rank_one_tensor, self.step_num))  
            
        if check_over and self.result != 0:
            self.is_game_over()

    def deltamove(self,tensor_next):
        _,_,moves = self.get_legal_moves()
        env = self.copy()
        for move,index in moves:
            env.step(move, check_over=False)
            tensor = env.get_state()
            if tensor == tensor_next:
                return move,index

    def copy(self):
        env = copy.copy(self)
        env.board = copy.copy(self.board)
        return env
    
    def canonical_input_tensor(self):
        """
        Returns a canonical version of the tensor chain
        """
        # Split tensors and indices from the chain
        tensors, indices = zip(*self.chain)
        # Stack tensors along a new dimension and convert indices to a vector
        return torch.stack(tensors), torch.tensor(indices)