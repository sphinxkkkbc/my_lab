import numpy as np
import torch
from player.player_tensor import TensorPlayer
import random
import math

def update_policy(self, policy):
        self.prior_policy = policy
        if self.prior_policy is not None:
            all_actions = self.board.get_legal_moves()
            sorted_actions = [action for _, action in 
                            sorted(zip(self.prior_policy, all_actions), 
                            key=lambda x: x[0], reverse=True)]
            self.untried_actions = sorted_actions

def get_action_probs(self, temp=1.0):
        visits = np.array([child.visits for child in self.children])
        if temp == 0:  # 选择访问次数最多的动作
            action_probs = np.zeros_like(visits, dtype=np.float32)
            action_probs[np.argmax(visits)] = 1.0
            return action_probs
        else:  # 根据访问次数计算概率
            visits = np.power(visits, 1.0/temp)
            probs = visits / np.sum(visits)
            return probs

def is_fully_expanded(self):
    return len(self.untried_actions) == 0

def best_child(self, c1=1.25, c2=19652):
    """
    选择最佳子节点
    使用神经网络预测的先验策略(self.prior_policy)作为pi参数
    """
    if not self.children:
        return None, 0
    
    total_visits = sum(child.visits for child in self.children)
    if total_visits == 0:
        return random.choice(self.children), 1.0
    
    average_visits = total_visits / len(self.children)
    
    # 计算每个子节点的tau值
    taus = []
    for child in self.children:
        if child.visits > average_visits:
            tau = np.log(child.visits) / average_visits
        else:
            tau = 1.0
        taus.append(tau)
    
    # 计算改进的pi参数
    improved_pi_params = []
    for child, tau in zip(self.children, taus):
        if self.visits == child.visits:  # 避免除以0
            improved_pi = 1.0
        else:
            try:
                improved_pi = math.pow(child.visits, 1/tau) / math.pow(self.visits - child.visits, 1/tau)
            except (ValueError, ZeroDivisionError):
                improved_pi = 1.0
        improved_pi_params.append(improved_pi)
    
    # 计算选择权重
    choices_weights = []
    for i, (child, improved_pi) in enumerate(zip(self.children, improved_pi_params)):
        if child.visits == 0:
            q_value = 0
        else:
            q_value = child.value / child.visits
        
        # 使用神经网络的先验策略
        prior_prob = self.prior_policy[i] if self.prior_policy is not None else 1.0/len(self.children)
        
        exploration = math.sqrt(total_visits - child.visits) / (1 + child.visits)
        log_term = math.log((total_visits - child.visits + c2 + 1) / c2)
        
        # 使用先验策略代替固定的pi_param
        weight = q_value + prior_prob * exploration * (c1 + log_term)
        choices_weights.append(weight)
    
    # 选择权重最大的子节点
    best_index = choices_weights.index(max(choices_weights))
    return self.children[best_index], improved_pi_params[best_index]

def expand(self):
    if not self.untried_actions:
        return None
        
    action = self.untried_actions.pop(0)  # 取出第一个未尝试的动作
    new_board = self.board.copy()  # 创建游戏状态的副本
    new_board.move(action)  # 执行动作
    
    child_node = TensorPlayer(new_board, parent=self)
    self.children.append(child_node)
    return child_node

def rollout(self):
    while not self.board.is_game_over():
        ####### possible moves undefined #######
        self.board.move()

    return

class MCTS:
    def __init__(self, iterations=500, model=None):
        self.iterations = iterations
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def get_policy_value(self, state):
        """使用神经网络获取策略和价值"""
        if self.model is None:
            return None, None
            
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                tensor_state = torch.FloatTensor(state).to(self.device)
            elif isinstance(state, torch.Tensor):
                tensor_state = state.to(self.device)
            else:
                raise TypeError("State must be either numpy array or torch.Tensor")
                
            tensor_state = tensor_state.unsqueeze(0)
            policy, value = self.model.get_policy_value(
                tensor_state, 
                ################ list scalar ----> time index undefined ###############
                torch.zeros(1, device=self.device)
            )
            print(f"policy.shape : {policy.shape},value.shape : {value.shape}")
            # 取value的平均值作为标量返回
            return policy.squeeze(0).cpu().numpy(), value.item()
    
    def search(self, game_state):
        """执行MCTS搜索并返回改进后的动作概率和价值"""
        self.root = TensorPlayer(game_state)
        
        # 使用神经网络评估根节点
        policy, value = self.get_policy_value(self.root.get_state())
        self.root.update_policy(policy)
        
        print(f"policy_network.shape : {policy},value_network.shape : {value}")

        for _ in range(self.iterations):
            node = self.root
            # Selection
            while node.is_fully_expanded() and not node.board.is_game_over():
                node, _ = node.best_child()
            
            # Expansion
            if not node.board.is_game_over():
                node = node.expand()
                # 使用神经网络评估
                policy, value = self.get_policy_value(node.get_state())
                if policy is not None:
                    node.update_policy(policy)
                    node.backpropagate(value)
                else:
                    result = node.rollout()
                    node.backpropagate(result)
        
        # 获取改进后的策略
        improved_policy = np.zeros(len(self.root.board.get_legal_moves()))
        for i, child in enumerate(self.root.children):
            _, pi = self.root.best_child()  # 获取改进后的pi值
            improved_policy[i] = pi
        
        # 归一化策略
        if np.sum(improved_policy) > 0:
            improved_policy = improved_policy / np.sum(improved_policy)
        
        return improved_policy, self.root.value / self.root.visits