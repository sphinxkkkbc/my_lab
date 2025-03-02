import unittest
import torch
import numpy as np
import sys
import time
sys.path.append('/home/Code')
from src.player.tensor_env import TensorGame
from src.config.config import Playconfig

class TestTensorGame(unittest.TestCase):
    def setUp(self):
        self.config = Playconfig()
        self.config.R_limit = 7
        self.config.mul_size = [2, 2, 2]
        self.config.set_values = [-1, 0, 1]
        self.config.len_chain = 7
        self.game = TensorGame(self.config)

    def test_initialization(self):
        """测试游戏初始化"""
        self.assertIsNone(self.game.board)
        self.assertEqual(self.game.tensor, 0)
        self.assertIsNone(self.game.result)
        self.assertEqual(self.game.step_num, 0)
        self.assertEqual(self.game.limit, 7)
        self.assertEqual(self.game.size_tensor, [2, 2, 2])
        self.assertEqual(self.game.set, [-1, 0, 1])
        self.assertEqual(len(self.game.chain), 0)

    def test_set_tensor(self):
        """测试设置张量"""
        # 测试numpy数组
        np_tensor = np.random.randn(2, 2, 2)
        self.game.set_tensor(np_tensor)
        self.assertTrue(torch.is_tensor(self.game.tensor))
        
        # 测试torch张量
        torch_tensor = torch.randn(4, 4, 4)
        self.game.set_tensor(torch_tensor)
        self.assertTrue(torch.is_tensor(self.game.tensor))

    def test_legal_moves(self):
        time1 = time.time()
        """测试合法移动生成"""
        moves = self.game.legal_moves()
        print(len(moves))
        time2 = time.time()
        print(time2-time1)

    def test_step(self):
        """测试执行一步移动"""
        # 创建一个简单的移动
        u = (1, 0, 0, 0)
        v = (1, 0, 0, 0)
        w = (1, 0, 0, 0)
        action = (u, v, w)
        
        # 设置初始张量
        initial_tensor = torch.ones(4, 4, 4)
        self.game.set_tensor(initial_tensor)
        
        # 执行移动
        self.game.step(action)
        
        # 检查状态更新
        self.assertEqual(self.game.step_num, 1)
        self.assertEqual(len(self.game.chain), 1)
        self.assertTrue(len(self.game.element['u']) == 1)

    def test_game_over(self):
        """测试游戏结束条件"""
        # 测试步数超限
        self.game.step_num = self.game.limit + 1
        self.assertEqual(self.game.is_game_over(), -1)
        
        # 测试张量全零
        self.game.step_num = 5
        self.game.tensor = torch.zeros(4, 4, 4)
        self.assertEqual(self.game.is_game_over(), 2)  # 在步数限制内完成

    def test_canonical_input_tensor(self):
        """测试规范化输入张量"""
        # 执行几步移动来构建张量链
        u = (1, 0, 0, 0)
        v = (1, 0, 0, 0)
        w = (1, 0, 0, 0)
        action = (u, v, w)
        
        self.game.set_tensor(torch.ones(4, 4, 4))
        self.game.step(action)
        
        tensors, indices = self.game.canonical_input_tensor()
        self.assertEqual(len(tensors.shape), 4)  # batch, h, w, d
        self.assertTrue(torch.is_tensor(indices))

if __name__ == '__main__':
    unittest.main() 