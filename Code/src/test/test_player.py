import unittest
import torch
import numpy as np
import sys
sys.path.append('/home/Code')
from src.player.player_tensor import TensorPlayer, get_legal_moves, ActionStats, VisitStats
from src.player.tensor_env import TensorGame
from src.config.config import Playconfig

class TestTensorPlayer(unittest.TestCase):
    def setUp(self):
        self.config = Playconfig()
        self.config.R_limit = 7
        self.config.mul_size = [2, 2, 2]
        self.config.set_values = [-1, 0, 1]
        self.config.simulation_num_per_move = 100
        self.config.search_threads = 4
        self.config.c_puct = 1.25
        self.config.noise_eps = 0.25
        self.config.dirichlet_alpha = 0.3
        self.config.virtual_loss = 3
        self.config.N_bar = 50
        
        self.player = TensorPlayer(play_config=self.config)
        self.game = TensorGame(self.config)

    def test_get_legal_moves(self):
        """测试获取合法移动"""
        len_moves, moves, move_indices = get_legal_moves(
            self.config.set_values,
            self.config.mul_size
        )
        self.assertTrue(len_moves > 0)
        self.assertEqual(len(moves), len_moves)
        self.assertEqual(len(move_indices), len_moves)

    def test_action_stats(self):
        """测试动作统计"""
        stats = ActionStats()
        self.assertEqual(stats.n, 0)
        self.assertEqual(stats.w, 0)
        self.assertEqual(stats.q, 0)
        self.assertEqual(stats.p, 0)

    def test_visit_stats(self):
        """测试访问统计"""
        stats = VisitStats()
        self.assertEqual(stats.sum_visits, 0)
        self.assertEqual(len(stats.a), 0)

    def test_reset(self):
        """测试重置"""
        self.player.reset()
        self.assertEqual(len(self.player.tree), 0)

    def test_select_action(self):
        """测试动作选择"""
        # 设置初始状态
        self.game.set_tensor(torch.ones(2, 2, 2))
        
        # 模拟MCTS搜索
        state = self.game.get_state()
        self.player.tree[state] = VisitStats()
        
        # 测试动作选择
        action = self.player.select_action_q_and_u(self.game, is_root_node=True)
        self.assertIsNotNone(action)
        self.assertEqual(len(action), 3)  # (u, v, w)

    def test_policy_calculation(self):
        """测试策略计算"""
        # 设置初始状态
        self.game.set_tensor(torch.ones(2, 2, 2))
        
        # 添加一些访问统计
        state = self.game.get_state()
        self.player.tree[state] = VisitStats()
        stats = self.player.tree[state]
        
        # 获取一个合法移动并添加访问记录
        _, moves, _ = get_legal_moves(self.config.set_values, self.config.mul_size)
        action = moves[0]
        stats.a[action] = ActionStats()
        stats.a[action].n = 10
        
        # 计算策略
        policy = self.player.calc_policy(self.game)
        self.assertTrue(isinstance(policy, np.ndarray))
        self.assertEqual(sum(policy), 1.0)

    def test_temperature_application(self):
        """测试温度参数应用"""
        policy = np.array([0.5, 0.3, 0.2])
        modified_policy = self.player.apply_temperature(policy, turn=0)
        self.assertEqual(sum(modified_policy), 1.0)

    def test_game_finish(self):
        """测试游戏结束处理"""
        # 添加一些移动记录
        self.player.move = [
            [(torch.zeros(3,3,3), torch.tensor([1])), [0.5, 0.3, 0.2], 0.1]
        ]
        
        # 测试游戏结束时的更新
        self.player.finish_game(1)  # 胜利
        self.assertEqual(len(self.player.move[0]), 4)  # 应该添加了结果
        self.assertEqual(self.player.move[0][-1], 1)  # 最后一个元素应该是结果

    def test_tensor_self_play(self, model, init_tensor=None):
        """
        Play one game and add the play data to the buffer

        :param Config config: config for how to play
        :param list(Connection) cur: list of pipes to use to get a pipe to send observations to for getting
            predictions. One will be removed from this list during the game, then added back
        :return (TensorEnv,list((tensor,vector,list(float),list(float)): a tuple containing the final TensorEnv state and then a list
            of data to be appended to the SelfPlayWorker.buffer
        """
        env = TensorGame().reset()
        if init_tensor :
            env.set_tensor(init_tensor)
        data = []
        player = TensorPlayer(self.config)

        while not env.done:
            action = player.action(env)
            env.step(action)

        for i in range(len(player.move)):
            data.append(player.move[i])

        player.finish_game(env.result)
        
        if env.result == 2:
            self.update_buffer('best',data)
        else:
            self.update_buffer('replay',data)

        return env

if __name__ == '__main__':
    unittest.main() 