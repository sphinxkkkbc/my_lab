import threading
import torch
import numpy as np
from multiprocessing import Connection, Pipe

class ModelAPI:
    """
    模型API接口，用于处理预测请求和进程间通信
    """
    def __init__(self, model):
        """
        初始化API
        Args:
            model: 你的PyTorch模型实例
        """
        self.model = model
        self.model.eval()  # 设置为评估模式
        self.pipes = []  # 存储所有通信管道
        self.running = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def start(self):
        """启动API服务线程"""
        self.running = True
        self.thread = threading.Thread(target=self._predict_loop)
        self.thread.start()

    def create_pipe(self) -> Connection:
        """创建新的通信管道"""
        me, them = Pipe()
        self.pipes.append(me)
        return them

    def _predict_loop(self):
        """预测循环，处理所有管道上的请求"""
        while self.running:
            ready = connection.wait(self.pipes)  # 等待请求
            for pipe in ready:
                try:
                    state = pipe.recv()  # 接收状态数据
                    # 转换为PyTorch张量
                    state_tensor = torch.FloatTensor(state).to(self.device)
                    with torch.no_grad():
                        # 获取策略和价值预测
                        policy, value = self.model(state_tensor)
                        # 转换为numpy数组
                        policy = policy.cpu().numpy()
                        value = value.cpu().numpy()
                    pipe.send((policy, value))
                except EOFError:
                    self.pipes.remove(pipe)

    def stop(self):
        """停止API服务"""
        self.running = False
        self.thread.join()

class YourModel:
    def __init__(self, config):
        self.config = config
        self.model = None  # 你的PyTorch模型
        self.api = None

    def get_pipes(self, num=1):
        """获取指定数量的通信管道"""
        if self.api is None:
            self.api = ModelAPI(self.model)
            self.api.start()
        return [self.api.create_pipe() for _ in range(num)]

    def predict(self, state):
        """直接预测接口"""
        if self.api is None:
            self.api = ModelAPI(self.model)
            self.api.start()
        pipe = self.api.create_pipe()
        pipe.send(state)
        return pipe.recv()
    
if __name__ == "main":
        # 使用示例
    model = YourModel(config)
    # 创建通信管道
    pipes = model.get_pipes(num=4)  # 创建
    
    # 在其他进程中使用管道
    def worker(pipe):
        state = get_state()  # 获取游戏状态
        pipe.send(state)  # 发送状态
        policy, value = pipe.recv()  # 接收预测结果