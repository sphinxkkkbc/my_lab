from concurrent.futures import ProcessPoolExecutor
import datetime
from logging import getLogger
import torch
from glob import glob
import os
import sys
pythonpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(pythonpath)
from model.model_tensor import AlphaTensorNet,Quantile_loss,KL_divergence_loss
from player.player_tensor import TensorPlayer
from player.tensor_env import TensorGame
from genetation_demo import generate_demo
from model.model_helper import load_best_model_weight
from config.config import Playconfig,WorkerConfig,ResourceConfig
from collections import deque
from random import shuffle
# from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter

logger = getLogger(__name__)

# class TensorDataset(Dataset):
#     def __init__(self, states, policies, values):
#         """
#         初始化张量游戏数据集
        
#         Args:
#             states: 状态数据
#             policies: 策略数据
#             values: 价值数据
#         """
#         self.states = states
#         self.policies = policies
#         self.values = values
        
#     def __len__(self):
#         return len(self.states)
    
#     def __getitem__(self, idx):
#         return self.states[idx], self.policies[idx], self.values[idx]

class TensorWorker:
    def __init__(self,config:Playconfig):
        self.config = Playconfig
        self.buffers = {
            'replay': deque(maxlen=config.replay_buffer_size),
            'demo': deque(maxlen=config.synthetic_demos),
            'best': deque(maxlen=config.best_games_buffer_size)
        }
        self.dataset = deque(),deque(),deque()
        self.model = None
        self.executor = ProcessPoolExecutor(max_workers=self.config.trainer.cleaning_processes)

    def extend_synthetic_demos(self):
        p_entry = [0.1, 0.8, 0.1]  
        synthetic_demo = generate_demo(
            S=self.config.mul_size,
            R_limit=self.config.R_limit,
            p_entry=p_entry,
            N=self.config.synthetic_demos,
            random_seed=42,
        )
        for demo in synthetic_demo:
            self.buffers['demo'].extend(demo)

    def tensor_self_play(self, cur, init_tensor=None):
        """
        Play one game and add the play data to the buffer

        :param Config config: config for how to play
        :param list(Connection) cur: list of pipes to use to get a pipe to send observations to for getting
            predictions. One will be removed from this list during the game, then added back
        :return (TensorEnv,list((tensor,vector,list(float),list(float)): a tuple containing the final TensorEnv state and then a list
            of data to be appended to the SelfPlayWorker.buffer
        """
        pipes = cur.pop() # borrow
        env = TensorGame().reset()
        if init_tensor :
            env.set_tensor(init_tensor)
        data = []
        player = TensorPlayer(self.config, pipes=pipes)

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

        cur.append(pipes)
        return env

    def train(self):       
        # for i in range(step):
        #     self.tensor_self_play(self.load_model(best_model_path) if model_changed else model)
        #     self.step+=1
        #     self.fill_queue()
        #     self.train_epoch()
        #     self.backup_model()
        #     self.save_current_model_if_is_best()

        self.model = self.load_model()
        optimizer = torch.optim.AdamW(self.model.parameters(),lr=self.config.learning_rate,weight_decay=self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=self.config.lr_decay_steps,gamma=self.config.lr_decay)
        writer = SummaryWriter(log_dir='./log/tensorboard_file',filename_suffix=self.config.n_steps)
        self.extend_synthetic_demos()
        results = generate_demo(S=self.config.mul_size,R_limit=self.config.R_limit,p_entry=[0.1,0.8,0.1],N=self.config.n_steps)
        tensors = [result[0] for result in results]
        for step in range(self.config.n_steps): 
            #player reward错误，随step减小而不是固定值

            #model输入(chain,index) + 输出(policy distribution,value distribution)
            #model输出policy格式未确定

            #mcts-play输入未确定,如果随机可能出现重复列
            #mcts-play输出不对,应该与model的policy output适配,且policy和value一一对应   
            self.tensor_self_play(self.model,init_tensor=tensors[step])
            self.fill_tensor_queue(step=step)

            states, target_policies, target_values = self.collect_all_loaded_data()
            # for batch_idx, (states, target_policies, target_values) in enumerate(dataloader): 
            policy_loss, value_loss = self.model.train(states, target_policies, target_values)

            optimizer.zero_grad()
            policy_loss.backward()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            optimizer.step()
            scheduler.step()
            
            writer.add_scalar('Loss/policy_loss', policy_loss.item(), step)
            writer.add_scalar('Loss/value_loss', value_loss.item(), step)
            writer.add_scalar('Learning_rate', scheduler.get_last_lr()[0], step)
            writer.flush()

            self.save_current_model()

            print(f"Step {step} - Policy Loss: {policy_loss.item()}, Value Loss: {value_loss.item()}")
        writer.close()


    def fill_tensor_queue(self,step):
        futures = deque()
        index = ['replay','demo','best']
        self.dataset.clear()
        with ProcessPoolExecutor(max_workers=self.config.trainer.cleaning_processes) as executor:
            for i in range(self.config.trainer.cleaning_processes):
                logger.debug(f"loading data from {index[i]}")
                futures.append(executor.submit(self.load_data_from_buffer(i,step=step)))
        while futures and len(self.dataset[0]) < self.config.batch_size:
            for x,y in zip(self.dataset,futures.popleft().result()):
                x.extend(y)

    def update_buffer(self,buffer_index,data):
        if len(self.buffers[buffer_index]) < self.config.replay_buffer_size:
            self.buffers[buffer_index].extend(data)
        else:
            self.buffers[buffer_index].popleft()
            self.buffers[buffer_index].extend(data)

    def load_data_from_buffer(self,buffer_index,step):
        data_ratio = [0.0, 1.0, 0.0]
        if step > 1000:
            data_ratio = [0.1, 0.9, 0.0]
            if step > 10000:
                data_ratio = [0.7, 0.25, 0.05]

        buffer = self.buffers[buffer_index]
        size = self.config.batch_size * data_ratio[buffer_index]
        shuffle(buffer)
        return buffer[:size]

    def collect_all_loaded_data(self):
        """

        :return: a tuple containing the data in self.dataset, split into
        (state, policy, and value).
        """
        state_ary,policy_ary,value_ary=self.dataset

        state_tensor = torch.tensor(list(state_ary), dtype=torch.float32)
        policy_tensor = torch.tensor(list(policy_ary), dtype=torch.float32) 
        value_tensor = torch.tensor(list(value_ary), dtype=torch.float32)

        # dataset = TensorDataset(state_tensor, policy_tensor, value_tensor)
    
        # dataloader = DataLoader(
        #     dataset,
        #     batch_size=self.config.batch_size,
        #     shuffle=True,
        #     num_workers=self.config.worker.search_threads,
        #     pin_memory=True if torch.cuda.is_available() else False
        # )
        return state_tensor, policy_tensor, value_tensor

    def save_current_model(self):
        """
        Saves the current model as the next generation model to the appropriate directory
        """
        rc = self.config.resource
        model_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        model_dir = os.path.join(rc.next_generation_model_dir, rc.next_generation_model_dirname_tmpl % model_id)
        os.makedirs(model_dir, exist_ok=True)
        config_path = os.path.join(model_dir, rc.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
        self.model.save(config_path, weight_path)

    def load_model(self):
        """
        Loads the next generation model from the appropriate directory. If not found, loads
        the best known model.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AlphaTensorNet(c=512).to(device)
        rc = self.config.resource

        dirs = get_next_generation_model_dirs(rc)
        if not dirs:
            logger.debug("loading best model")
            if not load_best_model_weight(model):
                raise RuntimeError("Best model can not loaded!")
        else:
            latest_dir = dirs[-1]
            logger.debug("loading latest model")
            config_path = os.path.join(latest_dir, rc.next_generation_model_config_filename)
            weight_path = os.path.join(latest_dir, rc.next_generation_model_weight_filename)
            model.load(config_path, weight_path)
        return model

def get_next_generation_model_dirs(rc: ResourceConfig):
    dir_pattern = os.path.join(rc.next_generation_model_dir, rc.next_generation_model_dirname_tmpl % "*")
    dirs = list(sorted(glob(dir_pattern)))
    return dirs
