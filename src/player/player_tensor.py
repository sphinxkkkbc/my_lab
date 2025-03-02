from collections import defaultdict
import sys
sys.path.append('F:\\我的大学\\大四\\毕设\\Code\\src')
from player.tensor_env import TensorGame
from concurrent.futures import ThreadPoolExecutor
import itertools
from threading import Lock
import numpy as np

class ActionStats:
    def __init__(self):  
        self.n = 0
        self.w = 0
        self.q = 0
        self.p = 0

class VisitStats:
    def __init__(self):
        self.a = defaultdict(ActionStats)
        self.sum_visits = 0

def get_legal_moves(set_values,size):    
    u_combinations = list(itertools.product(set_values, repeat=size[0]*size[1]))
    v_combinations = list(itertools.product(set_values, repeat=size[1]*size[2]))
    w_combinations = list(itertools.product(set_values, repeat=size[2]*size[0]))
    u_moves = []
    v_moves = []
    w_moves = []
    # Generate valid moves
    # Generate all possible combinations without zero vectors
    for u in u_combinations:
        if any(u):  # Checks if any element in u is non-zero
            for v in v_combinations:
                if any(v):
                    for w in w_combinations:
                        if any(w):
                            u_moves.append(u)
                            v_moves.append(v)
                            w_moves.append(w)
    
    moves = list(zip(u_moves, v_moves, w_moves))
    len_moves = len(moves)
    # Create an index mapping for each legal move 
    move_indices = {move : idx for idx, move in enumerate(moves)}
    return len_moves,moves,move_indices

#u,v,w在搜索时出现一次之后就不再出现,避免重复
class TensorPlayer:
    def __init__(self, pipes=None,play_config=None):
        self.move = []
        self.tree = defaultdict(ActionStats)
        self.limit = play_config.R_limit
        self.move_n,self.moves,self.move_lookup = get_legal_moves(play_config.set_values, play_config.mul_size)
        self.pipe_pool = pipes
        self.node_lock = defaultdict(Lock)
        self.playconfig = play_config

    def reset(self):
        self.tree = defaultdict(VisitStats)

    def search_moves(self,env):
        """
        Looks at all the possible moves using the AlphaTensor MCTS algorithm
        and finds the highest value possible move. Does so using multiple threads to get multiple
        estimates from the AlphaTensor MCTS algorithm so we can pick the best.

        :param AlphaTensorEnv env: env to search for moves within
        :return (float,float): the maximum value of all values predicted by each thread,
            and the first value that was predicted.
        """
        futures = []
        with ThreadPoolExecutor(max_workers=self.play_config.search_threads) as executor:
            for _ in range(self.play_config.simulation_num_per_move):
                futures.append(executor.submit(self.search_my_move,env=env.copy(),is_root_node=True))

        vals = [f.result() for f in futures]
        return np.max(vals), vals[0], vals # vals[0] is kind of racy

    def search_my_move(self,env: TensorGame, is_root_node=False):
        """
        Q, V is value for this Player.
        P is value for the player of next_player

        This method searches for possible moves, adds them to a search tree, and eventually returns the
        best move that was found during the search.

        :param TensorGame env: environment in which to search for the move
        :param boolean is_root_node: whether this is the root node of the search.
        :return float: value of the move. This is calculated by getting a prediction
            from the value network.
        """
        if env.done:
            return env.result

        state = state_key(env)

        with self.node_lock[state]:
            if state not in self.tree:
                leaf_p, leaf_v = self.expand_and_evaluate(env)
                self.tree[state].p = leaf_p
                return leaf_v # I'm returning everything from the POV of side to move

            # SELECT STEP
            action_t = self.select_action_q_and_u(env, is_root_node)

            virtual_loss = self.play_config.virtual_loss

            my_visit_stats = self.tree[state]
            my_stats = my_visit_stats.a[action_t]

            my_visit_stats.sum_n += virtual_loss
            my_stats.n += virtual_loss  
            my_stats.w += -virtual_loss
            my_stats.q = my_stats.w / my_stats.n

        env.step(action_t)
        leaf_v = self.search_my_move(env)  # next move from enemy POV
        #leaf_v = -leaf_v

        # BACKUP STEP
        # on returning search path
        # update: N, W, Q
        with self.node_lock[state]:
            my_visit_stats.sum_n += -virtual_loss + 1
            my_stats.n += -virtual_loss + 1
            my_stats.w += virtual_loss + leaf_v
            my_stats.q = my_stats.w / my_stats.n

        return leaf_v

    def action(self,env,can_stop=True):
        """
        Figures out the next best move
        within the specified environment and returns a string describing the action to take.

        :param ChessEnv env: environment in which to figure out the action
        :param boolean can_stop: whether we are allowed to take no action (return None)
        :return: None if no action should be taken (indicating a resign). Otherwise, returns a tuple
            indicating the action to take
        """
        self.reset()

        # for tl in range(self.play_config.thinking_loop):
        
        #self.search_moves(env)
        root_value, naked_value, distribution_value = self.search_moves(env)

        #policy = self.calc_policy(env)

        improved_policy = self.cal_policy_and_apply_temperature(env)
        my_action = int(np.random.choice(range(self.labels_n), p = improved_policy))

        if can_stop and env.result != 0:
            # noinspection PyTypeChecker
            return None
        else:
            tensor_chain,index = env.canonical_input_tensor()
            self.move.append([(tensor_chain, index), list(improved_policy),distribution_value])
            #self.move.append([tensor_chain, index, list(improved_policy)])
            return self.labels[my_action]

    def predict(self,state_planes):
        """
        Gets a prediction from the policy and value network

        :param state_planes: the observation state represented as planes
        :return (float,float): policy (prior probability of taking the action leading to this state)
            and value network (value of the state) prediction for this state.
        """
        pipe = self.pipe_pool.pop()
        pipe.send(state_planes)
        ret = pipe.recv()
        self.pipe_pool.append(pipe)
        return ret

    def expand_and_evaluate(self, env):
        """ expand new leaf, this is called only once per state
        this is called with state locked
        insert P(a|s), return leaf_v

        This gets a prediction for the policy and value of the state within the given env
        :return (float, float): the policy and value predictions for this state
        """
        state_planes = env.canonical_input_tensor()

        leaf_p, leaf_v = self.predict(state_planes)
        # these are canonical policy and value 

        return leaf_p, leaf_v
    
    def select_action_q_and_u(self, env, is_root_node, c1=1.25, c2=19652):
        """
        Picks the next action to explore using the AlphaTensor MCTS algorithm.

        Picks based on the action which maximizes the maximum action value
        (ActionStats.q) + an upper confidence bound on that action.

        :param Environment env: env to look for the next moves within
        :param is_root_node: whether this is for the root node of the MCTS search.
        :return tensor tuple: the move to explore
        """
        # this method is called with state locked
        state = state_key(env)

        my_visitstats = self.tree[state]

        if my_visitstats.p is not None: #push p to edges
            tot_p = 1e-8
            for mov in env.legal_moves:
                mov_p = my_visitstats.p[self.move_lookup[mov]]
                my_visitstats.a[mov].p = mov_p
                tot_p += mov_p
            for a_s in my_visitstats.a.values():
                a_s.p /= tot_p
            my_visitstats.p = None

        xx_ = np.sqrt(my_visitstats.sum_n + 1)  # sqrt of sum(N(s, b); for all b)

        e = self.play_config.noise_eps
        c_puct = self.play_config.c_puct
        dir_alpha = self.play_config.dirichlet_alpha

        c_puct = c1 + (np.sqrt((my_visitstats.sum_n + c2 + 1)/c2))

        best_s = -999
        best_a = None
        if is_root_node:
            noise = np.random.dirichlet([dir_alpha] * len(my_visitstats.a))
        
        i = 0
        for action, a_s in my_visitstats.a.items():
            p_ = a_s.p
            if is_root_node:
                p_ = (1-e) * p_ + e * noise[i]
                i += 1
            b = a_s.q + c_puct * p_ * xx_ / (1 + a_s.n)
            if b > best_s:
                best_s = b
                best_a = action

        return best_a

    def cal_policy_and_apply_temperature(self, env):
        '''
        Calculate the policy based on the visit counts and apply the temperature to it.

        :param env: the environment to calculate the policy for
        :return: the policy, with the temperature applied to it
        '''
        state = state_key(env)
        my_visitstats = self.tree[state]
        n_bar = self.playconfig.N_bar
        tau = np.log(my_visitstats.sum_n)/np.log(n_bar) if my_visitstats.sum_n > n_bar else 1

        policy = np.zeros(self.labels_n)
        for action, a_s in my_visitstats.a.items():
            policy[self.move_lookup[action]] = np.power(a_s.n,1/tau)

        policy /= np.sum(policy)
        return policy

    def apply_temperature(self, policy, turn):
        """
        Applies a random fluctuation to probability of choosing various actions
        :param policy: list of probabilities of taking each action
        :param turn: number of turns that have occurred in the game so far
        :return: policy, randomly perturbed based on the temperature. High temp = more perturbation. Low temp
            = less.
        """
        tau = np.power(self.play_config.tau_decay_rate, turn + 1)
        if tau < 0.1:
            tau = 0
        if tau == 0:
            action = np.argmax(policy)
            ret = np.zeros(self.labels_n)
            ret[action] = 1.0
            return ret
        else:
            ret = np.power(policy, 1/tau)
            ret /= np.sum(ret)
            return ret

    def calc_policy(self, env):
        """calc π(a|s0)
        :return list(float): a list of probabilities of taking each action, calculated based on visit counts.
        """
        state = state_key(env)
        my_visitstats = self.tree[state]
        policy = np.zeros(self.labels_n)
        for action, a_s in my_visitstats.a.items():
            policy[self.move_lookup[action]] = a_s.n

        policy /= np.sum(policy)
        return policy

    def finish_game(self, z):
        """
        When game is done, updates the value of all past moves based on the result.

        :param self:
        :param z: win=1, lose=-1
        :return:
        """
        for move in self.move:  # add this game winner result to all past moves.
            move += [z]

def state_key(env):
    return env.get_state