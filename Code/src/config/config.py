import os
class Playconfig:
    def __init__(self):
        self.R_limit = 100
        self.set_values = [-1,0,1]
        self.mul_size = [2,2,2]
        self.batch_size = 2
        self.n_simulations_before_50k = 200
        self.n_simulations_after_50k = 800
        self.replay_buffer_size = 100000
        self.synthetic_demos = 5000000
        self.best_games_buffer_size = 1000
        self.n_steps = 5000000
        self.gradient_clip = 4.0
        self.weight_decay = 1e-5
        self.learning_rate = 1e-4
        self.lr_decay = 0.1
        self.lr_decay_steps = 500000
        self.c_puct = 1.5
        self.noise_eps = 0.25
        self.dirichlet_alpha = 0.3
        self.virtual_loss = 3
        self.N_bar = 100
        self.len_chain = 7
        self.tau_decay_rate = 0.15

class WorkerConfig:
    def __init__(self):
        self.search_threads = 8

class ResourceConfig:
    def __init__(self):
        self.model_dir = 'F:/我的大学/大四/毕设/Code/src/pkg_model'
        self.model_best_config_path = os.path.join(self.model_dir, "model_best_config.json")
        self.model_best_weight_path = os.path.join(self.model_dir, "model_best_weight.h5")

        self.next_generation_model_dir = os.path.join(self.model_dir, "next_generation")
        self.next_generation_model_dirname_tmpl = "model_%s"
        self.next_generation_model_config_filename = "model_config.json"
        self.next_generation_model_weight_filename = "model_weight.h5"