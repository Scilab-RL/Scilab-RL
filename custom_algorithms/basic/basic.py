import torch as th
import numpy as np


def prepare_obs(obs):
    return th.cat([th.tensor(o.flatten(), dtype=th.float32) for o in obs.values()])


class BASIC:
    """
    This is the most basic algorithm that works with our framework.
    """
    # possible todos:
    # TODO use separate env for evaluation
    # TODO policy saving and loading
    # TODO nn backprop
    # TODO do not use the callback AT ALL (modify train.py)
    def __init__(self, env, net_arch=None, noise_factor=0.1):
        self.env = env
        self.logger = None
        self.num_timesteps = 0
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._last_obs = self.env.reset()
        self.nn = self._setup_nn(net_arch)
        self.noise_factor = noise_factor
        self.eval_freq = 0
        self.n_eval_episodes = 0

    def set_logger(self, logger):
        self.logger = logger

    def learn(self, total_timesteps, callback, log_interval):
        callback = callback[0]
        self.eval_freq = callback.eval_freq
        self.n_eval_episodes = callback.n_eval_episodes
        while self.num_timesteps < total_timesteps:
            action = self._get_action(self._last_obs, deterministic=False)
            obs, rewards, done, info = self.env.step(action)
            self._last_obs = obs
            self.num_timesteps += 1
            if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
                self.evaluate()

    def evaluate(self):
        successful_episodes = 0
        for _ in range(self.n_eval_episodes):
            done = False
            while not done:
                action = self._get_action(self._last_obs, deterministic=False) # TODO true
                obs, reward, done, _info = self.env.step(action)
                self._last_obs = obs
                success = _info['is_success']
                if success:
                    successful_episodes += 1
                    done = True
                    self.env.reset()
        self.logger.record('test/success_rate', successful_episodes / self.n_eval_episodes)
        self.logger.dump(step=self.num_timesteps)
        # TODO early stopping

    def load(self, path):
        pass

    def save(self, path):
        pass

    def _get_action(self, obs, deterministic):
        obs = prepare_obs(obs)
        action = self.nn(obs).detach().numpy()
        if not deterministic:
            action += self.noise_factor * (np.random.normal(size=len(action))-0.5)
        return action

    def _setup_nn(self, net_arch):
        input_dim = np.sum([space.shape for space in self.observation_space.spaces.values()])
        output_dim = self.action_space.shape[0]
        if net_arch is None:
            return th.nn.Sequential(th.nn.Linear(input_dim, output_dim))
        modules = [th.nn.Linear(input_dim, net_arch[0]), th.nn.ReLU()]
        for i in range(len(net_arch) - 1):
            modules.append(th.nn.Linear(net_arch[i], net_arch[i + 1]))
            modules.append(th.nn.ReLU())
        last_layer_dim = net_arch[-1]
        modules.append(th.nn.Linear(last_layer_dim, output_dim))
        nn = th.nn.Sequential(*modules)
        return nn
