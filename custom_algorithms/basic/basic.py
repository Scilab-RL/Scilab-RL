import torch as th
import numpy as np


def prepare_obs(obs):
    return th.cat([th.tensor(o.flatten(), dtype=th.float32) for o in obs.values()])


class BASIC:
    """
    This is the most basic algorithm that works with our framework.
    """
    def __init__(self, env, net_arch=None, noise_factor=0.1):
        self.env = env
        self.logger = None
        self.num_timesteps = 0
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._last_obs = self.env.reset()
        self.nn = self._setup_nn(net_arch)
        self.noise_factor = noise_factor

    def set_logger(self, logger):
        self.logger = logger

    def learn(self, total_timesteps, callback, log_interval):
        callback = callback[0]
        callback.init_callback(self)
        while self.num_timesteps < total_timesteps:
            action, _ = self.predict(self._last_obs, state=None, deterministic=False)
            obs, rewards, done, info = self.env.step(action)
            self._last_obs = obs
            self.num_timesteps += 1
            if callback.on_step() is False:
                return

    def predict(self, obs, state, deterministic):
        obs = prepare_obs(obs)
        action = self.nn(obs).detach().numpy()
        if not deterministic:
            action += self.noise_factor * (np.random.normal(size=len(action))-0.5)
        return action, state

    def get_env(self):
        return self.env

    def load(self, path):
        pass

    def save(self, path):
        pass

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
