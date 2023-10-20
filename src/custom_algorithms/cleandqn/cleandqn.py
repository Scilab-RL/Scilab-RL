import random
from typing import Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x)


class CLEANDQN:
    """
    Custom version of DQN adapted from CleanRL
    https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py
    https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy

    :param env: a gymnasium environment with discrete action space
    :param learning_rate: the learning rate of the optimizer
    :param start_e: the starting epsilon for exploration
    :param end_e: the final epsilon for exploration
    :param exploration_fraction: the fraction of the total steps it takes to go from start_e to end_e exploration
    :param learning_starts: start the training after learning_starts steps
    :param train_frequency: train the policy every train_frequency steps
    :param target_network_frequency: update the target network every target_network_frequency steps
    :param tau: target network smoothing coefficient
    :param gamma: the discount factor gamma
    :param batch_size: size of the batches sampled from the replay buffer
    :param buffer_size: size of the replay buffer memory
    """
    def __init__(self,
                 env: GymEnv,
                 learning_rate: float,
                 start_e: Union[float, int],
                 end_e: Union[float, int],
                 exploration_fraction: float,
                 learning_starts: int,
                 train_frequency: int,
                 target_network_frequency: int,
                 tau: float,
                 gamma: float,
                 batch_size: int,
                 buffer_size: int = 10_000):

        self.env = env
        assert isinstance(env.action_space, gym.spaces.Discrete), "only discrete action space is supported"

        self.start_e = start_e
        self.end_e = end_e
        self.exploration_fraction = exploration_fraction
        self.learning_starts = learning_starts
        self.train_frequency = train_frequency
        self.target_network_frequency = target_network_frequency
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size

        self.q_network = QNetwork(self.env).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.target_network = QNetwork(self.env).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.rb = ReplayBuffer(
            buffer_size,
            self.env.observation_space,
            self.env.action_space,
            device,
            handle_timeout_termination=False,
        )

        self.num_timesteps = 0
        self.n_updates = 0
        self.logger = None
        self.callback = None

    def learn(self, total_timesteps: int, callback, log_interval):
        """
        learn to get a good reward for the environment
        :param total_timesteps: the maximum number of timesteps to train the agent
        :param callback: a Callback or CallbackList to call every step, e.g. EvalCallback
        """
        self.callback = callback
        self.callback.init_callback(self)

        obs = self.env.reset()
        for global_step in range(total_timesteps):
            self.num_timesteps += 1

            epsilon = linear_schedule(self.start_e, self.end_e, int(self.exploration_fraction * total_timesteps),
                                      global_step)
            self.logger.record("rollout/exploration_rate", epsilon)

            actions, _ = self.predict(obs=obs, state=None, deterministic=False, episode_start=False, epsilon=epsilon)

            next_obs, rewards, done, infos = self.env.step(actions)  # VecEnv automatically resets

            real_next_obs = next_obs.copy()
            for idx, d in enumerate(done):
                if d:
                    real_next_obs[idx] = infos[idx]["terminal_observation"]
            self.rb.add(obs, real_next_obs, actions, rewards, done, infos)

            obs = next_obs

            if global_step > self.learning_starts:
                if global_step % self.train_frequency == 0:
                    self.train(global_step)

            if callback.on_step() is False:
                return

    def train(self, global_step):
        data = self.rb.sample(self.batch_size)
        with torch.no_grad():
            target_max, _ = self.target_network(data.next_observations).max(dim=1)
            td_target = data.rewards.flatten() + self.gamma * target_max * (1 - data.dones.flatten())
        old_val = self.q_network(data.observations).gather(1, data.actions).squeeze()
        loss = F.mse_loss(td_target, old_val)

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.n_updates += 1
        self.logger.record("train/n_updates", self.n_updates)
        self.logger.record_mean("train/loss", loss.item())
        # self.logger.record("train/q_values", old_val.mean().item())

        if global_step % self.target_network_frequency == 0:
            self.update_target_network()

    def update_target_network(self):
        for target_network_param, q_network_param in zip(self.target_network.parameters(),
                                                         self.q_network.parameters()):
            target_network_param.data.copy_(
                self.tau * q_network_param.data + (1.0 - self.tau) * target_network_param.data
            )

    def predict(self, obs, state, episode_start, deterministic, epsilon=0.):
        if random.random() < epsilon and not deterministic:
            actions = np.array([self.env.action_space.sample() for _ in range(self.env.num_envs)])
        else:
            q_values = self.q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        return actions, state

    def save(self, path):
        # Copy parameter list, so we don't mutate the original dict
        data = self.__dict__.copy()
        for to_exclude in ["logger", "env", "num_timesteps", "n_updates",
                           "rb", "q_network", "target_network", "callback"]:
            del data[to_exclude]
        # save network parameters
        data["state_dict"] = self.q_network.state_dict()
        torch.save(data, path)

    @classmethod
    def load(cls, path, env, **kwargs):
        model = cls(env=env, **kwargs)
        loaded_dict = torch.load(path)
        for k in loaded_dict:
            if k not in ["state_dict"]:
                model.__dict__[k] = loaded_dict[k]
        # load network states
        model.q_network.load_state_dict(loaded_dict["state_dict"])
        model.target_network.load_state_dict(loaded_dict["state_dict"])
        return model

    def set_logger(self, logger):
        self.logger = logger

    def get_env(self):
        return self.env
