from typing import Dict, Optional, Tuple, Union

import pathlib
import io
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from gymnasium import spaces

from stable_baselines3.common.logger import Logger
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_shape = np.sum([obs_space.shape for obs_space in env.observation_space.spaces.values()])
        self.fc1 = nn.Linear(obs_shape, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob


class Critic(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_shape = np.sum([obs_space.shape for obs_space in env.observation_space.spaces.values()])
        self.fc1 = nn.Linear(obs_shape + np.prod(env.action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def flatten_obs(obs, device):
    observation, ag, dg = obs['observation'], obs['achieved_goal'], obs['desired_goal']
    if isinstance(observation, np.ndarray):
        observation = torch.from_numpy(observation).to(device)
    if isinstance(ag, np.ndarray):
        ag = torch.from_numpy(ag).to(device)
    if isinstance(dg, np.ndarray):
        dg = torch.from_numpy(dg).to(device)
    return torch.cat([observation, ag, dg], dim=1).to(dtype=torch.float32)


class MEINSAC:
    """
    :param env: The Gym environment to learn from
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically
    :param use_her: whether to use hindsight experience replay (HER) by using the SB3 HerReplayBuffer
    """
    def __init__(
            self,
            env: GymEnv,
            learning_rate: float = 3e-4,
            buffer_size: int = 1_000_000,
            learning_starts: int = 100,
            batch_size: int = 256,
            tau: float = 0.005,
            gamma: float = 0.99,
            ent_coef: Union[str, float] = "auto",
            use_her: bool = False
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.tau = tau
        self.gamma = gamma

        self.env = env
        if isinstance(self.env.action_space, spaces.Box):
            assert np.all(
                np.isfinite(np.array([self.env.action_space.low, self.env.action_space.high]))
            ), "Continuous action space must have a finite lower and upper bound"

        # initialize replay buffer
        if use_her:
            self.replay_buffer = HerReplayBuffer(
                self.buffer_size,
                self.env.observation_space,
                self.env.action_space,
                env=self.env,
                device=self.device,
                n_envs=self.env.num_envs
            )
        else:
            self.replay_buffer = DictReplayBuffer(
                self.buffer_size,
                self.env.observation_space,
                self.env.action_space,
                device=self.device,
                n_envs=self.env.num_envs
            )

        self._create_actor_critic()

        self.ent_coef = ent_coef
        if self.ent_coef == "auto":
            self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))
            self.log_ent_coef = torch.zeros(1, device=self.device).requires_grad_(True)
            self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=self.learning_rate)
        else:
            self.ent_coef_tensor = torch.tensor(float(self.ent_coef), device=self.device)

        self.logger = None
        self._last_obs = None
        self.num_timesteps = 0
        self._n_updates = 0

    def _create_actor_critic(self) -> None:
        self.actor = Actor(self.env).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)

        self.crit_1 = Critic(self.env).to(self.device)
        self.crit_2 = Critic(self.env).to(self.device)
        self.crit_1_target = Critic(self.env).to(self.device)
        self.crit_2_target = Critic(self.env).to(self.device)
        self.crit_1_target.load_state_dict(self.crit_1.state_dict())
        self.crit_2_target.load_state_dict(self.crit_2.state_dict())
        self.critic_optimizer = torch.optim.Adam(list(self.crit_1.parameters()) + list(self.crit_2.parameters()),
                                                 lr=self.learning_rate)

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval=None,
    ):
        callback.init_callback(self)
        callback.on_training_start(locals(), globals())

        self._last_obs = self.env.reset()

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollout(callback=callback)

            if continue_training is False:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                self.train()

        callback.on_training_end()

        return self

    def collect_rollout(
            self,
            callback: BaseCallback
    ):
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :return: True if the training should continue, else False
        """
        # Select action randomly or according to policy
        if self.num_timesteps < self.learning_starts:
            actions = np.array([self.env.action_space.sample()])
        else:
            actions, _ = self.predict(self._last_obs)

        # perform action
        new_obs, rewards, dones, infos = self.env.step(actions)

        self.num_timesteps += self.env.num_envs

        # save data to replay buffer; handle `terminal_observation`
        real_next_obs = new_obs.copy()
        for idx, done in enumerate(dones):
            if done:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
        self.replay_buffer.add(self._last_obs, real_next_obs, actions, rewards, dones, infos)

        self._last_obs = new_obs

        # Only stop training if return value is False, not when it is None.
        if callback.on_step() is False:
            return False
        return True

    def train(self):
        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

        # Sample replay buffer
        replay_data = self.replay_buffer.sample(self.batch_size)
        observations = flatten_obs(replay_data.observations, self.device)
        next_observations = flatten_obs(replay_data.next_observations, self.device)

        # optimize entropy coefficient
        if self.ent_coef == "auto":
            with torch.no_grad():
                _, log_pi = self.actor.get_action(observations)
            ent_coef_loss = (-self.log_ent_coef * (log_pi + self.target_entropy)).mean()
            self.logger.record("train/ent_coef_loss", ent_coef_loss.item())

            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()
            ent_coef = self.log_ent_coef.exp().item()
        else:
            ent_coef = self.ent_coef_tensor
        self.logger.record("train/ent_coef", ent_coef)

        # train critic
        with torch.no_grad():
            next_state_actions, next_state_log_pi = self.actor.get_action(next_observations)
            crit_1_next_target = self.crit_1_target(next_observations, next_state_actions)
            crit_2_next_target = self.crit_2_target(next_observations, next_state_actions)
            min_crit_next_target = torch.min(crit_1_next_target, crit_2_next_target) - ent_coef * next_state_log_pi
            next_q_value = replay_data.rewards.flatten() + \
                           (1 - replay_data.dones.flatten()) * self.gamma * min_crit_next_target.flatten()

        crit_1_a_values = self.crit_1(observations, replay_data.actions).view(-1)
        crit_2_a_values = self.crit_2(observations, replay_data.actions).view(-1)
        crit_1_loss = F.mse_loss(crit_1_a_values, next_q_value)
        crit_2_loss = F.mse_loss(crit_2_a_values, next_q_value)
        crit_loss = 0.5 * (crit_1_loss + crit_2_loss)
        self.logger.record("train/critic_loss", crit_loss.item())

        self.critic_optimizer.zero_grad()
        crit_loss.backward()
        self.critic_optimizer.step()

        # train actor
        pi, log_pi = self.actor.get_action(observations)
        crit_1_pi = self.crit_1(observations, pi)
        crit_2_pi = self.crit_2(observations, pi)
        min_crit_pi = torch.min(crit_1_pi, crit_2_pi).view(-1)
        actor_loss = ((ent_coef * log_pi) - min_crit_pi).mean()
        self.logger.record("train/actor_loss", actor_loss.item())

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks with polyak update
        with torch.no_grad():
            for param, target_param in zip(self.crit_1.parameters(), self.crit_1_target.parameters()):
                target_param.data.mul_(1 - self.tau)
                torch.add(target_param.data, param.data, alpha=self.tau, out=target_param.data)
            for param, target_param in zip(self.crit_2.parameters(), self.crit_2_target.parameters()):
                target_param.data.mul_(1 - self.tau)
                torch.add(target_param.data, param.data, alpha=self.tau, out=target_param.data)

    def predict(
            self,
            obs: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, None]:
        """
        Get the policy action given an observation.

        :param obs: the input observation
        :return: the model's action
        """
        observation = flatten_obs(obs, self.device)
        action, _ = self.actor.get_action(observation)
        return action.detach().cpu().numpy(), None

    def save(self, path: Union[str, pathlib.Path, io.BufferedIOBase]):
        # Copy parameter list, so we don't mutate the original dict
        data = self.__dict__.copy()
        for to_exclude in ["logger", "env", "num_timesteps", "_n_updates", "_last_obs",
                           "replay_buffer", "actor", "crit_1", "crit_2", "crit_1_target", "crit_2_target"]:
            del data[to_exclude]
        # save network parameters
        data["_actor"] = self.actor.state_dict()
        data["_crit_1"] = self.crit_1.state_dict()
        data["_crit_2"] = self.crit_2.state_dict()
        torch.save(data, path)

    @classmethod
    def load(cls, path, env, **kwargs):
        model = cls(env=env, **kwargs)
        loaded_dict = torch.load(path)
        for k in loaded_dict:
            if k not in ["_actor", "_crit_1", "_crit_2"]:
                model.__dict__[k] = loaded_dict[k]
        # load network states
        model.actor.load_state_dict(loaded_dict["_actor"])
        model.crit_1.load_state_dict(loaded_dict["_crit_1"])
        model.crit_2.load_state_dict(loaded_dict["_crit_2"])
        model.crit_1_target.load_state_dict(loaded_dict["_crit_1"])
        model.crit_2_target.load_state_dict(loaded_dict["_crit_2"])
        return model

    def set_logger(self, logger: Logger) -> None:
        self.logger = logger

    def get_env(self):
        return self.env
