from typing import Dict, Optional, Tuple, Union
from copy import deepcopy

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
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer


class Actor(nn.Module):
    def __init__(self, env, action_scale_factor=1.0):
        self.action_scale_factor = action_scale_factor
        super().__init__()
        if isinstance(env.observation_space, spaces.Dict):
            obs_shape = np.sum([obs_space.shape for obs_space in env.observation_space.spaces.values()])
            self.flatten = True
        else:
            obs_shape = np.array(env.observation_space.shape).prod()
            self.flatten = False
        self.fc1 = nn.Linear(obs_shape, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc_action = nn.Linear(256, np.prod(env.action_space.shape))
        # action rescaling
        action_scale = torch.tensor((env.action_space.high - env.action_space.low) / 2.0 * self.action_scale_factor, dtype=torch.float32)
        action_bias = torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)

        self.register_buffer(
            "action_scale", action_scale
        )
        self.register_buffer(
            "action_bias", action_bias
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action = torch.tanh(self.fc_action(x))
        return action * self.action_scale + self.action_bias

class Critic(nn.Module):
    def __init__(self, env):
        super().__init__()
        if isinstance(env.observation_space, spaces.Dict):
            obs_shape = np.sum([obs_space.shape for obs_space in env.observation_space.spaces.values()])
            self.flatten = True
        else:
            obs_shape = np.array(env.observation_space.shape).prod()
            self.flatten = False
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

class CriticEnsemble(nn.Module):
    def __init__(self, env, n_critics: int):
        super().__init__()
        self._critics = nn.ModuleList(
            [
                Critic(env)
                for _ in range(n_critics)
            ]
        )

    def forward(self, x, a):
        return torch.stack([critic(x, a) for critic in self._critics])
    

def flatten_obs(obs, device):
    observation, ag, dg = obs['observation'], obs['achieved_goal'], obs['desired_goal']
    if isinstance(observation, np.ndarray):
        observation = torch.from_numpy(observation).to(device)
    if isinstance(ag, np.ndarray):
        ag = torch.from_numpy(ag).to(device)
    if isinstance(dg, np.ndarray):
        dg = torch.from_numpy(dg).to(device)
    return torch.cat([observation, ag, dg], dim=1).to(dtype=torch.float32)


class CLEANTD3:
    """
    A one-file version of DDPG derived from the stable-baselines3 versions of DDPG.
    :param env: The Gym environment to learn from
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param action_noise: the action noise type (None by default), this can help for hard exploration problem.
    :param policy_delay: Policy and target networks will only be updated once every policy_delay steps
        per training steps. The Q values will be updated policy_delay more often (update every training step).
    :param target_noise_std: Standard deviation of Gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: Limit for absolute value of target policy smoothing noise.
    :param use_her: whether to use hindsight experience replay (HER) by using the SB3 HerReplayBuffer
    """
    def __init__(
            self,
            env: GymEnv,
            learning_rate: float = 1e-3,
            buffer_size: int = 1_000_000,
            learning_starts: int = 1000,
            batch_size: int = 256,
            tau: float = 0.005,
            gamma: float = 0.99,
            action_noise_std: float = 0.0,
            policy_delay: int = 2,
            target_noise_std: float = 0.2,
            noise_clip: float = 0.5,
            action_scale_factor: float = 1.0,
            n_critics: int = 2,
            use_her: bool = True,
            log_obs_step: bool = False,
            log_act_step: bool = False,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.action_scale_factor = action_scale_factor
        self.policy_delay = policy_delay
        self.target_noise_std = target_noise_std
        self.noise_clip = noise_clip
        self.action_noise_std = action_noise_std
        self.log_obs_step = log_obs_step
        self.log_act_step = log_act_step
        self.learning_starts = learning_starts
        self.n_critics = n_critics

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
            buffer_class = DictReplayBuffer if isinstance(self.env.observation_space, spaces.Dict) else ReplayBuffer
            self.replay_buffer = buffer_class(
                self.buffer_size,
                self.env.observation_space,
                self.env.action_space,
                device=self.device,
                n_envs=self.env.num_envs
            )
        self.flatten_obs = isinstance(self.env.observation_space, spaces.Dict)
        self._create_actor_critic()

        self.logger = None
        self._last_obs = None
        self.num_timesteps = 0
        self.episode_steps = 0
        self._n_updates = 0

        self.action_low = torch.tensor(self.env.action_space.low, device=self.device, dtype=torch.float32)
        self.action_high = torch.tensor(self.env.action_space.high, device=self.device, dtype=torch.float32)


    def _create_actor_critic(self) -> None:
        self.actor = Actor(self.env, self.action_scale_factor).to(self.device)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic = CriticEnsemble(self.env, self.n_critics).to(self.device)
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        self.noise = lambda: np.random.normal(0, self.action_noise_std, size=self.env.action_space.shape)

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval=None,
    ):
        callback.init_callback(self)
        callback.on_training_start(locals(), globals())

        self._last_obs = self.env.reset()
        self.episode_steps = 0
        while self.num_timesteps < total_timesteps:
            continue_training = self.step_env(callback=callback)

            if continue_training is False:
                break
            
            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                self.train()

        callback.on_training_end()

        return self

    def step_env(
            self,
            callback: BaseCallback
    ):
        """
        Take one step in the environment and store the transition in a ``ReplayBuffer``.
        In the stable-baselines3 version, this function is called "collect_rollouts" and it is possible to take
        multiple steps in the environment, but here we hard-code it to one step for simplicity.

        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :return: True if the training should continue, else False
        """
        if self.num_timesteps < self.learning_starts:
            action = np.array([self.env.action_space.sample()])
        else:
            action, _ = self.predict(self._last_obs)
        if self.flatten_obs:
            flat_obs = flatten_obs(self._last_obs, self.device)
        else:
            flat_obs = torch.tensor(self._last_obs, device=self.device, dtype=torch.float32).detach().clone()
        torch_obs = torch.tensor(flat_obs, dtype=torch.float32)
        torch_action = torch.tensor(action,device=self.device, dtype=torch.float32)
        q_val = float(self.critic(torch_obs, torch_action).mean())
        self.logger.record("train/rollout_q_step", q_val)
        self.logger.record_mean("train/rollout_q_mean", q_val)

        # perform action
        new_obs, rewards, dones, infos = self.env.step(action)
        self.episode_steps += 1
        self.logger.record("train/rollout_rewards_step", np.mean(rewards))
        self.logger.record_mean("train/rollout_rewards_mean", np.mean(rewards))
        if self.log_obs_step:
            for n in range(new_obs['observation'].shape[1]):
                dim_obs = new_obs['observation'][:,n]
                self.logger.record(f"train/obs_{n}", np.mean(dim_obs))
        if self.log_act_step:
            for n in range(action.shape[1]):
                dim_act = action[:, n]
                self.logger.record(f"train/act_{n}", np.mean(dim_act))

        self.num_timesteps += self.env.num_envs

        # save data to replay buffer; handle `terminal_observation`
        next_obs = deepcopy(new_obs)
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                self.logger.record_mean(f'train/mean_ep_length', self.episode_steps)
                self.episode_steps = 0
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
        self.replay_buffer.add(self._last_obs, next_obs, action, rewards, dones, infos)

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
        if self.flatten_obs:
            observations = flatten_obs(replay_data.observations, self.device)
            next_observations = flatten_obs(replay_data.next_observations, self.device)
        else:
            observations = torch.tensor(replay_data.observations, device=self.device, dtype=torch.float32).detach().clone()
            next_observations = torch.tensor(replay_data.next_observations, device=self.device, dtype=torch.float32).detach().clone()

        # train critic
        with torch.no_grad():
            next_actions = self.actor_target(next_observations)

            noise = torch.normal(0, self.target_noise_std, size=self.env.action_space.shape, device=self.device)
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_actions = torch.clamp(next_actions + noise, self.action_low, self.action_high)
            target_q_values = self.critic_target(next_observations, next_actions)
            next_q_value = replay_data.rewards + (1 - replay_data.dones) * self.gamma * torch.min(target_q_values, dim=0).values

        critic_a_values = self.critic(observations, replay_data.actions)

        critic_loss = torch.stack([F.mse_loss(_a_v, next_q_value.view(-1, 1)) for _a_v in critic_a_values]).sum()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.logger.record("train/critic_loss", critic_loss.item())
        self.logger.record("train/train_rewards", replay_data.rewards.flatten().mean().item())

        # train actor
        if self._n_updates % self.policy_delay == 0:
            actor_loss = -self.critic(observations, self.actor(observations)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.logger.record("train/actor_loss", actor_loss.item())

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

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
        if self.flatten_obs:
            observation = flatten_obs(obs, self.device)
        else:
            observation = torch.tensor(obs, device=self.device, dtype=torch.float32).detach().clone()
        with torch.no_grad():
            action = self.actor(observation).cpu().numpy()
            if not deterministic:
                action += self.noise()
        return action, state

    def save(self, path: Union[str, pathlib.Path, io.BufferedIOBase]):
        # Copy parameter list, so we don't mutate the original dict
        data = self.__dict__.copy()
        for to_exclude in ["logger", "env", "num_timesteps", "_n_updates", "_last_obs",
                           "replay_buffer", "actor", "critic", "critic_target", "actor_target", "noise"]:
            del data[to_exclude]
        # save network parameters
        data["_actor"] = self.actor.state_dict()
        data["_critic"] = self.critic.state_dict()
        torch.save(data, path)

    @classmethod
    def load(cls, path, env, **kwargs):
        model = cls(env=env, **kwargs)
        loaded_dict = torch.load(path)
        for k in loaded_dict:
            if k not in ["_actor", "_critic"]:
                model.__dict__[k] = loaded_dict[k]
        # load network states
        model.actor.load_state_dict(loaded_dict["_actor"])
        model.critic.load_state_dict(loaded_dict["_critic"])
        return model

    def set_logger(self, logger: Logger) -> None:
        self.logger = logger

    def get_env(self):
        return self.env
