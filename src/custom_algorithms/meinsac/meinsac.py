from typing import Any, Dict, List, Optional, Tuple, Type, Union, Iterable

import pathlib
import io
import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.logger import Logger
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from custom_algorithms.mysacher.mysacher import SoftQNetwork, flatten_obs, Actor


class MEINSAC:
    """
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param seed: Seed for the pseudo random generators
    """
    actor: Actor
    critic: SoftQNetwork
    critic_target: SoftQNetwork

    def __init__(
        self,
        env: Union[GymEnv, str],
        learning_rate: float = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        ent_coef: Union[str, float] = "auto",
        seed: Optional[int] = None,
    ):
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

        self.env = env

        if isinstance(self.env.action_space, spaces.Box):
            assert np.all(
                np.isfinite(np.array([self.env.action_space.low, self.env.action_space.high]))
            ), "Continuous action space must have a finite lower and upper bound"

        self.seed = seed
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.tau = tau
        self.gamma = gamma

        self.replay_buffer = DictReplayBuffer(
            self.buffer_size,
            self.env.observation_space,
            self.env.action_space,
            device=self.device,
            n_envs=self.env.num_envs
        )

        self._create_nets()

        self.ent_coef = ent_coef
        if self.ent_coef == "auto":
            self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))
            self.log_ent_coef = th.zeros(1, device=self.device).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.learning_rate)
        else:
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

        self.logger = None
        self._last_obs = None
        self._episode_num = 0
        self.num_timesteps = 0
        self._n_updates = 0

    def _create_nets(self) -> None:
        self.actor = Actor(self.env).to(self.device)
        self.actor_optimizer = th.optim.Adam(self.actor.parameters(), lr=self.learning_rate)

        self.crit_1 = SoftQNetwork(self.env).to(self.device)
        self.crit_2 = SoftQNetwork(self.env).to(self.device)
        self.crit_1_target = SoftQNetwork(self.env).to(self.device)
        self.crit_2_target = SoftQNetwork(self.env).to(self.device)
        self.crit_1_target.load_state_dict(self.crit_1.state_dict())
        self.crit_2_target.load_state_dict(self.crit_2.state_dict())
        self.critic_optimizer = th.optim.Adam(list(self.crit_1.parameters()) + list(self.crit_2.parameters()),
                                              lr=self.learning_rate)

    def train(self):
        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        # Sample replay buffer
        replay_data = self.replay_buffer.sample(self.batch_size)
        observations = flatten_obs(replay_data.observations)
        next_observations = flatten_obs(replay_data.next_observations)

        # optimize entropy coefficient
        if self.ent_coef == "auto":
            with th.no_grad():
                _, log_pi = self.actor.get_action(observations)
            ent_coef_loss = (-self.log_ent_coef * (log_pi + self.target_entropy)).mean()
            ent_coef_losses.append(ent_coef_loss.item())

            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()
            ent_coef = self.log_ent_coef.exp().item()
        else:
            ent_coef = self.ent_coef_tensor
        ent_coefs.append(ent_coef)

        # train critic
        with th.no_grad():
            next_state_actions, next_state_log_pi = self.actor.get_action(next_observations)
            qf1_next_target = self.crit_1_target(next_observations, next_state_actions)
            qf2_next_target = self.crit_2_target(next_observations, next_state_actions)
            min_qf_next_target = th.min(qf1_next_target, qf2_next_target) - ent_coef * next_state_log_pi
            next_q_value = replay_data.rewards.flatten() + (1 - replay_data.dones.flatten()) * self.gamma * min_qf_next_target.flatten()

        qf1_a_values = self.crit_1(observations, replay_data.actions).view(-1)
        qf2_a_values = self.crit_2(observations, replay_data.actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = 0.5 * (qf1_loss + qf2_loss)
        critic_losses.append(qf_loss.item())

        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        # train actor
        pi, log_pi = self.actor.get_action(observations)
        qf1_pi = self.crit_1(observations, pi)
        qf2_pi = self.crit_2(observations, pi)
        min_qf_pi = th.min(qf1_pi, qf2_pi).view(-1)
        actor_loss = ((ent_coef * log_pi) - min_qf_pi).mean()
        actor_losses.append(actor_loss.item())

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks with polyak update
        with th.no_grad():
            for param, target_param in zip(self.crit_1.parameters(), self.crit_1_target.parameters()):
                target_param.data.mul_(1 - self.tau)
                th.add(target_param.data, param.data, alpha=self.tau, out=target_param.data)
            for param, target_param in zip(self.crit_2.parameters(), self.crit_2_target.parameters()):
                target_param.data.mul_(1 - self.tau)
                th.add(target_param.data, param.data, alpha=self.tau, out=target_param.data)

        self._n_updates += 1

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        self._last_obs = self.env.reset()

        callback.init_callback(self)

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(callback=callback)

            if rollout.continue_training is False:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                self.train()

        callback.on_training_end()

        return self

    def collect_rollouts(
        self,
        callback: BaseCallback
    ):
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :return:
        """
        # Select action randomly or according to policy
        if self.num_timesteps < self.learning_starts:
            actions = np.array([self.env.action_space.sample()])
        else:
            actions, _ = self.predict(self._last_obs)

        # perform action
        new_obs, rewards, dones, infos = self.env.step(actions)

        self.num_timesteps += self.env.num_envs

        num_collected_episodes = 0

        # save data to replay buffer; handle `terminal_observation`
        real_next_obs = new_obs.copy()
        for idx, done in enumerate(dones):
            if done:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
                num_collected_episodes += 1
                self._episode_num += 1
        self.replay_buffer.add(self._last_obs, real_next_obs, actions, rewards, dones, infos)

        self._last_obs = new_obs

        # Only stop training if return value is False, not when it is None.
        if callback.on_step() is False:
            return RolloutReturn(1, 0, continue_training=False)
        return RolloutReturn(1, num_collected_episodes, True)

    def set_logger(self, logger: Logger) -> None:
        self.logger = logger

    def get_env(self):
        return self.env

    def predict(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param obs: the input observation
        :return: the model's action
        """
        observation = flatten_obs(obs)
        action, _ = self.actor.get_action(observation)
        return action.detach().cpu().numpy(), None

    def save(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        exclude: Optional[Iterable[str]] = None,
        include: Optional[Iterable[str]] = None,
    ) -> None:
        return
