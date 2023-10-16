from typing import Any, Dict, List, Optional, Tuple, Type, Union, Iterable

import pathlib
import io
import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.logger import Logger
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn
# from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from custom_algorithms.mysacher.mysacher import SoftQNetwork, flatten_obs, Actor


device = th.device("cuda" if th.cuda.is_available() else "cpu")


class Critic(th.nn.Module):
    def __init__(self, env, lr):
        super().__init__()
        self.crit_1 = SoftQNetwork(env).to(device)
        self.crit_2 = SoftQNetwork(env).to(device)
        self.optimizer = th.optim.Adam(list(self.crit_1.parameters()) + list(self.crit_2.parameters()), lr=lr)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        obs = flatten_obs(obs).to(device)

        return tuple([self.crit_1(obs, actions), self.crit_2(obs, actions)])


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
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param seed: Seed for the pseudo random generators
    """
    actor: Actor
    critic: Critic
    critic_target: Critic

    def __init__(
        self,
        env: Union[GymEnv, str],
        learning_rate: float = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        ent_coef: Union[str, float] = "auto",
        seed: Optional[int] = None,
    ):
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

        self.num_timesteps = 0
        self.seed = seed
        self.learning_rate = learning_rate
        self._last_obs = None

        self._episode_num = 0

        # Buffers for logging
        # For logging (and TD3 delayed updates)
        self._n_updates = 0  # type: int

        # Create and wrap the env if needed
        if env is not None:
            self.observation_space = env.observation_space
            self.action_space = env.action_space
            self.n_envs = env.num_envs
            self.env = env

            if isinstance(self.action_space, spaces.Box):
                assert np.all(
                    np.isfinite(np.array([self.action_space.low, self.action_space.high]))
                ), "Continuous action space must have a finite lower and upper bound"

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.tau = tau
        self.gamma = gamma
        self.replay_buffer: Optional[ReplayBuffer] = None
        self.replay_buffer_class = replay_buffer_class
        self.replay_buffer_kwargs = replay_buffer_kwargs or {}

        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.ent_coef_optimizer: Optional[th.optim.Adam] = None

        self._setup_model()

        self.logger = None

    def _setup_model(self) -> None:
        if self.replay_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.replay_buffer_class = DictReplayBuffer
            else:
                self.replay_buffer_class = ReplayBuffer

        if self.replay_buffer is None:
            # Make a local copy as we should not pickle
            # the environment when using HerReplayBuffer
            replay_buffer_kwargs = self.replay_buffer_kwargs.copy()
            # if issubclass(self.replay_buffer_class, HerReplayBuffer):
            #     assert self.env is not None, "You must pass an environment when using `HerReplayBuffer`"
            #     replay_buffer_kwargs["env"] = self.env
            self.replay_buffer = self.replay_buffer_class(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                device=self.device,
                n_envs=self.n_envs,
                **replay_buffer_kwargs,  # pytype:disable=wrong-keyword-args
            )

        # Convert train freq parameter to TrainFreq object
        self._create_nets()
        # Target entropy is used when learning the entropy coefficient
        self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))  # type: ignore

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if self.ent_coef == "auto":
            # Default initial value of ent_coef when learned
            init_value = 1.0

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.learning_rate)
        else:
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

    def _create_nets(self) -> None:
        self.actor = Actor(self.env).to(self.device)
        self.actor.optimizer = th.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic = Critic(self.env, lr=self.learning_rate).to(self.device)

        self.critic_target = Critic(self.env, lr=self.learning_rate).to(self.device)

        self.critic_target.crit_1.load_state_dict(self.critic.crit_1.state_dict())
        self.critic_target.crit_2.load_state_dict(self.critic.crit_2.state_dict())

    def train(self):
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        # Sample replay buffer
        replay_data = self.replay_buffer.sample(self.batch_size)

        # Action by the current actor for the sampled state
        actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
        log_prob = log_prob.reshape(-1, 1)

        ent_coef_loss = None
        if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
            # Important: detach the variable from the graph
            # so we don't change it with other losses
            # see https://github.com/rail-berkeley/softlearning/issues/60
            ent_coef = th.exp(self.log_ent_coef.detach())
            ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
            ent_coef_losses.append(ent_coef_loss.item())
        else:
            ent_coef = self.ent_coef_tensor

        ent_coefs.append(ent_coef.item())

        # Optimize entropy coefficient, also called
        # entropy temperature or alpha in the paper
        if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()

        with th.no_grad():
            # Select action according to policy
            next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
            # Compute the next Q values: min over all critics targets
            next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
            next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
            # add entropy term
            next_q_values = next_q_values - ent_coef * next_log_prob
            # td error + entropy term
            target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

        # Get current Q-values estimates for each critic network
        # using action from the replay buffer
        current_q_values = self.critic(replay_data.observations, replay_data.actions)

        # Compute critic loss
        critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
        assert isinstance(critic_loss, th.Tensor)  # for type checker
        critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

        # Optimize the critic
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # Compute actor loss
        # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
        # Min over all critic networks
        q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
        min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
        actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
        actor_losses.append(actor_loss.item())

        # Optimize the actor
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # Update target networks
        # polyak update
        with th.no_grad():
            for param, target_param in zip(self.critic.crit_1.parameters(), self.critic_target.crit_1.parameters()):
                target_param.data.mul_(1 - self.tau)
                th.add(target_param.data, param.data, alpha=self.tau, out=target_param.data)
            for param, target_param in zip(self.critic.crit_2.parameters(), self.critic_target.crit_2.parameters()):
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
            rollout = self.collect_rollouts(self.env, callback=callback)

            if rollout.continue_training is False:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                self.train()

        callback.on_training_end()

        return self

    def collect_rollouts(
        self,
        env,
        callback: BaseCallback
    ):
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :return:
        """
        # Select action randomly or according to policy
        if self.num_timesteps < self.learning_starts:
            actions = np.array([self.action_space.sample()])
        else:
            actions, _ = self.predict(self._last_obs)

        # perform action
        new_obs, rewards, dones, infos = env.step(actions)

        self.num_timesteps += env.num_envs

        # Only stop training if return value is False, not when it is None.
        if callback.on_step() is False:
            return RolloutReturn(1, 0, continue_training=False)

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

        :param observation: the input observation
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
