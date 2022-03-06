from typing import Any, Dict, List, Optional, Tuple, Type, Union

import random
import numpy as np
import torch as th
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import RolloutReturn, TrainFreq
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.sac import SAC
from torch.nn import functional as F

from ideas_baselines.oo_sac.oo_monitor import OO_Monitor


def _get_idx(obs, n_obj):
    for i in range(n_obj+1):
        if obs['achieved_goal'][i] == 1:
            return i


class OO_SAC(SAC):
    def __init__(
            self,
            policy: Union[str, Type[SACPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 3e-4,
            buffer_size: int = 1000000,  # 1e6
            learning_starts: int = 100,
            batch_size: int = 256,
            tau: float = 0.005,
            gamma: float = 0.99,
            train_freq: Union[int, Tuple[int, str]] = 1,
            gradient_steps: int = 1,
            action_noise: Optional[ActionNoise] = None,
            replay_buffer_class: Optional[ReplayBuffer] = None,
            replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
            optimize_memory_usage: bool = False,
            ent_coef: Union[str, float] = "auto",
            target_update_interval: int = 1,
            target_entropy: Union[str, float] = "auto",
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            use_sde_at_warmup: bool = False,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Dict[str, Any] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
    ):
        super(OO_SAC, self).__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            tensorboard_log=tensorboard_log,
            create_eval_env=create_eval_env,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model
        )

        #env.env = OO_Monitor(env.env)
        env.env._is_success = self._is_success
        env.env.goal_distance = self.goal_distance
        self.distance_threshold = env.env.distance_threshold

    def _is_success(self, achieved_goal, desired_goal):
        obj_idx = self._current_obj_idx
        d = self.goal_distance(achieved_goal[obj_idx * 3:obj_idx * 3 + 3], desired_goal[obj_idx * 3:obj_idx * 3 + 3])
        return (d < self.distance_threshold).astype(np.float32)

    def goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    _current_obj_idx = 0

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            train_freq: TrainFreq,
            replay_buffer: ReplayBuffer,
            action_noise: Optional[ActionNoise] = None,
            learning_starts: int = 0,
            log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        episode_rewards, total_timesteps = [], []
        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert env.num_envs == 1, "OffPolicyAlgorithm only support single environment"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        #env.envs[0].env.env._is_success = self._is_success
        #env.envs[0].env.env.goal_distance = self.goal_distance
        #self.distance_threshold = env.envs[0].env.env.distance_threshold

        if self.use_sde:
            self.actor.reset_noise()

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            done = False
            episode_reward, episode_timesteps = 0.0, 0

            obj_idx = random.randint(1, env.envs[0].n_objects)
            self._current_obj_idx = obj_idx

            while not done:

                if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.actor.reset_noise()

                # Select action randomly or according to policy
                action, buffer_action = self._sample_action(learning_starts, action_noise)

                # Rescale and perform action
                new_obs, reward, done, infos = env.step(action)

                oo_obs = self.transform_obs_to_oo_obs(new_obs, obj_idx)

                oo_reward = env.envs[0].compute_reward(oo_obs['achieved_goal'], oo_obs['desired_goal'], infos)
                reward = oo_reward

                self.num_timesteps += 1
                episode_timesteps += 1
                num_collected_steps += 1

                # Give access to local variables
                callback.update_locals(locals())
                # Only stop training if return value is False, not when it is None.
                if callback.on_step() is False:
                    return RolloutReturn(0.0, num_collected_steps, num_collected_episodes, continue_training=False)

                episode_reward += reward

                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos, done)

                # Store data in replay buffer (normalized action and unnormalized observation)
                self._store_transition(replay_buffer, buffer_action, oo_obs, reward, done, infos)

                self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

                # For DQN, check if the target network should be updated
                # and update the exploration schedule
                # For SAC/TD3, the update is done as the same time as the gradient update
                # see https://github.com/hill-a/stable-baselines/issues/900
                self._on_step()

                if not should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
                    break

            if done:
                num_collected_episodes += 1
                self._episode_num += 1
                episode_rewards.append(episode_reward)
                total_timesteps.append(episode_timesteps)

                if action_noise is not None:
                    action_noise.reset()

                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()

        mean_reward = np.mean(episode_rewards) if num_collected_episodes > 0 else 0.0

        callback.on_rollout_end()

        return RolloutReturn(mean_reward, num_collected_steps, num_collected_episodes, continue_training)

    def _is_one_hot(self, vec):
        return np.isin(vec, [0., 1.]).all()

    def transform_obs_to_oo_obs(self, obs, obj_idx):
        oo_obs = obs.copy()
        original_len = obs['achieved_goal'].shape[1]
        n_obj = self.env.envs[0].n_objects
        oneHot_idx = np.eye(n_obj + 1)[obj_idx]

        # Considering one-hot vector
        if self._is_one_hot(oo_obs['achieved_goal'][0][:n_obj+1]):
            achieved_coords = oo_obs['achieved_goal'][0][1 + n_obj: 1 + n_obj + 3]
            desired_coords = oo_obs['desired_goal'][0][1 + n_obj: 1 + n_obj + 3]
        else:
            achieved_coords = oo_obs['achieved_goal'][0][obj_idx * 3: obj_idx * 3 + 3]
            desired_coords = oo_obs['desired_goal'][0][obj_idx * 3: obj_idx * 3 + 3]

        oo_obs['achieved_goal'] = np.expand_dims(np.concatenate([oneHot_idx, achieved_coords]), axis=0)
        oo_obs['desired_goal'] = np.expand_dims(np.concatenate([oneHot_idx, desired_coords]), axis=0)
        # Zero-pad values if vector is too long
        len_diff = abs(len(oo_obs['achieved_goal'][0]) - original_len)
        if len_diff != 0:
            oo_obs['achieved_goal'] = np.expand_dims(
                np.concatenate([oo_obs['achieved_goal'][0], np.zeros(len_diff)]), axis=0)
            oo_obs['desired_goal'] = np.expand_dims(
                np.concatenate([oo_obs['desired_goal'][0], np.zeros(len_diff)]), axis=0)
        return oo_obs

    def transform_reward_to_oo_reward(self, reward, idx, n_obj):
        oo_reward = np.full(n_obj, float(-1))
        oo_reward[idx] = reward[0]
        return oo_reward

    # TODO: Remove this function if we don't need to modify it.
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
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
            if ent_coef_loss is not None:
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
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Mean over all critic networks
            q_values_pi = th.cat(self.critic.forward(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

