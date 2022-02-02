import io
import pathlib
import random
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, \
    TrainFrequencyUnit
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from stable_baselines3.sac import SAC


class OO_SAC(SAC):
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

        if self.use_sde:
            self.actor.reset_noise()

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            done = False
            episode_reward, episode_timesteps = 0.0, 0

            obj_idx = random.randint(1, env.envs[0].n_objects)

            while not done:

                if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.actor.reset_noise()

                self._last_obs = self.transform_obs_to_oo_obs(self._last_obs, obj_idx)

                # Select action randomly or according to policy
                action, buffer_action = self._sample_action(learning_starts, action_noise)

                # Rescale and perform action
                new_obs, reward, done, infos = env.step(action)

                new_obs = self.transform_obs_to_oo_obs(new_obs, obj_idx)
                # TODO: reward = get_new_oo_reward() # implement new reward function based on obj_idx,
                reward = self.transform_reward_to_oo_reward(reward, obj_idx, env.envs[0].n_objects + 1) # n_obj + gripper

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
                self._store_transition(replay_buffer, buffer_action, new_obs, reward, done, infos)

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

    def transform_obs_to_oo_obs(self, obs, obj_idx):
        oo_obs = obs.copy()
        original_len = obs['achieved_goal'].shape[1]
        oneHot_idx = np.eye(self.env.envs[0].n_objects + 1)[obj_idx]

        achieved_coords = obs['achieved_goal'][0][obj_idx * 3: obj_idx * 3 + 3]
        oo_obs['achieved_goal'] = np.expand_dims(np.concatenate([oneHot_idx, achieved_coords]), axis=0)

        desired_coords = obs['desired_goal'][0][obj_idx * 3: obj_idx * 3 + 3]
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
