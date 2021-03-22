import os
import copy
import numpy as np

import gym
from stable_baselines3.common import logger
from gym import error, spaces
from gym.utils import seeding
from gym.envs.robotics import FetchReachEnv
from typing import Any, Callable, List, Optional, Sequence, Union
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv
from copy import deepcopy
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn
from util.custom_evaluation import get_success
# from ideas_baselines.mbchac.mbchac import MBCHAC
# try:
#     import mujoco_py
# except ImportError as e:
#     raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

DEFAULT_SIZE = 500


def get_h_envs_from_env(bottom_env: gym.wrappers.TimeLimit,
                        level_steps_str: str,
                        is_testing_env: bool = False, model: OffPolicyAlgorithm = None) -> List[gym.wrappers.TimeLimit]:

    def recursive_get_henvs(bottom_env: gym.wrappers.TimeLimit,
                        level_steps_str: str, env_list: List[gym.GoalEnv] = [],
                        is_testing_env: bool = False, model: OffPolicyAlgorithm = None) -> List[gym.wrappers.TimeLimit]:

        if level_steps_str == '':
            return env_list
        level_steps = [int(s) for s in level_steps_str.split(",")]
        if len(level_steps) > 1:
            env = HierarchicalHLEnv(bottom_env, is_testing_env=is_testing_env, model=model)
            env = gym.wrappers.TimeLimit(env, max_episode_steps=level_steps[0])
        else:
            env = bottom_env
            env.env.spec.max_episode_steps = level_steps[0]#TODO make independent of number of Wrappers
            env.env._max_episode_steps = level_steps[0]
            if model is not None:
                env.env.model = model
        env_list.append(env)
        next_level_steps_str = ",".join([str(s) for s in level_steps[1:]])
        if model is not None and model.sub_model is not None:
            next_level_model = model.sub_model
        else:
            next_level_model = None
        env_list = recursive_get_henvs(bottom_env, next_level_steps_str, env_list, is_testing_env, next_level_model)

        return env_list

    env_list = recursive_get_henvs(bottom_env=bottom_env, level_steps_str=level_steps_str,
                                   env_list=[], is_testing_env=is_testing_env, model=model)

    # iterate through reversed list to set sub_envs correctly; necessary for recursive action_space determination.
    for level, e in enumerate(reversed(env_list)):
        j = len(env_list) - level - 1
        if level > 0:
            env_list[j].set_sub_env(env_list[j+1])
            # re-set action space also for TimeLimit Wrapper class
            env_list[j].action_space = env_list[j].env.action_space
        env_list[j].is_top_level_env = j == 0
        env_list[j].level = level

    return env_list

class HierarchicalVecEnv(DummyVecEnv):
    """
    This class has the same functionality as DummyVecEnv, but it does not reset the simulator when a low-level episode ends.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        super().__init__(env_fns)

    # The same function as in DummyVecEnv, but without resetting the simulation when the episode ends.
    # This function also sets done to true on success
    def step_wait(self) -> VecEnvStepReturn:
        for env_idx in range(self.num_envs):
            obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx] = self.envs[env_idx].step(
                self.actions[env_idx]
            )
            # Set done to true if success is achieved or if it is done any ways (by TimeLimit)
            # self.buf_dones[env_idx] = np.isclose(self.buf_infos[env_idx]['is_success'], 1) or self.buf_dones[env_idx]
            if self.buf_dones[env_idx]:
                # save final observation where user can get it, but don't reset
                self.buf_infos[env_idx]["terminal_observation"] = obs
                # if self.envs[env_idx].is_top_level_env:
                #     obs = self.envs[env_idx].reset()
                self.envs[env_idx]._elapsed_steps = 0
            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))

    def reset_all(self):
        for env_idx in range(self.num_envs):
            obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return self.buf_obs

class HierarchicalHLEnv(gym.GoalEnv):
    def __init__(self, bottom_env, is_testing_env=None, model=None):
        self.bottom_env = bottom_env
        self.action_space = bottom_env.observation_space.spaces['desired_goal']
        if np.inf in self.action_space.high or -np.inf in self.action_space.low:
            logger.warn("Warning, subgoal space of hierarchical environment not defined. I will guess the subgoal bounds based on drawing goal samples when the sub env is set.")

        self.observation_space = bottom_env.env.observation_space
        self._sub_env = None
        self.model = model
        self.is_testing_env = is_testing_env

    def update_action_bound_guess(self):
        n_samples = 5000
        self.action_space.high = [-np.inf] * len(self.action_space.high)
        self.action_space.low = [np.inf] * len(self.action_space.low)
        for i in range(n_samples):
            goal = self._sub_env.env.unwrapped._sample_goal()
            self.action_space.high = np.maximum(goal, self.action_space.high)
            self.action_space.low = np.minimum(goal, self.action_space.low)
        # Add some small extra margin.
        epsilon = np.zeros_like(self.action_space.high) + 0.01
        self.action_space.high += np.abs(self.action_space.high - self.action_space.low) * 0.01
        self.action_space.low -= np.abs(self.action_space.high - self.action_space.low) * 0.01
        action_space_range = self.action_space.high - self.action_space.low
        action_space_range = np.maximum(action_space_range, epsilon)
        self.action_space.high = self.action_space.low + action_space_range
        # self.action_space.low = self.action_space.low - epsilon
        # Reset action space to determine whether the space is bounded.
        self.action_space = gym.spaces.Box(self.action_space.low, self.action_space.high)
        logger.info("Updated action bound guess by random sampling: Action space high: {}, Action space low: {}".format(self.action_space.high, self.action_space.low))

    def set_sub_env(self, env):
        self._sub_env = env
        if np.inf in self.action_space.high or -np.inf in self.action_space.low:
            self.update_action_bound_guess()
        return

    def step(self, action):
        subgoal = np.clip(action, self.action_space.low, self.action_space.high)
        self._sub_env.env._elapsed_steps = 0 # Set elapsed steps to 0 but don't reset the whole simulated environment
        self._sub_env.env.unwrapped.goal = subgoal
        self._sub_env.display_subgoals(subgoal, size=0.03, form='cylinder')

        assert self.model is not None, "Step not possible because no model defined yet."
        if self.is_testing_env:
            info = self.test_step()
        else:
            info = {}
            self.train_step()
        if not self.is_testing_env:
            if self.model.sub_model is not None:
                if self.model.sub_model.in_subgoal_test_mode:
                    info['is_subgoal_testing_trans'] = 1
                else:
                    info['is_subgoal_testing_trans'] = 0
        obs = self._get_obs()
        # obs['desired_goal'] = subgoal
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)

        self._step_callback()
        succ = self._is_success(obs['achieved_goal'], self.goal)
        info['is_success'] = succ
        done = False  # Returning done = true after time steps are done is not necessary here because it is done in TimeLimit wrapper. #TODO: Check if done=True should be returned after goal is achieved.
        # done = np.isclose(succ, 1)
        return obs, reward, done, info

    def train_step(self):
        self.model.train_step()

    def test_step(self):
        info_list = self.model.test_step(self)
        return info_list

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with nself._sub_env._step_callback()umerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        super(HierarchicalHLEnv, self).reset()
        obs = self._sub_env.reset()
        self.goal = self._sub_env.goal
        # if self.is_testing_env: ## DEBUG
        #     print("setting new testing goal: {}".format(self.goal))
        self.last_obs = obs
        return obs

    def close(self):
        self._sub_env.close()

    def render(self, **kwargs):
        self._sub_env.render(**kwargs)

    def _get_viewer(self, **kwargs):
        return self._sub_env._get_viewer(**kwargs)

    def _get_obs(self):
        """Returns the observation.
        """
        obs = self._sub_env.env.unwrapped._get_obs()
        obs['desired_goal'] = self.goal
        return obs

    # def _set_action(self, action):
    #     """Applies the given action to the simulation.
    #     """
    #     raise NotImplementedError()

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        return self._sub_env.env.unwrapped._is_success(achieved_goal, desired_goal)

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """
        return self._sub_env.env._sample_goal()

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        self._sub_env._env_setup()

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        self._sub_env._viewer_setup()

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass
        # self._sub_env._render_callback()

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass
        # self._sub_env._step_callback()

    def compute_reward(self, achieved_goal, goal, info):
        return self._sub_env.env.compute_reward(achieved_goal, goal, info)
