import os
import copy
import numpy as np

import gym
from gym import error, spaces
from gym.utils import seeding
from typing import List
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from util.custom_evaluation import get_success
# try:
#     import mujoco_py
# except ImportError as e:
#     raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

DEFAULT_SIZE = 500


def get_h_envs_from_env(bottom_env: gym.wrappers.TimeLimit,
                        level_steps_str: str, env_list: List[gym.GoalEnv] = [],
                        is_testing_env: bool = False, model: OffPolicyAlgorithm = None) -> List[gym.wrappers.TimeLimit]:
    if level_steps_str == '':
        return env_list
    level_steps = [int(s) for s in level_steps_str.split(",")]
    action_dim = len(bottom_env.env._sample_goal())
    obs_sample = bottom_env.env._get_obs()
    if len(level_steps) > 1:
        env = HierarchicalHLEnv(action_dim, obs_sample, is_testing_env=is_testing_env, model=model)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=level_steps[0])
    else:
        env = bottom_env
        env.spec.max_episode_steps = level_steps[0]
        env._max_episode_steps = level_steps[0]
    if len(env_list) >= 1:
        env_list[-1].set_sub_env(env)
    env_list.append(env)
    next_level_steps_str = ",".join([str(s) for s in level_steps[1:]])
    if model is not None and model.sub_model is not None:
        next_level_model = model.sub_model
    else:
        next_level_model = None
    env_list = get_h_envs_from_env(bottom_env, next_level_steps_str, env_list, is_testing_env, next_level_model)

    return env_list



class HierarchicalHLEnv(gym.GoalEnv):
    def __init__(self, action_dim, obs_sample, is_testing_env=None, model=None):
        # self.action_space = spaces.Box(-np.inf, np.inf, shape=(action_dim,), dtype='float32')
        self.action_space = spaces.Box(-1.0, 1.0, shape=(action_dim,), dtype='float32')
        # TODO: fix action space based on goal space.
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs_sample['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs_sample['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs_sample['observation'].shape, dtype='float32'),
        ))
        self._sub_env = None
        self.model = model
        self.is_testing_env = is_testing_env
        # self.test_reward_list = []
        # self.test_success_list = []

    # def reset_buffers(self):
    #     self.test_reward_list = []
    #     self.test_success_list = []
    #     if type(self._sub_env, HierarchicalHLEnv):
    #         self._sub_env.reset_buffers()

    def set_sub_env(self, env):
        self._sub_env = env

    # def step(self, action):
    #     subgoal = np.clip(action, self.action_space.low, self.action_space.high)
    #     self._sub_env.goal = subgoal
    #     assert self.model is not None, "Step not possible because no model defined yet."
    #     info = {'reward_list_{}'.format(self.model.layer): [],
    #             'success_list_{}'.format(self.model.layer): []}
    #     if self.is_testing_env:
    #         info = self.test_step()
    #         # info['reward_list_{}'.format(self.model.layer)].append(reward_list)
    #         # info['success_list_{}'.format(self.model.layer)].append(success_list)
    #     else:
    #         self.train_step()
    #     obs = self._get_obs()
    #     reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
    #     done = False  # Returning done = true after time steps are done is not necessary here because it is done in TimeLimit wrapper. #TODO: Check if done=True should be returned after goal is achieved.
    #     self._step_callback()
    #     succ = self._is_success(obs['achieved_goal'], self.goal)
    #     if self.model.is_top_layer:
    #         if 'is_success' not in info.keys():
    #             info['is_success'] = []
    #         info_list['is_success'].append(succ)
    #     if 'is_success{}'.format(self.model.layer) not in info_list.keys():
    #         info_list['is_success{}'.format(self.model.layer)] = []
    #     info_list['is_success{}'.format(self.model.layer)].append(succ)
    #     return obs, reward, done, info_list

    def step(self, action):
        subgoal = np.clip(action, self.action_space.low, self.action_space.high)
        self._sub_env.goal = subgoal
        assert self.model is not None, "Step not possible because no model defined yet."
        if self.is_testing_env:
            info = self.test_step()
        else:
            info = {}
            self.train_step()
        obs = self._get_obs()
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        done = False  # Returning done = true after time steps are done is not necessary here because it is done in TimeLimit wrapper. #TODO: Check if done=True should be returned after goal is achieved.
        self._step_callback()
        succ = self._is_success(obs['achieved_goal'], self.goal)
        if self.model.is_top_layer:
            if 'is_success' not in info.keys():
                info['is_success'] = []
            info['is_success'].append(succ)
        if 'is_success{}'.format(self.model.layer) not in info.keys():
            info['is_success{}'.format(self.model.layer)] = []
        info['is_success{}'.format(self.model.layer)].append(succ)

        return obs, reward, done, info

    def train_step(self):
        self.model.train_step()

    def test_step(self):
        info_list = self.model.test_step(self)
        # info_list = {'rewards{}'.format(self.model.layer): []}
        # done = False
        # while not done:
        #     obs = self._get_obs()
        #     action, state = self.model.sub_model.model.predict(obs, state=None, mask=None, deterministic=True)
        #     new_obs, reward, done, infos = self._sub_env.step(action)
        #     info_list['rewards{}'.format(self.model.layer)].append(reward)
        #     for k,v in infos.items():
        #         if k+str(self.model.layer) not in info_list.keys():
        #             info_list[k+str(self.model.layer)] = []
        #         info_list[k+str(self.model.layer)].append(v)
        return info_list

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with nself._sub_env._step_callback()umerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        super(HierarchicalHLEnv, self).reset()
        self.goal = self._sample_goal()
        obs = self._sub_env.reset()
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
        return self._sub_env.env._get_obs()

    # def _set_action(self, action):
    #     """Applies the given action to the simulation.
    #     """
    #     raise NotImplementedError()

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        return self._sub_env.env._is_success(achieved_goal, desired_goal)

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
