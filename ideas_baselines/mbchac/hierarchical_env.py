import os
import copy
import numpy as np

import gym
from gym import error, spaces
from gym.utils import seeding
from typing import List

# try:
#     import mujoco_py
# except ImportError as e:
#     raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

DEFAULT_SIZE = 500


def get_h_envs_from_env(bottom_env: gym.wrappers.TimeLimit,
                        level_steps: List[int], env_list: List[gym.GoalEnv] = []) -> List[gym.wrappers.TimeLimit]:
    if len(level_steps) == 0:
        return env_list
    action_dim = len(bottom_env.env._sample_goal())
    obs_sample = bottom_env.env._get_obs()
    if len(level_steps) > 1:
        env = HierarchicalHLEnv(action_dim, obs_sample, bottom_env)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=level_steps[0])
    else:
        env = bottom_env
        env.spec.max_episode_steps = level_steps[0]
        env._max_episode_steps = level_steps[0]
    if len(env_list) >= 1:
        env_list[-1].set_sub_env(env)
    env_list.append(env)

    env_list = get_h_envs_from_env(bottom_env, level_steps[1:], env_list)


    return env_list




class HierarchicalHLEnv(gym.GoalEnv):
    def __init__(self, action_dim, obs_sample, bottom_env):
        self.action_space = spaces.Box(-1., 1., shape=(action_dim,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs_sample['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs_sample['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs_sample['observation'].shape, dtype='float32'),
        ))
        self._sub_env = None
        self.model = None


    def set_sub_env(self, env):
        self._sub_env = env

    # @property
    # def dt(self):
    #     return self.sim.model.opt.timestep * self.sim.nsubsteps

    # Env methods
    # ----------------------------

    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    def step(self, action):
        subgoal = np.clip(action, self.action_space.low, self.action_space.high)
        self._sub_env.goal = subgoal
        if self.model is not None:
            self.model.sub_model.learn(total_timesteps=1, tb_log_name="MBCHAC_{}".format(self.model.layer-1),
                                      reset_num_timesteps=False)
        else:
            print("Step not possible because no model defined yet.")

        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with nself._sub_env._step_callback()umerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        super(HierarchicalHLEnv, self).reset()
        self.goal = self._sample_goal()
        obs = self._sub_env.reset()
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
        self._sub_env.get_obs()

    # def _set_action(self, action):
    #     """Applies the given action to the simulation.
    #     """
    #     raise NotImplementedError()

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        return self._sub_env._is_success(achieved_goal, desired_goal)

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
