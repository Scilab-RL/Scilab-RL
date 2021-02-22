from typing import Dict

import numpy as np
from gym import spaces
import gym

from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper
# from ideas_baselines.mbchac import MBCHAC
from stable_baselines3.common.base_class import BaseAlgorithm


class HEnvWrapper(ObsDictWrapper):
    """
    Wrapper for a ObsDictWrapper Env which overrides the action steps and action space to support hierarchical RL.

    :param env: The environment to wrap.
    """

    def __init__(self, venv: VecEnv, sub_model: BaseAlgorithm, n_steps: int):
        # venv = gym.make(env.spec.id) # make a new environment to not overwrite anything.
        venv.spec.max_episode_steps = n_steps
        venv._max_episode_steps = n_steps
        if sub_model is not None:
            tmp_goal = venv.sample_goal()
            venv.action_space = spaces.Box(-np.inf, np.inf, shape=tmp_goal.shape, dtype='float32')
        super().__init__(venv)

        self.sub_model = sub_model



    # def reset(self):
    #     return self.level_env.reset()
    #
    # def step_wait(self):
    #     return self.level_env.step_wait()
