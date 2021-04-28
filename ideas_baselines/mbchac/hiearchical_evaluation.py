from typing import Callable, List, Optional, Tuple, Union, Dict

import gym
import numpy as np
import cv2

from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
from util.custom_evaluation import get_success
from collections import OrderedDict
from ideas_baselines.mbchac.util import merge_list_dicts

def evaluate_hierarchical_policy(
    layer_alg: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    maybe_reset_env: bool = True
) -> OrderedDict:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    This is made to work only with one env.

    :param layer_alg: The RL agent you want to evaluate.
    :param env: The gym environment. In the case of a ``VecEnv``
        this must contain only one environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of reward per episode
        will be returned instead of the mean.
    :return: Mean reward per episode, std of reward per episode
        returns ([float], [int]) when ``return_episode_rewards`` is True
    """

    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    info_list = OrderedDict()
    for i in range(n_eval_episodes):
        # Avoid double reset, as VecEnv are reset automatically
        if maybe_reset_env:
            if not isinstance(env, VecEnv) or i == 0:
                obs = env.reset()
        assert isinstance(env.venv, DummyVecEnv), "Error environment must be a DummyVecEnv"
        layer_alg.reset_eval_info_list()
        this_info_list = layer_alg.test_episode(env)
        info_list = merge_list_dicts(this_info_list, info_list)

    if layer_alg.is_top_layer:
        # For compatibility with HER, add a few redundant extra fields:
        copy_fields = {'test/success_rate': 'test_{}/ep_success'.format(layer_alg.layer),
                       'test/mean_ep_length': 'test_{}/ep_length'.format(layer_alg.layer),
                       'test/mean_reward': 'test_{}/ep_reward'.format(layer_alg.layer)
                       }
        for k, v in copy_fields.items():
            try:
                info_list[k] = info_list[v]
            except:
                "Warning, column {} not in info_list".format(v)
    return info_list