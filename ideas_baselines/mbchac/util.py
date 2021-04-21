import numpy as np
import gym
from stable_baselines3.common.type_aliases import GymEnv

def get_concat_dict_from_dict_list(dict_list):
    concat_info = {}
    for inf in dict_list:
        for k,v in inf.items():
            if k not in concat_info:
                concat_info[k] = []
            if type(v) == list:
                concat_info[k] += v
            else:
                concat_info[k].append(v)
    return concat_info

def merge_list_dicts(dict_of_lists1, dict_of_lists2):
    for k,v in dict_of_lists1.items():
        assert type(v) == list, "Error not a list."
        if k not in dict_of_lists2.keys():
            dict_of_lists2[k] = []
        dict_of_lists2[k] += v
    return dict_of_lists2

def check_for_correct_spaces(env: GymEnv, observation_space: gym.spaces.Space, action_space: gym.spaces.Space) -> None:
    """
    Checks that the environment has same spaces as provided ones. Used by BaseAlgorithm to check if
    spaces match after loading the model with given env.
    I have copied this function from stable_baselines3.commom.utils. the difference is that this one compares the string
    represetations of the spaces and the stable_baselines3 version compares the spaces. However, I found that comparing
    the same spaces in the SB3 version yields not equal even if the spaces are equal.
    Checked parameters:
    - observation_space
    - action_space

    :param env: Environment to check for valid spaces
    :param observation_space: Observation space to check against
    :param action_space: Action space to check against
    """
    if str(observation_space) != str(env.observation_space):
        raise ValueError(f"Observation spaces do not match: {observation_space} != {env.observation_space}")
    if str(action_space) != str(env.action_space):
        raise ValueError(f"Action spaces do not match: {action_space} != {env.action_space}")