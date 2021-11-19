from typing import Any, Tuple, Union, Optional, List
import os
import datetime
import tempfile
import gym
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.logger import Logger, KVWriter, make_output_format

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


class LayerLogger(Logger):
    def __init__(self, folder: Optional[str], output_formats: List[KVWriter]):
        self.layer = ''
        super().__init__(folder, output_formats)

    def record(self, key: str, value: Any, exclude: Optional[Union[str, Tuple[str, ...]]] = None) -> None:
        if key.startswith('train'):
            key = 'train' + self.layer + '/' + key.split('/')[1]
        super().record(key, value, exclude)

    def set_layer(self, layer: str):
        self.layer = layer


def configure(folder: Optional[str] = None, format_strings: Optional[List[str]] = None) -> LayerLogger:
    """
    Configure the current logger.

    :param folder: the save location
        (if None, $SB3_LOGDIR, if still None, tempdir/SB3-[date & time])
    :param format_strings: the output logging format
        (if None, $SB3_LOG_FORMAT, if still None, ['stdout', 'log', 'csv'])
    :return: The logger object.
    """
    if folder is None:
        folder = os.getenv("SB3_LOGDIR")
    if folder is None:
        folder = os.path.join(tempfile.gettempdir(), datetime.datetime.now().strftime("SB3-%Y-%m-%d-%H-%M-%S-%f"))
    assert isinstance(folder, str)
    os.makedirs(folder, exist_ok=True)

    log_suffix = ""
    if format_strings is None:
        format_strings = os.getenv("SB3_LOG_FORMAT", "stdout,log,csv").split(",")

    format_strings = list(filter(None, format_strings))
    output_formats = [make_output_format(f, folder, log_suffix) for f in format_strings]

    logger = LayerLogger(folder=folder, output_formats=output_formats)
    # Only print when some files will be saved
    if len(format_strings) > 0 and format_strings != ["stdout"]:
        logger.log(f"Logging to {folder}")
    return logger
