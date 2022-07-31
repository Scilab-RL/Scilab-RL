import random
import subprocess
import numpy as np
from omegaconf import DictConfig


def flatten_dictConf(cfg, prefix=""):
    flat_cfg = {}
    for k, v in cfg.items():
        if type(v) == DictConfig:
            sub_dict = flatten_dictConf(v, prefix=k + ".")
            flat_cfg.update(sub_dict)
        else:
            flat_cfg[prefix + k] = v
    return flat_cfg


def get_git_label():
    try:
        git_label = str(subprocess.check_output(["git", 'describe', '--always'])).strip()[2:-3]
    except:
        git_label = ''
    return git_label


def set_global_seeds(i):
    try:
        import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0

    myseed = i + 1000 * rank if i is not None else None
    try:
        import tensorflow as tf
        tf.random.set_seed(myseed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(myseed)
    except ImportError:
        pass
    np.random.seed(myseed)
    random.seed(myseed)


def get_train_video_schedule(every_n_steps):
    def train_schedule(step_id):
        return step_id % every_n_steps == 1
    return train_schedule


def get_eval_video_schedule(every_n_epochs, n_eval_episodes):
    def eval_schedule(episode_id):
        return episode_id % (every_n_epochs * n_eval_episodes) == 1
    return eval_schedule


def avoid_start_learn_before_first_episode_finishes(alg_kwargs, env):
    try:
        max_ep_steps = env.get_attr("spec")[0].max_episode_steps
        # Raise the error because the attribute is present but is None
        if max_ep_steps is None:
            raise AttributeError
    # if not available check if a valid value was passed as an argument
    except AttributeError:
        raise ValueError(
            "The max episode length could not be inferred.\n"
            "You must specify a `max_episode_steps` when registering the environment,\n"
            "use a `gym.wrappers.TimeLimit` wrapper "
            "or pass `max_episode_length` to the model constructor"
        )
    if 'learning_starts' in alg_kwargs:
        alg_kwargs['learning_starts'] = max(alg_kwargs['learning_starts'], max_ep_steps)
    else:
        alg_kwargs['learning_starts'] = max_ep_steps
    return alg_kwargs
