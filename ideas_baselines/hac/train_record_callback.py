from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, EventCallback
from typing import Any, Callable, Dict, List, Optional, Union
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
import gym
import numpy as np
import os
import warnings
from util.custom_evaluation import evaluate_policy
from ideas_baselines.hac.hiearchical_evaluation import evaluate_hierarchical_policy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common import logger
from collections import deque
import time
from stable_baselines3.common.utils import safe_mean
from ideas_baselines.hac.hierarchical_env import get_h_envs_from_env
import matplotlib.pyplot as plt
import cv2

class TrainVideoCallback(BaseCallback):
    """
    Callback for evaluating an agent.

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every eval_freq call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param deterministic: Whether to render or not the environment during evaluation
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    """

    def __init__(
        self,
        env: Union[gym.Env, VecEnv],
        log_path: str = None,
        render: str = 'record',
        model = None
    ):
        super(TrainVideoCallback, self).__init__(
            training_env=env,
            model=model)

        self.render = render
        self.vid_size = 1024, 768
        self.vid_fps = 25
        self.eval_count = 0
        if self.render == 'record':
            self.render_info = {'size': self.vid_size, 'fps': self.vid_fps, 'eval_count': self.eval_count,
                                'path': self.log_path}
        else:
            self.render_info = None

    def _on_step(self, log_prefix='') -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            if self.render_info is not None:
                self.render_info['eval_count'] = self.eval_count
            info_list = evaluate_hierarchical_policy(
                self.top_level_layer,
                self.eval_env,
                self.render_info,
                n_eval_episodes=self.n_eval_episodes
            )
            for k,v in info_list.items():
                new_k = k
                if len(v) == 0 or type(v[0]) == bool:
                    continue
                mean = np.mean(v)
                std = np.std(v)
                logger.record(new_k + '', mean)
                logger.record(new_k + '_std', std)
                if k not in self.eval_histories.keys():
                    self.eval_histories[new_k] = []
                self.eval_histories[new_k].append(mean)

            if self.top_level_layer is not None:
                self.top_level_layer._dump_logs()
                print("Log path: {}".format(self.log_path))
                if self.early_stop_data_column in self.eval_histories.keys():
                    if self.eval_histories[self.early_stop_data_column][-1] >= self.best_early_stop_val:
                        self.best_early_stop_val = self.eval_histories[self.early_stop_data_column][-1]
                        if self.verbose > 0:
                            print("New best mean {}: {:.5f}!".format(self.early_stop_data_column, self.best_early_stop_val))
                        if self.log_path is not None:
                            self.model.save(os.path.join(self.log_path, "best_model"))

                    if len(self.eval_histories[self.early_stop_data_column]) >= self.early_stop_last_n:
                        mean_val = np.mean(self.eval_histories[self.early_stop_data_column][-self.early_stop_last_n:])
                        if mean_val >= self.early_stop_threshold:
                            print(
                                "Early stop threshold for {} met: Average over last {} evaluations is {:5f} and threshold is {}. Stopping training.".format(
                                    self.early_stop_data_column, self.early_stop_last_n, mean_val,
                                    self.early_stop_threshold))
                            if self.log_path is not None:
                                self.model.save(os.path.join(self.log_path, "early_stop_model"))
                            return False
                else:
                    logger.warn("Warning, early stop data column {} not in eval history keys {}. This should only happen once during initialization.".format(self.early_stop_data_column, self.eval_histories.keys()))
            self.eval_count += 1
        return True



