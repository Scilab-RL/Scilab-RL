from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, EventCallback
from typing import Any, Callable, Dict, List, Optional, Union
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
import gym
import numpy as np
import os
import warnings
from util.custom_evaluation import evaluate_policy
from ideas_baselines.mbchac.hiearchical_evaluation import evaluate_hierarchical_policy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common import logger
from collections import deque
import time
from stable_baselines3.common.utils import safe_mean
from ideas_baselines.mbchac.hierarchical_env import get_h_envs_from_env

class HierarchicalEvalCallback(EvalCallback):
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
        eval_env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: str = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        early_stop_data_column: str = 'test/success_rate',
        early_stop_threshold: float = 1.0,
        early_stop_last_n: int = 5,
        top_level_model = None
    ):
        super(EvalCallback, self).__init__(verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_early_stop_val = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.eval_histories = {}
        self.early_stop_data_column = early_stop_data_column
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_last_n = early_stop_last_n
        self.top_level_model = top_level_model

        layer_envs = get_h_envs_from_env(eval_env, top_level_model.time_scales, env_list=[], is_testing_env=True, model=top_level_model)
        eval_env = layer_envs[0]
        eval_env = BaseAlgorithm._wrap_env(eval_env)
        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        if isinstance(eval_env, VecEnv):
            assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"

        self.eval_env = eval_env
        self.log_path = log_path
        self.best_model_save_path = None
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []


    def _on_step(self, log_prefix='') -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            info_list = evaluate_hierarchical_policy(
                self.top_level_model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                # deterministic=self.deterministic,
                # return_episode_rewards=True,
            )
            for k,v in info_list:
                try:
                    mean = np.mean(v)
                    std = np.std(v)
                    logger.record(k+'_mean', mean)
                    logger.record(k + '_std', std)
                    if k not in self.eval_histories.keys():
                        self.eval_histories[k] = []
                    self.eval_histories[k].append(mean)
                except:
                    pass
            # for layer, ep_rew, ep_len, ep_succ in zip(range(len(episode_rewards)), episode_rewards, episode_lengths, episode_successes):
            #     mean_reward, std_reward = np.mean(ep_rew), np.std(ep_rew)
            #     mean_ep_length, std_ep_length = np.mean(ep_len), np.std(ep_len)
            #     mean_success, std_success = np.mean(ep_succ), np.std(ep_succ)
            #
            #     if self.model.is_top_layer:
            #         test_pf = "test"
            #     else:
            #         test_pf = "test{}".format(str(layer))
            #     logger.record(test_pf + "/mean_reward", float(mean_reward))
            #     logger.record(test_pf + "/std_reward", float(std_reward))
            #     logger.record(test_pf + "/mean_ep_length", mean_ep_length)
            #     logger.record(test_pf + "/success_rate", mean_success)
            #
            #     k = test_pf + "/success_rate"
            #     if k not in self.eval_histories.keys():
            #         self.eval_histories[k] = []
            #     self.eval_histories[k].append(mean_success)
            #
            #     k = test_pf + "/mean_reward"
            #     if k not in self.eval_histories.keys():
            #         self.eval_histories[k] = []
            #     self.eval_histories[k].append(mean_reward)
            #
            #     k = test_pf + "/std_reward"
            #     if k not in self.eval_histories.keys():
            #         self.eval_histories[k] = []
            #     self.eval_histories[k].append(std_reward)
            #
            #     k = test_pf + "/mean_ep_length"
            #     if k not in self.eval_histories.keys():
            #         self.eval_histories[k] = []
            #     self.eval_histories[k].append(mean_ep_length)
            if self.top_level_model is not None:
                self.top_level_model._dump_logs()
                if self.early_stop_data_column in self.eval_histories.keys():
                    if self.eval_histories[self.early_stop_data_column] >= self.best_early_stop_val:
                        if self.verbose > 0:
                            print("New best mean {}!".format(self.best_early_stop_val))
                        if self.log_path is not None:
                            self.model.save(os.path.join(self.log_path, "best_model"))
                        self.best_early_stop_val = self.eval_histories[self.early_stop_data_column]
                    if len(self.eval_histories[self.early_stop_data_column]) >= self.early_stop_last_n:
                        mean_val = np.mean(self.eval_histories[self.early_stop_data_column][-self.early_stop_last_n:])
                        if mean_val >= self.early_stop_threshold:
                            logger.info("Early stop threshold for {} met: Average over last {} evaluations is {} and threshold is {}. Stopping training.".format(self.early_stop_data_column, self.early_stop_last_n, mean_val, self.early_stop_threshold))
                            if self.log_path is not None:
                                self.model.save(os.path.join(self.log_path, "early_stop_model"))
                            return False
                else:
                    logger.warn("Warning, early stop data column {} not in eval history keys {}. This should only happen once during initialization.".format(self.early_stop_data_column, self.eval_histories.keys()))

        return True



