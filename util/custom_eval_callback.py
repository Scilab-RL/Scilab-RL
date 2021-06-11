import os
from typing import Union

import cv2
import gym
import numpy as np
from stable_baselines3.common import logger
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization

from util.custom_evaluation import evaluate_policy


class CustomEvalCallback(EvalCallback):
    """
    Callback for evaluating an agent.

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best agent according to the ``mean_reward``
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every eval_freq call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render_train: none -> don't render during training
                         display -> render and display during training
                         record -> render and record during training
    :param render_test: Same as render train but during evaluation
    :param verbose:
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: str = None,
        deterministic: bool = True,
        render_train: str = 'none',
        render_test: str = 'none',
        verbose: int = 1,
        early_stop_data_column: str = 'test/success_rate',
        early_stop_threshold: float = 1.0,
        early_stop_last_n: int = 5,
        agent: OffPolicyAlgorithm = None
    ):
        super(EvalCallback, self).__init__(verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.best_mean_success = -np.inf
        self.deterministic = deterministic
        self.render_train = render_train
        self.render_test = render_test
        self.best_model_save_path = None
        eval_history_column_names = ['test/mean_reward', 'test/success_rate']
        self.eval_histories = {}
        for name in eval_history_column_names:
            self.eval_histories[name] = []
        self.early_stop_data_column = early_stop_data_column
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_last_n = early_stop_last_n
        self.agent = agent

        eval_env = BaseAlgorithm._wrap_env(eval_env)

        if isinstance(eval_env, VecEnv):
            assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"

        self.eval_env = eval_env
        self.log_path = log_path
        self.best_agent_save_path = None
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []

        self.vid_size = 1024, 768
        self.vid_fps = 25
        self.eval_count = 0
        self.train_count = 0
        self.render_train_info = None
        self.render_test_info = None
        self.last_step_was_train = False
        self.video_writer = None

        if self.render_train == 'record':
            self.render_train_info = {'size': self.vid_size, 'fps': self.vid_fps, 'train_count': self.train_count,
                                      'path': self.log_path}
        elif self.render_train == 'display':
            self.render_train_info = {'mode': 'human'}
        if self.render_test == 'record':
            self.render_test_info = {'size': self.vid_size, 'fps': self.vid_fps, 'eval_count': self.eval_count,
                                     'path': self.log_path}
        elif self.render_test == 'display':
            self.render_test_info = {'mode': 'human'}

    def _on_step(self, log_prefix='') -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.video_writer is not None and self.last_step_was_train:
                # save the training video
                self.video_writer.release()
                self.video_writer = None
                self.last_step_was_train = False
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)
            if self.render_test_info is not None:
                self.render_test_info['eval_count'] = self.eval_count
            episode_rewards, episode_lengths, episode_successes = evaluate_policy(
                self.agent,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                render_info=self.render_test_info,
            )
            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            mean_success, std_success = np.mean(episode_successes), np.std(episode_successes)

            if self.verbose > 0:
                logger.info(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                logger.info(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

            logger.record("test/mean_reward", float(mean_reward))
            logger.record("test/std_reward", float(std_reward))
            logger.record("test/mean_ep_length", mean_ep_length)
            logger.record("test/success_rate", mean_success)

            self.eval_histories['test/success_rate'].append(mean_success)
            self.eval_histories['test/mean_reward'].append(mean_reward)

            # if mean_reward > self.best_mean_reward:
            #     if self.verbose > 0:
            #         logger.info("New best mean reward!")
            #     if self.best_agent_save_path is not None:
            #         self.agent.save(os.path.join(self.best_agent_save_path, "best_agent"))
            #     self.best_mean_reward = mean_reward
            #     # Trigger callback if needed
            #     if self.callback is not None:
            #         return self._on_event()
            if mean_success > self.best_mean_success:
                if self.verbose > 0:
                    logger.info("New best mean success rate!")
                if self.log_path is not None:
                    self.agent.save(os.path.join(self.log_path, "best_agent"))
                self.best_mean_success = mean_success
            if self.agent is not None:
                self.agent._dump_logs()
            if len(self.eval_histories[self.early_stop_data_column]) >= self.early_stop_last_n:
                mean_val = np.mean(self.eval_histories[self.early_stop_data_column][-self.early_stop_last_n:])
                if mean_val >= self.early_stop_threshold:
                    logger.info("Early stop threshold for {} met: Average over last {} evaluations is {} and threshold is {}. Stopping training.".format(self.early_stop_data_column, self.early_stop_last_n, mean_val, self.early_stop_threshold))
                    if self.log_path is not None:
                        self.agent.save(os.path.join(self.log_path, "early_stop_agent"))
                    return False
            self.eval_count += 1
        else:
            if self.render_train == 'display':
                if hasattr(self.training_env, 'venv'):
                    env = self.training_env.venv.envs[0]
                else:
                    env = self.training_env
                env.render(mode=self.render_train_info['mode'])
            elif self.render_train == 'record':
                if not self.last_step_was_train:
                    self.train_count += 1
                    self.render_train_info['train_count'] = self.train_count
                    # create a new VideoWriter for each episode
                    try:
                        self.video_writer = cv2.VideoWriter(
                            self.render_train_info['path'] + '/train_{}.avi'.format(self.render_train_info['train_count']),
                            cv2.VideoWriter_fourcc('F', 'M', 'P', '4'),
                            self.render_train_info['fps'], self.render_train_info['size'])
                    except:
                        logger.info("Error creating video writer")
                else:
                    if hasattr(self.training_env, 'venv'):
                        frame = self.training_env.venv.envs[0].render(mode='rgb_array',
                                                                      width=self.render_train_info['size'][0],
                                                                      height=self.render_train_info['size'][1])
                    else:
                        frame = self.training_env.render(mode='rgb_array')
                    self.video_writer.write(frame)
            self.last_step_was_train = True

        return True
