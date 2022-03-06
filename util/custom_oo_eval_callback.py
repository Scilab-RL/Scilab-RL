from util.custom_eval_callback import CustomEvalCallback

import os

import cv2
import numpy as np
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization

from util.custom_oo_evaluation import evaluate_oo_policy


class CustomOOEvalCallback(CustomEvalCallback):
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
            episode_rewards, episode_lengths, episode_successes = evaluate_oo_policy(
                self.agent,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                render_info=self.render_test_info,
                logger=self.logger,
                distance_threshold=self.eval_env.envs[0].distance_threshold
            )
            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            mean_success, std_success = np.mean(episode_successes), np.std(episode_successes)

            if self.verbose > 0:
                self.logger.info(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                self.logger.info(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

            self.logger.record("test/mean_reward", float(mean_reward))
            self.logger.record("test/std_reward", float(std_reward))
            self.logger.record("test/mean_ep_length", mean_ep_length)
            self.logger.record("test/success_rate", mean_success)

            self.eval_histories['test/success_rate'].append(mean_success)
            self.eval_histories['test/mean_reward'].append(mean_reward)

            # if mean_reward > self.best_mean_reward:
            #     if self.verbose > 0:
            #         self.logger.info("New best mean reward!")
            #     if self.best_agent_save_path is not None:
            #         self.agent.save(os.path.join(self.best_agent_save_path, "best_agent"))
            #     self.best_mean_reward = mean_reward
            #     # Trigger callback if needed
            #     if self.callback is not None:
            #         return self._on_event()
            if mean_success > self.best_mean_success:
                if self.verbose > 0:
                    self.logger.info("New best mean success rate!")
                if self.log_path is not None:
                    self.agent.save(os.path.join(self.log_path, "best_agent"))
                self.best_mean_success = mean_success
            if self.agent is not None:
                self.agent._dump_logs()
            if len(self.eval_histories[self.early_stop_data_column]) >= self.early_stop_last_n:
                mean_val = np.mean(self.eval_histories[self.early_stop_data_column][-self.early_stop_last_n:])
                if mean_val >= self.early_stop_threshold:
                    self.logger.info("Early stop threshold for {} met: Average over last {} evaluations is {} and threshold is {}. Stopping training.".format(self.early_stop_data_column, self.early_stop_last_n, mean_val, self.early_stop_threshold))
                    if self.log_path is not None:
                        self.agent.save(os.path.join(self.log_path, "early_stop_agent"))
                    return False
            self.eval_count += 1
        else:
            if not self.last_step_was_train:
                self.train_count += 1
            if (self.train_count-1) % self.render_every_n_train == 0:
                if self.render_train == 'display':
                    if hasattr(self.training_env, 'venv'):
                        env = self.training_env.venv.envs[0]
                    else:
                        env = self.training_env
                    env.render(mode=self.render_train_info['mode'])
                elif self.render_train == 'record':
                    if not self.last_step_was_train:
                        self.render_train_info['train_count'] = self.train_count
                        # create a new VideoWriter for each episode
                        try:
                            self.video_writer = cv2.VideoWriter(
                                self.render_train_info['path'] + '/train_{}.avi'.format(self.train_count-1),
                                cv2.VideoWriter_fourcc('F', 'M', 'P', '4'),
                                self.render_train_info['fps'], self.render_train_info['size'])
                        except:
                            self.logger.info("Error creating video writer")
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