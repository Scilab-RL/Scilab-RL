import mlflow
import os
import warnings
import gym

import numpy as np

from typing import Any, Dict, Optional, Union
from util.custom_evaluation import evaluate_policy
from util.animation_util import LiveAnimationPlot
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from stable_baselines3.common.callbacks import BaseCallback, EventCallback


class EarlyStopCallback(BaseCallback):
    """
    This callback checks whether to stop the experiment early because the agent is already good enough.
    If the agent achieved an average value better than *threshold* for the *metric* over the last *n_episodes*,
    it ends the training and saves an early-stopping agent.
    param metric: The metric to consider for early stopping.
    param eval_freq: The frequency of evaluation, so that this callback is only called after each evaluation.
    param threshold: The early-stopping-threshold for the metric-average value.
    param n_episodes: The number of episodes over which to average the metric.
    """

    def __init__(
            self,
            metric: str = 'eval/success_rate',
            eval_freq: int = 2000,
            threshold: float = 0.9,
            n_episodes: int = 3
    ):
        super(EarlyStopCallback, self).__init__(verbose=0)
        self.metric = metric
        self.eval_freq = eval_freq
        self.threshold = threshold
        self.n_episodes = n_episodes

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            client = mlflow.tracking.MlflowClient()
            hist = client.get_metric_history(mlflow.active_run().info.run_id, self.metric)
            data_val_hist = [h.value for h in hist]
            if len(data_val_hist) >= self.n_episodes:
                avg = sum(data_val_hist[-self.n_episodes:])/self.n_episodes
                if avg >= self.threshold:
                    self.logger.info(f"Early stop threshold for {self.metric} met: "
                                     f"Average over last {self.n_episodes} evaluations is {avg} "
                                     f"and threshold is {self.threshold}. Stopping training.")
                    return False
        return True


class DisplayMetricCallBack(BaseCallback):
    """
    This callback
    param logger: THe logger in which the recorded value can be accessed.
    param episodic: visualize episodic metric values
    param save_anim: bool indicating whether to save the animation

    """

    def __init__(
        self,
        metric_keys,
        logger,
        episodic=True,
        save_anim=False,
        display_nth_rollout=1
    ):
        super(DisplayMetricCallBack, self).__init__(verbose=0)
        self.animation_started = False,
        self.curr_recorded_value = []
        self.new_animation = False
        self.logger = logger
        self.num_iteration = 0
        self.episodic = episodic
        self.metric_keys = metric_keys
        self.animation = None
        self.curr_rollout = -1
        self.save_anim = save_anim
        self.animation = LiveAnimationPlot(y_axis_labels=self.metric_keys)
        self.num_metrics = len(self.metric_keys)
        self.display_nth_rollout = display_nth_rollout
    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        '''
        if not self.curr_rollout:
            self.animation = LiveAnimationPlot(y_axis_label=self.metric_key)
        '''
        if self.episodic and self.curr_rollout:
            if self.save_anim:
                self.animation.save_animation('metric_anim_' + str(self.curr_rollout))
            self.animation.reset_fig()
            # reset data
            self.animation.x_data = [[] for _ in range(len(self.metric_keys))]
            self.animation.y_data = [[] for _ in range(len(self.metric_keys))]
            self.num_iteration = 0

            #self.animation = LiveAnimationPlot(y_axis_label=self.metric_key)
        self.curr_rollout = self.curr_rollout + 1

    def _on_step(self) -> bool:
        if self.curr_rollout % self.display_nth_rollout == 0:
            for i in range(self.num_metrics):
                self.curr_recorded_value = self.logger.name_to_value[self.metric_keys[i]]
                self.animation.x_data[i].append(self.num_iteration)
                self.animation.y_data[i].append(self.curr_recorded_value)
            self.animation.start_animation()
            self.num_iteration = self.num_iteration + 1
        return True

    def _on_training_end(self) -> None:
        if not self.episodic and self.save_anim and self.curr_rollout % self.display_nth_rollout == 0:
            self.animation.save_animation('metric_anim_all_rollouts')


class EvalCallback(EventCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    :param callback_metric_viz: Callback to handle metric visualization.
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: str = None,
        best_model_save_path: str = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        callback_metric_viz = None,
    ):
        super(EvalCallback, self).__init__(callback_on_new_best, verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn
        self.callback_metric_viz = callback_metric_viz

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            # Reset success rate buffer
            self._is_success_buffer = []

            # Start metric visualization
            if self.callback_metric_viz:
                self.callback_metric_viz._on_rollout_start()

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
                callback_metric_viz = self.callback_metric_viz
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "rl_model_best"))
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)
