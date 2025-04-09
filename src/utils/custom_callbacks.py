import os
import mlflow
import numpy as np
from omegaconf import OmegaConf
from typing import Dict, Any

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import sync_envs_normalization
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym

from utils.environment_pertubation_wrapper import NoiseAction

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


class EvalCallback(EvalCallback):
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
    """

    def _log_data_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]
        maybe_is_success = None
        if locals_["done"]:
            if "is_success" in info.keys():
                maybe_is_success = info.get("is_success")
            elif "success" in info.keys():
                maybe_is_success = info.get("success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)
        if 'rewards' in locals_.keys():
            reward = float(locals_['rewards'][0])
            self.logger.record('eval/rollout_rewards_step', reward)

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_data_callback,
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
            # for meta-world environments, "is_success" is named "success"
            maybe_success = info.get("success")
            if maybe_success is not None:
                self._is_success_buffer.append(maybe_success)

class PostTrainingNoiseEvalCallback(BaseCallback):
    def __init__(self, cfg, run_dir, mlflow_run, verbose=0):
        super().__init__(verbose)
        self.cfg = cfg
        self.run_dir = run_dir
        self.mlflow_run = mlflow_run

    def _on_training_end(self) -> None:
        self.logger.info("Running post-training evaluation with action noise...")

        base_env = self.training_env.envs[0]  # unwrap base gym env from DummyVecEnv
        noise_levels = [0.0, 0.1, 0.5, 1.0, 2.5, 5.0]
        results = {}

        for noise in noise_levels:
            try:
                noisy_env = NoiseAction(base_env, value=noise, toggle_at_episode=0)
                eval_env = DummyVecEnv([lambda: noisy_env])

                success_list = []
                for _ in range(200):
                    obs = eval_env.reset()
                    done = False
                    while not done:
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, reward, done, info = eval_env.step(action)
                    success_list.append(info[0].get("is_success", 0.0))

                success_rate = float(np.mean(success_list))
                results[noise] = success_rate
                self.logger.record(f"eval/success_rate_noise_{noise}", success_rate)
                mlflow.log_metric(f"success_rate_noise_{noise}", success_rate)

                eval_env.close()

            except Exception as e:
                self.logger.warn(f"Noise level {noise} failed: {e}")
                results[noise] = -1

        file_path = os.path.join(self.run_dir, "action_noise_eval_results.yaml")
        OmegaConf.save(config=OmegaConf.create(results), f=file_path)
        mlflow.log_artifact(file_path)

    def _on_step(self) -> bool:
        return True

class PostTrainingNoiseRewardEvalCallback(BaseCallback):
    def __init__(self, cfg, run_dir, mlflow_run, verbose=0):
        super().__init__(verbose)
        self.cfg = cfg
        self.run_dir = run_dir
        self.mlflow_run = mlflow_run

    def _on_training_end(self) -> None:
        self.logger.info("Running post-training evaluation with action noise (avg reward)...")

        base_env = self.training_env.envs[0]  # unwrap base env from DummyVecEnv
        noise_levels = [0.0, 0.1, 0.5, 1.0, 2.5, 5.0, 20, 200]
        results = {}

        for noise in noise_levels:
            try:
                noisy_env = NoiseAction(base_env, value=noise, toggle_at_episode=0)
                eval_env = DummyVecEnv([lambda: noisy_env])

                episode_rewards = []
                for _ in range(200):
                    obs = eval_env.reset()
                    done = False
                    total_reward = 0
                    while not done:
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, reward, done, info = eval_env.step(action)
                        total_reward += reward
                    episode_rewards.append(total_reward)

                avg_reward = float(np.mean(episode_rewards))
                results[noise] = avg_reward
                self.logger.record(f"eval/avg_reward_noise_{noise}", avg_reward)
                mlflow.log_metric(f"avg_reward_noise_{noise}", avg_reward)

                eval_env.close()

            except Exception as e:
                self.logger.warn(f"Noise level {noise} failed: {e}")
                results[noise] = "error"

        file_path = os.path.join(self.run_dir, "action_noise_avg_reward_results.yaml")
        OmegaConf.save(config=OmegaConf.create(results), f=file_path)
        mlflow.log_artifact(file_path)

    def _on_step(self) -> bool:
        return True

class PostTrainingGravityEvalCallback(BaseCallback):
    def __init__(self, cfg, run_dir, mlflow_run, verbose=0):
        super().__init__(verbose)
        self.cfg = cfg
        self.run_dir = run_dir
        self.mlflow_run = mlflow_run

    def _on_training_end(self) -> None:
        self.logger.info("Running post-training evaluation with modified gravity...")

        # Gravity values to test (Z-axis)
        gravity_values = [0, -9.81, -50, -100, -300, -500]
        results = {}

        for g in gravity_values:
            try:
                # Create a fresh environment each time to avoid weird state carryover
                def make_env_with_gravity():
                    env = gym.make(self.cfg.env, **self.cfg.env_kwargs)
                    env.unwrapped.model.opt.gravity[2] += g
                    return env

                eval_env = DummyVecEnv([make_env_with_gravity])

                success_list = []
                for _ in range(200):
                    obs = eval_env.reset()
                    done = False
                    while not done:
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, reward, done, info = eval_env.step(action)
                    success_list.append(info[0].get("is_success", 0.0))

                success_rate = float(np.mean(success_list))
                results[g] = success_rate
                self.logger.record(f"eval/success_rate_gravity_{g}", success_rate)
                mlflow.log_metric(f"success_rate_gravity_{g}", success_rate)

                eval_env.close()

            except Exception as e:
                self.logger.warn(f"Gravity {g} failed: {e}")
                results[g] = -1

        file_path = os.path.join(self.run_dir, "gravity_eval_results.yaml")
        OmegaConf.save(config=OmegaConf.create(results), f=file_path)
        mlflow.log_artifact(file_path)

    def _on_step(self) -> bool:
        return True

class PostTrainingTorqueEvalCallback(BaseCallback):
    def __init__(self, cfg, run_dir, mlflow_run, verbose=0):
        super().__init__(verbose)
        self.cfg = cfg
        self.run_dir = run_dir
        self.mlflow_run = mlflow_run

    def _on_training_end(self) -> None:
        self.logger.info("Running post-training evaluation with reduced actuator torque...")

        # Torque multipliers to test
        torque_scales = [1.0, 0.75, 0.5, 0.25, 0.1]
        results = {}

        for scale in torque_scales:
            try:
                def make_env_with_torque_limit():
                    env = gym.make(self.cfg.env, **self.cfg.env_kwargs)
                    # Correct path to access sim inside Fetch envs
                    env.unwrapped.model.actuator_ctrlrange[:, 0] *= scale
                    env.unwrapped.model.actuator_ctrlrange[:, 1] *= scale
                    return env

                eval_env = DummyVecEnv([make_env_with_torque_limit])

                success_list = []
                for _ in range(200):
                    obs = eval_env.reset()
                    done = False
                    while not done:
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, reward, done, info = eval_env.step(action)
                    success_list.append(info[0].get("is_success", 0.0))

                success_rate = float(np.mean(success_list))
                results[scale] = success_rate
                self.logger.record(f"eval/success_rate_torque_{scale}", success_rate)
                mlflow.log_metric(f"success_rate_torque_{scale}", success_rate)

                eval_env.close()

            except Exception as e:
                self.logger.warn(f"Torque scale {scale} failed: {e}")
                results[scale] = -1

        file_path = os.path.join(self.run_dir, "torque_eval_results.yaml")
        OmegaConf.save(config=OmegaConf.create(results), f=file_path)
        mlflow.log_artifact(file_path)

    def _on_step(self) -> bool:
        return True

class PostTrainingGravityRewardEvalCallback(BaseCallback):
    def __init__(self, cfg, run_dir, mlflow_run, verbose=0):
        super().__init__(verbose)
        self.cfg = cfg
        self.run_dir = run_dir
        self.mlflow_run = mlflow_run

    def _on_training_end(self) -> None:
        self.logger.info("Running post-training evaluation with scaled gravity (avg reward)...")

        # Detect default gravity value from a fresh env
        default_env = gym.make(self.cfg.env, **self.cfg.env_kwargs)
        base_gravity = default_env.unwrapped.g
        default_env.close()
        self.logger.info(f"Detected base gravity: {base_gravity}")

        # Gravity scaling factors to test (1.0 = default gravity first)
        gravity_scales = [1.0, 0.5, 1.5, 2.0, 3.0, 5.0]
        results = {}

        for scale in gravity_scales:
            g = base_gravity * scale
            try:
                def make_env_with_gravity():
                    env = gym.make(self.cfg.env, **self.cfg.env_kwargs)
                    env.unwrapped.g = g
                    return env

                eval_env = DummyVecEnv([make_env_with_gravity])

                episode_rewards = []
                for _ in range(200):
                    obs = eval_env.reset()
                    done = False
                    total_reward = 0
                    while not done:
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, reward, done, info = eval_env.step(action)
                        total_reward += reward
                    episode_rewards.append(total_reward)

                avg_reward = float(np.mean(episode_rewards))
                results[g] = avg_reward
                self.logger.record(f"eval/avg_reward_gravity_{g}", avg_reward)
                mlflow.log_metric(f"avg_reward_gravity_{g}", avg_reward)

                eval_env.close()

            except Exception as e:
                self.logger.warn(f"Gravity {g} (scale {scale}) failed: {e}")
                results[g] = "error"

        file_path = os.path.join(self.run_dir, "gravity_scaled_avg_reward_results.yaml")
        OmegaConf.save(config=OmegaConf.create(results), f=file_path)
        mlflow.log_artifact(file_path)

    def _on_step(self) -> bool:
        return True