from typing import Union, Optional
import gym
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv


class EarlyStopEvalCallback(EvalCallback):
    """
    Adds the functionality of early stopping dependent on the eval/success_rate to the EvalCallback.
    :param threshold: threshold
    """

    def __init__(
            self,
            eval_env: Union[gym.Env, VecEnv],
            threshold: float = 0.9,
            n_episodes: int = 3,
            callback_on_new_best: Optional[BaseCallback] = None,
            n_eval_episodes: int = 5,
            eval_freq: int = 10000,
            log_path: str = None,
            best_model_save_path: str = None,
            deterministic: bool = True,
            render: bool = False,
            verbose: int = 1,
            warn: bool = True,
    ):
        super(EarlyStopEvalCallback, self).__init__(eval_env=eval_env, callback_on_new_best=callback_on_new_best,
                                                    n_eval_episodes=n_eval_episodes, eval_freq=eval_freq,
                                                    log_path=log_path, best_model_save_path=best_model_save_path,
                                                    deterministic=deterministic, render=render, verbose=verbose,
                                                    warn=warn)
        self.threshold = threshold
        self.n_episodes = n_episodes

    def _on_step(self) -> bool:
        assert self.parent is not None, "``StopTrainingOnMinimumReward`` callback must be used with an ``EvalCallback``"
        continue_training = bool(self.parent.best_mean_reward < self.reward_threshold)
        if self.verbose > 0 and not continue_training:
            print(
                f"Stopping training because the mean reward {self.parent.best_mean_reward:.2f} "
                f" is above the threshold {self.reward_threshold}"
            )
        return continue_training
