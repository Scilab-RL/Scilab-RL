import mlflow
from stable_baselines3.common.callbacks import BaseCallback


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
