from stable_baselines3.common.callbacks import BaseCallback
from util.plot_multiple_metrics_animation import LiveAnimationPlot
import mlflow


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
        save_anim=False

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
        self.curr_rollout = 0
        self.save_anim = save_anim
        self.animation = LiveAnimationPlot(y_axis_labels=self.metric_keys)
        self.num_metrics = len(self.metric_keys)
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
        for i in range(self.num_metrics):
            self.curr_recorded_value = self.logger.name_to_value[self.metric_keys[i]]
            self.animation.x_data[i].append(self.num_iteration)
            self.animation.y_data[i].append(self.curr_recorded_value)
        self.animation.start_animation()
        self.num_iteration = self.num_iteration + 1
        return True

    def _on_training_end(self) -> None:
        if not self.episodic and self.save_anim:
            self.animation.save_animation('metric_anim_all_rollouts')
