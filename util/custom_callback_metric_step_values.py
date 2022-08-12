from stable_baselines3.common.callbacks import BaseCallback
from util.plot_animation_episode import LiveAnimationPlot
import matplotlib.pyplot as plt

import mlflow


class DisplayMetricCallBack(BaseCallback):
    """
    This callback
    param logger: THe logger in which the recorded value can be accessed.
    param auto_close: Automatically generated Plots
    param episodic: visualize episodic metric values
    param n_episodes: The number of episodes over which to average the metric.
    """

    def __init__(
            self,
            metric_key,
            logger,
            episodic=True,
            auto_close=True
    ):
        super(DisplayMetricCallBack, self).__init__(verbose=0)
        self.animation_started = False,
        self.curr_recorded_value = None
        self.new_animation = False
        self.num_iteration = 0
        self.logger = logger
        self.auto_close = auto_close
        self.episodic = episodic
        self.metric_key = metric_key

        self.animation = LiveAnimationPlot()

    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        if self.episodic:
            if self.auto_close:
                plt.close()
            self.animation = LiveAnimationPlot()

            # reset data
            self.animation.x_data = []
            self.animation.y_data = []

    def _on_step(self) -> bool:
        self.curr_recorded_value = self.logger.name_to_value[self.metric_key]
        self.animation.x_data.append(self.num_iteration)
        self.animation.y_data.append(self.curr_recorded_value)
        self.animation.start_animation()
        self.num_iteration = self.num_iteration + 1
        return True
