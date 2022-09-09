from stable_baselines3.common.callbacks import BaseCallback
from util.plot_animation_episode import LiveAnimationPlot
import matplotlib.pyplot as plt
import inspect
import mlflow


class DisplayMetricCallBack(BaseCallback):
    """
    This callback
    param logger: THe logger in which the recorded value can be accessed.
    param episodic: visualize episodic metric values
    param save_anim: bool indicating whether to save the animation

    """

    def __init__(
            self,
            metric_key,
            logger,
            episodic=True,
            save_anim=False

    ):
        super(DisplayMetricCallBack, self).__init__(verbose=0)
        self.animation_started = False,
        self.curr_recorded_value = None
        self.new_animation = False
        self.logger = logger
        self.num_iteration = 0
        self.episodic = episodic
        self.metric_key = metric_key
        self.animation = None
        self.curr_rollout = 0
        self.save_anim = save_anim
        #self.animation = LiveAnimationPlot(y_axis_label=self.metric_key)

    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        if not self.curr_rollout:
            self.animation = LiveAnimationPlot(y_axis_label=self.metric_key)
        if self.episodic and self.curr_rollout:
            if self.save_anim:
                self.animation.save_animation('metric_anim_' + str(self.curr_rollout))
            plt.close()
            # reset data
            self.animation.x_data = []
            self.animation.y_data = []
            self.num_iteration = 0
            self.animation = LiveAnimationPlot(y_axis_label=self.metric_key)
        self.curr_rollout = self.curr_rollout + 1

    def _on_step(self) -> bool:
        self.curr_recorded_value = self.logger.name_to_value[self.metric_key]
        self.animation.x_data.append(self.num_iteration)
        self.animation.y_data.append(self.curr_recorded_value)
        self.animation.start_animation()
        self.num_iteration = self.num_iteration + 1
        return True

    def _on_training_end(self) -> None:
        if not self.episodic and self.save_anim:
            self.animation.save_animation('metric_anim_all_rollouts')
