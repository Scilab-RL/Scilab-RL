from stable_baselines3.common.callbacks import BaseCallback
from util.plot_animation_episode import LiveAnimationPlot
import matplotlib.pyplot as plt

import mlflow


class DisplayMetricCallBack(BaseCallback):
    """
    This callback
    param metric: The metric to consider for early stopping.
    param eval_freq: The frequency of evaluation, so that this callback is only called after each evaluation.
    param threshold: The early-stopping-threshold for the metric-average value.
    param n_episodes: The number of episodes over which to average the metric.
    """

    def __init__(
            self,
            logger
    ):
        super(DisplayMetricCallBack, self).__init__(verbose=0)
        self.animation_started = False,
        self.curr_recorded_value = None
        self.new_animation = False
        # There are no values to use in the first iteration and
        # so we have to skip this step
        self.num_iteration = 0
        self.logger = logger
        self.animation = LiveAnimationPlot()

    # to be deleted
    def _on_training_start(self) -> None:
        self.num_iteration = 0

    def _on_step(self) -> bool:
        # if self.num_iteration > 0:
        print("Currently in _on_step and iteration number {}! ######################".format(self.num_iteration))
        # client = mlflow.tracking.MlflowClient()
        # recorded_value_history = client.get_metric_history(mlflow.active_run().info.run_id, 'val_to_record')
        # self.curr_recorded_value = recorded_value_history[-1].value
        self.curr_recorded_value = self.logger.name_to_value['val_to_record']
        print(self.curr_recorded_value)
        self.animation.x_data.append(self.num_iteration)
        self.animation.y_data.append(self.curr_recorded_value)
        self.animation.start_animation()
        self.num_iteration = self.num_iteration + 1
        '''
        else:
            self.first_iteration = True
            self.new_animation = LiveAnimationPlot()
            self.animation_started = False
        '''

        return True
