import gym
from typing import Callable
from util.animation_util import LiveAnimationPlot


class DisplayWrapper(gym.Wrapper):
    """
    Display episodes based on step_trigger, episode_trigger or episode_in_epoch_trigger.
    step_trigger uses step_id,
    episode_trigger uses episode_id,
    episode_of_epoch_trigger uses epoch_id and episode_in_epoch_id
    """
    def __init__(
        self,
        env,
        steps_per_epoch = None,
        episode_in_epoch_trigger: Callable[[int], bool] = None,
        episode_trigger: Callable[[int], bool] = None,
        step_trigger: Callable[[int], bool] = None,
        metric_keys = [],
        logger = None,
    ):
        super().__init__(env)

        trigger_count = sum([x is not None for x in [episode_in_epoch_trigger, episode_trigger, step_trigger]])
        assert trigger_count == 1, "Must specify exactly one trigger"
        steps_per_epoch_trigger_count = sum([x is not None for x in [episode_in_epoch_trigger, steps_per_epoch]])
        assert steps_per_epoch_trigger_count != 1, "If episode_in_epoch_trigger is used, steps_per_epoch must be specified"

        self.episode_in_epoch_trigger = episode_in_epoch_trigger
        self.episode_trigger = episode_trigger
        self.step_trigger = step_trigger

        self.step_in_episode_id = 0
        self.episode_in_epoch_id = 0
        self.epoch_id = 0
        self.episode_id = 0
        self.step_id = 0

        self.steps_per_epoch = steps_per_epoch

        self.displaying = False
        self.is_vector_env = getattr(env, "is_vector_env", False)

        self.metric_keys = metric_keys
        self.num_metrics = len(self.metric_keys)
        self.display_metrics = self.num_metrics > 0
        self.animation = LiveAnimationPlot(y_axis_labels=self.metric_keys) if self.display_metrics else None
        self.logger = logger


    def reset(self, **kwargs):
        observations = self.env.reset(**kwargs)
        if not self.displaying and self._display_enabled():
            self.start_displayer()
        return observations

    def start_displayer(self):
        self.close_displayer()
        # Might be an option to put metric display init here
        self.displaying = True

    def _display_enabled(self):
        if self.step_trigger:
            return self.step_trigger(self.step_id)
        elif self.episode_in_epoch_trigger:
            return self.episode_in_epoch_trigger(self.episode_in_epoch_id, self.epoch_id)
        else:
            return self.episode_trigger(self.episode_id)

    def step(self, action):
        observations, rewards, dones, infos = self.env.step(action)

        # increment steps, episodes and epochs
        if self.steps_per_epoch:
            epoch_id_tmp = self.epoch_id
            self.epoch_id = self.step_id // self.steps_per_epoch
            if self.epoch_id > epoch_id_tmp:
                self.episode_in_epoch_id = 0
        self.step_id += 1
        self.step_in_episode_id +=1
        if not self.is_vector_env:
            if dones:
                self.episode_id += 1
                if self.steps_per_epoch:
                    self.episode_in_epoch_id += 1
        elif dones[0]:
            self.episode_id += 1
            if self.steps_per_epoch:
                self.episode_in_epoch_id += 1

        if self.displaying:
            self.env.render(mode='human')
            # metrics stuff
            if self.display_metrics:
                for i in range(self.num_metrics):
                    self.curr_recorded_value = self.logger.name_to_value[self.metric_keys[i]]
                    self.animation.x_data[i].append(self.step_in_episode_id)
                    self.animation.y_data[i].append(self.curr_recorded_value)
                self.animation.start_animation()

            if not self.is_vector_env:
                if dones:
                    self.close_displayer()
            elif dones[0]:
                self.close_displayer()

        elif self._display_enabled():
            self.start_displayer()

        if not self.is_vector_env:
            if dones:
                self.step_in_episode_id = 0
        elif dones[0]:
            self.step_in_episode_id = 0
        return observations, rewards, dones, infos

    def close_displayer(self) -> None:
        if self.displaying:
            # Metric stuff
            if self.display_metrics:
                self.animation.reset_fig()
                # reset data
                self.animation.x_data = [[] for _ in range(len(self.metric_keys))]
                self.animation.y_data = [[] for _ in range(len(self.metric_keys))]
            #close metric displayer
        self.displaying = False
