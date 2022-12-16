import gym
from typing import Callable

from gym import logger

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
    ):
        super().__init__(env)

        trigger_count = sum([x is not None for x in [episode_in_epoch_trigger, episode_trigger, step_trigger]])
        assert trigger_count == 1, "Must specify exactly one trigger"
        steps_per_epoch_trigger_count = sum([x is not None for x in [episode_in_epoch_trigger, steps_per_epoch]])
        assert steps_per_epoch_trigger_count != 1, "If episode_in_epoch_trigger is used, steps_per_epoch must be specified"

        self.episode_in_epoch_trigger = episode_in_epoch_trigger
        self.episode_trigger = episode_trigger
        self.step_trigger = step_trigger

        self.episode_in_epoch_id = 0
        self.epoch_id = 0
        self.episode_id = 0
        self.step_id = 0

        self.steps_per_epoch = steps_per_epoch

        self.displaying = False
        self.is_vector_env = getattr(env, "is_vector_env", False)

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
            if not self.is_vector_env:
                if dones:
                    self.close_displayer()
            elif dones[0]:
                self.close_displayer()

        elif self._display_enabled():
            self.start_displayer()

        return observations, rewards, dones, infos

    def close_displayer(self) -> None:
        if self.displaying:
            pass
            #close metric displayer
        self.displaying = False
