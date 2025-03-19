from typing import Any, Callable, SupportsFloat

import gymnasium as gym
from gymnasium.spaces import Space, Box
from gymnasium.core import ActType, ObsType, WrapperActType

"""
This code is from https://github.com/dohmjan/aerl/
"""


class SleepingTransformAction(
    gym.Wrapper[ObsType, WrapperActType, ObsType, ActType], gym.utils.RecordConstructorArgs
):
    """Applies an inactive function to the ``action`` before passing the modified value to the environment ``step`` function.

    Adapted from https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/wrappers/transform_action.py
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        func: Callable[[WrapperActType], ActType],
        action_space: Space[WrapperActType] | None,
        toggle_at_step: int | list[int] | float | None = None,
        toggle_at_episode: int | list[int] | float | None = None
    ):
        """Initialize SleepingTransformAction.

        Args:
            env: The environment to wrap.
            func: Function to apply to the :meth:`step`'s ``action``
            action_space: The updated action space of the wrapper given the function.
            toggle_at_step: Decides whether to apply transformation. If type is int, it's considered
            True for all steps ongoing. If type is list[int], that represents a list of steps at
            which it's alternately toggled True/False. If type is float, it's considered True for
            that frequency of steps.
            toggle_at_episode: Decides whether to apply transformation. If type is int, it's
            considered True for all episodes ongoing. If type is list[int], that represents a list
            of episodes at which it's alternately toggled True/False. If type is float, it's
            considered True for that frequency of episodes.
        """
        gym.utils.RecordConstructorArgs.__init__(
            self,
            func=func,
            action_space=action_space,
            toggle_at_step=toggle_at_step,
            toggle_at_episode=toggle_at_episode
        )
        gym.Wrapper.__init__(self, env)

        if action_space is not None:
            self.action_space = action_space

        self.step_count = 0
        self.episode_count = 0

        self.func = func

        assert (toggle_at_step is None) != (toggle_at_episode is None), (
            "Make sure either toggle_at_step or toggle_at_episode is defined, not both."
        )
        if isinstance(toggle_at_step, float):
            assert 0.0 <= toggle_at_step <= 1.0, "Toggle frequency has to be between 0.0 and 1.0."
        elif isinstance(toggle_at_step, int):
            toggle_at_step = [toggle_at_step]
        elif isinstance(toggle_at_step, list):
            toggle_at_step = sorted(toggle_at_step)
        if isinstance(toggle_at_episode, float):
            assert 0.0 <= toggle_at_episode <= 1.0, "Toggle frequency has to be between 0.0 and 1.0."
        elif isinstance(toggle_at_episode, int):
            toggle_at_episode = [toggle_at_episode]
        elif isinstance(toggle_at_episode, list):
            toggle_at_episode = sorted(toggle_at_episode)

        self._toggle_at_step = toggle_at_step
        self._toggle_at_episode = toggle_at_episode

    def step(
        self, action: WrapperActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Runsthe :attr:`env` :meth:`env.step` using the modified ``action`` from :meth:`self.action`."""
        obs, reward, terminated, truncated, info = self.env.step(self.action(action))
        self.step_count += 1
        if truncated or terminated:
            self.episode_count += 1
        return obs, reward, terminated, truncated, info

    def action(self, action: WrapperActType) -> ActType:
        """Apply function to action."""
        return self.func(action) if self.is_active() else action

    def is_active(self):
        if self._toggle_at_step is not None:
            if isinstance(self._toggle_at_step, list):
                return sum([self.step_count >= t for t in self._toggle_at_step]) % 2 == 1
            elif isinstance(self._toggle_at_step, float):
                return (self.step_count * self._toggle_at_step) % 1 == 0
            else:
                raise NotImplementedError()
        elif self._toggle_at_episode is not None:
            if isinstance(self._toggle_at_episode, list):
                return sum([self.episode_count >= t for t in self._toggle_at_episode]) % 2 == 1
            elif isinstance(self._toggle_at_episode, float):
                return (self.episode_count * self._toggle_at_episode) % 1 == 0
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

class InvertAction(
    SleepingTransformAction[ObsType, WrapperActType, ActType], gym.utils.RecordConstructorArgs
):
    """Inverts one or all dimensions of ``action`` which is passed to ``step``.

    Example:
        >>> import gymnasium as gym
        >>> import numpy as np
        >>> from aerl import InvertAction
        >>> env = gym.make("Hopper-v4", disable_env_checker=True)
        >>> env = InvertAction(env, dim=0, toggle_at_step=0)
        >>> _ = env.reset(seed=42)
        >>> _ = env.step(np.array([0.5, 0.5, 0.5], dtype=np.float32))
        ... # Executes the action np.array([-0.5, 0.5, 0.5]) in the base environment
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        dim: int | None = None,
        toggle_at_step: int | list[int] | float | None = None,
        toggle_at_episode: int | list[int] | float | None = None
    ):
        """A wrapper for inverting one or all dimensions of the continuous action.

        Args:
            env: The environment to wrap.
            dim: Action dimension that is transformed. None corresponds to all dimensions.
            toggle_at_step: Decides whether to apply transformation. If type is int, it's considered
            True for all steps ongoing. If type is list[int], that represents a list of steps at
            which it's alternately toggled True/False. If type is float, it's considered True for
            that frequency of steps.
            toggle_at_episode: Decides whether to apply transformation. If type is int, it's
            considered True for all episodes ongoing. If type is list[int], that represents a list
            of episodes at which it's alternately toggled True/False. If type is float, it's
            considered True for that frequency of episodes.
        """
        assert isinstance(env.action_space, Box)

        def _action_func(action):
            _action = action.copy()
            if dim is not None:
                _action[dim] *= -1
            else:
                _action *= -1
            return _action

        gym.utils.RecordConstructorArgs.__init__(self)
        SleepingTransformAction.__init__(
            self,
            env=env,
            func=_action_func,
            action_space=None,
            toggle_at_step=toggle_at_step,
            toggle_at_episode=toggle_at_episode
        )