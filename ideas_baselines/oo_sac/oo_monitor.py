import gym
import numpy as np
from typing import Optional, Tuple

from stable_baselines3.common.monitor import Monitor
from gym.envs.robotics.robot_env import RobotEnv


class OO_Monitor(Monitor):
    def __init__(
            self,
            env: gym.Env,
            filename: Optional[str] = None,
            allow_early_resets: bool = True,
            reset_keywords: Tuple[str, ...] = (),
            info_keywords: Tuple[str, ...] = (),
    ):
        super(OO_Monitor, self).__init__(
            env=env,
            filename=filename,
            allow_early_resets=allow_early_resets,
            reset_keywords=reset_keywords,
            info_keywords=info_keywords
        )
        env._is_success = self._is_success
        env.goal_distance = self.goal_distance

    def _is_success(self, achieved_goal, desired_goal):
        d = self.goal_distance(achieved_goal[:3], desired_goal[:3])
        return (d < self.distance_threshold).astype(np.float32)

    def goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)


class OO_RobotEnv(RobotEnv):
    def __init__(self, model_path, initial_qpos, n_actions, n_substeps):
        super(OO_RobotEnv, self).__init__(
            model_path=model_path,
            initial_qpos=initial_qpos,
            n_actions=n_actions,
            n_substeps=n_substeps
        )

    def _is_success(self, achieved_goal, desired_goal):
        d = self.goal_distance(achieved_goal[:3], desired_goal[:3])
        return (d < self.distance_threshold).astype(np.float32)
