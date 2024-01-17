import gymnasium_robotics
import numpy as np


# ParentClass = gymnasium_robotics.envs.maze.ant_maze_v4
from gymnasium.envs.registration import load_env_creator
GymnasiumPointMazeEnvClass = load_env_creator('gymnasium_robotics.envs.maze.point_maze:PointMazeEnv')

class PointGymMod(GymnasiumPointMazeEnvClass):
    metadata = GymnasiumPointMazeEnvClass.metadata
    metadata['render_fps'] = 30
    def __init__(self, distance_threshold=0.45, **kwargs):
        self.distance_threshold = distance_threshold
        super().__init__(**kwargs)
        
    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> float:
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        success = (distance <= self.distance_threshold).astype(np.float64)
        if self.reward_type == "dense":
            return np.exp(-distance)
        elif self.reward_type == "sparse":
            return success
        elif self.reward_type == "sparseneg":
            return success - 1
        else: assert False, "reward type of ant env must be either dense or sparse or sparseneg"

    def compute_terminated(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> bool:
        if not self.continuing_task:
            # If task is episodic terminate the episode when the goal is reached
            return bool(np.linalg.norm(achieved_goal - desired_goal) <= self.distance_threshold)
        else:
            # Continuing tasks don't terminate, episode will be truncated when time limit is reached (`max_episode_steps`)
            return False

    def update_goal(self, achieved_goal: np.ndarray) -> None:
        """Update goal position if continuing task and within goal radius."""

        if (
            self.continuing_task
            and self.reset_target
            and bool(np.linalg.norm(achieved_goal - self.goal) <= self.distance_threshold)
            and len(self.maze.unique_goal_locations) > 1
        ):
            # Generate a goal while within 0.45 of achieved_goal. The distance check above
            # is not redundant, it avoids calling update_target_site_pos() unless necessary
            while np.linalg.norm(achieved_goal - self.goal) <= self.distance_threshold:
                # Generate another goal
                goal = self.generate_target_goal()
                # Add noise to goal position
                self.goal = self.add_xy_position_noise(goal)

            # Update the position of the target site for visualization
            self.update_target_site_pos()
