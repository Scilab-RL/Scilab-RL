import gymnasium_robotics
import numpy as np


# ParentClass = gymnasium_robotics.envs.maze.ant_maze_v4
from gymnasium.envs.registration import load_env_creator
GymnasiumPointMazeEnvClass = load_env_creator('gymnasium_robotics.envs.maze.point_maze:PointMazeEnv')

class MazeMap:
    RESET = R = "r"  # Initial Reset position of the agent
    GOAL = G = "g"
    COMBINED = C = "c"  # These cells can be selected as goal or reset locations

    OPEN = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ]
    OPEN_DIVERSE_G = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, R, G, G, G, G, 1],
        [1, G, G, G, G, G, 1],
        [1, G, G, G, G, G, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ]
    OPEN_DIVERSE_GR = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, C, C, C, C, C, 1],
        [1, C, C, C, C, C, 1],
        [1, C, C, C, C, C, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ]
    SMALL_OPEN_DIVERSE_GR = [
        [1, 1, 1, 1, 1],
        [1, C, C, C, 1],
        [1, C, C, C, 1],
        [1, C, C, C, 1],
        [1, 1, 1, 1, 1],
    ]
    SMALL_OPEN_DIVERSE_G = [
        [1, 1, 1, 1, 1],
        [1, G, G, G, 1],
        [1, G, G, G, 1],
        [1, G, G, G, 1],
        [1, 1, 1, 1, 1],
    ]
    MEDIUM_CUSTOM_DIVERSE_GR = [[1, 1, 1, 1, 1, 1, 1, 1],
                              [1, C, C, 1, 1, C, C, 1],
                              [1, C, C, 1, C, C, C, 1],
                              [1, 1, C, C, C, 1, 1, 1],
                              [1, C, C, 1, C, C, C, 1],
                              [1, C, 1, C, C, 1, C, 1],
                              [1, C, C, C, 1, C, C, 1],
                              [1, 1, 1, 1, 1, 1, 1, 1]]
    name2map = {"open": OPEN,
                "open_dg": OPEN_DIVERSE_G,
                "open_dgr": OPEN_DIVERSE_GR,
                "small_open_dg": SMALL_OPEN_DIVERSE_G,
                "small_open_dgr": SMALL_OPEN_DIVERSE_GR,
                "medium_custom_dgr": MEDIUM_CUSTOM_DIVERSE_GR,
                }

class PointGymMod(GymnasiumPointMazeEnvClass):
    metadata = GymnasiumPointMazeEnvClass.metadata
    metadata['render_fps'] = 30
    def __init__(self, distance_threshold=0.45, **kwargs):
        kwargs["maze_map"] = MazeMap.name2map[kwargs["maze_map"]]
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
