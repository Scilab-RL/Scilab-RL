import gymnasium_robotics
import numpy as np


# ParentClass = gymnasium_robotics.envs.maze.ant_maze_v4
from gymnasium.envs.registration import load_env_creator
GymnasiumAntMazeEnvClass = load_env_creator('gymnasium_robotics.envs.maze.ant_maze_v4:AntMazeEnv')

class AntGymMod(GymnasiumAntMazeEnvClass):
    metadata = GymnasiumAntMazeEnvClass.metadata
    metadata['render_fps'] = 30
    def __init__(self, distance_threshold=0.45, **kwargs):
        self.distance_threshold = distance_threshold
        super().__init__(**kwargs)
    # EnvSpec(id='AntMaze_Open_Diverse_GR-v4', entry_point='gymnasium_robotics.envs.maze.ant_maze_v4:AntMazeEnv',
    #         reward_threshold=None, nondeterministic=False, max_episode_steps=700, order_enforce=True, autoreset=False,
    #         disable_env_checker=False, apply_api_compatibility=False, kwargs={
    #         'maze_map': [[1, 1, 1, 1, 1, 1, 1], [1, 'c', 'c', 'c', 'c', 'c', 1], [1, 'c', 'c', 'c', 'c', 'c', 1],
    #                      [1, 'c', 'c', 'c', 'c', 'c', 1], [1, 1, 1, 1, 1, 1, 1]], 'reward_type': 'sparse'},
    #         namespace=None, name='AntMaze_Open_Diverse_GR', version=4, additional_wrappers=(), vector_entry_point=None)
    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> float:
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        if self.reward_type == "dense":
            return np.exp(-distance)
        elif self.reward_type == "sparse":
            positive_reward = (distance <= self.distance_threshold).astype(np.float64)
            negative_reward = positive_reward - 1
            return negative_reward
        else: assert False, "reward type of ant env must be either dense or sparse"

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
