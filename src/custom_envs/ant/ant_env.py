import gymnasium_robotics
import numpy as np


# ParentClass = gymnasium_robotics.envs.maze.ant_maze_v4
from gymnasium.envs.registration import load_env_creator
ParentClass = load_env_creator('gymnasium_robotics.envs.maze.ant_maze_v4:AntMazeEnv')

class AntGymMod(ParentClass):
    def __init__(self, **kwargs):
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
            positive_reward = (distance <= 0.45).astype(np.float64)
            negative_reward = positive_reward - 1
            return negative_reward