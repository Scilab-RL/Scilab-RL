import gymnasium_robotics
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv

# ParentClass = gymnasium_robotics.envs.maze.ant_maze_v4
from gymnasium.envs.registration import load_env_creator
GymnasiumAntMazeEnvClass = load_env_creator('gymnasium_robotics.envs.maze.ant_maze_v4:AntMazeEnv')
CAMERA_CONFIG = {
        "distance": 5.0,
    }
class AntGymMod(GymnasiumAntMazeEnvClass):
    metadata = GymnasiumAntMazeEnvClass.metadata
    metadata['render_fps'] = 30

    def __init__(self, distance_threshold=0.45, frame_skip=5, **kwargs):
        self.distance_threshold = distance_threshold
        super().__init__(**kwargs)
        self.ant_env.frame_skip = frame_skip
        return
        
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

    def step(self, action):
        ant_obs, _, _, _, info = self.ant_env.step(action)
        obs = self._get_obs(ant_obs)

        reward = self.compute_reward(obs["achieved_goal"], self.goal, info)
        terminated = self.compute_terminated(obs["achieved_goal"], self.goal, info)
        truncated = self.compute_truncated(obs["achieved_goal"], self.goal, info)
        info["success"] = bool(np.linalg.norm(obs["achieved_goal"] - self.goal) <= self.distance_threshold)
        # TODO: observarion[0]=0.25 seems to denote that ant has flipped. add penalty for this.
        if self.render_mode == "human":
            self.render()

        # Update the goal position if necessary
        self.update_goal(obs["achieved_goal"])

        return obs, reward, terminated, truncated, info

