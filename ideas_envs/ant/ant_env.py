import numpy as np
from gym.envs.robotics.robot_env import RobotEnv
from gym.envs.robotics.fetch_env import goal_distance


class AntEnv(RobotEnv):
    def __init__(self, model_path, reward_type='sparse', distance_threshold=0.4):
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        super().__init__(model_path=model_path, initial_qpos={}, n_actions=8, n_substeps=20)

    def compute_reward(self, achieved_goal, desired_goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, desired_goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        return -d

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _get_obs(self):
        obs = np.concatenate((self.sim.data.qpos, self.sim.data.qvel))
        achieved_goal = obs[:3]
        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _set_action(self, action):
        self.sim.data.ctrl[:] = action

    def _render_callback(self):
        site_id = self.sim.model.site_name2id("goal")
        self.sim.model.site_pos[site_id] = self.goal[:3].copy()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.sim.step()
        # Return state
        return self._get_obs()

    def _sample_goal(self):
        raise NotImplementedError
