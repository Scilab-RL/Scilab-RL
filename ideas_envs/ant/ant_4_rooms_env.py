import os
import numpy as np
from gym.envs.robotics.robot_env import RobotEnv
from gym.envs.robotics.fetch_env import goal_distance

MODEL_XML_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'ant_4_rooms.xml')


class Ant4RoomsEnv(RobotEnv):
    def __init__(self, distance_threshold=9000, reward_type='sparse'):
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        super().__init__(MODEL_XML_PATH, initial_qpos={}, n_actions=8, n_substeps=20)

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

        # Reset controls
        self.sim.data.ctrl[:] = 0
        while False:
            # Reset joint positions
            for i in range(2):
                self.sim.data.qpos[i] = np.random.uniform(-6, 6)
            # Ensure initial ant position is more than min_dist away from goal
            min_dist = 8
            if np.linalg.norm(self.goal[:2] - self.sim.data.qpos[:2]) > min_dist:
                break
        if True:
            # Choose initial start state to be different than room containing the end goal
            # Determine which of four rooms contains goal
            goal_room = 0
            if self.goal[0] < 0 and self.goal[1] > 0:
                goal_room = 1
            elif self.goal[0] < 0 and self.goal[1] < 0:
                goal_room = 2
            elif self.goal[0] > 0 and self.goal[1] < 0:
                goal_room = 3

            # Place ant in room different than room containing goal

            initial_room = np.random.randint(0, 4)
            while initial_room == goal_room:
                initial_room = np.random.randint(0, 4)

            # Move ant to correct room
            self.sim.data.qpos[0] = np.random.uniform(3, 6.5)
            self.sim.data.qpos[1] = np.random.uniform(3, 6.5)

            # If goal should be in top left quadrant
            if initial_room == 1:
                self.sim.data.qpos[0] *= -1

            # Else if goal should be in bottom left quadrant
            elif initial_room == 2:
                self.sim.data.qpos[0] *= -1
                self.sim.data.qpos[1] *= -1

            # Else if goal should be in bottom right quadrant
            elif initial_room == 3:
                self.sim.data.qpos[1] *= -1

        self.sim.step()

        # Return state
        return self._get_obs()

    def _sample_goal(self):
        goal = np.zeros(3)

        # Randomly select one of the four rooms in which the goal will be located
        room_num = np.random.randint(0, 4)

        # Pick exact goal location
        goal[0] = np.random.uniform(3, 6.5)
        goal[1] = np.random.uniform(3, 6.5)
        goal[2] = np.random.uniform(0.45, 0.55)

        # If goal should be in top left quadrant
        if room_num == 1:
            goal[0] *= -1

        # Else if goal should be in bottom left quadrant
        elif room_num == 2:
            goal[0] *= -1
            goal[1] *= -1

        # Else if goal should be in bottom right quadrant
        elif room_num == 3:
            goal[1] *= -1

        return goal
