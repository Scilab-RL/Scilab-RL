import os
import numpy as np
from src.custom_envs.ant.ant_env import AntEnv

MODEL_XML_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'ant_reacher.xml')


class AntReacherEnv(AntEnv):
    def __init__(self, reward_type='sparse', distance_threshold=0.4):
        super().__init__(MODEL_XML_PATH, reward_type, distance_threshold)

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Reset controls
        self.sim.data.ctrl[:] = 0

        while True:
            # Reset joint positions
            for i in range(2):
                self.sim.data.qpos[i] = self.np_random.uniform(-6, 6)
            # Ensure initial ant position is more than min_dist away from goal
            min_dist = 8
            if np.linalg.norm(self.goal[:2] - self.sim.data.qpos[:2]) > min_dist:
                break

        self.sim.step()
        return self._get_obs()

    def _sample_goal(self):
        goal = np.zeros(3)
        goal[0] = self.np_random.uniform(-6.5, 6.5)
        goal[1] = self.np_random.uniform(-6.5, 6.5)
        goal[2] = self.np_random.uniform(0.45, 0.55)
        return goal
