"""adapted from https://github.com/tensorflow/models/tree/master/research/efficient-hrl/environments"""

import os
import numpy as np
from ideas_envs.ant.ant_env import AntEnv
from ideas_envs.ant.xml_creator import create_xml

MODEL_XML_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'ant_maze_push_fall.xml')


class AntMazePushFallEnv(AntEnv):
    def __init__(self, reward_type='sparse', distance_threshold=0.4, task='Maze'):
        xml_path = create_xml(MODEL_XML_PATH, maze_id=task)
        super().__init__(xml_path, reward_type, distance_threshold)

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Reset controls
        self.sim.data.ctrl[:] = 0

        while True:
            # Reset joint positions
            for i in range(2):
                self.sim.data.qpos[i] = np.random.uniform(-6, 6)
            # Ensure initial ant position is more than min_dist away from goal
            min_dist = 8
            if np.linalg.norm(self.goal[:2] - self.sim.data.qpos[:2]) > min_dist:
                break

        self.sim.step()
        return self._get_obs()

    def _sample_goal(self):
        goal = np.zeros(3)
        goal[0] = np.random.uniform(-6.5, 6.5)
        goal[1] = np.random.uniform(-6.5, 6.5)
        goal[2] = np.random.uniform(0.45, 0.55)
        return goal

    def _render_callback(self):
        pass
