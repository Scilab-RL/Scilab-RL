"""adapted from https://github.com/tensorflow/models/tree/master/research/efficient-hrl/environments"""

import os
import numpy as np
from ideas_envs.ant.ant_env import AntEnv
from ideas_envs.ant.xml_creator import create_xml

MODEL_XML_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'ant_maze_push_fall.xml')
MAX_GOAL_DIST = 5.0


def get_subgoal_bounds(task):
    if task == 'Maze':
        return np.array([[-3.75, 19.75], [-3.75, 19.75], [0.5, 0.5]])
    if task == 'Push':
        return np.array([[-11.75, 11.75], [-3.75, 21.75], [0.5, 0.5]])
    if task == 'Fall':
        return np.array([[-5.75, 15.75], [-5.75, 29.75], [4.5, 4.5]])
    assert False, 'Unknown env'


class AntMazePushFallEnv(AntEnv):
    def __init__(self, reward_type='sparse', distance_threshold=0.4, task='Maze'):
        xml_path = create_xml(MODEL_XML_PATH, maze_id=task)
        self.task = task
        self.evaluate = False
        self.subgoal_bound = get_subgoal_bounds(task)

        super().__init__(xml_path, reward_type, distance_threshold)

        low = np.array([-10, -10, -0.5, -1, -1, -1, -1,
                        -0.5, -0.3, -0.5, -0.3, -0.5, -0.3, -0.5, -0.3])
        self.observation_space['observation'].low[:15] = low
        self.observation_space['observation'].high[:15] = -low
        site_id = self.sim.model.site_name2id("goal")
        self.sim.model.site_size[site_id][0] = MAX_GOAL_DIST

    def _set_action(self, action):
        super()._set_action(action*30)

    def _obs2goal(self, obs):
        return obs[:3].copy()

    def _get_obs(self):
        obs = np.concatenate((self.sim.data.qpos, self.sim.data.qvel))
        achieved_goal = obs[:3]
        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Reset controls
        self.sim.data.ctrl[:] = 0

        self.sim.step()
        return self._get_obs()

    def _sample_goal(self):
        if self.task == 'Maze':
            # NOTE: When evaluating, a fixed goal is used (paper HIRO)
            if self.evaluate:
                return np.array([0., 16., 0.5])
            sampled_goal = self.np_random.uniform((-4, -4, 0.5), (20, 20, 0.5))
            # end goal should not be inside wall
            while sampled_goal[0] < 12.25 and sampled_goal[1] < 12.25 and sampled_goal[1] > 3.75:
                sampled_goal = self.np_random.uniform((-4, -4, 0.5), (20, 20, 0.5))
            return sampled_goal
        if self.task == 'Push':
            return np.array([0., 19., 0.5])
        if self.task == 'Fall':
            return np.array([0., 27., 4.5])
        assert False, 'Unknown env'

    def _render_callback(self):
        site_id = self.sim.model.site_name2id("goal")
        self.sim.model.site_pos[site_id][:3] = self.goal.copy()
