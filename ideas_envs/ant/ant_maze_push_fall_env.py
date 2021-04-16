"""adapted from https://github.com/tensorflow/models/tree/master/research/efficient-hrl/environments"""

import os
import numpy as np
from gym import spaces
from ideas_envs.ant.ant_env import AntEnv
from ideas_envs.ant.xml_creator import create_xml

MODEL_XML_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'ant_maze_push_fall.xml')
MAX_GOAL_DIST = 5.0


def get_reward_fn(task):
    if task in ['Maze', 'Push']:
        return lambda obs, goal: -np.sum(np.square(obs[:2] - goal)) ** 0.5
    if task == 'Fall':
        return lambda obs, goal: -np.sum(np.square(obs[:3] - goal)) ** 0.5
    assert False, 'Unknown env'


def get_subgoal_bounds(task):
    if task == 'Maze':
        return np.array([[-3.75, 19.75], [-3.75, 19.75]])
    if task == 'Push':
        return np.array([[-11.75, 11.75], [-3.75, 21.75]])
    if task == 'Fall':
        return np.array([[-5.75, 15.75], [-5.75, 29.75], [4.5, 4.5]])
    assert False, 'Unknown env'


def success_fn(last_reward):
    return last_reward > -MAX_GOAL_DIST


class AntMazePushFallEnv(AntEnv):
    def __init__(self, reward_type='sparse', distance_threshold=0.4, task='Maze'):
        xml_path = create_xml(MODEL_XML_PATH, maze_id=task)
        self.task = task
        self.evaluate = False
        self.g_slice = 3 if self.task == 'Fall' else 2
        self.reward_fn = get_reward_fn(task)
        self.subgoal_bound = get_subgoal_bounds(task)
        super().__init__(xml_path, reward_type, distance_threshold)
        low = np.array([-10, -10, -0.5, -1, -1, -1, -1,
                        -0.5, -0.3, -0.5, -0.3, -0.5, -0.3, -0.5, -0.3])
        self.observation_space['observation'].low[:15] = low
        self.observation_space['observation'].high[:15] = -low

    def _obs2goal(self, obs):
        return obs[:self.g_slice].copy()

    def _get_obs(self):
        obs = np.concatenate((self.sim.data.qpos, self.sim.data.qvel))
        achieved_goal = obs[:self.g_slice]
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
                return np.array([0., 16.])
            sampled_goal = self.np_random.uniform((-4, -4), (20, 20))
            # end goal should not be inside wall
            while sampled_goal[0] < 12.25 and sampled_goal[1] < 12.25 and sampled_goal[1] > 3.75:
                sampled_goal = self.np_random.uniform((-4, -4), (20, 20))
            return sampled_goal
        if self.task == 'Push':
            return np.array([0., 19.])
        if self.task == 'Fall':
            return np.array([0., 27., 4.5])
        assert False, 'Unknown env'

    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = self.reward_fn(achieved_goal, desired_goal)
        self.success = success_fn(reward)
        if self.reward_type == 'sparse':
            reward = -1 + int(self.success)
        return reward

    def _render_callback(self):
        pass
