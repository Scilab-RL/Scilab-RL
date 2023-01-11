import random
import numpy as np


def apply_oo_wrapper(env, n_attrs, n_objects):
    env._original_get_obs = env._get_obs
    env._original_sample_goal = env._sample_goal
    t = OOWrappper(env, n_objects, n_attrs)
    env._get_obs = t._get_obs
    env._sample_goal = t._sample_goal
    return env


def expand_goal_vector(obs, goal):
    original_len = obs['achieved_goal'].shape[0]
    len_diff = abs(len(goal) - original_len)
    if len_diff != 0:
        goal = np.expand_dims(np.concatenate([goal, np.zeros(len_diff)]), axis=0)[0]
    return goal


class OOInterface:
    def _get_obs(self):
        raise NotImplementedError()

    def _sample_goal(self):
        raise NotImplementedError()


class OOWrappper:
    def __init__(self, base_env, n_objects, n_attrs=3):
        self.base_env = base_env
        self.n_attrs = n_attrs
        self.n_objects = n_objects

    def _get_obs(self):
        obs = self.base_env._original_get_obs()
        obj_idx = np.where(self.base_env.goal[:-3] != 0)[0][0]
        oneHot_idx = np.eye(self.n_objects)[obj_idx]
        #assert # Assert that it is divisable

        # reshape goal vector
        self.base_env.goal = expand_goal_vector(obs, self.base_env.goal)
        obs['desired_goal'] = self.base_env.goal

        achieved_goal = obs['achieved_goal']
        achieved_oo_values = list(achieved_goal[obj_idx * 3: obj_idx * 3 + 3])
        achieved_oo_goal = np.array(list(oneHot_idx) + achieved_oo_values)
        obs['achieved_goal'] = expand_goal_vector(obs, achieved_oo_goal)
        return obs.copy()

    def _sample_goal(self):
        goal = self.base_env._original_sample_goal()

        # pick object index
        obj_idx = random.randint(0, self.n_objects - 1)
        oneHot_idx = np.eye(self.n_objects)[obj_idx]

        # build oo_obs
        oo_values = list(goal[obj_idx * 3: obj_idx * 3 + 3])
        oo_goal = np.array(list(oneHot_idx) + oo_values)
        return oo_goal.copy()
