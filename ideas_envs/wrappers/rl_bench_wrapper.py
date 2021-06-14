from itertools import chain
import numpy as np
from gym.core import Wrapper
import gym.spaces as spaces
from gym.wrappers import TimeLimit
import rlbench.gym  # unused, but do not remove. It registers the RL Bench environments.
from rlbench.backend.conditions import DetectedCondition, NothingGrasped, GraspedCondition, JointCondition, \
    DetectedSeveralCondition, ConditionSet, EmptyCondition
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape


class RLBenchWrapper(Wrapper):
    """
    Wrapper for RL Bench tasks that provides the methods necessary to work with our framework.
    RL Bench tasks consist of "conditions". This wrapper extracts goals from these "conditions",
    so that we can use them for goal-conditioned reinforcement learning.
    """
    def __init__(self, env):
        env = TimeLimit(env, 50)
        super().__init__(env)
        self._max_episode_steps = self.env._max_episode_steps
        self.env.reset()
        obs, _, _, _ = self.step(self.env.unwrapped.action_space.sample())
        self.obs = obs.copy()  # We cache the obs because RL Bench Environments do not have a _get_obs() method.
        obs_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))
        obs_space.spaces['desired_goal'] = self._guess_goal_space(obs_space.spaces['desired_goal'])
        self.observation_space = obs_space
        self.vis = {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        achieved_goal, desired_goal = self._get_goals()
        obs = {
            'observation': obs,
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal
        }
        self.obs = obs.copy()
        info['is_success'] = int(reward)
        reward = reward - 1
        return obs, reward, done, info

    def reset(self, **kwargs):
        # max_episode_steps is updated after __init__(), so we apply the change to the TimeLimit env here
        self.env._max_episode_steps = self._max_episode_steps
        obs = self.env.reset()
        achieved_goal, desired_goal = self._get_goals()
        obs = {
            'observation': obs,
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal
        }
        self.obs = obs.copy()
        return obs

    def _get_obs(self):
        return self.obs

    def _sample_goal(self):
        self.env.unwrapped.reset()
        _, goal = self._get_goals()
        return goal

    def _guess_goal_space(self, goal_space):
        n_samples = 5# FIXME set back to 1000
        goal_space.high = -goal_space.high
        goal_space.low = -goal_space.low
        for _ in range(n_samples):
            goal = self._sample_goal()
            goal_space.high = np.maximum(goal, goal_space.high)
            goal_space.low = np.minimum(goal, goal_space.low)
        # Add some small extra margin.
        goal_space.high += np.abs(goal_space.high - goal_space.low) * 0.01
        goal_space.low -= np.abs(goal_space.high - goal_space.low) * 0.01
        return goal_space

    def _get_goals(self):
        achieved_goal = []
        desired_goal = []
        conditions = []
        for cond in self.env.unwrapped.task._task._success_conditions:
            if isinstance(cond, ConditionSet):
                for c in cond._conditions:
                    conditions.append(c)
            else:
                conditions.append(cond)
        for cond in conditions:
            if isinstance(cond, DetectedCondition):
                desired_goal.append(cond._detector.get_position())
                achieved_goal.append(cond._obj.get_position())
            elif isinstance(cond, (NothingGrasped, GraspedCondition, EmptyCondition)):
                desired_goal.append([1.])
                met, _ = cond.condition_met()
                achieved_goal.append([float(met)])
            elif isinstance(cond, JointCondition):
                desired_goal.append([cond._pos + 1e-20])
                achieved_goal.append([np.abs(cond._joint.get_joint_position() - cond._original_pos)])
            elif isinstance(cond, DetectedSeveralCondition):
                detector_pos = cond._detector.get_position()
                for o in cond._objects:
                    achieved_goal.append(o.get_position())
                    desired_goal.append(detector_pos)
            else:
                raise NotImplementedError("Converting this condition-type to a goal is not supported yet.")
        achieved_goal = np.array(list(chain.from_iterable(achieved_goal)))
        desired_goal = np.array(list(chain.from_iterable(desired_goal)))
        self.goal = desired_goal
        return achieved_goal, desired_goal

    def render(self, **kwargs):
        pass  # CoppeliaSim environments are not explicitly rendered

    def compute_reward(self, achieved_goal, desired_goal, info):
        if len(achieved_goal.shape) == 1:
            achieved_goal = np.array([achieved_goal])
            desired_goal = np.array([desired_goal])
        unmet_conditions = np.zeros(len(achieved_goal))
        conditions = []
        for cond in self.env.unwrapped.task._task._success_conditions:
            if isinstance(cond, ConditionSet):
                # Computing the reward for a ConditionSet like this means that all conditions in the set have to be
                # met simultaneously (instead of in order). This might be impossible in some cases (e.g. when the
                # task is to first move an object to position A and then to position B). However, we can not check
                # whether a condition has been met before here anyway, because we are looking at one time-step at a
                # time.
                for c in cond._conditions:
                    conditions.append(c)
            else:
                conditions.append(cond)
        # loop through all conditions. If there are any unmet conditions, give a reward of -1, else give a reward of 0.
        for cond in conditions:
            if isinstance(cond, DetectedCondition):
                unmet_conditions += self.compute_reward_detected_condition(achieved_goal[:, :3], desired_goal[:, :3], cond)
                achieved_goal, desired_goal = achieved_goal[:, 3:], desired_goal[:, 3:]
            elif isinstance(cond, (NothingGrasped, GraspedCondition, EmptyCondition)):
                unmet_conditions += self.compute_reward_binary_condition(achieved_goal[:, :1])
                achieved_goal, desired_goal = achieved_goal[:, 1:], desired_goal[:, 1:]
            elif isinstance(cond, JointCondition):
                unmet_conditions += self.compute_reward_joint_condition(achieved_goal[:, :1], desired_goal[:, :1])
                achieved_goal, desired_goal = achieved_goal[:, 1:], desired_goal[:, 1:]
            elif isinstance(cond, DetectedSeveralCondition):
                num_obj = len(cond._objects)
                unmet_conditions += self.compute_reward_detected_several_condition(achieved_goal[:, :3 * num_obj],
                                                                                   desired_goal[:, :3 * num_obj], cond)
                achieved_goal, desired_goal = achieved_goal[:, 3 * num_obj:], desired_goal[:, 3 * num_obj:]
            else:
                raise NotImplementedError("This condition-type ({}) is not supported yet.".format(cond.__class__))
        rewards = [-1 if unmet_cond > 0 else 0 for unmet_cond in unmet_conditions]
        return np.array(rewards)

    def compute_reward_detected_condition(self, achieved_goal, desired_goal, cond):
        # For DetectedConditions, we set the positions in the simulation and then query the sensor.
        # HER manipulates some achieved goals (object position), so that they are the same as the desired goal
        # (detector position). Note that the position of the detector itself might NOT be a position that it detects.
        # cache positions to reset them later
        dpos = cond._detector.get_position()
        opos = cond._obj.get_position()

        unmet = np.zeros(len(achieved_goal))
        for i, (ag, dg) in enumerate(zip(achieved_goal, desired_goal)):
            cond._detector.set_position(dg, reset_dynamics=False)
            cond._obj.set_position(ag, reset_dynamics=False)
            r = int(cond.condition_met()[0]) - 1
            unmet[i] -= r

        cond._detector.set_position(dpos, reset_dynamics=False)
        cond._obj.set_position(opos, reset_dynamics=False)
        return unmet

    def compute_reward_binary_condition(self, achieved_goal):
        # Either the gripper has grasped something or not. (Or the container is empty or not.)
        # So we can infer the reward from just the achieved goal.
        achieved_goal = achieved_goal.flatten()
        return 1 - achieved_goal

    def compute_reward_joint_condition(self, achieved_goal, desired_goal):
        achieved_goal = achieved_goal.flatten()
        desired_goal = desired_goal.flatten()
        return 1 - (achieved_goal >= desired_goal)

    def compute_reward_detected_several_condition(self, achieved_goal, desired_goal, cond):
        # This is very similar to compute_reward_detected_condition()
        # cache positions to reset them later
        dpos = cond._detector.get_position()
        opos = [obj.get_position() for obj in cond._objects]

        unmet = np.zeros(len(achieved_goal))
        for i, (ag, dg) in enumerate(zip(achieved_goal, desired_goal)):
            # Attention! We need the following workaround for HER, because HER sets desired_goal=achieved_goal to
            # receive a positive reward but we can not set the position of one detector to be the same as that of
            # n different objects. So we just give a positive reward if desired_goal==achieved_goal
            if np.all(ag == dg):
                unmet[i] = 0
            else:
                cond._detector.set_position(dg[:3], reset_dynamics=False)
                ag = ag.reshape(int(len(ag) / 3), 3)
                for j, pos in enumerate(ag):
                    cond._objects[j].set_position(pos, reset_dynamics=False)
                r = int(cond.condition_met()[0]) - 1
                unmet[i] -= r

        cond._detector.set_position(dpos, reset_dynamics=False)
        for obj, pos in zip(cond._objects, opos):
            obj.set_position(pos, reset_dynamics=False)
        return unmet

    def _is_success(self, achieved_goal, desired_goal):
        return int(self.compute_reward(achieved_goal, desired_goal, []) == 0)

    def display_subgoals(self, subgoals):
        # receives the subgoals in a dict of the form
        # 'subgoal_SHAPEX' : (position_array[x, y, z], float size, rgba[r, g, b, a])
        # where SHAPE is the visualization-shape and X is the number of the subgoal.
        to_visualize = {}
        for k in subgoals:
            to_visualize[k] = {'pos': subgoals[k][0], 'col': subgoals[k][2][:3], 'shape': k[8:-1]}
        self.visualize(to_visualize)

    def visualize(self, names_pos_col={}):
        """
        Takes a dictionary with names, positions, shapes and colors that is structured as follows:
        {'name': {'pos': [0.8, -0.1, 1.1], 'col': [.0, .9, .0], 'shape': 'sphere'},
        'name2': {'pos': [1.0, 0.1, 1.3], 'col': [.0, .0, .9], 'shape': 'box'}}
        Then PrimitiveShapes with the name are created in the specified shape and color and moved to the position.
        """
        for name in names_pos_col:
            if names_pos_col[name]['shape'] == 'box':
                shape = PrimitiveShape.CUBOID
            elif names_pos_col[name]['shape'] == 'sphere':
                shape = PrimitiveShape.SPHERE
            elif names_pos_col[name]['shape'] == 'cylinder':
                shape = PrimitiveShape.CYLINDER
            if name not in self.vis:
                self.vis[name] = Shape.create(shape, [0.04]*3, mass=0, respondable=False, static=True,
                                              position=names_pos_col[name]['pos'], color=names_pos_col[name]['col'])
            else:
                self.vis[name].set_position(names_pos_col[name]['pos'])
                self.vis[name].set_color(names_pos_col[name]['col'])

    @property
    def unwrapped(self):
        # This breaks the unwrapped property because it does not return the pure env.
        # We do this here because the RL Bench environments do not provide methods like _get_obs().
        return self
