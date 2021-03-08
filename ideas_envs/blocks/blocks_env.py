import os
import numpy as np
from gym.utils import EzPickle
from gym.envs.robotics import fetch_env, rotations, utils

MODEL_XML_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'blocks.xml')


class BlocksEnv(fetch_env.FetchEnv, EzPickle):
    """
    Environment for block-stacking.
    """

    def __init__(self, n_objects, gripper_goal, reward_type='sparse'):
        """
        :param n_objects: How many blocks should be stacked
        :param gripper_goal: 3 possibilities:
            gripper_none: The position of the gripper is not relevant for the goal
            gripper_random: The gripper should reach a random position after stacking the blocks
            gripper_above: The gripper should be above the stacked blocks
        :param reward_type: whether the reward should be sparse or dense
        """
        initial_qpos = {
            # robot xyz
            'robot0:slide0': -0.65,
            'robot0:slide1': 0,
            'robot0:slide2': 0,
        }

        self.n_objects = n_objects
        self.gripper_goal = gripper_goal

        self.goal_size = self.n_objects * 3
        if self.gripper_goal != 'gripper_none':
            self.goal_size += 3

        has_object = self.n_objects > 0

        super().__init__(MODEL_XML_PATH, has_object=has_object, block_gripper=False, n_substeps=20,
                         gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
                         obj_range=0.15, target_range=0.15, distance_threshold=0.05,
                         initial_qpos=initial_qpos, reward_type=reward_type)
        EzPickle.__init__(self)

    def _set_action(self, action):
        super(BlocksEnv, self)._set_action(np.array([0.2, 0, 0, 0.1]))

    def _get_obs(self):
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt

        object_pos, object_rot, object_velp, object_velr = (np.empty(self.n_objects * 3) for _ in range(4))
        for i in range(self.n_objects):
            # position
            object_pos[i * 3:(i + 1) * 3] = self.sim.data.get_geom_xpos('object0')
            # rotation
            object_rot[i * 3:(i + 1) * 3] = rotations.mat2euler(self.sim.data.get_geom_xmat('object0'))
            # velocities
            object_velp[i * 3:(i + 1) * 3] = self.sim.data.get_geom_xvelp('object0') * dt
            object_velr[i * 3:(i + 1) * 3] = self.sim.data.get_geom_xvelr('object0') * dt

        obs = np.concatenate([grip_pos, object_pos, gripper_state, object_rot,
                              grip_velp, object_velp, object_velr, gripper_vel])
        achieved_goal = obs[:self.goal_size]

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _reset_sim(self):
        super(BlocksEnv, self)._reset_sim()
        return True

    def _sample_goal(self):
        return super(BlocksEnv, self)._sample_goal()

    def _env_setup(self, initial_qpos):
        super(BlocksEnv, self)._env_setup(initial_qpos)
        # Die Unterschiede sind:
        # 1.
        # self.random_gripper_goal_pos_offset = (0.0, 0.0, 0.14)
        # 2.
        # if self.n_objects > 0 statt if self.has_object

# Was hat es mit der tower_height auf sich? Brauchen wir die?
# --> scheint überflüssig zu sein, da sie eh immer == n_objects sein soll
