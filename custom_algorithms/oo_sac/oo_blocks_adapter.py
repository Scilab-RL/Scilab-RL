import os
import random

import numpy as np
from gym.utils import EzPickle
from gym.envs.robotics import fetch_env, rotations, utils

from custom_envs.blocks.blocks_env import BlocksEnv


class OOBlocksAdapter(BlocksEnv):
    """
    Adapter enables use of OO_SAC with Blocks environment
    """

    def _get_obs(self):
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        gripper_state = robot_qpos[-2:] #np.concatenate((robot_qpos[-2:], [0]))
        gripper_vel = robot_qvel[-2:] #np.concatenate((robot_qvel[-2:] * dt, [0]))

        object_pos, object_rot, object_velp, object_velr = (np.empty(self.n_objects * 3) for _ in range(4))
        for i in range(self.n_objects):
            # position
            object_pos[i * 3:(i + 1) * 3] = self.sim.data.get_geom_xpos('object{}'.format(i))
            # rotation
            object_rot[i * 3:(i + 1) * 3] = rotations.mat2euler(self.sim.data.get_geom_xmat('object{}'.format(i)))
            # velocities
            object_velp[i * 3:(i + 1) * 3] = self.sim.data.get_geom_xvelp('object{}'.format(i)) * dt
            object_velr[i * 3:(i + 1) * 3] = self.sim.data.get_geom_xvelr('object{}'.format(i)) * dt

        obs = np.concatenate([grip_pos, object_pos, gripper_state, object_rot,
                              grip_velp, object_velp, object_velr, gripper_vel])
        # 1. get object index from self.goal (any index != 0 is object, since 0 is gripper index)
        obj_idx = np.where(self.goal[:-3] != 0)[0][0]
        self.obj_idx = obj_idx
        # 2. Detect the values of achieved_goal that correspond to self.goal
        if self.gripper_goal == 'gripper_none':
            # TODO: gripper_none is probably not working for any object number
            oo_achieved_goal = obs[3:3 + self.goal_size]
        else:
            # one hot vector of goal object + object position from observation
            oo_achieved_goal = np.append(self.goal[:1 + self.n_objects], obs[obj_idx * 3:obj_idx * 3 + 3])

        oo_achieved_goal = self._reshape_goal_vec(oo_achieved_goal)
        self.goal = self._reshape_goal_vec(self.goal)
        return {
            'observation': obs.copy(),
            'achieved_goal': oo_achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _reshape_goal_vec(self, goal):
        original_len = self.observation_space['achieved_goal'].shape[0]
        len_diff = abs(len(goal) - original_len)
        if len_diff != 0:
            goal = np.expand_dims(np.concatenate([goal, np.zeros(len_diff)]), axis=0)[0]
        return goal
    """
    def _render_callback(self):
        start_idx = 0
        goal_len = self.goal[3:].shape[0]
        if self.gripper_goal != 'gripper_none':
            start_idx = 3
            site_id = self.sim.model.site_name2id('gripper_goal')
            pos_len = self.sim.model.site_pos[site_id].shape[0]
            len_diff = abs(pos_len - goal_len)
            self.sim.model.site_pos[site_id] = self.goal[3:-len_diff].copy()
        for i in range(self.n_objects):
            site_id = self.sim.model.site_name2id('object{}_goal'.format(i))
            self.sim.model.site_pos[site_id] = self._get_obs()['observation'][
                                               start_idx + 3 * i:start_idx + 3 * i + 3].copy()
    """
    def _sample_goal(self):
        full_goal = np.empty((self.n_objects + 1) * 3)
        if self.n_objects > 0:
            # Find a random position for the tower
            lowest_block_xy = self.initial_gripper_xpos[:2] \
                              + self.np_random.uniform(-self.target_range, self.target_range, size=2)
            # the height z is set below table height for now because we add the object height later
            z = np.array([self.table_height - self.object_height * 0.5])
            # if the gripper position is included in the goal, leave the first 3 values free
            if self.gripper_goal == 'gripper_none':
                start_idx = 0
            else:
                start_idx = 3
            # set goal positions for the blocks
            z += self.object_height
            for i in range(self.n_objects):
                # z += self.object_height
                # pos = np.concatenate([lowest_block_xy, z])
                block_xy = self.initial_gripper_xpos[:2] \
                           + self.np_random.uniform(-self.target_range, self.target_range, size=2)
                pos = np.concatenate([block_xy, z])
                full_goal[start_idx + i * 3:start_idx + i * 3 + 3] = pos

            # set the gripper_goal, if there is any
            if not self.gripper_goal == 'gripper_none':
                if self.gripper_goal == 'gripper_above' or self.gripper_goal == 'object_move':
                    #z += 2 * self.object_height
                    full_goal[:3] = self.initial_gripper_xpos[:3] \
                              + self.np_random.uniform(-self.target_range, self.target_range, size=3)#np.concatenate([lowest_block_xy, z])
                if self.gripper_goal == 'gripper_random':
                    too_close = True
                    while too_close:
                        grip_goal = self.initial_gripper_xpos \
                                    + self.np_random.uniform(-self.target_range, self.target_range, size=3)
                        closest_dist = np.inf
                        for i in range(self.n_objects):
                            closest_dist = min(closest_dist, np.linalg.norm(grip_goal - full_goal[3 + 3 * i:6 + 3 * i]))
                        if closest_dist > self.sample_dist_threshold:
                            too_close = False
                    full_goal[:3] = grip_goal
        else:
            # n_objects == 0 is only possible with 'gripper_random'
            full_goal[:] = self.initial_gripper_xpos \
                           + self.np_random.uniform(-self.target_range, self.target_range, size=3)

        #  E.g. Let gripper be object 0 and block be object 1 and x,y,z are goal coordinates:
        # Then if you randomly decide to choose a goal for the block: oogoal = (0, 1, x,y,z)
        # obj_idx = 1
        #self.full_goal = full_goal
        if self.gripper_goal == 'object_move':
            obj_idx = random.randint(1, self.n_objects)
        else:
            obj_idx = random.randint(0, self.n_objects)
        # Number of objects + 1 (gripper) to get one-hot representation of object in env
        oneHot_idx = np.eye(self.n_objects + 1)[obj_idx]
        oo_values = list(full_goal[obj_idx * 3: obj_idx * 3 + 3])
        oo_goal = np.array(list(oneHot_idx) + oo_values)
        # Example, the gripper (object 0) should be at the values below.
        # oo_goal = np.array([1, 0, -0.05111022,  0.03454098,  0.525])
        return oo_goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = fetch_env.goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)
