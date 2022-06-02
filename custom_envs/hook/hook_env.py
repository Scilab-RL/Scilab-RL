import os
import numpy as np
from gym.envs.robotics import rotations
from custom_envs.blocks.blocks_env import BlocksEnv

MODEL_XML_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'hook.xml')
POS_MIN, POS_MAX = [0.28, -0.15, 0.42], [0.34, 0.15, 0.42]


class HookEnv(BlocksEnv):
    """
    This environment enhances the Blocks environment with a hook.
    The hook can be used to pull objects that are out of reach.
    """
    def __init__(self, n_objects, reward_type='sparse', model_xml_path=MODEL_XML_PATH):
        """
        :param n_objects: How many blocks should be stacked
        :param reward_type: whether the reward should be sparse or dense
        :param model_xml_path: The path to the XML that defines the MuJoCo environment
        """
        super().__init__(n_objects, gripper_goal='gripper_random', reward_type=reward_type, model_xml_path=model_xml_path)

    def _get_obs(self):
        # append the hook position, orientation and velocities to the normal BlocksEnv observation
        obs = super()._get_obs()
        observation = obs['observation']

        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        # position
        hook_pos = self.sim.data.get_geom_xpos('hook').copy()
        # rotation
        hook_rot = rotations.mat2euler(self.sim.data.get_geom_xmat('hook')).copy()
        # velocities
        hook_velp = self.sim.data.get_geom_xvelp('hook').copy() * dt
        hook_velr = self.sim.data.get_geom_xvelr('hook').copy() * dt

        obs['observation'] = np.concatenate([observation, hook_pos, hook_rot, hook_velp, hook_velr])
        return obs

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        # randomize the position of the hook
        hook_pos = self.np_random.uniform(POS_MIN, POS_MAX)
        hook_qpos = self.sim.data.get_joint_qpos('hook:joint')
        assert hook_qpos.shape == (7,)
        hook_qpos[:3] = hook_pos
        self.sim.data.set_joint_qpos('hook:joint', hook_qpos)

        # randomize start position of objects
        for o in range(self.n_objects):
            oname = 'object{}'.format(o)
            # find a position that is not too close to the hook or another object
            too_close = True
            while too_close:
                object_xpos = self.np_random.uniform(POS_MIN, POS_MAX)[:2]

                closest_dist = np.linalg.norm(object_xpos - hook_pos[:2])
                # Iterate through all previously placed boxes and select closest:
                for o_other in range(o):
                    other_xpos = self.sim.data.get_geom_xpos('object{}'.format(o_other))[:2]
                    dist = np.linalg.norm(object_xpos - other_xpos)
                    closest_dist = min(dist, closest_dist)
                if closest_dist > self.sample_dist_threshold:
                    too_close = False

            object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(oname))
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            object_qpos[2] = self.table_height + (self.object_height / 2)
            self.sim.data.set_joint_qpos('{}:joint'.format(oname), object_qpos)
        self.sim.forward()
        return True
