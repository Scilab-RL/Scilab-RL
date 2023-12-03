import os
import numpy as np
from gymnasium.utils import EzPickle
from gymnasium_robotics.envs.fetch.fetch_env import MujocoFetchEnv
from gymnasium import spaces

MODEL_XML_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'blocks.xml')


class Reach1DOFEnv(MujocoFetchEnv, EzPickle):
    """
    This is a very easy reacher-environment with only one degree of freedom.
    """
    def __init__(self, reward_type='sparse', model_xml_path=MODEL_XML_PATH):
        """
        : param reward_type: Whether the reward should be sparse or dense
        : param model_xml_path: The path to the XML that defines the MuJoCo environment
        """
        initial_qpos = {
            # robot xyz
            'robot0:slide0': -0.6,
            'robot0:slide1': 0,
            'robot0:slide2': 0,
        }
        self.goal_size = 1
        super().__init__(model_path=model_xml_path, has_object=False, block_gripper=False, n_substeps=20,
                         gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
                         obj_range=0.15, target_range=0.15, distance_threshold=0.05,
                         initial_qpos=initial_qpos, reward_type=reward_type)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype="float32")
        EzPickle.__init__(self)

    def _get_obs(self):
        grip_pos = self._utils.get_site_xpos(self.model, self.data, 'robot0:grip')
        dt = self.n_substeps * self.model.opt.timestep
        grip_velp = self._utils.get_site_xvelp(self.model, self.data, 'robot0:grip') * dt

        obs = np.concatenate([grip_pos, grip_velp])
        achieved_goal = obs[:self.goal_size]

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _render_callback(self):
        goal_pos = self.initial_gripper_xpos.copy()
        goal_pos[0] = self.goal[0]
        site_id = self._model_names.site_name2id['gripper_goal']
        self.model.site_pos[site_id] = goal_pos

    def _reset_sim(self):
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        self._mujoco.mj_forward(self.model, self.data)
        return True

    def _sample_goal(self):
        goal = np.array([self.initial_gripper_xpos[0]])
        goal[0] += self.np_random.uniform(-self.target_range, self.target_range)
        return goal.copy()

    def _set_action(self, action):
        action = np.concatenate([action, np.zeros(3)])
        super()._set_action(action)
