import os
import numpy as np
from gymnasium import spaces
from gymnasium.utils import EzPickle
from gymnasium_robotics.envs.fetch.fetch_env import MujocoFetchEnv
from gymnasium_robotics.utils.mujoco_utils import get_joint_qpos, set_joint_qpos
from custom_envs.mujoco_utils import get_geom_xpos

MODEL_XML_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'button_unlock.xml')

class ButtonUnlockEnv(MujocoFetchEnv, EzPickle):
    """
    Environment with causal dependencies:
    The robot has to reach n key locations (buttons) to unlock the goal location.
    The robot can only move in the x-y-plane.
    """

    def __init__(self, n_buttons, reward_type='sparse', model_xml_path=MODEL_XML_PATH):
        """
        :param n_buttons: How many buttons the robot has to press to unlock the goal
        :param reward_type: Whether the reward should be sparse or dense
        :param model_xml_path: The path to the XML that defines the MuJoCo environment
        """
        initial_qpos = {
            # robot xyz
            'robot0:slide0': -0.6,
            'robot0:slide1': 0,
            'robot0:slide2': 0,
        }

        self.n_buttons = n_buttons
        self.locked = np.ones(n_buttons-1)

        self.table_height = 0.4
        self.sample_dist_threshold = 0.11

        self.subgoals = {}
        self.goal = np.zeros(3)
        self.reset_since_last_goal_sampled = False

        super().__init__(model_path=model_xml_path, has_object=False, block_gripper=True, n_substeps=20,
                         gripper_extra_height=0, target_in_the_air=True, target_offset=0.0,
                         obj_range=0.15, target_range=0.15, distance_threshold=0.03,
                         initial_qpos=initial_qpos, reward_type=reward_type)
        self.action_space = spaces.Box(-1., 1., shape=(2,), dtype='float32')
        EzPickle.__init__(self)

    def _set_action(self, action):
        assert action.shape == (2,)
        action = action.copy()  # ensure that we don't change the action outside of this scope

        action *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion

        # little changes in the grippers height would accumulate over time.
        # The following forces the gripper to stay at its initial height.
        z = 0.435 - self._utils.get_site_xpos(self.model, self.data, 'robot0:grip')[2]

        action = np.concatenate([action, [z], rot_ctrl])

        # Apply action to simulation.
        self._utils.mocap_set_action(self.model, self.data, action)

    def _get_obs(self):
        grip_pos = self._utils.get_site_xpos(self.model, self.data, 'robot0:grip')
        dt = self.n_substeps * self.model.opt.timestep
        grip_velp = self._utils.get_site_xvelp(self.model, self.data, 'robot0:grip') * dt

        buttons_pos = np.empty(self.n_buttons * 2)
        for i in range(self.n_buttons):
            # we only care for the x-y-position of the buttons,
            # since the buttons do not rotate nor move and are always at the same height.
            pos = get_geom_xpos(self.model, self.data, 'object{}'.format(i))[:2]
            buttons_pos[2 * i:2 * i + 2] = pos
            # remove subgoal if robot is close to it
            if i > 0:
                if np.linalg.norm(grip_pos[:2] - pos) < self.distance_threshold:
                    self.locked[i - 1] = 0
                    geom_id = self._model_names.geom_name2id['object{}'.format(i)]
                    self.model.geom_rgba[geom_id] = [0.1, 0.9, 0.1, 1]

        # open cage if all subgoals were reached
        if np.all([l == 0 for l in self.locked]) or len(self.locked) == 0:
            set_joint_qpos(self.model, self.data, 'cage:glassjoint', -1.99)

        obs = np.concatenate([grip_pos, buttons_pos, grip_velp, self.locked])

        achieved_goal = obs[:3]
        desired_goal = self.goal

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': desired_goal.copy(),
        }

    def _render_callback(self):
        pass

    def _reset_sim(self):
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        self.locked = np.ones(self.n_buttons-1)
        # randomize start position of buttons
        for i in range(self.n_buttons):
            # find a position that is not too close to the gripper or another button
            too_close = True
            while too_close:
                button_xy = self.initial_gripper_xpos[:2] \
                              + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)

                closest_dist = np.linalg.norm(button_xy - self.initial_gripper_xpos[:2])
                # Iterate through all previously placed buttons and select closest:
                for o_other in range(i):
                    other_xpos = get_joint_qpos(self.model, self.data, 'object{}:joint'.format(o_other)).copy()[:2]
                    dist = np.linalg.norm(button_xy - other_xpos)
                    closest_dist = min(dist, closest_dist)
                if closest_dist > self.sample_dist_threshold:
                    too_close = False

            object_qpos = get_joint_qpos(self.model, self.data, 'object{}:joint'.format(i))
            assert object_qpos.shape == (7,)
            object_qpos[:2] = button_xy
            object_qpos[2] = self.table_height + 0.008
            set_joint_qpos(self.model, self.data, 'object{}:joint'.format(i), object_qpos)
            # also reset the buttons color
            if i > 0:
                geom_id = self._model_names.geom_name2id['object{}'.format(i)]
                self.model.geom_rgba[geom_id] = [0.1, 0.1, 0.9, 1]

        # set cage position
        goal_pos = get_joint_qpos(self.model, self.data, 'object0:joint').copy()
        goal_pos[2] = self.table_height + 0.05
        set_joint_qpos(self.model, self.data, 'cage:joint', goal_pos)

        self._mujoco.mj_forward(self.model, self.data)
        self.reset_since_last_goal_sampled = True
        return True

    def _sample_goal(self):
        if not self.reset_since_last_goal_sampled:
            self._reset_sim()
        self.reset_since_last_goal_sampled = False
        obs = self._get_obs()
        if obs is not None:
            goal = obs['observation'].copy()[3:6]
            goal[2] = 0.435  # z position of gripper
        return goal
