import os
import numpy as np
from ideas_envs.ant.ant_env import AntEnv

MODEL_XML_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'ant_button_unlock.xml')


class AntButtonUnlockEnv(AntEnv):
    def __init__(self, reward_type='sparse', distance_threshold=0.05, n_buttons=1):
        self.n_buttons = n_buttons
        self.locked = np.ones(n_buttons - 1)
        self.sample_dist_threshold = 0.3
        self.range = 0.55
        super().__init__(MODEL_XML_PATH, reward_type, distance_threshold)

    def _get_obs(self):
        # get ant joint positions and velocities like in the ant env
        ant_obs = np.concatenate((self.sim.data.qpos[:15], self.sim.data.qvel[:14]))

        buttons_pos = np.empty(self.n_buttons * 2)
        for i in range(self.n_buttons):
            # we only care for the x-y-position of the buttons,
            # since the buttons do not rotate nor move and are always at the same height.
            pos = self.sim.data.get_geom_xpos('object{}'.format(i))[:2]
            buttons_pos[2 * i:2 * i + 2] = pos
            # remove subgoal if robot is close to it
            if i > 0:
                if np.linalg.norm(ant_obs[:2] - pos) < self.distance_threshold:
                    self.locked[i - 1] = 0
                    geom_id = self.sim.model._geom_name2id['object{}'.format(i)]
                    self.sim.model.geom_rgba[geom_id] = [0.1, 0.9, 0.1, 1]

        # open cage if all subgoals were reached
        if np.all([l == 0 for l in self.locked]) or len(self.locked) == 0:
            self.sim.data.set_joint_qpos('cage:glassjoint', -1.99)

        obs = np.concatenate([ant_obs, buttons_pos, self.locked])

        achieved_goal = ant_obs[:2]
        desired_goal = buttons_pos[:2]

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': desired_goal.copy(),
        }

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.locked = np.ones(self.n_buttons - 1)

        # Reset controls
        self.sim.data.ctrl[:] = 0

        # randomize ant starting position
        ant_pos = self.sim.data.get_joint_qpos('root')
        ant_pos[:2] = np.random.uniform(-self.range, self.range, size=2)
        self.sim.data.set_joint_qpos('root', ant_pos)

        # randomize start position of buttons
        for i in range(self.n_buttons):
            # find a position that is not too close to the ant or another button
            too_close = True
            while too_close:
                button_xy = self.np_random.uniform(-self.range, self.range, size=2)

                closest_dist = np.linalg.norm(button_xy - ant_pos[:2])
                # Iterate through all previously placed buttons and select closest:
                for o_other in range(i):
                    other_xpos = self.sim.data.get_joint_qpos('object{}:joint'.format(o_other)).copy()[:2]
                    dist = np.linalg.norm(button_xy - other_xpos)
                    closest_dist = min(dist, closest_dist)
                if closest_dist > self.sample_dist_threshold:
                    too_close = False

            object_qpos = self.sim.data.get_joint_qpos('object{}:joint'.format(i))
            assert object_qpos.shape == (7,)
            object_qpos[:2] = button_xy
            object_qpos[2] = 0.008
            self.sim.data.set_joint_qpos('object{}:joint'.format(i), object_qpos)
            # also reset the buttons color
            if i > 0:
                geom_id = self.sim.model._geom_name2id['object{}'.format(i)]
                self.sim.model.geom_rgba[geom_id] = [0.1, 0.1, 0.9, 1]

        # set cage position
        goal_pos = self.sim.data.get_joint_qpos('object0:joint').copy()
        goal_pos[2] = 0.05
        self.sim.data.set_joint_qpos('cage:joint', goal_pos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        obs = self._get_obs()
        if obs is not None:
            goal = obs['observation'].copy()[3:5]
        return goal

    def _render_callback(self):
        pass
