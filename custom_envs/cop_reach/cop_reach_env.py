from pyrep import PyRep
from pyrep.const import PrimitiveShape, ObjectType, RenderMode
from pyrep.robots.arms.arm import Arm
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.errors import IKError, PyRepError
import numpy as np
from os.path import dirname, join, abspath
import gym
from gym import spaces
from gym.utils import seeding

CACHED_ENV = None

SCENE_FILE = join(dirname(abspath(__file__)),
                  'CopReacherEnv.ttt')


class ManipulatorPro (Arm):
    def __init__(self, count: int = 0):
        super().__init__(count, 'mp', num_joints=6)


POS_MIN, POS_MAX = [0.8, -0.2, 1.0], [1.0, 0.2, 1.4]


class ReacherEnvMaker:
    """
    Creates a Reacher Environment or returns an existing instance if one has already been created.
    """
    def __new__(cls, *args, **kwargs):
        global CACHED_ENV
        if CACHED_ENV is None:
            return ReacherEnv(*args, **kwargs)
        else:
            print('\033[92m' + 'Using cached Env' + '\033[0m')
            return CACHED_ENV


class ReacherEnv(gym.GoalEnv):
    """
    Environment with Reacher task that uses CoppeliaSim and the ManipulatorPro robotic arm.
    Args:
        render_mode: If render_mode != 'human', CoppeliaSim will run in headless mode.
        ik: whether to use inverse kinematics. If not, the actuators will be controlled directly.
            Note, that also the observation changes, when ik is set.
            !!!The IK can not always be computed. In that case, the action is not carried out!!!
    """
    def __init__(self, render_mode='none', ik=1, reward_type='sparse'):
        print('\033[92m' + 'Creating new Env' + '\033[0m')
        render = render_mode == 'human'
        self.ik = bool(ik)
        self.reward_type = reward_type
        self.seed()

        # PyRep initialization
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=not render)
        self.pr.start()

        self.metadata['render.modes'] = ['human', 'rgb_array']
        if render_mode == 'rgb_array':
            cam_placeholder = Dummy('cam_placeholder')
            self.cam = VisionSensor.create([640, 360])
            self.cam.set_pose(cam_placeholder.get_pose())
            self.cam.set_render_mode(RenderMode.OPENGL3)

        # load robot and set position
        self.agent = ManipulatorPro()
        if not self.ik:
            self.agent.set_control_loop_enabled(False)
            self.agent.set_motor_locked_at_zero_velocity(True)
        else:
            print('\033[91m' + 'You are running with inverse kinematics. Sometimes the IK are not working and the '
                               'action can not be carried out. In that case, \'Attempting to reach out of reach\' is '
                               'printed.' + '\033[0m')
        self.target = Shape('target')
        self.vis = {}
        self.agent_ee_tip = self.agent.get_tip()
        self.agent_parts = self.pr.get_objects_in_tree(self.agent)
        self.poses = [ap.get_pose() for ap in self.agent_parts]
        self.initial_joint_positions = self.agent.get_joint_positions()

        # define goal
        self.goal = self._sample_goal()

        self.distance_threshold = 0.05
        self.step_ctr = 0

        # set action space
        if self.ik:
            self.action_space = spaces.Box(-1., 1., shape=(3,), dtype='float32')
        else:
            self.action_space = spaces.Box(-1., 1., shape=(6,), dtype='float32')
        obs = self._get_obs()
        obs_space_high = np.ones_like(obs['observation'])
        obs_space_low = obs_space_high.copy()*(-1)
        obs_space_low[:3] = np.array(POS_MIN)
        obs_space_high[:3] = np.array(POS_MAX)
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(np.array(POS_MIN), np.array(POS_MAX), dtype='float32'),
            achieved_goal=spaces.Box(np.array(POS_MIN), np.array(POS_MAX), dtype='float32'),
            observation=spaces.Box(obs_space_low, obs_space_high, dtype='float32'),))

        global CACHED_ENV
        CACHED_ENV = self

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='none', **kwargs):
        if mode in ['none', 'human']:
            return
        frame = self.cam.capture_rgb()
        frame = np.clip((frame * 255.).astype(np.uint8), 0, 255)
        return frame

    def _get_obs(self):
        achieved_goal = self.agent_ee_tip.get_position()
        if self.ik:
            obs = achieved_goal
        else:
            obs = np.concatenate([achieved_goal,
                                  self.agent.get_joint_positions(),
                                  self.agent.get_joint_velocities()])

        obs = {'observation': obs.copy(), 'achieved_goal': achieved_goal.copy(),
               'desired_goal': np.array(self.goal.copy()),
               'non_noisy_obs': obs.copy()}
        return obs

    def _sample_goal(self):
        # Get a random position within a cuboid and set the target position
        pos = list(self.np_random.uniform(POS_MIN, POS_MAX))
        self.target.set_position(pos)
        return pos

    def reset(self):
        self.step_ctr = 0
        # reset the joint positions
        self.agent.set_joint_positions(self.initial_joint_positions)

        # reset the robot-parts positions
        for ap, pos in zip(self.agent_parts, self.poses):
            t = ap.get_type()
            if t == ObjectType.JOINT:
                pass
            else:
                ap.set_pose(pos, reset_dynamics=True)

        self.goal = self._sample_goal()
        obs = self._get_obs()
        return obs

    def compute_reward(self, achieved_goal, goal, info):
        if len(achieved_goal.shape) == 2:
            reward = [self.compute_reward(g1, g2, info) for g1, g2 in zip(achieved_goal, goal)]
        else:
            ax, ay, az = achieved_goal  # self.agent_ee_tip.get_position()
            tx, ty, tz = goal  # self.target.get_position()
            dist = np.sqrt((ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2)
            if self.reward_type != 'sparse':
                reward = -dist
            elif dist < self.distance_threshold:
                reward = 0
            else:
                reward = -1
        return np.array(reward)

    def _is_success(self, achieved_goal, desired_goal):
        ax, ay, az = achieved_goal
        tx, ty, tz = desired_goal
        dist = np.sqrt((ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2)
        return int(dist < self.distance_threshold)

    def _set_action(self, action):
        self.step_ctr += 1
        if self.ik:
            pos = self.agent_ee_tip.get_position().copy()
            quat = self.agent_ee_tip.get_quaternion().copy()
            pos += (action * 0.05)
            try:
                new_joint_angles = self.agent.solve_ik(pos, quaternion=quat)
                self.agent.set_joint_target_positions(new_joint_angles)
            except IKError:
                pass  # sometimes the inverse kinematics cannot be computed.
                # In that case, the action is not carried out.
        else:
            self.agent.set_joint_target_velocities(action)

    def step(self, action):
        self._set_action(action)
        self.pr.step()  # Step the physics simulation
        obs = self._get_obs()
        is_success = self._is_success(obs['achieved_goal'], obs['desired_goal'])
        done = bool(is_success)
        info = {'is_success': is_success}

        r = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], {})
        return obs, r, done, info

    def _obs2goal(self, obs):
        if len(obs.shape) == 1:
            obs_arr = np.array([obs])
        else:
            obs_arr = obs
        assert len(obs_arr.shape) == 2
        goals = []
        for o in obs_arr:
            goals.append(o[:3])
        goals = np.array(goals)
        if len(obs.shape) == 1:
            return goals[0]
        else:
            return goals

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

    def close(self):
        print('\033[91m' + 'Closing Env' + '\033[0m')
        try:
            self.pr.stop()
            self.pr.shutdown()
        except PyRepError:
            print('Cannot shut down CoppeliaSim because it hasn\'t been launched')
