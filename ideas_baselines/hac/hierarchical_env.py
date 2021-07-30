from copy import deepcopy
from typing import Callable, List
import numpy as np
import gym
from stable_baselines3.common import logger
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn

GOAL_MARKER_COLORS = [[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [1, 1, 0],
                      [1, 0, 1],
                      [0, 1, 1],
                      [1, 0.5, 0],
                      [1, 0, 0.5],
                      [0.5, 1, 0.5],
                      [0, 0.5, 1]]

GOAL_MARKER_SHAPES = ['sphere', 'box', 'cylinder']
GOAL_MARKER_MIN_ALPHA = 0.1
GOAL_MARKER_MAX_ALPHA = 0.9


def get_h_envs_from_env(bottom_env: gym.wrappers.TimeLimit,
                        level_steps: List,
                        is_testing_env: bool = False, layer_alg: OffPolicyAlgorithm = None) -> List[gym.wrappers.TimeLimit]:
    """
    Creates HierarchicalHLEnvs on top of the bottom_env.
    :param bottom_env: The original simulation environment wrapped in a TimeLimit wrapper
    :param level_steps: A list that contains the steps that each layer should perform before a
                        next higher-level step is performed. From highest to lowest layer.
                        e.g. [5, 11]: The lowest layer performs 11 steps before the highest layer performs one.
                        after 5 steps of the highest layer, the episode is done.
    :param is_testing_env: True if the bottom_env is the env intended for testing
    :param layer_alg: The highest layer of the algorithm. Every environment-layer receives a reference to the
                      corresponding algorithm-layer.
    :return: A list that contains the hierarchy of environments, from highest to lowest.
    """
    # reverse the level_steps list so that it is sorted from lowest to highest layer
    level_steps = level_steps.copy()  # don't modify original
    level_steps.reverse()

    # The bottom_env has a list that stores the goals of all layers. This is necessary for the subgoal visualization.
    bottom_env.goal_list = [[0, 0, 0]] * len(level_steps)
    env_list = [bottom_env]
    # set the maximum steps for the bottom_env
    bottom_env.unwrapped.spec.max_episode_steps = level_steps[0]
    bottom_env._max_episode_steps = level_steps[0]
    level_steps = level_steps[1:]
    # find the lowest layer of the algorithm and assign it to the bottom_env
    if layer_alg is not None:
        while layer_alg.sub_layer is not None:
            layer_alg = layer_alg.sub_layer
        bottom_env.env.layer_alg = layer_alg


    for lvl_steps in level_steps:
        if layer_alg is not None:
            layer_alg = layer_alg.parent_layer
        env = HierarchicalHLEnv(bottom_env, is_testing_env=is_testing_env, layer_alg=layer_alg)
        env.set_sub_env(env_list[-1])
        env = gym.wrappers.TimeLimit(env, max_episode_steps=lvl_steps)
        env_list.append(env)

    # the env_list should be sorted from highest to lowest env, so we need to reverse it.
    env_list.reverse()

    return env_list


class HierarchicalVecEnv(DummyVecEnv):
    """
    This class has the same functionality as DummyVecEnv, but it does not reset the simulator when a low-level
    episode ends. Every HierarchicalHLEnv gets wrapped into a HierarchicalVecEnv.
    It also handles the (sub)goal visualization.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]], early_done_on_success: bool):
        super().__init__(env_fns)
        self.goal_viz_shape = 'sphere'
        self.goal_viz_size = 0.035
        self.early_done_on_success = early_done_on_success

    def prepare_goal_viz(self, env_idx):
        # get the list of (sub)goals to display
        goal_list = self.envs[env_idx].goal_list
        # create a list of sites that represent the (sub)goals
        goals_to_viz = {}
        alphas = np.arange(GOAL_MARKER_MIN_ALPHA, GOAL_MARKER_MAX_ALPHA, GOAL_MARKER_MAX_ALPHA / len(goal_list))
        for layer_idx, layer_goals_to_render in enumerate(goal_list):
            if len(layer_goals_to_render) % 3 != 0:
                return
            n_goals = len(layer_goals_to_render) // 3
            colors = GOAL_MARKER_COLORS[:n_goals]
            for i in range(n_goals):
                shape_idx = layer_idx % len(GOAL_MARKER_SHAPES)
                goal_marker_shape = GOAL_MARKER_SHAPES[shape_idx]
                color_and_alpha = colors[i] + [alphas[layer_idx]]
                goals_to_viz['subgoal_{}{}'.format(goal_marker_shape, i)] = (
                    layer_goals_to_render[i * 3: (i + 1) * 3], self.goal_viz_size, color_and_alpha)
        # set the sites' attributes in the simulation
        viz_env = self.envs[env_idx].unwrapped
        if hasattr(viz_env, 'display_subgoals'):
            viz_env.display_subgoals(goals_to_viz)
            return
        for name in goals_to_viz:
            try:
                site_id = viz_env.sim.model.site_name2id(name)
                viz_env.sim.model.site_pos[site_id] = goals_to_viz[name][0].copy()
                size = [goals_to_viz[name][1]] * 3
                viz_env.sim.model.site_size[site_id] = size
                viz_env.sim.model.site_rgba[site_id] = goals_to_viz[name][2]
            except ValueError as e:
                raise ValueError("Site {} does not exist. Please include the ideas_envs.assets.subgoal_viz.xml "
                                 "in your environment xml.".format(name)) from e

    def render(self, mode='rgb_array', width=1024, height=768):
        env_idx = 0
        self.prepare_goal_viz(env_idx)
        frame = self.envs[env_idx].render(mode=mode, width=width, height=height)
        return frame

    # The same function as in DummyVecEnv, but without resetting the simulation when the episode ends.
    # This function also sets done to true on success
    def step_wait(self) -> VecEnvStepReturn:
        for env_idx in range(self.num_envs):
            obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx] = self.envs[env_idx].step(
                self.actions[env_idx]
            )
            # Set done to true if success is achieved or if it is done any ways (by TimeLimit)
            if self.buf_dones[env_idx]:
                # save final observation where user can get it, but don't reset
                self.buf_infos[env_idx]["terminal_observation"] = obs
                self.envs[env_idx]._elapsed_steps = 0
            self._save_obs(env_idx, obs)
        return self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos)

    def reset_all(self):
        for env_idx in range(self.num_envs):
            obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return self.buf_obs


class HierarchicalHLEnv(gym.GoalEnv):
    """
    A HierarchicalHLEnv functions as a hierarchical abstraction of lower-level goal-conditioned environments.
    An action taken in a HierarchicalHLEnv is the goal of its _sub_env.
    Example structure:
    <HierarchicalHLEnv>
        self._sub_env = <TimeLimit<BlocksEnv<Blocks-o3-gripper_random-v1>>>
    """
    def __init__(self, bottom_env, is_testing_env=None, layer_alg=None):
        self.bottom_env = bottom_env
        self.action_space = bottom_env.observation_space.spaces['desired_goal']
        self.observation_space = bottom_env.unwrapped.observation_space
        self._sub_env = None
        self.layer_alg = layer_alg
        self.is_testing_env = is_testing_env

    def update_action_bound_guess(self):
        print("guessing the subgoal bounds based on goal samples")  # cannot use the logger because this happens during
                                                                    # the initialization of the algorithm.
        n_samples = 5000
        self.action_space.high = [-np.inf] * len(self.action_space.high)
        self.action_space.low = [np.inf] * len(self.action_space.low)
        for _ in range(n_samples):
            goal = self._sub_env.unwrapped._sample_goal()
            self.action_space.high = np.maximum(goal, self.action_space.high)
            self.action_space.low = np.minimum(goal, self.action_space.low)
        # Add some small extra margin.
        epsilon = np.zeros_like(self.action_space.high) + 0.01
        self.action_space.high += np.abs(self.action_space.high - self.action_space.low) * 0.01
        self.action_space.low -= np.abs(self.action_space.high - self.action_space.low) * 0.01
        action_space_range = self.action_space.high - self.action_space.low
        action_space_range = np.maximum(action_space_range, epsilon)
        self.action_space.high = self.action_space.low + action_space_range
        # Reset action space to determine whether the space is bounded.
        self.action_space = gym.spaces.Box(self.action_space.low, self.action_space.high)

    def set_sub_env(self, env):
        self._sub_env = env
        if np.inf in self.action_space.high or -np.inf in self.action_space.low:
            self.update_action_bound_guess()

    def step(self, action):
        subgoal = np.clip(action, self.action_space.low, self.action_space.high)
        self._sub_env.env._elapsed_steps = 0 # Set elapsed steps to 0 but don't reset the whole simulated environment
        self._sub_env.unwrapped.goal = subgoal
        # store the goals in the bottom_env, so that they can be displayed
        current_layer = self.layer_alg.layer
        self.bottom_env.goal_list[current_layer] = self.goal
        self.bottom_env.goal_list[current_layer-1] = subgoal

        assert self.layer_alg is not None, "Step not possible because no layer_alg defined yet."
        if self.is_testing_env:
            info = self.test_step()
        else:
            info = {}
            self.train_step()
        if not self.is_testing_env:
            if self.layer_alg.sub_layer is not None:
                if self.layer_alg.sub_layer.in_subgoal_test_mode:
                    info['is_subgoal_testing_trans'] = 1
                else:
                    info['is_subgoal_testing_trans'] = 0
        obs = self._get_obs()
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)

        self._step_callback()
        succ = self._is_success(obs['achieved_goal'], self.goal)
        info['is_success'] = succ
        done = False  # Returning done = true after time steps are done is not necessary here because it is done in TimeLimit wrapper. #TODO: Check if done=True should be returned after goal is achieved.
        # done = np.isclose(succ, 1)
        return obs, reward, done, info

    def train_step(self):
        self.layer_alg.train_step()

    def test_step(self):
        info_list = self.layer_alg.test_step(self)
        return info_list

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with self._sub_env._step_callback()numerical issues (e.g. due to penetration
        # or Gimbal lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        super().reset()
        obs = self._sub_env.reset()
        self.goal = self._sub_env.goal
        return obs

    def close(self):
        self._sub_env.close()

    def render(self, **kwargs):
        self._sub_env.render(**kwargs)

    def _get_viewer(self, **kwargs):
        return self._sub_env._get_viewer(**kwargs)

    def _get_obs(self):
        """Returns the observation.
        """
        obs = self._sub_env.unwrapped._get_obs()
        obs['desired_goal'] = self.goal
        return obs

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        return self._sub_env.unwrapped._is_success(achieved_goal, desired_goal)

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """
        return self._sub_env.env._sample_goal()

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        self._sub_env._env_setup(initial_qpos)

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        self._sub_env._viewer_setup()

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self._sub_env.unwrapped.compute_reward(achieved_goal, desired_goal, info)
