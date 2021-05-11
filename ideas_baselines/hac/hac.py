import io
import pathlib
import warnings
from typing import Any, Iterable, List, Optional, Tuple, Type, Union
import numpy as np
import torch as th
import gym
import matplotlib.pyplot as plt
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import load_from_zip_file, recursive_setattr
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn
from ideas_baselines.hac.util import check_for_correct_spaces
# from stable_baselines3.common.utils import check_for_correct_spaces
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper
from ideas_baselines.hac.goal_selection_strategy import KEY_TO_GOAL_STRATEGY, GoalSelectionStrategy
from ideas_baselines.hac.hher_replay_buffer import HHerReplayBuffer
from ideas_baselines.hac.hierarchical_env import get_h_envs_from_env
from stable_baselines3.common import logger
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.preprocessing import is_image_space
from ideas_baselines.hac.hierarchical_env import HierarchicalVecEnv
from gym.wrappers import TimeLimit
import time
from ideas_baselines.hac.util import get_concat_dict_from_dict_list, merge_list_dicts
from ideas_envs.wrappers.subgoal_viz_wrapper import SubgoalVisualizationWrapper
import numbers
from copy import deepcopy
from stable_baselines3.common.logger import HumanOutputFormat
import sys
from stable_baselines3.common.vec_env import VecVideoRecorder
import cv2
try:
    from stable_baselines3.common.env_util import is_wrapped # stable-baselines v.3
except Exception as e:
    from stable_baselines3.common.vec_env import is_wrapped # stable-baselines v.0.1

from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecNormalize,
    VecTransposeImage,
    unwrap_vec_normalize,
)

def get_time_limit(env: VecEnv, current_max_episode_length: Optional[int]) -> int:
    """
    Get time limit from environment.

    :param env: Environment from which we want to get the time limit.
    :param current_max_episode_length: Current value for max_episode_length.
    :return: max episode length
    """
    # try to get the attribute from environment
    if current_max_episode_length is None:
        try:
            current_max_episode_length = env.get_attr("spec")[0].max_episode_steps
            # Raise the error because the attribute is present but is None
            if current_max_episode_length is None:
                raise AttributeError
        # if not available check if a valid value was passed as an argument
        except AttributeError:
            raise ValueError(
                "The max episode length could not be inferred.\n"
                "You must specify a `max_episode_steps` when registering the environment,\n"
                "use a `gym.wrappers.TimeLimit` wrapper "
                "or pass `max_episode_length` to the algorithm constructor"
            )
    return current_max_episode_length


def compute_time_scales(scales, env):
    max_steps = env.spec.max_episode_steps
    for i,s in enumerate(scales):
        if s == 0:
            raise ValueError("Use -1 in time scale instead of 0")
        elif s == -1:
            defined_steps = np.product([int(step) for step in scales[:i]])
            defined_after_steps = np.product([int(step) for step in scales[i+1:]])
            defined_steps *= defined_after_steps
            scales[i] = int(max_steps / defined_steps)
    return scales



class HAC(BaseAlgorithm):
    """
    Model-based curious hierarchical actor-critic HAC
    Cobed based on Hindsight Experience Replay (HER) code of stable baselines 3 repository

    .. warning::

      For performance reasons, the maximum number of steps per episodes must be specified.
      In most cases, it will be inferred if you specify ``max_episode_steps`` when registering the environment
      or if you use a ``gym.wrappers.TimeLimit`` (and ``env.spec`` is not None).
      Otherwise, you can directly pass ``max_episode_length`` to the algorithm constructor


    For additional offline algorithm specific arguments please have a look at the corresponding documentation.

    :param policy: The policy to use.
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param layer_classes: Array of Off policy algorithms
        which will be used with hindsight experience replay. (SAC, TD3, DDPG, DQN)
    :param n_sampled_goal: Number of sampled goals for replay. (offline sampling)
    :param goal_selection_strategy: Strategy for sampling goals for replay.
        One of ['episode', 'final', 'future', 'random']
    :param learning_rate: learning rate for the optimizer,
        it can be a function of the current progress remaining (from 1 to 0)
    :param max_episode_length: The maximum length of an episode. If not specified,
        it will be automatically inferred if the environment uses a ``gym.wrappers.TimeLimit`` wrapper.
        it will be automatically inferred if the environment uses a ``gym.wrappers.TimeLimit`` wrapper.
    """


    save_attrs_to_exclude = ['layer_alg', 'train_video_writer', 'test_video_writer', 'sub_layer', 'parent_layer', 'env',
                             'episode_storage', 'device', 'train_callback', 'tmp_train_logger', 'policy_class',
                             'policy_kwargs', 'lr_schedule',
                             'gradient_steps', 'train_freq', # These two are not required because they are overwritten in the train() function of the model any ways.
                             '_episode_storage', 'eval_info_list', 'sub_layer_classes', 'goal_selection_strategy', 'layer_class', 'action_space', 'observation_space', # These require more than 1MB to save
                             'episode_steps',
                             'render_train', 'render_test', 'render_every_n_eval', 'train_render_info', 'test_render_info' # Overwrite rendering variables.
                             ]
    def __init__(
        self,
        policy: Union[str, Type[BasePolicy]],
        env: Union[GymEnv, str],
        layer_class: Type[OffPolicyAlgorithm],
        sub_layer_classes: List[Type[OffPolicyAlgorithm]] = [],
        time_scales: int = 1,
        learning_rates: str = '3e-4',
        n_sampled_goal: int = 4,
        goal_selection_strategy: Union[GoalSelectionStrategy, str] = "future",
        max_episode_length: Optional[int] = None,
        is_top_layer: bool = True,
        layer_envs: List[GymEnv] = [],
        n_train_batches: int = 0,
        train_freq: int = 0,
        subgoal_test_perc: float = 0.3,
        render_train: str = 'none',
        render_test: str = 'none',
        render_every_n_eval: int = 1,
        use_action_replay: bool = True,
        ep_early_done_on_succ: bool = True,
        hindsight_sampling_done_if_success: int = 1,
        set_fut_ret_zero_if_done: int = 1, # FIXME: should be in kwargs
        *args,
        **kwargs,
    ):
        self.epoch_count = 0
        self.ep_early_done_on_succ = ep_early_done_on_succ
        self.gradient_steps = n_train_batches
        self.learn_callback = None
        self.render_every_n_eval = render_every_n_eval
        self.is_top_layer = is_top_layer
        self.time_scales = time_scales
        self.train_freq = train_freq
        self.is_bottom_layer = len(time_scales) == 1
        self.parent_layer = None
        self.hindsight_sampling_done_if_success = hindsight_sampling_done_if_success
        self.reset_train_info_list()
        assert len(time_scales) == (
                    len(sub_layer_classes) + 1), "Error, number of time scales is not equal to number of layers."
        assert time_scales.count(
            "_") <= 1, "Error, only one wildcard character \'_\' allowed in time_scales argument {}".format(time_scales)
        if self.is_top_layer == 1:  # Determine time_scales. Only do this once at top layer.
            self.time_scales = compute_time_scales(time_scales, env)
            # Build hierarchical layer_envs from env, depending on steps and action space.
            layer_envs = get_h_envs_from_env(env, time_scales)
        self.learning_rates = learning_rates
        learning_rates_float = [float(lr) for lr in self.learning_rates]
        self.learning_rate = learning_rates_float[0]
        if max_episode_length is None:
            max_episode_length = int(self.time_scales[0])
        self.max_steps_per_layer_action = np.product(self.time_scales[1:]) # the max. number of low-level steps per action on this layer.
        bottom_env = layer_envs[-1]
        this_env = layer_envs[0]
        this_env.env.layer_alg = self
        # Build policy, convert env into <ObstDictWrapper> class.
        super(HAC, self).__init__(policy=BasePolicy, env=this_env, policy_base=BasePolicy, learning_rate=self.learning_rate)
        # we will use the policy from the layer_alg.
        del self.policy

        self.learning_starts = kwargs['learning_starts']
        _init_setup_model = kwargs.get("_init_setup_model", True)

        if "_init_setup_model" in kwargs:
            del kwargs["_init_setup_model"]

        # layer algorithm initialization
        self.layer_class = layer_class
        layer_args = kwargs.copy()
        if 'sub_layer_classes' in layer_args:
            del layer_args['sub_layer_classes']

        self.sub_layer_classes = sub_layer_classes
        self.layer = len(self.sub_layer_classes)

        assert (len(sub_layer_classes) + 1) == len(layer_envs), "Error, number of sub_layer classes should be one less than number of envs"
        next_level_steps = 0
        if len(sub_layer_classes) > 0:
            next_level_steps = int(self.time_scales[1])
            self.sub_layer = HAC('MlpPolicy', bottom_env, sub_layer_classes[0], sub_layer_classes[1:],
                                    time_scales=self.time_scales[1:],
                                    n_sampled_goal=n_sampled_goal,
                                    goal_selection_strategy=goal_selection_strategy,
                                    train_freq=self.train_freq,
                                    is_top_layer=False,
                                    layer_envs=layer_envs[1:],
                                    render_train=render_train,
                                    render_test=render_test,
                                    learning_rates=learning_rates[1:],
                                    render_every_n_eval=render_every_n_eval,
                                    use_action_replay=use_action_replay,
                                    ep_early_done_on_succ=ep_early_done_on_succ,
                                    **kwargs)
            self.sub_layer.parent_layer = self
        else:
            self.sub_layer = None

        self.layer_alg = layer_class(
            policy=policy,
            env=self.env,
            _init_setup_model=False,  # pytype: disable=wrong-keyword-args
            learning_rate=self.learning_rate,
            *args,
            **layer_args,  # pytype: disable=wrong-keyword-args
        )

        self.verbose = self.layer_alg.verbose
        self.tensorboard_log = self.layer_alg.tensorboard_log

        # convert goal_selection_strategy into GoalSelectionStrategy if string
        if isinstance(goal_selection_strategy, str):
            self.goal_selection_strategy = KEY_TO_GOAL_STRATEGY[goal_selection_strategy.lower()]
        else:
            self.goal_selection_strategy = goal_selection_strategy

        # check if goal_selection_strategy is valid
        assert isinstance(
            self.goal_selection_strategy, GoalSelectionStrategy
        ), f"Invalid goal selection strategy, please use one of {list(GoalSelectionStrategy)}"

        self.n_sampled_goal = n_sampled_goal
        # compute ratio between HER replays and regular replays in percent for online HER sampling
        self.her_ratio = 1 - (1.0 / (self.n_sampled_goal + 1))
        # maximum steps in episode
        self.max_episode_length = get_time_limit(self.env, max_episode_length)
        # storage for transitions of current episode for offline sampling
        # for online sampling, it replaces the "classic" replay buffer completely
        her_buffer_size = self.buffer_size
        self.subgoal_test_perc = subgoal_test_perc

        sample_test_trans_fraction = self.subgoal_test_perc / (self.n_sampled_goal + 1)
        sample_test_trans_fraction = (not self.is_bottom_layer) * sample_test_trans_fraction
        subgoal_test_fail_penalty = next_level_steps

        perform_action_replay = not self.is_bottom_layer
        self.perform_action_replay = perform_action_replay and use_action_replay

        self._episode_storage = HHerReplayBuffer(
            self.env,
            her_buffer_size,
            self.max_episode_length,
            self.goal_selection_strategy,
            self.env.observation_space,
            self.env.action_space,
            self.device,
            self.n_envs,
            self.her_ratio,  # pytype: disable=wrong-arg-types
            self.perform_action_replay,
            sample_test_trans_fraction,
            subgoal_test_fail_penalty,
            self.hindsight_sampling_done_if_success,
        )

        # counter for steps in episode
        self.episode_steps = 0
        if _init_setup_model:
            self._setup_model()

        self.train_callback = None
        self.tmp_train_logger = logger.Logger(folder=None, output_formats=[]) # HumanOutputFormat(sys.stdout)
        self.test_render_info = None
        self.train_render_info = None

        self.vid_size = 1024, 768
        self.vid_fps = 25

        self.render_train = render_train
        self.render_test = render_test

        if self.render_train == 'record':
            train_render_info = {'size': self.vid_size, 'fps': self.vid_fps,
                                'path': logger.get_dir()}
            self.set_train_render_info(train_render_info)
            self.train_video_writer = None
        else:
            self.train_render_info = None

        if self.render_test == 'record':
            test_render_info = {'size': self.vid_size, 'fps': self.vid_fps,
                                'path': logger.get_dir()}
            self.set_test_render_info(test_render_info)
            self.test_video_writer = None
        else:
            self.test_render_info = None

        self.in_subgoal_test_mode = False
        self.continue_training = True
        self.actions_since_last_train = 0
        self.learning_enabled = False
        if self.layer == 0:  # Always start training on lowest layer
            self.set_learning_enabled()
        else:
            # Start learning for higher layers only if using action replay.
            # Otherwise wait until the lower level layer is stable (see  _record_logs() below)
            if use_action_replay:
                self.set_learning_enabled()

    def get_continue_training(self):
        if self.sub_layer is None:
            return self.continue_training
        else:
            return self.sub_layer.get_continue_training()

    def start_train_video_writer(self, filename_postfix):
        self.train_video_writer = cv2.VideoWriter(self.train_render_info['path'] + '/train_{}.avi'.format(filename_postfix),
                        cv2.VideoWriter_fourcc('F', 'M', 'P', '4'), self.train_render_info['fps'],
                        self.train_render_info['size'])

    def stop_train_video_writer(self):
        self.train_video_writer.release()

    def start_test_video_writer(self, filename_postfix):
        self.test_video_writer = cv2.VideoWriter(self.test_render_info['path'] + '/test_{}.avi'.format(filename_postfix),
                        cv2.VideoWriter_fourcc('F', 'M', 'P', '4'), self.test_render_info['fps'],
                        self.test_render_info['size'])

    def stop_test_video_writer(self):
        self.test_video_writer.release()

    def _setup_model(self) -> None:
        self.layer_alg._setup_model()
        # assign episode storage to replay buffer when using online HER sampling
        self.layer_alg.replay_buffer = self._episode_storage

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        # if self.continuous_subgoals and
        if self.sub_layer is not None:
            subgoal = self.layer_alg.predict(observation, state, mask, deterministic)
            observation['desired_goal'] = subgoal[0]
            action = self.sub_layer.predict(observation, state, mask, deterministic)
        else:
            action = self.layer_alg.predict(observation, state, mask, deterministic)
        return action

    def train_layer(self, n_gradient_steps: int):
        # if self.num_timesteps > self.learning_starts and self.replay_buffer.size() > 0 and self.learning_enabled is True:
        rb_size = self.replay_buffer.size()
        if self.num_timesteps > self.learning_starts and self.replay_buffer.size() > self.learning_starts and self.learning_enabled is True:
            # logger.info("Training layer {} for {} steps.".format(self.layer, n_gradient_steps))
            # assign temporary logger to avoid generating duplicate keys for the different layers.
            real_logger = logger.Logger.CURRENT
            logger.Logger.CURRENT = self.tmp_train_logger
            self.train(batch_size=self.batch_size, gradient_steps=n_gradient_steps)
            logger.Logger.CURRENT = real_logger
            logged_kvs = self.tmp_train_logger.name_to_value
            for k, v in logged_kvs.items():
                try:
                    postfix = k.split("/")[1]
                    prefix = k.split("/")[0]
                    new_k = prefix + "_{}".format(self.layer) + "/" + postfix
                except:
                    new_k = k
                logger.record_mean(new_k, v)
            self.tmp_train_logger.dump()
            self.actions_since_last_train = 0
        else:
            pass  # logger.info("Did not yet train layer {} because I have not yet enough experience collected.".format(self.layer))

    def run_and_maybe_train(self, n_episodes: int):
        rollout = self.collect_rollouts(
            self.env,
            n_episodes=n_episodes,
            n_steps=-1,
            action_noise=self.action_noise,
            learning_starts=self.learning_starts
        )
        if rollout.continue_training is False:
            return False
        if self.train_freq == 0: # If training frequency is 0, train every episode for the number of gradient steps equal to the number of actions performed.
            self.train_layer(rollout.episode_timesteps)
        return True

    def set_learning_enabled(self):
        if self.learning_enabled is False:
            self.actions_since_last_train = 0
            self.learning_enabled = True
            self.replay_buffer.reset()


    def init_learn(self, total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps):
        tb_log_name = "HAC_{}".format(self.layer)
        layer_total_timesteps = total_timesteps / self.max_steps_per_layer_action
        layer_total_timesteps, callback = self._setup_learn(
            layer_total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )
        # self.epoch_count = 0
        self.layer_alg.start_time = self.start_time
        self.layer_alg.ep_info_buffer = self.ep_info_buffer
        self.layer_alg.ep_success_buffer = self.ep_success_buffer
        self.layer_alg.num_timesteps = self.num_timesteps
        self.layer_alg._episode_num = self._episode_num
        self.layer_alg._last_obs = self._last_obs
        self.layer_alg._total_timesteps = self._total_timesteps
        if self.sub_layer is not None:
            self.sub_layer.init_learn(total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps)
        self.train_callback = callback
        return layer_total_timesteps, callback

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = None,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "HAC",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = False,
    ) -> BaseAlgorithm:

        total_env_timesteps = total_timesteps
        total_layer_actions, callback = self.init_learn(total_env_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps)
        if self.is_top_layer:
            self.env.reset()
        callback.on_training_start(locals(), globals())
        self.actions_since_last_train = 0
        continue_training = True
        if self.is_top_layer and self.train_render_info is not None:
            self.start_train_video_writer(self.get_env_steps())
        if self.is_top_layer and self.test_render_info is not None:
            self.start_test_video_writer(self.get_env_steps())

        ### THIS IS THE MAIN TRAINING LOOP
        while self.get_env_steps() < total_env_timesteps and continue_training:
            continue_training = self.run_and_maybe_train(n_episodes=1)
            # logger.info(f"Performed {self.get_env_steps()} / {total_env_timesteps} action steps. Continue another training round: {continue_training}")

        callback.on_training_end()
        return self    # callback: Optional[Callable] = None,

    def collect_rollouts(
        self,
        env: VecEnv,
        n_episodes: int = 1,
        n_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ReplayBuffer.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param n_episodes: Number of episodes to use to collect rollout data
            You can also specify a ``n_steps`` instead
        :param n_steps: Number of steps to use to collect rollout data
            You can also specify a ``n_episodes`` instead.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :return:
        """

        episode_rewards, total_timesteps = [], []
        total_steps, total_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert env.num_envs == 1, "OffPolicyAlgorithm only support single environment"

        if self.layer_alg.use_sde:
            self.actor.reset_noise()

        callback = self.train_callback

        callback.on_rollout_start()
        continue_training = True

        while total_steps < n_steps or total_episodes < n_episodes:
            done = False
            episode_reward, episode_timesteps = 0.0, 0
            while not done:
                self.update_venv_buf_obs(self.env)
                observation = self.env.venv.buf_obs
                observation = deepcopy(observation) # Required so that values don't get changed.
                self._last_obs = ObsDictWrapper.convert_dict(observation)
                self.layer_alg._last_obs = self._last_obs

                if self.layer_alg.use_sde and self.layer_alg.sde_sample_freq > 0 and total_steps % self.layer_alg.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.actor.reset_noise()

                subgoal_test = False
                if not self.in_subgoal_test_mode and not self.is_bottom_layer: # Next layer can only go in subgoal test mode if this layer is not already in subgoal testing mode
                    subgoal_test = True if np.random.random_sample() < self.subgoal_test_perc else False
                    if subgoal_test:
                        self.set_subgoal_test_mode() # set sub_layer to testing mode is applicable.

                ls = learning_starts
                # if self.layer != 0: ## DEBUG: learning starts to inf causes random action to be selected.
                #     ls = np.inf
                if self.in_subgoal_test_mode:
                    action, buffer_action = self._sample_action(observation=self._last_obs, learning_starts=ls, deterministic=True)
                else:
                    action, buffer_action = self._sample_action(observation=self._last_obs, learning_starts=ls, deterministic=False)
                # if self.layer==1 and episode_timesteps == (self.max_episode_length-1): # Un-Comment this to hard-set the last subgoal to the final goal
                #     action = observation['desired_goal']
                new_obs, reward, done, infos = env.step(action)
                success = np.array([info['is_success'] for info in infos])
                # last_steps_succ.append(success)
                if subgoal_test: # if the subgoal test has started here, unset testing mode of sub_layer if applicable.
                    self.unset_subgoal_test_mode()
                if self.is_bottom_layer and self.train_render_info != 'none' and self.epoch_count % self.render_every_n_eval == 0:
                    if self.render_train == 'record':
                        frame = self.env.unwrapped.render(mode='rgb_array', width=self.train_render_info['size'][0],
                                                             height=self.train_render_info['size'][1])
                        # cv2.imwrite('tmp.png', frame) ## DEBUG
                        self.get_top_layer().train_video_writer.write(frame)
                    elif self.render_train == 'display':
                        self.env.unwrapped.render(mode='human')

                episode_reward += reward

                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos, done)
                self.layer_alg.ep_info_buffer = self.ep_info_buffer
                self.layer_alg.ep_success_buffer = self.ep_success_buffer

                # == Store transition in the replay buffer and/or in the episode storage ==

                if self._vec_normalize_env is not None:
                    # Store only the unnormalized version
                    new_obs_ = self._vec_normalize_env.get_original_obs()
                    reward_ = self._vec_normalize_env.get_original_reward()
                else:
                    # Avoid changing the original ones
                    self._last_original_obs, new_obs_, reward_ = observation, new_obs, reward

                    self.layer_alg._last_original_obs = self._last_original_obs

                # As the VecEnv resets automatically, new_obs is already the
                # first observation of the next episode
                if done and infos[0].get("terminal_observation") is not None:
                    # The saved terminal_observation is not passed through other
                    # VecEnvWrapper, so no need to unnormalize
                    # NOTE: this may be an issue when using other wrappers
                    next_obs = infos[0]["terminal_observation"]
                else:
                    next_obs = new_obs_
                # try:
                finished_early = False
                if self.ep_early_done_on_succ > 1:
                    finished_early = self.check_last_succ(n_ep=self.ep_early_done_on_succ-1) # Check the last episodes except this one.
                    finished_early = np.logical_and(finished_early, success) # Then check if this episode is also finished.
                elif self.ep_early_done_on_succ == 1:
                    finished_early = success # If there is only one step to determined early stop, check if this episode is successful now.

                done = np.logical_or(done, finished_early) # The episode is done if it is finshed early or if it is done via timeout
                self.replay_buffer.add(self._last_original_obs, next_obs, buffer_action, reward_, done, infos)

                self._last_obs = new_obs
                self.layer_alg._last_obs = self._last_obs

                # Save the unnormalized new observation
                if self._vec_normalize_env is not None:
                    self._last_original_obs = new_obs_
                    self.layer_alg._last_original_obs = self._last_original_obs

                # Update progress only for top layer because _total_timesteps is not known for lower-level layers.
                if self.is_top_layer:
                    self.layer_alg._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

                self._current_progress_remaining = self.layer_alg._current_progress_remaining
                # For DQN, check if the target network should be updated
                # and update the exploration schedule
                # For SAC/TD3, the update is done at the same time as the gradient update
                # see https://github.com/hill-a/stable-baselines/issues/900
                self.layer_alg._on_step()

                # Only perform on_steps rollout in lowest layer
                if self.is_bottom_layer:
                    # Only stop training if return value is False, not when it is None.
                    if callback.on_step() is False:
                        self.continue_training = False
                # Ask all sub-layers if we should continue training (self.continue_training is only set in bottom layer)
                if self.get_continue_training() is False:
                    return RolloutReturn(0.0, total_steps, total_episodes, continue_training=False)

                self.actions_since_last_train += 1
                if self.actions_since_last_train >= self.train_freq and self.train_freq != 0:
                    self.train_layer(self.actions_since_last_train)

                # update paramters with model parameters
                self._n_updates = self.layer_alg._n_updates
                self._last_dones = self.layer_alg._last_dones

                if 0 < n_steps < total_steps:
                    break

                self.episode_steps += 1
                self.num_timesteps += 1
                self.layer_alg.num_timesteps = self.num_timesteps
                episode_timesteps += 1 # TODO: Why is there episode_timesteps and self.episode_steps? Is this redundant?
                total_steps += 1

            if done or self.episode_steps >= self.max_episode_length:
                self.replay_buffer.store_episode()

                if self.is_top_layer: # Reset environments and _last_obs of all sub_layers.
                    new_obs = self.env.venv.reset_all()
                    tmp_layer = self
                    while True:
                        tmp_layer._last_obs = new_obs
                        tmp_layer._last_obs = self._last_obs
                        if tmp_layer.sub_layer is None:
                            break
                        else:
                            tmp_layer = self.sub_layer
                self.train_info_list['ep_length'].append(episode_timesteps)
                assert episode_timesteps == self.episode_steps, "DEBUG TEST: If this assertion never triggers, episode_timesteps can be removed and replaced by self.episode_steps to avoid redundancy."
                total_episodes += 1
                self._episode_num += 1
                self.layer_alg._episode_num = self._episode_num
                episode_rewards.append(episode_reward)
                total_timesteps.append(episode_timesteps)

                if action_noise is not None:
                    action_noise.reset()

                self.episode_steps = 0

        mean_reward = np.mean(episode_rewards) if total_episodes > 0 else 0.0

        callback.on_rollout_end()

        return RolloutReturn(mean_reward, total_steps, total_episodes, continue_training)

    def check_last_succ(self, n_ep=None):
        if n_ep is None:
            n_ep = self.ep_early_done_on_succ
        buffer_pos = self.replay_buffer.pos
        ep_info_buffer = self.replay_buffer.info_buffer[buffer_pos]
        if n_ep > 0 and len(ep_info_buffer) >= n_ep:
            last_infos = list(ep_info_buffer)[-n_ep:]
            success = np.array([[inf['is_success'] for inf in info] for info in last_infos])[:, 0]
            finished = np.all(success)
            return finished
        else:
            return False

    def set_subgoal_test_mode(self):
        if self.sub_layer is not None:
            self.sub_layer.in_subgoal_test_mode = True
            self.sub_layer.set_subgoal_test_mode()

    def unset_subgoal_test_mode(self):
        if self.sub_layer is not None:
            self.sub_layer.in_subgoal_test_mode = False
            self.sub_layer.unset_subgoal_test_mode()

    def train_step(self):
        self.sub_layer.run_and_maybe_train(n_episodes=1)

    def test_step(self, eval_env):
        return self.sub_layer.test_episode(eval_env._sub_env)

    def reset_eval_info_list(self):
        self.eval_info_list = {}
        if self.sub_layer is not None:
            self.sub_layer.reset_eval_info_list()

    def update_venv_buf_obs(self, env):
        for i,e in enumerate(env.venv.envs):
            env._save_obs(i, e.env.unwrapped._get_obs())


    def test_episode(self, eval_env):
        done = False
        step_ctr = 0
        info = {}
        ep_reward = 0
        last_succ = []
        # For consistency wrap env.
        eval_env = self._wrap_env(eval_env)
        assert hasattr(eval_env, 'venv'), "Error, vectorized environment required."
        if self.is_top_layer: # Reset simulator for all layers.
            eval_env.venv.reset_all()

        while not done:
            # logger.info("Level {} step {}".format(self.layer, step_ctr)) ## DEBUG
            self.update_venv_buf_obs(eval_env)
            obs = eval_env.venv.buf_obs
            obs = ObsDictWrapper.convert_dict(obs)
            action, _ = self._sample_action(observation=obs,learning_starts=0, deterministic=True)
            q_mean, q_std = self.maybe_get_layer_q_value(action, obs)
            # If it is the last high-level action, then set subgoal to end goal.
            # if self.layer != 0 and step_ctr+1 == eval_env.venv.envs[0]._max_episode_steps:
            #     action = [eval_env.venv.envs[0].goal.copy()]
            # if self.layer == 1: ## DEBUG
            #     logger.info("Setting new subgoal {} for observation {}".format(action, obs))
            # else:
            #     logger.info("Executing low-level action {} for observation {}".format(action, obs))
            new_obs, reward, done, info = eval_env.step(action)
            info.append({'q_mean': q_mean, 'q_std': q_std})
            # if self.layer == 0: ## DEBUG
            #     logger.info(" New obs after ll-action: {}".format(ObsDictWrapper.convert_dict(new_obs)))
            #     logger.info(" desired goal after ll-action: {}".format(new_obs['desired_goal']))
            #     logger.info(" achieved goal after ll-action: {}".format(new_obs['achieved_goal']))
            if self.is_bottom_layer and self.test_render_info != 'none' and self.epoch_count % self.render_every_n_eval == 0:
                if self.render_test == 'record':
                    frame = eval_env.unwrapped.render(mode='rgb_array', width=self.test_render_info['size'][0],
                                                         height=self.test_render_info['size'][1])
                    # cv2.imwrite('tmp_test.png', frame)  ## DEBUG
                    self.get_top_layer().test_video_writer.write(frame)
                elif self.render_test == 'display':
                    eval_env.unwrapped.render(mode='human')
            if self.sub_layer is not None:
                self.sub_layer.reset_eval_info_list()
            step_ctr += 1
            if type(info) == list:
                info = get_concat_dict_from_dict_list(info)
            if 'is_success' in info.keys():
                last_succ.append(info['is_success'].copy())
                info['step_success'] = info['is_success'].copy()
                del info['is_success']
                # logger.info("Success in layer {}: {}".format(self.layer, info['step_success'])) ## DEBUG:
                if self.ep_early_done_on_succ > 0 and len(last_succ) >= self.ep_early_done_on_succ:
                    last_succ_steps = last_succ[-self.ep_early_done_on_succ:]
                    finished = np.all(last_succ_steps)
                    done = np.logical_or(done, finished)

            ep_reward += np.sum(reward)
            for k,v in info.items():
                if k.find("test_") != 0:
                    layered_info_key = "test_{}/{}".format(self.layer, k)
                else:
                    layered_info_key = k
                if layered_info_key not in self.eval_info_list.keys():
                    self.eval_info_list[layered_info_key] = []
                if type(v) == list and len(v) > 0: # TODO: Something is wrong here!
                    if isinstance(v[0], numbers.Number):
                        self.eval_info_list[layered_info_key] += v
                else:
                    if isinstance(v, numbers.Number):
                        self.eval_info_list[layered_info_key].append(v)

        eplen_key = 'test_{}/ep_length'.format(self.layer)
        success_key = 'test_{}/ep_success'.format(self.layer)
        reward_key = 'test_{}/ep_reward'.format(self.layer)
        if eplen_key not in self.eval_info_list.keys():
            self.eval_info_list[eplen_key] = [step_ctr]
        if success_key not in self.eval_info_list.keys():
            if 'step_success' in info.keys():
                self.eval_info_list[success_key] = last_succ[-1].copy()
            else:
                assert False, "Error, step_success key not in info values."
                # logger.info("Episode success in layer {}: {}".format(self.layer, last_succ)) ## DEBUG
        if reward_key not in self.eval_info_list.keys():
            self.eval_info_list[reward_key] = [ep_reward]
        return self.eval_info_list

    def maybe_get_layer_q_value(self, action, obs):
        try: # if we are lucky we obtain the layer_alg's q value like this.
            if self.device.type != 'cpu':
                th_obs = th.from_numpy(obs).cuda()
                th_act = th.from_numpy(action).cuda()
            else:
                th_obs = th.from_numpy(obs)
                th_act = th.from_numpy(action)
            with th.no_grad():
                q = th.stack(self.layer_alg.critic(th_obs, th_act))
                q_mean = float(th.mean(q))
                q_std = float(th.std(q))
        except Exception as e: # otherwise just return none.
            q_mean = None
            q_std = None
        return q_mean, q_std

    def _sample_her_transitions(self) -> None:
        """
        Sample additional goals and store new transitions in replay buffer
        when using offline sampling.
        """

        # Sample goals and get new observations
        # maybe_vec_env=None as we should store unnormalized transitions,
        # they will be normalized at sampling time
        observations, next_observations, actions, rewards = self._episode_storage.sample_offline(
            n_sampled_goal=self.n_sampled_goal
        )

        # store data in replay buffer
        dones = np.zeros((len(observations)), dtype=bool)
        self.replay_buffer.extend(observations, next_observations, actions, rewards, dones)

    def __getattr__(self, item: str) -> Any:
        """
        Find attribute from layer class if this class does not have it.
        """
        if hasattr(self.layer_alg, item):
            return getattr(self.layer_alg, item)
        else:
            raise AttributeError(f"{self} has no attribute {item}")

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        return self.layer_alg._get_torch_save_params()

    def save(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        exclude: Optional[Iterable[str]] = None,
        include: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Save all the attributes of the object and the layer_alg parameters in a zip-file.

        :param path: path to the file where the rl agent should be saved
        :param exclude: name of parameters that should be excluded in addition to the default one
        :param include: name of parameters that might be excluded but should be included anyway
        """
        # add HAC parameters to model
        layer_data = self.__dict__.copy()
        exclude_params = HAC.save_attrs_to_exclude
        for k,v in layer_data.items():
            if k in self.layer_alg.__dict__.keys() and k not in exclude_params:
                try:
                    valid = layer_data[k] == v
                    if type(valid) == np.ndarray:
                        valid = valid.all()
                    if not valid:
                        logger.info(f"Warning, mismatch of parameter {k} in model of HAC ({v}) and HAC itself ({layer_data[k]}).")
                        exclude_params.append(k)
                except:
                    logger.info(f"Warning, cannot compare parameter {k} in model of HAC and HAC itself.")
                    exclude_params.append(k)
        for param_name in exclude_params:
            layer_data.pop(param_name, None)

        self.layer_alg.layer_data = layer_data
        layer_path = path + f"_lay{self.layer}"
        model_excludes = ['replay_buffer']
        self.layer_alg.save(layer_path, model_excludes, include)
        if self.sub_layer is not None:
            self.sub_layer.save(path)

    @classmethod
    def load(
        cls,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto",
        **policy_args,
    ) -> "BaseAlgorithm":
        """
        Load the layer_alg from a zip-file

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded agent on
            (can be None if you only need prediction from a trained agent) has priority over any saved environment
        :param device: Device on which the code should run.
        :param kwargs: extra arguments to change the agent when loading
        """
        parent_loaded_model = cls('MlpPolicy', env, **policy_args)
        layer_model = parent_loaded_model
        n_layers = len(policy_args['time_scales'])
        for lay in reversed(range(n_layers)):
            layer_path = path + f"_lay{lay}"
            data, params, pytorch_variables = load_from_zip_file(layer_path, device=device)

            # Remove stored device information and replace with ours
            if "policy_kwargs" in data:
                if "device" in data["policy_kwargs"]:
                    del data["policy_kwargs"]["device"]

            if "observation_space" not in data or "action_space" not in data:
                raise KeyError("The observation_space and action_space were not given, can't verify new environments")

            layer_model._setup_model()
            ld_copy = data['layer_data']
            for k in HAC.save_attrs_to_exclude:
                if k in ld_copy.keys():
                    del ld_copy[k]
            layer_model.__dict__.update(ld_copy)
            del data['layer_data']
            layer_model.layer_alg.__dict__.update(data)

            # put state_dicts back in place
            layer_model.layer_alg.set_parameters(params, exact_match=True, device=device)

            # put other pytorch variables back in place
            if pytorch_variables is not None:
                for name in pytorch_variables:
                    recursive_setattr(layer_model.layer_alg, name, pytorch_variables[name])

            # Sample gSDE exploration matrix, so it uses the right device
            # see issue #44
            if layer_model.layer_alg.use_sde:
                layer_model.layer_alg.policy.reset_noise()  # pytype: disable=attribute-error

            layer_model = layer_model.sub_layer
        return parent_loaded_model

    def load_replay_buffer(
        self, path: Union[str, pathlib.Path, io.BufferedIOBase], truncate_last_trajectory: bool = True
    ) -> None:
        """
        Load a replay buffer from a pickle file and set environment for replay buffer.

        :param path: Path to the pickled replay buffer.
        :param truncate_last_trajectory: Only for online sampling.
            If set to ``True`` we assume that the last trajectory in the replay buffer was finished.
            If it is set to ``False`` we assume that we continue the same trajectory (same episode).
        """
        self.layer_alg.load_replay_buffer(path=path)

        # set environment
        self.replay_buffer.set_env(self.env)
        # If we are at the start of an episode, no need to truncate
        current_idx = self.replay_buffer.current_idx

        # truncate interrupted episode
        if truncate_last_trajectory and current_idx > 0:
            warnings.warn(
                "The last trajectory in the replay buffer will be truncated.\n"
                "If you are in the same episode as when the replay buffer was saved,\n"
                "you should use `truncate_last_trajectory=False` to avoid that issue."
            )
            # get current episode and transition index
            pos = self.replay_buffer.pos
            # set episode length for current episode
            self.replay_buffer.episode_lengths[pos] = current_idx
            # set done = True for current episode
            # current_idx was already incremented
            self.replay_buffer.buffer["done"][pos][current_idx - 1] = np.array([True], dtype=np.float32)
            # reset current transition index
            self.replay_buffer.current_idx = 0
            # increment episode counter
            self.replay_buffer.pos = (self.replay_buffer.pos + 1) % self.replay_buffer.max_episode_stored
            # update "full" indicator
            self.replay_buffer.full = self.replay_buffer.full or self.replay_buffer.pos == 0

    def get_env_steps(self) -> int:
        if self.sub_layer is None:
            return self.num_timesteps
        else:
            return self.sub_layer.get_env_steps()

    def _record_logs(self) -> None:
        """
        Write log.
        """
        time_pf = "time_{}".format(self.layer)
        rollout_pf = "rollout_{}".format(self.layer)
        train_pf = "train_{}".format(self.layer)

        fps = int(self.num_timesteps / (time.time() - self.start_time))
        logger.record(time_pf + "/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            logger.record(rollout_pf + "/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            logger.record(rollout_pf + "/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        logger.record(time_pf + "/fps", fps)
        logger.record(time_pf + "/total layer steps", self.num_timesteps, exclude="tensorboard")
        # env_steps = self.get_env_steps()
        # logger.record(time_pf + "/total timesteps", env_steps, exclude="tensorboard")
        if self.is_top_layer:
            env_steps = self.get_env_steps()
            logger.record("time/total timesteps", env_steps, exclude="tensorboard")
            logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
        if self.use_sde:
            logger.record(train_pf + "/std", (self.actor.get_std()).mean().item())

        if len(self.ep_success_buffer) > 0:
            logger.record(rollout_pf + "/success_rate", safe_mean(self.ep_success_buffer))
            if safe_mean(self.ep_success_buffer) > 0.9: # Start training of higher level layer if the success rate is above 90%. This has only an effect if action replay is disabled.
                if self.parent_layer is not None:
                    self.parent_layer.set_learning_enabled()

        if self.sub_layer is not None:
            self.sub_layer._record_logs()

        for k,v in self.train_info_list.items():
            logger.record(train_pf + f"/{k}", safe_mean(v))
            logger.record(train_pf + f"/{k}_std", np.std(v))
        self.reset_train_info_list()

        self.epoch_count += 1

        if self.is_top_layer:
            logger.record("epoch", self.epoch_count)
            try:
                succ_rate = logger.Logger.CURRENT.name_to_value['test/success_rate']
            except Exception as e:
                logger.info("Error getting test success rate")
                succ_rate = 0
            hyperopt_score = float(succ_rate/self.epoch_count)
            logger.record("hyperopt_score", hyperopt_score, exclude="tensorboard")
            if self.epoch_count % self.render_every_n_eval == 0:
                if self.train_render_info is not None:
                    self.start_train_video_writer(self.get_env_steps())
                if self.test_render_info is not None:
                    self.start_test_video_writer(self.get_env_steps())
            if (self.epoch_count - 1) % self.render_every_n_eval == 0:
                if self.train_render_info is not None:
                    self.stop_train_video_writer()
                if self.test_render_info is not None:
                    self.stop_test_video_writer()

    def reset_train_info_list(self):
        self.train_info_list = {'ep_length': []}

    def _dump_logs(self) -> None:
        self._record_logs()
        top_layer = self.layer
        # # For compatibility with HER, add a few redundant extra fields:
        # copy_fields = {'time/total timesteps': 'time_{}/total timesteps'.format(top_layer)
        #                }
        # copy_fields = {}
        # for k,v in copy_fields.items():
        #     logger.record(k, logger.Logger.CURRENT.name_to_value[v])
        logger.info("Writing log data to {}".format(logger.get_dir()))
        logger.dump(step=self.num_timesteps)

    def _sample_action(
        self, observation: np.ndarray = None, learning_starts: int = 0, deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function is copied from OffPolicyAlgorithm class, but takes as additional input a "deterministic"
        parameter to determine whether or not add action noise, instead of the action_noise argument which
        is more or less unused, as far as I can tell.
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param observation: An optional observation to base the action prediction upon. If not provided,
            layer_alg._last_obs will be used instead.
        :param deterministic: Whether the policy selects a determinsitic action or adds random noise to it.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        assert isinstance(self.layer_alg, OffPolicyAlgorithm), "Error, layer_alg ist not an OffPolicyAlgorithm"
        if observation is None:
            observation = self.layer_alg._last_obs
        # Select action randomly or according to policy
        # if self.layer_alg.num_timesteps < learning_starts or self.learning_enabled is False:
        if self.replay_buffer.size() < learning_starts or self.learning_enabled is False:
        # if self.layer_alg.num_timesteps < learning_starts and (not (self.layer_alg.use_sde and self.layer_alg.use_sde_at_warmup)):
            # Warmup phase
            unscaled_action = np.array([self.layer_alg.action_space.sample()])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            try:
                unscaled_action, _ = self.layer_alg.predict(observation, deterministic=deterministic)
            except Exception as e:
                logger.info("ohno {}".format(e))

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.layer_alg.action_space, gym.spaces.Box):
            scaled_action = self.layer_alg.policy.scale_action(unscaled_action)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.layer_alg.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action

    #@staticmethod
    def _wrap_env(self, env: GymEnv, verbose: int = 0) -> HierarchicalVecEnv:
        if not isinstance(env, ObsDictWrapper):
            if not isinstance(env, HierarchicalVecEnv):
                env = HierarchicalVecEnv([lambda: env], self.ep_early_done_on_succ)

                if is_image_space(env.observation_space) and not is_wrapped(env, VecTransposeImage):
                    env = VecTransposeImage(env)
            # check if wrapper for dict support is needed when using HER
            if isinstance(env.observation_space, gym.spaces.dict.Dict):
                env = ObsDictWrapper(env)

        return env
    # wrap_env = _wrap_env

    def get_top_layer(self):
        if self.is_top_layer:
            return self
        else:
            return self.parent_layer.get_top_layer()

    def set_test_render_info(self, render_info):
        self.test_render_info = render_info
        if self.sub_layer is not None:
            self.sub_layer.set_test_render_info(render_info)

    def set_train_render_info(self, render_info):
        self.train_render_info = render_info
        if self.sub_layer is not None:
            self.sub_layer.set_train_render_info(render_info)