import io
import pathlib
import warnings
from typing import Any, Iterable, List, Optional, Tuple, Type, Union
import numpy as np
import torch as th
import gym
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import load_from_zip_file, recursive_setattr
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn
from stable_baselines3.common.utils import check_for_correct_spaces
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper
from ideas_baselines.mbchac.goal_selection_strategy import KEY_TO_GOAL_STRATEGY, GoalSelectionStrategy
from ideas_baselines.mbchac.hher_replay_buffer import HHerReplayBuffer
from ideas_baselines.mbchac.hierarchical_env import get_h_envs_from_env
from stable_baselines3.common import logger
from stable_baselines3.common.utils import safe_mean
from gym.wrappers import TimeLimit
import time
from ideas_baselines.mbchac.util import get_concat_dict_from_dict_list, merge_list_dicts
import numbers
from stable_baselines3.common.logger import HumanOutputFormat
import sys
from stable_baselines3.common.vec_env import VecVideoRecorder
import cv2
from watchpoints import watch


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
                "or pass `max_episode_length` to the model constructor"
            )
    return current_max_episode_length


def compute_time_scales(time_scales_str, env):
    scales = time_scales_str.split(",")
    max_steps = env.spec.max_episode_steps
    for i,s in enumerate(scales):
        if s == '_':
            defined_steps = np.product([int(step) for step in scales[:i]])
            defined_after_steps = np.product([int(step) for step in scales[i+1:]])
            defined_steps *= defined_after_steps
            assert max_steps % defined_steps == 0, "Error defined time_scale not compatible with environment max steps. Max. number of environment steps {} needs to be divisible by product of all defined steps {}.".format(max_steps, defined_steps)
            this_steps = int(max_steps / defined_steps)
            scales[i] = str(this_steps)
    assert np.product([int(step) for step in scales]) == max_steps, "Error defined time_scale not compatible with environment max steps. Product of all steps needs to be {}".format(max_steps)
    return ",".join(scales)



class MBCHAC(BaseAlgorithm):
    """
    Model-based curious hierarchical actor-critic MBCHAC
    Cobed based on Hindsight Experience Replay (HER) code of stable baselines 3 repository

    .. warning::

      For performance reasons, the maximum number of steps per episodes must be specified.
      In most cases, it will be inferred if you specify ``max_episode_steps`` when registering the environment
      or if you use a ``gym.wrappers.TimeLimit`` (and ``env.spec`` is not None).
      Otherwise, you can directly pass ``max_episode_length`` to the model constructor


    For additional offline algorithm specific arguments please have a look at the corresponding documentation.

    :param policy: The policy model to use.
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param model_classes: Array of Off policy models which will be used with hindsight experience replay. (SAC, TD3, DDPG, DQN)
    :param n_sampled_goal: Number of sampled goals for replay. (offline sampling)
    :param goal_selection_strategy: Strategy for sampling goals for replay.
        One of ['episode', 'final', 'future', 'random']
    :param learning_rate: learning rate for the optimizer,
        it can be a function of the current progress remaining (from 1 to 0)
    :param max_episode_length: The m
    aximum length of an episode. If not specified,
        it will be automatically inferred if the environment uses a ``gym.wrappers.TimeLimit`` wrapper.
    """

    def __init__(
        self,
        policy: Union[str, Type[BasePolicy]],
        env: Union[GymEnv, str],
        model_class: Type[OffPolicyAlgorithm],
        sub_model_classes: List[Type[OffPolicyAlgorithm]] = [],
        time_scales: str = '_',
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
        *args,
        **kwargs,
    ):
        self.gradient_steps = n_train_batches
        self.learn_callback = None
        # determine time_scales
        self.is_top_layer = is_top_layer
        self.time_scales = time_scales
        self.train_freq = train_freq
        self.is_bottom_layer = len(self.time_scales.split(",")) == 1
        assert len(time_scales.split(",")) == (
                    len(sub_model_classes) + 1), "Error, number of time scales is not equal to number of layers."
        assert time_scales.count(
            "_") <= 1, "Error, only one wildcard character \'_\' allowed in time_scales argument {}".format(time_scales)
        if self.is_top_layer == 1:  # Only do this once at top layer.
            self.time_scales = compute_time_scales(time_scales, env)
            # Build layer_env from env, depending on steps and action space.
            layer_envs = get_h_envs_from_env(env, self.time_scales)
        time_scales_int = [int(s) for s in self.time_scales.split(",")]
        self.level_steps_per_episode = int(self.time_scales.split(",")[0])
        if max_episode_length is None:
            max_episode_length = self.level_steps_per_episode
        self.max_steps_per_layer_action = np.product(time_scales_int[1:]) # the max. number of low-level steps per action on this layer.
        bottom_env = layer_envs[-1]
        this_env = layer_envs[0]
        this_env.env.model = self
        # Build policy, convert env into <ObstDictWrapper> class.
        super(MBCHAC, self).__init__(policy=BasePolicy, env=this_env, policy_base=BasePolicy, learning_rate=0.0)
        # we will use the policy and learning rate from the model.
        del self.policy, self.learning_rate

        self.learning_starts = kwargs['learning_starts']
        # watch(self.learning_starts)

        _init_setup_model = kwargs.get("_init_setup_model", True)
        if "_init_setup_model" in kwargs:
            del kwargs["_init_setup_model"]
        # model initialization
        self.model_class = model_class
        model_args = kwargs.copy()
        del model_args['model_classes']

        self.sub_model_classes = sub_model_classes
        self.layer = len(self.sub_model_classes)

        assert (len(sub_model_classes) + 1) == len(layer_envs), "Error, number of sub model classes should be one less than number of envs"
        sub_level_steps = '0'
        if len(sub_model_classes) > 0:
            sub_level_steps = ",".join(self.time_scales.split(",")[1:])
            self.sub_model = MBCHAC('MlpPolicy', bottom_env, sub_model_classes[0], sub_model_classes[1:],
                                    time_scales=sub_level_steps,
                                    n_sampled_goal=n_sampled_goal, goal_selection_strategy=goal_selection_strategy,
                                    train_freq=self.train_freq,
                                    is_top_layer=False, layer_envs=layer_envs[1:],
                                    **kwargs)
        else:
            self.sub_model = None

        self.model = model_class(
            policy=policy,
            env=self.env,
            _init_setup_model=False,  # pytype: disable=wrong-keyword-args
            *args,
            **model_args,  # pytype: disable=wrong-keyword-args
        )

        self.verbose = self.model.verbose
        self.tensorboard_log = self.model.tensorboard_log

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
        perform_action_replay = not self.is_bottom_layer
        sample_test_trans_fraction = (not self.is_bottom_layer) * 0.1
        self._episode_storage = HHerReplayBuffer(
            self.env,
            her_buffer_size,
            max_episode_length,
            self.goal_selection_strategy,
            self.env.observation_space,
            self.env.action_space,
            self.device,
            self.n_envs,
            self.her_ratio,  # pytype: disable=wrong-arg-types
            perform_action_replay,
            sample_test_trans_fraction,
            int(sub_level_steps)
        )

        # counter for steps in episode
        self.episode_steps = 0
        if _init_setup_model:
            self._setup_model()

        self.train_callback = None
        self.tmp_train_logger = logger.Logger(folder=None, output_formats=[]) # HumanOutputFormat(sys.stdout)
        self.test_render_frames = []
        self.test_render_info = None
        self.train_render_frames = []
        self.train_render_info = None
        self.subgoal_test_perc = subgoal_test_perc

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

        self.epoch_count = 0

        self.in_subgoal_test_mode = False

        self.train_overwrite_goals = []
        self.test_overwrite_goals = []

    def start_train_video_writer(self, n):
        self.reset_train_render_frames()
        self.train_video_writer = cv2.VideoWriter(self.train_render_info['path'] + '/train_{}.avi'.format(n),
                        cv2.VideoWriter_fourcc('F', 'M', 'P', '4'), self.train_render_info['fps'],
                        self.train_render_info['size'])
    def stop_train_video_writer(self):
        frames = self.get_train_render_frames()
        if frames is not None and len(frames) > 0:
            for f in frames:
                self.train_video_writer.write(f)
        self.train_video_writer.release()
        self.reset_train_render_frames()
        
    def start_test_video_writer(self, n):
        self.reset_test_render_frames()
        self.test_video_writer = cv2.VideoWriter(self.test_render_info['path'] + '/test_{}.avi'.format(n),
                        cv2.VideoWriter_fourcc('F', 'M', 'P', '4'), self.test_render_info['fps'],
                        self.test_render_info['size'])
    def stop_test_video_writer(self):
        frames = self.get_test_render_frames()
        if frames is not None and len(frames) > 0:
            for f in frames:
                self.test_video_writer.write(f)
        self.test_video_writer.release()
        self.reset_test_render_frames()

    def _setup_model(self) -> None:
        self.model._setup_model()
        # assign episode storage to replay buffer when using online HER sampling
        self.model.replay_buffer = self._episode_storage

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        # if self.continuous_subgoals and
        if self.sub_model is not None:
            subgoal = self.model.predict(observation, state, mask, deterministic)
            observation['desired_goal'] = subgoal[0]
        if self.sub_model is None:
            action = self.model.predict(observation, state, mask, deterministic)
        else:
            action = self.sub_model.predict(observation, state, mask, deterministic)
        return action


    def train_model(self, n_gradient_steps: int):
        if self.num_timesteps > self.learning_starts and self.replay_buffer.size() > 0:
            logger.info("Training layer {} for {} steps.".format(self.layer, n_gradient_steps))
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
            logger.info("Did not yet train layer {} because I have not yet enough experience collected.".format(self.layer))

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
            self.train_model(rollout.episode_timesteps)
        return True


    def init_learn(self, total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps):
        tb_log_name = "MBCHAC_{}".format(self.layer)
        layer_total_timesteps = total_timesteps / self.max_steps_per_layer_action
        layer_total_timesteps, callback = self._setup_learn(
            layer_total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )
        self.epoch_count = 0
        self.model.start_time = self.start_time
        self.model.ep_info_buffer = self.ep_info_buffer
        self.model.ep_success_buffer = self.ep_success_buffer
        self.model.num_timesteps = self.num_timesteps
        self.model._episode_num = self._episode_num
        self.model._last_obs = self._last_obs
        self.model._total_timesteps = self._total_timesteps
        if self.sub_model is not None:
            self.sub_model.init_learn(total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps)
        self.train_callback = callback
        self.actions_since_last_train = 0
        return layer_total_timesteps, callback

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = None,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "MBCHAC",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> BaseAlgorithm:

        self.epoch_count = 0

        total_timesteps, callback = self.init_learn(total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps)

        callback.on_training_start(locals(), globals())
        self.actions_since_last_train = 0
        continue_training = True
        if self.is_top_layer and self.train_render_info is not None:
            self.start_train_video_writer(self.epoch_count)
        if self.is_top_layer and self.test_render_info is not None:
            self.start_test_video_writer(self.epoch_count)
        while self.num_timesteps < total_timesteps and continue_training:
            continue_training = self.run_and_maybe_train(n_episodes=1)
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

        if self.model.use_sde:
            self.actor.reset_noise()

        callback = self.train_callback

        callback.on_rollout_start()
        continue_training = True

        while total_steps < n_steps or total_episodes < n_episodes:
            done = False
            episode_reward, episode_timesteps = 0.0, 0

            while not done:
                # concatenate observation and (desired) goal
                observation = self._last_obs
                self._last_obs = ObsDictWrapper.convert_dict(observation)

                if self.model.use_sde and self.model.sde_sample_freq > 0 and total_steps % self.model.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.actor.reset_noise()

                # Select action randomly or according to policy
                self.model._last_obs = self._last_obs
                step_obs = observation
                if self.train_overwrite_goals != []:
                    step_obs['desired_goal'] = self.train_overwrite_goals.copy()
                    self.train_overwrite_goals = []
                step_obs = ObsDictWrapper.convert_dict(step_obs)

                subgoal_test = False
                if not self.in_subgoal_test_mode and not self.is_bottom_layer: # Next layer can only go in subgoal test mode if this layer is not already in subgoal testing mode
                    subgoal_test = True if np.random.random_sample() < self.subgoal_test_perc else False
                    if subgoal_test:
                        self.set_subgoal_test_mode() # set submodel to testing mode is applicable.
                if self.in_subgoal_test_mode:
                    action, buffer_action = self._sample_action(observation=step_obs, learning_starts=learning_starts, deterministic=True)
                else:
                    action, buffer_action = self._sample_action(observation=step_obs, learning_starts=learning_starts, deterministic=False)
                # if self.layer==1 and episode_timesteps == (self.max_episode_length-1):
                #     action = observation['desired_goal']
                new_obs, reward, done, infos = env.step(action)
                if subgoal_test: # if the subgoal test has started here, unset testing mode of submodel if applicable.
                    self.unset_subgoal_test_mode()
                if self.is_bottom_layer and self.train_render_info is not None:
                    frame = self.env.venv.envs[0].render(mode='rgb_array', width=self.train_render_info['size'][0],
                                                             height=self.train_render_info['size'][1])
                    self.train_render_frames.append(frame)

                self.num_timesteps += 1
                self.model.num_timesteps = self.num_timesteps
                episode_timesteps += 1
                total_steps += 1

                episode_reward += reward

                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos, done)
                self.model.ep_info_buffer = self.ep_info_buffer
                self.model.ep_success_buffer = self.ep_success_buffer

                # == Store transition in the replay buffer and/or in the episode storage ==

                if self._vec_normalize_env is not None:
                    # Store only the unnormalized version
                    new_obs_ = self._vec_normalize_env.get_original_obs()
                    reward_ = self._vec_normalize_env.get_original_reward()
                else:
                    # Avoid changing the original ones
                    self._last_original_obs, new_obs_, reward_ = observation, new_obs, reward

                    self.model._last_original_obs = self._last_original_obs

                # As the VecEnv resets automatically, new_obs is already the
                # first observation of the next episode
                if done and infos[0].get("terminal_observation") is not None:
                    # The saved terminal_observation is not passed through other
                    # VecEnvWrapper, so no need to unnormalize
                    # NOTE: this may be an issue when using other wrappers
                    next_obs = infos[0]["terminal_observation"]
                else:
                    next_obs = new_obs_
                self.replay_buffer.add(self._last_original_obs, next_obs, buffer_action, reward_, done, infos)

                self._last_obs = new_obs
                self.model._last_obs = self._last_obs

                # Save the unnormalized new observation
                if self._vec_normalize_env is not None:
                    self._last_original_obs = new_obs_
                    self.model._last_original_obs = self._last_original_obs

                # Update progress only for top layer because _total_timesteps is not known for lower-level layers.
                if self.is_top_layer:
                    self.model._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

                # For DQN, check if the target network should be updated
                # and update the exploration schedule
                # For SAC/TD3, the update is done as the same time as the gradient update
                # see https://github.com/hill-a/stable-baselines/issues/900
                self.model._on_step()

                self.episode_steps += 1

                # Only perform on_steps rollout in lowest layer
                if self.is_bottom_layer:
                    # Only stop training if return value is False, not when it is None.
                    if callback.on_step() is False:
                        return RolloutReturn(0.0, total_steps, total_episodes, continue_training=False)

                self.actions_since_last_train += 1
                if self.actions_since_last_train >= self.train_freq and self.train_freq != 0:
                    self.train_model(self.actions_since_last_train)

                if 0 < n_steps <= total_steps:
                    break

            if done or self.episode_steps >= self.max_episode_length:
                self.replay_buffer.store_episode()

                total_episodes += 1
                self._episode_num += 1
                self.model._episode_num = self._episode_num
                episode_rewards.append(episode_reward)
                total_timesteps.append(episode_timesteps)

                if action_noise is not None:
                    action_noise.reset()

                self.episode_steps = 0

        mean_reward = np.mean(episode_rewards) if total_episodes > 0 else 0.0

        callback.on_rollout_end()

        return RolloutReturn(mean_reward, total_steps, total_episodes, continue_training)

    def set_subgoal_test_mode(self):
        if self.sub_model is not None:
            self.sub_model.in_subgoal_test_mode = True
            self.sub_model.set_subgoal_test_mode()

    def unset_subgoal_test_mode(self):
        if self.sub_model is not None:
            self.sub_model.in_subgoal_test_mode = False
            self.sub_model.unset_subgoal_test_mode()

    def train_step(self):
        self.sub_model.run_and_maybe_train(n_episodes=1)

    def test_step(self, eval_env):
        return self.sub_model.test_episode(eval_env._sub_env)

    def reset_eval_info_list(self):
        self.eval_info_list = {}
        if self.sub_model is not None:
            self.sub_model.reset_eval_info_list()

    def test_episode(self, eval_env, return_ep_info=False):
        done = False
        step_ctr = 0
        info = {}
        ep_reward = 0
        last_succ = 0
        eval_env = BaseAlgorithm._wrap_env(eval_env)
        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])
        while not done:
            if hasattr(eval_env, 'venv'):
                obs = eval_env.venv.buf_obs
            # elif hasattr(eval_env, 'env'):
            #     obs = eval_env.env._get_obs()
            # elif hasattr(eval_env, '_get_obs'):
            #     obs = eval_env._get_obs()
            else:
                assert False, "eval_env type not supported!"
            if self.test_overwrite_goals != []:
                obs['desired_goal'] = self.test_overwrite_goals
                self.test_overwrite_goals = []
            obs = ObsDictWrapper.convert_dict(obs)
            action, _ = self._sample_action(observation=obs,learning_starts=0, deterministic=True)
            new_obs, reward, done, info = eval_env.step(action)
            if self.is_bottom_layer and self.test_render_info is not None:
                if hasattr(eval_env, 'env'):
                    frame = eval_env.env.render(mode='rgb_array', width=self.test_render_info['size'][0],
                                                         height=self.test_render_info['size'][1])
                elif hasattr(eval_env, 'venv'):
                    frame = eval_env.venv.envs[0].render(mode='rgb_array', width=self.test_render_info['size'][0],
                                                     height=self.test_render_info['size'][1])
                else:
                    assert False, "Eval env is neither TimeLimit nor Vectorized Env class and has neither env nor venv property."
                self.test_render_frames.append(frame)
            if self.sub_model is not None:
                self.sub_model.reset_eval_info_list()
            step_ctr += 1
            if type(info) == list:
                info = get_concat_dict_from_dict_list(info)
            if 'is_success' in info.keys():
                last_succ = info['is_success']
                info['step_success'] = info['is_success']
                del info['is_success']
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
                self.eval_info_list[success_key] = [last_succ]
        if reward_key not in self.eval_info_list.keys():
            self.eval_info_list[reward_key] = [ep_reward]
        return self.eval_info_list

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
        Find attribute from model class if this class does not have it.
        """
        if hasattr(self.model, item):
            return getattr(self.model, item)
        else:
            raise AttributeError(f"{self} has no attribute {item}")

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        return self.model._get_torch_save_params()

    def save(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        exclude: Optional[Iterable[str]] = None,
        include: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Save all the attributes of the object and the model parameters in a zip-file.

        :param path: path to the file where the rl agent should be saved
        :param exclude: name of parameters that should be excluded in addition to the default one
        :param include: name of parameters that might be excluded but should be included anyway
        """

        # add HER parameters to model
        self.model.n_sampled_goal = self.n_sampled_goal
        self.model.goal_selection_strategy = self.goal_selection_strategy
        self.model.model_class = self.model_class
        self.model.max_episode_length = self.max_episode_length

        self.model.save(path, exclude, include)

    @classmethod
    def load(
        cls,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto",
        **kwargs,
    ) -> "BaseAlgorithm":
        """
        Load the model from a zip-file

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param kwargs: extra arguments to change the model when loading
        """
        data, params, pytorch_variables = load_from_zip_file(path, device=device)

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        # check if observation space and action space are part of the saved parameters
        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        # check if given env is valid
        if env is not None:
            # Wrap first if needed
            env = cls._wrap_env(env, data["verbose"])
            # Check if given env is valid
            check_for_correct_spaces(env, data["observation_space"], data["action_space"])
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env" in data:
                env = data["env"]

        if "use_sde" in data and data["use_sde"]:
            kwargs["use_sde"] = True

        # Keys that cannot be changed
        for key in {"model_class", "max_episode_length"}:
            if key in kwargs:
                del kwargs[key]

        # Keys that can be changed
        for key in {"n_sampled_goal", "goal_selection_strategy"}:
            if key in kwargs:
                data[key] = kwargs[key]  # pytype: disable=unsupported-operands
                del kwargs[key]

        # noinspection PyArgumentList
        her_model = cls(
            policy=data["policy_class"],
            env=env,
            model_class=data["model_class"],
            n_sampled_goal=data["n_sampled_goal"],
            goal_selection_strategy=data["goal_selection_strategy"],
            max_episode_length=data["max_episode_length"],
            policy_kwargs=data["policy_kwargs"],
            _init_setup_model=False,  # pytype: disable=not-instantiable,wrong-keyword-args
            **kwargs,
        )

        # load parameters
        her_model.model.__dict__.update(data)
        her_model.model.__dict__.update(kwargs)
        her_model._setup_model()

        her_model._total_timesteps = her_model.model._total_timesteps
        her_model.num_timesteps = her_model.model.num_timesteps
        her_model._episode_num = her_model.model._episode_num

        # put state_dicts back in place
        her_model.model.set_parameters(params, exact_match=True, device=device)

        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                recursive_setattr(her_model.model, name, pytorch_variables[name])

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if her_model.model.use_sde:
            her_model.model.policy.reset_noise()  # pytype: disable=attribute-error
        return her_model

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
        self.model.load_replay_buffer(path=path)


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
        if self.sub_model is None:
            return self.num_timesteps
        else:
            return self.sub_model.get_env_steps()

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
        logger.record(time_pf + "/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
        logger.record(time_pf + "/total layer steps", self.num_timesteps, exclude="tensorboard")
        env_steps = self.get_env_steps()
        logger.record(time_pf + "/total timesteps", env_steps, exclude="tensorboard")
        if self.use_sde:
            logger.record(train_pf + "/std", (self.actor.get_std()).mean().item())

        if len(self.ep_success_buffer) > 0:
            logger.record(rollout_pf + "/success_rate", safe_mean(self.ep_success_buffer))

        if self.sub_model is not None:
            self.sub_model._record_logs()

        self.epoch_count += 1
        if self.is_top_layer and self.train_render_info is not None:
            self.stop_train_video_writer()
            self.start_train_video_writer(self.epoch_count)

        if self.is_top_layer and self.test_render_info is not None:
            self.stop_test_video_writer()
            self.start_test_video_writer(self.epoch_count)
        # Pass the number of timesteps for tensorboard


    def _dump_logs(self) -> None:
        self._record_logs()
        top_layer = self.layer
        # # For compatibility with HER, add a few redundant extra fields:
        copy_fields = {'time/total timesteps': 'time_{}/total timesteps'.format(top_layer)
                       }

        for k,v in copy_fields.items():
            logger.record(k, logger.Logger.CURRENT.name_to_value[v])
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
            model._last_obs will be used instead.
        :param deterministic: Whether the policy selects a determinsitic action or adds random noise to it.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        assert isinstance(self.model, OffPolicyAlgorithm), "Error, model ist not an OffPolicyAlgorithm"
        if observation is None:
            observation = self.model._last_obs
        # Select action randomly or according to policy
        if self.model.num_timesteps < learning_starts and not (self.model.use_sde and self.model.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.model.action_space.sample()])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            unscaled_action, _ = self.model.predict(observation, deterministic=deterministic)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.model.action_space, gym.spaces.Box):
            scaled_action = self.model.policy.scale_action(unscaled_action)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.model.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action

    def set_test_render_info(self, render_info):
        self.test_render_info = render_info
        if self.sub_model is not None:
            self.sub_model.set_test_render_info(render_info)

    def set_train_render_info(self, render_info):
        self.train_render_info = render_info
        if self.sub_model is not None:
            self.sub_model.set_train_render_info(render_info)

    def get_test_render_frames(self):
        if self.sub_model is not None:
            return self.sub_model.get_test_render_frames()
        else:
            return self.test_render_frames

    def get_train_render_frames(self):
        if self.sub_model is not None:
            return self.sub_model.get_train_render_frames()
        else:
            return self.train_render_frames

    def reset_test_render_frames(self):
        self.test_render_frames = []
        if self.sub_model is not None:
            self.sub_model.reset_test_render_frames()

    def reset_train_render_frames(self):
        self.train_render_frames = []
        if self.sub_model is not None:
            self.sub_model.reset_train_render_frames()
