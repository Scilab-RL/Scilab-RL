import io
import pathlib
import warnings
from typing import Any, Iterable, List, Optional, Tuple, Type, Union
import numpy as np
import torch as th
import gym
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import load_from_zip_file, recursive_setattr
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn
from stable_baselines3.common.utils import check_for_correct_spaces
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper
from stable_baselines3.her.goal_selection_strategy import KEY_TO_GOAL_STRATEGY, GoalSelectionStrategy
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from ideas_baselines.mbchac.hierarchical_env import get_h_envs_from_env



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
    :param online_sampling: Sample HER transitions online.
    :param learning_rate: learning rate for the optimizer,
        it can be a function of the current progress remaining (from 1 to 0)
    :param max_episode_length: The maximum length of an episode. If not specified,
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
        online_sampling: bool = False,
        max_episode_length: Optional[int] = None,
        is_top_layer: bool = True,
        layer_envs: List[GymEnv] = [],
        continuous_subgoals = False,
        n_episodes_rollout: int = -1,
        *args,
        **kwargs,
    ):
        # determine time_scales
        self.is_top_layer = is_top_layer
        self.time_scales = time_scales
        self.is_bottom_env = len(self.time_scales.split(",")) == 1
        assert len(time_scales.split(",")) == (
                    len(sub_model_classes) + 1), "Error, number of time scales is not equal to number of layers."
        assert time_scales.count(
            "_") <= 1, "Error, only one wildcard character \'_\' allowed in time_scales argument {}".format(time_scales)
        if self.is_top_layer == 1:  # Only do this once at top layer.
            self.time_scales = compute_time_scales(time_scales, env)
            time_scales_int = [int(s) for s in self.time_scales.split(",")]
            # Build layer_env from env, depending on steps and action space.
            layer_envs = get_h_envs_from_env(env, time_scales_int)
        self.level_steps = int(self.time_scales.split(",")[0])
        if max_episode_length is None:
            max_episode_length = self.level_steps
        bottom_env = layer_envs[-1]

        this_env = layer_envs[0]
        this_env.env.model = self
        # Build policy, convert env into <ObstDictWrapper> class.
        super(MBCHAC, self).__init__(policy=BasePolicy, env=this_env, policy_base=BasePolicy, learning_rate=0.0)
        # Set model of env to self, so that we can perform hierarchical actions.
        self.total_num_timesteps = 0
        # we will use the policy and learning rate from the model.
        del self.policy, self.learning_rate
        if self.get_vec_normalize_env() is not None:
            assert online_sampling, "You must pass `online_sampling=True` if you want to use `VecNormalize` with `HER`"

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
        if len(sub_model_classes) > 0:
            sub_level_steps = ",".join(self.time_scales.split(",")[1:])
            self.sub_model = MBCHAC('MlpPolicy', bottom_env, sub_model_classes[0], sub_model_classes[1:],
                                    time_scales=sub_level_steps,
                                    n_sampled_goal=n_sampled_goal, goal_selection_strategy=goal_selection_strategy,
                                    online_sampling=online_sampling, max_episode_length=int(sub_level_steps[0]),
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
        # if we sample her transitions online use custom replay buffer
        self.online_sampling = online_sampling
        # compute ratio between HER replays and regular replays in percent for online HER sampling
        self.her_ratio = 1 - (1.0 / (self.n_sampled_goal + 1))
        # maximum steps in episode
        self.max_episode_length = get_time_limit(self.env, max_episode_length)
        # storage for transitions of current episode for offline sampling
        # for online sampling, it replaces the "classic" replay buffer completely
        her_buffer_size = self.buffer_size if online_sampling else self.max_episode_length
        self._episode_storage = HerReplayBuffer(
            self.env,
            her_buffer_size,
            max_episode_length,
            self.goal_selection_strategy,
            self.env.observation_space,
            self.env.action_space,
            self.device,
            self.n_envs,
            self.her_ratio,  # pytype: disable=wrong-arg-types
        )

        # counter for steps in episode
        self.episode_steps = 0

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self.model._setup_model()
        # assign episode storage to replay buffer when using online HER sampling
        if self.online_sampling:
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

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "MBCHAC",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> BaseAlgorithm:

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )
        self.model.start_time = self.start_time
        self.model.ep_info_buffer = self.ep_info_buffer
        self.model.ep_success_buffer = self.ep_success_buffer
        self.model.num_timesteps = self.num_timesteps
        self.model._episode_num = self._episode_num
        self.model._last_obs = self._last_obs
        self.model._total_timesteps = self._total_timesteps

        callback.on_training_start(locals(), globals())

        while self.total_num_timesteps < total_timesteps:

            # Equivalent to CHACpolicy.layer.train() - called on l. 120
            rollout = self.collect_rollouts(
                self.env,
                n_episodes=self.n_episodes_rollout,
                n_steps=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                log_interval=log_interval,
            )

            if rollout.continue_training is False:
                break

            # Equivalent to CHACpolicy.learn() - called on l. 126
            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts and self.replay_buffer.size() > 0 and do_train is True:
                # If no `gradient_steps` is specified,        policy: Union[str, Type[BasePolicy]],
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps > 0 else rollout.episode_timesteps
                self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

        callback.on_training_end()

        return self

    # def train(self, batch_size, gradient_steps):
    #     super().train(batch_size=batch_size, gradient_steps=gradient_steps)
    #     if self.sub_model is not None:
    #         self.sub_model.train(batch_size, gradient_steps)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        n_episodes: int = 1,
        n_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
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
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """

        episode_rewards, total_timesteps = [], []
        total_steps, total_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert env.num_envs == 1, "OffPolicyAlgorithm only support single environment"

        if self.model.use_sde:
            self.actor.reset_noise()

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
                action, buffer_action = self._sample_action(learning_starts, action_noise)

                # Perform action
                if self.sub_model is None:
                    new_obs, reward, done, infos = env.step(action)
                else:
                    new_obs, reward, done, infos = env.step(action)

                self.num_timesteps += 1
                self.model.num_timesteps = self.num_timesteps
                if hasattr(self.sub_model, 'total_num_timesteps'):
                    self.total_num_timesteps = self.sub_model.total_num_timesteps
                else:
                    self.total_num_timesteps = self.sub_model.num_timesteps
                episode_timesteps += 1
                total_steps += 1

                # Only stop training if return value is False, not when it is None.
                if callback.on_step() is False:
                    return RolloutReturn(0.0, total_steps, total_episodes, continue_training=False)

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

                if self.online_sampling:
                    self.replay_buffer.add(self._last_original_obs, next_obs, buffer_action, reward_, done, infos)
                else:
                    # concatenate observation with (desired) goal
                    flattened_obs = ObsDictWrapper.convert_dict(self._last_original_obs)
                    flattened_next_obs = ObsDictWrapper.convert_dict(next_obs)
                    # add to replay buffer
                    self.replay_buffer.add(flattened_obs, flattened_next_obs, buffer_action, reward_, done)
                    # add current transition to episode storage
                    self._episode_storage.add(self._last_original_obs, next_obs, buffer_action, reward_, done, infos)

                self._last_obs = new_obs
                self.model._last_obs = self._last_obs

                # Save the unnormalized new observation
                if self._vec_normalize_env is not None:
                    self._last_original_obs = new_obs_
                    self.model._last_original_obs = self._last_original_obs

                self.model._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

                # For DQN, check if the target network should be updated
                # and update the exploration schedule
                # For SAC/TD3, the update is done as the same time as the gradient update
                # see https://github.com/hill-a/stable-baselines/issues/900
                self.model._on_step()

                self.episode_steps += 1

                if 0 < n_steps <= total_steps:
                    break

            if done or self.episode_steps >= self.max_episode_length:
                if self.online_sampling:
                    self.replay_buffer.store_episode()
                else:
                    self._episode_storage.store_episode()
                    # sample virtual transitions and store them in replay buffer
                    self._sample_her_transitions()
                    # clear storage for current episode
                    self._episode_storage.reset()

                total_episodes += 1
                self._episode_num += 1
                self.model._episode_num = self._episode_num
                episode_rewards.append(episode_reward)
                total_timesteps.append(episode_timesteps)

                if action_noise is not None:
                    action_noise.reset()

                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()

                self.episode_steps = 0

        mean_reward = np.mean(episode_rewards) if total_episodes > 0 else 0.0

        callback.on_rollout_end()

        return RolloutReturn(mean_reward, total_steps, total_episodes, continue_training)

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
        self.model.online_sampling = self.online_sampling
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
        for key in {"model_class", "online_sampling", "max_episode_length"}:
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
            online_sampling=data["online_sampling"],
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
        Load a replay buffer from a pickle file and set environment for replay buffer (only online sampling).

        :param path: Path to the pickled replay buffer.
        :param truncate_last_trajectory: Only for online sampling.
            If set to ``True`` we assume that the last trajectory in the replay buffer was finished.
            If it is set to ``False`` we assume that we continue the same trajectory (same episode).
        """
        self.model.load_replay_buffer(path=path)

        if self.online_sampling:
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

    # def make_h_env(self, env):
