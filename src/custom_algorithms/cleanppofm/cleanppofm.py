import copy
import math
import io
import pathlib
import warnings
from collections import deque, OrderedDict
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
from gymnasium import spaces
from gymnasium import logger as gymnasium_logger
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.vec_env import VecEnv
from torch.nn import functional as F

from custom_algorithms.cleanppofm.forward_model import ProbabilisticSimpleForwardNet, \
    ProbabilisticForwardNetPositionPrediction, ProbabilisticSimpleForwardNetIncludingReward, \
    ProbabilisticForwardNetPositionPredictionIncludingReward
from custom_algorithms.cleanppofm.utils import flatten_obs, get_position_and_object_positions_of_observation, \
    get_next_observation_gridworld, reward_estimation, calculate_prediction_error, \
    get_next_position_observation_moonlander, calculate_difficulty, normalize_rewards, get_next_whole_observation, \
    get_observation_of_position_and_object_positions
from custom_algorithms.cleanppofm.agent import Agent
from custom_envs.moonlander.helper_functions import calculate_gaussian_reward
from utils.custom_buffer import CustomDictRolloutBuffer as DictRolloutBuffer
from utils.custom_buffer import CustomRolloutBuffer as RolloutBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CLEANPPOFM:
    """
    Proximal Policy Optimization algorithm (PPO) (clip version) with a forward model (FM).
    By now, only implemented for gridworld and moonlander environments!
    NOT TESTED WITH OTHER ENVIRONMENTS!
    This is a simplified one-file version of the stable-baselines3 PPO implementation.

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param env: The environment to learn from
    :param learning_rate: The learning rate
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter
    :param clip_range_vf: Clipping parameter for the value function
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    """

    def __init__(
            self,
            env: Union[GymEnv, str],
            learning_rate: float = 3e-4,
            n_steps: int = 2048,
            batch_size: int = 64,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: float = 0.2,
            clip_range_vf: Union[None, float] = None,
            normalize_advantage: bool = True,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            # up until here the same as in the stable-baselines3 implementation
            fm_parameters: dict = None,
            position_predicting: bool = False,
            reward_predicting: bool = False,
            normalized_rewards: bool = False,
            number_of_future_steps: int = 10,
            fm_trained_with_input_noise: bool = True,
            input_noise_on: bool = False,
            maximum_number_of_objects: int = 5,
            model_based: bool = False
    ):
        if fm_parameters is None:
            fm_parameters = {}
        self.num_timesteps = 0
        self.learning_rate = learning_rate
        self._last_obs = None  # type: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
        self._last_episode_starts = None  # type: Optional[np.ndarray]
        # Buffers for logging
        self.ep_info_buffer = None  # type: Optional[deque]

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.n_envs = env.num_envs
        self.env = env
        # use gymnasium logger for yellow colored logging
        gymnasium_logger.warn("This algorithm is only tested under the Gridworld and Moonlander Envs")

        if isinstance(self.action_space, spaces.Box):
            assert np.all(
                np.isfinite(np.array([self.action_space.low, self.action_space.high]))
            ), "Continuous action space must have a finite lower and upper bound"

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                    batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        # Check that `n_steps * n_envs > 1` to avoid NaN
        # when doing advantage normalization
        buffer_size = self.env.num_envs * self.n_steps
        assert buffer_size > 1 or (
            not normalize_advantage
        ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
        # Check that the rollout buffer size is a multiple of the mini-batch size
        untruncated_batches = buffer_size // batch_size
        if buffer_size % batch_size > 0:
            warnings.warn(
                f"You have specified a mini-batch size of {batch_size},"
                f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                f" after every {untruncated_batches} untruncated mini-batches,"
                f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
            )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage

        # Forward model (own implementation)
        self.fm_parameters = fm_parameters
        # (boolean) in moonlander env, you can choose if the forward model predicts the complete next observation
        # or just the next position of the agent in the observation
        self.position_predicting = position_predicting
        # boolean if the forward model should also predict the reward
        self.reward_predicting = reward_predicting
        # boolean if the rewards are normalized
        self.normalized_rewards = normalized_rewards
        # number of future steps for the reward predicting forward model
        self.number_of_future_steps = number_of_future_steps
        # boolean if the training data for the forward model is generated with input noise or not
        self.fm_trained_with_input_noise = fm_trained_with_input_noise
        # boolean if the input noise is on
        self.input_noise_on = input_noise_on
        # maximal number of objects considered in the forward model
        self.maximum_number_of_objects = maximum_number_of_objects
        # boolean if the forward model is used for selecting the next action of the RL agent
        self.model_based = model_based
        # prediction error of the prediction of the forward model and the actual next observation
        self.soc = 1

        # get the env name as described here: https://github.com/DLR-RM/stable-baselines3/blob/master/docs/guide/vec_envs.rst
        # Note: you should use vec_env.env_method("get_wrapper_attr", "attribute_name") in Gymnasium v1.0
        self.env_name = self.env.env_method("get_wrapper_attr", "name")[0]
        if not (self.env_name == "GridWorldEnv" or self.env_name == "MoonlanderWorldEnv"):
            raise NotImplementedError("This algorithm is not implemented for this environment yet!")

        # position predicting only possible for moonlander env
        if self.position_predicting and not self.env_name == "MoonlanderWorldEnv":
            raise NotImplementedError("Position predicting is only possible for the moonlander environment by now!")
        if self.position_predicting:
            fm_cls = ProbabilisticForwardNetPositionPredictionIncludingReward if self.reward_predicting else \
                ProbabilisticForwardNetPositionPrediction
        else:
            fm_cls = ProbabilisticSimpleForwardNetIncludingReward if self.reward_predicting else ProbabilisticSimpleForwardNet
        if not fm_cls == ProbabilisticForwardNetPositionPredictionIncludingReward:
            self.fm_network = fm_cls(self.env, self.fm_parameters).to(device)
        else:
            self.fm_network = fm_cls(self.env, self.fm_parameters, self.maximum_number_of_objects).to(device)
        self.fm_optimizer = torch.optim.Adam(
            self.fm_network.parameters(),
            # FIXME
            # lr=self.fm_parameters["learning_rate"]
            lr=0.001
        )
        self.fm_best_loss = math.inf
        self.best_model = copy.deepcopy(self.fm_network.state_dict())
        self.logger = None

        self._setup_model()

    def _setup_model(self) -> None:
        buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, spaces.Dict) else RolloutBuffer

        self.rollout_buffer = buffer_cls(
            buffer_size=self.n_steps,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = Agent(env=self.env, reward_predicting=self.reward_predicting, model_based=self.model_based).to(
            device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate, eps=1e-5)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        This implementation is mostly from the stable-baselines3 PPO implementation.
        """
        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            #####
            # PROBLEM: rollout_buffer.get(self.batch_size) returns a random sample of the buffer,
            # it is not ordered anymore -> solution: own implementation of the buffer
            #####

            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                #####
                observations = rollout_data.observations

                # our custom implementation of the buffer includes the next observation for training the forward model
                next_observations = rollout_data.next_observations
                rewards = rollout_data.rewards

                self.train_fm(observations, next_observations, actions, rewards)
                #####

                _, log_prob, entropy, values, _ = self.policy.get_action_and_value_and_forward_model_prediction(
                    fm_network=self.fm_network,
                    obs=observations,
                    action=actions, logger=self.logger,
                    position_predicting=self.position_predicting,
                    maximum_number_of_objects=self.maximum_number_of_objects)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -self.clip_range_vf, self.clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        var_y = np.var(self.rollout_buffer.values.flatten())
        explained_var = np.nan if var_y == 0 else (
                1 - np.var(self.rollout_buffer.returns.flatten() - self.rollout_buffer.values.flatten()) / var_y)

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1
    ):
        """
        This implementation is mostly from the stable-baselines3 PPO implementation.
        No changes were made by me.
        """
        iteration = 0
        print("resetting env")
        self._last_obs = self.env.reset()
        callback.init_callback(self)
        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer)

            if continue_training is False:
                break

            iteration += 1

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean",
                                       np.mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean",
                                       np.mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            rollout_buffer: RolloutBuffer,
    ) -> bool:
        """
        This implementation is mostly from the stable-baselines3 PPO implementation.
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :return: True if function returned with at least `self.n_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"

        n_steps = 0
        elements_in_rollout_buffer = 0
        rollout_buffer.reset()

        callback.on_rollout_start()

        while elements_in_rollout_buffer < self.n_steps:
            with torch.no_grad():
                # get action and forward model prediction
                actions, log_probs, _, values, forward_normal = self.policy.get_action_and_value_and_forward_model_prediction(
                    fm_network=self.fm_network,
                    obs=self._last_obs, logger=self.logger,
                    position_predicting=self.position_predicting,
                    maximum_number_of_objects=self.maximum_number_of_objects)

            # log logarithmic probability of action distribution (one value in tensor)
            log_prob_float = float(np.mean(log_probs.cpu().numpy()))
            self.logger.record("train/rollout_logprob_step", float(log_prob_float))
            self.logger.record_mean("train/rollout_logprob_mean", float(log_prob_float))
            # Rescale and perform action
            actions = actions.cpu().numpy()
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
            elif isinstance(self.action_space, spaces.Discrete):
                clipped_actions = actions[0]

            new_obs, rewards, dones, infos, prediction_error, difficulty, soc, reward_with_future_reward_estimation_corrective, _, _ = self.step_in_env(
                actions=clipped_actions, forward_normal=forward_normal)

            # FIXME: is it possible that multiple actions are taken here?
            ##### LOGGING #####
            if self.reward_predicting:
                self.logger.record("train/predicted_rewards", float(forward_normal.mean[:, -1].mean()))
                self.logger.record_mean("train/predicted_rewards_mean", float(forward_normal.mean[:, -1].mean()))
                self.logger.record("train/reward_with_future_reward_estimation_corrective",
                                   reward_with_future_reward_estimation_corrective.mean())
                self.logger.record_mean("train/reward_with_future_reward_estimation_corrective_mean",
                                        reward_with_future_reward_estimation_corrective.mean())
            self.logger.record("train/prediction_error", prediction_error)
            self.logger.record_mean("train/prediction_error_mean", prediction_error)
            self.logger.record("train/difficulty", difficulty)
            self.logger.record_mean("train/difficulty_mean", difficulty)
            self.logger.record("train/soc", soc)
            self.logger.record_mean("train/soc_mean", soc)
            self.logger.record("train/rollout_rewards_step", float(rewards.mean()))
            self.logger.record_mean("train/rollout_rewards_mean", float(rewards.mean()))

            # this is only logged when no hyperparameter tuning is running?
            # dodge/collect env
            if "simple" in infos[0].keys():
                self.logger.record("rollout_reward_simple", float(infos[0]["simple"]))
                self.logger.record("rollout_reward_gaussian", float(infos[0]["gaussian"]))
                self.logger.record("rollout_reward_pos_neg",
                                   float(infos[0]["pos_neg"]["pos"][0] + infos[0]["pos_neg"]["neg"][0]))
                self.logger.record("rollout_number_of_crashed_or_collected_objects",
                                   float(infos[0]["number_of_crashed_or_collected_objects"]))
            # gridworld env
            if "self.input_noise_is_applied_in_this_episode" in infos[0].keys():
                self.logger.record("input_noise_applied", infos[0]["self.input_noise_is_applied_in_this_episode"])
            # meta env
            if "dodge" in infos[0].keys():
                self.logger.record("dodge_gaussian_reward", infos[0]["dodge"]["gaussian"])
                self.logger.record("collect_gaussian_reward", infos[0]["collect"]["gaussian"])
                self.logger.record("task_switching_costs", infos[0]["task_switching_costs"])

            # stable baselines 3 implementation
            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            # FIXME: go here to get actual terminal observation
            temporary_new_obs = new_obs
            for idx, done in enumerate(dones):
                if done and infos[idx].get("terminal_observation") is not None:
                    # fixme: what about multiple elements in the list?
                    if self.env_name == "GridWorldEnv":
                        temporary_new_obs = OrderedDict(infos[idx]["terminal_observation"])
                    elif self.env_name == "MoonlanderWorldEnv":
                        temporary_new_obs = infos[idx]["terminal_observation"]

                    # TimeLimit.truncated = truncated and not terminated --> when episode is done because of time limit (steps)
                    if infos[idx].get("TimeLimit.truncated", False):
                        terminal_obs = infos[idx]["terminal_observation"]
                        with torch.no_grad():
                            # FIXME: why do we need to expand the dimensions?
                            if "agent" in terminal_obs and "target" in terminal_obs:
                                terminal_obs["agent"] = np.expand_dims(terminal_obs["agent"], axis=0)
                                terminal_obs["target"] = np.expand_dims(terminal_obs["target"], axis=0)
                            else:
                                terminal_obs = np.expand_dims(terminal_obs, axis=0)
                            terminal_value = self.policy.get_value(terminal_obs)[0]
                        rewards[idx] += self.gamma * terminal_value

            # this is needed because otherwise the last observation and action does not match the new observation
            # FIXME: but maybe this removes one step? but otherwise the forward model does not learn well because
            # it does not understand that a new episode has started and therefore a new observation is given
            if not infos[0]["TimeLimit.truncated"]:
                rollout_buffer.add(
                    obs=self._last_obs,
                    next_obs=temporary_new_obs,
                    action=actions,
                    reward=rewards,
                    reward_with_future_reward_estimation_corrective=reward_with_future_reward_estimation_corrective,
                    episode_start=self._last_episode_starts,
                    value=values,
                    log_prob=log_probs,
                )
                elements_in_rollout_buffer += 1
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with torch.no_grad():
            if self.model_based:
                # fixme: this is hardcoded, it does not work when we have a non-deterministic forward model because we calculate the next obs at two places
                observation_height = self.env.env_method("get_wrapper_attr", "observation_height")[0]
                observation_width = self.env.env_method("get_wrapper_attr", "observation_width")[0]
                agent_size = self.env.env_method("get_wrapper_attr", "size")[0]
                task = self.env.env_method("get_wrapper_attr", "task")[0]
                comb_obs = torch.tensor(new_obs).clone().detach()
                comb_obj_positions = get_position_and_object_positions_of_observation(comb_obs,
                                                                                      maximum_number_of_objects=self.maximum_number_of_objects,
                                                                                      observation_width=observation_width,
                                                                                      observation_height=observation_height,
                                                                                      agent_size=agent_size)

                # Convert comb_obj_positions to a tensor
                cop_tensor = torch.tensor(comb_obj_positions).float()

                obs_after_every_action = comb_obs.clone().detach()
                rewards_for_every_action = {}
                if task == "dodge":
                    task_type = "obstacle"
                elif task == "collect":
                    task_type = "coin"

                for i in range(0, env.action_space.n):
                    # Fixme: for hardcoded next obs, we had to change the ordering
                    current_action = torch.full((cop_tensor.shape[0], 1), i)
                    current_action_hardcoded = torch.full((1, cop_tensor.shape[0]), i)

                    # FIXME: hardcoded next obs
                    next_positions = get_next_position_observation_moonlander(
                        observations=cop_tensor,
                        actions=current_action_hardcoded[0],
                        observation_width=observation_width,
                        observation_height=observation_height,
                        agent_size=agent_size,
                        maximum_number_of_objects=self.maximum_number_of_objects)

                    obs_after_action = get_observation_of_position_and_object_positions(
                        # agent_and_object_positions=nra_mean,
                        agent_and_object_positions=next_positions,
                        observation_height=observation_height,
                        observation_width=observation_width,
                        agent_size=agent_size, task=task)

                    # calculate reward for new obs
                    x_position_of_agent = int(
                        min(max(agent_size, next_positions[0][0]), observation_width - agent_size + 1))
                    y_position_of_agent = int(next_positions[0][1])

                    collected_objects = []
                    for index in range(2, len(next_positions[0]), 2):
                        if not (next_positions[0][index] == 0 and next_positions[0][index + 1] == 0):

                            if (
                                    (
                                            ((next_positions[0][index] - 1) == (x_position_of_agent - 1))
                                            or ((next_positions[0][index] - 1) == x_position_of_agent)
                                            or ((next_positions[0][index] - 1) == (x_position_of_agent + 1))
                                            or (next_positions[0][index] == (x_position_of_agent - 1))
                                            or (next_positions[0][index] == x_position_of_agent)
                                            or (next_positions[0][index] == (x_position_of_agent + 1))
                                            or ((next_positions[0][index] + 1) == (x_position_of_agent - 1))
                                            or ((next_positions[0][index] + 1) == x_position_of_agent)
                                            or ((next_positions[0][index] + 1) == (x_position_of_agent + 1))
                                    )
                                    and
                                    (
                                            ((next_positions[0][index + 1] - 1) == (y_position_of_agent - 1))
                                            or ((next_positions[0][index + 1] - 1) == y_position_of_agent)
                                            or ((next_positions[0][index + 1] - 1) == (y_position_of_agent + 1))
                                            or (next_positions[0][index + 1] == (y_position_of_agent - 1))
                                            or (next_positions[0][index + 1] == y_position_of_agent)
                                            or (next_positions[0][index + 1] == (y_position_of_agent + 1))
                                            or ((next_positions[0][index + 1] + 1) == (y_position_of_agent - 1))
                                            or ((next_positions[0][index + 1] + 1) == y_position_of_agent)
                                            or ((next_positions[0][index + 1] + 1) == (y_position_of_agent + 1))
                                    )
                            ):
                                collected_objects.append(
                                    {'x': int(next_positions[0][index]),
                                     'y': int(next_positions[0][index + 1]),
                                     'size': agent_size})

                    rewards_for_every_action[i] = [calculate_gaussian_reward(
                        state=np.array(row).reshape(observation_height, observation_width + 2),
                        collected_objects=collected_objects,
                        agent_size=agent_size,
                        task_type=task_type,
                        current_reward_function="gaussian",
                        x_position_of_agent=x_position_of_agent,
                        y_position_of_agent=y_position_of_agent)[0] for row in obs_after_action]

                    obs_after_every_action = torch.cat(
                        (obs_after_every_action.to(device=device), obs_after_action.to(device=device)), dim=1)
                for i in range(0, env.action_space.n):
                    rewards_for_every_action_tensor = torch.tensor(rewards_for_every_action[i]).reshape(
                        obs_after_every_action.shape[0],
                        1)
                    obs_after_every_action = torch.cat((obs_after_every_action, rewards_for_every_action_tensor), dim=1)
                new_obs = obs_after_every_action.to(device=device)

            # Compute value for the last timestep
            values = self.policy.get_value(new_obs)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train_fm(self, observations: torch.Tensor, next_observations: torch.Tensor, actions: torch.Tensor,
                 rewards: torch.Tensor) -> None:
        """
        Train the forward model with the actual next observations and rewards
        Args:
            observations: observations
            next_observations: actual next observations
            actions: action taken in observations leading to next_observations
            rewards: actual rewards

        Returns:

        """
        if self.policy.flatten:
            observations = flatten_obs(observations)
            next_observations = flatten_obs(next_observations)

        observation_height = self.env.env_method("get_wrapper_attr", "observation_height")[0]
        observation_width = self.env.env_method("get_wrapper_attr", "observation_width")[0]
        agent_size = self.env.env_method("get_wrapper_attr", "size")[0]

        ##### FORMAT OBSERVATION FOR FORWARD MODEL #####
        # 1. with or without reward (line 572)
        # 2. with or without position predicting (moonlander)
        # 3. with or without input noise

        # gridworld or moonlander without position predicting
        if not self.position_predicting:
            next_observations_duplicated = copy.deepcopy(next_observations)

            # create next observations without input noise
            if self.env_name == "MoonlanderWorldEnv" and not self.fm_trained_with_input_noise:
                next_observations_duplicated = get_next_whole_observation(next_observations=next_observations,
                                                                          actions=actions,
                                                                          observation_width=observation_width,
                                                                          observation_height=observation_height)
            elif self.env_name == "GridWorldEnv" and not self.fm_trained_with_input_noise:
                next_observations_duplicated = get_next_observation_gridworld(observations=observations,
                                                                              actions=actions)

            next_observations_formatted = next_observations_duplicated if not self.reward_predicting else torch.cat(
                (next_observations, rewards), dim=1)
        # position predicting only for moonlander
        else:
            # get position out of observation
            observations = get_position_and_object_positions_of_observation(observations,
                                                                            maximum_number_of_objects=self.maximum_number_of_objects,
                                                                            observation_width=observation_width,
                                                                            observation_height=observation_height,
                                                                            agent_size=agent_size)
            if not self.fm_trained_with_input_noise:
                next_observations_formatted = get_next_position_observation_moonlander(observations=observations,
                                                                                       actions=actions,
                                                                                       observation_width=observation_width,
                                                                                       observation_height=observation_height,
                                                                                       agent_size=agent_size,
                                                                                       maximum_number_of_objects=self.maximum_number_of_objects)
            else:
                next_observations_formatted = get_position_and_object_positions_of_observation(next_observations,
                                                                                               maximum_number_of_objects=self.maximum_number_of_objects,
                                                                                               observation_width=observation_width,
                                                                                               observation_height=observation_height,
                                                                                               agent_size=agent_size)
            if self.reward_predicting:
                next_observations_formatted = torch.cat((next_observations_formatted, rewards), dim=1)

        ##### FORWARD MODEL TRAINING #####
        # forward model prediction
        forward_model_prediction_normal_distribution = self.fm_network(observations, actions.float().unsqueeze(1))
        # log probs is the logarithm of the maximum likelihood
        # log because the deviation is easier (addition instead of multiplication)
        # negative because likelihood normally maximizes
        fw_loss = -forward_model_prediction_normal_distribution.log_prob(next_observations_formatted)
        loss = fw_loss.mean()

        # Track best performance, and save the model's state
        if loss < self.fm_best_loss:
            self.fm_best_loss = loss
            model_path = 'best_model'
            torch.save(self.fm_network.state_dict(), model_path)
            self.best_model = copy.deepcopy(self.fm_network.state_dict())

        self.logger.record("fm/fw_loss", loss.item())
        self.fm_optimizer.zero_grad()
        loss.backward()
        self.fm_optimizer.step()

    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]], torch.distributions.Normal]:
        """
        From the stable-baselines3 PPO implementation.
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).
        Args:
            observation: the input observation
            state: The last hidden states (can be None, used in recurrent policies)
            episode_start: The last masks (can be None, used in recurrent policies)
                this corresponds to beginning of episodes, where the hidden states of the RNN must be reset.
            deterministic: Whether or not to return deterministic actions.

        Returns:
            the model's action and the next hidden state (used in recurrent policies)
        """
        with torch.no_grad():
            action, _, _, _, forward_model_prediction_normal_distribution = self.policy.get_action_and_value_and_forward_model_prediction(
                fm_network=self.fm_network,
                obs=observation,
                deterministic=deterministic,
                logger=self.logger,
                position_predicting=self.position_predicting, maximum_number_of_objects=self.maximum_number_of_objects)
        return action.cpu().numpy(), state, forward_model_prediction_normal_distribution

    def step_in_env(self, actions, forward_normal, use_reward_of_env: bool = False,
                    # for the moment only for meta env
                    use_prediction_error: bool = True, use_difficulty: bool = True) -> tuple[
        np.ndarray, float, bool, dict, float, float, float, float, int, float]:
        """
        Step in the environment with the given actions and the forward model prediction.
        This includes the displaying of the forward model prediction and the calculation of the prediction error.
        The reward is modified through the prediction error
        Args:
            actions: action to take in the environment
            forward_normal: prediction of the forward model (normal distribution)
            use_reward_of_env: if the reward of the environment should be used to calculate the reward estimation or from the forward model prediction
            use_prediction_error: if the prediction error should be used to calculate the SoC
            use_difficulty: if the difficulty should be used to calculate the SoC

        Returns:
            new_obs: new observation
            rewards: rewards
            dones: if the episode is done
            infos: additional information
            prediction_error: calculated prediction error
            difficulty: calculated difficulty
            soc: calculated sense of control
            reward_with_future_reward_estimation_corrective: reward corrected by prediction error
            input_noise: applied input noise
            rewards_normalized: normalized rewards
        """
        ##### DISPLAYING THE FORWARD MODEL PREDICTION #####
        # modify the env attributes as described here:
        # https://github.com/DLR-RM/stable-baselines3/blob/master/docs/guide/vec_envs.rst
        if not self.reward_predicting:
            # FIXME
            # this does not work
            # self.env.set_attr("forward_model_prediction", forward_normal.mean.cpu())
            # but this throws a warning?
            # same below for input noise
            self.env.env_method("set_forward_model_prediction", forward_normal.mean.cpu())
        # remove reward because it is not needed to display the predicted observation
        else:
            # FIXME
            # this does not work
            # self.env.set_attr("forward_model_prediction", forward_normal.mean[0][:-1].cpu().unsqueeze(0))
            # but this throws a warning?
            # same below for input noise
            self.env.env_method("set_forward_model_prediction", forward_normal.mean[0][:-1].cpu().unsqueeze(0))

        # if input noise is applied!
        input_noise = 0
        if self.input_noise_on and not actions[0] == 1:
            mu = 0
            sigma = 1.5
            input_noise = np.random.normal(loc=mu, scale=sigma)
            input_noise = int(round(input_noise, 0))
        self.env.env_method("set_input_noise", input_noise)

        ##### STEP IN ENVIRONMENT #####
        # dones = terminated or truncated
        new_obs, rewards, dones, infos = self.env.step(actions)

        prediction_error = 0
        difficulty = 0
        if use_prediction_error:
            ##### CALCULATING PREDICTION ERROR #####
            prediction_error = calculate_prediction_error(env_name=self.env_name, env=self.env,
                                                          next_obs=torch.tensor(new_obs, device=device),
                                                          forward_model_prediction_normal_distribution=forward_normal,
                                                          maximum_number_of_objects=self.maximum_number_of_objects)
        if use_difficulty:
            ##### CALCULATING DIFFICULTY #####
            difficulty, summed_up_rewards_default = calculate_difficulty(env=self.env, policy=self.policy,
                                                                         fm_network=self.fm_network,
                                                                         logger=self.logger, env_name=self.env_name,
                                                                         prediction_error=prediction_error,
                                                                         position_predicting=self.position_predicting,
                                                                         maximum_number_of_objects=self.maximum_number_of_objects,
                                                                         reward_predicting=self.reward_predicting,
                                                                         use_reward_of_env=use_reward_of_env)

        ##### CALCULATING SOC #####
        # prediction error is high, if the prediction and actual observation do not match
        # difficulty is high if the rewards of the optimal trajectory are quite different to the rewards of the default trajectory
        # soc = mean of prediction error and difficulty
        self.soc = 1 - ((prediction_error + difficulty) / 2)

        task = self.env.env_method("get_wrapper_attr", "task")[0]
        # normalize actual reward
        rewards_normalized = normalize_rewards(task=task, absolute_reward=rewards)
        # add normalized reward to summed up rewards + normalize by mean
        # FIXME: changed this!!!
        # summed_up_rewards_default = (rewards_normalized + summed_up_rewards_default) / 2

        if self.normalized_rewards:
            rewards = rewards_normalized

        ##### CALCULATE REWARD ESTIMATION FOR DEFAULT TRAJECTORY CORRECTED BY SOC #####
        # FIXME: changed this!!!
        # reward_estimation = (summed_up_rewards_default + self.soc) / 2
        reward_estimation = (rewards_normalized + self.soc) / 2

        # input noise only for debugging
        return new_obs, rewards, dones, infos, prediction_error, difficulty, self.soc, reward_estimation, input_noise, rewards_normalized

    def save(
            self,
            path: Union[str, pathlib.Path, io.BufferedIOBase],
    ) -> None:
        # Copy parameter list, so we don't mutate the original dict
        data = self.__dict__.copy()
        for to_exclude in ["logger", "env", "num_timesteps", "policy",
                           "_last_obs", "_last_episode_starts"]:
            del data[to_exclude]
        # save network parameters
        data["_policy"] = self.policy.state_dict()
        print("save", self.policy.state_dict())
        # changed to save the best forward model
        data["_fm"] = self.best_model
        torch.save(data, path)

    @classmethod
    def load(cls, path, env, **kwargs):
        model = cls(env=env, **kwargs)
        loaded_dict = torch.load(path, map_location=torch.device(device))
        for k in loaded_dict:
            if k not in ["_policy", "_fm"]:
                model.__dict__[k] = loaded_dict[k]
        # FIXME: help needed to update agent again to modelbased observation shape
        model._setup_model()
        # load network states
        model.policy.load_state_dict(loaded_dict["_policy"])
        model.fm_network.load_state_dict(loaded_dict["_fm"])
        return model

    def set_logger(self, logger):
        self.logger = logger

    def get_env(self) -> VecEnv:
        return self.env
