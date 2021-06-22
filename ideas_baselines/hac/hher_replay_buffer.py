from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from ideas_baselines.common.type_aliases import ReplayBufferSamplesWithTestTrans
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper
from ideas_baselines.hac.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common import logger

class HHerReplayBuffer(ReplayBuffer):
    """
    Replay buffer for sampling Hierarchical HER (Hindsight Experience Replay) transitions, as described in the HAC paper by Levy et al. 2019.
    In the online sampling case, these new transitions will not be saved in the replay buffer
    and will only be created at sampling time.

    :param env: The training environment
    :param buffer_size: The size of the buffer measured in transitions.
    :param max_episode_length: The length of an episode. (time horizon)
    :param goal_selection_strategy: Strategy for sampling goals for replay.
        One of ['episode', 'final', 'future']
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :her_ratio: The ratio between HER transitions and regular transitions in percent
        (between 0 and 1, for online sampling)
        The default value ``her_ratio=0.8`` corresponds to 4 virtual transitions
        for one real transition (4 / (4 + 1) = 0.8)
    :hindsight_sampling_done_if_success: This determines whether an episodes is considered as done if it is successful *in hindsight*. This is important e.g. in SAC when computing the q-value because the discounted future return is set to 0 when an episode is done.
    """

    def __init__(
        self,
        env: ObsDictWrapper,
        buffer_size: int,
        max_episode_length: int,
        goal_selection_strategy: GoalSelectionStrategy,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        her_ratio: float = 0.8,
        perform_action_replay_transitions = True,
        test_trans_sampling_fraction = 0.1,
        subgoal_test_fail_penalty = 1,
        hindsight_sampling_done_if_success = True,
    ):
        """

        Args:
            env:
            buffer_size:
            max_episode_length:
            goal_selection_strategy:
            observation_space:
            action_space:
            device:
            n_envs:
            her_ratio:
            perform_action_replay_transitions:
            test_trans_sampling_fraction:
            subgoal_test_fail_penalty:
            hindsight_sampling_done_if_success:
            set_dones_one: Whether to set all dones for all steps to one. This is used for disabling setting the discounted future reward to 0 if using SAC.
        """

        super(HHerReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs)
        self.hindsight_sampling_done_if_success = hindsight_sampling_done_if_success
        self.env = env
        self.buffer_size = buffer_size
        self.max_episode_length = max_episode_length

        # buffer with episodes
        # number of episodes which can be stored until buffer size is reached
        self.max_episode_stored = self.buffer_size // self.max_episode_length

        self.episode_lengths = np.zeros(self.max_episode_stored, dtype=np.int64)

        self.goal_selection_strategy = goal_selection_strategy
        # percentage of her indices
        self.her_ratio = her_ratio

        self.perform_action_replay_transitions = perform_action_replay_transitions

        self.test_trans_sampling_fraction = test_trans_sampling_fraction

        self.subgoal_test_fail_penalty = max(subgoal_test_fail_penalty, 1)

        # input dimensions for buffer initialization
        self.input_shape = {
            "observation": (self.env.num_envs, self.env.obs_dim),
            "achieved_goal": (self.env.num_envs, self.env.goal_dim),
            "desired_goal": (self.env.num_envs, self.env.goal_dim),
            "action": (self.action_dim,),
            "reward": (1,),
            "next_obs": (self.env.num_envs, self.env.obs_dim),
            "next_achieved_goal": (self.env.num_envs, self.env.goal_dim),
            "next_desired_goal": (self.env.num_envs, self.env.goal_dim),
            "done": (1,),
            "is_subgoal_testing_trans": (1,),
        }

        self.reset()

    def __getstate__(self) -> Dict[str, Any]:
        """
        Gets state for pickling.

        Excludes self.env, as in general Env's may not be pickleable."""
        state = self.__dict__.copy()
        # these attributes are not pickleable
        del state["env"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Restores pickled state.

        User must call ``set_env()`` after unpickling before using.

        :param state:
        """
        self.__dict__.update(state)
        assert "env" not in state
        self.env = None

    def set_env(self, env: ObsDictWrapper) -> None:
        """
        Sets the environment.

        :param env:
        """
        if self.env is not None:
            raise ValueError("Trying to set env of already initialized environment.")

        self.env = env

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> Union[ReplayBufferSamplesWithTestTrans, RolloutBufferSamples]:
        """
        Abstract method from base class.
        """
        raise NotImplementedError()

    def sample(
        self,
        batch_size: int,
        env: Optional[VecNormalize],
    ) -> Union[ReplayBufferSamplesWithTestTrans, Tuple[np.ndarray, ...]]:
        """
        Sample function for online sampling of HER transition,
        this replaces the "regular" replay buffer ``sample()``
        method in the ``train()`` function.
        :param batch_size: Number of element to sample
        :param env: Associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return: Samples.
        """
        return self._sample_transitions(batch_size, maybe_vec_env=env)

    def sample_goals(
        self,
        episode_indices: np.ndarray,
        her_indices: np.ndarray,
        transitions_indices: np.ndarray,
    ) -> np.ndarray:
        """
        Sample goals based on goal_selection_strategy.
        This is a vectorized (fast) version.

        :param episode_indices: Episode indices to use.
        :param her_indices: HER indices.
        :param transitions_indices: Transition indices to use.
        :return: Return sampled goals.
        """
        her_episode_indices = episode_indices[her_indices]

        if self.goal_selection_strategy == GoalSelectionStrategy.FINAL:
            # replay with final state of current episode
            transitions_indices = self.episode_lengths[her_episode_indices] - 1
            goals = self.buffer["achieved_goal"][her_episode_indices, transitions_indices]

        elif self.goal_selection_strategy == GoalSelectionStrategy.FUTURE2:
            # replay with random state which comes from the same episode and was observed after current transition
            transitions_indices = np.random.randint(
                transitions_indices[her_indices] + 1, self.episode_lengths[her_episode_indices]
            )
            goals = self.buffer["achieved_goal"][her_episode_indices, transitions_indices]

        elif self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            # replay with random state which comes from the same episode and was observed after current transition
            transitions_indices = np.random.randint(
                transitions_indices[her_indices], self.episode_lengths[her_episode_indices]
            )
            goals = self.buffer["next_achieved_goal"][her_episode_indices, transitions_indices]

        elif self.goal_selection_strategy == GoalSelectionStrategy.EPISODE:
            # replay with random state which comes from the same episode as current transition
            transitions_indices = np.random.randint(self.episode_lengths[her_episode_indices])
            goals = self.buffer["achieved_goal"][her_episode_indices, transitions_indices]

        elif self.goal_selection_strategy == GoalSelectionStrategy.RNDEND2:
            # replay with random state which comes from the same episode and was observed after current transition
            # This distribution is such that if the length of an episode is max, then we have a uniform distribution, but the shorter it gets the more we sample from the last sample.

            n_final_sample_prob = (self.max_episode_length - self.episode_lengths[her_episode_indices]) / self.max_episode_length
            rnd_sample = np.random.random_sample(n_final_sample_prob.shape)
            sample_final_idxs = np.where(rnd_sample < n_final_sample_prob)
            # sample_ps = np.zeros_like(self.buffer["achieved_goal"][her_episode_indices])
            # sample_ps
            # sample_ranges = np.arange(transitions_indices[her_indices] + 1, self.episode_lengths[her_episode_indices])
            # all_transition_idxs =
            # transitions_indices = np.random.choice(all_transition_idxs, p=sample_ps)
            transitions_indices = np.random.randint(
                transitions_indices[her_indices] + 1, self.episode_lengths[her_episode_indices]
            )
            replace_trans_idxs = self.episode_lengths[her_episode_indices][sample_final_idxs] - 1
            transitions_indices[sample_final_idxs] = replace_trans_idxs
            goals = self.buffer["achieved_goal"][her_episode_indices, transitions_indices]

        elif self.goal_selection_strategy == GoalSelectionStrategy.RNDEND: # Here we do also select the last transition but select the next_achieved_goal instead.
            # replay with random state which comes from the same episode and was observed after current transition
            # This distribution is such that if the length of an episode is max, then we have a uniform distribution, but the shorter it gets the more we sample from the last sample.

            # First sample future transition indices just like in future.
            transitions_indices = np.random.randint(
                transitions_indices[her_indices], self.episode_lengths[her_episode_indices]
            )

            # Then determine the episodes where to sample from the end.
            n_final_sample_prob = (self.max_episode_length - self.episode_lengths[her_episode_indices]) / self.max_episode_length
            rnd_sample = np.random.random_sample(n_final_sample_prob.shape)
            sample_final_idxs = np.where(rnd_sample < n_final_sample_prob)
            replace_trans_idxs = self.episode_lengths[her_episode_indices][sample_final_idxs] - 1
            # Finally update those transition indices where to sample from the end.
            transitions_indices[sample_final_idxs] = replace_trans_idxs
            goals = self.buffer["next_achieved_goal"][her_episode_indices, transitions_indices]

        else:
            raise ValueError(f"Strategy {self.goal_selection_strategy} for sampling goals not supported!")

        return goals

    def _sample_transitions(self,
        batch_size: Optional[int],
        maybe_vec_env: Optional[VecNormalize],
    ) -> Union[ReplayBufferSamplesWithTestTrans, Tuple[np.ndarray, ...]]:

        # Make sure that testing transitions exist in replay buffer.
        tt_sample_frac = self.test_trans_sampling_fraction
        if len(np.argwhere(self.buffer['is_subgoal_testing_trans'] == 1)) == 0:
            tt_sample_frac = 0

        n_ga_trans = int(batch_size * (1 - tt_sample_frac))
        n_test_trans = batch_size - n_ga_trans

        replay_trans = self.sample_goal_action_replay_transitions(batch_size=n_ga_trans, maybe_vec_env=maybe_vec_env)

        if n_test_trans > 0:
            test_replay_trans = self.sample_test_transitions(batch_size=n_test_trans, maybe_vec_env=maybe_vec_env)
            tmp_list = []
            for key in replay_trans._fields:

                if key == 'is_test_trans':
                    rt_t = replay_trans._asdict()[key] * 0 # Set the testing transitions flag of the goal replay transitions to 0 when sampling because these are treated as normal transitions when computing the q value.
                else:
                    rt_t = replay_trans._asdict()[key]

                trt_t = test_replay_trans._asdict()[key]
                concat_t = th.cat((rt_t, trt_t), 0)
                tmp_list.append(concat_t)

            replay_trans = ReplayBufferSamplesWithTestTrans(*tuple(tmp_list))

        return replay_trans

    def sample_test_transitions(
        self,
        batch_size: Optional[int],
        maybe_vec_env: Optional[VecNormalize],
    ) -> Union[ReplayBufferSamplesWithTestTrans, Tuple[np.ndarray, ...]]:
        """
        :param batch_size: Number of element to sample (only used for online sampling)
        :param env: associated gym VecEnv to normalize the observations/rewards
            Only valid when using online sampling
        :param n_sampled_goal: Number of sampled goals for replay. (offline sampling)
        :return: Samples.
        """
        assert batch_size is not None, "No batch_size specified for online sampling of HER transitions"
        episode_trans_test_indices = self.buffer['is_subgoal_testing_trans']
        all_testing_indices = np.argwhere(episode_trans_test_indices == 1)
        testing_indices_indices = np.random.randint(0, len(all_testing_indices), batch_size)

        testing_indices = all_testing_indices[testing_indices_indices]
        ti_reshaped = list(np.transpose(testing_indices)[:2])
        transitions = {key: self.buffer[key][tuple(ti_reshaped)].copy() for key in self.buffer.keys()}

        # Convert info buffer to numpy array
        successes = np.array(
            [
                self.info_buffer[episode_idx][transition_idx][0]['is_success']
                # for episode_idx, transition_idx in zip(ti_reshaped[0][0], ti_reshaped[0][1])
                for episode_idx, transition_idx, _ in testing_indices
            ]
        )

        rewards = successes * self.subgoal_test_fail_penalty
        rewards -= self.subgoal_test_fail_penalty
        transitions["reward"][:, 0] = rewards

        # concatenate observation with (desired) goal
        observations = ObsDictWrapper.convert_dict(self._normalize_obs(transitions, maybe_vec_env))
        # HACK to make normalize obs work with the next observation
        transitions["observation"] = transitions["next_obs"]
        next_observations = ObsDictWrapper.convert_dict(self._normalize_obs(transitions, maybe_vec_env))

        data = (
            observations[:, 0],
            transitions["action"],
            next_observations[:, 0],
            transitions["done"],
            self._normalize_reward(transitions["reward"], maybe_vec_env),
            transitions['is_subgoal_testing_trans'],
        )
        return ReplayBufferSamplesWithTestTrans(*tuple(map(self.to_torch, data)))

    def sample_goal_action_replay_transitions(
        self,
        batch_size: Optional[int],
        maybe_vec_env: Optional[VecNormalize],
    ) -> Union[ReplayBufferSamplesWithTestTrans, Tuple[np.ndarray, ...]]:
        """
        :param batch_size: Number of element to sample (only used for online sampling)
        :param env: associated gym VecEnv to normalize the observations/rewards
            Only valid when using online sampling
        :param n_sampled_goal: Number of sampled goals for replay. (offline sampling)
        :return: Samples.
        """
        assert batch_size is not None, "No batch_size specified for online sampling of HER transitions"
        episode_indices = np.random.randint(0, self.n_episodes_stored, batch_size)
        # A subset of the transitions will be relabeled using HER algorithm
        her_indices = np.arange(batch_size)[: int(self.her_ratio * batch_size)]

        ep_lengths = self.episode_lengths[episode_indices]

        # Special case when using the "future" or "rndend" goal sampling strategy
        # we cannot sample all transitions, we have to remove the last timestep
        if self.goal_selection_strategy in [GoalSelectionStrategy.FUTURE2, GoalSelectionStrategy.RNDEND2]:
            # restrict the sampling domain when ep_lengths > 1
            # otherwise filter out the indices
            her_indices = her_indices[ep_lengths[her_indices] > 1]
            ep_lengths[her_indices] -= 1

        transitions_indices = np.random.randint(ep_lengths)
        # get selected transitions
        transitions = {key: self.buffer[key][episode_indices, transitions_indices].copy() for key in self.buffer.keys()}

        # sample new desired goals and relabel the transitions
        new_goals = self.sample_goals(episode_indices, her_indices, transitions_indices)
        transitions["desired_goal"][her_indices] = new_goals

        # Convert info buffer to numpy array
        transitions["info"] = np.array(
            [
                self.info_buffer[episode_idx][transition_idx]
                for episode_idx, transition_idx in zip(episode_indices, transitions_indices)
            ]
        )

        # Vectorized computation of the new reward
        transitions["reward"][her_indices, 0] = self.env.env_method(
            "compute_reward",
            # the new state depends on the previous state and action
            # s_{t+1} = f(s_t, a_t)
            # so the next_achieved_goal depends also on the previous state and action
            # because we are in a GoalEnv:
            # r_t = reward(s_t, a_t) = reward(next_achieved_goal, desired_goal)
            # therefore we have to use "next_achieved_goal" and not "achieved_goal"
            transitions["next_achieved_goal"][her_indices, 0],
            # here we use the new desired goal
            transitions["desired_goal"][her_indices, 0],
            transitions["info"][her_indices, 0],
        )
        infos = transitions['info']
        success = np.array([[inf['is_success'] for inf in info] for info in infos])
        if self.hindsight_sampling_done_if_success:
            transitions['done'] = success
        # Perform action replay
        if self.perform_action_replay_transitions:
            assert len(transitions['next_achieved_goal'].shape) == 3 and \
                   transitions['next_achieved_goal'].shape[1] == 1 and \
                   len(transitions['action'].shape) == 2, "Error! Unexpected dimension during action replay transition sampling."
            # perform action replay only where the action was not successful.
            no_success_idxs = np.where(np.isclose(success, 0.0))
            unscaled_action = transitions['next_achieved_goal'].reshape([transitions['next_achieved_goal'].shape[0], transitions['next_achieved_goal'].shape[2]])
            scaled_action = self.env.unwrapped.envs[0].unwrapped.layer_alg.policy.scale_action(unscaled_action)
            transitions['action'][no_success_idxs] = scaled_action[no_success_idxs]

        # concatenate observation with (desired) goal
        observations = ObsDictWrapper.convert_dict(self._normalize_obs(transitions, maybe_vec_env))
        # HACK to make normalize obs work with the next observation
        transitions["observation"] = transitions["next_obs"]
        next_observations = ObsDictWrapper.convert_dict(self._normalize_obs(transitions, maybe_vec_env))

        data = (
            observations[:, 0],
            transitions["action"],
            next_observations[:, 0],
            transitions["done"],
            self._normalize_reward(transitions["reward"], maybe_vec_env),
            transitions["is_subgoal_testing_trans"],
        )
        return ReplayBufferSamplesWithTestTrans(*tuple(map(self.to_torch, data)))

    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[dict],
    ) -> None:

        if self.current_idx == 0 and self.full:
            # Clear info buffer
            self.info_buffer[self.pos] = deque(maxlen=self.max_episode_length)
        self.buffer["observation"][self.pos][self.current_idx] = obs["observation"]
        self.buffer["achieved_goal"][self.pos][self.current_idx] = obs["achieved_goal"]
        self.buffer["desired_goal"][self.pos][self.current_idx] = obs["desired_goal"]
        self.buffer["action"][self.pos][self.current_idx] = action
        self.buffer["done"][self.pos][self.current_idx] = done
        self.buffer["reward"][self.pos][self.current_idx] = reward
        self.buffer["next_obs"][self.pos][self.current_idx] = next_obs["observation"]
        self.buffer["next_achieved_goal"][self.pos][self.current_idx] = next_obs["achieved_goal"]
        self.buffer["next_desired_goal"][self.pos][self.current_idx] = next_obs["desired_goal"]
        has_test_trans = True in ['is_subgoal_testing_trans' in inf.keys() for inf in infos]
        if has_test_trans:
            test_trans = [inf.get('is_subgoal_testing_trans') for inf in infos]
            test_trans = np.array(test_trans)
            self.buffer["is_subgoal_testing_trans"][self.pos][self.current_idx] = test_trans

        self.info_buffer[self.pos].append(infos)

        # update current pointer
        self.current_idx += 1

    def store_episode(self) -> None:
        """
        Increment episode counter
        and reset transition pointer.
        """
        # add episode length to length storage
        self.episode_lengths[self.pos] = self.current_idx

        # update current episode pointer
        # Note: in the OpenAI implementation
        # when the buffer is full, the episode replaced
        # is randomly chosen
        self.pos += 1
        if self.pos == self.max_episode_stored:
            self.full = True
            self.pos = 0
        # reset transition pointer
        self.current_idx = 0

    @property
    def n_episodes_stored(self) -> int:
        if self.full:
            return self.max_episode_stored
        return self.pos

    def size(self) -> int:
        """
        :return: The current number of transitions in the buffer.
        """
        return int(np.sum(self.episode_lengths))

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.buffer = {
            key: np.zeros((self.max_episode_stored, self.max_episode_length, *dim), dtype=np.float32)
            for key, dim in self.input_shape.items()
        }
        # Store info dicts are it can be used to compute the reward (e.g. continuity cost)
        self.info_buffer = [deque(maxlen=self.max_episode_length) for _ in range(self.max_episode_stored)]
        # episode length storage, needed for episodes which has less steps than the maximum length
        self.pos = 0
        self.current_idx = 0
        self.full = False
        self.episode_lengths = np.zeros(self.max_episode_stored, dtype=np.int64)
