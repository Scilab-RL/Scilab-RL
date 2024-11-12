import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch

from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped

from custom_algorithms.cleanppofm.utils import get_position_and_object_positions_of_observation, \
    get_observation_of_position_and_object_positions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# modified copy from stable baselines
def evaluate_policy(
        model: "base_class.BaseAlgorithm",
        env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
        reward_threshold: Optional[float] = None,
        return_episode_rewards: bool = False,
        warn: bool = True,
        # from us
        callback_metric_viz=None,
        logger=None,
) -> Union[Tuple[float, float], Tuple[List[float], List[int], List[int]]]:
    """
    From the stable-baselines3 evaluation implementation.

    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate.
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    current_number_of_crashed_or_collected_objects = np.zeros(n_envs, dtype="int")
    print("reset")
    observations = env.reset()
    states = None

    ### from me
    episode_number_of_crashed_or_collected_objects = []
    # Note: you should use vec_env.env_method("get_wrapper_attr", "attribute_name") in Gymnasium v1.0
    env_name = env.env_method("get_wrapper_attr", "name")[0]
    ###

    while (episode_counts < episode_count_targets).any():

        ### custom code
        actions, states, forward_normal = model.predict(observations, state=states, deterministic=deterministic)
        # for logging: get last position, predicted position and new position
        observation_width = env.env_method("get_wrapper_attr", "observation_width")[0]
        observation_height = env.env_method("get_wrapper_attr", "observation_height")[0]
        agent_size = env.env_method("get_wrapper_attr", "size")[0]
        position = get_position_and_object_positions_of_observation(obs=torch.tensor(observations),
                                                                    observation_width=observation_width,
                                                                    observation_height=observation_height,
                                                                    agent_size=agent_size)[0][0]
        predicted_x_position = min(max(1, forward_normal.mean.cpu().detach().numpy()[0][0]), 10)
        expected_new_positon = min(max(1, position + (actions[0] - 1)), 10)

        observations, rewards, dones, infos, prediction_error, difficulty, soc, reward_with_future_reward_estimation_corrective, _, _ = model.step_in_env(
            actions=actions,
            forward_normal=forward_normal)

        new_position = get_position_and_object_positions_of_observation(torch.tensor(observations),
                                                                        observation_width=observation_width,
                                                                        observation_height=observation_height,
                                                                        agent_size=agent_size)[0][0]

        if model.reward_predicting:
            logger.record("eval/predicted_rewards", float(forward_normal.mean[:, -1].mean()))
            logger.record_mean("eval/predicted_rewards_mean", float(forward_normal.mean[:, -1].mean()))
            logger.record("eval/reward_with_future_reward_estimation_corrective",
                          reward_with_future_reward_estimation_corrective.mean())
            logger.record_mean("eval/reward_with_future_reward_estimation_corrective_mean",
                               reward_with_future_reward_estimation_corrective.mean())
        logger.record("eval/prediction_error", prediction_error)
        logger.record_mean("eval/prediction_error_mean", prediction_error)
        logger.record("eval/difficulty", difficulty)
        logger.record_mean("eval/difficulty_mean", difficulty)
        logger.record("eval/soc", soc)
        logger.record_mean("eval/soc_mean", soc)
        logger.record("eval/action", actions[0])
        logger.record("eval/last_position", position)
        logger.record("eval/expected_new_position", expected_new_positon)
        logger.record("eval/predicted_x_position", predicted_x_position)
        logger.record("eval/new_position", new_position)
        # already logged in custom callback
        logger.record("eval/rollout_rewards_step", float(rewards.mean()))
        logger.record_mean("eval/rollout_rewards_mean", float(rewards.mean()))

        # dodge/collect env
        if "simple" in infos[0].keys():
            logger.record("eval/number_of_crashed_or_collected_objects",
                          float(infos[0]["number_of_crashed_or_collected_objects"]))
        # trigger metric visualization
        if callback_metric_viz:
            callback_metric_viz._on_step()

        if "simple" in infos[0].keys():
            current_number_of_crashed_or_collected_objects += infos[0]["number_of_crashed_or_collected_objects"]

        ### until here

        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:

                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1

                        ### from me
                        episode_number_of_crashed_or_collected_objects.append(
                            current_number_of_crashed_or_collected_objects[i])

                    current_rewards[i] = 0
                    current_lengths[i] = 0

                    ### from me
                    current_number_of_crashed_or_collected_objects[i] = 0

                    if states is not None:
                        states[i] *= 0

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths, episode_number_of_crashed_or_collected_objects
    return mean_reward, std_reward


def evaluate_policy_meta_agent(
        model: "type_aliases.PolicyPredictor",
        env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
        reward_threshold: Optional[float] = None,
        return_episode_rewards: bool = False,
        warn: bool = True,
        # from us
        callback_metric_viz=None,
        logger=None,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    From the stable-baselines3 evaluation implementation.

    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)

    ### from me
    last_action = np.array([0])
    current_number_of_crashed_objects = np.zeros(n_envs, dtype="int")
    current_number_of_collected_objects = np.zeros(n_envs, dtype="int")
    current_number_of_switches = np.zeros(n_envs, dtype="int")
    current_number_of_dodge_actions = np.zeros(n_envs, dtype="int")
    current_number_of_collect_actions = np.zeros(n_envs, dtype="int")
    episode_number_of_crashed_objects = []
    episode_number_of_collected_objects = []
    episode_number_of_switches = []
    episode_number_of_dodge_actions = []
    episode_number_of_collect_actions = []
    ###

    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )

        # ### evaluate the forward model
        # observation_width = env.env_method("get_wrapper_attr", "observation_width")[0]
        # observation_height = env.env_method("get_wrapper_attr", "observation_height")[0]
        # agent_size = env.env_method("get_wrapper_attr", "agent_size")[0]
        # maximum_number_of_objects = env.env_method("get_wrapper_attr", "maximum_number_of_objects")[0]
        #
        # last_observations = np.array([])
        # if actions[0] == 0:
        #     for i in range(0, observation_height * 2, 2):
        #         last_observations = np.concatenate(
        #             (last_observations, observations[0][i * (observation_width + 2):(i + 1) * (observation_width + 2)]),
        #             axis=0)
        #     trained_agent = env.env_method("get_wrapper_attr", "trained_dodge_asteroids")[0]
        #     task = "dodge"
        # else:
        #     for i in range(1, observation_height * 2 + 1, 2):
        #         last_observations = np.concatenate(
        #             (last_observations, observations[0][i * (observation_width + 2):(i + 1) * (observation_width + 2)]),
        #             axis=0)
        #     trained_agent = env.env_method("get_wrapper_attr", "trained_collect_asteroids")[0]
        #     task = "collect"
        #
        # last_observations = np.expand_dims(last_observations, axis=0)
        # last_observations = get_position_and_object_positions_of_observation(
        #     obs=torch.tensor(last_observations, device=device),
        #     maximum_number_of_objects=maximum_number_of_objects,
        #     observation_width=observation_width,
        #     observation_height=observation_height,
        #     agent_size=agent_size)
        #
        # forward_model = trained_agent.fm_network
        # # tensor (1, 22) & tensor (1, 1)
        # forward_model_prediction = forward_model(last_observations, torch.tensor([actions]).float())

        # belief_state = get_observation_of_position_and_object_positions(agent_and_object_positions=
        #                                                                 forward_model_prediction.mean[
        #                                                                     0][:-1].cpu().unsqueeze(0),
        #                                                                 observation_height=30,
        #                                                                 observation_width=40,
        #                                                                 agent_size=2,
        #                                                                 task=task).flatten().cpu().numpy()
        # belief_state = np.expand_dims(belief_state, 0)
        new_observations, rewards, dones, infos = env.step(actions)

        # new_observations_current_task = np.array([])
        # if actions[0] == 0:
        #     for i in range(0, observation_height * 2, 2):
        #         new_observations_current_task = np.concatenate(
        #             (new_observations_current_task,
        #              new_observations[0][i * (observation_width + 2):(i + 1) * (observation_width + 2)]),
        #             axis=0)
        # else:
        #     for i in range(1, observation_height * 2 + 1, 2):
        #         new_observations_current_task = np.concatenate(
        #             (new_observations_current_task,
        #              new_observations[0][i * (observation_width + 2):(i + 1) * (observation_width + 2)]),
        #             axis=0)
        #
        # new_observations_current_task = np.expand_dims(new_observations_current_task, axis=0)
        # positions_of_new_observation = \
        #     get_position_and_object_positions_of_observation(
        #         obs=torch.tensor(new_observations_current_task, device=device),
        #         # fixme: not hardcoded
        #         maximum_number_of_objects=10,
        #         observation_width=observation_width,
        #         observation_height=observation_height,
        #         agent_size=agent_size)

        # print("forward_model_prediction", torch.round(forward_model_prediction.mean))
        # print("positions_of_new_observation", positions_of_new_observation)

        ### custom code
        # EXAMPLE INFOS:
        # [{'info_dodge': [{'simple': 10, 'gaussian': 10, 'pos_neg': {'pos': [0], 'neg': [0]}, 'number_of_crashed_or_collected_objects': 0, 'TimeLimit.truncated': False}],
        # 'info_collect': [{'simple': 0, 'gaussian': 0, 'pos_neg': {'pos': [0], 'neg': [0]}, 'number_of_crashed_or_collected_objects': 0, 'TimeLimit.truncated': False}],
        # 'reward_dodge': array([0.96], dtype=float32), 'reward_collect': 0.45,
        # 'action_meta': 0, 'dodge_position_before': 1, 'collect_position_before': 5,
        # 'dodge_action': array([0]), 'collect_action': 1, 'input_noise': 2,
        # 'dodge_next_position': 2, 'collect_next_position': 5,
        # 'predicted_dodge_next_position': 1, 'predicted_collect_next_position': 5,
        # 'prediction_error': 0.11068782380292025, 'difficulty': array([0.01], dtype=float32),
        # 'SoC_dodge': array([0.94], dtype=float32), 'SoC_collect': 0.9, 'TimeLimit.truncated': False}]

        info_dict = infos[0]
        logger.record("eval/action_meta", info_dict["action_meta"])
        logger.record("eval/dodge_position_before", info_dict["dodge_position_before"])
        logger.record("eval/collect_position_before", info_dict["collect_position_before"])
        logger.record("eval/dodge_action", info_dict["dodge_action"])
        logger.record("eval/collect_action", info_dict["collect_action"])
        logger.record("eval/input_noise", info_dict["input_noise"])
        logger.record("eval/dodge_next_position", info_dict["dodge_next_position"])
        logger.record("eval/collect_next_position", info_dict["collect_next_position"])
        logger.record("eval/predicted_dodge_next_position", info_dict["predicted_dodge_next_position"])
        logger.record("eval/predicted_collect_next_position", info_dict["predicted_collect_next_position"])
        logger.record("eval/prediction_error", info_dict["prediction_error"])
        logger.record("eval/difficulty", info_dict["difficulty"])
        logger.record("eval/SoC_dodge", info_dict["SoC_dodge"])
        logger.record("eval/SoC_collect", info_dict["SoC_collect"])
        logger.record("eval/reward_dodge", info_dict["reward_dodge"])
        logger.record("eval/reward_collect", info_dict["reward_collect"])
        logger.record("eval/dodge_object_crashed",
                      info_dict["info_dodge"][0]["number_of_crashed_or_collected_objects"])
        logger.record("eval/collect_object_collected",
                      info_dict["info_collect"][0]["number_of_crashed_or_collected_objects"])
        # print("eval/action_meta", info_dict["action_meta"])
        # print("eval/dodge_position_before", info_dict["dodge_position_before"])
        # print("eval/collect_position_before", info_dict["collect_position_before"])
        # print("eval/dodge_action", info_dict["dodge_action"])
        # print("eval/collect_action", info_dict["collect_action"])
        # print("eval/input_noise", info_dict["input_noise"])
        # print("eval/dodge_next_position", info_dict["dodge_next_position"])
        # print("eval/collect_next_position", info_dict["collect_next_position"])
        # print("eval/predicted_dodge_next_position", info_dict["predicted_dodge_next_position"])
        # print("eval/predicted_collect_next_position", info_dict["predicted_collect_next_position"])
        # print("eval/prediction_error", info_dict["prediction_error"])
        # print("eval/difficulty", info_dict["difficulty"])
        # print("eval/SoC_dodge", info_dict["SoC_dodge"])
        # print("eval/SoC_collect", info_dict["SoC_collect"])
        # print("eval/reward_dodge", info_dict["reward_dodge"])
        # print("eval/reward_collect", info_dict["reward_collect"])
        # print("eval/dodge_object_crashed_or_collected",
        #       info_dict["info_dodge"][0]["number_of_crashed_or_collected_objects"])
        # print("eval/collect_object_crashed_or_collected",
        #       info_dict["info_collect"][0]["number_of_crashed_or_collected_objects"])
        logger.record("eval/rollout_rewards_step", float(rewards.mean()))
        logger.record_mean("eval/rollout_rewards_mean", float(rewards.mean()))

        # dodge/collect env
        if "simple" in infos[0].keys():
            logger.record("eval/number_of_crashed_or_collected_objects",
                          float(infos[0]["number_of_crashed_or_collected_objects"]))
        # trigger metric visualization
        if callback_metric_viz:
            callback_metric_viz._on_step()

        current_number_of_crashed_objects += info_dict["info_dodge"][0]["number_of_crashed_or_collected_objects"]
        current_number_of_collected_objects += info_dict["info_collect"][0]["number_of_crashed_or_collected_objects"]
        if not (last_action == actions).item():
            current_number_of_switches += 1
            last_action = actions
        if actions == np.array([0]):
            current_number_of_dodge_actions += 1
        elif actions == np.array([1]):
            current_number_of_collect_actions += 1

        ### until here

        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1

                        ### from me
                        episode_number_of_crashed_objects.append(current_number_of_crashed_objects[i])
                        episode_number_of_collected_objects.append(current_number_of_collected_objects[i])
                        episode_number_of_switches.append(current_number_of_switches[i])
                        episode_number_of_dodge_actions.append(current_number_of_dodge_actions[i])
                        episode_number_of_collect_actions.append(current_number_of_collect_actions[i])
                        ###
                    current_rewards[i] = 0
                    current_lengths[i] = 0

                    ### from me
                    current_number_of_crashed_objects[i] = 0
                    current_number_of_collected_objects[i] = 0
                    current_number_of_switches[i] = 0
                    current_number_of_dodge_actions[i] = 0
                    current_number_of_collect_actions[i] = 0

        observations = new_observations

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths, episode_number_of_crashed_objects, episode_number_of_collected_objects, episode_number_of_switches, episode_number_of_dodge_actions, episode_number_of_collect_actions
    return mean_reward, std_reward
