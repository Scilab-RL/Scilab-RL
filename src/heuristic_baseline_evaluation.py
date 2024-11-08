import torch
import gymnasium as gym
from src.custom_envs.register_envs import register_custom_envs

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped


# USE EVALUATE POLICY OF STABLE BASELINES 3 WITHOUT A MODEL BUT A HEURISTIC
def evaluate_policy(
        env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
        reward_threshold: Optional[float] = None,
        return_episode_rewards: bool = False,
        warn: bool = True,
        action_sequence: list[int] = None,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
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
    :param action_sequence: List of actions to be executed in the environment
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
    ##### MY CODE #####
    # ndarray (1,)
    if action_sequence is None:
        actions = np.array([0])

    counter = 0

    current_number_of_crashed_objects = np.zeros(n_envs, dtype="int")
    current_number_of_collected_objects = np.zeros(n_envs, dtype="int")
    episode_number_of_crashed_objects = []
    episode_number_of_collected_objects = []
    ###################

    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        ##### MY CODE #####
        # normally here would the model be used to predict the action
        # but we use a heuristic instead
        if action_sequence is None:
            if actions[0] == 0:
                actions = np.array([1])
            else:
                actions = np.array([0])
        else:
            actions = np.array([action_sequence[counter]])

        ###################
        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1

        ##### MY CODE #####
        counter += 1

        info_dict = infos[0]
        current_number_of_crashed_objects += info_dict["info_dodge"][0]["number_of_crashed_or_collected_objects"]
        current_number_of_collected_objects += info_dict["info_collect"][0]["number_of_crashed_or_collected_objects"]
        ###################

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

                        ##### MY CODE #####
                        episode_number_of_crashed_objects.append(current_number_of_crashed_objects[i])
                        episode_number_of_collected_objects.append(current_number_of_collected_objects[i])
                        ###################

                    current_rewards[i] = 0
                    current_lengths[i] = 0

                    ##### MY CODE #####
                    current_number_of_crashed_objects[i] = 0
                    current_number_of_collected_objects[i] = 0
                    ###################

        observations = new_observations

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    ##### MY CODE #####
    mean_crashed_objects = np.mean(episode_number_of_crashed_objects)
    std_crashed_objects = np.std(episode_number_of_crashed_objects)
    mean_collected_objects = np.mean(episode_number_of_collected_objects)
    std_collected_objects = np.std(episode_number_of_collected_objects)

    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Mean number of crashed objects: {mean_crashed_objects:.2f} +/- {std_crashed_objects:.2f}")
    print(f"Mean number of collected objects: {mean_collected_objects:.2f} +/- {std_collected_objects:.2f}")
    print(f"Episode rewards: {episode_rewards}")
    print(f"Episode number of crashed objects: {episode_number_of_crashed_objects}")
    print(f"Episode number of collected objects: {episode_number_of_collected_objects}")
    ###################

    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward


def calculate_action_sequence_of_means_of_frame_number_of_humans(human_mean_dodge_switch_frame_number: float,
                                                                 human_std_dodge_switch_frame_number: float,
                                                                 human_mean_collect_switch_frame_number: float,
                                                                 human_std_collect_switch_frame_number: float,
                                                                 n_eval_episodes: int) -> List[
    int]:
    """

    Args:
        human_mean_dodge_switch_frame_number: Mean number of frames a human stays in the dodge task
        human_std_dodge_switch_frame_number: Standard deviation of the number of frames a human stays
        in the dodge task
        human_mean_collect_switch_frame_number: Mean number of frames a human stays in the collect task
        human_std_collect_switch_frame_number: Standard deviation of the number of frames a human stays
        in the collect task
        n_eval_episodes: Number of episodes to evaluate the policy

    Returns: action sequence in form of a list of integers (which are the actions)

    """
    # in total we need ~500 steps in one episode
    action_sequence = []

    while len(action_sequence) < (500 * n_eval_episodes):
        # sample from dodge mean and std and add the corresponding number of 0s
        num_zeros = int(np.random.normal(human_mean_dodge_switch_frame_number, human_std_dodge_switch_frame_number))
        action_sequence.extend([0] * num_zeros)

        # sample from collect mean and std and add the corresponding number of 1s
        num_ones = int(np.random.normal(human_mean_collect_switch_frame_number, human_std_collect_switch_frame_number))
        action_sequence.extend([1] * num_ones)

    return action_sequence


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    register_custom_envs()

    n_eval_episodes = 10

    print("CURRENTLY EVALUATING HARD HARD INPUT NOISE")
    print("The standard deviation of task switches frames is too high!")
    # dodge: 2.788788 (± 5.240052)
    # collect: 3.996455 (± 3.889264)
    human_mean_dodge_switch_frame_number = 2.788788 * 10
    human_std_dodge_switch_frame_number = 5.240052 * 10
    human_mean_collect_switch_frame_number = 3.996455 * 10
    human_std_collect_switch_frame_number = 3.889264 * 10

    action_sequence = calculate_action_sequence_of_means_of_frame_number_of_humans(
        human_mean_dodge_switch_frame_number=human_mean_dodge_switch_frame_number,
        human_std_dodge_switch_frame_number=human_std_dodge_switch_frame_number,
        human_mean_collect_switch_frame_number=human_mean_collect_switch_frame_number,
        human_std_collect_switch_frame_number=human_std_collect_switch_frame_number, n_eval_episodes=n_eval_episodes)

    # Initialise the environment
    env = gym.make("MetaEnv-pretrained-human-subtask-modelbased-v0", render_mode="human",
                   dodge_best_model_name="dodge_MB_reward_included_rl_model_best",
                   collect_best_model_name="collect_reward_in_mb_rl_model_best")
    # mean_reward, std_reward = evaluate_policy(env=env, n_eval_episodes=10, deterministic=True, render=True)
    mean_reward, std_reward = evaluate_policy(env=env, n_eval_episodes=n_eval_episodes, deterministic=True, render=True,
                                              action_sequence=action_sequence)
    print(mean_reward, std_reward)
