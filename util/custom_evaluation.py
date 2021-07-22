from typing import Callable, List, Optional, Tuple, Union, Dict
from stable_baselines3.common.logger import Logger
import gym
import numpy as np
import cv2

from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import VecEnv

def get_success(info_list):
    succ = np.nan
    for entry in info_list:
        try:
            succ = entry['is_success']
        except:
            continue
    return succ



def evaluate_policy(
    agent: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render_info: Dict = None,
    callback: Optional[Callable] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    logger: Logger = None
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    This is made to work only with one env.

    :param agent: The RL agent you want to evaluate.
    :param env: The gym environment. In the case of a ``VecEnv``
        this must contain only one environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param callback: callback function to do additional checks,
        called after each step.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of reward per episode
        will be returned instead of the mean.
    :return: Mean reward per episode, std of reward per episode
        returns ([float], [int]) when ``return_episode_rewards`` is True
    """

    video_writer = None
    render = render_info is not None and (render_info['eval_count']-1) % render_info['render_every_n_eval'] == 0
    if render:
        if render_info is not None and 'fps' in render_info:
            try:
                video_writer = cv2.VideoWriter(render_info['path'] + '/eval_{}.avi'.format(render_info['eval_count']),
                                                cv2.VideoWriter_fourcc('F', 'M', 'P', '4'), render_info['fps'], render_info['size'])
            except:
                logger.info("Error creating video writer")

    info_list = []
    episode_rewards, episode_lengths, episode_successes = [], [], []
    for i in range(n_eval_episodes):
        # Avoid double reset, as VecEnv are reset automatically
        if not isinstance(env, VecEnv) or i == 0:
            obs = env.reset()
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        episode_success = 0.0
        while not done:
            if render:
                if video_writer is not None:
                    if hasattr(env, 'venv'):
                        frame = env.venv.envs[0].render(mode='rgb_array', width=render_info['size'][0],
                                                                height=render_info['size'][1])
                    else:
                        frame = env.render(mode='rgb_array')
                    video_writer.write(frame)
                elif render_info is not None and 'mode' in render_info:
                    if hasattr(env, 'venv'):
                        env.venv.envs[0].render(mode=render_info['mode'])
                    else:
                        env.render(mode=render_info['mode'])

            action, state = agent.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, _info = env.step(action)
            this_episode_success = get_success(_info)
            if not episode_success or episode_success == np.nan:
                episode_success = this_episode_success
            episode_reward += reward
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            if episode_success and episode_success is not np.nan: # Early abort on success.
                done = True
                if isinstance(env, VecEnv):
                    env.reset()
        episode_successes.append(episode_success)
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    if video_writer is not None:
        video_writer.release()
    mean_success = np.mean(episode_successes)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths, episode_successes
    return mean_reward, std_reward, mean_length, mean_success
