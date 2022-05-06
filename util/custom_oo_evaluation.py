from typing import Callable, List, Optional, Tuple, Union, Dict
from stable_baselines3.common.logger import Logger
import gym
import numpy as np
import cv2
import random

from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import VecEnv
from ideas_baselines.oo_sac.oo_blocks_adapter import OO_Blocks_Adapter


def get_success(info_list):
    succ = np.nan
    for entry in info_list:
        try:
            succ = entry['is_success']
        except:
            continue
    return succ


threshold = 0
obj_idx = 1


def _is_success(achieved_goal, desired_goal):
    d = goal_distance(achieved_goal[obj_idx * 3:obj_idx * 3 + 3], desired_goal[obj_idx * 3:obj_idx * 3 + 3])
    return (d < threshold).astype(np.float32)


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


def evaluate_oo_policy(
    agent: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render_info: Dict = None,
    callback: Optional[Callable] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    logger: Logger = None,
    distance_threshold = 0.05
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
    global threshold
    threshold = distance_threshold
    #env.envs[0].env.env._is_success = _is_success

    global obj_idx

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
    tries_per_object = np.zeros(env.envs[0].n_objects + 1)  # + gripper
    success_per_object = np.zeros(env.envs[0].n_objects + 1)

    env.envs[0].__class__ = OO_Blocks_Adapter

    for i in range(n_eval_episodes):
        obj_idx = random.randint(1, env.envs[0].n_objects)
        tries_per_object[obj_idx] += 1
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

            obs = transform_obs_to_oo_obs(obs, env, obj_idx)
            action, state = agent.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, _info = env.step(action)
            this_episode_success = get_success(_info)

            if this_episode_success > 0:
                success_per_object[obj_idx] += this_episode_success

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
    success_per_object = success_per_object / tries_per_object
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths, episode_successes, success_per_object
    return mean_reward, std_reward, mean_length, mean_success, success_per_object


def _is_one_hot(vec):
    return np.isin(vec, [0., 1.]).all()


def transform_obs_to_oo_obs(obs, env, obj_idx):
    oo_obs = obs.copy()
    original_len = obs['achieved_goal'].shape[1]
    n_obj = env.envs[0].n_objects
    oneHot_idx = np.eye(n_obj + 1)[obj_idx]

    # Considering one-hot vector
    if _is_one_hot(oo_obs['achieved_goal'][0][:n_obj+1]):
        achieved_coords = oo_obs['achieved_goal'][0][1 + n_obj: 1 + n_obj + 3]
        desired_coords = oo_obs['desired_goal'][0][1 + n_obj: 1 + n_obj + 3]
    else:
        achieved_coords = oo_obs['achieved_goal'][0][obj_idx * 3: obj_idx * 3 + 3]
        desired_coords = oo_obs['desired_goal'][0][obj_idx * 3: obj_idx * 3 + 3]

    oo_obs['achieved_goal'] = np.expand_dims(np.concatenate([oneHot_idx, achieved_coords]), axis=0)
    oo_obs['desired_goal'] = np.expand_dims(np.concatenate([oneHot_idx, desired_coords]), axis=0)
    # Zero-pad values if vector is too long
    len_diff = abs(len(oo_obs['achieved_goal'][0]) - original_len)
    if len_diff != 0:
        oo_obs['achieved_goal'] = np.expand_dims(
            np.concatenate([oo_obs['achieved_goal'][0], np.zeros(len_diff)]), axis=0)
        oo_obs['desired_goal'] = np.expand_dims(
            np.concatenate([oo_obs['desired_goal'][0], np.zeros(len_diff)]), axis=0)
    return oo_obs

