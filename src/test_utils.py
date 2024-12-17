import unittest
from unittest import mock
import torch
import gymnasium as gym
import numpy as np
import copy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from src.custom_algorithms.cleanppofm.utils import (get_summed_up_reward_of_env_with_predicted_states_hardcoded, \
                                                    get_position_and_object_positions_of_observation,
                                                    get_observation_of_position_and_object_positions, \
                                                    get_next_position_observation_moonlander,
                                                    calculate_prediction_error,
                                                    calculate_need_for_control,
                                                    normalize_rewards,
                                                    get_collected_objects,
                                                    calculate_trajectory_length,
                                                    get_next_normalized_reward)
from src.custom_envs.register_envs import register_custom_envs
from custom_algorithms.cleanppofm.agent import Agent
from custom_algorithms.cleanppofm.forward_model import ProbabilisticForwardNetPositionPredictionIncludingReward


class TestUtils(unittest.TestCase):
    @mock.patch("stable_baselines3.common.vec_env.dummy_vec_env.DummyVecEnv.env_method")
    def test_get_summed_up_reward_of_env_with_predicted_states_hardcoded_are_errors_raised(self, env_method_mock):
        # env_method is called 5 times, for env_name, task, observation_height, observation_width, agent_size
        # we call get_summed_up_reward_of_env_with_predicted_states_hardcoded two times, so 10 mock return values are needed
        env_method_mock.side_effect = [["bla"], ["bla"], ["bla"], ["bla"], ["bla"], ["MoonlanderWorldEnv"], ["bla"],
                                       ["bla"], ["bla"], ["bla"]]

        register_custom_envs()
        env = gym.make("MoonlanderWorld-dodge-gaussian-v0")
        dummy_vec_env = DummyVecEnv([lambda: env])

        with self.subTest("other env than MoonlanderWorldEnv"):
            self.assertRaises(NotImplementedError, get_summed_up_reward_of_env_with_predicted_states_hardcoded,
                              env=dummy_vec_env, last_observation_positions=torch.tensor([[2., 1.]]),
                              number_of_future_steps=1)
        with self.subTest("other task than dodge or collect"):
            self.assertRaises(ValueError, get_summed_up_reward_of_env_with_predicted_states_hardcoded,
                              env=dummy_vec_env, last_observation_positions=torch.tensor([[2., 1.]]),
                              number_of_future_steps=1)

    def test_get_summed_up_reward_of_env_with_predicted_states_hardcoded(self) -> None:
        register_custom_envs()
        dodge_env = gym.make("MoonlanderWorld-dodge-gaussian-v0")
        dodge_dummy_vec_env = DummyVecEnv([lambda: dodge_env])
        collect_env = gym.make("MoonlanderWorld-collect-gaussian-v0")
        collect_dummy_vec_env = DummyVecEnv([lambda: collect_env])

        with self.subTest("Dodge"):
            with self.subTest("1 future step, zero objects"):
                reward = get_summed_up_reward_of_env_with_predicted_states_hardcoded(
                    env=dodge_dummy_vec_env, last_observation_positions=torch.tensor([[2., 1.]]),
                    number_of_future_steps=1)
                self.assertEqual(reward, 0.5)

            with self.subTest("1 future step, crashing one object"):
                reward = get_summed_up_reward_of_env_with_predicted_states_hardcoded(
                    env=dodge_dummy_vec_env, last_observation_positions=torch.tensor([[2., 1., 4., 1.]]),
                    number_of_future_steps=1)
                self.assertEqual(reward, 0)

            with self.subTest("3 future step, crashing objects"):
                reward = get_summed_up_reward_of_env_with_predicted_states_hardcoded(
                    env=dodge_dummy_vec_env, last_observation_positions=torch.tensor([[2., 1., 4., 4.]]),
                    number_of_future_steps=3)
                self.assertEqual(reward, (0 + 0 + 0) / 3)

            with self.subTest("5 future step, object is flying out"):
                reward = get_summed_up_reward_of_env_with_predicted_states_hardcoded(
                    env=dodge_dummy_vec_env, last_observation_positions=torch.tensor([[2., 1., 15., 1., 19., 5.]]),
                    number_of_future_steps=5)
                self.assertEqual(reward, (0.5 + 0.5 + 0.5 + 0.5 + 0.5) / 5)

        with self.subTest("Collect"):
            with self.subTest("1 future step, zero objects"):
                reward = get_summed_up_reward_of_env_with_predicted_states_hardcoded(
                    env=collect_dummy_vec_env, last_observation_positions=torch.tensor([[2., 1.]]),
                    number_of_future_steps=1)
                self.assertEqual(reward, 0.5)

            with self.subTest("1 future step, collecting one object"):
                reward = get_summed_up_reward_of_env_with_predicted_states_hardcoded(
                    env=collect_dummy_vec_env, last_observation_positions=torch.tensor([[2., 1., 4., 1.]]),
                    number_of_future_steps=1)
                self.assertEqual(reward, 1)

            with self.subTest("3 future step, collecting one objects"):
                reward = get_summed_up_reward_of_env_with_predicted_states_hardcoded(
                    env=collect_dummy_vec_env, last_observation_positions=torch.tensor([[2., 1., 4., 4.]]),
                    number_of_future_steps=3)
                # got first number manually
                self.assertEqual(reward, (0.7016129032258065 + 1 + 0.5) / 3)

            with self.subTest("5 future step, object is flying out"):
                reward = get_summed_up_reward_of_env_with_predicted_states_hardcoded(
                    env=collect_dummy_vec_env, last_observation_positions=torch.tensor([[2., 1., 15., 1., 19., 5.]]),
                    number_of_future_steps=5)
                self.assertEqual(reward, (0.5 + 0.5 + 0.5 + 0.5 + 0.5) / 5)

    def test_get_position_and_object_positions_of_observation(self) -> None:
        # observation (64, 1260), 1260 = 30 * 42
        with self.subTest(
                "standard example, multiple observations, more and less than maximum number of objects, agent size 1"):
            observation_0 = torch.tensor([
                [-1., 0., 1., 2., -1.],
                [-1., 0., 0., 0., -1.],
                [-1., 2., 0., 2., -1.],
                [-1., 0., 2., 0., -1.],
                [-1., 0., 0., 0., -1.],
            ]).flatten()
            observation_1 = torch.tensor([
                [-1., 3., 1., 0., -1.],
                [-1., 0., 0., 0., -1.],
                [-1., 0., 0., 0., -1.],
                [-1., 0., 0., 0., -1.],
                [-1., 3., 0., 0., -1.],
            ]).flatten()
            observation = torch.cat((observation_0.unsqueeze(0), observation_1.unsqueeze(0)), dim=0)
            agent_and_object_positions_tensor = get_position_and_object_positions_of_observation(obs=observation,
                                                                                                 maximum_number_of_objects=3,
                                                                                                 observation_width=3,
                                                                                                 observation_height=5,
                                                                                                 agent_size=1)
            self.assertTrue(torch.equal(agent_and_object_positions_tensor, torch.tensor(
                [[2., 0., 3., 0., 1., 2., 3., 2.], [2., 0., 1., 0., 1., 4., 0., 0.]])))

        with self.subTest("different versions of objects (collect) flying out of the observation, agent size 2"):
            observation = torch.tensor([
                [-1., 1., 1., 1., 0., 2., 2., 2., 0., 2., 2., 2., 0., 2., 2., 2., -1.],
                [-1., 1., 1., 1., 0., 2., 2., 2., 0., 2., 2., 2., 0., 0., 0., 0., -1.],
                [-1., 1., 1., 1., 0., 2., 2., 2., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
            ]).flatten().unsqueeze(0)
            agent_and_object_positions_tensor = get_position_and_object_positions_of_observation(obs=observation,
                                                                                                 maximum_number_of_objects=3,
                                                                                                 observation_width=15,
                                                                                                 observation_height=4,
                                                                                                 agent_size=2)
            self.assertTrue(
                torch.equal(agent_and_object_positions_tensor, torch.tensor([[2., 1., 10., 0., 14., -1., 6., 1]])))

        with self.subTest("different versions of objects (dodge) flying into the observation, agent size 2"):
            observation = torch.tensor([
                [-1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 1., 1., 1., 0., 3., 3., 3., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 1., 1., 1., 0., 3., 3., 3., 0., 3., 3., 3., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 3., 3., 3., 0., 3., 3., 3., 0., 3., 3., 3., -1.],
            ]).flatten().unsqueeze(0)
            agent_and_object_positions_tensor = get_position_and_object_positions_of_observation(obs=observation,
                                                                                                 maximum_number_of_objects=3,
                                                                                                 observation_width=15,
                                                                                                 observation_height=4,
                                                                                                 agent_size=2)
            self.assertTrue(
                torch.equal(agent_and_object_positions_tensor, torch.tensor([[2., 1., 6., 2., 10., 3., 14., 4]])))

        with self.subTest("objects (dodge) next to agent, agent size 2"):
            observation = torch.tensor([
                [-1., 3., 3., 3., 1., 1., 1., 3., 3., 3., -1.],
                [-1., 3., 3., 3., 1., 1., 1., 3., 3., 3., -1.],
                [-1., 3., 3., 3., 1., 1., 1., 3., 3., 3., -1.],
                [-1., 0., 0., 0., 3., 3., 3., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 3., 3., 3., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 3., 3., 3., 0., 0., 0., -1.],
            ]).flatten().unsqueeze(0)
            agent_and_object_positions_tensor = get_position_and_object_positions_of_observation(obs=observation,
                                                                                                 maximum_number_of_objects=3,
                                                                                                 observation_width=9,
                                                                                                 observation_height=6,
                                                                                                 agent_size=2)
            self.assertTrue(
                torch.equal(agent_and_object_positions_tensor, torch.tensor([[5., 1., 2., 1., 8., 1., 5., 4]])))

        with self.subTest("objects (dodge) covered by agent, agent size 2"):
            observation_0 = torch.tensor([
                [-1., 0., 0., 0., 1., 1., 1., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 1., 1., 1., 0., 0., 0., -1.],
                [-1., 0., 3., 3., 1., 1., 1., 3., 3., 0., -1.],
                [-1., 0., 3., 3., 3., 0., 3., 3., 3., 0., -1.],
                [-1., 0., 3., 3., 3., 0., 3., 3., 3., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
            ]).flatten().unsqueeze(0)
            observation_1 = torch.tensor([
                [-1., 0., 0., 0., 1., 1., 1., 0., 0., 0., -1.],
                [-1., 0., 3., 3., 1., 1., 1., 3., 3., 0., -1.],
                [-1., 0., 3., 3., 1., 1., 1., 3., 3., 0., -1.],
                [-1., 0., 3., 3., 3., 0., 3., 3., 3., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
            ]).flatten().unsqueeze(0)
            observation_2 = torch.tensor([
                [-1., 0., 3., 3., 1., 1., 1., 3., 3., 0., -1.],
                [-1., 0., 3., 3., 1., 1., 1., 3., 3., 0., -1.],
                [-1., 0., 3., 3., 1., 1., 1., 3., 3., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
            ]).flatten().unsqueeze(0)
            observation_3 = torch.tensor([
                [-1., 0., 3., 3., 1., 1., 1., 3., 3., 0., -1.],
                [-1., 0., 3., 3., 1., 1., 1., 3., 3., 0., -1.],
                [-1., 0., 0., 0., 1., 1., 1., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
            ]).flatten().unsqueeze(0)
            observation_4 = torch.tensor([
                [-1., 0., 3., 3., 1., 1., 1., 3., 3., 0., -1.],
                [-1., 0., 0., 0., 1., 1., 1., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 1., 1., 1., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
            ]).flatten().unsqueeze(0)

            observation_5 = torch.tensor([
                [-1., 0., 0., 0., 1., 1., 1., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 1., 1., 1., 0., 0., 0., -1.],
                [-1., 0., 0., 3., 1., 1., 1., 3., 0., 0., -1.],
                [-1., 0., 0., 3., 3., 3., 3., 3., 0., 0., -1.],
                [-1., 0., 0., 3., 3., 3., 3., 3., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
            ]).flatten().unsqueeze(0)
            observation_6 = torch.tensor([
                [-1., 0., 0., 0., 1., 1., 1., 0., 0., 0., -1.],
                [-1., 0., 0., 3., 1., 1., 1., 3., 0., 0., -1.],
                [-1., 0., 0., 3., 1., 1., 1., 3., 0., 0., -1.],
                [-1., 0., 0., 3., 3., 3., 3., 3., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
            ]).flatten().unsqueeze(0)
            observation_7 = torch.tensor([
                [-1., 0., 0., 3., 1., 1., 1., 3., 0., 0., -1.],
                [-1., 0., 0., 3., 1., 1., 1., 3., 0., 0., -1.],
                [-1., 0., 0., 3., 1., 1., 1., 3., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
            ]).flatten().unsqueeze(0)
            observation_8 = torch.tensor([
                [-1., 0., 0., 3., 1., 1., 1., 3., 0., 0., -1.],
                [-1., 0., 0., 3., 1., 1., 1., 3., 0., 0., -1.],
                [-1., 0., 0., 0., 1., 1., 1., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
            ]).flatten().unsqueeze(0)
            observation_9 = torch.tensor([
                [-1., 0., 0., 3., 1., 1., 1., 3., 0., 0., -1.],
                [-1., 0., 0., 0., 1., 1., 1., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 1., 1., 1., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
            ]).flatten().unsqueeze(0)

            observation_10 = torch.tensor([
                [-1., 0., 0., 0., 1., 1., 1., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 1., 1., 1., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 1., 1., 1., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 3., 3., 3., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 3., 3., 3., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
            ]).flatten().unsqueeze(0)
            observation_11 = torch.tensor([
                [-1., 0., 0., 0., 1., 1., 1., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 1., 1., 1., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 1., 1., 1., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 3., 3., 3., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
            ]).flatten().unsqueeze(0)
            observation = torch.cat((observation_0, observation_1, observation_2, observation_3, observation_4,
                                     observation_5, observation_6, observation_7, observation_8, observation_9,
                                     observation_10, observation_11), dim=0)
            agent_and_object_positions_tensor = get_position_and_object_positions_of_observation(obs=observation,
                                                                                                 maximum_number_of_objects=3,
                                                                                                 observation_width=9,
                                                                                                 observation_height=6,
                                                                                                 agent_size=2)
            self.assertTrue(
                torch.equal(agent_and_object_positions_tensor, torch.tensor([
                    [5., 1., 3., 3., 7., 3., 0., 0.],
                    [5., 1., 3., 2., 7., 2., 0., 0.],
                    [5., 1., 3., 1., 7., 1., 0., 0.],
                    [5., 1., 3., 0., 7., 0., 0., 0.],
                    [5., 1., 3., -1., 7., -1., 0., 0.],
                    [5., 1., 4., 3., 5., 3., 6., 3.],
                    [5., 1., 4., 2., 5., 2., 6., 2.],
                    [5., 1., 4., 1., 6., 1., 0., 0.],
                    [5., 1., 4., 0., 6., 0., 0., 0.],
                    [5., 1., 4., -1., 6., -1., 0., 0.],
                    [5., 1., 5., 3., 0., 0., 0., 0.],
                    [5., 1., 5., 2., 0., 0., 0., 0.],
                ])))

        with self.subTest("objects (dodge) are overlapping, agent size 2, more than maximum number of objects"):
            observation_0 = torch.tensor([
                [-1., 1., 1., 1., 3., 3., 3., 0., 3., 3., 3., -1.],
                [-1., 1., 1., 1., 3., 3., 3., 0., 3., 3., 3., -1.],
                [-1., 1., 1., 1., 3., 3., 3., 3., 3., 3., 3., -1.],
                [-1., 0., 0., 0., 0., 0., 3., 3., 3., 0., 0., -1.],
                [-1., 0., 0., 0., 3., 3., 3., 3., 3., 3., 3., -1.],
                [-1., 0., 0., 0., 3., 3., 3., 0., 3., 3., 3., -1.],
                [-1., 0., 0., 0., 3., 3., 3., 0., 3., 3., 3., -1.],
            ]).flatten().unsqueeze(0)

            observation_1 = torch.tensor([
                [-1., 1., 1., 1., 0., 3., 3., 3., 0., 0., 0., -1.],
                [-1., 1., 1., 1., 0., 3., 3., 3., 0., 0., 0., -1.],
                [-1., 1., 1., 1., 0., 3., 3., 3., 3., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 3., 3., 3., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 3., 3., 3., 3., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 3., 3., 3., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 3., 3., 3., 0., -1.],
            ]).flatten().unsqueeze(0)
            observation_2 = torch.tensor([
                [-1., 1., 1., 1., 0., 0., 3., 3., 3., 0., 0., -1.],
                [-1., 1., 1., 1., 0., 0., 3., 3., 3., 0., 0., -1.],
                [-1., 1., 1., 1., 0., 0., 3., 3., 3., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 3., 3., 3., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 3., 3., 3., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 3., 3., 3., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 3., 3., 3., 0., 0., -1.],
            ]).flatten().unsqueeze(0)
            observation_3 = torch.tensor([
                [-1., 1., 1., 1., 0., 0., 0., 3., 3., 3., 0., -1.],
                [-1., 1., 1., 1., 0., 0., 0., 3., 3., 3., 0., -1.],
                [-1., 1., 1., 1., 0., 0., 3., 3., 3., 3., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 3., 3., 3., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 3., 3., 3., 3., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 3., 3., 3., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 3., 3., 3., 0., 0., 0., -1.],
            ]).flatten().unsqueeze(0)

            observation_4 = torch.tensor([
                [-1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 1., 1., 1., 0., 0., 0., 3., 3., 3., 0., -1.],
                [-1., 1., 1., 1., 0., 0., 3., 3., 3., 3., 0., -1.],
                [-1., 0., 0., 0., 0., 3., 3., 3., 3., 3., 0., -1.],
                [-1., 0., 0., 0., 0., 3., 3., 3., 3., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 3., 3., 3., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
            ]).flatten().unsqueeze(0)
            observation_5 = torch.tensor([
                [-1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 1., 1., 1., 0., 3., 3., 3., 3., 3., 0., -1.],
                [-1., 0., 0., 0., 0., 3., 3., 3., 3., 3., 0., -1.],
                [-1., 0., 0., 0., 0., 3., 3., 3., 3., 3., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
            ]).flatten().unsqueeze(0)
            observation_6 = torch.tensor([
                [-1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 1., 1., 1., 0., 3., 3., 3., 0., 0., 0., -1.],
                [-1., 1., 1., 1., 0., 3., 3., 3., 3., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 3., 3., 3., 3., 3., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 3., 3., 3., 3., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 3., 3., 3., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
            ]).flatten().unsqueeze(0)
        observation = torch.cat((observation_0, observation_1, observation_2, observation_3, observation_4,
                                 observation_5, observation_6), dim=0)
        agent_and_object_positions_tensor = get_position_and_object_positions_of_observation(obs=observation,
                                                                                             maximum_number_of_objects=8,
                                                                                             observation_width=10,
                                                                                             observation_height=7,
                                                                                             agent_size=2)
        self.assertTrue(
            torch.equal(agent_and_object_positions_tensor, torch.tensor([
                [2., 1., 5., 1., 9., 1., 7., 3., 5., 5., 9., 5., 0., 0., 0., 0., 0., 0.],
                [2., 1., 6., 1., 7., 3., 8., 5., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [2., 1., 7., 1., 7., 2., 7., 3., 7., 4., 7., 5., 0., 0., 0., 0., 0., 0.],
                [2., 1., 8., 1., 7., 3., 6., 5., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [2., 1., 8., 2., 7., 3., 6., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [2., 1., 6., 3., 7., 3., 8., 3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [2., 1., 6., 2., 7., 3., 8., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            ])))

        with self.subTest("agent size not supported"):
            self.assertRaises(ValueError, get_position_and_object_positions_of_observation,
                              obs=torch.tensor([[-1., 0., 1., 0., -1]]), maximum_number_of_objects=10,
                              observation_width=10, observation_height=10, agent_size=3)

        with self.subTest("observation width and height does not match observation tensor"):
            self.assertRaises(ValueError, get_position_and_object_positions_of_observation,
                              obs=torch.tensor([[-1., 0., 1., 0., -1]]), maximum_number_of_objects=10,
                              observation_width=10, observation_height=10, agent_size=3)

    def test_get_observation_of_position_and_object_positions(self) -> None:
        with self.subTest("multiple agent and object position tensors, with x and y position out of boundaries"):
            observation = get_observation_of_position_and_object_positions(
                agent_and_object_positions=torch.tensor(
                    [[-1., 1., 0., 0.], [7., 1., 0., 0.], [4., -2., 0., 0.], [4., 10., 0., 0.]]),
                observation_height=5,
                observation_width=5, agent_size=2, task="dodge")

            tensor_0 = torch.tensor([
                [-1., 1., 1., 1., 0., 0., -1.],
                [-1., 1., 1., 1., 0., 0., -1.],
                [-1., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., -1.]]).flatten().unsqueeze(0)
            tensor_1 = torch.tensor([
                [-1., 0., 0., 1., 1., 1., -1.],
                [-1., 0., 0., 1., 1., 1., -1.],
                [-1., 0., 0., 1., 1., 1., -1.],
                [-1., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., -1.]]).flatten().unsqueeze(0)
            tensor_2 = torch.tensor([
                [-1., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 1., 1., 1., -1.],
                [-1., 0., 0., 1., 1., 1., -1.],
                [-1., 0., 0., 1., 1., 1., -1.]]).flatten().unsqueeze(0)

            self.assertTrue(torch.equal(observation, torch.cat((tensor_0, tensor_1, tensor_1, tensor_2), dim=0)))

        with self.subTest("dodge objects flying out or in"):
            observation = get_observation_of_position_and_object_positions(
                agent_and_object_positions=torch.tensor(
                    [[5., 1., 2., 1.], [5., 1., 2., 0.], [5., 1., 2., -1.], [5., 1., 2., -2.],
                     [5., 1., 2., 8.], [5., 1., 2., 7.], [5., 1., 2., 6.], [5., 1., 2., 5.]]),
                observation_height=7,
                observation_width=6, agent_size=2, task="dodge")

            tensor_0 = torch.tensor([
                [-1., 3., 3., 3., 1., 1., 1., -1.],
                [-1., 3., 3., 3., 1., 1., 1., -1.],
                [-1., 3., 3., 3., 1., 1., 1., -1.],
                [-1., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., -1.]]).flatten().unsqueeze(0)
            tensor_1 = torch.tensor([
                [-1., 3., 3., 3., 1., 1., 1., -1.],
                [-1., 3., 3., 3., 1., 1., 1., -1.],
                [-1., 0., 0., 0., 1., 1., 1., -1.],
                [-1., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., -1.]]).flatten().unsqueeze(0)
            tensor_2 = torch.tensor([
                [-1., 3., 3., 3., 1., 1., 1., -1.],
                [-1., 0., 0., 0., 1., 1., 1., -1.],
                [-1., 0., 0., 0., 1., 1., 1., -1.],
                [-1., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., -1.]]).flatten().unsqueeze(0)
            tensor_3 = torch.tensor([
                [-1., 0., 0., 0., 1., 1., 1., -1.],
                [-1., 0., 0., 0., 1., 1., 1., -1.],
                [-1., 0., 0., 0., 1., 1., 1., -1.],
                [-1., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., -1.]]).flatten().unsqueeze(0)
            tensor_4 = torch.tensor([
                [-1., 0., 0., 0., 1., 1., 1., -1.],
                [-1., 0., 0., 0., 1., 1., 1., -1.],
                [-1., 0., 0., 0., 1., 1., 1., -1.],
                [-1., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 3., 3., 3., 0., 0., 0., -1.]]).flatten().unsqueeze(0)
            tensor_5 = torch.tensor([
                [-1., 0., 0., 0., 1., 1., 1., -1.],
                [-1., 0., 0., 0., 1., 1., 1., -1.],
                [-1., 0., 0., 0., 1., 1., 1., -1.],
                [-1., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 3., 3., 3., 0., 0., 0., -1.],
                [-1., 3., 3., 3., 0., 0., 0., -1.]]).flatten().unsqueeze(0)
            tensor_6 = torch.tensor([
                [-1., 0., 0., 0., 1., 1., 1., -1.],
                [-1., 0., 0., 0., 1., 1., 1., -1.],
                [-1., 0., 0., 0., 1., 1., 1., -1.],
                [-1., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 3., 3., 3., 0., 0., 0., -1.],
                [-1., 3., 3., 3., 0., 0., 0., -1.],
                [-1., 3., 3., 3., 0., 0., 0., -1.]]).flatten().unsqueeze(0)

            self.assertTrue(torch.equal(observation, torch.cat(
                (tensor_0, tensor_1, tensor_2, tensor_3, tensor_3, tensor_4, tensor_5, tensor_6), dim=0)))

        with self.subTest("overlapping object with agent"):
            observation = get_observation_of_position_and_object_positions(
                agent_and_object_positions=torch.tensor(
                    [[4., 1., 2., -1.], [4., 1., 3., -1.], [4., 1., 4., -1.], [4., 1., 5., -1.], [4., 1., 6., -1.],
                     [4., 1., 2., 0.], [4., 1., 3., 0.], [4., 1., 4., 0.], [4., 1., 5., 0.], [4., 1., 6., 0.],
                     [4., 1., 2., 1.], [4., 1., 3., 1.], [4., 1., 4., 1.], [4., 1., 5., 1.], [4., 1., 6., 1.],
                     [4., 1., 2., 2.], [4., 1., 3., 2.], [4., 1., 4., 2.], [4., 1., 5., 2.], [4., 1., 6., 2.],
                     [4., 1., 2., 3.], [4., 1., 3., 3.], [4., 1., 4., 3.], [4., 1., 5., 3.], [4., 1., 6., 3.]
                     ]),
                observation_height=7,
                observation_width=7, agent_size=2, task="dodge")

            tensor_0 = torch.tensor([
                [-1., 3., 3., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 0., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 0., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.]]).flatten().unsqueeze(0)
            tensor_1 = torch.tensor([
                [-1., 0., 3., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 0., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 0., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.]]).flatten().unsqueeze(0)
            tensor_2 = torch.tensor([
                [-1., 0., 0., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 0., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 0., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.]]).flatten().unsqueeze(0)
            tensor_3 = torch.tensor([
                [-1., 0., 0., 1., 1., 1., 3., 0., -1.],
                [-1., 0., 0., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 0., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.]]).flatten().unsqueeze(0)
            tensor_4 = torch.tensor([
                [-1., 0., 0., 1., 1., 1., 3., 3., -1.],
                [-1., 0., 0., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 0., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.]]).flatten().unsqueeze(0)
            tensor_5 = torch.tensor([
                [-1., 3., 3., 1., 1., 1., 0., 0., -1.],
                [-1., 3., 3., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 0., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.]]).flatten().unsqueeze(0)
            tensor_6 = torch.tensor([
                [-1., 0., 3., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 3., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 0., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.]]).flatten().unsqueeze(0)
            tensor_7 = torch.tensor([
                [-1., 0., 0., 1., 1., 1., 3., 0., -1.],
                [-1., 0., 0., 1., 1., 1., 3., 0., -1.],
                [-1., 0., 0., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.]]).flatten().unsqueeze(0)
            tensor_8 = torch.tensor([
                [-1., 0., 0., 1., 1., 1., 3., 3., -1.],
                [-1., 0., 0., 1., 1., 1., 3., 3., -1.],
                [-1., 0., 0., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.]]).flatten().unsqueeze(0)
            tensor_9 = torch.tensor([
                [-1., 3., 3., 1., 1., 1., 0., 0., -1.],
                [-1., 3., 3., 1., 1., 1., 0., 0., -1.],
                [-1., 3., 3., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.]]).flatten().unsqueeze(0)
            tensor_10 = torch.tensor([
                [-1., 0., 3., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 3., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 3., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.]]).flatten().unsqueeze(0)
            tensor_11 = torch.tensor([
                [-1., 0., 0., 1., 1., 1., 3., 0., -1.],
                [-1., 0., 0., 1., 1., 1., 3., 0., -1.],
                [-1., 0., 0., 1., 1., 1., 3., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.]]).flatten().unsqueeze(0)
            tensor_12 = torch.tensor([
                [-1., 0., 0., 1., 1., 1., 3., 3., -1.],
                [-1., 0., 0., 1., 1., 1., 3., 3., -1.],
                [-1., 0., 0., 1., 1., 1., 3., 3., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.]]).flatten().unsqueeze(0)
            tensor_13 = torch.tensor([
                [-1., 0., 0., 1., 1., 1., 0., 0., -1.],
                [-1., 3., 3., 1., 1., 1., 0., 0., -1.],
                [-1., 3., 3., 1., 1., 1., 0., 0., -1.],
                [-1., 3., 3., 3., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.]]).flatten().unsqueeze(0)
            tensor_14 = torch.tensor([
                [-1., 0., 0., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 3., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 3., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 3., 3., 3., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.]]).flatten().unsqueeze(0)
            tensor_15 = torch.tensor([
                [-1., 0., 0., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 0., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 0., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 0., 3., 3., 3., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.]]).flatten().unsqueeze(0)
            tensor_16 = torch.tensor([
                [-1., 0., 0., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 0., 1., 1., 1., 3., 0., -1.],
                [-1., 0., 0., 1., 1., 1., 3., 0., -1.],
                [-1., 0., 0., 0., 3., 3., 3., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.]]).flatten().unsqueeze(0)
            tensor_17 = torch.tensor([
                [-1., 0., 0., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 0., 1., 1., 1., 3., 3., -1.],
                [-1., 0., 0., 1., 1., 1., 3., 3., -1.],
                [-1., 0., 0., 0., 0., 3., 3., 3., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.]]).flatten().unsqueeze(0)
            tensor_18 = torch.tensor([
                [-1., 0., 0., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 0., 1., 1., 1., 0., 0., -1.],
                [-1., 3., 3., 1., 1., 1., 0., 0., -1.],
                [-1., 3., 3., 3., 0., 0., 0., 0., -1.],
                [-1., 3., 3., 3., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.]]).flatten().unsqueeze(0)
            tensor_19 = torch.tensor([
                [-1., 0., 0., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 0., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 3., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 3., 3., 3., 0., 0., 0., -1.],
                [-1., 0., 3., 3., 3., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.]]).flatten().unsqueeze(0)
            tensor_20 = torch.tensor([
                [-1., 0., 0., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 0., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 0., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 0., 3., 3., 3., 0., 0., -1.],
                [-1., 0., 0., 3., 3., 3., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.]]).flatten().unsqueeze(0)
            tensor_21 = torch.tensor([
                [-1., 0., 0., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 0., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 0., 1., 1., 1., 3., 0., -1.],
                [-1., 0., 0., 0., 3., 3., 3., 0., -1.],
                [-1., 0., 0., 0., 3., 3., 3., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.]]).flatten().unsqueeze(0)
            tensor_22 = torch.tensor([
                [-1., 0., 0., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 0., 1., 1., 1., 0., 0., -1.],
                [-1., 0., 0., 1., 1., 1., 3., 3., -1.],
                [-1., 0., 0., 0., 0., 3., 3., 3., -1.],
                [-1., 0., 0., 0., 0., 3., 3., 3., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 0., 0., -1.]]).flatten().unsqueeze(0)

            self.assertTrue(torch.equal(observation, torch.cat((tensor_0, tensor_1, tensor_2, tensor_3, tensor_4,
                                                                tensor_5, tensor_6, tensor_2, tensor_7, tensor_8,
                                                                tensor_9, tensor_10, tensor_2, tensor_11, tensor_12,
                                                                tensor_13, tensor_14, tensor_15, tensor_16, tensor_17,
                                                                tensor_18, tensor_19, tensor_20, tensor_21, tensor_22),
                                                               dim=0)))

        with self.subTest("collect object, different agent size"):
            observation = get_observation_of_position_and_object_positions(
                agent_and_object_positions=torch.tensor([[3., 2., 8., 7.]]), observation_height=10,
                observation_width=10, agent_size=3, task="collect")
            self.assertTrue(torch.equal(observation, torch.tensor([
                [-1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., -1.],
                [-1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., -1.],
                [-1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., -1.],
                [-1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., -1.],
                [-1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., -1.],
                [-1., 0., 0., 0., 0., 0., 2., 2., 2., 2., 2., -1.],
                [-1., 0., 0., 0., 0., 0., 2., 2., 2., 2., 2., -1.],
                [-1., 0., 0., 0., 0., 0., 2., 2., 2., 2., 2., -1.],
                [-1., 0., 0., 0., 0., 0., 2., 2., 2., 2., 2., -1.],
                [-1., 0., 0., 0., 0., 0., 2., 2., 2., 2., 2., -1.]]).flatten().unsqueeze(0)))

        with self.subTest("task not implemented"):
            self.assertRaises(ValueError, get_observation_of_position_and_object_positions,
                              agent_and_object_positions=torch.tensor([[4., 1.]]), observation_height=5,
                              observation_width=5, agent_size=2, task="bla")

    def test_get_next_position_observation_moonlander(self) -> None:
        with self.subTest("action movement agent size 1"):
            new_positions = get_next_position_observation_moonlander(
                observations=torch.tensor([
                    [5., 0.],  # left
                    [1., 0.],  # left in wall
                    [5., 0.],  # stay
                    [5., 0.],  # right
                    [40., 0.],  # right in wall
                ]),
                observation_width=40, actions=torch.tensor([0, 0, 1, 2, 2]), agent_size=1)
            self.assertTrue(torch.equal(new_positions, torch.tensor([
                [4., 0.],
                [1., 0.],
                [5., 0.],
                [6., 0.],
                [40., 0.]])))
        with self.subTest("action movement agent size 2"):
            new_positions = get_next_position_observation_moonlander(
                observations=torch.tensor([
                    [5., 1.],  # left
                    [2., 1.],  # left in wall
                    [5., 1.],  # stay
                    [5., 1.],  # right
                    [39., 1.],  # right in wall
                ]),
                observation_width=40, actions=torch.tensor([0, 0, 1, 2, 2]), agent_size=2)
            self.assertTrue(torch.equal(new_positions, torch.tensor([
                [3., 1.],
                [2., 1.],
                [5., 1.],
                [7., 1.],
                [39., 1.]])))
        with self.subTest("objects movement size 1"):
            new_positions = get_next_position_observation_moonlander(
                # object flying out, object at last position, normal object, object flying in
                observations=torch.tensor([[5., 1., 1., 0., 3., 1., 5., 2., 7., 30.]]),
                actions=torch.tensor([0]),
                observation_width=40,
                agent_size=1)
            self.assertTrue(torch.equal(new_positions, torch.tensor([[4., 1., 0., 0., 3., 0., 5., 1., 7., 29.]])))
        with self.subTest("objects movement size 2"):
            new_positions = get_next_position_observation_moonlander(
                # object flying out, object at last position, normal object, object flying in
                observations=torch.tensor([[5., 1., 1., -1., 3., 0., 5., 1., 7., 30.]]),
                actions=torch.tensor([0]),
                observation_width=40,
                agent_size=2)
            self.assertTrue(torch.equal(new_positions, torch.tensor([[3., 1., 0., 0., 3., -1., 5., 0., 7., 29.]])))
        with self.subTest("objects movement size 3"):
            new_positions = get_next_position_observation_moonlander(
                # object flying out, object at last position, normal object, object flying in
                observations=torch.tensor([[6., 1., 1., -2., 3., -1., 5., 0., 7., 30.]]),
                actions=torch.tensor([0]),
                observation_width=40,
                agent_size=3)
            self.assertTrue(torch.equal(new_positions, torch.tensor([[3., 1., 0., 0., 3., -2., 5., -1., 7., 29.]])))

    def test_calculate_prediction_error(self) -> None:
        forward_model_prediction_normal_distribution = torch.distributions.Normal(torch.tensor([[1., 1.]]),
                                                                                  scale=torch.ones(2))
        with self.subTest("perfect prediction"):
            prediction_error = calculate_prediction_error(
                env_name="MoonlanderWorldEnv",
                next_obs_positions=torch.tensor([[1., 1.]]),
                forward_model_prediction_normal_distribution=forward_model_prediction_normal_distribution,
                first_possible_x_position=1,
                last_possible_x_position=9)
            self.assertEqual(prediction_error, 0)

        with self.subTest("medium bad prediction"):
            prediction_error = calculate_prediction_error(
                env_name="MoonlanderWorldEnv",
                next_obs_positions=torch.tensor([[5., 1.]]),
                forward_model_prediction_normal_distribution=forward_model_prediction_normal_distribution,
                first_possible_x_position=1,
                last_possible_x_position=9)
            self.assertEqual(prediction_error, 0.5)

        with self.subTest("worst prediction"):
            prediction_error = calculate_prediction_error(
                env_name="MoonlanderWorldEnv",
                next_obs_positions=torch.tensor([[9., 1.]]),
                forward_model_prediction_normal_distribution=forward_model_prediction_normal_distribution,
                first_possible_x_position=1,
                last_possible_x_position=9)
            self.assertEqual(prediction_error, 1)

        with self.subTest("other env than MoonlanderWorldEnv"):
            self.assertRaises(ValueError, calculate_prediction_error,
                              env_name="bla",
                              next_obs_positions=torch.tensor([]),
                              forward_model_prediction_normal_distribution=torch.tensor([]),
                              first_possible_x_position=0,
                              last_possible_x_position=0)

    def test_calculate_need_for_control(self) -> None:
        register_custom_envs()
        env = gym.make("MoonlanderWorld-dodge-gaussian-v0")
        dummy_vec_env = DummyVecEnv([lambda: env])

        policy = Agent(env=dummy_vec_env, reward_predicting=True, model_based=False)
        fm_network = ProbabilisticForwardNetPositionPredictionIncludingReward(env=dummy_vec_env,
                                                                              cfg={'hidden_size': 256,
                                                                                   'learning_rate': 0.001,
                                                                                   'reward_eta': 0.2},
                                                                              maximum_number_of_objects=10)

        tmp_path = "/tmp/sb3_log/"
        logger = configure(tmp_path, ["stdout", "csv"])

        with self.subTest("other env than MoonlanderWorldEnv"):
            test_env = gym.make("MoonlanderWorld-dodge-gaussian-v0")
            test_env.name = "bla"
            test_dummy_vec_env = DummyVecEnv([lambda: test_env])
            self.assertRaises(ValueError, calculate_need_for_control,
                              env=test_dummy_vec_env,
                              policy=policy,
                              fm_network=fm_network,
                              logger=logger,
                              position_predicting=True,
                              prediction_error=0,
                              maximum_number_of_objects=10)

        with self.subTest("not position predicting"):
            self.assertRaises(NotImplementedError, calculate_need_for_control,
                              env=dummy_vec_env,
                              policy=policy,
                              fm_network=fm_network,
                              logger=logger,
                              position_predicting=False,
                              prediction_error=0,
                              maximum_number_of_objects=10)

    @mock.patch("src.custom_algorithms.cleanppofm.agent.Agent")
    def test_calculate_need_for_control_policy_mock(self, policy_mock) -> None:
        policy_mock.get_action_and_value_and_forward_model_prediction.side_effect = (
                [(torch.tensor([[1]]), None, None, None, None)] * 15 +
                [(torch.tensor([[2]]), None, None, None, None)] +
                [(torch.tensor([[2]]), None, None, None, None)] * 2 +
                [(torch.tensor([[1]]), None, None, None, None)])

        register_custom_envs()
        env = gym.make("MoonlanderWorld-dodge-gaussian-v0")
        dummy_vec_env = DummyVecEnv([lambda: env])

        fm_network = ProbabilisticForwardNetPositionPredictionIncludingReward(env=dummy_vec_env,
                                                                              cfg={'hidden_size': 256,
                                                                                   'learning_rate': 0.001,
                                                                                   'reward_eta': 0.2},
                                                                              maximum_number_of_objects=10)

        tmp_path = "/tmp/sb3_log/"
        logger = configure(tmp_path, ["stdout", "csv"])

        # build empty obs
        matrix = np.zeros(shape=(30, 40 + 2), dtype=np.int16)
        # add wall
        matrix[:, 0] = -1
        matrix[:, -1] = -1

        matrix[0:3, 5:8] = 1

        matrix_copy_0 = copy.deepcopy(matrix)

        with self.subTest("empty observation"):
            need_for_control, summed_up_reward_default = calculate_need_for_control(
                env=dummy_vec_env,
                policy=policy_mock,
                fm_network=fm_network,
                logger=logger,
                position_predicting=True,
                prediction_error=0,
                maximum_number_of_objects=10,
                last_observation_state=torch.tensor([matrix.flatten()])
            )
            self.assertEqual(need_for_control, 0)
            self.assertEqual(summed_up_reward_default, (
                    0.5 + 0.5 + 0.5 + 0.5 + 0.5 + 0.5 + 0.5 + 0.5 + 0.5 + 0.5 + 0.5 + 0.5 + 0.5 + 0.5 + 0.5) / 15)

        with self.subTest("difference in rewards"):
            matrix_copy_0[3:6, 3:6] = 3

            need_for_control, summed_up_reward_default = calculate_need_for_control(
                env=dummy_vec_env,
                policy=policy_mock,
                fm_network=fm_network,
                logger=logger,
                position_predicting=True,
                # trajectory lengths of one
                prediction_error=14 / 15,
                maximum_number_of_objects=10,
                last_observation_state=torch.tensor([matrix_copy_0.flatten()])
            )
            # 2 (0.19230769230769232) is the absolute (relative) reward when not crashing but being near the object
            # (optimal policy)
            # 0 is the reward when crashing (default policy)
            self.assertEqual(need_for_control, (0.5 * (2 - (-3))) / 13)
            self.assertEqual(summed_up_reward_default, 0)

        with self.subTest("difference in rewards, multiple steps"):
            matrix_copy_0[3:6, 3:6] = 3

            need_for_control, summed_up_reward_default = calculate_need_for_control(
                env=dummy_vec_env,
                policy=policy_mock,
                fm_network=fm_network,
                logger=logger,
                position_predicting=True,
                # trajectory lengths of two
                prediction_error=13 / 15,
                maximum_number_of_objects=10,
                last_observation_state=torch.tensor([matrix_copy_0.flatten()])
            )
            # 2 (0.19230769230769232) and 10 (0.5) are the absolute (relative) reward
            # when not crashing but being near the object (optimal policy)
            # 0 is the reward when crashing (default policy) --> 2 times because in dodge the object does not disappear
            self.assertEqual(need_for_control, (((0.5 * (2 - (-3))) / 13) + ((0.5 * (10 - (-3))) / 13)) / 2)
            self.assertEqual(summed_up_reward_default, 2 * 0)

        with self.subTest("last observation not given"):
            need_for_control, summed_up_reward_default = calculate_need_for_control(
                env=dummy_vec_env,
                policy=policy_mock,
                fm_network=fm_network,
                logger=logger,
                position_predicting=True,
                # trajectory lengths of one
                prediction_error=14 / 15,
                maximum_number_of_objects=10
            )
            # starting a new env --> no objects in the first step
            self.assertEqual(need_for_control, 0)
            self.assertEqual(summed_up_reward_default, 0.5)

    def test_normalize_rewards(self) -> None:
        with self.subTest("dodge"):
            task = "dodge"
            with self.subTest(">10 should raise ValueError"):
                self.assertRaises(ValueError, normalize_rewards, task=task, absolute_reward=11)
            with self.subTest("10 should be 0.5"):
                normalized_reward = normalize_rewards(task=task, absolute_reward=10)
                self.assertEqual(normalized_reward, 0.5)
            with self.subTest("the possible half of the absolute rewards (3.5) should be 0.25"):
                normalized_reward = normalize_rewards(task=task, absolute_reward=3.5)
                self.assertEqual(normalized_reward, 0.25)
            with self.subTest("-3 should be 0"):
                normalized_reward = normalize_rewards(task=task, absolute_reward=-3)
                self.assertEqual(normalized_reward, 0)
            with self.subTest("<-3 should be 0"):
                normalized_reward = normalize_rewards(task=task, absolute_reward=-100000)
                self.assertEqual(normalized_reward, 0)

        with self.subTest("collect"):
            task = "collect"
            with self.subTest(">62 should be 1"):
                normalized_reward = normalize_rewards(task=task, absolute_reward=1000)
                self.assertEqual(normalized_reward, 1)
            with self.subTest("62 should be 1"):
                normalized_reward = normalize_rewards(task=task, absolute_reward=62)
                self.assertEqual(normalized_reward, 1)
            with self.subTest("the possible half of the absolute rewards (41) should be 0.75"):
                normalized_reward = normalize_rewards(task=task, absolute_reward=31)
                self.assertEqual(normalized_reward, 0.75)
            with self.subTest("0 should be 0.5"):
                normalized_reward = normalize_rewards(task=task, absolute_reward=0)
                self.assertEqual(normalized_reward, 0.5)
            with self.subTest("<0 should be 0.5"):
                normalized_reward = normalize_rewards(task=task, absolute_reward=-100000)
                self.assertEqual(normalized_reward, 0.5)

        with self.subTest("task not implemented"):
            self.assertRaises(NotImplementedError, normalize_rewards, task="blablabla", absolute_reward=0)

    def test_get_collected_objects(self) -> None:
        with self.subTest("didn't collect anything"):
            collected_objects = get_collected_objects(observation_positions=torch.tensor([[2., 1., 3., 1., 2., 2.]]),
                                                      agent_size=1,
                                                      observation_width=5)
            self.assertListEqual(collected_objects, [])

        with self.subTest("collect an object, agent size 1"):
            collected_objects = get_collected_objects(observation_positions=torch.tensor([[2., 1., 2., 1.]]),
                                                      agent_size=1,
                                                      observation_width=5)
            self.assertListEqual(collected_objects, [{"x": 2, "y": 1, "size": 1}])

        with self.subTest("collect multiple objects, agent size 2"):
            collected_objects = get_collected_objects(
                observation_positions=torch.tensor([[5., 1.,
                                                     2., -1., 3., -1., 4., -1., 5., -1., 6., -1., 7., -1., 8., -1.,
                                                     2., 0., 3., 0., 4., 0., 5., 0., 6., 0., 7., 0., 8., 0.,
                                                     2., 1., 3., 1., 4., 1., 5., 1., 6., 1., 7., 1., 8., 1.,
                                                     2., 2., 3., 2., 4., 2., 5., 2., 6., 2., 7., 2., 8., 2.,
                                                     2., 3., 3., 3., 4., 3., 5., 3., 6., 3., 7., 3., 8., 3.,
                                                     2., 4., 3., 4., 4., 4., 5., 4., 6., 4., 7., 4., 8., 4.]]),
                agent_size=2,
                observation_width=10)
            self.assertListEqual(collected_objects, [
                {"x": 3, "y": -1, "size": 2},
                {"x": 4, "y": -1, "size": 2},
                {"x": 5, "y": -1, "size": 2},
                {"x": 6, "y": -1, "size": 2},
                {"x": 7, "y": -1, "size": 2},
                {"x": 3, "y": 0, "size": 2},
                {"x": 4, "y": 0, "size": 2},
                {"x": 5, "y": 0, "size": 2},
                {"x": 6, "y": 0, "size": 2},
                {"x": 7, "y": 0, "size": 2},
                {"x": 3, "y": 1, "size": 2},
                {"x": 4, "y": 1, "size": 2},
                {"x": 5, "y": 1, "size": 2},
                {"x": 6, "y": 1, "size": 2},
                {"x": 7, "y": 1, "size": 2},
                {"x": 3, "y": 2, "size": 2},
                {"x": 4, "y": 2, "size": 2},
                {"x": 5, "y": 2, "size": 2},
                {"x": 6, "y": 2, "size": 2},
                {"x": 7, "y": 2, "size": 2},
                {"x": 3, "y": 3, "size": 2},
                {"x": 4, "y": 3, "size": 2},
                {"x": 5, "y": 3, "size": 2},
                {"x": 6, "y": 3, "size": 2},
                {"x": 7, "y": 3, "size": 2},
            ])

        with self.subTest("agent size > 2 not supported"):
            self.assertRaises(NotImplementedError, get_collected_objects,
                              observation_positions=torch.tensor([[2., 1.]]), agent_size=3, observation_width=5)

    def test_calculate_trajectory_length(self) -> None:
        observation_height = 10
        with self.subTest("prediction error of zero"):
            trajectory_length = calculate_trajectory_length(observation_height=observation_height, prediction_error=0)
            self.assertEqual(trajectory_length, observation_height / 2)
        with self.subTest("prediction error of 0.5"):
            trajectory_length = calculate_trajectory_length(observation_height=observation_height, prediction_error=0.5)
            self.assertEqual(trajectory_length, observation_height / 4)
        with self.subTest("prediction error of 1"):
            trajectory_length = calculate_trajectory_length(observation_height=observation_height, prediction_error=1)
            self.assertEqual(trajectory_length, 0)

    def test_get_next_normalized_reward(self) -> None:
        # build empty obs
        matrix = np.zeros(shape=(30, 40 + 2), dtype=np.int16)
        # add wall
        matrix[:, 0] = -1
        matrix[:, -1] = -1

        matrix[0:3, 5:8] = 1

        matrix_copy_0 = copy.deepcopy(matrix)
        matrix_copy_1 = copy.deepcopy(matrix)
        matrix_copy_2 = copy.deepcopy(matrix)
        matrix_copy_3 = copy.deepcopy(matrix)
        matrix_copy_4 = copy.deepcopy(matrix)

        matrix = torch.tensor(matrix.flatten()).unsqueeze(0)
        with self.subTest("empty observation"):
            normalized_reward, new_state = get_next_normalized_reward(last_observation_state=matrix,
                                                                      action=torch.tensor([1]),
                                                                      maximum_number_of_objects=10,
                                                                      observation_width=40,
                                                                      observation_height=30, agent_size=2,
                                                                      task="dodge",
                                                                      task_type="obstacle")
            self.assertEqual(normalized_reward, 0.5)
            np.testing.assert_array_equal(new_state, matrix)
            normalized_reward, new_state = get_next_normalized_reward(last_observation_state=matrix,
                                                                      action=torch.tensor([1]),
                                                                      maximum_number_of_objects=10,
                                                                      observation_width=40,
                                                                      observation_height=30, agent_size=2,
                                                                      task="collect",
                                                                      task_type="coin")
            self.assertEqual(normalized_reward, 0.5)
            np.testing.assert_array_equal(new_state, matrix)

        with (self.subTest("crashing/collecting object")):
            # state where agent is one step before crashing
            matrix_copy_0[3:6, 3:6] = 3
            matrix_copy_1[3:6, 3:6] = 2
            matrix_copy_0 = torch.tensor(matrix_copy_0.flatten()).unsqueeze(0)
            matrix_copy_1 = torch.tensor(matrix_copy_1.flatten()).unsqueeze(0)

            # only two elements for the first row, because the agent is overlapping
            matrix_copy_2[2, 3:5] = 3
            matrix_copy_2[3:5, 3:6] = 3
            matrix_copy_2 = matrix_copy_2.flatten()
            matrix_copy_2 = np.expand_dims(matrix_copy_2.astype(np.float), axis=0)

            matrix_copy_3[2, 3:5] = 2
            matrix_copy_3[3:5, 3:6] = 2
            matrix_copy_3 = matrix_copy_3.flatten()
            matrix_copy_3 = np.expand_dims(matrix_copy_3.astype(np.float), axis=0)

            normalized_reward, new_state = get_next_normalized_reward(last_observation_state=matrix_copy_0,
                                                                      action=torch.tensor([1]),
                                                                      maximum_number_of_objects=10,
                                                                      observation_width=40,
                                                                      observation_height=30, agent_size=2,
                                                                      task="dodge",
                                                                      task_type="obstacle")
            self.assertEqual(normalized_reward, 0)
            np.testing.assert_array_equal(new_state, matrix_copy_2)

            normalized_reward, new_state = get_next_normalized_reward(last_observation_state=matrix_copy_1,
                                                                      action=torch.tensor([1]),
                                                                      maximum_number_of_objects=10,
                                                                      observation_width=40,
                                                                      observation_height=30, agent_size=2,
                                                                      task="collect",
                                                                      task_type="coin")
            self.assertEqual(normalized_reward, 1)
            np.testing.assert_array_equal(new_state, matrix_copy_3)

        with self.subTest("collect object --> object disappears"):
            matrix_copy_4[2, 3:5] = 2
            matrix_copy_4[3:5, 3:6] = 2
            matrix_copy_4 = torch.tensor(matrix_copy_4.flatten()).unsqueeze(0)

            normalized_reward, new_state = get_next_normalized_reward(last_observation_state=matrix_copy_4,
                                                                      action=torch.tensor([1]),
                                                                      maximum_number_of_objects=10,
                                                                      observation_width=40,
                                                                      observation_height=30, agent_size=2,
                                                                      task="collect",
                                                                      task_type="coin")
            self.assertEqual(normalized_reward, 0.5)
            np.testing.assert_array_equal(new_state, matrix)

        with self.subTest("state does not match observation_width & observation_height"):
            self.assertRaises(ValueError, get_next_normalized_reward, last_observation_state=matrix,
                              action=torch.tensor([1]), maximum_number_of_objects=10, observation_width=10,
                              observation_height=10, agent_size=2, task="dodge", task_type="obstacle")

        with self.subTest("task or task type does not exists or do not match"):
            self.assertRaises(NotImplementedError, get_next_normalized_reward, last_observation_state=matrix,
                              action=torch.tensor([1]), maximum_number_of_objects=10, observation_width=40,
                              observation_height=30, agent_size=2, task="bla", task_type="obstacle")
            self.assertRaises(NotImplementedError, get_next_normalized_reward, last_observation_state=matrix,
                              action=torch.tensor([1]), maximum_number_of_objects=10, observation_width=40,
                              observation_height=30, agent_size=2, task="dodge", task_type="bla")
            self.assertRaises(NotImplementedError, get_next_normalized_reward, last_observation_state=matrix,
                              action=torch.tensor([1]), maximum_number_of_objects=10, observation_width=40,
                              observation_height=30, agent_size=2, task="dodge", task_type="coin")
            self.assertRaises(NotImplementedError, get_next_normalized_reward, last_observation_state=matrix,
                              action=torch.tensor([1]), maximum_number_of_objects=10, observation_width=40,
                              observation_height=30, agent_size=2, task="collect", task_type="obstacle")


if __name__ == '__main__':
    unittest.main()
