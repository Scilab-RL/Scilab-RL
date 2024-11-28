import unittest
import torch
from src.custom_algorithms.cleanppofm.utils import (get_summed_up_reward_of_env_or_fm_with_predicted_states_of_fm, \
                                                    get_position_and_object_positions_of_observation,
                                                    get_next_whole_observation,
                                                    get_observation_of_position_and_object_positions, \
                                                    get_next_position_observation_moonlander,
                                                    calculate_prediction_error,
                                                    calculate_need_for_control,
                                                    normalize_rewards)


class TestUtils(unittest.TestCase):

    def test_summed_up_reward_of_env_or_fm_with_predicted_states_of_fm(self) -> None:
        pass

    def test_get_position_and_object_positions_of_observation(self) -> None:
        pass

    def test_get_next_whole_observation(self) -> None:
        pass

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
        pass

    def test_calculate_need_for_control(self) -> None:
        pass

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


if __name__ == '__main__':
    unittest.main()
