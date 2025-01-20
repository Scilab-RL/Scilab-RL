from typing import List, Dict
import numpy as np
from gymnasium import spaces, Env
from src.custom_envs.moonlander.moonlander_env import MoonlanderWorldEnv
from src.custom_algorithms.cleanppofm.utils import get_next_position_observation_moonlander


class ModelbasedMoonlanderEnv(Env):
    def __init__(self, task: str = "dodge", reward_function: str = "pos_neg",
                 list_of_object_dict_lists: List[Dict] = None):
        self.name = "MoonlanderWorldEnv"
        self.actual_environment = MoonlanderWorldEnv(task=task, reward_function=reward_function,
                                                     list_of_object_dict_lists=list_of_object_dict_lists)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-10,
            high=3,
            # Four observations packed into one: current, and the prediction for left/stay/right
            shape=(4, self.actual_environment.observation_space.shape[0]),
            dtype=np.float64,
        )
        # reward_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
        self.observation_space = spaces.Dict(
            {
                "observations": self.observation_space,
                # "rewards": reward_space
            }
        )

        self.current_observation = {
            "observations": np.array(
                [
                    np.zeros(self.observation_space.shape),
                    np.zeros(self.observation_space.shape),
                    np.zeros(self.observation_space.shape),
                    np.zeros(self.observation_space.shape),
                ]
            ),
            # "rewards": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        }

    def step(self, action: int):
        current_observation_matrix, reward, is_done, truncated, info = self.actual_environment.step(action)

        # new predictions
        observation_predictions = []
        # reward_predictions = []

        # get predictions
        if not self.actual_environment.is_done():
            for action in [0, 1, 2]:
                predicted_observation = get_next_position_observation_moonlander(
                    observations=current_observation_matrix,
                    actions=action,
                    observation_width=self.actual_environment.observation_width,
                    observation_height=self.actual_environment.observation_height,
                    agent_size=self.actual_environment.agent_size,
                    maximum_number_of_objects=self.actual_environment.maximum_number_of_objects)
                observation_predictions.append(predicted_observation)
                # predicted_reward = ...
                # reward_predictions.append(predicted_reward)
        else:
            observation_predictions = [
                np.zeros(self.current_observation["observations"][0].shape),
                np.zeros(self.current_observation["observations"][0].shape),
                np.zeros(self.current_observation["observations"][0].shape),
            ]
            reward_predictions = [0.0, 0.0, 0.0]

        self.current_observation["observations"] = np.array(
            [
                current_observation_matrix,
                observation_predictions[0],
                observation_predictions[1],
                observation_predictions[2],
            ]
        )
        # self.current_observation["rewards"] = np.array(
        #     reward_predictions, dtype=np.float32
        # )

        return self.current_observation, reward, is_done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        current_observation = self.actual_environment.reset()
        self.current_observation = {
            "observations": np.array(
                [
                    current_observation,
                    np.zeros(current_observation.shape),
                    np.zeros(current_observation.shape),
                    np.zeros(current_observation.shape),
                ]
            ),
            # "rewards": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        }

        # set placeholder for info
        return self.current_observation, {"simple": 0, "gaussian": 0, "pos_neg": 0,
                                          "number_of_crashed_or_collected_objects": 0}
