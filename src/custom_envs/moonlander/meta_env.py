import copy
import os
import sys

import gymnasium as gym
import numpy as np

# FIXME: needed for rendering rgb array
np.set_printoptions(threshold=sys.maxsize)
import yaml
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

from src.custom_envs.moonlander.moonlander_env import MoonlanderWorldEnv


class MetaEnv(gym.Env):
    render_mode = None
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(self, reward_function: str = "pos_neg"):
        self.ROOT_DIR = "."
        config_path_dodge_asteroids = os.path.join(os.path.dirname(os.path.realpath(__file__)), "standard_config.yaml")
        config_path_collect_asteroids = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                     "standard_config_second_task.yaml")

        with open(config_path_dodge_asteroids, "r") as file:
            config_dodge_asteroids = yaml.safe_load(file)
        with open(config_path_collect_asteroids, "r") as file:
            config_collect_asteroids = yaml.safe_load(file)

        # FIXME: missing tests that configurations are the same in both files
        agent_config = config_dodge_asteroids["agent"]
        world_config = config_dodge_asteroids["world"]

        ### ACTION SPACE ###
        # one action to choose the task + three actions for each task
        # action 0 --> left action
        # action 1 --> stay action
        # action 2 --> right action
        # action 3 --> switch action
        self.action_space = gym.spaces.Discrete(4)

        ### OBSERVATION SPACE ###
        # 10x12 grid for each task
        # TODO: in the future --> SoC for each task
        self.y_position_of_agent = agent_config["size"]
        self.following_observations_size = min(
            agent_config["observation_height"],
            int(world_config["y_height"] - self.y_position_of_agent + 1),
        )
        self.observation_space = gym.spaces.Box(
            low=-10,
            high=5,
            shape=(
                self.following_observations_size * (world_config["x_width"] + 2) * 2,
            ),
            dtype=np.int64,
        )

        ### REWARD RANGE ###
        # FIXME: not sure about this
        # reward is added from each task

        # first state
        self.dodge_asteroids = MoonlanderWorldEnv(task="dodge", reward_function=reward_function)
        self.collect_asteroids = MoonlanderWorldEnv(task="collect", reward_function=reward_function)
        self.state_of_dodge_asteroids, _ = self.dodge_asteroids.reset()
        self.state_of_collect_asteroids, _ = self.collect_asteroids.reset()
        # concatenate state with vector of zeros
        self.mask = np.full(shape=self.state_of_collect_asteroids.shape, fill_value=5)
        self.current_task = 0
        self.state = np.concatenate(
            (self.state_of_dodge_asteroids.reshape(10, 12), self.mask.reshape(10, 12)),
            axis=1,
        ).flatten()

        # for rendering
        plt.ion()
        self.fig, self.ax = plt.subplots()
        eximg = np.zeros((10, 24))
        eximg[0] = -10
        eximg[1] = 5
        self.im = self.ax.imshow(eximg)

        # counter
        self.episode_counter = 0
        self.step_counter = 0
        self.switch_counter = 0

    def step(self, action: int):
        task_switching_costs = 0
        # action 0,1, or 2 (left, stay, right) for each task
        match action:
            case 0:
                # left action
                if self.current_task == 0:
                    (
                        self.state_of_dodge_asteroids,
                        reward_dodge_asteroids,
                        is_done_dodge,
                        _,
                        info_dodge,
                    ) = self.dodge_asteroids.step(action=0)
                    (
                        self.state_of_collect_asteroids,
                        reward_collect_asteroids,
                        is_done_collect,
                        _,
                        info_collect,
                    ) = self.collect_asteroids.step(action=1)
                elif self.current_task == 1:
                    (
                        self.state_of_dodge_asteroids,
                        reward_dodge_asteroids,
                        is_done_dodge,
                        _,
                        info_dodge,
                    ) = self.dodge_asteroids.step(action=1)
                    (
                        self.state_of_collect_asteroids,
                        reward_collect_asteroids,
                        is_done_collect,
                        _,
                        info_collect,
                    ) = self.collect_asteroids.step(action=0)
            case 1:
                # stay action
                (
                    self.state_of_dodge_asteroids,
                    reward_dodge_asteroids,
                    is_done_dodge,
                    _,
                    info_dodge,
                ) = self.dodge_asteroids.step(action=1)
                (
                    self.state_of_collect_asteroids,
                    reward_collect_asteroids,
                    is_done_collect,
                    _,
                    info_collect,
                ) = self.collect_asteroids.step(action=1)
            case 2:
                # right action
                if self.current_task == 0:
                    (
                        self.state_of_dodge_asteroids,
                        reward_dodge_asteroids,
                        is_done_dodge,
                        _,
                        info_dodge,
                    ) = self.dodge_asteroids.step(action=2)
                    (
                        self.state_of_collect_asteroids,
                        reward_collect_asteroids,
                        is_done_collect,
                        _,
                        info_collect,
                    ) = self.collect_asteroids.step(action=1)
                elif self.current_task == 1:
                    (
                        self.state_of_dodge_asteroids,
                        reward_dodge_asteroids,
                        is_done_dodge,
                        _,
                        info_dodge,
                    ) = self.dodge_asteroids.step(action=1)
                    (
                        self.state_of_collect_asteroids,
                        reward_collect_asteroids,
                        is_done_collect,
                        _,
                        info_collect,
                    ) = self.collect_asteroids.step(action=2)
            case 3:
                # switch
                # TODO: switch only possible every 0.5 second = 5 frames?

                # still to step so that episode does not run forever
                (
                    self.state_of_dodge_asteroids,
                    reward_dodge_asteroids,
                    is_done_dodge,
                    _,
                    info_dodge,
                ) = self.dodge_asteroids.step(action=1)
                (
                    self.state_of_collect_asteroids,
                    reward_collect_asteroids,
                    is_done_collect,
                    _,
                    info_collect,
                ) = self.collect_asteroids.step(action=1)

                if self.current_task == 0:
                    self.current_task = 1
                elif self.current_task == 1:
                    self.current_task = 0

                ### TODO: TASK-SWITCHING COSTS ###
                task_switching_costs = -0
                reward_dodge_asteroids -= 0
                reward_collect_asteroids -= 0

            # If an exact match is not confirmed, this last case will be used if provided
            case _:
                raise ValueError("action must be 0, 1, 2, or 3")

        if self.current_task == 0:
            self.state = np.concatenate(
                (
                    self.state_of_dodge_asteroids.reshape(10, 12),
                    self.mask.reshape(10, 12),
                ),
                axis=1,
            ).flatten()
        elif self.current_task == 1:
            self.state = np.concatenate(
                (
                    self.mask.reshape(10, 12),
                    self.state_of_collect_asteroids.reshape(10, 12),
                ),
                axis=1,
            ).flatten()

        self.step_counter += 1
        info = {"info_dodge": info_dodge, "info_collect": info_collect, "task_switching_costs": task_switching_costs,
                "reward_dodge": reward_dodge_asteroids, "reward_collect": reward_collect_asteroids}
        return (
            self.state,
            reward_dodge_asteroids + reward_collect_asteroids,
            is_done_dodge or is_done_collect,
            False,
            info,
        )

    def render(self):
        observation = copy.deepcopy(self.state).reshape(10, 24)
        self.im.set_data(observation)
        if self.render_mode == "human":
            self.fig.canvas.draw_idle()
        elif self.render_mode == "rgb_array":
            self.fig.canvas.draw()
            return np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
                self.fig.canvas.get_width_height()[::-1] + (3,))

    def reset(self, seed=None, options=None):
        self.state_of_dodge_asteroids, _ = self.dodge_asteroids.reset()
        self.state_of_collect_asteroids, _ = self.collect_asteroids.reset()
        # concatenate state with vector of zeros
        self.mask = np.full(shape=self.state_of_collect_asteroids.shape, fill_value=5)
        self.current_task = 0
        self.state = np.concatenate(
            (self.state_of_dodge_asteroids.reshape(10, 12), self.mask.reshape(10, 12)),
            axis=1,
        ).flatten()

        # counter
        self.episode_counter += 1
        self.step_counter = 0
        return self.state, {}
