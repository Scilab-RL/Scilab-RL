import copy
import csv
import datetime
import logging
import math
import os
import sys
import random as rnd
from typing import List, Dict

import custom_envs.moonlander.helper_functions as hlp
import cv2
import numpy as np
import torch
import yaml
from gymnasium import Env
from gymnasium import spaces
from matplotlib import pyplot as plt

# FIXME: needed for rendering rgb array
# print with two decimals
np.set_printoptions(threshold=sys.maxsize, formatter={'float': lambda x: "{0:0.2f}".format(x)})


class MoonlanderWorldEnv(Env):
    render_mode = None
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(self, task: str = "dodge", reward_function: str = "pos_neg",
                 list_of_object_dict_lists: List[Dict] = None):
        """
        initialises the environment
        Args:
        """
        self.name = "MoonlanderWorldEnv"
        self.ROOT_DIR = "."
        if task == "dodge":
            config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "standard_config.yaml")
        elif task == "collect":
            config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       "standard_config_second_task.yaml")
        else:
            raise ValueError("Task {} not implemented".format(task))

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        # overwrite current reward function
        config["reward_function"] = reward_function

        # FIXED VARIABLES
        if (
                config["agent"]["size"] < 1
                or config["world"]["y_height"] < 1
                or config["world"]["x_width"] < 1
                or config["agent"]["observation_height"] < 1
        ):
            raise ValueError(
                "Only numbers greater than zero are allowed. Please redefine the size, world height or "
                "agent observation height"
            )

        if (
                (not 0.0 <= config["world"]["drift"]["invisible_drift_probability"] <= 1.0)
                or (not 0.0 <= config["world"]["drift"]["fake_drift_probability"] <= 1.0)
                or (
                not config["world"]["drift"]["invisible_drift_probability"]
                    + config["world"]["drift"]["fake_drift_probability"]
                    <= 1.0
        )
        ):
            raise ValueError(
                "invisible_drift_probability and fake_drift_probability must be in the range of [0, 1] and"
                "sum to less than or exactly one!"
            )
        if config["world"]["objects"]["type"] not in ["obstacle", "coin"]:
            raise ValueError(
                "object_type must be either 'obstacle' or 'coin' but was {}".format(
                    config["world"]["objects"]["type"]
                )
            )

        if config["world"]["drift"]["drift_at_whole_level"] not in [
            "empty",
            "left",
            "right",
            "ranges",
        ]:
            raise ValueError(
                "drift_at_whole_level must be one of 'empty', 'left', 'right' or 'ranges', but was {}".format(
                    config["world"]["drift"]["drift_at_whole_level"]
                )
            )

        self.config = config
        self.reward_function = config["reward_function"]
        self.already_crashed_objects = []
        if self.reward_function not in ["simple", "gaussian", "pos_neg"]:
            raise ValueError(
                "Reward function {} not implemented".format(self.reward_function)
            )
        self.pos_neg_reward_info_dict_per_step = {}
        self.gaussian_reward_info_per_step = 0
        self.simple_reward_info_per_step = 0

        if "no_crashes" in config:
            self.no_crashes = config["no_crashes"]
        else:
            self.no_crashes = False

        if "funnel_ranges" in config["world"]:
            self.funnel_ranges = config["world"]["funnel_ranges"]
        else:
            self.funnel_ranges = True

        # Actions we can take: left, stay, right
        self.action_space = spaces.Discrete(3)

        # verbose level
        verbose_level = config["verbose_level"]
        if verbose_level == 0:
            logging.basicConfig(level=logging.WARNING)
        elif verbose_level == 1:
            logging.basicConfig(level=logging.INFO)
        elif verbose_level == 2:
            logging.basicConfig(level=logging.DEBUG)

        self.current_time = str(datetime.datetime.now())
        self.episode_counter = 0
        self.step_counter = 0

        # DYNAMIC VARIABLES
        logging.info("initialisation" + self.current_time + str(self.episode_counter))
        self.current_object_sizes = None

        agent_config = config["agent"]
        world_config = config["world"]
        drift_config = world_config["drift"]
        objects_config = world_config["objects"]

        size = agent_config["size"]
        # random x position of agent
        if agent_config["initial_x_position"] is None:
            x_width = world_config["x_width"]
            self.x_position_of_agent = rnd.randint(size, x_width - size + 1)
        else:
            self.x_position_of_agent = agent_config["initial_x_position"]

        # needed to read out the sizes of the moonlander world
        self.first_possible_x_position = size
        self.last_possible_x_position = world_config["x_width"] - size + 1
        self.observation_height = agent_config["observation_height"]
        self.observation_width = world_config["x_width"]
        self.task = task
        self.size = size
        self.difficulty = world_config["difficulty"]


        self.y_position_of_agent = agent_config["size"]

        # objects list includes the absolute x and y position of the objects + its size

        # list of free ranges says where no objects are defined --> used for defining the funnel
        # the lists in this free ranges list consist of two numbers:
        # where a free range starts and where it ends
        # the numbers are included in this range ([x,y] and not (x,y))

        # setup of the list of drift ranges is similar to the free ranges lists but is independent from funnels and objects
        (
            object_range_list,
            list_of_free_ranges,
            list_of_drift_ranges_with_drift_number,
        ) = hlp.create_ranges_of_objects_funnels_and_drifts(
            world_x_width=world_config["x_width"],
            world_y_height=world_config["y_height"],
            height_padding_areas=agent_config["observation_height"],
            level_difficulty=world_config["difficulty"],
            agent_size=agent_config["size"],
            drift_length=drift_config["length"],
            use_variable_drift_intensity=drift_config["variable_intensity"],
            invisible_drift_probability=drift_config["invisible_drift_probability"],
            fake_drift_probability=drift_config["fake_drift_probability"],
            funnel_range=self.funnel_ranges,
        )

        drift_at_whole_level = drift_config["drift_at_whole_level"]
        if drift_at_whole_level == "ranges":
            self.drift_ranges_with_drift_number = list_of_drift_ranges_with_drift_number
        elif drift_at_whole_level == "empty":
            self.drift_ranges_with_drift_number = []
        elif drift_at_whole_level == "left":
            # start, stop, intensity, visibility, fake
            self.drift_ranges_with_drift_number = [
                [1, world_config["y_height"], 1, False, False]
            ]
        elif drift_at_whole_level == "right":
            # start, stop, intensity, visibility, fake
            self.drift_ranges_with_drift_number = [
                [1, world_config["y_height"], -1, False, False]
            ]

        logging.info("object_range_list" + str(object_range_list))
        logging.info("free ranges" + str(list_of_free_ranges))
        logging.info("drift ranges" + str(self.drift_ranges_with_drift_number))

        if (
                objects_config["type"] == "coin"
                and world_config["difficulty"] == "hard"
        ):
            number_of_objects = 30
        else:
            number_of_objects = None

        ### OBJECTS
        self.list_of_object_dict_lists = list_of_object_dict_lists
        if self.list_of_object_dict_lists is None or len(self.list_of_object_dict_lists) <= self.episode_counter:
            object_dict_list = hlp.create_list_of_object_dicts(
                object_range_list=object_range_list,
                object_size=agent_config["size"],
                world_x_width=world_config["x_width"],
                level_difficulty=world_config["difficulty"],
                normalized_object_placement=objects_config["normalized_placement"],
                allow_overlapping_objects=objects_config["allow_overlap"],
                number_of_objects=number_of_objects,
            )
        else:
            object_dict_list = self.list_of_object_dict_lists[self.episode_counter]
        self.object_dict_list = object_dict_list

        ### WALLS
        # the walls are always the same with the same input arguments
        # it doesn't change when resetting the environment!
        # the walls_dict defines for each absolute y value in the world, where the walls are (funnel or not)
        # when the value of the key y is [0, world_x_width] then there is no funnel
        # smaller values define a funnel, e.g. [1, world_x_width - 1] indicate that the wall is one step indented
        walls_dict = hlp.create_dict_of_world_walls(
            list_of_free_ranges=list_of_free_ranges,
            world_y_height=world_config["y_height"],
            world_x_width=world_config["x_width"],
            agent_size=agent_config["size"],
            no_crashes=config["no_crashes"],
        )
        self.walls_dict = walls_dict
        logging.info("walls_dict" + str(walls_dict))

        self.crashed = False
        self.following_observations_size = min(
            agent_config["observation_height"],
            int(world_config["y_height"] - self.y_position_of_agent + 1),
        )

        self.update_observation()
        self.rendering_first_time = True

        # INITIAL STATE
        self.observation_space = spaces.Box(
            low=-10,
            high=3,
            shape=(self.following_observations_size * (world_config["x_width"] + 2),),
            dtype=np.int64,
        )

        self.information_for_each_step = [[self.state, "Nan", "Nan"]]
        # save all x and y positions of the agent + action
        self.positions_and_action = [
            [int(self.x_position_of_agent), int(self.y_position_of_agent), 1]
        ]

        # forward model prediction
        self.forward_model_prediction = None

        # input noise
        self.input_noise = 0

        ### LOGGING
        if verbose_level > 0:
            os.mkdir(self.ROOT_DIR + "/logs/" + self.current_time)

            ### OBJECTS
            self.filepath_for_object_list = (
                    self.ROOT_DIR + "/logs/" + self.current_time + "/object_list.csv"
            )
            # write the objects list of each episode to file
            with open(self.filepath_for_object_list, "a") as file:
                writer = csv.writer(file)
                writer.writerow([self.episode_counter, self.object_dict_list])

            ### WALLS
            self.filepath_for_walls_dict = (
                    self.ROOT_DIR + "/logs/" + self.current_time + "/walls_dict.csv"
            )
            # write the walls definition to file --> same for every episode
            if not os.path.exists(self.filepath_for_walls_dict):
                with open(self.filepath_for_walls_dict, "w") as file:
                    writer = csv.writer(file)
                    writer.writerow([self.walls_dict])

            ### DRIFT
            self.filepath_for_drift_ranges_list = (
                    self.ROOT_DIR + "/logs/" + self.current_time + "/drift_ranges.csv"
            )
            # write the drift ranges of each episode to file
            with open(self.filepath_for_drift_ranges_list, "a") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [self.episode_counter, self.drift_ranges_with_drift_number]
                )

            ### LOGGING EVERY EPISODE
            if verbose_level == 2:
                self.filepath = (
                        self.ROOT_DIR
                        + "/logs/"
                        + self.current_time
                        + "/"
                        + str(self.episode_counter)
                        + ".csv"
                )

                self.filepath_for_vis = (
                        self.ROOT_DIR
                        + "/logs/"
                        + self.current_time
                        + "/"
                        + str(self.episode_counter)
                        + "_vis.csv"
                )

                # write the initial state to file
                with open(self.filepath, "a") as file:
                    writer = csv.writer(file)
                    writer.writerow(["state", "reward", "done"])
                    writer.writerow(self.information_for_each_step[0])

                # write initial state to file
                with open(self.filepath_for_vis, "a") as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        ["x_position_of_agent", "y_position_of_agent", "action"]
                    )
                    writer.writerow(self.positions_and_action[0])

    def is_done(self) -> bool:
        """
        checks if agent is done by going through the world
        either by crashing in the wall or in an obstacle or by being at the end
        Returns: bool, if agent is done

        """
        return (
                self.crashed
                or self.y_position_of_agent + self.config["agent"]["observation_height"]
                == self.config["world"]["y_height"]
        )

    def apply_action(self, action: int, step_width: int) -> None:
        """
        applies an action 0,1, or 2 (left, stay, right) and updates the x position + the widths to the walls
        Args:
            action (int): integer representing left or right step or staying on same position
            step_width: input_noise added to action movement

        """
        # action 0 - 1 = -1 --> left
        # action 1 - 1 = 0 --> stay
        # action 2 -1 = 1 --> right
        self.crashed = False
        # go down --> go one step further --> y-position changes
        self.y_position_of_agent = int(self.y_position_of_agent) + 1
        self.following_observations_size = min(
            self.config["agent"]["observation_height"],
            int(self.config["world"]["y_height"] - self.y_position_of_agent + 1),
        )

        # action_movement is -1 to go left, 0 to stay and 1 to go right for agent size 1
        # for agent size 2 it is -2, 0, 2
        # for agent size 3 it is -3, 0, 3 ...
        # this allows simply adding the drift force to the action step to compute the
        # next location.
        # input noise variable because with wrapping the env it is not possible to have more than one argument for the step function
        action_movement = self.config["agent"]["size"] * action - self.config["agent"][
            "size"] + step_width + self.input_noise

        # Pick out the first drift range that contains the current y position, and take its drift direction value
        (_, _, drift, _, is_drift_fake) = next(
            filter(
                lambda drift_range: drift_range[0]
                                    <= self.y_position_of_agent
                                    <= drift_range[1],
                self.drift_ranges_with_drift_number,
            ),
            [0, 0, 0, True, False],
        )

        # Only apply drift of intensity n at every nth step
        if (
                not is_drift_fake
                and drift != 0
                and self.y_position_of_agent % abs(drift) == 0
        ):
            # Keep the direction (sign) but only move one step in the specified direction.
            # The magnitude indicates the intensity of the drift, but we don't have to
            # consider that here as the drift dict already leaves rows without drift if
            # required by the intensity.
            drift_movement = int(math.copysign(1, drift))
        else:
            # This is a non-moving step, drift contributes nothing
            drift_movement = 0

        self.x_position_of_agent += action_movement + drift_movement

        # Clamp x position to the allowed range, to avoid the agents clipping out of bounds when
        # a strong drift occurs and the agent simultaneously takes a step.

        # agent cannot be in a wall
        if self.config["no_crashes"] == True:
            if self.x_position_of_agent < self.config["agent"]["size"]:
                self.x_position_of_agent = self.config["agent"]["size"]
            elif (
                    self.x_position_of_agent
                    > self.config["world"]["x_width"] + 1 - self.config["agent"]["size"]
            ):
                self.x_position_of_agent = (
                        self.config["world"]["x_width"] + 1 - self.config["agent"]["size"]
                )
        # agent can end in a wall
        else:
            if self.x_position_of_agent < self.config["agent"]["size"] - 1:
                self.x_position_of_agent = self.config["agent"]["size"] - 1
            elif (
                    self.x_position_of_agent
                    > self.config["world"]["x_width"] + 2 - self.config["agent"]["size"]
            ):
                self.x_position_of_agent = (
                        self.config["world"]["x_width"] + 2 - self.config["agent"]["size"]
                )

    def update_observation(self) -> None:
        """
        generates the new observation/state based on the current environment parameters (agent position, objects,
        drift, etc.)
        """
        self.state = hlp.create_agent_observation(
            following_observation_size=self.following_observations_size,
            drift_ranges=self.drift_ranges_with_drift_number,
            walls_dict=self.walls_dict,
            object_dict_list=self.object_dict_list,
            agent_x_position=self.x_position_of_agent,
            agent_y_position=self.y_position_of_agent,
            world_x_width=self.config["world"]["x_width"],
            agent_size=self.config["agent"]["size"],
            object_type=self.config["world"]["objects"]["type"],
            no_crashes=self.config["no_crashes"],
        )

    def calculate_reward(self) -> tuple[int, int]:
        """
        calculates reward if the agent has crashed in the wall or in an obstacle
        if agent is in obstacle or wall, reward is -100 & crashed is True
        if agent is near an obstacle or wall, reward is 0
        if agent does a successful step, reward is 10
        Returns (int): reward for the current step

        """
        # sometimes the x_position of the agent is converted to a tensor (when? why?)
        # which results in errors in calculating the rewards
        self.x_position_of_agent = int(self.x_position_of_agent)
        actual_reward = 0
        collected_objects = self.find_intersections(self.object_dict_list)
        number_of_crashed_or_collected_objects = 0
        for obj in collected_objects:
            if obj not in self.already_crashed_objects:
                number_of_crashed_or_collected_objects += 1
        # state is current observation
        # agent is in obstacle or wall = crash and crashes are lead to end the episode --> same reward for both tasks
        if (((self.config["world"]["objects"]["type"] == "obstacle" and
              (self.has_agent_collided_with_wall() or len(collected_objects) > 0)) or
             (self.config["world"]["objects"]["type"] == "coin" and self.has_agent_collided_with_wall())) and
                not self.no_crashes):
            self.crashed = True
            self.simple_reward_info_per_step = -100
            self.gaussian_reward_info_per_step = -1000
            self.pos_neg_reward_info_dict_per_step["pos"] = None
            self.pos_neg_reward_info_dict_per_step["neg"] = None
            if self.reward_function == "simple":
                actual_reward = -100
            elif self.reward_function == "gaussian":
                actual_reward = -1000
            elif self.reward_function == "pos_neg":
                raise ValueError(
                    "Reward function {} can not be used with crashes".format(
                        self.reward_function
                    )
                )

        # no crash in obstacle or wall OR crash but crashes do not end the episode
        else:
            ##### SIMPLE REWARD #####
            reward_simple = self.calculate_simple_reward(collected_objects=collected_objects)
            if self.reward_function == "simple":
                actual_reward = reward_simple

            ##### GAUSSIAN REWARD #####
            reward_gaussian = self.calculate_gaussian_reward(collected_objects=collected_objects)
            if self.reward_function == "gaussian":
                actual_reward = reward_gaussian

            ##### POS NEG REWARD #####
            reward_pos_neg = self.calculate_pos_neg_reward(collected_objects=collected_objects)
            if self.reward_function == "pos_neg":
                actual_reward = reward_pos_neg

        return actual_reward, number_of_crashed_or_collected_objects

    def calculate_simple_reward(self, collected_objects: list[dict]) -> int:
        current_simple_reward = 0
        relevant_shortened_state = list()
        for row in self.state[0: 2 * self.config["agent"]["size"]]:
            relevant_shortened_state.append(
                row[
                max(
                    self.x_position_of_agent - self.config["agent"]["size"], 0
                ): min(
                    self.x_position_of_agent + self.config["agent"]["size"] + 1,
                    self.state.shape[1],
                )
                ]
            )
        relevant_shortened_state = np.array(relevant_shortened_state)

        # obstacles
        if self.config["world"]["objects"]["type"] == "obstacle":
            # if agent is in obstacles and crashes to not lead to end the episode
            if len(collected_objects) > 0 and self.no_crashes:
                self.simple_reward_info_per_step = -100
                current_simple_reward = -100
            # if agent is near an obstacle (3) or wall (-1)
            # wall only relevant when crashing in wall leads to end the episode
            elif 3 in relevant_shortened_state or (-1 in relevant_shortened_state and not self.no_crashes):
                self.simple_reward_info_per_step = 0
                current_simple_reward = 0
            else:
                self.simple_reward_info_per_step = 10
                current_simple_reward = 10
        # coins
        else:
            # if one or more coins are collected -> reward 10 per coin
            if len(collected_objects) > 0:
                self.simple_reward_info_per_step = len(collected_objects) * 10
                current_simple_reward = len(collected_objects) * 10
                if self.reward_function == "simple":
                    # Prevent coins from being collected multiple times
                    for coin in collected_objects:
                        self.object_dict_list.remove(coin)
            else:
                self.simple_reward_info_per_step = 0
                current_simple_reward = 0

        return current_simple_reward

    def calculate_gaussian_reward(self, collected_objects: list[dict]) -> int:
        current_reward_gaussian = 0
        # remove agent from state
        blurred_state = copy.deepcopy(self.state)
        blurred_state[blurred_state == 1] = 0
        # handle drift as wall tiles
        blurred_state[blurred_state == -5] = -1
        # if no crashes in walls (and obstacles) are possible, remove walls from state
        if self.no_crashes:
            blurred_state[blurred_state == -1] = 0
        # replace walls (-1)(maybe already removed through no crashes) with the highest value (255)
        blurred_state[blurred_state == -1] = 255

        range_of_agent = range(
            -(self.config["agent"]["size"] - 1),
            self.config["agent"]["size"]
        )

        if len(collected_objects) > 0:
            for obj in collected_objects:
                # Prevent coins from being collected multiple times
                if self.config["world"]["objects"]["type"] == "coin" and self.reward_function == "gaussian":
                    self.object_dict_list.remove(obj)

                # find positions where agent is on object --> only last row of agent is possible
                x_positions_of_agent = []
                y_positions_of_agent = []
                for index in range_of_agent:
                    x_positions_of_agent.append(
                        self.x_position_of_agent + index
                    )
                    y_positions_of_agent.append(
                        self.y_position_of_agent + index
                    )
                agent_positions = [(x_positions_of_agent[index_1], y_positions_of_agent[index_2]) for index_2 in
                                   range(len(y_positions_of_agent)) for index_1 in
                                   range(len(x_positions_of_agent))]

                range_of_object = range(
                    -(math.floor(obj["size"] / 2)),
                    (math.floor(obj["size"] / 2) + 1),
                )
                x_positions_of_object = []
                y_positions_of_object = []
                for index in range_of_object:
                    x_positions_of_object.append(obj["x"] + index)
                    y_positions_of_object.append(obj["y"] + index)
                object_positions = [(x_positions_of_object[index_1], y_positions_of_object[index_2]) for index_2 in
                                    range(len(y_positions_of_object)) for index_1 in
                                    range(len(x_positions_of_object))]

                positions_where_agent_is_on_object = list(
                    set(agent_positions).intersection(object_positions)
                )
                # FIXME: what about multiple objects???

                if self.config["world"]["objects"]["type"] == "coin":
                    for pos in positions_where_agent_is_on_object:
                        # replace values of intersection of agent and coin with 2
                        blurred_state[pos[1] - self.y_position_of_agent + 1, pos[0]] = 2
                # obstacle
                else:
                    for pos in positions_where_agent_is_on_object:
                        # replace values of intersection of agent and obstacle with 3
                        blurred_state[pos[1] - self.y_position_of_agent + 1, pos[0]] = 3

        if self.config["world"]["objects"]["type"] == "coin":
            # replace nothing (0) with middle value (127)
            blurred_state[blurred_state == 0] = 127
            # replace coins (2) with the lowest value (0)
            blurred_state[blurred_state == 2] = 0
            # replace collect (-10) with the lowest value (0)
            # -10 here not possible for crashes, because it would already handled above
            blurred_state[blurred_state == -10] = 0
        else:
            # replace obstacles (3) with the highest value (255)
            blurred_state[blurred_state == 3] = 255
            # replace crash (-10) with the highest value (255)
            # -10 here not possible for crashes, because it would already handled above
            blurred_state[blurred_state == -10] = 255

        # apply gaussian filter (7x7)
        # gymnasium expects an int64 numpy array, but gaussian blur only works with int16
        blurred_state = cv2.GaussianBlur(blurred_state.astype(np.int16), (7, 7), 0)

        # get values of each pixel of current agent position
        values_of_agent_position = []
        # rows
        for i in range(2 * self.config["agent"]["size"] - 1):
            # columns
            if len(range_of_agent) == 0:
                values_of_agent_position.append(
                    blurred_state[i][self.x_position_of_agent]
                )
            else:
                for j in range_of_agent:
                    values_of_agent_position.append(
                        blurred_state[i][self.x_position_of_agent - j]
                    )

        for value in values_of_agent_position:
            current_reward_gaussian += abs(value - 255)
        # normalize reward
        if self.config["world"]["objects"]["type"] == "coin":
            # agent size 1 (1x1)
            # reward when no coin is near: 128 --> should be 0
            # lowest reward possible when collecting a coin: 146 --> should be 500
            if self.config["agent"]["size"] == 1:
                normalized_reward = ((current_reward_gaussian - 128) / 0.036) / 10
            # agent size 2 (3x3)
            # reward when no coin is near: 128*9=1152 --> should be 0
            # lowest reward possible when collecting a coin: 1309 --> should be 500
            # the function for this is: f(x) = 0.314x + 1152
            # we calculate the corresponding normalized reward x for the current reward f(x)
            # for reward of 10 when no coin is near:
            # normalized_reward = ((reward - 1152) / 0.314)/10
            elif self.config["agent"]["size"] == 2:
                normalized_reward = ((current_reward_gaussian - 1152) / 0.314) / 10
            # agent size 3 (5x5)
            # reward when no coin is near: 3200 -->  shouldbe 0
            # lowest reward possible when collecting a coin: 3373 --> should be 500
            elif self.config["agent"]["size"] == 3:
                normalized_reward = ((current_reward_gaussian - 3200) / 0.346) / 10
            # agent size 4 (7x7)
            # reward when no coin is near: 6272 --> should be 0
            # lowest reward possible when collecting a coin: 6445 --> should be 500
            elif self.config["agent"]["size"] == 4:
                normalized_reward = ((current_reward_gaussian - 6272) / 0.346) / 10
            # agent size 5 (9x9)
            # reward when no coin is near: 10368 --> should be 0
            # lowest reward possible when collecting a coin: 10541 --> should be 500
            elif self.config["agent"]["size"] == 5:
                normalized_reward = ((current_reward_gaussian - 10368) / 0.346) / 10
            # other sizes:
            # when an agent "collides" with the coin on just one corner the reward is always 127, 126, 126, 124,
            #                                                                                126, 124, 121, 116,
            #                                                                                126, 121, 111, 98,
            #                                                                                124, 116, 98, 75
            # in the colliding corner, while the other values in the agent position are 127
            else:
                agent_size = self.config["agent"]["size"]
                reward_no_coin = agent_size * 128  # reward when no coin is near
                normalized_reward = ((current_reward_gaussian - reward_no_coin) / 0.346) / 10
        else:
            # agent size 1 (1x1)
            # biggest reward possible when near an obstacle: 231 --> should be 0
            # biggest reward possible when not near an obstacle: 255 --> should be 10
            if self.config["agent"]["size"] == 1:
                normalized_reward = (current_reward_gaussian - 231) / 2.4
            # agent size 2 (3x3)
            # we want the same distance as before 10 for successful step, 0 if near an obstacle (obstacle task)
            # the biggest reward possible when near an obstacle is 2219 --> this should be 0
            # the highest reward possible when not near an obstacle is 255*9=2295 --> this should be 10
            # the function for this is: f(x) = 7.6x + 2219
            # we calculate the corresponding normalized reward x for the current reward f(x)
            elif self.config["agent"]["size"] == 2:
                normalized_reward = (current_reward_gaussian - 2219) / 7.6
            # agent size 3 (5x5)
            # biggest reward possible when near an obstacle: 6303 --> should be 0
            # biggest reward possible when not near an obstacle: 255*25=6375 --> should be 10
            elif self.config["agent"]["size"] == 3:
                normalized_reward = (current_reward_gaussian - 6303) / 7.2
            # agent size 4 (7x7)
            # biggest reward possible when near an obstacle: 12423 --> should be 0
            # biggest reward possible when not near an obstacle: 255*49=12495 --> should be 10
            elif self.config["agent"]["size"] == 4:
                normalized_reward = (current_reward_gaussian - 12423) / 7.2
            # agent size 5 (9x9)
            # biggest reward possible when near an obstacle: 20583 --> should be 0
            # biggest reward possible when not near an obstacle: 255*81=20655 --> should be 10
            elif self.config["agent"]["size"] == 5:
                normalized_reward = (current_reward_gaussian - 20583) / 7.2
            # other sizes
            else:
                agent_size = self.config["agent"]["size"]
                agent_size_matrix = (agent_size * 2 - 1) * (agent_size * 2 - 1)  # e.g. agent size = 4 --> 7x7=49
                smallest_reward_obstacle = 2223 + (agent_size_matrix - 9) * 255  # reward when near an obstacle
                normalized_reward = (current_reward_gaussian - smallest_reward_obstacle) / 7.2
        self.gaussian_reward_info_per_step = int(normalized_reward)
        return int(normalized_reward)

    def calculate_pos_neg_reward(self, collected_objects: list[dict]) -> int:
        current_reward_pos_neg = 0
        if (
                self.config["world"]["difficulty"] != "easy"
                and self.config["world"]["difficulty"] != "hard"
                and self.reward_function == "pos_neg"
        ):
            raise ValueError(
                "Reward function {} can only be used with easy and hard difficulty".format(
                    self.reward_function
                )
            )
        elif (
                self.config["world"]["difficulty"] != "easy"
                and self.config["world"]["difficulty"] != "hard"
                and self.reward_function != "pos_neg"
        ):
            self.pos_neg_reward_info_dict_per_step["pos"] = None
            self.pos_neg_reward_info_dict_per_step["neg"] = None
        else:
            # positive reward because of passing
            self.pos_neg_reward_info_dict_per_step["pos"] = [0]
            self.pos_neg_reward_info_dict_per_step["neg"] = [0]
            for object in self.object_dict_list:
                if (
                        object["y"] + (2 * self.config["agent"]["size"] - 1)
                        == self.y_position_of_agent
                        and object not in self.already_crashed_objects
                ):
                    if self.config["world"]["difficulty"] == "easy":
                        if self.config["world"]["objects"]["type"] == "coin":
                            current_reward_pos_neg -= 7
                            if self.pos_neg_reward_info_dict_per_step["neg"] != [0]:
                                self.pos_neg_reward_info_dict_per_step["neg"].append(-7)
                            else:
                                self.pos_neg_reward_info_dict_per_step["neg"] = [-7]
                        else:
                            current_reward_pos_neg += 7
                            if self.pos_neg_reward_info_dict_per_step["pos"] != [0]:
                                self.pos_neg_reward_info_dict_per_step["pos"].append(7)
                            else:
                                self.pos_neg_reward_info_dict_per_step["pos"] = [7]
                    elif self.config["world"]["difficulty"] == "hard":
                        if self.config["world"]["objects"]["type"] == "coin":
                            current_reward_pos_neg -= 3
                            if self.pos_neg_reward_info_dict_per_step["neg"] != [0]:
                                self.pos_neg_reward_info_dict_per_step["neg"].append(-3)
                            else:
                                self.pos_neg_reward_info_dict_per_step["neg"] = [-3]
                        else:
                            current_reward_pos_neg += 1
                            if self.pos_neg_reward_info_dict_per_step["pos"] != [0]:
                                self.pos_neg_reward_info_dict_per_step["pos"].append(1)
                            else:
                                self.pos_neg_reward_info_dict_per_step["pos"] = [1]
            for crash in self.find_intersections(self.object_dict_list):
                if crash not in self.already_crashed_objects:
                    if self.config["world"]["difficulty"] == "easy":
                        if self.config["world"]["objects"]["type"] == "coin":
                            # Prevent coins from being collected multiple times
                            if self.reward_function == "pos_neg":
                                self.object_dict_list.remove(crash)
                            current_reward_pos_neg += 7
                            if self.pos_neg_reward_info_dict_per_step["pos"] != [0]:
                                self.pos_neg_reward_info_dict_per_step["pos"].append(7)
                            else:
                                self.pos_neg_reward_info_dict_per_step["pos"] = [7]
                        else:
                            current_reward_pos_neg -= 7
                            if self.pos_neg_reward_info_dict_per_step["neg"] != [0]:
                                self.pos_neg_reward_info_dict_per_step["neg"].append(-7)
                            else:
                                self.pos_neg_reward_info_dict_per_step["neg"] = [-7]
                    elif self.config["world"]["difficulty"] == "hard":
                        if self.config["world"]["objects"]["type"] == "coin":
                            # Prevent coins from being collected multiple times
                            if self.reward_function == "pos_neg":
                                self.object_dict_list.remove(crash)
                            current_reward_pos_neg += 3
                            if self.pos_neg_reward_info_dict_per_step["pos"] != [0]:
                                self.pos_neg_reward_info_dict_per_step["pos"].append(3)
                            else:
                                self.pos_neg_reward_info_dict_per_step["pos"] = [3]
                        else:
                            current_reward_pos_neg -= 3
                            if self.pos_neg_reward_info_dict_per_step["neg"] != [0]:
                                self.pos_neg_reward_info_dict_per_step["neg"].append(-3)
                            else:
                                self.pos_neg_reward_info_dict_per_step["neg"] = [-3]

                    self.already_crashed_objects.append(crash)
        return current_reward_pos_neg

    def has_agent_collided_with_wall(self) -> bool:
        """
        Returns: Whether the agent has collided with a wall.
        """
        size = self.config["agent"]["size"]
        # We have to check each row of the agent because there may be
        # varying levels of wall indentation (like in a funnel)
        for agent_y_index in range(size * 2 - 1):
            # - 1 because the left wall isn't included in the world width
            wall_index = str(self.y_position_of_agent - (size - 1) + agent_y_index)
            if (
                    self.x_position_of_agent - size < self.walls_dict[wall_index][0]
                    or self.x_position_of_agent + size - 1
                    > self.walls_dict[wall_index][1] - 1
            ):
                return True
        return False

    def find_intersections(self, objects: List[Dict[str, int]]) -> List[Dict[str, int]]:
        """
        Returns: The list of obstacles/coins intersecting with the agent

        Args:
            objects: List of objects (obstacles/coins/...) to check for an intersection.
        """

        def collides_with_agent(obj) -> bool:
            # Check for collisions by ensuring that the center of the agent is not in a radius around the
            # object equal to the size of the object, plus the size of the agent (-2 because we do want to allow
            # them to be exactly side by side, and they are minimum width 1 each)
            radius = obj["size"] + self.config["agent"]["size"] - 2
            return (
                    abs(self.y_position_of_agent - obj["y"]) <= radius
                    and abs(self.x_position_of_agent - obj["x"]) <= radius
            )

        return list(filter(collides_with_agent, objects))

    def step(self, action: int, step_width: int = 0):
        """
        performs a whole step of an agent including applying an action, updating the observation and getting a reward
        Args:
            action (int): integer representing left or right step or staying on same position
            step_width: input_noise added to action movement

        Returns:
            the state in form of the observation matrix
            the reward
            if the agent is done going through the world
            an empty info dictionary

        """
        # logging.info("step in env")
        if self.is_done():
            raise EnvironmentError(
                "no more action steps possible at current position in the environment"
            )

        # APPLY ACTION
        self.apply_action(action=action, step_width=step_width)

        # UPDATE OBSERVATION
        self.update_observation()

        # set placeholder for truncated
        truncated = False

        # CALCULATE REWARD
        reward, number_of_crashed_or_collected_objects = self.calculate_reward()

        # info of rewards
        info = {"simple": self.simple_reward_info_per_step, "gaussian": self.gaussian_reward_info_per_step,
                "pos_neg": self.pos_neg_reward_info_dict_per_step,
                "number_of_crashed_or_collected_objects": number_of_crashed_or_collected_objects}

        self.positions_and_action = self.positions_and_action + [
            [
                self.x_position_of_agent,
                self.y_position_of_agent,
                action,
            ]
        ]
        self.information_for_each_step = self.information_for_each_step + [
            [self.state, reward, self.is_done()]
        ]

        # at the end of the episode, write log files
        if self.config["verbose_level"] == 2 and self.is_done():
            # write the current step of the agent to the file
            with open(self.filepath, "a") as file:
                writer = csv.writer(file)
                # first element is already added in the initialization
                for step in self.information_for_each_step[1:]:
                    writer.writerow(step)

            ### VISUALIZATION
            with open(self.filepath_for_vis, "a") as file:
                writer = csv.writer(file)
                # first element is already added in the initialization
                for step in self.positions_and_action[1:]:
                    writer.writerow(step)

        self.step_counter += 1
        # return step information
        return self.state.flatten(), reward, self.is_done(), truncated, info

    def render(self):
        if self.rendering_first_time:
            plt.ion()
            if self.forward_model_prediction is None:
                self.fig, self.ax = plt.subplots()
                eximg = np.zeros((self.state.shape))
                eximg[0] = -10
                eximg[1] = 3
                self.im = self.ax.imshow(eximg)
            # or model-based rendering
            else:
                self.fig_mb, self.ax_mb = plt.subplots()
                eximg_mb = np.zeros((self.state.shape[0], self.state.shape[1] * 2))
                eximg_mb[0] = -10
                eximg_mb[1] = 3
                self.im_mb = self.ax_mb.imshow(eximg_mb)
            self.rendering_first_time = False
        # FIXME: this is hardcoded for our custom moonlander env
        if self.forward_model_prediction is not None:
            # tensor of (1,12) or (1, 1260)
            copy_of_forward_model_prediction = copy.deepcopy(self.forward_model_prediction)[0]
            # build empty obs
            matrix = np.zeros(shape=(self.observation_height, self.observation_width + 2), dtype=np.int16)

            # add objects
            counter = 2
            while counter < len(copy_of_forward_model_prediction):
                x_position_of_object = int(torch.round(copy_of_forward_model_prediction[counter]))
                if self.size <= x_position_of_object <= self.observation_width + 1 - self.size:
                    matrix[
                    max(0, min(self.observation_height - (2 * self.size - 1),
                               int(torch.round(
                                   copy_of_forward_model_prediction[
                                       counter + 1]) - self.size + 1))):  # y start of object
                    max(2 * self.size - 2, min(self.observation_height - 1, int(torch.round(
                        copy_of_forward_model_prediction[counter + 1]) + self.size - 1))) + 1,  # y end of object
                    x_position_of_object - self.size + 1:  # x start of object
                    x_position_of_object + self.size] = 2  # x end of object
                counter += 2

            # add agent
            x_position_of_agent = int(torch.round(copy_of_forward_model_prediction[0]))
            # make sure that the agent is within the matrix
            if x_position_of_agent < self.size:
                x_position_of_agent = self.size
            elif x_position_of_agent > self.observation_width + 1 - self.size:
                x_position_of_agent = self.observation_width + 1 - self.size

            # first element is the y position of the agent, second element is the x position of the agent
            matrix[
            max(0, min(self.observation_height - (2 * self.size - 1),
                       int(torch.round(copy_of_forward_model_prediction[1]) - self.size + 1))):  # y start of agent
            max(2 * self.size - 2,
                min(self.observation_height - 1,
                    int(torch.round(copy_of_forward_model_prediction[1]) + self.size - 1))) + 1,
            # y end of agent
            x_position_of_agent - self.size + 1:  # x start of agent
            x_position_of_agent + self.size] = 1  # x end of agent

            # add wall
            matrix[:, 0] = -1
            matrix[:, -1] = -1

            forward_model_pred = matrix
            plotted_image = np.concatenate((self.state, forward_model_pred), axis=1)
            self.im_mb.set_data(plotted_image)
            self.fig_mb.canvas.draw()
        else:
            self.im.set_data(self.state)
            self.fig.canvas.draw()
        if self.render_mode == "rgb_array":
            if self.forward_model_prediction is not None:
                return np.frombuffer(self.fig_mb.canvas.tostring_rgb(), dtype=np.uint8).reshape(
                    self.fig_mb.canvas.get_width_height()[::-1] + (3,))
            else:
                return np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
                    self.fig.canvas.get_width_height()[::-1] + (3,))

    def reset(self, seed=None, options=None):
        """
        resets the environment
        """
        super().reset(seed=seed)
        # logging.info("reset " + self.current_time + str(self.episode_counter))
        self.episode_counter += 1
        self.step_counter = 0

        agent_config = self.config["agent"]
        world_config = self.config["world"]
        drift_config = world_config["drift"]
        objects_config = world_config["objects"]

        # random x position of agent
        size = agent_config["size"]
        if agent_config["initial_x_position"] is None:
            self.x_position_of_agent = rnd.randint(
                size, world_config["x_width"] - size + 1
            )
        else:
            self.x_position_of_agent = agent_config["initial_x_position"]

        # needed to read out the sizes of the moonlander world
        self.first_possible_x_position = size
        self.last_possible_x_position = world_config["x_width"] - size + 1

        self.y_position_of_agent = size

        (
            object_range_list,
            list_of_free_ranges,
            drift_ranges,
        ) = hlp.create_ranges_of_objects_funnels_and_drifts(
            world_x_width=world_config["x_width"],
            world_y_height=world_config["y_height"],
            height_padding_areas=agent_config["observation_height"],
            level_difficulty=world_config["difficulty"],
            drift_length=drift_config["length"],
            use_variable_drift_intensity=drift_config["variable_intensity"],
            invisible_drift_probability=drift_config["invisible_drift_probability"],
            fake_drift_probability=drift_config["fake_drift_probability"],
            funnel_range=self.funnel_ranges,
        )
        drift_at_whole_level = drift_config["drift_at_whole_level"]
        if drift_at_whole_level == "ranges":
            self.drift_ranges_with_drift_number = drift_ranges
        elif drift_at_whole_level == "no":
            self.drift_ranges_with_drift_number = []
        elif drift_at_whole_level == "left":
            # start, stop, intensity, visibility, fake
            self.drift_ranges_with_drift_number = [
                [1, world_config["y_height"], 1, False, False]
            ]
        elif drift_at_whole_level == "right":
            # start, stop, intensity, visibility, fake
            self.drift_ranges_with_drift_number = [
                [1, world_config["y_height"], -1, False, False]
            ]

        if (
                objects_config["type"] == "coin"
                and self.reward_function == "pos_neg"
                and world_config["difficulty"] == "hard"
        ):
            number_of_objects = 15
        else:
            number_of_objects = None

        ### OBJECTS
        if self.list_of_object_dict_lists is None or len(self.list_of_object_dict_lists) <= self.episode_counter:
            object_dict_list = hlp.create_list_of_object_dicts(
                object_range_list=object_range_list,
                object_size=agent_config["size"],
                world_x_width=world_config["x_width"],
                level_difficulty=world_config["difficulty"],
                normalized_object_placement=objects_config["normalized_placement"],
                allow_overlapping_objects=objects_config["allow_overlap"],
                number_of_objects=number_of_objects,
            )
        else:
            object_dict_list = self.list_of_object_dict_lists[self.episode_counter]
        self.object_dict_list = object_dict_list

        ### WALLS --> always the same with the same game settings

        self.crashed = False
        self.following_observations_size = min(
            agent_config["observation_height"],
            int(world_config["y_height"] - self.y_position_of_agent + 1),
        )

        self.update_observation()
        self.rendering_first_time = True

        # save all x and y positions of the agent + action
        self.positions_and_action = [
            [int(self.x_position_of_agent), int(self.y_position_of_agent), 1]
        ]
        self.information_for_each_step = [[self.state, "Nan", "Nan"]]

        # forward model prediction
        self.forward_model_prediction = None

        # input noise
        self.input_noise = 0

        if self.config["verbose_level"] > 0:
            ### OBJECTS
            with open(self.filepath_for_object_list, "a") as file:
                writer = csv.writer(file)
                writer.writerow([self.episode_counter, self.object_dict_list])

            ### DRIFT
            with open(self.filepath_for_drift_ranges_list, "a") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [self.episode_counter, self.drift_ranges_with_drift_number]
                )

            ### LOGGING EVERY EPISODE
            if self.config["verbose_level"] == 2:
                self.filepath = (
                        self.ROOT_DIR
                        + "/logs/"
                        + self.current_time
                        + "/"
                        + str(self.episode_counter)
                        + ".csv"
                )
                self.filepath_for_vis = (
                        self.ROOT_DIR
                        + "/logs/"
                        + self.current_time
                        + "/"
                        + str(self.episode_counter)
                        + "_vis.csv"
                )

        # set placeholder for info
        self.pos_neg_reward_info_dict_per_step = {}
        self.gaussian_reward_info_per_step = 0
        self.simple_reward_info_per_step = 0
        info = {"simple": self.simple_reward_info_per_step, "gaussian": self.gaussian_reward_info_per_step,
                "pos_neg": self.pos_neg_reward_info_dict_per_step, "number_of_crashed_or_collected_objects": 0}

        return self.state.flatten(), info

    def set_forward_model_prediction(self, new_forward_model_prediction: torch.tensor) -> None:
        self.forward_model_prediction = new_forward_model_prediction

    def set_input_noise(self, new_input_noise: float) -> None:
        self.input_noise = new_input_noise

    def set_state(self, new_state: np.ndarray) -> None:
        self.state = new_state

    def set_object_dict_list(self, object_dict_list: List[Dict]) -> None:
        self.object_dict_list = object_dict_list

    def __deepcopy__(self, memo):
        cls = self.__class__
        obj = cls.__new__(cls)
        memo[id(self)] = obj
        for k, v in self.__dict__.items():
            if k in ["fig_mb", "ax_mb", "im_mb"]:
                v = dict()
            # detach needed, otherwise deepcopy would not work
            if torch.is_tensor(v):
                setattr(obj, k, copy.deepcopy(v.detach(), memo))
            else:
                setattr(obj, k, copy.deepcopy(v, memo))
            pass
        return obj
