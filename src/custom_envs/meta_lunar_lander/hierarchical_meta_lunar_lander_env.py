import copy
import math
import os
import gymnasium as gym
from gymnasium.envs.box2d.lunar_lander import LunarLander
import numpy as np
from matplotlib import pyplot as plt
from src.custom_algorithms.cleanppofm.cleanppofm import CLEANPPOFM
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure


class HierarchicalMetaLunarLanderEnv(gym.Env):
    render_mode = None
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(self):
        ### ACTION SPACE ###
        # one action to decide which task to control
        # action 0 --> task 0 can be controlled
        # action 1 --> task 1 can be controlled
        self.action_space = gym.spaces.Discrete(2)

        ### OBSERVATION SPACE ###
        # 8-dimensional vector for each task:
        # the coordinates of the lander in x
        # the coordinates of the lander in y
        # its linear velocities in x
        # its linear velocities in y
        # its angle
        # its angular velocity
        # and two booleans that represent whether each leg is in contact with the ground or not.
        low = np.array(
            [
                # these are bounds for position
                # realistically the environment should have ended
                # long before we reach more than 50% outside
                -2.5,  # x coordinate
                -2.5,  # y coordinate
                # velocity bounds is 5x rated speed
                -10.0,
                -10.0,
                -2 * math.pi,
                -10.0,
                -0.0,
                -0.0,
                # second environment
                -2.5,  # x coordinate
                -2.5,  # y coordinate
                -10.0,
                -10.0,
                -2 * math.pi,
                -10.0,
                -0.0,
                -0.0,
            ]
        ).astype(np.float32)
        high = np.array(
            [
                # these are bounds for position
                # realistically the environment should have ended
                # long before we reach more than 50% outside
                2.5,  # x coordinate
                2.5,  # y coordinate
                # velocity bounds is 5x rated speed
                10.0,
                10.0,
                2 * math.pi,
                10.0,
                1.0,
                1.0,
                # second environment
                2.5,  # x coordinate
                2.5,  # y coordinate
                10.0,
                10.0,
                2 * math.pi,
                10.0,
                1.0,
                1.0,
            ]
        ).astype(np.float32)

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=(16,), dtype=np.float32)

        # first state
        self.lunar_lander_one = LunarLander()
        self.lunar_lander_two = LunarLander()
        self.state_of_lunar_lander_one, _ = self.lunar_lander_one.reset()
        self.state_of_lunar_lander_two, _ = self.lunar_lander_two.reset()

        tmp_path = "/tmp/sb3_log/"
        self.logger = configure(tmp_path, ["stdout", "csv"])

        # Load the trained agents
        # FIXME: this is an ugly hack to load the trained agents
        with open(
                os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "../../../policies/lunar_lander_rl_model_finished"), "rb"
        ) as file:
            print("start loading agents", file)
            self.trained_lunar_lander_one = CLEANPPOFM.load(path=file,
                                                            env=make_vec_env("LunarLander-v2", n_envs=1))
            self.trained_lunar_lander_one.set_logger(logger=self.logger)
        with open(
                os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "../../../policies/lunar_lander_stable_rl_model_finished"), "rb"
        ) as file:
            # same model cannot be loaded twice -> copy does also not work
            self.trained_lunar_lander_two = CLEANPPOFM.load(path=file,
                                                            env=make_vec_env("LunarLander-v2", n_envs=1))
            self.trained_lunar_lander_two.set_logger(logger=self.logger)
            print("finish loading agents")

        # concatenate state with vector of dummy value
        # FIXME: what should be the value of the mask?
        self.mask = np.copy(low[0:8])
        self.current_task = 0
        self.state = np.concatenate((self.state_of_lunar_lander_one, self.mask)).flatten()

        self.rendering_first_time = True
        # counter
        self.episode_counter = 0
        self.step_counter = 0

    def step(self, action: int):
        """
        action: selects the task
                0: one
                1: two
        """
        # action 0 or 1 for selecting the task
        match action:
            case 0:
                # task one
                active_model = self.trained_lunar_lander_one
                active_env = self.lunar_lander_one
                inactive_env = self.lunar_lander_two
                last_state = self.state_of_lunar_lander_one
                self.current_task = 0
            case 1:
                # task two
                active_model = self.trained_lunar_lander_two
                active_env = self.lunar_lander_two
                inactive_env = self.lunar_lander_one
                last_state = self.state_of_lunar_lander_two
                self.current_task = 1

            # If an exact match is not confirmed, this last case will be used if provided
            case _:
                raise ValueError("action must be 0 or 1")

        # predict next action
        action_of_task_agent, _, _ = active_model.predict(np.expand_dims(last_state, 0), deterministic=True)
        # perform action
        new_state, active_reward, active_is_done, _, active_info = active_env.step(action=action_of_task_agent[0])

        # perform action in inactive task
        _, inactive_reward, inactive_is_done, _, inactive_info = inactive_env.step(action=0)

        # update last state
        # one only has the last state of the last actively acting TODO: prediction of FM?
        match action:
            case 0:
                # task one
                self.state_of_lunar_lander_one = new_state
            case 1:
                # task two
                self.state_of_lunar_lander_two = new_state

        if self.current_task == 0:
            self.state = np.concatenate((self.state_of_lunar_lander_one, self.mask)).flatten()
        elif self.current_task == 1:
            self.state = np.concatenate((self.mask, self.state_of_lunar_lander_two)).flatten()

        self.step_counter += 1
        return (
            self.state,
            active_reward + inactive_reward,
            active_is_done or inactive_is_done,
            False,
            {},
        )

    def render(self):
        self.lunar_lander_one.render_mode = "rgb_array"
        self.lunar_lander_two.render_mode = "rgb_array"
        # get rgb image of current lunar landers
        if self.current_task == 0:
            img_lunar_lander_one = self.lunar_lander_one.render()
            img_lunar_lander_two = np.zeros_like(img_lunar_lander_one)
        elif self.current_task == 1:
            img_lunar_lander_two = self.lunar_lander_two.render()
            img_lunar_lander_one = np.zeros_like(img_lunar_lander_two)
        if self.rendering_first_time:
            plt.ion()
            self.fig, self.ax = plt.subplots(1, 2)
            self.im_0 = self.ax[0].imshow(img_lunar_lander_one)
            self.im_1 = self.ax[1].imshow(img_lunar_lander_two)
            self.rendering_first_time = False
        self.im_0.set_data(img_lunar_lander_one)
        self.im_1.set_data(img_lunar_lander_two)
        self.fig.canvas.draw()
        if self.render_mode == "rgb_array":
            return np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
                self.fig.canvas.get_width_height()[::-1] + (3,))

    def reset(self, seed=None, options=None):
        self.state_of_lunar_lander_one, _ = self.lunar_lander_one.reset()
        self.state_of_lunar_lander_two, _ = self.lunar_lander_two.reset()
        # concatenate state with vector of dummy value
        # FIXME: what should be the value of the mask?
        self.current_task = 0
        self.state = np.concatenate((self.state_of_lunar_lander_one, self.mask)).flatten()

        self.rendering_first_time = True

        # counter
        self.episode_counter += 1
        self.step_counter = 0
        return self.state, {}
