import time

import gymnasium_robotics
import numpy as np
from os import path
from typing import Dict, List, Optional, Union

import numpy as np
from gymnasium import spaces
from gymnasium.utils.ezpickle import EzPickle

# from gymnasium_robotics.envs.point_maze.point_env import PointEnv
from gymnasium_robotics.envs.maze.maps import U_MAZE
from gymnasium_robotics.envs.maze.maze_v4 import MazeEnv
from gymnasium_robotics.envs.maze.point import PointEnv
from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames

# ParentClass = gymnasium_robotics.envs.maze.ant_maze_v4
from gymnasium.envs.registration import load_env_creator
from os import path
from typing import Optional

import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv

GymnasiumPointMazeEnvClass = load_env_creator('gymnasium_robotics.envs.maze.point_maze:PointMazeEnv')


class MultiObjPointEnv(MujocoEnv):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(self, n_objects=1, xml_file: Optional[str] = None, **kwargs):

        self.n_objects = n_objects
        if xml_file is None:
            xml_file = path.join(
                path.dirname(path.realpath(__file__)), "../assets/point/point.xml"
            )
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4+4*self.n_objects,), dtype=np.float64
        )
        n_tries = 10
        success = False
        for _ in range(n_tries):
            try:
                super().__init__(
                    model_path=xml_file,
                    frame_skip=1,
                    observation_space=observation_space,
                    **kwargs
                )
                success = True
            except Exception as e:
                print(f"Error instantiating point env: {e}. Retrying...")
                time.sleep(1)
        if not success:
            raise RuntimeError

    def reset_model(self) -> np.ndarray:
        self.set_state(self.init_qpos, self.init_qvel)
        obs, _ = self._get_obs()

        return obs

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self._clip_velocity()
        self.do_simulation(action, self.frame_skip)
        obs, info = self._get_obs()
        # This environment class has no intrinsic task, thus episodes don't end and there is no reward
        reward = 0
        terminated = False
        truncated = False

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos[:2]
        qvel = self.data.qvel[:2]
        for o in range(self.n_objects):
            o_pos_idx = 2 + (o * 7)
            o_vel_idx = 2 + (o * 6)
            qpos = np.concatenate([qpos, self.data.qpos[o_pos_idx:o_pos_idx + 2]])
            qvel = np.concatenate([qvel, self.data.qvel[o_vel_idx:o_vel_idx + 2]])
        return np.concatenate([qpos, qvel]).ravel(), {}

    def _clip_velocity(self):
        """The velocity needs to be limited because the ball is
        force actuated and the velocity can grow unbounded."""
        qvel = np.clip(self.data.qvel, -5.0, 5.0)
        self.set_state(self.data.qpos, qvel)

class PointObjEnv(GymnasiumPointMazeEnvClass):
    metadata = GymnasiumPointMazeEnvClass.metadata
    metadata['render_fps'] = 30

    def __init__(
            self,
            maze_map: List[List[Union[str, int]]] = U_MAZE,
            render_mode: Optional[str] = None,
            reward_type: str = "sparse",
            continuing_task: bool = True,
            reset_target: bool = False,
            distance_threshold = 0.45,
            n_objects = 1,
            **kwargs,
    ):
        self.distance_threshold=distance_threshold
        self.n_objects = n_objects
        point_xml_file_path = path.join(
            path.dirname(path.realpath(__file__)), "../assets/point_multi_object.xml"
        )
        MazeEnv.__init__(self,
            agent_xml_path=point_xml_file_path,
            maze_map=maze_map,
            maze_size_scaling=1,
            maze_height=0.4,
            reward_type=reward_type,
            continuing_task=continuing_task,
            reset_target=reset_target,
            **kwargs,
        )
        # self.tmp_xml_file_path = path.join(path.dirname(path.realpath(__file__)), "../assets/maze/point_obj_env.xml")
        maze_length = len(maze_map)
        default_camera_config = {"distance": 12.5 if maze_length > 8 else 8.8}

        self.point_env = MultiObjPointEnv(
            xml_file=self.tmp_xml_file_path,
            render_mode=render_mode,
            default_camera_config=default_camera_config,
            n_objects=self.n_objects,
            **kwargs,
        )
        self._model_names = MujocoModelNames(self.point_env.model)
        self.target_site_id = self._model_names.site_name2id["target"]

        self.action_space = self.point_env.action_space
        obs_shape: tuple = self.point_env.observation_space.shape
        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs_shape, dtype="float64"
                ),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=(2,), dtype="float64"),
                desired_goal=spaces.Box(-np.inf, np.inf, shape=(2,), dtype="float64"),
            )
        )

        self.render_mode = render_mode

        EzPickle.__init__(
            self,
            maze_map,
            render_mode,
            reward_type,
            continuing_task,
            reset_target,
            **kwargs,
        )

    def compute_reward(
            self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> float:
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        success = (distance <= self.distance_threshold).astype(np.float64)
        if self.reward_type == "dense":
            return np.exp(-distance)
        elif self.reward_type == "sparse":
            return success
        elif self.reward_type == "sparseneg":
            return success - 1
        else:
            assert False, "reward type of ant env must be either dense or sparse or sparseneg"

    def compute_terminated(
            self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> bool:
        if not self.continuing_task:
            # If task is episodic terminate the episode when the goal is reached
            return bool(np.linalg.norm(achieved_goal - desired_goal) <= self.distance_threshold)
        else:
            # Continuing tasks don't terminate, episode will be truncated when time limit is reached (`max_episode_steps`)
            return False

    def update_goal(self, achieved_goal: np.ndarray) -> None:
        """Update goal position if continuing task and within goal radius."""

        if (
                self.continuing_task
                and self.reset_target
                and bool(np.linalg.norm(achieved_goal - self.goal) <= self.distance_threshold)
                and len(self.maze.unique_goal_locations) > 1
        ):
            # Generate a goal while within 0.45 of achieved_goal. The distance check above
            # is not redundant, it avoids calling update_target_site_pos() unless necessary
            while np.linalg.norm(achieved_goal - self.goal) <= self.distance_threshold:
                # Generate another goal
                goal = self.generate_target_goal()
                # Add noise to goal position
                self.goal = self.add_xy_position_noise(goal)

            # Update the position of the target site for visualization
            self.update_target_site_pos()
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        **kwargs,
    ):
        MazeEnv.reset(self, seed=seed, **kwargs)
        self.point_env.model.site_rgba[self.target_site_id] = [0.5, 0.5, 0.5, 0.3]

        self.point_env.init_qpos[:2] = self.reset_pos
        positions = [self.reset_pos]
        for o in range(self.n_objects):
            pos_free = False
            obj_pos = None
            while pos_free is False:
                obj_pos = self.generate_reset_pos()
                obj_pos = self.add_xy_position_noise(obj_pos)
                pos_free = True
                for p in positions:
                    if np.linalg.norm(obj_pos - p) <= 0.5 * self.maze.maze_size_scaling:
                        pos_free = False
            positions.append(obj_pos)
            obj_qpos_idx = 2 + o * 7
            self.point_env.init_qpos[obj_qpos_idx:obj_qpos_idx+2] = obj_pos
        obs, info = self.point_env.reset(seed=seed)
        obs_dict = self._get_obs(obs)
        info["success"] = bool(
            np.linalg.norm(obs_dict["achieved_goal"] - self.goal) <= self.distance_threshold
        )

        return obs_dict, info