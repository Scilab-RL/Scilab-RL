import os
import subprocess
import gymnasium as gym

from typing import Callable
from utils.animation_util import LiveAnimationPlot
from gymnasium.envs.mujoco import MujocoEnv

from gymnasium.wrappers.monitoring import video_recorder
from moviepy.editor import VideoFileClip, clips_array

def recursive_set_render_mode(env, mode):
    """
    Sets the render mode for the environment object and all environments that are part of the object.
    """
    try:
        env.unwrapped.render_mode = mode
        env_dict = vars(env.unwrapped)
        for k,v in env_dict.items():
            if isinstance(v, MujocoEnv):
                if k != "unwrapped":
                    recursive_set_render_mode(v, mode)
    except Exception as e:
        print(f"{e}. Error, no valid environment provided")



class DisplayWrapper(gym.Wrapper):
    """
    Display episodes based on step_trigger, episode_trigger or episode_in_epoch_trigger.
    step_trigger uses step_id,
    episode_trigger uses episode_id,
    episode_of_epoch_trigger uses epoch_id and episode_in_epoch_id
    """
    def __init__(
        self,
        env,
        steps_per_epoch = None,
        episode_in_epoch_trigger: Callable[[int], bool] = None,
        episode_trigger: Callable[[int], bool] = None,
        step_trigger: Callable[[int], bool] = None,
        metric_keys = [],
        logger = None,
    ):
        super().__init__(env)

        trigger_count = sum([x is not None for x in [episode_in_epoch_trigger, episode_trigger, step_trigger]])
        assert trigger_count == 1, "Must specify exactly one trigger"
        steps_per_epoch_trigger_count = sum([x is not None for x in [episode_in_epoch_trigger, steps_per_epoch]])
        assert steps_per_epoch_trigger_count != 1, "If episode_in_epoch_trigger is used, steps_per_epoch must be specified"

        self.episode_in_epoch_trigger = episode_in_epoch_trigger
        self.episode_trigger = episode_trigger
        self.step_trigger = step_trigger

        self.step_in_episode_id = 0
        self.episode_in_epoch_id = 0
        self.epoch_id = 0
        self.episode_id = 0
        self.step_id = 0

        self.steps_per_epoch = steps_per_epoch

        self.displaying = False
        self.is_vector_env = getattr(env, "is_vector_env", False)

        self.metric_keys = metric_keys
        self.num_metrics = len(self.metric_keys)
        self.display_metrics = self.num_metrics > 0
        self.animation = LiveAnimationPlot(y_axis_labels=self.metric_keys,
                                           env=self.env) if self.display_metrics else None
        self.logger = logger
        recursive_set_render_mode(self.env, 'human')


    def reset(self, **kwargs):
        observations = self.env.reset(**kwargs)
        if not self.displaying and self._display_enabled():
            self.start_displayer()
        return observations

    def start_displayer(self):
        self.close_displayer()
        # Might be an option to put metric display init here
        self.displaying = True

    def _display_enabled(self):
        if self.step_trigger:
            return self.step_trigger(self.step_id)
        elif self.episode_in_epoch_trigger:
            return self.episode_in_epoch_trigger(self.episode_in_epoch_id, self.epoch_id)
        else:
            return self.episode_trigger(self.episode_id)

    def step(self, action):
        observations, rewards, terminated, truncated, infos = self.env.step(action)

        dones = terminated | truncated

        # increment steps, episodes and epochs
        if self.steps_per_epoch:
            epoch_id_tmp = self.epoch_id
            self.epoch_id = self.step_id // self.steps_per_epoch
            if self.epoch_id > epoch_id_tmp:
                self.episode_in_epoch_id = 0
        self.step_id += 1
        self.step_in_episode_id +=1
        if not self.is_vector_env:
            if dones:
                self.episode_id += 1
                if self.steps_per_epoch:
                    self.episode_in_epoch_id += 1
        elif dones[0]:
            self.episode_id += 1
            if self.steps_per_epoch:
                self.episode_in_epoch_id += 1

        if self.displaying:
            self.env.render()
            # metrics stuff
            if self.display_metrics:
                for i in range(self.num_metrics):
                    self.curr_recorded_value = self.logger.name_to_value[self.metric_keys[i]]
                    if len(self.animation.x_data[i]) > 0:
                        self.animation.x_data[i].append(self.animation.x_data[i][-1]+1)
                    else:
                        self.animation.x_data[i].append(self.step_in_episode_id)
                    self.animation.y_data[i].append(self.curr_recorded_value)
                self.animation.start_animation()

            if not self.is_vector_env:
                if dones:
                    self.close_displayer()
            elif dones[0]:
                self.close_displayer()

        elif self._display_enabled():
            self.start_displayer()

        if not self.is_vector_env:
            if dones:
                self.step_in_episode_id = 0
        elif dones[0]:
            self.step_in_episode_id = 0
        return observations, rewards, terminated, truncated, infos

    def close_displayer(self) -> None:
        if self.displaying:
            # Metric stuff
            if self.display_metrics:
                self.animation.reset_fig()
                # reset data
                self.animation.x_data = [[] for _ in range(len(self.metric_keys))]
                self.animation.y_data = [[] for _ in range(len(self.metric_keys))]
            #close metric displayer
        self.displaying = False


class RecordVideo(gym.Wrapper):
    def __init__(
        self,
        env,
        video_folder: str,
        steps_per_epoch = None,
        episode_in_epoch_trigger: Callable[[int], bool] = None,
        episode_trigger: Callable[[int], bool] = None,
        step_trigger: Callable[[int], bool] = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
        metric_keys = [],
        logger = None,
    ):
        super(RecordVideo, self).__init__(env)

        trigger_count = sum([x is not None for x in [episode_in_epoch_trigger, episode_trigger, step_trigger]])
        assert trigger_count == 1, "Must specify exactly one trigger"
        steps_per_epoch_trigger_count = sum([x is not None for x in [episode_in_epoch_trigger, steps_per_epoch]])
        assert steps_per_epoch_trigger_count != 1, "If episode_in_epoch_trigger is used, steps_per_epoch must be specified"

        self.episode_in_epoch_trigger = episode_in_epoch_trigger
        self.episode_trigger = episode_trigger
        self.step_trigger = step_trigger

        self.step_in_episode_id = 0
        self.episode_in_epoch_id = 0
        self.epoch_id = 0
        self.episode_id = 0
        self.step_id = 0

        self.steps_per_epoch = steps_per_epoch

        self.video_recorder = None

        self.video_folder = os.path.abspath(video_folder)
        # Create output folder if needed
        if os.path.isdir(self.video_folder):
            logger.warn(
                f"Overwriting existing videos at {self.video_folder} folder (try specifying a different `video_folder` for the `RecordVideo` wrapper if this is not desired)"
            )
        os.makedirs(self.video_folder, exist_ok=True)
        self.base_path = None

        self.name_prefix = name_prefix
        episode_length = env.spec.max_episode_steps
        assert episode_length <= video_length, "Video length (render_frames_per_clip in main.yaml) must be at least the number of steps in one episode"

        self.video_length = video_length
        self.recording = False
        self.recorded_frames = 0
        self.is_vector_env = getattr(env, "is_vector_env", False)

        self.metric_keys = metric_keys
        self.num_metrics = len(self.metric_keys)
        self.record_metrics = self.num_metrics > 0
        self.animation = LiveAnimationPlot(y_axis_labels=self.metric_keys,
                                           env=self.env, ) if self.record_metrics else None
        self.logger = logger
        recursive_set_render_mode(self.env, 'rgb_array')

    def reset(self, **kwargs):
        observations = super(RecordVideo, self).reset(**kwargs)
        if not self.recording and self._video_enabled():
            self.start_video_recorder()
        return observations

    def start_video_recorder(self):
        self.close_video_recorder()

        if self.step_trigger:
            video_name = f"{self.name_prefix}-step-{self.step_id}"
        elif self.episode_in_epoch_trigger:
            video_name = f"{self.name_prefix}-epochs-{self.epoch_id}"
        else:
            video_name = f"{self.name_prefix}-episode-{self.episode_id}"

        self.base_path = os.path.join(self.video_folder, video_name)
        # self.env.unwrapped.render_mode = 'rgb_array'
        self.video_recorder = video_recorder.VideoRecorder(
            env=self.env,
            base_path=self.base_path,
            metadata={"step_id": self.step_id, "episode_id": self.episode_id},
        )

        # self.video_recorder.capture_frame()
        # if self.record_metrics:
        #     for i in range(self.num_metrics):
        #         self.animation.x_data[i].append(self.step_in_episode_id)
        #         # if self.logger.name_to_count[self.metric_keys[i]] == 0:
        #         #     self.animation.y_data[i].append(None)
        #         # else:
        #         self.curr_recorded_value = self.logger.name_to_value[self.metric_keys[i]]
        #         self.animation.y_data[i].append(self.curr_recorded_value)
        self.recorded_frames = 0
        self.recording = True

    def _video_enabled(self):
        if self.step_trigger:
            return self.step_trigger(self.step_id)
        elif self.episode_in_epoch_trigger:
            return self.episode_in_epoch_trigger(self.episode_in_epoch_id, self.epoch_id)
        else:
            return self.episode_trigger(self.episode_id)

    def step(self, action):
        observations, rewards, terminated, truncated, infos = super(RecordVideo, self).step(action)

        dones = terminated | truncated

        # increment steps, episodes and epochs
        if self.steps_per_epoch:
            epoch_id_tmp = self.epoch_id
            self.epoch_id = self.step_id // self.steps_per_epoch
            if self.epoch_id > epoch_id_tmp:
                self.episode_in_epoch_id = 0
        self.step_id += 1
        self.step_in_episode_id +=1
        if not self.is_vector_env:
            if dones:
                self.episode_id += 1
                if self.steps_per_epoch:
                    self.episode_in_epoch_id += 1
        elif dones[0]:
            self.episode_id += 1
            if self.steps_per_epoch:
                self.episode_in_epoch_id += 1

        if self.recording:
            # metrics stuff
            if self.record_metrics:
                for i in range(self.num_metrics):
                    self.curr_recorded_value = self.logger.name_to_value[self.metric_keys[i]]
                    if len(self.animation.x_data[i]) > 0:
                        self.animation.x_data[i].append(self.animation.x_data[i][-1]+1)
                    else:
                        self.animation.x_data[i].append(self.step_in_episode_id)
                    # self.animation.x_data[i].append(self.step_in_episode_id)
                    self.animation.y_data[i].append(self.curr_recorded_value)
            self.video_recorder.capture_frame()
            self.recorded_frames += 1
            if self.video_length > 0:
                if self.recorded_frames > self.video_length:
                    self.close_video_recorder()
            else:
                if not self.is_vector_env:
                    if dones:
                        self.close_video_recorder()
                elif dones[0]:
                    self.close_video_recorder()

        elif self._video_enabled():
            self.start_video_recorder()

        if not self.is_vector_env:
            if dones:
                self.step_in_episode_id = 0
        elif dones[0]:
            self.step_in_episode_id = 0

        return observations, rewards, terminated, truncated, infos

    def close_video_recorder(self) -> None:
        if self.recording:
            render_frames = len(self.video_recorder.recorded_frames)
            self.video_recorder.close()
            # Metric stuff
            if self.record_metrics:
                metrics_frames = len(self.animation.y_data[0])
                assert render_frames==metrics_frames, "Error, number of frames of rendered video not the same as number of frames of metrics video."
                self.animation.save_animation(self.base_path + ".metric")
                self.animation.reset_fig()
                # reset data
                self.animation.x_data = [[] for _ in range(len(self.metric_keys))]
                self.animation.y_data = [[] for _ in range(len(self.metric_keys))]
                self.join_animation()
        self.recording = False
        self.recorded_frames = 0

    def join_animation(self):
        self.cmdline = (
            "ffmpeg",
            "-nostats",
            "-loglevel",
            "error",  # suppress warnings
            "-y",
            # input
            "-i",
            self.base_path + ".mp4",
            "-i",
            self.base_path + ".metric.mp4",
            # output
            "-filter_complex",
            "hstack",
            self.base_path + ".joint.mp4",
        )

        metric_clip = VideoFileClip(self.base_path + ".metric.mp4")
        metric_n_frames = int(metric_clip.fps * metric_clip.duration)

        render_clip = VideoFileClip(self.base_path + ".mp4")
        render_n_frames = int(render_clip.fps * render_clip.duration)
        metric_clip = metric_clip.set_duration(render_clip.duration)
        metric_n_frames = int(metric_clip.fps * metric_clip.duration)

        joint_clip = clips_array([[render_clip, metric_clip]])
        joint_clip.write_videofile(self.base_path + ".joint.mp4")
