import matplotlib
from math import ceil,sqrt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

cwd = os.getcwd()


class LiveAnimationPlot:
    def __init__(
            self,
            env,
            x_axis_labels=["Rollout_step"],
            y_axis_labels=["recorded_value"]
    ):
        self.frames_per_sec = env.metadata.get("video.frames_per_second", 30)
        self.output_frames_per_sec = env.metadata.get(
            "video.output_frames_per_second", self.frames_per_sec
        )
        self.x_axis_labels = x_axis_labels * len(y_axis_labels)
        self.y_axis_labels = y_axis_labels
        self.num_metrics = len(y_axis_labels)
        self.x_data = [[] for _ in range(self.num_metrics)]
        self.y_data = [[] for _ in range(self.num_metrics)]
        rows = ceil(sqrt(self.num_metrics))
        self.fig, self.axs = plt.subplots(nrows=rows,
                                          ncols=self.num_metrics // rows,
                                          constrained_layout=True,
                                          figsize=[5, 5])

        # for n = 1, matplotlib returns an ax object without an array
        if self.num_metrics == 1:
            self.axs = [self.axs]
        else:
            self.axs = self.axs.flatten()
        self.axs = self.axs[:self.num_metrics]

        self.lines = [ax_i.plot([])[0] for ax_i in self.axs]
        self.animation = None

    def reset_fig(self):
        for ax_i in self.axs:
            ax_i.cla()
        self.lines = [ax_i.plot([])[0] for ax_i in self.axs]
        self.animation = None

    def animation_frame(self, j):
        for i, line in enumerate(self.lines):
            line.set_xdata(self.x_data[i])
            line.set_ydata(self.y_data[i])
        for ax_i in self.axs:
            ax_i.relim()
            ax_i.autoscale()
        return self.lines

    def start_animation(self):
        for i, ax_i in enumerate(self.axs):
            ax_i.set_xlabel(self.x_axis_labels[i])
            ax_i.set_ylabel(self.y_axis_labels[i])
        self.animation = FuncAnimation(self.fig, func=self.animation_frame, interval=500, blit=False)
        plt.ion()
        plt.pause(0.01)
        #self.fig.set_layout_engine(None)

    def create_to_save_anim(self, j):
        for i, line in enumerate(self.lines):
            line.set_xdata(self.x_data[i][0:j + 1])
            line.set_ydata(self.y_data[i][0:j + 1])
        return self.lines

    def save_animation(self, base_path):
        FFwriter = matplotlib.animation.FFMpegWriter(fps=self.output_frames_per_sec, codec="h264")
        # PillowWriter = matplotlib.animation.PillowWriter(fps=self.output_frames_per_sec, codec="h264")
        for i, ax_i in enumerate(self.axs):
            y_range = max(self.y_data[i]) - min(self.y_data[i])
            x_range = max(self.x_data[i]) - min(self.x_data[i])
            ax_i.set_xlim(min(self.x_data[i]) - x_range * 0.05, max(self.x_data[i]) + x_range * 0.05)
            ax_i.set_ylim(min(self.y_data[i]) - y_range * 0.05, max(self.y_data[i]) + y_range * 0.05)
            ax_i.set_xlabel(self.x_axis_labels[i])
            ax_i.set_ylabel(self.y_axis_labels[i])
        self.animation = FuncAnimation(self.fig, func=self.create_to_save_anim, blit=False, frames=len(self.x_data[0]))
        self.animation.save(base_path + '.mp4', dpi=100, writer=FFwriter)
