import matplotlib
import sys
from math import ceil,sqrt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import numpy as np
from pathlib import Path

cwd = os.getcwd()


class LiveAnimationPlot:
    def __init__(
            self,
            x_axis_labels=["Rollout_step"],
            y_axis_labels=["recorded_value"]
    ):
        self.x_axis_labels = x_axis_labels * len(y_axis_labels)
        self.y_axis_labels = y_axis_labels
        self.num_metrics = len(y_axis_labels)
        self.x_data = [[] for _ in range(self.num_metrics)]
        self.y_data = [[] for _ in range(self.num_metrics)]
        self.fig, self.axs = plt.subplots(nrows=ceil(sqrt(self.num_metrics)), ncols=ceil(sqrt(self.num_metrics)),constrained_layout=True)

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
            # ax_i = ax_i.gca()
            ax_i.relim()
            ax_i.autoscale()
        return self.lines

    def start_animation(self):
        for i, ax_i in enumerate(self.axs):
            ax_i.set_xlabel(self.x_axis_labels[i])
            ax_i.set_ylabel(self.y_axis_labels[i])
        self.animation = FuncAnimation(self.fig, func=self.animation_frame, frames=20, interval=500, blit=False)
        plt.ion()
        plt.pause(0.01)
        self.fig.set_layout_engine(None)

    def create_to_save_anim(self, j):
        for i, line in enumerate(self.lines):
            line.set_xdata(self.x_data[i][0:i + 1])
            line.set_ydata(self.y_data[i][0:i + 1])
        return self.lines

    def save_animation(self, name=''):
        Path(cwd + '/animations/').mkdir(parents=True, exist_ok=True)
        FFwriter = matplotlib.animation.FFMpegWriter(fps=20, codec="h264")
        for i, ax_i in enumerate(self.axs):
            ax_i.set_xlim(0.0, max(self.x_data[-1]))
            ax_i.set_ylim(0.0, max(self.y_data[i]) * 1.1)
        self.animation = FuncAnimation(self.fig, func=self.create_to_save_anim, frames=80, interval=50, blit=False,
                                       save_count=sys.maxsize)
        self.animation.save(cwd + '/animations/' + name + '.mp4', dpi=350, writer=FFwriter)
