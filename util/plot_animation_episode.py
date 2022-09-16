import matplotlib
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from pathlib import Path

cwd = os.getcwd()


class LiveAnimationPlot:
    def __init__(
            self,
            x_axis_label="Episode_step",
            y_axis_label="recorded_value"
    ):
        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label
        self.x_data = []
        self.y_data = []
        self.fig, self.ax = plt.subplots()
        self.line = plt.plot([])[0]
        self.animation = None

    def init_func(self):
        self.line.set_xdata(self.x_data)
        self.line.set_ydata(self.y_data)
        return self.line

    def reset_fig(self):
        plt.clf()
        #plt.close(self.fig)
        #self.fig, self.ax = plt.subplots()
        self.line = plt.plot([])[0]
        self.animation = None

    def animation_frame(self, i):
        self.line.set_xdata(self.x_data)
        self.line.set_ydata(self.y_data)
        ax = plt.gca()
        ax.relim()
        ax.autoscale()
        return self.line

    def start_animation(self):
        plt.xlabel(self.x_axis_label)
        plt.ylabel(self.y_axis_label)
        self.animation = FuncAnimation(self.fig, func=self.animation_frame, frames=20, interval=500, blit=False)
        plt.ion()
        plt.pause(0.01)


    def create_to_save_anim(self, i):
        self.line.set_xdata(self.x_data[0:i+1])
        self.line.set_ydata(self.y_data[0:i+1])
        return self.line

    def save_animation(self, name):
        # writervideo = matplotlib.animation.FFMpegWriter(fps=60)
        Path(cwd + '/animations/').mkdir(parents=True, exist_ok=True)
        FFwriter = matplotlib.animation.FFMpegWriter(fps=20, codec="h264")
        plt.xlim(0.0, max(self.x_data))
        plt.ylim(0.0, max(self.y_data)*1.1)
        self.animation = FuncAnimation(self.fig, func=self.create_to_save_anim, frames=80, interval=50, blit=False,save_count=sys.maxsize)
        self.animation.save(cwd + '/animations/' + name + '.mp4', dpi=350, writer=FFwriter)
