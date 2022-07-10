import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class LiveAnimationPlot:
    def __init__(
            self,
            x_axis_label="Episode_step",
            y_axis_label="recorded_value"
    ):
        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label
        self.x_data = [0]
        self.y_data = [0]
        self.fig, self.ax = plt.subplots()
        #self.ax.set_xlim(0, 200)
        #self.ax.set_ylim(0, 200)
        self.line = plt.plot([])[0]
        self.animation = None

    '''
    def init(self):
        self.line.set_ydata(np.ma.array(x, mask=True))
        return self.line,
    '''
    def animation_frame(self, i):
        self.line.set_xdata(self.x_data)
        self.line.set_ydata(self.y_data)
        plt.xlabel(self.x_axis_label)
        plt.ylabel(self.y_axis_label)
        ax = plt.gca()
        ax.relim()
        ax.autoscale()
        return self.line

    def start_animation(self):
        self.animation = FuncAnimation(self.fig, func=self.animation_frame, frames=10, interval=500, blit=False)
        plt.ion()
        plt.pause(0.01)


