import os
import numpy as np
from custom_envs.ant.ant_env import AntEnv

MODEL_XML_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'ant_4_rooms.xml')


class Ant4RoomsEnv(AntEnv):
    def __init__(self, reward_type='sparse', distance_threshold=0.4):
        super().__init__(MODEL_XML_PATH, reward_type, distance_threshold)

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Reset controls
        self.sim.data.ctrl[:] = 0

        # Choose initial start state to be different than room containing the end goal
        # Determine which of four rooms contains goal
        goal_room = 0
        if self.goal[0] < 0 and self.goal[1] > 0:
            goal_room = 1
        elif self.goal[0] < 0 and self.goal[1] < 0:
            goal_room = 2
        elif self.goal[0] > 0 and self.goal[1] < 0:
            goal_room = 3

        # Place ant in room different than room containing goal

        initial_room = self.np_random.randint(0, 4)
        while initial_room == goal_room:
            initial_room = self.np_random.randint(0, 4)

        # Move ant to correct room
        self.sim.data.qpos[0] = self.np_random.uniform(3, 6.5)
        self.sim.data.qpos[1] = self.np_random.uniform(3, 6.5)

        # If goal should be in top left quadrant
        if initial_room == 1:
            self.sim.data.qpos[0] *= -1

        # Else if goal should be in bottom left quadrant
        elif initial_room == 2:
            self.sim.data.qpos[0] *= -1
            self.sim.data.qpos[1] *= -1

        # Else if goal should be in bottom right quadrant
        elif initial_room == 3:
            self.sim.data.qpos[1] *= -1

        self.sim.step()
        return self._get_obs()

    def _sample_goal(self):
        goal = np.zeros(3)

        # Randomly select one of the four rooms in which the goal will be located
        room_num = self.np_random.randint(0, 4)

        # Pick exact goal location
        goal[0] = self.np_random.uniform(3, 6.5)
        goal[1] = self.np_random.uniform(3, 6.5)
        goal[2] = self.np_random.uniform(0.45, 0.55)

        # If goal should be in top left quadrant
        if room_num == 1:
            goal[0] *= -1

        # Else if goal should be in bottom left quadrant
        elif room_num == 2:
            goal[0] *= -1
            goal[1] *= -1

        # Else if goal should be in bottom right quadrant
        elif room_num == 3:
            goal[1] *= -1

        return goal
