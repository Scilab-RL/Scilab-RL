from gym.core import Wrapper


class SubgoalVisualizationWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def render(self, mode='human', **kwargs):
        self._prepare_subgoals()
        return self.env.render(mode, **kwargs)

    def _prepare_subgoals(self):
        print("YO")

    def visualize_subgoals(self, subgoals):
        """
            :param subgoals a one dimensional array with the subgoal positions
            shape: [x, y, z, x, y, z, x, y, z]
        """
        pass
