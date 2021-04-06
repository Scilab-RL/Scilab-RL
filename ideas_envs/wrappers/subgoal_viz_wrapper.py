from gym.core import Wrapper

SITE_COLORS = [1, 0, 0, 0.3,
               0, 1, 0, 0.3,
               0, 0, 1, 0.3,
               1, 1, 0, 0.3,
               1, 0, 1, 0.3,
               0, 1, 1, 0.3,
               1, 0.5, 0, 0.3,
               1, 0, 0.5, 0.3,
               0.5, 1, 0.5, 0.3,
               0, 0.5, 1, 0.3]


class SubgoalVisualizationWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.subgoals = {}

    def render(self, mode='human', **kwargs):
        self._prepare_subgoals()
        return self.env.render(mode, **kwargs)

    def _prepare_subgoals(self):
        for name in self.subgoals:
            try:
                site_id = self.sim.model.site_name2id(name)
                self.sim.model.site_pos[site_id] = self.subgoals[name][0].copy()
                size = [self.subgoals[name][1]]*3
                self.sim.model.site_size[site_id] = size
                self.sim.model.site_rgba[site_id] = self.subgoals[name][2]
            except ValueError as e:
                raise ValueError("Site {} does not exist. Please include the ideas_envs.assets.subgoal_viz.xml "
                                 "in your environment xml.".format(name)) from e

    def display_subgoals(self, subgoals, shape='sphere', size=0.025, colors=None):
        """
            :param subgoals is a one dimensional array with the subgoal positions
                            with the shape: [x, y, z, x, y, z, ...]
            :param shape is the geometric shape of the subgoal visualization site
                            it can be either 'sphere', 'box' or 'cylinder'
            :param size is the size of the subgoal visualization site
            :param colors is a list of colors for the visualizations
                            with the shape: [r, g, b, a, r, g, b, a, ...]
        """
        if hasattr(self.env.unwrapped, 'display_subgoals'):
            self.env.unwrapped.display_subgoals(subgoals)
            return
        assert len(subgoals) % 3 == 0, "The subgoals must be provided in the form [x, y, z, x, y, z, ...]"
        if colors is None:
            colors = SITE_COLORS[:int((len(subgoals)/3) * 4)]
        for i in range(int(len(subgoals)/3)):
            self.subgoals['subgoal_{}{}'.format(shape, i)] = (subgoals[i*3: (i+1)*3], size, colors[i*4: (i+1)*4])
