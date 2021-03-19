from gym.core import Wrapper

SPHERE, BOX = 'sphere', 'box'


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
                self.sim.model.site_rgba[site_id][3] = 0.3
            except ValueError as e:
                raise ValueError("Site {} does not exist. Please include the ideas_envs.assets.subgoal_viz.xml "
                                 "in your environment xml.".format(name)) from e

    def display_subgoals(self, subgoals, form=SPHERE, size=0.025):
        """
            :param subgoals a one dimensional array with the subgoal positions
            shape: [x, y, z, x, y, z, ...]
        """
        assert len(subgoals) % 3 == 0, "The subgoals must be provided in the form [x, y, z, x, y, z, ...]"
        if hasattr(self.env.unwrapped, 'display_subgoals'):
            self.env.unwrapped.display_subgoals(subgoals)
        for i in range(int(len(subgoals)/3)):
            self.subgoals['subgoal_{}{}'.format(form, i)] = (subgoals[i*3: (i+1)*3], size)
