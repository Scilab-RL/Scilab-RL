from gym.core import Wrapper
# During hyperopt, this module is reimported after env-registration. Therefore, the env-registration has to be repeated:
import custom_envs.register_envs

SITE_COLORS = [1, 0, 0, 0.2,
               0, 1, 0, 0.2,
               0, 0, 1, 0.2,
               1, 1, 0, 0.2,
               1, 0, 1, 0.2,
               0, 1, 1, 0.2,
               1, 0.5, 0, 0.2,
               1, 0, 0.5, 0.2,
               0.5, 1, 0.5, 0.2,
               0, 0.5, 1, 0.2]


class ObjectVisualizationWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.objects = {}

    def render(self, mode='human', **kwargs):
        self._prepare_objects()
        return self.env.render(mode, **kwargs)

    def _prepare_objects(self):
        for name in self.objects:
            try:
                site_id = self.sim.model.site_name2id(name)
                self.sim.model.site_pos[site_id] = self.objects[name][0].copy()
                size = [self.objects[name][1]]*3
                self.sim.model.site_size[site_id] = size
                self.sim.model.site_rgba[site_id] = self.objects[name][2]
            except ValueError as e:
                raise ValueError("Site {} does not exist. Please include the custom_envs.assets.subgoal_viz.xml "
                                 "in your environment xml.".format(name)) from e

    def display_objects(self, objects, shape='sphere', size=0.025, colors=None, alpha = 1):
        """
            :param objects is a one dimensional array with the object positions
                            with the shape: [x, y, z, x, y, z, ...]
            :param shape is the geometric shape of the object visualization site
                            it can be either 'sphere', 'box' or 'cylinder'
            :param size is the size of the object visualization site
            :param colors is a list of colors for the visualizations
                            with the shape: [r, g, b, a, r, g, b, a, ...]
            :param alpha is the value of the opacity of the object and must be between 0 and 1
        """
        assert 0 <= alpha <= 1, "Alpha(Opacity) must be between 0 and 1."
        if hasattr(self.env.unwrapped, 'display_subgoals'):
            self.env.unwrapped.display_objects(objects)
            return
        assert len(objects) % 3 == 0, "The objects must be provided in the form [x, y, z, x, y, z, ...]"
        colors = [color * alpha for color in colors]
        if colors is None:
            colors = SITE_COLORS[:int((len(objects)/3) * 4)] * alpha
        for i in range(int(len(objects)/3)):
            self.objects['object_{}{}'.format(shape, i)] = (objects[i*3: (i+1)*3], size, colors[i*4: (i+1)*4])

