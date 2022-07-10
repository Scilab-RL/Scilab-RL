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
                raise ValueError("Site {} does not exist. Please include the custom_envs.assets.subgoal_viz.xml "
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
import os
from shutil import copyfile
import xml.etree.ElementTree
import custom_envs.wrappers.subgoal_viz_wrapper as svw


def goal_viz_for_gym_robotics():
    """
    patch the gym.envs.robotics environment.xml files so that we can render subgoals
    Only if MuJoCo is installed
    """
    try:
        import mujoco_py
        import gym.envs.robotics.fetch.reach as reach
    except ImportError as e:
        return
    base_path = reach.__file__[:-14]+'assets/'
    # copy the goal visualization XML to the gym.envs.robotics.assets directory
    # If it is already there, assume the required changes have already been made and return
    if os.path.exists(base_path+'subgoal_viz.xml'):
        return

    copyfile(svw.__file__[:-31]+'assets/subgoal_viz.xml', base_path+'subgoal_viz.xml')

    xml_names = ['fetch/pick_and_place.xml', 'fetch/push.xml', 'fetch/reach.xml', 'fetch/slide.xml',
                 'hand/manipulate_block.xml', 'hand/manipulate_block_touch_sensors.xml',
                 'hand/manipulate_egg.xml', 'hand/manipulate_egg_touch_sensors.xml',
                 'hand/manipulate_pen.xml', 'hand/manipulate_pen_touch_sensors.xml',
                 'hand/reach.xml']
    for xml_name in xml_names:
        path = base_path + xml_name
        # get the xml
        et = xml.etree.ElementTree.parse(path)

        # include the subgoal_viz.xml
        worldbody = et.find(".//worldbody")
        include_element = xml.etree.ElementTree.Element('include', {'file': '../subgoal_viz.xml'})
        worldbody.append(include_element)

        # overwrite the old xml
        et.write(path)


if __name__ == '__main__':
    goal_viz_for_gym_robotics()
