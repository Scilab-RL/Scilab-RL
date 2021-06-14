import os
from shutil import copyfile
import xml.etree.ElementTree
import ideas_envs.wrappers.subgoal_viz_wrapper as svw


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
