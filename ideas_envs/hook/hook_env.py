import os
from ideas_envs.blocks.blocks_env import BlocksEnv

MODEL_XML_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'hook.xml')


class HookEnv(BlocksEnv):
    """
    This environment enhances the Blocks environment with a hook.
    The hook can be used to pull objects that are out of reach.
    """
    def __init__(self, n_objects, gripper_goal, reward_type='sparse', model_xml_path=MODEL_XML_PATH):
        """
        :param n_objects: How many blocks should be stacked
        :param gripper_goal: 3 possibilities:
            gripper_none: The position of the gripper is not relevant for the goal
            gripper_random: The gripper should reach a random position after stacking the blocks
            gripper_above: The gripper should be above the stacked blocks
        :param reward_type: whether the reward should be sparse or dense
        :param model_xml_path: The path to the XML that defines the MuJoCo environment
        """
        super().__init__(n_objects, gripper_goal, reward_type, model_xml_path)
