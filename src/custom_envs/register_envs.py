"""
All custom environments must be registered here, otherwise they won't be found.
"""
from gymnasium.envs.registration import register
import highway_env
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from utils.custom_wrappers import MakeDictObs
RESET = R = "r"  # Initial Reset position of the agent
GOAL = G = "g"
COMBINED = C = "c"  # These cells can be selected as goal or reset locations

class MazeMap:

    OPEN = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ]
    OPEN_DIVERSE_G = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, R, G, G, G, G, 1],
        [1, G, G, G, G, G, 1],
        [1, G, G, G, G, G, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ]
    OPEN_DIVERSE_GR = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, C, C, C, C, C, 1],
        [1, C, C, C, C, C, 1],
        [1, C, C, C, C, C, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ]
    SMALL_OPEN_DIVERSE_GR = [
        [1, 1, 1, 1, 1],
        [1, C, C, C, 1],
        [1, C, C, C, 1],
        [1, C, C, C, 1],
        [1, 1, 1, 1, 1],
    ]
    SMALL_OPEN_DIVERSE_G = [
        [1, 1, 1, 1, 1],
        [1, G, G, G, 1],
        [1, G, G, G, 1],
        [1, G, G, G, 1],
        [1, 1, 1, 1, 1],
    ]
    MEDIUM_CUSTOM_DIVERSE_GR = [[1, 1, 1, 1, 1, 1, 1, 1],
                              [1, C, C, 1, 1, C, C, 1],
                              [1, C, C, 1, C, C, C, 1],
                              [1, 1, C, C, C, 1, 1, 1],
                              [1, C, C, 1, C, C, C, 1],
                              [1, C, 1, C, C, 1, C, 1],
                              [1, C, C, C, 1, C, C, 1],
                              [1, 1, 1, 1, 1, 1, 1, 1]]
    name2map = {"open": OPEN,
                "open_dg": OPEN_DIVERSE_G,
                "open_dgr": OPEN_DIVERSE_GR,
                "small_open_dg": SMALL_OPEN_DIVERSE_G,
                "small_open_dgr": SMALL_OPEN_DIVERSE_GR,
                "medium_custom_dgr": MEDIUM_CUSTOM_DIVERSE_GR,
                }

def _merge(a, b):
    a.update(b)
    return a

def register_custom_envs():
    for n_objects in range(5):
        for gripper_goal in ['gripper_none', 'gripper_random', 'gripper_above']:
            if gripper_goal != 'gripper_random' and n_objects == 0:  # Disallow because there would be no goal
                continue
            distance_threshold = 0.05  # was originally 0.05
            register(id=f'Blocks-o{n_objects}-{gripper_goal}-v1',
                     entry_point='custom_envs.blocks.blocks_env:BlocksEnv',
                     kwargs={'n_objects': n_objects, 'gripper_goal': gripper_goal, 'distance_threshold': distance_threshold},
                     max_episode_steps=max(50, 50*n_objects))

    ## Custom Ant environments
    for reward_type in ["sparse", "sparseneg", "dense"]:
        for fs in [5,10,15,20]:
            for dt in [0.5,1.0,1.5]:
                for map in MazeMap.name2map.keys():
                    for continuing_task in [1, 0]:
                        for reset_target in [1, 0]:
                                for max_ep_Steps in [300, 500, 700]:
                                    kwargs = {
                                        "reward_type": reward_type,
                                        'frame_skip': fs,
                                        "distance_threshold": dt,
                                        "maze_map": MazeMap.name2map[map],
                                        "continuing_task": continuing_task,
                                        "reset_target": reset_target,
                                    }
                                    register(id=f'AntGym-{reward_type}-{fs}-{dt}-{map}-c{continuing_task}-rt{reset_target}-s{max_ep_Steps}-v0',
                                        entry_point='custom_envs.maze.ant_env:AntGymMod',
                                        kwargs = kwargs,
                                        max_episode_steps = max_ep_Steps,
                                        )
    ## Custom PointMaze environments
    for reward_type in ["sparse", "sparseneg", "dense"]:
            for dt in [0.5, 1.0, 1.5]:
                for map in MazeMap.name2map.keys():
                    for continuing_task in [1, 0]:
                        for reset_target in [1, 0]:
                            for max_ep_Steps in [300, 500, 700]:
                                kwargs = {
                                    "reward_type": reward_type,
                                    "distance_threshold": dt,
                                    "maze_map": MazeMap.name2map[map],
                                    "continuing_task": continuing_task,
                                    "reset_target": reset_target,
                                }
                                register(
                                    id=f'PointGym-{reward_type}-{dt}-{map}-c{continuing_task}-rt{reset_target}-s{max_ep_Steps}-v0',
                                    entry_point='custom_envs.maze.point_env:PointGymMod',
                                    kwargs=kwargs,
                                    max_episode_steps=max_ep_Steps,
                                    )

    register(id='Reach1DOF-v0',
             entry_point='custom_envs.reach1dof.reach1dof_env:Reach1DOFEnv',
             max_episode_steps=50)

    for n_objects in range(3):
        register(id=f'Hook-o{n_objects}-v1',
                 entry_point='custom_envs.hook.hook_env:HookEnv',
                 kwargs={'n_objects': n_objects},
                 max_episode_steps=max(50, 100 * n_objects))

        register(id=f'ButtonUnlock-o{n_objects}-v1',
                 entry_point='custom_envs.button_unlock.button_unlock_env:ButtonUnlockEnv',
                 kwargs={'n_buttons': n_objects+1},
                 max_episode_steps=max(50, 50*n_objects))

    register(
        id='parking-limited-v0',
        entry_point='highway_env.envs:ParkingEnv',
        max_episode_steps=100,
    )

    register_metaworld_envs()


def register_metaworld_envs():
    for env_name, env_class in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE.items():
        for env_type in ["original", "sparse", "dense"]:
            """
            original - don't use the MakeDictObs wrapper
            sparse - use the MakeDictObs wrapper and sparse rewards
            dense - use the MakeDictObs wrapper and dense rewards
            """
            def make_variable_goal_env(environment_class, environment_type):
                def variable_goal_env(**kwargs):
                    """
                    set _freeze_rand_vec to False after instantiation so that the goal is not always the same.
                    """
                    env = environment_class(**kwargs)
                    env._freeze_rand_vec = False
                    if environment_type == "original":
                        pass
                    elif environment_type == "sparse":
                        env = MakeDictObs(env, dense=False)
                    elif environment_type == "dense":
                        env = MakeDictObs(env, dense=True)
                    else:
                        raise ValueError(f"unknown environment type {environment_type}")
                    return env

                return variable_goal_env

            if env_type == "original":
                register(id=f"MetaW-{env_name[:-len('-goal-observable')]}", entry_point=make_variable_goal_env(env_class, env_type), max_episode_steps=500)
            elif env_type == "sparse":
                register(id=f"MetaW-{env_name[:-len('-goal-observable')]}-sparse", entry_point=make_variable_goal_env(env_class, env_type), max_episode_steps=500)
            elif env_type == "dense":
                register(id=f"MetaW-{env_name[:-len('-goal-observable')]}-dense", entry_point=make_variable_goal_env(env_class, env_type), max_episode_steps=500)
