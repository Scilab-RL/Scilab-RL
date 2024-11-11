"""
All custom environments must be registered here, otherwise they won't be found.
"""
import csv
import ast
from custom_envs import ROOT_DIR
from gymnasium.envs.registration import register
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
                     kwargs={'n_objects': n_objects, 'gripper_goal': gripper_goal,
                             'distance_threshold': distance_threshold},
                     max_episode_steps=max(50, 50 * n_objects))

    ## Custom Ant environments
    for reward_type in ["sparse", "sparseneg", "dense"]:
        for fs in [5, 10, 15, 20]:
            for dt in [0.5, 1.0, 1.5]:
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
                                register(
                                    id=f'AntGym-{reward_type}-{fs}-{dt}-{map}-c{continuing_task}-rt{reset_target}-s{max_ep_Steps}-v0',
                                    entry_point='custom_envs.maze.ant_env:AntGymMod',
                                    kwargs=kwargs,
                                    max_episode_steps=max_ep_Steps,
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
                            for n_obj in range(5):
                                obj_kwargs = kwargs.copy()
                                obj_kwargs['n_objects'] = n_obj
                                register(
                                    id=f'PointObjGym-{reward_type}-o{n_obj}-{dt}-{map}-c{continuing_task}-rt{reset_target}-s{max_ep_Steps}-v0',
                                    entry_point='custom_envs.maze.point_obj_env:PointObjEnv',
                                    kwargs=obj_kwargs,
                                    max_episode_steps=max_ep_Steps,
                                )
    # kwargs = {
    #     "reward_type": reward_type,
    #     "distance_threshold": 0.45,
    #     "maze_map": MazeMap.name2map["small_open_dgr"],
    #     "continuing_task": 1,
    #     "reset_target": 0,
    #     "n_objects": 4,
    # }
    # register(
    #     id=f'PointObjGym-v0',
    #     entry_point='custom_envs.maze.point_obj_env:PointObjEnv',
    #     kwargs=kwargs,
    #     max_episode_steps=700,
    # )

    register(id='Reach1DOF-v0',
             entry_point='custom_envs.reach1dof.reach1dof_env:Reach1DOFEnv',
             max_episode_steps=50)

    register(id="MoonlanderWorld-dodge-simple-v0",
             entry_point="custom_envs.moonlander.moonlander_env:MoonlanderWorldEnv",
             kwargs={'task': 'dodge', 'reward_function': 'simple'},
             max_episode_steps=500)
    register(id="MoonlanderWorld-dodge-gaussian-v0",
             entry_point="custom_envs.moonlander.moonlander_env:MoonlanderWorldEnv",
             kwargs={'task': 'dodge', 'reward_function': 'gaussian'},
             max_episode_steps=500)
    register(id="MoonlanderWorld-dodge-pos_neg-v0",
             entry_point="custom_envs.moonlander.moonlander_env:MoonlanderWorldEnv",
             kwargs={'task': 'dodge', 'reward_function': 'pos_neg'},
             max_episode_steps=500)
    register(id="MoonlanderWorld-collect-simple-v0",
             entry_point="custom_envs.moonlander.moonlander_env:MoonlanderWorldEnv",
             kwargs={'task': 'collect', 'reward_function': 'simple'},
             max_episode_steps=500)
    register(id="MoonlanderWorld-collect-gaussian-v0",
             entry_point="custom_envs.moonlander.moonlander_env:MoonlanderWorldEnv",
             kwargs={'task': 'collect', 'reward_function': 'gaussian'},
             max_episode_steps=500)
    register(id="MoonlanderWorld-collect-pos_neg-v0",
             entry_point="custom_envs.moonlander.moonlander_env:MoonlanderWorldEnv",
             kwargs={'task': 'collect', 'reward_function': 'pos_neg'},
             max_episode_steps=500)

    filename_small = "hard_object_list_10_times_10.csv"
    filename_small_1 = "hard_object_list_10_times_10_1.csv"
    filename_collect_easy = "collect_easy_object_list_30_times_40.csv"
    filename_collect_hard = "collect_hard_object_list_30_times_40.csv"
    filename_dodge_easy = "dodge_easy_object_list_30_times_40.csv"
    filename_dodge_hard = "dodge_hard_object_list_30_times_40.csv"

    list_of_filenames = [filename_small, filename_small_1, filename_collect_easy, filename_collect_hard,
                         filename_dodge_easy, filename_dodge_hard]
    dict_of_filename_to_object_dict_list = {}
    for filename in list_of_filenames:
        list_of_object_dict_lists = []
        with open(ROOT_DIR / "moonlander" / filename, "r") as file:
            lines = csv.reader(file)
            for line in lines:
                # first element is index
                # second element is the object list
                # form string to list of dictionaries
                list_of_object_dict_lists.append(ast.literal_eval(line[1]))
        dict_of_filename_to_object_dict_list[filename] = list_of_object_dict_lists

    # BENCHMARKS
    register(id="MoonlanderWorld-collect-gaussian-benchmark-small-v0",
             entry_point="custom_envs.moonlander.moonlander_env:MoonlanderWorldEnv",
             kwargs={'task': 'collect', 'reward_function': 'gaussian',
                     'list_of_object_dict_lists': dict_of_filename_to_object_dict_list[
                         "hard_object_list_10_times_10.csv"]},
             max_episode_steps=500)
    register(id="MoonlanderWorld-dodge-gaussian-benchmark-small-v0",
             entry_point="custom_envs.moonlander.moonlander_env:MoonlanderWorldEnv",
             kwargs={'task': 'dodge', 'reward_function': 'gaussian',
                     'list_of_object_dict_lists': dict_of_filename_to_object_dict_list[
                         "hard_object_list_10_times_10.csv"]},
             max_episode_steps=500)
    register(id="MoonlanderWorld-collect-gaussian-benchmark-easy-v0",
             entry_point="custom_envs.moonlander.moonlander_env:MoonlanderWorldEnv",
             kwargs={'task': 'collect', 'reward_function': 'gaussian',
                     'list_of_object_dict_lists': dict_of_filename_to_object_dict_list[
                         "collect_easy_object_list_30_times_40.csv"]},
             max_episode_steps=500)
    register(id="MoonlanderWorld-collect-gaussian-benchmark-hard-v0",
             entry_point="custom_envs.moonlander.moonlander_env:MoonlanderWorldEnv",
             kwargs={'task': 'collect', 'reward_function': 'gaussian',
                     'list_of_object_dict_lists': dict_of_filename_to_object_dict_list[
                         "collect_hard_object_list_30_times_40.csv"]},
             max_episode_steps=500)
    register(id="MoonlanderWorld-dodge-gaussian-benchmark-easy-v0",
             entry_point="custom_envs.moonlander.moonlander_env:MoonlanderWorldEnv",
             kwargs={'task': 'dodge', 'reward_function': 'gaussian',
                     'list_of_object_dict_lists': dict_of_filename_to_object_dict_list[
                         "dodge_easy_object_list_30_times_40.csv"]},
             max_episode_steps=500)
    register(id="MoonlanderWorld-dodge-gaussian-benchmark-hard-v0",
             entry_point="custom_envs.moonlander.moonlander_env:MoonlanderWorldEnv",
             kwargs={'task': 'dodge', 'reward_function': 'gaussian',
                     'list_of_object_dict_lists': dict_of_filename_to_object_dict_list[
                         "dodge_hard_object_list_30_times_40.csv"]},
             max_episode_steps=500)

    register(id="MetaEnv-v0",
             entry_point="custom_envs.moonlander.meta_env:MetaEnv",
             kwargs={'reward_function': 'simple'},
             max_episode_steps=500)
    register(id="MetaEnv-gaussian-v0",
             entry_point="custom_envs.moonlander.meta_env:MetaEnv",
             kwargs={'reward_function': 'gaussian'},
             max_episode_steps=500)
    register(id="MetaEnv-pos_neg-v0",
             entry_point="custom_envs.moonlander.meta_env:MetaEnv",
             kwargs={'reward_function': 'pos_neg'},
             max_episode_steps=500)

    register(id="MetaEnv-pretrained-small-v0",
             entry_point="custom_envs.moonlander.meta_env_pretrained:MetaEnvPretrained",
             kwargs={'dodge_best_model_name': "dodge_best_fm_23_08_rl_model_best",
                     'collect_best_model_name': "collect_best_fm_23_08_rl_model_best"},
             max_episode_steps=500)
    register(id="MetaEnv-pretrained-human-v0",
             entry_point="custom_envs.moonlander.meta_env_pretrained:MetaEnvPretrained",
             kwargs={'dodge_best_model_name': "dodge_human_27_09_rl_model_best",
                     'collect_best_model_name': "collect_human_27_09_rl_model_best"},
             max_episode_steps=500)
    register(id="MetaEnv-pretrained-benchmark-small-v0",
             entry_point="custom_envs.moonlander.meta_env_pretrained:MetaEnvPretrained",
             kwargs={'dodge_best_model_name': "dodge_best_fm_23_08_rl_model_best",
                     'collect_best_model_name': "collect_best_fm_23_08_rl_model_best",
                     'dodge_list_of_object_dict_lists': dict_of_filename_to_object_dict_list[
                         "hard_object_list_10_times_10.csv"],
                     'collect_list_of_object_dict_lists': dict_of_filename_to_object_dict_list[
                         "hard_object_list_10_times_10_1.csv"]},
             max_episode_steps=500)
    register(id="MetaEnv-pretrained-benchmark-easy-easy-v0",
             entry_point="custom_envs.moonlander.meta_env_pretrained:MetaEnvPretrained",
             kwargs={'dodge_best_model_name': "dodge_human_size_hard_rl_model_best",
                     'collect_best_model_name': "collect_human_size_hard_rl_model_best",
                     'dodge_list_of_object_dict_lists': dict_of_filename_to_object_dict_list[
                         "dodge_easy_object_list_30_times_40.csv"],
                     'collect_list_of_object_dict_lists': dict_of_filename_to_object_dict_list[
                         "collect_easy_object_list_30_times_40.csv"]},
             max_episode_steps=500)
    register(id="MetaEnv-pretrained-benchmark-easy-hard-v0",
             entry_point="custom_envs.moonlander.meta_env_pretrained:MetaEnvPretrained",
             kwargs={'dodge_best_model_name': "dodge_human_size_hard_rl_model_best",
                     'collect_best_model_name': "collect_human_size_hard_rl_model_best",
                     'dodge_list_of_object_dict_lists': dict_of_filename_to_object_dict_list[
                         "dodge_easy_object_list_30_times_40.csv"],
                     'collect_list_of_object_dict_lists': dict_of_filename_to_object_dict_list[
                         "collect_hard_object_list_30_times_40.csv"]},
             max_episode_steps=500)
    register(id="MetaEnv-pretrained-benchmark-hard-easy-v0",
             entry_point="custom_envs.moonlander.meta_env_pretrained:MetaEnvPretrained",
             kwargs={'dodge_best_model_name': "dodge_human_size_hard_rl_model_best",
                     'collect_best_model_name': "collect_human_size_hard_rl_model_best",
                     'dodge_list_of_object_dict_lists': dict_of_filename_to_object_dict_list[
                         "dodge_hard_object_list_30_times_40.csv"],
                     'collect_list_of_object_dict_lists': dict_of_filename_to_object_dict_list[
                         "collect_easy_object_list_30_times_40.csv"]},
             max_episode_steps=500)
    register(id="MetaEnv-pretrained-benchmark-hard-hard-v0",
             entry_point="custom_envs.moonlander.meta_env_pretrained:MetaEnvPretrained",
             kwargs={'dodge_best_model_name': "dodge_human_size_hard_rl_model_best",
                     'collect_best_model_name': "collect_human_size_hard_rl_model_best",
                     'dodge_list_of_object_dict_lists': dict_of_filename_to_object_dict_list[
                         "dodge_hard_object_list_30_times_40.csv"],
                     'collect_list_of_object_dict_lists': dict_of_filename_to_object_dict_list[
                         "collect_hard_object_list_30_times_40.csv"]},
             max_episode_steps=500)

    register(id="GridworldEnv-v0",
             entry_point="custom_envs.grid_world.grid_world_env:GridWorldEnv",
             max_episode_steps=10)
    register(id="GridworldEnv-v1",
             entry_point="custom_envs.grid_world.grid_world_env:GridWorldEnv",
             kwargs={'is_it_possible_that_input_noise_is_applied': True},
             max_episode_steps=10)
    register(id="GridworldEnv-v2",
             entry_point="custom_envs.grid_world.grid_world_env:GridWorldEnv",
             kwargs={'scene_of_input_noise': True},
             max_episode_steps=10)

    register(id="MetaLunarLanderEnv-v0",
             entry_point="custom_envs.meta_lunar_lander.meta_lunar_lander_env:MetaLunarLanderEnv",
             max_episode_steps=1000)
    register(id="MetaLunarLanderEnv-v1",
             entry_point="custom_envs.meta_lunar_lander.hierarchical_meta_lunar_lander_env:HierarchicalMetaLunarLanderEnv",
             max_episode_steps=1000)

    for n_objects in range(3):
        register(id=f'Hook-o{n_objects}-v1',
                 entry_point='custom_envs.hook.hook_env:HookEnv',
                 kwargs={'n_objects': n_objects},
                 max_episode_steps=max(50, 100 * n_objects))

        register(id=f'ButtonUnlock-o{n_objects}-v1',
                 entry_point='custom_envs.button_unlock.button_unlock_env:ButtonUnlockEnv',
                 kwargs={'n_buttons': n_objects + 1},
                 max_episode_steps=max(50, 50 * n_objects))

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
                register(id=f"MetaW-{env_name[:-len('-goal-observable')]}",
                         entry_point=make_variable_goal_env(env_class, env_type), max_episode_steps=500)
            elif env_type == "sparse":
                register(id=f"MetaW-{env_name[:-len('-goal-observable')]}-sparse",
                         entry_point=make_variable_goal_env(env_class, env_type), max_episode_steps=500)
            elif env_type == "dense":
                register(id=f"MetaW-{env_name[:-len('-goal-observable')]}-dense",
                         entry_point=make_variable_goal_env(env_class, env_type), max_episode_steps=500)
