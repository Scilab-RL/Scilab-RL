"""
All custom environments must be registered here, otherwise they won't be found.
"""
from gymnasium.envs.registration import register
import highway_env
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from src.utils.custom_wrappers import MakeDictObs


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
                register(id=env_name[:-len('-goal-observable')], entry_point=make_variable_goal_env(env_class, env_type), max_episode_steps=500)
            elif env_type == "sparse":
                register(id=f"{env_name[:-len('-goal-observable')]}-sparse", entry_point=make_variable_goal_env(env_class, env_type), max_episode_steps=500)
            elif env_type == "dense":
                register(id=f"{env_name[:-len('-goal-observable')]}-dense", entry_point=make_variable_goal_env(env_class, env_type), max_episode_steps=500)
