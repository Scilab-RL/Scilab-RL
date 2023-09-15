"""
All custom environments must be registered here, otherwise they won't be found.
"""
from gymnasium.envs.registration import register
import highway_env

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

        register(id=f'AntButtonUnlock-o{n_objects}-v1',
                 entry_point='custom_envs.ant.ant_button_unlock_env:AntButtonUnlockEnv',
                 kwargs={'n_buttons': n_objects+1},
                 max_episode_steps=500*(n_objects+1))

    register(id='Ant4Rooms-v1',
             entry_point='custom_envs.ant.ant_4_rooms_env:Ant4RoomsEnv',
             kwargs={},
             max_episode_steps=600)

    register(id='AntReacher-v1',
             entry_point='custom_envs.ant.ant_reacher_env:AntReacherEnv',
             kwargs={},
             max_episode_steps=500)

    # Ant environments from HIRO paper
    for task in ['Maze', 'Push', 'Fall']:
        kwargs = {'task': task}
        register(id=f'Ant{task}-v0',
                 entry_point='custom_envs.ant.ant_maze_push_fall_env:AntMazePushFallEnv',
                 kwargs=kwargs,
                 max_episode_steps=600)

    register(
        id='parking-limited-v0',
        entry_point='highway_env.envs:ParkingEnv',
        max_episode_steps=100,
    )
