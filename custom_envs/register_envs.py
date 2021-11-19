from gym.envs.registration import register


for n_objects in range(5):
    for gripper_goal in ['gripper_none', 'gripper_random', 'gripper_above']:
        if gripper_goal != 'gripper_random' and n_objects == 0:  # Disallow because there would be no goal
            continue
        register(id='Blocks-o{}-{}-v1'.format(n_objects, gripper_goal),
                 entry_point='custom_envs.blocks.blocks_env:BlocksEnv',
                 kwargs={'n_objects': n_objects, 'gripper_goal': gripper_goal},
                 max_episode_steps=max(50, 50*n_objects))

for n_objects in range(3):
    register(id='Hook-o{}-v1'.format(n_objects),
             entry_point='custom_envs.hook.hook_env:HookEnv',
             kwargs={'n_objects': n_objects},
             max_episode_steps=max(50, 100 * n_objects))

    register(id='ButtonUnlock-o{}-v1'.format(n_objects),
             entry_point='custom_envs.button_unlock.button_unlock_env:ButtonUnlockEnv',
             kwargs={'n_buttons': n_objects+1},
             max_episode_steps=max(50, 50*n_objects))

    register(id='AntButtonUnlock-o{}-v1'.format(n_objects),
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
    register(id='Ant{}-v0'.format(task),
             entry_point='custom_envs.ant.ant_maze_push_fall_env:AntMazePushFallEnv',
             kwargs=kwargs,
             max_episode_steps=600)

# ReacherEnv using CoppeliaSim
for IK in [0, 1]:  # whether to use inverse kinematics
    kwargs = {'ik': IK, 'render': 1}
    register(id='CopReach-ik{}-v0'.format(kwargs['ik']),
             entry_point='custom_envs.cop_reach.cop_reach_env:ReacherEnvMaker',
             kwargs=kwargs,
             max_episode_steps=200)