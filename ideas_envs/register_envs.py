from gym.envs.registration import register


for n_objects in range(5):
    for gripper_goal in ['gripper_none', 'gripper_random', 'gripper_above']:
        if gripper_goal != 'gripper_random' and n_objects == 0:  # Disallow because there would be no goal
            continue
        # register Blocks environment
        register(id='Blocks-o{}-{}-v1'.format(n_objects, gripper_goal),
                 entry_point='ideas_envs.blocks.blocks_env:BlocksEnv',
                 kwargs={'n_objects': n_objects, 'gripper_goal': gripper_goal},
                 max_episode_steps=max(50, 50*n_objects))
        # register Hook environment
        register(id='Hook-o{}-{}-v1'.format(n_objects, gripper_goal),
                 entry_point='ideas_envs.hook.hook_env:HookEnv',
                 kwargs={'n_objects': n_objects, 'gripper_goal': gripper_goal},
                 max_episode_steps=max(50, 100 * n_objects))
