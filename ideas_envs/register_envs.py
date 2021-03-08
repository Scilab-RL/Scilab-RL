from gym.envs.registration import register


register(id='Blocks-v1',
         entry_point='ideas_envs.blocks.blocks_env:BlocksEnv',
         max_episode_steps=50)
