# ideas_deep_rl2

This is the IDEAS / LeCAREbot deep RL repository focusing on hierarchical goal-conditioned reinforcement learning using the stable baselines 3 methods and OpenAI gym interface

## Getting started

To generate a file with all possible environment/algorithm combinations run 

`./run_testing.sh g <gpu_ids> p <max processes> m <free GPU memory required to star a new process> s <time to sleep before executing next command> t <type of testing <function|performance>>`

This will create a file `test_cmds.txt` in the base directory with all currently supported environment-algorithm combinations and useful hyperparameters. It will also execute them.

## Limitations
Currently, only off-policy algorithms are supported: DQN, DDPG, TD3 and SAC. PPO