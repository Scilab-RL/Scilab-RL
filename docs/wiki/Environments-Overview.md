---
layout: default
title: Custom Environments
parent: Wiki
has_children: false
nav_order: 3
---

This page gives a short overview over the `custom_envs`.

# MuJoCo Environments

## Robotic Arm Environments
### Blocks
![Screenshot_from_2021-10-15_16-03-24](uploads/05d2ff86845aabc6bcd859faf14a7566/Screenshot_from_2021-10-15_16-03-24.png)

> Blocks-o2-gripper_random-v1

The _blocks_ environment directly inherits from the `gym.envs.robotics.fetch_env`. The basic task is to stack up to 4 blocks on top of each other. The number of blocks is configured with the number after the `o`. There are three configurations for the gripper-goal:
- `gripper_none`: The position of the gripper is not relevant for the goal
- `gripper_random`: The gripper should reach a random position after stacking the blocks
- `gripper_above`: The gripper should be above the stacked blocks

### Hook
![Screenshot_from_2021-10-15_16-20-15](uploads/c36a9828469e1f8ce1e8e7fd61d00c3e/Screenshot_from_2021-10-15_16-20-15.png)

> Hook-o1-v1

Basically the same as the _blocks_ environment, but the blocks can be out of reach and have to be pulled closer with a hook. Tool use! The gripper-goal is always `gripper_none`.

### ButtonUnlock
![Screenshot_from_2021-10-15_16-24-06](uploads/ba7ac134d3abca640f0b61a3b1143c69/Screenshot_from_2021-10-15_16-24-06.png)

> ButtonUnlock-o2-v1

This time the goal is to push the red button. The robotic arm can only move in the x-y-plane. Before it can push the red button, it has to unlock access to it by pushing up to 2 blue buttons. In the picture, one button was already pressed. Causal Dependencies!

## Ant
We have a variety of _Ant_ environments that all inherit from `src/custom_envs.ant.ant_env`.
### AntReacher
![Screenshot_from_2021-10-15_16-29-50](uploads/af4446ffa8d61c7b255a0ceb4d1f39b2/Screenshot_from_2021-10-15_16-29-50.png)

> AntReacher-v1

The ant should reach the position marked with the red shpere.

### Ant Four Rooms
![Screenshot_from_2021-10-15_16-37-29](uploads/6c14069d3dca6e572633546c1f365384/Screenshot_from_2021-10-15_16-37-29.png)

> Ant4Rooms-v1

The same as _AntReacher_, but with four rooms.

### AntButtonUnlock
![Screenshot_from_2021-10-15_16-43-39](uploads/cc6441e7854268d717cec36a2d4007ff/Screenshot_from_2021-10-15_16-43-39.png)

> AntButtonUnlock-o2-v1

The same as _ButtonUnlock_ with the robotic arm, but with the Ant robot.

### AntMaze, AntPush and AntFall
![EnvAntMazeThresh](uploads/414e29d7f86f8ce5e2deacb53c110e5d/EnvAntMazeThresh.png)

> AntMaze-v0 

![EnvAntPush](uploads/f2ed1ccabe09e67cb2dd3e8058b027d2/EnvAntPush.png)

> AntPush-v0

![EnvAntFall](uploads/d8d8dbae15715400b02db75454dd1ad2/EnvAntFall.png)

> AntFall-v0

These environments were created to showcase the abilities of the [HIRO algorithm](https://arxiv.org/abs/1805.08296). We adapted them to fit in our framework. The goal of the Ant is to reach the yellow sphere (the purple sphere is a subgoal created by the HAC algorithm).

# CoppeliaSim
## RLBench
![Screenshot_from_2021-10-15_16-55-36](uploads/3ebe6df1ceea1bba39da6ee6dde51631/Screenshot_from_2021-10-15_16-55-36.png)

> turn_tap-state-v0

[RLBench](https://github.com/stepjam/RLBench) is a collection of ~100 environments, which we can use in our framework by wrapping them with our `src/custom_envs.wrappers.rl_bench_wrapper`. You can find a list of them in issue #66. Thoroughly tested RLBench environments are _reach_target-state-v0_, _turn_tap-state-v0_, _close_laptop_lid-state-v0_, _close_drawer-state-v0_ and _close_box-state-v0_.
