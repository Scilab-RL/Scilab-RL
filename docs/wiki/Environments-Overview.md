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
