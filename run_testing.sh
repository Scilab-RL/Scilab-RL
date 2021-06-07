#!/usr/bin/env bash

main() {
  local ALGS="hac,her"

  local ENVS=""
  ENVS+="Blocks-o1-gripper_random-v1,"
  ENVS+="AntReacher-v1,"
  ENVS+="ButtonUnlock-o1-v1,"
  ENVS+="FetchReach-v1,"
  ENVS+="AntMaze-v0,"
  ENVS+="FetchPush-v1,"
  ENVS+="FetchSlide-v1,"
  ENVS+="FetchPickAndPlace-v1,"
  ENVS+="FetchReach-v1,"
  ENVS+="HandManipulateBlock-v0,"
  ENVS+="Hook-o1-v1,"
  ENVS+="ButtonUnlock-o2-v1,"
  ENVS+="ButtonUnlock-o1-v1,"
  ENVS+="AntReacher-v1,"
  ENVS+="Ant4Rooms-v1,"
  ENVS+="AntMaze-v0,"
  ENVS+="AntPush-v0,"
  ENVS+="AntFall-v0,"
  ENVS+="BlockStackMujocoEnv-gripper_random-o0-v1,"
  ENVS+="BlockStackMujocoEnv-gripper_random-o2-v1,"
  ENVS+="BlockStackMujocoEnv-gripper_above-o1-v1,"
  ENVS+="BlockStackMujocoEnv-gripper_none-o1-v1"

  python3 experiment/train.py algorithm=$ALGS env=$ENVS ++n_epochs=2 +defaults=smoke_test --multirun;
}

main;
