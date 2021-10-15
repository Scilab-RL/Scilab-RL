#!/usr/bin/env bash

main() {
  local ALGS="hac,sac"

  local ENVS=""
  #MuJoCo
  ENVS+="FetchReach-v1,"
  ENVS+="FetchPush-v1,"
  ENVS+="FetchSlide-v1,"
  ENVS+="FetchPickAndPlace-v1,"
  ENVS+="HandManipulateBlock-v0,"
  ENVS+="Hook-o1-v1,"
  ENVS+="ButtonUnlock-o2-v1,"
  ENVS+="AntReacher-v1,"
  ENVS+="Ant4Rooms-v1,"
  ENVS+="AntMaze-v0,"
  ENVS+="AntPush-v0,"
  ENVS+="AntFall-v0,"
  ENVS+="Blocks-o0-gripper_random-v1,"
  ENVS+="Blocks-o1-gripper_above-v1,"
  ENVS+="Blocks-o3-gripper_none-v1,"
  #RLBench
  ENVS+="reach_target-state-v0,"
  ENVS+="turn_tap-state-v0,"
  ENVS+="close_laptop_lid-state-v0,"
  ENVS+="close_drawer-state-v0,"
  ENVS+="close_box-state-v0"

  python3 experiment/train.py algorithm=$ALGS env=$ENVS ++n_epochs=1 +defaults=smoke_test --multirun;
}

main;
