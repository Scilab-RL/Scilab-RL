#!/usr/bin/env bash

main() {
  # test selection of environments with sac
  local ALGS="sac"

  local ENVS=""
  #MuJoCo
  # ENVS+="FetchReach-v1,"
  ENVS+="FetchPickAndPlace-v1,"
  ENVS+="HandManipulateBlock-v0,"
  ENVS+="Hook-o1-v1,"
  ENVS+="ButtonUnlock-o2-v1,"
  ENVS+="AntReacher-v1,"
  # ENVS+="Ant4Rooms-v1,"
  ENVS+="AntButtonUnlock-o2-v1,"
  ENVS+="AntMaze-v0,"
  ENVS+="Blocks-o0-gripper_random-v1,"
  ENVS+="Blocks-o3-gripper_none-v1,"
  #RLBench
  # ENVS+="reach_target-state-v0,"
  ENVS+="close_drawer-state-v0"

  # Don't have xvfb? install it with sudo apt-get install xvfb
  if ! xvfb-run -a python3 experiment/train.py algorithm=$ALGS env=$ENVS ++n_epochs=1 ++wandb=0 +defaults=smoke_test --multirun;
  then
    exit 1
  fi

  # test other Algorithms
  local ALG_TEST_ENVS="FetchReach-v1,AntReacher-v1,reach_target-state-v0"
  # test HER
  if ! xvfb-run -a python3 experiment/train.py algorithm=sac env=$ALG_TEST_ENVS +replay_buffer=her ++n_epochs=1 ++wandb=0 +defaults=smoke_test --multirun;
  then
    exit 1
  fi

  # test ddpg and td3
  if ! xvfb-run -a python3 experiment/train.py algorithm=ddpg,td3 env=$ALG_TEST_ENVS ++n_epochs=1 ++wandb=0 +defaults=smoke_test --multirun;
  then
    exit 1
  fi

  echo "All smoke tests passed successfully."
}

main
