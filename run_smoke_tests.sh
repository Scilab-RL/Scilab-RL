#!/usr/bin/env bash
# We first test all the algorithms and then test a selection of environments

test_algos() {
  # test all algorithms that have a config in conf/algorithm
  ALGOS=""
  for config in "conf/algorithm"/*
  do
    config="${config%.*}"
    ALGOS+="${config##*/},"
  done
  ALGOS="${ALGOS%,*}"
  echo "Smoke-testing algorithms $ALGOS"

  # environments with which to test the algorithms
  local ENVS="FetchReach-v1,AntReacher-v1,reach_target-state-v0,parking-limited-v0"

  # Don't have xvfb? install it with sudo apt-get install xvfb
  if ! xvfb-run -a python3 main.py env=$ENVS algorithm=$ALGOS +performance=smoke_test --multirun;
  then
    exit 1
  fi
}

test_envs() {
  local ENVS=""
  #MuJoCo
  # ENVS+="FetchReach-v1,"
  ENVS+="FetchPickAndPlace-v1,"
  ENVS+="HandManipulateBlock-v0,"
  ENVS+="Hook-o1-v1,"
  ENVS+="ButtonUnlock-o2-v1,"
  # ENVS+="AntReacher-v1,"
  ENVS+="AntMaze-v0,"
  ENVs+="AntButtonUnlock-o2-v1,"
  ENVS+="Blocks-o0-gripper_random-v1,"
  ENVS+="Blocks-o3-gripper_none-v1,"
  ENVS+="Reach1DOF-v0,"
  #RLBench
  # ENVS+="reach_target-state-v0,"
  ENVS+="close_box-state-v0,"
  ENVS+="CopReach-ik1-v0"
  # ADD NEW ENVIRONMENTS HERE
  ENVS+="parking-limited-v0"

  echo "Smoke-testing environments $ENVS"

  # Don't have xvfb? install it with sudo apt-get install xvfb
  if ! xvfb-run -a python3 main.py algorithm=sac env=$ENVS +performance=smoke_test --multirun;
  then
    exit 1
  fi
}
export CUDA_VISIBLE_DEVICES=""
test_algos
test_envs
echo "All smoke tests passed successfully."
