#!/usr/bin/env bash
# We first test all the algorithms and then test a selection of environments

test_algos() {
  # test all algorithms that have a config in conf/algorithm.
  # For now, we only consider algorithms with a continuous action space, so DQN will not work.
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
  if ! xvfb-run -a python3 main.py env=$ENVS algorithm=$ALGOS +performance=smoke_test render=none --multirun;
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
  ENVS+="CopReach-ik1-v0,"
  # ADD NEW ENVIRONMENTS HERE
  ENVS+="parking-limited-v0"

  echo "Smoke-testing environments $ENVS"

  # Don't have xvfb? install it with sudo apt-get install xvfb
  if ! xvfb-run -a python3 main.py algorithm=sac env=$ENVS +performance=smoke_test render=none --multirun;
  then
    exit 1
  fi
}
export CUDA_VISIBLE_DEVICES=""
test_algos
test_envs
echo "All smoke tests passed successfully."

test_render() {
  # test render on different types of environments
  local ENVS=""
  # MuJoCo
  ENVS+="FetchPickAndPlace-v1,"
  # CoppeliaSim
  ENVS+="close_box-state-v0,"
  # Box2D physics engine
  ENVS+="parking-limited-v0"
  if ! xvfb-run -a python3 main.py algorithm=sac env=$ENVS +performance=smoke_test render="record" render_freq=1 --multirun;
  then
    exit 1
  fi

}
export CUDA_VISIBLE_DEVICES=""
test_render
echo "All render tests passed successfully."

# find all pre-trained algorithms from the pervious smoke tests
TRIALS=( $(find /builds/*/Scilab-RL/data/ -name 'rl_model_best.zip*') )
N_TRIALS=${#TRIALS[@]}

test_loading() {
  # restore an agent from the previous randomly selected smoke test
	RND_INDEX=$(($RANDOM % $N_TRIALS))
  for ENV in "FetchPickAndPlace-v1" "AntMaze-v0" "Hook-o1-v1" "close_box-state-v0" "parking-limited-v0"; do
    for TRIAL_LINE in "${TRIALS[@]}"; do
      if [[ -n $(echo "$TRIAL_LINE" | grep -e "$ENV" ) ]]; then
        echo "Loading $TRIAL_LINE for $ENV"
        if ! xvfb-run -a python3 main.py env=${ENV} +restore_policy=${TRIAL_LINE} render=none wandb=0 n_epochs=1
        then
          exit 1
        fi
      fi
    done
  done
}
export CUDA_VISIBLE_DEVICES=""
for i in {0..10}; do
	test_loading
done
echo "All loading tests passed successfully."
