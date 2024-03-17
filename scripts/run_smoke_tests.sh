#!/usr/bin/env bash
# We first test all the algorithms and then test a selection of environments

export OMP_NUM_THREADS=1
# change into git root directory
cd $(git rev-parse --show-toplevel)

DISCRETE_ACTION_ALGOS=(dqn cleandqn onestepac)

test_algos() {
  # test all algorithms that have a config in conf/algorithm.
  # For now, we only consider algorithms with a continuous action space, so DQN will not work.
  ALGOS=""
  for config in "conf/algorithm"/*
  do
    config="${config%.*}"
    algo_name=$(basename "$config")
    if [[ " ${DISCRETE_ACTION_ALGOS[@]} " =~ " ${algo_name} " ]]; then
      # Skip discrete action algorithms
      continue
    else
      ALGOS+="${algo_name},"
    fi
  done
  ALGOS="${ALGOS%,*}"
  echo "Smoke-testing algorithms $ALGOS"

  # environments with which to test the algorithms
  local ENVS="FetchReach-v2,parking-limited-v0"

  # Don't have xvfb? install it with sudo apt-get install xvfb
  if ! xvfb-run -a python3 src/main.py env=$ENVS algorithm=$ALGOS +performance=smoke_test render=none --multirun;
  then
    exit 1
  fi
}

test_envs() {
  local ENVS=""
  #MuJoCo
  ENVS+="FetchReach-v2,"
  ENVS+="FetchPickAndPlace-v2,"
  ENVS+="HandManipulateBlock-v1,"
  ENVS+="Hook-o1-v1,"
  ENVS+="ButtonUnlock-o2-v1,"
  ENVS+="Blocks-o0-gripper_random-v1,"
  ENVS+="Blocks-o3-gripper_none-v1,"
  ENVS+="Reach1DOF-v0,"
  # ADD NEW ENVIRONMENTS HERE.
  # Don't forget to add a comma at the end of each environment name except for the last environment name.
  ENVS+="parking-limited-v0,"
  ENVS+="PointGym-sparse-0.5-small_open_dgr-c1-rt0-s500-v0,"
  ENVS+="AntGym-sparse-10-0.5-small_open_dgr-c1-rt0-s700-v0,"
  ENVS+="MetaW-peg-insert-side-v2-sparse"

  echo "Smoke-testing environments $ENVS"

  # Don't have xvfb? install it with sudo apt-get install xvfb
  if ! xvfb-run -a python3 src/main.py algorithm=sac env=$ENVS +performance=smoke_test render=none --multirun;
  then
    exit 1
  fi
}
test_algos
test_envs
echo "All smoke tests passed successfully."

test_render() {
  # test render on different types of environments
  local ENVS=""
  # MuJoCo
  ENVS+="FetchPickAndPlace-v2,"
  # Box2D physics engine
  ENVS+="parking-limited-v0"
  if ! xvfb-run -a python3 src/main.py algorithm=sac env=$ENVS +performance=smoke_test render="record" render_freq=1 --multirun;
  then
    exit 1
  fi

}
test_render
echo "All render tests passed successfully."

test_loading() {
  # Loop over algorithms and specify base_logdir corresponding to algorithms.
  # Otherwise, algorithm names would not be in the paths. That way it is easier
  # to restore trained policies.
  # Storing the policies with adapted base_logdir could also be done in test_algos().
  local ALGOS=()
  for config in "conf/algorithm"/*
  do
    config="${config%.*}"
    algo_name=$(basename "$config")
    if [[ " ${DISCRETE_ACTION_ALGOS[@]} " =~ " ${algo_name} " ]]; then
      # Skip discrete action algorithms
      continue
    else
      ALGOS+=("${algo_name}")
    fi
  done
  local ENVS="FetchReach-v2,parking-limited-v0"
  # delete directory for loading_tests so that we only load the policies that we save now
  rm -rf data/loading_tests
  for ALG in ${ALGOS[@]}; do
    # Don't have xvfb? install it with sudo apt-get install xvfb
    if ! xvfb-run -a python3 src/main.py env=$ENVS algorithm=$ALG +performance=smoke_test render=none base_logdir="data/loading_tests/$ALG" --multirun;
    then
      exit 1
    fi
  done
  # Overwrite ENVS here, as we'll need an array
  local ENVS=("FetchReach-v2" "parking-limited-v0")
  for ALG in ${ALGOS[@]}; do
    # Find pre-trained algorithms from above
    TRIALS=( $(find "$(pwd)/data/loading_tests/$ALG/" -name "rl_model_finished*") )
    for TRIAL_LINE in ${TRIALS[@]}; do
      for ENV in ${ENVS[@]}; do
        if [[ -n $(echo "$TRIAL_LINE" | grep -e "$ENV" ) ]]; then
          echo "Loading $TRIAL_LINE for $ENV and $ALG"
          if ! xvfb-run -a python3 src/main.py env=${ENV} algorithm=${ALG} +restore_policy=${TRIAL_LINE} render=none wandb=0 n_epochs=1
          then
            exit 1
          fi
	fi
      done
    done
  done
}
test_loading
echo "All loading tests passed successfully."
