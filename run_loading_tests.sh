test_loading() {
  # restore an agent from the previous smoke test
  SMOKE_TEST_DIR=/builds/$USER/Scilab-RL/data/2327a47/FetchPickAndPlace-v1/11-15-15/0/rl_model_best.zip
  if ! xvfb-run -a python3 main.py env=FetchReach-v1 algorithm=sac +restore_policy=$SMOKE_TEST_DIR render=none wandb=0
  then
    exit 1
  fi
}
echo "Passed loading test"
