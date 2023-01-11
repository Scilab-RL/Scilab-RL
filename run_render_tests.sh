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
