#!/bin/bash

export OMP_NUM_THREADS=1
# Run all performance-tests, except for those: (useful when you don't want to repeat already successful tests)
configs_to_ignore=()
# The following are known to work:
configs_to_ignore+=("Blocks/o0-random-cleansac_her-test")
configs_to_ignore+=("Blocks/o0-random-sac_her-test")
configs_to_ignore+=("Blocks/o1-above-cleansac_her-test")
#configs_to_ignore+=("Blocks/o1-above-sac_her-test")
configs_to_ignore+=("Blocks/o1-none-cleansac_her-test")
configs_to_ignore+=("Blocks/o1-none-sac_her-test")
#configs_to_ignore+=("Blocks/o1-random-cleansac_her-test")
#configs_to_ignore+=("Blocks/o1-random-sac_her-test")
configs_to_ignore+=("FetchPickAndPlace/sac_her-test")
configs_to_ignore+=("FetchPush/sac_her-test")
configs_to_ignore+=("FetchPush/cleansac_her-test")
configs_to_ignore+=("FetchReach/sac_her-test")
configs_to_ignore+=("FetchReach/cleansac_her-test")
configs_to_ignore+=("FetchReach/dense-ppo-test")
configs_to_ignore+=("FetchSlide/sac_her-test")
configs_to_ignore+=("HighwayParking/sac_her-test")
configs_to_ignore+=("HighwayParking/cleansac_her-test")
configs_to_ignore+=("Reach1DOF/actor_critic-test")
# The following are known not to work:
configs_to_ignore+=("AntMaze/AntMaze_Open_Diverse_GR-v4-sac_her-test")
configs_to_ignore+=("AntMaze/AntMaze_Open_Diverse_GR_Dense-v4-sac_her-test")


unsuccessful_configs=()
git_branch=$(git rev-parse --abbrev-ref HEAD)
commit_hash=$(git rev-parse $git_branch)
echo "Performance test starting on $(date)." > performance-test_results.log
echo "Branch: $git_branch. Commit: $commit_hash" >> performance-test_results.log
for env_folder in "conf/performance"/*
do
  for config in "$env_folder"/*
  do
    if [ ${config: -9} = "test.yaml" ]
    then
      config=${config:17}
      config=${config%.*}
      if [[ " ${configs_to_ignore[*]} " =~ " ${config} " ]]
      then
        echo "Skipping config $config"
        echo "Skipping config $config" >> performance-test_results.log
      else
        if ! xvfb-run -a python3 src/main.py +performance=$config wandb=1 render=none --multirun;
        then
          echo "Performance-test $config FAILED."
          echo "Performance-test $config FAILED." >> performance-test_results.log
          unsuccessful_configs+=($config)
        else
          echo "Performance-test $config successful."
          echo "Performance-test $config successful." >> performance-test_results.log
        fi
      fi
    fi
  done
done
if [ ${#unsuccessful_configs[@]} = 0 ]
then
  echo "All performance tests passed."
  exit 0
else
  echo "The following performance tests failed:"
  for config in "${unsuccessful_configs[@]}"
  do
    echo $config
  done
  exit 1
fi
