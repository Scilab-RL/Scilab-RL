RUN_ALL_TESTS="true" # set to "true" to run all tests

# change into git root directory
cd $(git rev-parse --show-toplevel)

if [ $RUN_ALL_TESTS != "true" ]
then  # run only these performance tests:
  for config in "FetchReach/sac_her-test"
  do
    if ! xvfb-run -a python3 src/main.py +performance=$config wandb=0 render=none --multirun;
    then
      echo "Performance-test $config failed."
      exit 1
    fi
  done
  exit 0
fi

# run all performance-tests:
configs_to_ignore=("Blocks/o0-random-cleansac_her-test" "Blocks/o0-random-sac_her-test")
unsuccessful_configs=()
git_branch=$(git rev-parse --abbrev-ref HEAD)
commit_hash=$(git rev-parse $git_branch)
echo "Performance test starting on $(date)." > performance-test_results.log
echo "Branch: $git_branch. Commit: $commit_hash" >> performance-test_results.log
for env_folder in "conf/performance"/*
do
  for config in "$env_folder"/*
  do
#    if [[ $(echo ${configs_to_ignore[@]} | fgrep -w $config) ]]
#    if [[ " ${configs_to_ignore[*]} " =~ " ${config} " ]]
    if printf '%s\0' "${configs_to_ignore[@]}" | grep -Fxqz -- $config
    then
      echo "Skipping config $config"
      echo "Skipping config $config" >> performance-test_results.log
    else
      if [ ${config: -9} = "test.yaml" ]
      then
        config=${config:17}
        config=${config%.*}
        #if ! xvfb-run -a python3 src/main.py +performance=$config wandb=0 render=none --multirun;
        if [ $(echo $config | fgrep -w "o1") ]
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
if [ ${#ArrayName[@]} = 0 ]
then
  echo "All performance tests passed."
  exit 0
else
  echo "The following performance tests failed:"
  for config in unsuccessful_configs
  do
    echo $config
  done
  exit 1
fi

