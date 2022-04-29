RUN_ALL_TESTS="false" # set to "true" to run all tests

if [ $RUN_ALL_TESTS != "true" ]
then  # run only these performance tests:
  for config in "FetchReach/sac_her-test" "RLB_reach_target/sac_her-test"
  do
    if ! xvfb-run -a python3 experiment/train.py +performance=$config wandb=0 --multirun;
    then
      exit 1
    fi
  done
  exit 0
fi

# run all performance-tests:
for env_folder in "conf/performance"/*
do
  for config in "$env_folder"/*
  do
    if [ ${config: -9} = "test.yaml" ]
    then
      config=${config:17}
      config=${config%.*}
      if ! xvfb-run -a python3 experiment/train.py +performance=$config wandb=0 --multirun;
      then
        echo "Performance-test $config failed."
        exit 1
      fi
    fi
  done
done
echo "All performance tests passed."
exit 0
