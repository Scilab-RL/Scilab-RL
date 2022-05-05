RUN_ALL_TESTS="false" # set to "true" to run all tests

if [ $RUN_ALL_TESTS != "true" ]
then  # run only these performance tests:
  for config in "FetchReach/sac_her-test" "FetchPush/sac_her-test" "FetchSlide/sac_her-test" "FetchPickAndPlace/sac_her-test" "Blocks/o0-random-sac_her-test" "Blocks/o1-random-sac_her-test" "Blocks/o1-above-sac_her-test" "Blocks/o1-none-sac_her-test" "Ant4Rooms/sac_her-test" "AntReacher/sac_her-test"
  do
    # echo "python3 experiment/train.py +performance=$config wandb=1 project_name=perf_test_$config --multirun;"
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
