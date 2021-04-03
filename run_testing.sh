#!/usr/bin/env bash
usage() { echo "$0 usage:" && grep " .)\ #" $0; exit 0; }
[ $# -eq 0 ] && usage

gpu_ids=(0)
min_mem_free=2500
max_active_procs=6
test_mode='performance'
sleep_time=15
while getopts ":ht:p:m:g:s:" arg; do
  case $arg in
    p) # Specify max. number of processes.
      echo "p is ${OPTARG}"
      max_active_procs=${OPTARG}
      ;;
    m) # Specify min free memory in MB on a GPU to start the next process.
      echo "m is ${OPTARG}"
      min_mem_free=${OPTARG}
      ;;
    g) # Specify a comma-sparated list of GPU ids to include in the testing, e.g. '0,1', or simply '0' to use GPU 0.
      echo "g is ${OPTARG}"
      IFS=', ' read -r -a gpu_ids <<< "${OPTARG}"
#      for element in "${array[@]}"
#      do
#          echo "$element"
#      done
#      gpu_ids=',' read -r -a array <<< "${OPTARG}"
      ;;
    s) # Specify time to sleep in seconds after each command. This is important to wait for the previous command to create directories and allocate memory before the next process is started.
      echo "s is ${OPTARG}"
      sleep_time=${OPTARG}
      ;;
    t) # Specify type of testing, either 'function' or 'performance'.
      test_mode=${OPTARG}
      if [ $test_mode = "function" -o $test_mode = "performance" ]; then
        echo "t is $test_mode."
      else
        echo "Error. Testing type needs to be either 'function' or 'performance', '$test_mode' found instead."
        exit 0
      fi
      ;;
    h | *) # Display help.
      usage
      exit 0
      ;;
  esac
done

echo "Starting $test_mode test."
echo "Max. number of parallel processes is $max_active_procs"
echo "Minimal required free memory is $min_mem_free"
echo "GPUs to be used are $gpu_ids"

logs_dir="test_logs"
cmd_file="test_cmds.txt"

rm -rf ${logs_dir}
rm ${cmd_file}

mkdir ${logs_dir}

echo "Generating test commands for ${test_mode} mode"

python3 experiment/generate_testing_commands.py $test_mode
sleep 2


cmd_ctr=0
n_cmds=$(cat $cmd_file | wc -l)
declare -a cmd_arr=()
while IFS= read -r cmd
do
    cmd_arr+=("${cmd[@]}")
done < $cmd_file


for ((i = 0; i < ${#cmd_arr[@]}; i++))
do
    cmd="${cmd_arr[$i]}"
    ((cmd_ctr++))
    echo "Next cmd in queue is:"
    echo $cmd
	#    cmd="sleep 12" # Uncomment for debugging this script with a simple sleep command
    n_active_procs=$(pgrep -c -P$$)
#    ps -ef | grep sleep
    echo "Currently, there are ${n_active_procs} active processes."

    while [ "$n_active_procs" -ge "$max_active_procs" ]; do

        echo "${n_active_procs} of ${max_active_procs} processes are running. Waiting..."
        sleep $sleep_time
        n_active_procs=$(pgrep -c -P$$)
    done
	# Only run if there is a GPU with enough free memory
    free_mem=0
	  this_gpu_id=-1
    while [ "$free_mem" -le "$min_mem_free" ]; do
        for gpu_id in "${gpu_ids[@]}"; do
            free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $gpu_id | grep -Eo [0-9]+)
            if [ "$free_mem" -ge "$min_mem_free" ]; then
              this_gpu_id=${gpu_id}
              break
            fi
            echo "${free_mem} MB is free on GPU ${gpu_id}, but ${min_mem_free} MB is required. Waiting..."
            sleep $sleep_time
        done
    done
    echo "Now executing cmd ${cmd_ctr} / $(( n_cmds )) on GPU ${this_gpu_id} with ${free_mem} MB free memory: "
    echo ${cmd}
    export CUDA_VISIBLE_DEVICES=${this_gpu_id}
#    ${cmd}
    $cmd 1> ${logs_dir}/${cmd_ctr}.log 2> ${logs_dir}/${cmd_ctr}_err.log || true & # Execute in background
    sleep $sleep_time
done
echo "All commands have been started. Waiting for last processes to finish."
while pgrep -c -P$$ > "0"
do
  echo "$(pgrep -c -P$$) procs remaining".
  sleep $sleep_time
done

echo "All commands finished."

#echo "Number of running procs: $(pgrep -c -P$$)"
if [ "$test_mode" = "performance" ]; then
  python3 experiment/validate_performance_testing.py
fi
python3 experiment/check_error_logs.py

