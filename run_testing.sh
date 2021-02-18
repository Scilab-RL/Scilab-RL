#!/usr/bin/env bash
test_mode=$1
logs_dir="test_logs"
cmd_file="test_cmds.txt"

#gpu_ids=(0 1)
gpu_ids=(0)
min_mem_free=1500

rm -rf ${logs_dir}
rm ${cmd_file}
mkdir ${logs_dir}
python3 experiment/generate_testing_commands.py ${test_mode}
sleep 2
# Use first argument $1 to determine number of active processes, otherwise use 5
max_active_procs="${1-6}"
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
    ps -ef | grep sleep
    echo "Currently, there are ${n_active_procs} active processes."
    while [ "$n_active_procs" -ge "$max_active_procs" ];do
        echo "${n_active_procs} of ${max_active_procs} processes are running. Waiting..."
        sleep 15
        n_active_procs=$(pgrep -c -P$$)
    done
	# Only run if there is a GPU with enough free memory
    free_mem=0
	  this_gpu_id=-1
    while [ "$free_mem" -le "$min_mem_free" ]; do
        for gpu_id in "${gpu_ids[@]}"; do
            free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $gpu_id | grep -Eo [0-9]+)
            echo "${free_mem} MB is free on GPU ${gpu_id}, but ${min_mem_free} MB is required. Waiting..."
            if [ "$free_mem" -ge "$min_mem_free" ]; then
              this_gpu_id=${gpu_id}
              break
            fi
            sleep 15
        done
    done
    echo "Now executing cmd ${cmd_ctr} / $(( n_cmds )) on GPU ${this_gpu_id} with ${free_mem} MB free memory: "
    echo ${cmd}
    export CUDA_VISIBLE_DEVICES=${this_gpu_id}
#    ${cmd}
    $cmd 1> ${logs_dir}/${cmd_ctr}.log 2> ${logs_dir}/${cmd_ctr}_err.log || true & # Execute in background
    sleep 15
done
echo "All commands have been started. Waiting for last processes to finish."
while pgrep -c -P$$ > "0"
do
  echo "$(pgrep -c -P$$) procs remaining".
  sleep 10
done

echo "All commands finished."

#echo "Number of running procs: $(pgrep -c -P$$)"
if [ $test_mode == "performance" ]; then
  python3 experiment/validate_performance_testing.py

python3 experiment/check_error_logs.py

