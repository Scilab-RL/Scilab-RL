#!/usr/bin/env bash
usage() { echo "$0 usage:" && grep " .)\ #" $0; exit 0; }
[ $# -eq 0 ] && usage

gpu_ids=(0)
min_mem_free=2500
max_active_procs=16
sleep_time=15
opt_duration=15
while getopts ":hd:p:m:g:s:" arg; do
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
#
#      gpu_ids=',' read -r -a array <<< "${OPTARG}"
      ;;
    s) # Specify time to sleep in seconds after each command. This is important to wait for the previous command to create directories and allocate memory before the next process is started.
      echo "s is ${OPTARG}"
      sleep_time=${OPTARG}
      ;;
    d) # Specify the duration in minutes wall time for how long you want to optimize.
      echo "d is ${OPTARG}"
      opt_duration=${OPTARG}
      opt_duration_seconds=$(( $opt_duration * 60 ))
      ;;
    h | *) # Display help.
      usage
      exit 0
      ;;
  esac
done

echo "Starting hyperopt for $opt_duration minutes"
echo "Max. number of parallel processes is $max_active_procs"
echo "Minimal required free memory is $min_mem_free"
echo "GPUs to be used are:"
for element in "${gpu_ids[@]}"
do
    echo "$element"
done

logs_dir="hyperopt_logs"

rm -rf ${logs_dir}
mkdir ${logs_dir}

cmd="python3 experiment/hyperopt.py --multirun"
cmd_ctr=0
while [ "$SECONDS" -lt "$opt_duration_seconds" ]; do
  cmd_ctr=$(($cmd_ctr+1))
  seconds_remaining=$(($opt_duration_seconds - $SECONDS))
  minutes_remaining=$((seconds_remaining / 60))
  hours_remaining=$((minutes_remaining / 60))
  echo "Running next parameterization, $minutes_remaining minutes or $hours_remaining hours left."

  n_active_procs=$(pgrep -c -P$$)
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
  echo "Starting next hyperopt run on GPU ${this_gpu_id} with ${free_mem} MB free memory: "
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


