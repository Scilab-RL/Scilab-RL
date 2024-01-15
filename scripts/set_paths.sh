#!/bin/bash
if [ -x "$(command -v nvidia-smi)" ]; then
  nv_version_long=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
  nv_version=${nv_version_long:0:3}
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-$nv_version
  #export LD_PRELOAD=$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libGLEW.so # Had to comment this out with NVIDIA driver 460 version.
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
  export MUJOCO_GL=osmesa
fi
