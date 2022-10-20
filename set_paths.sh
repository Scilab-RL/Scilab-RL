#For MuJoCo
export MUJOCO_PY_MUJOCO_PATH=${HOME}/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MUJOCO_PY_MUJOCO_PATH/bin

#For CoppeliaSim
export COPPELIASIM_ROOT=${HOME}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

#For both
nv_version_long=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
nv_version=${nv_version_long:0:3}
echo "Nvidia version: ${nv_version}."
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-$nv_version
#export LD_PRELOAD=$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libGLEW.so # Had to comment this out with NVIDIA driver 460 version.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
