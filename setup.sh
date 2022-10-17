#!/bin/bash

_conda_cuda_pytorch() {
	# reinstall gpu specific tools
	if [ -x "$(command -v nvidia-smi)" ]; then
		conda install cudatoolkit=11.3 pytorch -c pytorch -y
	fi
}

setup_conda() {
  echo "Setup conda"
	source $(conda info --base)/etc/profile.d/conda.sh
  # check if scilabrl already exists
  if [ -n "$(conda env list | grep 'scilabrl*')" ]; then
    conda activate scilabrl
    # just reinstall everything
    pip install -r requirements.txt
    _conda_cuda_pytorch
  else
    if [ $(uname -s) == "Linux" ]; then
      conda env create -f conda/linux_environment.yaml
			_conda_cuda_pytorch
    elif [ $(uname -s) == "Darwin" ]; then
      if [ $(uname -m) =~ "arm" ]; then
        conda env create -f conda/macos_arm_environment.yaml
      elif [ $(uname -m) =~ "x86" ]; then
        printf "Intel Macs are currently not supported"
        exit 1
        # conda env create -f macos_x86_environment.yaml
      fi
    fi
  fi
}

get_mujoco() {
  # Check if MuJoCo is already installed
  if ! [ -d "${HOME}/.mujoco/mujoco210" ]; then
    mkdir -p $HOME/.mujoco/
    # Get MuJoCo
    echo "Getting MuJoCo"
    MUJOCO_VERSION="2.1.1"
    if [ $(uname -s) == "Linux" ]; then
      MUJOCO_DISTRO="linux-x86_64.tar.gz"
      wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O "${HOME}/mujoco.tar.gz"
      tar -xf "${HOME}/mujoco.tar.gz" -C "${HOME}/.mujoco/"
      # NOTE: If we move to a newer version for linux
      # wget "https://github.com/deepmind/mujoco/releases/download/$MUJOCO_VERSION/mujoco-$MUJOCO_VERSION-linux-x86_64.tar.gz" -O "${HOME}/mujoco.tar.gz"
      # tar -xf "${HOME}/mujoco.tar.gz"
      # mv "${PWD}/mujoco-${MUJOCO_VERSION}" "${HOME}/.mujoco/mujoco210"
      rm "${HOME}/mujoco.tar.gz"
    elif [ $(uname -s) == "Darwin" ]; then
      MUJOCO_DISTRO="macos-universal2.dmg"
      wget "https://github.com/deepmind/mujoco/releases/download/$MUJOCO_VERSION/mujoco-$MUJOCO_VERSION-macos-universal2.dmg"
      VOLUME=`hdiutil attach mujoco-${MUJOCO_VERSION}-macos-universal2.dmg | grep Volumes | awk '{print $3}'`
      cp -rf $VOLUME/*.app /Applications
      hdiutil detach $VOLUME
      mkdir -p $HOME/.mujoco/mujoco210
      ln -sf /Applications/MuJoCo.app/Contents/Frameworks/MuJoCo.framework/Versions/Current/Headers/ $HOME/.mujoco/mujoco210/include
      mkdir -p $HOME/.mujoco/mujoco210/bin
      ln -sf /Applications/MuJoCo.app/Contents/Frameworks/MuJoCo.framework/Versions/Current/libmujoco.2.1.1.dylib $HOME/.mujoco/mujoco210/bin/libmujoco210.dylib
      ln -sf /Applications/MuJoCo.app/Contents/Frameworks/MuJoCo.framework/Versions/Current/libmujoco.2.1.1.dylib /usr/local/lib/
      # For M1 (arm64) mac users:
      if [ $(uname -m) =~ "arm" ]; then
        conda install -y glfw
        rm -rfiv $CONDA_PREFIX/lib/python3.*/site-packages/glfw/libglfw.3.dylib
        ln -sf $CONDA_PREFIX/lib/libglfw.3.dylib $HOME/.mujoco/mujoco210/bin
        if [ ! -x "$(command -v gcc-12)" ]; then
          brew install gcc
        fi
        export CC=/opt/homebrew/bin/gcc-12
      fi
      rm -rf mujoco-2.1.1-macos-universal2.dmg
    fi
  fi
  # Install mujoco-py
  echo "Installing mujoco-py and testing import"
  source set_paths.sh
  pip3 install mujoco-py && python3 -c 'import mujoco_py'
}

get_rlbench() {
  if [ $(uname -s) == "Darwin" ]; then
    echo "There is no PyRep support for macos"
    return
  fi
  # Check if CoppeliaSim is already installed
  if [ -d "${HOME}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04" ]; then
    echo "Skipping CoppeliaSim as it is already installed."
    return
  fi
  # Get CoppeliaSim
  echo "Getting CoppeliaSim"
  wget http://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -O "${HOME}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz"
  tar -xf "${HOME}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz" -C "$HOME"
  rm "${HOME}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz"
  # Get RLBench
  echo "Getting RLBench"
  source venv/bin/activate
  source set_paths.sh
  pip install git+https://github.com/stepjam/PyRep.git git+https://github.com/stepjam/RLBench.git pyquaternion natsort
}


install_conda() {
  if [ $(uname -s) == "Linux" ]; then
    PYOS="Linux"
  elif [ $(uname -s) == "Darwin" ]; then
    PYOS=MacOSX
  fi
  ARCHITECTURE=$(uname -m)
	CONDA_RELEASE="Miniforge3-$PYOS-$ARCHITECTURE"
  CONDA_HOME="$HOME/miniforge3"
  export PATH="$CONDA_HOME/bin:$PATH"
  curl -fsSLO https://github.com/conda-forge/miniforge/releases/latest/download/$CONDA_RELEASE.sh
  bash "$CONDA_RELEASE.sh" -b -p $CONDA_HOME
  rm "$CONDA_RELEASE.sh"
  conda init "$(basename $SHELL)"
  . $(conda info --base)/etc/profile.d/conda.sh
}


main() {
	if ! [ -x "$(command -v conda)" ]; then
		echo "Installing conda"
		install_conda
	fi
	setup_conda
	while getopts 'mr' OPTION; do
		case "$OPTION" in
			m)
				get_mujoco
				;;
			r)
				get_rlbench
				;;
			?)
				echo "Use -m to install MuJoCo and -r to install RLBench"
				exit 1
				;;
		esac
	done
}

main "$@"
