#!/bin/bash

setup_venv() {
  # Check if there is already a venv
  if [ -d "venv" ]; then
    echo "Skipping venv setup as there is already a venv."
    return
  fi
  # Set up the venv
  echo "Setting up venv"
  virtualenv -p python3 venv
  source venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
}

setup_conda() {
	source $(conda info --base)/etc/profile.d/conda.sh
	if [  $(uname -s) == "Linux" ]; then
		conda env create -f linux_environment.yaml
		conda install cudatoolkit=11.3 pytorch -c pytorch -y
	elif [ $(uname -s) == "Darwin" ]; then
		if [ $(uname -m) =~ "arm" ]; then
			conda env create -f macos_arm_environment.yaml
		elif [ $(uname -m) =~ "x86" ]; then
			conda env create -f macos_x86_environment.yaml
		fi
	fi
	conda activate scilabrl
}

get_mujoco() {

  pkgs='libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf'
  for pkg in $pkgs; do
    status="$(dpkg-query -W --showformat='${db:Status-Status}' "$pkg" 2>&1)"
    if [ ! $? = 0 ] || [ ! "$status" = installed ]; then
      echo "You appear to be missing dependencies for MuJoCo. Install them with"
      echo "sudo apt-get install libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf"
      return
    fi
  done

  # Check if MuJoCo is already installed
  if [ -d "${HOME}/mujoco210" ]; then
    echo "Skipping MuJoCo as it is already installed."
    return
  fi
  # Get MuJoCo
  echo "Getting MuJoCo"
  wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O "${HOME}/mujoco.tar.gz"
  tar -xf "${HOME}/mujoco.tar.gz" -C "${HOME}"
  rm "${HOME}/mujoco.tar.gz"
  # Install mujoco-py
  echo "Installing mujoco-py"
  source venv/bin/activate
  source set_paths.sh
  pip install mujoco-py
}

get_rlbench() {
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


# check if conda is installed
if [ ! -x "$(command -v conda)" ]; then
  setup_venv
else
  setup_conda
fi

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
