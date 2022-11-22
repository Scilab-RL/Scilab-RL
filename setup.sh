#!/bin/bash

info () {
  printf "\r  [ \033[00;34m..\033[0m ] $1\n"
}

success () {
  printf "\r\033[2K  [ \033[00;32mOK\033[0m ] $1\n"
}

warn () {
  printf "\r\033[2K  [\033[0;31mFAIL\033[0m] $1\n"
  echo ''
}

_conda_install_pytorch() {
  # install gpu specific tools
  if [ -x "$(command -v nvidia-smi)" ]; then
    # remove potential cpu versions
    conda uninstall pytorch -y
    pip uninstall torch -y
    conda uninstall cpuonly -y 
    # install gpu version
    conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia -y --force-reinstall
  else
    conda install pytorch==1.11.0 -y
  fi
}

setup_conda() {
  source $(conda info --base)/etc/profile.d/conda.sh
  # check if scilabrl already exists
  if ! [ -n "$(conda env list | grep 'scilabrl*')" ]; then
    if [ $(uname -s) == "Linux" ]; then
      conda env create -f conda/linux_environment.yaml
    elif [ $(uname -s) == "Darwin" ]; then
      if [ $(uname -m) =~ "arm" ]; then
        conda env create -f conda/macos_arm_environment.yaml
      elif [ $(uname -m) =~ "x86" ]; then
        warn "Intel Macs are currently not supported"
        exit 1
        # conda env create -f macos_x86_environment.yaml
      fi
    fi
  fi
  conda activate scilabrl
  pip install -r requirements.txt
  _conda_install_pytorch
}

install_mujoco() {
  if ! [ -d "${HOME}/.mujoco/mujoco210" ]; then
    mkdir -p $HOME/.mujoco/
    info "Getting MuJoCo"
    MUJOCO_VERSION="2.1.1"
    if [ $(uname -s) == "Linux" ]; then
      MUJOCO_DISTRO="linux-x86_64.tar.gz"
      wget -q https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O "${HOME}/mujoco.tar.gz"
      tar -xf "${HOME}/mujoco.tar.gz" -C "${HOME}/.mujoco/"
      # NOTE: If we move to a newer version for linux
      # wget "https://github.com/deepmind/mujoco/releases/download/$MUJOCO_VERSION/mujoco-$MUJOCO_VERSION-linux-x86_64.tar.gz" -O "${HOME}/mujoco.tar.gz"
      # tar -xf "${HOME}/mujoco.tar.gz"
      # mv "${PWD}/mujoco-${MUJOCO_VERSION}" "${HOME}/.mujoco/mujoco210"
      rm "${HOME}/mujoco.tar.gz"
    elif [ $(uname -s) == "Darwin" ]; then
      MUJOCO_DISTRO="macos-universal2.dmg"
      wget -q "https://github.com/deepmind/mujoco/releases/download/$MUJOCO_VERSION/mujoco-$MUJOCO_VERSION-macos-universal2.dmg"
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
    info "Installing mujoco-py and testing import"
    source set_paths.sh
    pip install mujoco-py && python -c 'import mujoco_py'
}

install_rlbench() {
  if [ $(uname -s) == "Darwin" ]; then
    warn "There is no PyRep support for macos"
    return
  fi
  # Check if CoppeliaSim is already installed
  if [ -d "${HOME}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04" ]; then
    warn "Skipping CoppeliaSim as it is already installed."
    return
  fi
  # Get CoppeliaSim
  echo "Getting CoppeliaSim"
  wget -q http://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -O "${HOME}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz"
  tar -xf "${HOME}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz" -C "$HOME"
  rm "${HOME}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz"
  # Get RLBench
  echo "Getting RLBench"
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
  source $(conda info --base)/etc/profile.d/conda.sh
}


main() {
  if ! [ -x "$(command -v conda)" ]; then
    info "Installing conda"
    install_conda
    success "Conda installed"
  fi
	# TODO: update conda
	# conda update -n base -c conda-forge conda
  setup_conda

  info "Adding source $PWD/set_paths.sh to rc file of the current shell"
  if [ -n "$ZSH_VERSION" ]; then
    grep -qxF "source $PWD/set_paths.sh" $HOME/.zshrc || echo "source $PWD/set_paths.sh" >> $HOME/.zshrc
    source $HOME/.zshrc
  elif [ -n "$BASH_VERSION" ]; then
    grep -qxF "source $PWD/set_paths.sh" $HOME/.bashrc || echo "source $PWD/set_paths.sh" >> $HOME/.bashrc
    source $HOME/.bashrc
  else
    warn "Unknown shell, could not setup set_paths.sh script correctly"
  fi

  conda activate scilabrl

  success "SciLab-RL environment created/updated"
  install_mujoco
  success "Mujoco installed/updated"
  install_rlbench
  success "RLBench installed/updated"
  success "Installation complete."
  info "You must now run 'source ~/.bashrc' to activate conda. Alternatively, you can just restart this shell"
  info "Then, activate the created environment with 'conda activate scilabrl'"
  info "You may check the installation (MuJoCo) via python main.py n_epochs=1 wandb=0 env=FetchReach-v1"
}

main "$@"
