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
  if [[ -x "$(command -v nvidia-smi)" ]]; then
    info "Installing pytorch gpu version"
    # remove potential cpu versions
    conda uninstall pytorch -y
    pip uninstall torch -y
    conda uninstall cpuonly -y
    # install gpu version
    conda install pytorch-cuda=11.7 -c pytorch -c nvidia -y --force-reinstall
  else
    info "Installing pytorch cpu version"
    conda install pytorch=2.0.1 -c pytorch -y
  fi
}

setup_conda() {
  source $(conda info --base)/etc/profile.d/conda.sh
  # check if scilabrl already exists
  if ! [[ -n "$(conda env list | grep 'scilabrl*')" ]]; then
    if [[ $(uname -s) == "Linux" ]]; then
      conda env create -f conda/linux_environment.yaml
    elif [[ $(uname -s) == "Darwin" ]]; then
      if [[ $(uname -m) =~ "arm" ]]; then
        conda env create -f conda/macos_arm_environment.yaml
      elif [[ $(uname -m) =~ "x86" ]]; then
        warn "Intel Macs are currently not supported"
        exit 1
        # conda env create -f macos_x86_environment.yaml
      fi
    fi
  fi
  conda activate scilabrl
  _conda_install_pytorch
  pip install -r requirements.txt
}

install_conda() {
  if [[ $(uname -s) == "Linux" ]]; then
    PYOS="Linux"
  elif [[ $(uname -s) == "Darwin" ]]; then
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
  # change into git root directory
  cd $(git rev-parse --show-toplevel)
  if ! [[ -x "$(command -v conda)" ]]; then
    info "Installing conda"
    install_conda
    success "Conda installed"
  fi
	conda update -n base -c conda-forge conda -y
  setup_conda

  info "Adding source $PWD/scripts/set_paths.sh to rc file of the current shell"
  if [[ -n "$($SHELL -c 'echo $ZSH_VERSION')" ]]; then
    grep -qxF "source $PWD/scripts/set_paths.sh" $HOME/.zshrc || echo "source $PWD/scripts/set_paths.sh" >> $HOME/.zshrc
    source $HOME/.zshrc
  elif [[ -n "$($SHELL -c 'echo $BASH_VERSION')" ]]; then
    grep -qxF "source $PWD/scripts/set_paths.sh" $HOME/.bashrc || echo "source $PWD/scripts/set_paths.sh" >> $HOME/.bashrc
    source $HOME/.bashrc
  else
    warn "Unknown shell, could not setup scripts/set_paths.sh script correctly"
  fi

  conda activate scilabrl

  success "SciLab-RL environment created/updated"
  success "Installation complete."
  info "You must now run 'source ~/.bashrc' to activate conda. Alternatively, you can just restart this shell"
  info "Then, activate the created environment with 'conda activate scilabrl'"
  info "You may check the installation (MuJoCo) via python src/main.py n_epochs=1 wandb=0 env=FetchReach-v2"
}

main "$@"
