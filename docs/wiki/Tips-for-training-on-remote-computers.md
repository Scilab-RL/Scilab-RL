---
layout: default
title: Train on remote machine
parent: Wiki
has_children: false
nav_order: 15
---

Training or running a hyperparameter optimization for your model takes much computational power that you might not have on your local machine. Luckily you can run your experiments on remote computers!

Here are some useful tools and hints for running on remote machines:

# tmux

To be able to run multiple experiments in parallel or to close your connection and not terminate your processes, `tmux` is very useful.

You can create a session with `tmux`. To detach from a _tmux_ session, press `Ctrl + B` and then `D` immediately afterwards. To reattach, run `tmux attach -t SESSION_INDEX` (SESSION_INDEX is 0 if you have only one _tmux_ session). List your _tmux_ sessions with `tmux ls`. Kill a _tmux_ session with `tmux kill-session -t SESSION_INDEX`. To scroll inside a _tmux_ session, press `Ctrl + B` `PageUp` (or `PageDown` for the other direction), `Ctrl + C` to leave the scroll mode.

In your _tmux_ session, activate your _venv_ with 
```
conda activate scilabrl
source ./set_paths.sh
```

# Choosing a GPU

You can select the GPU you want to train on with e.g. `export CUDA_VISIBLE_DEVICES=0` in your _venv_.

# Mounting storage

You can **mount the server storage** to your PC with

`mkdir /storage/remote_pc` (creates a folder to mount to)

`chmod 777 /storage/remote_pc` (sets the permissions for this folder)

`sshfs USERNAME@remote_server.de:/data/USERNAME /storage/remote_pc` (mounts the storage so you can access it from your local PC through the _Files_ UI.) If you're not mounting for the first time, this is the only command you have to run.

# Other

**See all your processes** with `ps -ef | grep USERNAME`

**Kill all your processes** with `pkill -u USERNAME`
