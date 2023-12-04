---
layout: default
title: File structure
parent: Wiki
has_children: false
nav_order: 18
---

* The main script from which all algorithms are started is `src/main.py`.
* The directory `scripts` contains shell scripts for automated testing.
* New algorithms should be added to the `src/custom_algorithms` folder.
* The folder `src/custom_envs` contains our custom environments.
* The file `conf/main.yaml` contains general configurations for the experiment parameters.
* The folder `conf/algorithm` contains configurations for each algorithm, both stable-baselines3 and custom algorithms.
  It determines the kwargs passed on to the model (SAC, TD3, etc).
  These are also overridable as command-line options, e.g. `algorithm.verbose=False`.
* The folder `conf/performance` contains optimization and performance-testing scripts for different environments.
* The folder `src/util` contains some misc utilities.
* The folder `src/hydra_plugins` contains some customized plugins for our hyperparameter management system.
