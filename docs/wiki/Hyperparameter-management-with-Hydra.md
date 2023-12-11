---
layout: default
title: Hyperparameter management
parent: Wiki
has_children: false
nav_order: 8
---

Each experiment can have a different hyperparameter configuration. Because there are many hyperparameters and many different configurations for our experiments, we manage our hyperparameters and arguments with [hydra](https://hydra.cc/). They also have [nice tutorials](https://hydra.cc/docs/tutorials/intro/)!

Hydra works with configuration files. We store our configurations in `conf`. The following screenshot shows an overview of our configurations.

![image](uploads/0991a38dfd18236520ef1bfa5af537cc/image.png)

## Default parameters
Hydra loads the hyperparameters from `main.yaml`, which specifies the most important default parameters. When we run `python src/main.py`, hydra automatically adds all the default parameters from this config. For example, the default environment is specified in `main.yaml` with `env: 'FetchReach-v2'`. `main.yaml` also contains good explanations for the most important arguments!

## Override arguments
Let's say we want to choose a different environment. Then we can specify it by setting the `env` argument like this: `python src/main.py env=FetchPush-v2`. Note that we do not need any quotes like `""` or `''` for the environment name.

## Add arguments
If the argument is not in the default arguments, we can add it by writing a `+` before it. For example, `env_kwargs.reward_type` is not in the default arguments. We can change the reward type of the environment to `dense` (default is `sparse`) with `+env_kwargs.reward_type=dense`.

## Override or add arguments
With `++`, you can override arguments or add them if they're not in the default arguments. E.g. `++n_epochs=3`.

## Hierarchical configurations
The advantage of hydra is that we do not have to type out all the hyperparameters every run, but can reuse them with hierarchical configurations. For example, we have a configuration file for each algorithm. This contains the default parameters for the algorithm. Hierarchical arguments are referenced with a `.`, e.g. the learning rate of the algorithm can be set with `++algorithm.learning_rate=0.007`.

## Performance
The `performance` folder contains configurations for hyperparameter optimization and performance testing. They have to be run with the `--multirun` flag. E.g. to run performance testing for the `FetchReach-v2` environment with SAC & HER, run `python src/main.py +performance=FetchReach/sac_her-opti --multirun`. More on this in the [Hyperparameter optimization tutorial](Hyperparameter-optimization).

## Smoke test config
`conf/performance/smoke_test.yaml` is the configuration for smoke tests.
