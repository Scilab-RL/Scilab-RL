---
layout: default
title: Performance tests
parent: Wiki
has_children: false
nav_order: 11
---

Performance tests check if the performance of an algorithm is still as good as before, e.g. after the upgrade of a python package, a common utility function or changes in the code. If you just want to check whether an algorithm or environment works with the rest of the framework _at all_, use [smoke tests](Smoke-tests).


We use `conf/performance/FetchReach/sac_her-test.yaml` as an example. Here is the config:
```
# @package _global_
# Changes specified in this config should be interpreted as relative to the _global_ package.
defaults:
  - override /algorithm: sac

env: 'FetchReach-v2'

algorithm:
  replay_buffer_class: HerReplayBuffer
  buffer_size: 1000000
  batch_size: 256
  learning_rate: 0.001
  gamma: 0.96
  tau: 0.07
  policy_kwargs:
    n_critics: 1
    net_arch:
      - 256
      - 256
      - 256
  replay_buffer_kwargs:
    n_sampled_goal: 4
    goal_selection_strategy: 'future'

```
This just specifies that the performance test is for the `sac` algorithm with the HER replay buffer and the `FetchReach-v2` environment.

```
performance_testing_conditions:
  # In 2 out of 3 tests, the eval/success rate should be at least 0.9 after 10k steps.

  total_runs: 3 # How many runs in total:

  succ_runs: 2 # This number of runs should meet the conditions:

  eval_columns: eval/success_rate # This is what we evaluate to determine success. Will use this to override the \'early_stop_data_column\' parameter of main.yaml

  eval_value: 0.9 # This is the value we determine for success. Will use this to determine and override the \'early_stop_threshold\' parameter of main.yaml

  # ca. 15 minutes on GPU
  max_steps: 10_000 # This is the time limit for checking the success. Will use this and the \'eval_after_n_steps\' parameter of main.yaml to determine the n_epochs parameter in main.yaml.
```
As the comments in the config already explain, the performance test tests whether sac&her can achieve at least 90% `eval/success_rate` after 10k steps in 2 of 3 runs.

```
hydra:
  sweeper:
    n_jobs: 3
    _target_: hydra_plugins.hydra_custom_optuna_sweeper.performance_testing_sweeper.PerformanceTestingSweeper
    study_name: sac_her_FetchReach-test
```
The last part of the config sets the number of parallel runs with `n_jobs`, specifies that we use the `PerformanceTestingSweeper` and names the study.

You can start this performance test with `python src/main.py +performance=FetchReach/sac_her-test --multirun`

If the test is successful, the `PerformanceTestingSweeper` prints something like `[2022-04-22 13:10:25,558][HYDRA] Performance test FetchReach/sac_her-test successful! The value for eval/success_rate was at least 0.9 in 3 runs.` to the console.

Use `run_performance_tests.sh` to run multiple performance tests sequentially.
