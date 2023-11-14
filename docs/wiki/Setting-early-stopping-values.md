---
layout: default
title: Early stopping
parent: Wiki
has_children: false
nav_order: 10
---

If the policy is good enough, we want to stop the experiment to not waste any computing resources. How to do this early stopping is defined by the following parameters:

- `early_stop_data_column: 'eval/success_rate'`, 
the data column on which early stopping is based.

- `early_stop_threshold: 0.9`, 
the early stopping threshold.

- `early_stop_last_n: 3`,
the n last epochs over which to average for determining early stopping condition. This is also the minimal number of epochs to run before early stopping.

With these default parameters, the experiment stops if the average `eval/success_rate` over the last 3 epochs is greater or equal 0.9.

As an example for a different early stopping setting, you could change it to immediately stop if the mean reward over the test episodes is more than -40 by setting: `early_stop_data_column=eval/mean_reward early_stop_threshold=-40 early_stop_last_n=1`
