---
layout: default
title: Run an experiment
parent: Wiki
has_children: false
nav_order: 4
---

This tutorial shows how to run the standard algorithm _soft actor-critic_ (SAC) with _hindsight experience replay_ (HER) from stable baselines 3 in the Scilab-RL framework. There are two main ways to start an algorithm. Either via the terminal:

- navigate to Scilab-RL folder
- activate the virtual environment with `conda activate scilabrl`
- set the paths with `source set_paths.sh`
- run SAC with HER on the FetchReach environment with `python src/main.py algorithm=sac env=FetchReach-v2`

Alternatively, you can run it via Pycharm (see the [tutorial](Pycharm)).

In the remainder of this tutorial, we explain the console output of the experiment we've just started!

```
MLFlow experiment name Default
wandb: Currently logged in as: USER (use `wandb login --relogin` to force relogin)
wandb: Tracking run with wandb version 0.12.14
wandb: Run data is saved locally in /home/USER/PycharmProjects/Scilab-RL/data/fa32268/FetchReach-v2/09-08-48/wandb/run-20220420_090848-2872w2rk
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run rose-breeze-83
wandb: ⭐️ View project at https://wandb.ai/USER/sac_FetchReach-v2
wandb: 🚀 View run at https://wandb.ai/USER/sac_FetchReach-v2/runs/2872w2rk
```
These lines are displayed because we use _MLFlow_ and _Weights and Biases_ to track our experiments more on that in the [Display logged data tutorial](Display-logged-data).

After that, the experiment configuration is printed ([configuration tutorial](Hyperparameter-management-with-Hydra)). The first lines show us that we run SAC with hindsight experience replay (HER) and that we train on the _FetchReach_ environment.

```
Starting training with the following configuration:
env: FetchReach-v2
env_kwargs: {}
render: none
render_freq: 5
render_metrics_train:
- train/rollout_rewards_step
render_metrics_test:
- eval/rollout_rewards_step
render_frames_per_clip: 0
seed: 0
base_logdir: data
restore_policy: null
eval_after_n_steps: 2000
n_epochs: 60
n_test_rollouts: 10
early_stop_last_n: 3
early_stop_threshold: 0.9
early_stop_data_column: eval/success_rate
info: ''
save_model_freq: 0
wandb: 1
project_name: null
entity: null
group: null
tags: null
hyperopt_criterion: eval/success_rate
algorithm:
  name: sac
  verbose: true
  replay_buffer_class: HerReplayBuffer
  policy: MultiInputPolicy
  learning_starts: 1000
```
This shows more of our configuration, including the maximal number of epochs `n_epochs`, the early stopping arguments ([early stopping tutorial](Setting-early-stopping-values)), the render arguments ([rendering tutorial](Visualization)) and whether to use Weights and Biases (`wandb`).

```
Log directory: /home/username/PycharmProjects/Scilab-RL/data/bd47b82/FetchReach-v2/09-34-28
Starting process id: 9843
Active mlflow run_id: d281cb1f6cdb4616a6f3774a85792f54
Using cuda device
Launching training
```
The next part shows the Log directory, the mlflow and process ID's, that we use the cpu (and not the gpu) for the neural network computations, what wrappers our environment is wrapped in, and that we now Launch the training!

Now, after each epoch, the following status report is printed to the console:
```
Eval num_timesteps=2000, episode_reward=-46.70 +/- 9.90
Episode length: 50.00 +/- 0.00
Success rate: 0.00%
---------------------------------
| eval/              |          |
|    mean_ep_length  | 50       |
|    mean_reward     | -46.7    |
|    success_rate    | 0.0      |
| time/              |          |
|    total timesteps | 2000     |
| train/             |          |
|    actor_loss      | -12.9    |
|    critic_loss     | 0.198    |
|    ent_coef        | 0.557    |
|    ent_coef_loss   | -3.92    |
|    learning_rate   | 0.0003   |
|    n_updates       | 1949     |
---------------------------------
```
This shows some of the metrics with which we monitor the experiment. The most important ones are 
- the `eval/success_rate`, which shows the rate of successful evaluation episodes
- the `time/total_timesteps`, which shows the total number of training steps so far.

After a while, in this case 18000 timesteps, the early stopping threshold is met and the training finishes:
```
Early stop threshold for eval/success_rate met: Average over last 3 evaluations is 1.0 and threshold is 0.9. Stopping training.
Training finished!
Saving policy to /home/username/PycharmProjects/Scilab-RL/data/bd47b82/FetchReach-v2/09-34-28/rl_model_finished
Hyperopt score: 0.15679012330961817, epochs: 9.
```
After that, Weights and Biases already summarizes the experiment:
```
wandb: Waiting for W&B process to finish... (success).
wandb: - 0.001 MB of 0.001 MB uploaded (0.000 MB deduped)
wandb:                                                                                
wandb: 
wandb: Run history:
wandb:  eval/mean_ep_length ▁▁▁▁▁▁▁
wandb:     eval/mean_reward ▁▁▁▁███
wandb:    eval/success_rate ▁▁▁▁███
wandb:       hyperopt_score ▁
wandb: time/total timesteps ▁▂▃▅▆▇█
wandb:     train/actor_loss ▂▁▃▆▇██
wandb:    train/critic_loss █▇▇▄▄▂▁
wandb:       train/ent_coef █▅▃▂▁▁▁
wandb:  train/ent_coef_loss █▅▂▁▂▅▇
wandb:  train/learning_rate ▁▁▁▁▁▁▁
wandb:      train/n_updates ▁▂▃▅▆▇█
wandb: 
wandb: Run summary:
wandb:  eval/mean_ep_length 50.0
wandb:     eval/mean_reward -1.4
wandb:    eval/success_rate 1.0
wandb:       hyperopt_score 0.20408
wandb: time/total timesteps 14000
wandb:     train/actor_loss -2.33283
wandb:    train/critic_loss 0.09809
wandb:       train/ent_coef 0.02225
wandb:  train/ent_coef_loss -5.70626
wandb:  train/learning_rate 0.0003
wandb:      train/n_updates 13949
wandb: 
wandb: Synced stilted-dawn-47: https://wandb.ai/username/sac_FetchReach-v2/runs/122u2g70
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20220527_180745-122u2g70/logs

Process finished with exit code 0
```
At the end of each experiment, we calculate a score for the hyperparameter optimization (hyperopt) and return it. In this case, we did not run hyperopt, so this is not important for us. Learn more about hyperopt in the [hyperparameter optimization tutorial](Hyperparameter-optimization).

If you would like to add your own algorithm you may go to [adding a new algorithm](Adding-a-new-Algorithm)