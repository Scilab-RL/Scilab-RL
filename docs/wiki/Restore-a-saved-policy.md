---
layout: default
title: Restore a policy
parent: Wiki
has_children: false
nav_order: 7
---

The early-stopping policy of a run is always stored in the `data` folder for the run. Let's say we ran an experiment with `python src/main.py env=FetchReach-v2 algorithm=sac`. Then, at the start of the console output, it will have printed the Log directory:
```
Log directory: /home/USER/PycharmProjects/Scilab-RL/data/fa32268/FetchReach-v2/15-26-33
```
In this folder, we find the `early_stop_agent.zip`. 

A common use case is to use stored policies to observe visually (by rendering) how an agent behaves. For example, assume that we'd like to look at the early-stopping agent to see if it really solved the reacher-task well. Therefore, 
- we use the `+restore_policy` parameter and set it to the path of the `early_stop_agent.zip` file. 
- We also set the [render_args](Visualization) so that we display the evaluation.
- Furthermore, we set `wandb=0`, because we do not want to track this run and `n_epochs=1` because we only want to display one epoch. 
- Finally, we set `eval_after_n_steps` to 1 because we do not want the policy to train, as it is already trained. `eval_after_n_steps` hast to be at least 1.

The resulting command looks like this:
```
python src/main.py env=FetchReach-v2 algorithm=sac +restore_policy=/home/USER/PycharmProjects/Scilab-RL/data/fa32268/FetchReach-v2/15-26-33/early_stop_agent.zip render_args=[[none,1],[display,1]] wandb=0 n_epochs=1 eval_after_n_steps=1
```

> :warning: Note that we do not store any replay buffers. That means that trained off-policy algorithms can be restored to be displayed, but not to be further trained as if the training had not been interrupted.

We can also record the evaluation. The video is stored in the log directory. The log directory has the same name as the directory from which the policy is restored, but with a `_restored` postfix. This is what 10 successful evaluation-episodes look like:

![eval_0](uploads/30c9bcea5b30138b8acb7ed220fb3890/eval_0.mp4)
