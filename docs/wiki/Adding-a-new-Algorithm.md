---
layout: default
title: Add new algorithms
parent: Wiki
has_children: false
nav_order: 1
---

You may want to create your own reinforcement learning algorithm by modifying an existing one.
As an example for this, we create a copy of the one-file soft actor-critic (SAC) algorithm implementation `cleansac` and modify it.

## copy cleansac

- copy the `cleansac` folder `src/custom_algorithms/cleansac`
- paste it into the `src/custom_algorithms` folder and name it something other than "cleansac", e.g. "cleansac_mod"
- rename the python file, `cleansac.py` ➡ `cleansac_mod.py`, class-name (e.g. to _CLEANSAC_MOD_), docstring, _super()_ calls etc.
- adjust the `__init__.py` in the cleansac_mod folder to import your custom version of CLEANSAC
- create an algorithm configuration file `cleansac_mod.yaml` in `conf/algorithm` by copy-pasting `cleansac.yaml` and change `name: 'cleansac'` to `name: 'cleansac_mod'`
- start the algorithm with `python src/main.py algorithm=cleansac_mod`

## start modifying it

CLEANSAC uses multiple critic-networks that assess the value of the actions from the actor network.
Empirically, it is best to take the minimal value of these critics (the most pessimistic one) as the q-value for the actor.
Let's say you have the hypothesis that it would be better to take the maximum of these q-values (the most optimistic value estimate).
You can modify the code as follows to test your hypothesis:

```
# original
min_crit_next_target = torch.min(crit_next_targets, dim=0).values
min_crit_next_target -= ent_coef * next_state_log_pi
    next_q_value = replay_data.rewards.flatten() + \
                   (1 - replay_data.dones.flatten()) * self.gamma * min_crit_next_target.flatten()

...

min_crit_pi = torch.min(self.critic(observations, pi), dim=0).values
actor_loss = ((ent_coef * log_pi) - min_crit_pi).mean()
```

```
# modified
max_crit_next_target = torch.max(crit_next_targets, dim=0).values
max_crit_next_target -= ent_coef * next_state_log_pi
    next_q_value = replay_data.rewards.flatten() + \
                   (1 - replay_data.dones.flatten()) * self.gamma * max_crit_next_target.flatten()

...

max_crit_pi = torch.max(self.critic(observations, pi), dim=0).values
actor_loss = ((ent_coef * log_pi) - max_crit_pi).mean()
```

It would be more convenient to have a command line parameter with which we can choose between min and max. 
- add the line `critic_select: 'min' # allows values ['min', 'max']` to `cleansac_mod.yaml`. 
- add the new parameter to the constructor of _CLEANSAC_MOD_ : `critic_select: str = 'min',`. You could also add other constructor parameters like the `learning_rate` to the config or just use hydras overwrite syntax (e.g. `++algorithm.learning_rate=0.007`, see also the [hydra tutorial](Hyperparameter-management-with-Hydra)).
- add the parameter to the instance of the _CLEANSAC_MOD_ object with `self.critic_select = critic_select`.
- now you can modify the code accordingly:

```
if self.critic_select == 'min':
    min_crit_next_target = torch.min(crit_next_targets, dim=0).values
    min_crit_next_target -= ent_coef * next_state_log_pi
        next_q_value = replay_data.rewards.flatten() + \
                       (1 - replay_data.dones.flatten()) * self.gamma * min_crit_next_target.flatten()
elif self.critic_select == 'max':
    max_crit_next_target = torch.max(crit_next_targets, dim=0).values
    max_crit_next_target -= ent_coef * next_state_log_pi
        next_q_value = replay_data.rewards.flatten() + \
                       (1 - replay_data.dones.flatten()) * self.gamma * max_crit_next_target.flatten()
else:
    assert False, "Error, invalid value for critic_select"
    
...

if self.critic_select == 'min':
    min_crit_pi = torch.min(self.critic(observations, pi), dim=0).values
    actor_loss = ((ent_coef * log_pi) - min_crit_pi).mean()
elif self.critic_select == 'max':
    max_crit_pi = torch.max(self.critic(observations, pi), dim=0).values
    actor_loss = ((ent_coef * log_pi) - max_crit_pi).mean()
else:
    assert False, "Error, invalid value for critic_select"
```

You can now choose min/max via the command line parameter `algorithm.critic_select` and run experiments to find out which is more successful. You could either start a number of runs with either configurations like in the [Weights and Biases tutorial](Display-logged-data), or [perform hyperparameter optimization](Hyperparameter-optimization) (recommended).

# integrate into framework
If you want to add your algorithm to the framework, create a merge request. GitLab will automatically run a pipeline with smoke-tests to see if your changes broke anything. Your algorithm will automatically be tested during the [smoke tests](Smoke-tests).

# Add new Python packages
If your new algorithm requires new python packages, you must put them into the requirements.txt
