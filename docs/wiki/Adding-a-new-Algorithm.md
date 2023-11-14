---
layout: default
title: Add new algorithms
parent: Wiki
has_children: false
nav_order: 1
---

You may want to create your own reinforcement learning algorithm by modifying an existing one. As an example for this, we create a copy of the soft actor-critic (SAC) algorithm implementation of stable baselines 3 and modify it.

## copy SAC

- copy the SAC folder `venv/lib/python3.10/site-packages/stable_baselines3/sac`
- paste it into the `src/custom_algorithms` folder and name it something other than "sac", e.g. "sac_mod"
- rename the python file, `sac.py` :arrow_right: `sac_mod.py`, class-name (e.g. to _SAC_MOD_), docstring, _super()_ calls etc.
- remove unnecessary duplicates that we do not want to modify, e.g. `policies.py`
- adjust the `__init__.py` in the sac_mod folder to import your custom version of SAC
- create an algorithm configuration file `sac_mod.yaml` in `conf/algorithm` and write `name: 'sac_mod'` and `verbose: True` into it
- start the algorithm with `python src/main.py algorithm=sac_mod`

## start modifying it

SAC uses multiple critic-networks that assess the value of the actions from the actor network. Empirically, it is best to take the minimal value of these critics (the most pessimistic one) as the q-value for the actor. Let's say you have the hypothesis that it would be better to take the maximum of these q-values (the most optimistic value estimate). You can modify the code as follows to test your hypothesis:

```
# original
next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
```

```
# modified
next_q_values, _ = th.max(next_q_values, dim=1, keepdim=True)
```

It would be more convenient to have a command line parameter with which we can choose between min and max. 
- add the line `critic_select: 'min' # allows values ['min', 'max']` to `sac_mod.yaml`. 
- add the new parameter to the constructor of _SAC_MOD_ : `critic_select: str = 'min',`. You could also add other constructor parameters like the `learning_rate` to the config or just use hydras overwrite syntax (e.g. `++algorithm.learning_rate=0.007`, see also the [hydra tutorial](Hyperparameter-management-with-Hydra)).
- add the parameter to the instance of the _SAC_MOD_ object with `self.critic_select = critic_select`.
- now you can modify the code accordingly:

```
# modified with selection
if self.critic_select == 'min':
    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
elif self.critic_select == 'max':
    next_q_values, _ = th.max(next_q_values, dim=1, keepdim=True)
else:
    assert False, "Error, invalid value for critic_select"
```

You can now choose min/max via the command line parameter `algorithm.critic_select` and run experiments to find out which is more successful. You could either start a number of runs with either configurations like in the [Weights and Biases tutorial](Display-logged-data), or [perform hyperparameter optimization](Hyperparameter-optimization) (recommended).

# integrate into framework
If you want to add your algorithm to the framework, create a merge request. GitLab will automatically run a pipeline with smoke- and performance-tests to see if your changes broke anything. Your algorithm will automatically be tested during the [smoke tests](Smoke-tests).

# Add new Python packages
If your new algorithm requires new python packages, you must put them into the requirements.txt
