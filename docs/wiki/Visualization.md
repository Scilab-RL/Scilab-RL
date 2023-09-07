---
layout: default
title: Visualization
parent: Wiki
has_children: false
nav_order: 6
---

The framework supports multiple ways of rendering and plotting features. Two of these are discussed here: 3D rendering and the plotting of episodoc metric values. 

The 3D rendering displays the simulated robot in its environment, and the plotting enables an engineer or researcher to display metrics that might be useful for presentation and debugging purposes. 

Both features are controlled via the `render_args` option in `main.yaml` config file (or, respectively, via the command line).

Details for using the `render_args` are provided as comments in `main.yaml`. 

> ⚠ There are the following limitations:

* MuJoCo: The training is only displayed in the first epoch if the evaluation should be recorded.

To visualize a certain metric when training using a custom algorithm implementing the stable baselines API, the logger (from util.custom_logger) must record the metric in the logger. It is important that the "_on_step" and "_on_rollout_start" functions are called for all custom callbacks. You can read more about custom callbacks in the stable baselines [documentation.](https://stable-baselines.readthedocs.io/en/master/guide/callbacks.html)


`
q_value = ...
self.logger.record('q_val',q_value)
`

Basic.py includes a concrete example. Then the algorithm can be run as usual. For example, in the case where we want to visualize the metric 'q_val' for each episode/rollout and close the animation after every episode in basic.py, we run:

`python3 src/main.py n_epochs=2 wandb=0 algorithm=basic env=FetchReach-v1 render_args=[['none',1,[['q_val',1],['q_val',1]]],['none',10,[['q_val',1],['q_val',1]]]]`

We concurrently display the interactions of the agent with the environment every step.
