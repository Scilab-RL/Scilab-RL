---
layout: default
title: Home
has_children: false
nav_order: 1
---

This is the Scilab-RL repository focusing on goal-conditioned reinforcement learning using the [stable baselines 3](https://stable-baselines3.readthedocs.io/en/master/) methods and [Gymnasium](https://gymnasium.farama.org/) interface.

![](overview.svg)

The framework is tailored towards the rapid prototyping, development and evaluation of new RL algorithms and methods. It has the following unique selling-points compared to others, like spinning up and stable baselines:
* Built-in data visualization for fast and efficient debugging using MLFLow and Weights & Biases
* Support for many state-of-the-art algorithms via stable baselines 3 and extensible to others
* Built-in hyperparameter optimization using Optuna
* Easy development of new robotic simulation and real robot environments based on MuJoCo
* Smoke and performance testing
* Compatibility between a multitude of state-of-the-art algorithms for quick empirical comparison and evaluation
* A focus on goal-conditioned reinforcement learning with hindsight experience replay to avoid environment-specific reward shaping
