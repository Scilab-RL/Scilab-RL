---
layout: default
title: Wiki
has_children: true
nav_order: 2
---

First information about setting up this repository can be found in our [readme](https://github.com/Scilab-RL/Scilab-RL#readme). This wiki contains tutorials and some additional tips and tricks. Take a look at the tutorials below!

The most common use case for a researcher is probably to extend an existing algorithm with a new feature or mechanism. This is covered in the tutorial to [Adding a new algorithm](Adding-a-new-Algorithm). 

# Tutorials

How to configure [Pycharm](Pycharm) for this project.

How to [run a standard algorithm](Run-a-standard-algorithm)(SAC with Hindsight Experience Replay from stable baselines 3) and interpret the console output. This is also a good starting point for checking out 
other tutorials.

How to [visualize and render](Visualization).

How to [manage hyperparameters and arguments with Hydra](Hyperparameter-management-with-Hydra).

How to [log data](Logger).

How to [display logged data](Display-logged-data) in MLFlow and Weights and Biases.

How to [restore a saved policy](Restore-a-saved-policy).

How to [set early stopping values](Setting-early-stopping-values).

As an example of how to create a new different algorithm from an existing one, we [create a copy of CLEANSAC and start modifying it](Adding-a-new-Algorithm)

How to [perform hyperparameter optimization](Hyperparameter-optimization).

How to [set up and perform a performance test](Performance-tests).

How to [set up and perform a smoke test](Smoke-tests).

How to [add a new robotic environment](Adding-a-new-Environment).

# Further information

[How are the files structured?](File-structure)

We created an [overview over our custom environments](Environments-Overview).

[Tips for training on remote computers](Tips-for-training-on-remote-computers).

Detailed instructions on [setting up the project, especially for Windows with WSL2](Detailed-Instructions-for-installation-and-getting-started.md).

How our [GitLab pipeline](GitLab-Pipeline) works.
