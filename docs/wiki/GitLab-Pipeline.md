---
layout: default
title: GitLab Pipeline
parent: Wiki
has_children: false
nav_order: 13
---

Each time someone pushes to a merge request, we run a GitLab pipeline to check if all [smoke-tests](Smoke-tests) pass. The pipeline fails, if the project cannot be set up or if any test fails. But how does it work?

# The .gitlab-ci.yml

The pipeline is defined in the `.gitlab-ci.yml`. Here, we tell GitLab to only run the pipeline for merge requests and not for every push to every branch. We also choose our custom _Docker image_ and tell GitLab to cache the pip-cache (so that it does not have to download all the packages every time the pipeline runs). After that, we define a job that makes GitLab run shell commands in the container which set up the framework by installing the python libraries and then run the smoke-tests.

# Docker 
Docker is a way to package software so it can run on any hardware. It solves the problem that some code may run on one machine but not on another. This is because everything is run inside a _Docker container_, which will be the same on every machine. A _Docker container_ is a running instance of a _Docker image_. For our project, we created a Docker image with Ubuntu 22.04 and MuJoCo. You can find it in GitLab under _Packages & Registries > Container Registry_. This image was created with the _Dockerfile_, which you can find under `Docker/Dockerfile`.

# GitLab runner
The pipeline has to run on a machine, in a _GitLab runner_. We currently run our experiments on a runner we set up at the TUHH. You can set the runner at _Settings > CI/CD > Runners_.

# Recommended tutorial videos
- [GitLab CI CD Pipeline Tutorial | Introduction | 2022](https://www.youtube.com/watch?v=mnYbOrj-hLY)
- [Learn Docker in 7 Easy Steps - Full Beginner's Tutorial](https://www.youtube.com/watch?v=gAkwW2tuIqE)
