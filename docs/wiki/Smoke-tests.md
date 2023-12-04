---
layout: default
title: Smoke tests
parent: Wiki
has_children: false
nav_order: 12
---

Smoke tests check whether everything still works e.g. after the upgrade of a python package or changes in the code. To check the performance of an algorithm, use [performance tests](Performance-tests).

> 💡 The smoke-tests run in our [GitLab pipeline](GitLab-Pipeline) each time someone pushes to a merge request.

# Run the smoke tests
- open the Scilab-RL project in a terminal
- activate the virtual environment with `source venv/bin/activate`
- set the environment variables with `source set_paths.sh`
- run the smoke tests with `./scripts/run_smoke_tests.sh`
- open another terminal and start _MLFlow_ with `mlflow ui --host 0.0.0.0`, then open _MLFlow_ in your browser at http://0.0.0.0:5000/#/
- choose _smoke_test_ under _Experiments_ on the left and see whether each algorithm-environment combination successfully finished ✅ or failed ❌

# Configure the smoke tests
We run smoke tests for all algorithms and a selection of environments in `run_smoke_tests.sh`. Each algorithm that has a config in `conf/algorithm` will automatically be tested. We do not test all environments, because there are too many. However, if you added a new type of environment to the framework, please add it to `run_smoke_tests.sh`.
