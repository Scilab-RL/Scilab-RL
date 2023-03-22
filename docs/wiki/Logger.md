---
layout: default
title: Logger
parent: Wiki
has_children: false
nav_order: 14
---

We use the [stable baselines 3 Logger](https://stable-baselines3.readthedocs.io/en/master/common/logger.html). To use the logger in a custom algorithm, just pass the logger instance to the instance of your algorithm and use it with `logger.info()` or `logger.error()` etc.

This logger allows to log to different outputs. Currently we log to four different outputs:
- To the console in the `FixedHumanOutputFormat`
- To a `train.log` file in the `FixedHumanOutputFormat`
- To _MLFlow_ with the `MLFlowOutputFormat`
- To _Weights and Biases_ with the `WandBOutputFormat`

These output formats are specified in `src/util/custom_logger.py`. You could create another custom output format, e.g. for another experiment tracking software, and add it to the logger with `logger.output_formats.append(AnotherCustomOutputFormat())`.
