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

# Logger and WandB

* Integration with Weights & Biases
The WandBOutputFormat class allows data logged by a Python application to be seamlessly integrated and visualized in WandB.
* It supports logging not only standard numerical metrics but also more complex data types like videos, which is particularly useful in contexts where visual data is essential (e.g., training computer vision models).

## Usage of Logger
* `logger.record`: is used to capture and store individual data points or metrics during the execution of a machine learning model. When you call logger.record, it takes key-value pairs as input, where the key is a string representing the name of the metric, and the value is the metric itself.
    * `self.logger.record("metric_name", metric_value)`

&nbsp;

* `logger.record_mean`: is used to record the average (mean) of a particular metric over a specified period or set of events. This is distinct from logger.record, which logs individual or instantaneous values of metrics.
    * `logger.record_mean("train/loss", loss.item())`

&nbsp;

* `logger.dump`: This command is used to output or save the recorded metrics. When called it automatically writes the logged data to a file, console, or external monitoring tools such as WandB and MLFlow for visualization. 

* Capturing Data: During model training, logger.record is used to capture various metrics and statistics. These are temporarily stored in the logger's internal state.

* Formatting and Logging: Periodically, or at specific points in the training process (like the end of an epoch), the logger's data is passed to its output formats for processing and logging. If WandBOutputFormat is one of the logger's output formats, it receives this data.
* WandB Logging: WandBOutputFormat processes the received data and logs it to WandB. It ensures that the data is in the correct format for WandB and uses WandB's API to send the data to the WandB server. This allows the data to be visualized and analyzed on the WandB dashboard.

You may also check the [display logged data](Display-logged-data) section for detailed information on how the WandB and MLFlow data can be seen and interpreted.