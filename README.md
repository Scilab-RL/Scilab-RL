# ideas_deep_rl2

This is the IDEAS / LeCAREbot deep RL repository focusing on hierarchical goal-conditioned reinforcement learning using the stable baselines 3 methods and OpenAI gym interface

## Requirements:
- Python 3.6+

## Getting started

1. Download MuJoCo (mujoco.org) and obtain a license
   (as student you can obtain a free one-year student license).
   Copy the mjpro200_linux folder from the downloaded archive
   as well as mjkey.txt that you will obtain from the registration
   to folders of your choice.

1. Set the environment variables in `set_paths.sh` according to the
   locations where you saved the mjpro200_linux folder and the mjkey.txt.
   If you are using an IDE, set the variables there as well.

1. generate a virtual python3 environment with

    `virtualenv -p python3 venv`

1. load the environment with

    `source venv/bin/activate`

1. upgrade the version of pip
    `pip install --upgrade pip`

1. set environment variables and paths with

    `source ./set_paths.sh`

1. install required python libraries with

    `pip install -r requirements.txt`

1. Add the mujoco binary path to the  `LD_LIBRARY_PATH` environment variable. If you first run the script without doing this it will automatically remind you and identify the path for you.


## Start training manually (hydra debugging)

```bash
python3 experiment/train.py env=FetchReach-v1 algorithm=hac algorithm.render_test=record algorithm.time_scales=[5,-1]
python experiment/train.py env=FetchReach-v1 algorithm=hac layer_classes=['sacvg','ddpg']
```

```bash
python experiment/train.py env=FetchReach-v1 algorithm=her2 layer_classes=['sac']
# also works with ddpg
```


## File structure
* The main script from which all algorithms are started is `train.py`.
* The root directory contains shell scripts for automated testing and data generation.
* The folder `experiment` contains all architectural stuff for evaluation.

    * `plot.py` is for plotting

    * `testing_algos.py` is for defining testing parameters

    * `testing_envs.py` is for defining testing environments

    * `click_options.py` is for setting the global (model-independent) command-line parameters

    * `check_error_logs.py`, `generate_testing_commands.py` and `validate_performance_testing.py` is for executing evaluations and tests.

* The folder `ideas_baselines` contains the new HAC implementation and an implementation of HER. Other new algorithms should be added here, too. For details on the specific algorithms, see below.
* The folder `ideas_envs` should contain new environments (but we may also choose to put environments in a completely different repository).
* The folder `interface` contains for each algorithm, both stable-baselines3 algorithms and the algorithms here, a file `config.py` and `click_options.py`. The click options file determines the kwargs passed on to the model (HAC, SAC, TD3, etc). These are specifyable as command-line options. The file `config.py` is right now just for determining the parameters to be used for generating a path name for the log directory.
* The folder `util` contains some misc utilities.
* The folder `hydra_plugins` contains some customized plugins for our hyperparameter management system.

## Testing

1. generate a file with all possible environment/algorithm combinations with:

    `./run_testing.sh -g <comma-separated list of gpu_ids in paremtheses, e.g. '(0,1)'> -p <max processes> -m <min. free GPU memory required to star a new process> -s <time to sleep before executing next command> -t <type of testing <function|performance>>`

    The defaults for these parameters are g=0, p=6, m=2500, t='performance', s=15

    The script will create a file `test_cmds.txt` in the base directory with all currently supported environment-algorithm combinations and useful hyperparameters. It will also execute them and perform two checks:

    1. A function check to determine whether a process crashed
    1. A performance test to determine whether certain predefined success rates are met. These success rates are defined in `experiment/testing_algos.py` (see Section on Testing below.)

## Algorithms

### HAC
TBD

## Limitations
Currently, only off-policy algorithms are supported: DQN, DDPG, TD3 and SAC. PPO is not supported

## Hyperparameter optimization and management
The framework has a sophisticated hyperparameter management and optimization pipeline. It builds on the following four tools: 
### Hydra
Hydra manages the command-line parameters and configuration options. 
The command line parameters are set in the `conf/main.yaml` file. 
All algorithm-specific parameters are set in the `conf/algorithm/<alg_name>/yaml` file.  

### Optuna
Optuna is a framework to perform the hyperparameter optimization algorithm (e.g. TPE). 
Optuna integrates flawlessly with hydra. 
To set up the search space for the hyperparameter optimization look at the `search_space` section of the `conf/main.yaml` file.
The default optuna launcher for hyperopting is the joblib launcher which spawns several loky processes. 
A disadvantage of this approach is that the logging to stdout is disabled. 
However, you will be able to read all console outputs produced by `logger.info()` in the log folder.
The hyperparameter search is implemented via a customized sweeper plugin located in the `hydra_plugins` subfolder. 
By default, the sweeper uses the TPE optimizer.
You can control the behavior of the sweeper in the `hydra/sweeper` section of `conf/main.yaml`.
The sweeper automatically learns when to early stop trials by remembering past trials.
It sets the max. number of epochs for a new trial to 1.5 times the number of epochs of the so-far fastest trial (when terminated by early stopping).
The sweeper stops after the set number of trials or the specified duration, as specified in the config file. 
For convenience, the sweeper also creates a file `delete_me_to_stop_hyperopt`, which you just need to delete to soft-stop the hyperopting after the current batch of jobs.  

### Mlflow
Mlflow collects studies and logs data of all runs. 
The information about all runs is stored in a subfolder called `mlruns`. 
You can watch the mlflow runs by executing `mlflow ui --host 0.0.0.0` in the root folder of this project, which will call a web server that you can access via port 5000 (by default). 
The `--host` tells the server to allow connections from all machines.

### Comet.ml 
  Comet.ml is a cloud-based framework for logging and visualizing all data, including GPU usage, memory usage, and even the Python code and console output. You can obtain a free academic license on the website. 
  Comet.ml seamlessly integrates with mlflow, just by having the `import comet_ml` at the top of the `experiment/train.py`. 
  Note that this import monkey-patches several other modules, including mlflow, which is why it has to be at the top of the `train.py` file. 
  Also, there are some conflicts with the joblib multiprocessing library if using the standard `_DEFAULT_START_METHOD` of joblib/loky because the standard multiprocessing re-loads all modules and overwrites the monkey-patched ones. 
  Therefore, there is an overwrite just below the comet_ml import, telling joblib/loky to use `loky_init_main`.
  To upload the results to comet.ml, you need to specify your API key that you obatain when you register with comet.ml. 
  There are several options to do this. 
  The recommended option is to create a config file `~/.comet.config` (in your home folder, note the `.` in the file name). 
  The config file should have the following content:
```
[comet]
   api_key=<your API key>
```


To start the hyperparameter optimization start `experiment/train.py --multirun`. The `--multirun` flag starts the hyperparameter optimization mode. 




  
