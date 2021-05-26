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


## Start training manually 

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

## Environments
Currently, all goal-conditioned gym environments are supported. A list of all tested environments can be found in `conf/main.yaml`. 
### Adding a new environment
To add a new environment, we suggest to proceed as follows. 

1. As a basis, use an environment that is as close as possible to the new environment you want to develop. A good start for a gripper-based environment is the gym version of FetchReach. Run your favorite algorithm with this base environment and make sure it works as intended. It is also a good idea to take notes about the learning performance (how many training steps required to achieve success). 
1. Make a copy of that environment and add it to the `ideas_envs` folder in the repository. Therefore, create a new subfolder for your new environment under `ideas_envs`, say `ideas_envs/new_env`. Then, say you want to start with the `FetchReach-v1` environment as a basis, and assuming that `venv` is the folder where you have your virtual python 3.6 environment, copy `venv/lib/python3.6/site-packages/gym/envs/robotics/fetch_env.py` to that subfolder. You may want to save it under a different filename, say `new_env.py`. Finally, if you use an OpenAI gym env like FetchReach, you have to create an entry point class for your environment, as found in `venv/lib/python3.6/site-packages/gym/envs/robotics/fetch/reach.py:FetchReachEnv`. Using an entry point class is the preferred way of implementing different variations for each environment. For example, the fetch environment has different entry points for `FetchReach` and `FetchPush`. You find these in `reach.py` and `push.py` respectively. For the following we assume that your entry point class is `NewClassEnv` and your environment name is `NewEnv-v0` (note that the `-vN` suffix is mandatory for each environment name, where `N` is the version number). 
1. Then you need to register your entry point class as a new environment. You can do this by adding it to the `ideas_envs/register_envs.py` file. You find examples for how exactly to do this in the file. 
1. Now try whether your copy of the base environment is running as intended, in the same way as the original one. Therefore, set the `env` parameter in `conf/main.yaml` to `NewEnv-v0`, or add the commandline parameter `env=NewEnv-v0`. 
1. That's it. Now you can take your copy as a basis for your new environment and modify it as you want to develop your new environment. 

## Limitations
Currently, only off-policy algorithms are supported: DQN, DDPG, TD3, SAC, HER and HAC. PPO is not yet supported but it should not be too hard to enable it. 

## Hyperparameter optimization and management
The framework has a sophisticated hyperparameter management and optimization pipeline. 
To start the hyperparameter optimization start `experiment/train.py --multirun`. The `--multirun` flag starts the hyperparameter optimization mode. 
[comment]: <> (A problem with the comet.ml integration is that if a script raises an Error and stops, all parallel processes will be blocked and the error will not be output. Therefore, if you spot that all processes of the hyperopt are idle you should re-start the process without comet.ml &#40;just remove the import in `train.py`&#41; and try to reproduce and find the error by observing the console output.  )
The hyperparameter management and optimization builds on the following four tools: 

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
  Comet.ml is supposed to seamlessly integrate with mlflow, just by having the `import comet_ml` at the top of the `experiment/train.py`. In theory, this should enable auto logging of all sorts of information, but this is not working in multiprocessing mode (see issue https://git.informatik.uni-hamburg.de/eppe/ideas_deep_rl2/-/issues/26).
  As a workaround, the sweeper now calls the `mlflow_to_cometml` script which uploads all data to cometml after each batch of jobs, i.e., we use the live logging functionality and the logging of CPU usage, GPU usage, etc. 
  
Note that the comet.ml import monkey-patches several other modules, including mlflow, which is why it has to be at the top of the `train.py` file. 
  Some conflicts with the joblib multiprocessing library occur when using the standard `_DEFAULT_START_METHOD` of joblib/loky because the standard multiprocessing re-loads all modules and overwrites the monkey-patched ones. 
  Therefore, there is an overwrite just below the comet_ml import, telling joblib/loky to use `loky_init_main`, as this seems not to overwrite the monkey-patched modules.
  
To upload the results to comet.ml in either way, using the import or the `mlflow_to_cometml` script, you need to specify your API key that you obtain when you register with comet.ml. 
  There are several options to do this. 
  The recommended option is to create a config file `~/.comet.config` (in your home folder, note the `.` in the file name). 
  The config file should have the following content:
```
[comet]
   api_key=<your API key>
```
  



  
