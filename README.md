# ideas_deep_rl2

This is the IDEAS / LeCAREbot deep RL repository focusing on hierarchical goal-conditioned reinforcement learning using the [stable baselines 3](https://stable-baselines3.readthedocs.io/en/master/) methods and [OpenAI gym](https://gym.openai.com/) interface.

## Requirements:
- Python 3.6+

## Getting started

1. generate a virtual python3 environment with

    `virtualenv -p python3 venv` or
    `python3 -m venv venv`

1. load the environment with

    `source venv/bin/activate`

1. upgrade the version of pip


    `pip install --upgrade pip`

1. install required python libraries with

    `pip install -r requirements.txt`

1. Install one or both simulators from the [environments section](#environments).


## Start training manually

```bash
python3 experiment/train.py env=FetchReach-v1 algorithm=hac algorithm.time_scales=[5,-1]
python experiment/train.py env=FetchReach-v1 algorithm=hac algorithm.layer_classes=['sacvg','ddpg'] algorithm.set_fut_ret_zero_if_done
```

[comment]: <> (```bash)

[comment]: <> (python experiment/train.py env=FetchReach-v1 algorithm=her2 algorithm.layer_classes=['sac'])

[comment]: <> (# also works with ddpg,td3,...)

[comment]: <> (```)

## Configure the training parameters
We use [hydra](https://hydra.cc/docs/next/intro) to manage the command-line parameters and configuration options.
The command line parameters are set in the `conf/main.yaml` file. 
It specifies that the default parameters are retrieved from `conf/exp_params/default.yaml`.
All algorithm-specific parameters are set in the `conf/algorithm/<alg_name>.yaml` file.
Parameters can be removed `~`, added `+` or overridden `++`.

## Load a stored policy
By default, the script stores the latest policy, the best policy (the best is the one with the highest value in `early_stop_data_column`), and it stores policies regularly in an interval of `save_model_freq` steps. To restore a saved policy, use the `restore_policy` commandline parameter. For example, say the best model is stored under the following directory:
`/storage/gdrive/Coding/ideas_deep_rl2/data/ac47785/FetchReach-v1/goaselstr=rndend&learat=0.0018,0.003&nsamgoa=6&subtesper=0.1&timsca=-1,7&100/best_model`
Then you can restore that policy py starting the script with the ``
`python experiment/train.py restore_policy='/storage/gdrive/Coding/ideas_deep_rl2/data/ac47785/FetchReach-v1/goaselstr=rndend&learat=0.0018,0.003&nsamgoa=6&subtesper=0.1&timsca=-1,7&100/best_model'`
It is important that you  **put the path to the store policy in single quotes**, otherwise the parser will fail because of the `=` symbols in the path! Double quotes won't work!

## File structure
* The main script from which all algorithms are started is `train.py`.
* The root directory contains shell scripts for automated testing and data generation.
* The folder `experiment` contains `train.py` and `plot.py`, which can plot data generated during the training.

* The folder `ideas_baselines` contains the new HAC implementation, an implementation of HER, and SACVG (a version of SAC with variable gamma).
  Other new algorithms should be added here, too. For details on the specific algorithms, see below.
* The folder `ideas_envs` should contain new environments (but we may also choose to put environments in a completely different repository).
* The folder `conf/algorithm` contains configurations for each algorithm, both stable-baselines3 algorithms and the algorithms here.
  It determines the kwargs passed on to the model (HAC, SAC, TD3, etc).
  These are also overridable as command-line options, e.g. `algorithm.verbose=False`.
* The folder `util` contains some misc utilities.
* The folder `hydra_plugins` contains some customized plugins for our hyperparameter management system.

## Algorithms

### HAC
TBD

## Environments
Currently, all goal-conditioned gym environments are supported. A list of all tested environments can be found in `conf/main.yaml`.
You can use MuJoCo, CoppeliaSim or both. The following sections show you how to install them.

### Install MuJoCo
1. Download [MuJoCo](mujoco.org) and obtain a license
   (as student you can obtain a free one-year student license).
   Copy the *mjpro200_linux* folder from the downloaded archive
   as well as *mjkey.txt* that you will obtain from the registration
   to folders of your choice.

1. Set the environment variables in `set_paths.sh` according to the
   locations where you saved the *mjpro200_linux* folder and the *mjkey.txt*.
   Run `source ./set_paths.sh`
   If you are using an IDE, set the variables there as well.

1. `pip install mujoco-py`
### Install CoppeliaSim and RL Bench
If you'd like to use environments simulated with CoppeliaSim,
[download CoppeliaSim Edu 4.1.0](https://www.coppeliarobotics.com/previousVersions) (4.2.0 causes problems with some environments)
and set the following paths accordingly.
```
COPPELIASIM_ROOT=EDIT/ME/PATH/TO/COPPELIASIM/INSTALL/DIR
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:EDIT/ME/PATH/TO/COPPELIASIM/INSTALL/DIR
QT_QPA_PLATFORM_PLUGIN_PATH=EDIT/ME/PATH/TO/COPPELIASIM/INSTALL/DIR
```
Then `pip install git+https://github.com/stepjam/PyRep.git`. You should now be able to use
CoppeliaSim environments.

If you'd also like to use the [RL Bench](https://github.com/stepjam/RLBench) environments,
`pip install git+https://github.com/stepjam/RLBench.git pyquaternion natsort`.
An example for an RL Bench environment is *reach_target-state-v0*.
### Adding a new environment
To add a new environment, we suggest to proceed as follows.

1. As a basis, use an environment that is as close as possible to the new environment you want to develop. A good start for a gripper-based environment is the gym version of FetchReach. Run your favorite algorithm with this base environment and make sure it works as intended. It is also a good idea to take notes about the learning performance (how many training steps required to achieve success).
1. Make a copy of that environment and add it to the `ideas_envs` folder in the repository. Therefore, create a new subfolder for your new environment under `ideas_envs`, say `ideas_envs/new_env`. Then, say you want to start with the `FetchReach-v1` environment as a basis, and assuming that `venv` is the folder where you have your virtual python 3.6 environment, copy `venv/lib/python3.6/site-packages/gym/envs/robotics/fetch_env.py` to that subfolder. You may want to save it under a different filename, say `new_env.py`. Finally, if you use an OpenAI gym env like FetchReach, you have to create an entry point class for your environment, as found in `venv/lib/python3.6/site-packages/gym/envs/robotics/fetch/reach.py:FetchReachEnv`. Using an entry point class is the preferred way of implementing different variations for each environment. For example, the fetch environment has different entry points for `FetchReach` and `FetchPush`. You find these in `reach.py` and `push.py` respectively. For the following we assume that your entry point class is `NewClassEnv` and your environment name is `NewEnv-v0` (note that the `-vN` suffix is mandatory for each environment name, where `N` is the version number).
1. Then you need to register your entry point class as a new environment. You can do this by adding it to the `ideas_envs/register_envs.py` file. You find examples for how exactly to do this in the file.
1. Now try whether your copy of the base environment is running as intended, in the same way as the original one. Therefore, set the `env` parameter in `conf/main.yaml` to `NewEnv-v0`, or add the commandline parameter `env=NewEnv-v0`.
1. That's it. Now you can take your copy as a basis for your new environment and modify it as you want to develop your new environment.

## Limitations
> :warning: Currently, only off-policy algorithms are supported: DQN, DDPG, TD3, SAC, HER and HAC. PPO is not yet supported but it should not be too hard to enable it.

## Hyperparameter optimization and management
The framework has a sophisticated hyperparameter management and optimization pipeline.
To start the hyperparameter optimization start `experiment/train.py --multirun`. The `--multirun` flag starts the hyperparameter optimization mode.

> :warning: (A problem with the comet.ml integration is that if a script raises an Error and stops, all parallel processes will be blocked and the error will not be output. Therefore, if you spot that all processes of the hyperopt are idle you should re-start the process without comet.ml &#40;just remove the import in `train.py`&#41; and try to reproduce and find the error by observing the console output.  )

The hyperparameter management and optimization builds on the following four tools:

### Optuna
Optuna is a framework to perform the hyperparameter optimization algorithm (e.g. TPE).
Optuna integrates flawlessly with hydra.
The default optuna launcher for hyperopting is the joblib launcher which spawns several loky processes.
A disadvantage of this approach is that the logging to stdout is disabled.
However, you will be able to read all console outputs produced by `logger.info()` in the log folder.
The hyperparameter search is implemented via a customized sweeper plugin located in the `hydra_plugins` subfolder.
By default, the sweeper uses the TPE optimizer.
The sweeper automatically learns when to early stop trials by remembering past trials.
It sets the max. number of epochs for a new trial to 1.5 times the number of epochs of the so-far fastest trial (when terminated by early stopping).
The sweeper stops after the set number of trials or the specified duration, as specified in the config file.
For convenience, the sweeper also creates a file `delete_me_to_stop_hyperopt`, which you just need to delete to soft-stop the hyperopting after the current batch of jobs.

#### Hyperparameter tuning
We configure the hyperparameter tuning with hydra. The configurations are stored in `conf/performance/<env_name>/<algo_name>-opti.yaml`. For example, to optimize `sac` with `her` for the RLBench reacher environment, run
```bash
python experiment/train.py +performance=RLB_reach_target/sac_her-opti.yaml --multirun
```
We use theses files to specify the algorithm (e.g. `override /algorithm: sac`) and its parameters, the environment (e.g. `env: reach_target-state-v0`), and the search space for the optimization. If you copy and change a config to optimize for a different environment or algorithm, it is also important to change `hydra:sweeper:study_name` and `hydra:sweeper:storage` accordingly.

#### Testing functionality (smoke test)
Simply execute
```bash
python experiment/train.py algorithm=hac,her env=FetchReach-v1,AntReacher-v1 ++n_epochs=2 +defaults=smoke_test --multirun
```
, to run experiments for hac and her for two epochs (here we use `++` to override the amount of epochs).
With `+defaults=smoke_test` we are loading the sweeper parameters from `confg/smoke_test.yaml`.
Crashed experiments can be found in mlflow, having a red cross symbol.

#### Performance testing
Run a performance test for an environment-algorithm combination. The conditions for a performance test are stored in
`conf/performance/<env_name>/<algo_name>-test.yaml`.
You can for example run:
```bash
python experiment/train.py +performance=FetchReach/sac_her-test.yaml --multirun
```
to test the performance of the current hyperparameters.
The joblib launcher allows to run `n_jobs` in parallel.

> :warning: **You cannot** run multiple performance tests by simply providing multiple configs separated by commas, for example:
`python experiment/train.py +performance=FetchReach/her-test,AntMaze/hac-2layer-test --multirun` does not work.
In that case, just call `experiment/train.py` twice with the different performance test configs.

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
