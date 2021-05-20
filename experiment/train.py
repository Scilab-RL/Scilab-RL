# import comet_ml # Direct comet.ml upload not supported in multiprocessing mode. See README.md and issue https://git.informatik.uni-hamburg.de/eppe/ideas_deep_rl2/-/issues/26
import joblib.externals.loky.backend.context as jl_ctx
jl_ctx._DEFAULT_START_METHOD = 'loky_init_main' # This is required for multiprocessing with joblib because by default, the loky multiprocessing backend of joblib re-imports all modules when calling main(). Since comet_ml monkey-patches several libraries, the changes to these libraries get lost. When setting the start_method to loky_init_main the modules are not re-imported and the monkey-path-changes don't get lost.
import mlflow

import matplotlib
matplotlib.use('Agg')
import os
import sys
sys.path.append(os.getcwd())
import importlib
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import csv
import numpy as np

# importing gym also imports cv2, which changes the QT_QPA_PLATFORM_PLUGIN_PATH.
# So we have to save it and set it again after importing gym.
QT_QPA_PLATFORM_PLUGIN_PATH = os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']
import gym
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = QT_QPA_PLATFORM_PLUGIN_PATH
from util.util import get_subdir_by_params,get_git_label,set_global_seeds,log_dict,get_last_epoch_from_logdir
import time
import ideas_envs.register_envs
import ideas_envs.wrappers.utils
from ideas_envs.wrappers.rl_bench_wrapper import RLBenchWrapper
from stable_baselines3.common import logger
from util.custom_logger import MatplotlibCSVOutputFormat, FixedHumanOutputFormat, MLFlowOutputFormat
from util.custom_eval_callback import CustomEvalCallback
from ideas_baselines.hac.hierarchical_eval_callback import HierarchicalEvalCallback
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from util.mlflow_util import setup_mlflow, get_hyperopt_score

def check_env_alg_compatibility(model, env):
    return isinstance(model.action_space, type(env.action_space)) \
        and isinstance(model.observation_space, type(env.observation_space['observation']))

def train(baseline, train_env, eval_env, cfg):
    total_steps = cfg.eval_after_n_steps * cfg.n_epochs
    callback = []
    if cfg.save_model_freq > 0:
        checkpoint_callback = CheckpointCallback(save_freq=cfg.save_model_freq, save_path=logger.get_dir())
        callback.append(checkpoint_callback)
    if hasattr(baseline, 'time_scales'):
        eval_callback = HierarchicalEvalCallback(eval_env,
                                                 log_path=logger.get_dir(),
                                                 eval_freq=cfg.eval_after_n_steps,
                                                 n_eval_episodes=cfg.n_test_rollouts,
                                                 early_stop_last_n=cfg.early_stop_last_n,
                                                 early_stop_data_column=cfg.early_stop_data_column,
                                                 early_stop_threshold=cfg.early_stop_threshold,
                                                 top_level_layer=baseline)
    else:
        eval_callback = CustomEvalCallback(eval_env,
                                           log_path=logger.get_dir(),
                                           eval_freq=cfg.eval_after_n_steps,
                                           n_eval_episodes=cfg.n_test_rollouts,
                                           early_stop_last_n=cfg.early_stop_last_n,
                                           early_stop_data_column=cfg.early_stop_data_column,
                                           early_stop_threshold=cfg.early_stop_threshold)

    # Create the callback list
    callback.append(eval_callback)
    baseline.learn(total_timesteps=total_steps, callback=callback, log_interval=None)
    train_env.close()
    eval_env.close()
    logger.info("Training finished!")

def launch(cfg, kwargs):
    set_global_seeds(cfg.seed)
    algo_name = cfg['algorithm'].name

    # remove name as we pass all arguments to the model constructor
    if 'name' in cfg.algorithm.keys():
        with open_dict(cfg):
            del cfg['algorithm']['name']

    try:
        BaselineClass = getattr(importlib.import_module('stable_baselines3.' + algo_name), algo_name.upper())
    except:
        BaselineClass = getattr(importlib.import_module('ideas_baselines.' + algo_name), algo_name.upper())
    if cfg.env.endswith('-state-v0') or cfg.env.endswith('-vision-v0'):  # if the environment is a rl_bench env
        render_mode = None
        if cfg.algorithm.render_train == "display":
            render_mode = "human"
        train_env = eval_env = RLBenchWrapper(gym.make(cfg.env, render_mode=render_mode))
    else:
        train_env = gym.make(cfg.env)
        eval_env = gym.make(cfg.env)
    if cfg.restore_policy is not None:
        baseline = BaselineClass.load(cfg.restore_policy, **cfg.algorithm, env=train_env, **kwargs)
    else:
        baseline = BaselineClass('MlpPolicy', train_env, **cfg.algorithm, **kwargs)
    env_alg_compatible = check_env_alg_compatibility(baseline, train_env)
    if not env_alg_compatible:
        logger.info("Environment {} and algorithm {} are not compatible.".format(train_env, baseline))
        sys.exit()
    logger.info("Launching training")
    train(baseline, train_env, eval_env, cfg)

# make git_label available in hydra
OmegaConf.register_new_resolver("git_label", lambda: get_git_label())

@hydra.main(config_name="main", config_path="../conf")
def main(cfg: DictConfig) -> (float, int):
    ctr = cfg['try_start_idx']
    max_ctr = cfg['max_try_idx']
    run_dir = 'tmp_logdir'
    original_dir = os.getcwd()

    if cfg.restore_policy is not None:
        run_dir = os.path.split(cfg.restore_policy)[:-1][0]
        run_dir = run_dir + "_restored"
        trial_no = None
        os.rename(original_dir, run_dir)
    else:
        path_dir_params = {key: cfg.algorithm[key] for key in cfg.algorithm.exp_path_params}
        dir_created = False
        while not dir_created:
            subdir_exists = True
            while subdir_exists:
                param_dir = get_subdir_by_params(path_dir_params, ctr)
                run_dir = os.path.join(os.path.split(original_dir)[0], param_dir)
                subdir_exists = os.path.exists(run_dir)
                ctr += 1
            try:
                os.rename(original_dir, run_dir)
                dir_created = True
            except Exception as e:
                print(f"Creating logdir did not work, trying again: {e}")
                time.sleep(1)
        trial_no = ctr - 1
    setup_mlflow(cfg, run_dir)
    mlflow_run = mlflow.active_run()


    # Output will only be logged appropriately after configuring the logger in the following lines:
    logger.configure(folder=run_dir, format_strings=[])
    logger.Logger.CURRENT.output_formats.append(FixedHumanOutputFormat(sys.stdout))
    logger.Logger.CURRENT.output_formats.append(FixedHumanOutputFormat(os.path.join(run_dir, "train.log")))
    logger.Logger.CURRENT.output_formats.append(MLFlowOutputFormat())
    logger.info(f"Starting training with the following configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info(f"Log directory: {run_dir}")
    # End configure logger

    #
    if trial_no is not None:
        if trial_no > max_ctr:
            logger.info("Already collected enough data for this parameterization.")
            sys.exit()
        logger.info("Trying this config for {}th time. ".format(trial_no))

    if 'exp_path_params' in cfg.algorithm.keys():
        with open_dict(cfg):
            del cfg['algorithm']['exp_path_params']
    kwargs = {}
    # TODO: function for folder name
    #  OmegaConf.register_resolver("git_label", lambda: get_git_label())
    # Get default options for model classes
    class_list = []
    for class_name in cfg.layer_classes:
        if class_name in dir(importlib.import_module('ideas_baselines')):
            class_list.append(getattr(importlib.import_module('ideas_baselines.' + class_name), class_name.upper()))
        elif class_name in dir(importlib.import_module('stable_baselines3')):
            class_list.append(getattr(importlib.import_module('stable_baselines3.' + class_name), class_name.upper()))
        else:
            raise ValueError(f"class name {class_name} not found")

    if class_list:
        kwargs['layer_class'] = class_list[0]
        if len(class_list) > 1:
            kwargs['sub_layer_classes'] = class_list[1:]

    logger.info("Starting process id: {}".format(os.getpid()))

    if cfg['seed'] == 0:
        cfg['seed'] = int(time.time())


    logdir = logger.get_dir()
    logger.info("Data dir: {} ".format(logdir))

    # Prepare xmls for subgoal visualizations
    ideas_envs.wrappers.utils.goal_viz_for_gym_robotics()
    OmegaConf.save(config=cfg, f='params.yaml')
    launch(cfg, kwargs)
    logger.info("Finishing main training function.")
    logger.info(f"MLflow run: {mlflow_run}.")
    hyperopt_score, n_epochs = get_hyperopt_score(cfg, mlflow_run)
    mlflow.log_metric("hyperopt_score", hyperopt_score)
    logger.info(f"Hyperopt score: {hyperopt_score}, epochs: {n_epochs}.")
    try:
        with open(os.path.join(run_dir, 'train.log'), 'r') as logfile:
            log_text = logfile.read()
            mlflow.log_text(log_text, 'train.log')
    except Exception as e:
        logger.info('Could not open logfile and log it in mlflow.')
    mlflow.end_run()

    return hyperopt_score, n_epochs



if __name__ == '__main__':
    main()
