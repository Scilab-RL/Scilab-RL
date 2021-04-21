import os
import importlib
import os
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import gym
from util.util import get_subdir_by_params,get_git_label,set_global_seeds,log_dict,get_last_epoch_from_logdir
# import logger
import os
import time
import json
import sys
from queue import deque
import numpy as np
import ideas_envs.register_envs
import ideas_envs.wrappers.utils
from stable_baselines3.common import logger
from util.custom_logger import MatplotlibOutputFormat, FixedHumanOutputFormat
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from util.compat_wrappers import make_robustGoalConditionedHierarchicalEnv, make_robustGoalConditionedModel
from stable_baselines3.common.bit_flipping_env import BitFlippingEnv
from stable_baselines3 import DDPG, DQN, SAC, TD3
from util.custom_eval_callback import CustomEvalCallback
from ideas_baselines.mbchac.hierarchical_eval_callback import HierarchicalEvalCallback
# from util.custom_train_callback import CustomTrainCallback
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from ideas_baselines import HER2, MBCHAC, SACVG

ALL_PATH_CONFIG_PARAMS = ['info', 'algorithm']

def check_env_alg_compatibility(model, env):
    return isinstance(model.action_space, type(env.action_space)) \
        and isinstance(model.observation_space, type(env.observation_space['observation']))

def train(model, train_env, eval_env, cfg):
    algorithm_cfg = cfg.algorithm
    total_steps = cfg.eval_after_n_steps * cfg.n_epochs
    checkpoint_callback = CheckpointCallback(save_freq=cfg.save_model_freq, save_path=logger.get_dir())
    if hasattr(model, 'time_scales'):
        eval_callback = HierarchicalEvalCallback(eval_env,
                                                 log_path=logger.get_dir(),
                                                 eval_freq=cfg.eval_after_n_steps,
                                                 n_eval_episodes=cfg.n_test_rollouts,
                                                 early_stop_last_n=cfg.early_stop_last_n,
                                                 early_stop_data_column=cfg.early_stop_data_column,
                                                 early_stop_threshold=cfg.early_stop_threshold,
                                                 top_level_model=model)
    else:
        eval_callback = CustomEvalCallback(eval_env,
                                           log_path=logger.get_dir(),
                                           eval_freq=cfg.eval_after_n_steps,
                                           n_eval_episodes=cfg.n_test_rollouts,
                                           early_stop_last_n=cfg.early_stop_last_n,
                                           early_stop_data_column=cfg.early_stop_data_column,
                                           early_stop_threshold=cfg.early_stop_threshold)

    # Create the callback list
    callback = CallbackList([checkpoint_callback, eval_callback])
    model.learn(total_timesteps=total_steps, callback=callback, log_interval=None)

    train_env.close()
    eval_env.close()
    logger.info("Training finished!")

def launch(model_class, model_classes, cfg):
    set_global_seeds(cfg.seed)
    algo_name = cfg['algorithm'].name
    try:
        ModelClass = getattr(importlib.import_module('stable_baselines3.' + algo_name), algo_name.upper())
    except:
        ModelClass = getattr(importlib.import_module('ideas_baselines.' + algo_name), algo_name.upper())
    train_env = gym.make(cfg.env)
    eval_env = gym.make(cfg.env)
    if cfg.restore_policy is not None:
        model = ModelClass.load(cfg.restore_policy, model_class, sub_model_classes=model_classes, **cfg['algorithm'])
    else:
        model = ModelClass('MlpPolicy', train_env, model_class, sub_model_classes=model_classes, **cfg['algorithm'])
    env_alg_compatible = check_env_alg_compatibility(model, train_env)
    if not env_alg_compatible:
        logger.info("Environment {} and algorithm {} are not compatible.".format(train_env, model))
        sys.exit()
    logger.info("Launching training")
    train(model, train_env, eval_env, cfg)


OmegaConf.register_resolver("git_label", lambda: get_git_label())
@hydra.main(config_name="main", config_path="conf")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    run_dir = os.getcwd()
    root_dir = hydra.utils.get_original_cwd()
    algorithm_cfg = cfg.algorithm
    config = importlib.import_module('interface.' + cfg['algorithm'].name + '.config')
    model_classes = []
    # Get default options for model classes
    if 'model_class' in algorithm_cfg.keys():
        model_class = getattr(importlib.import_module('stable_baselines3.' + algorithm_cfg['model_class']),
                                             algorithm_cfg['model_class'].upper())

    if 'model_classes' in algorithm_cfg.keys():
        class_list = []
        for class_name in algorithm_cfg.model_classes:
            try:
                class_list.append(getattr(importlib.import_module('stable_baselines3.' + class_name), class_name.upper()))
            except:
                class_list.append(getattr(importlib.import_module('ideas_baselines.' + class_name), class_name.upper()))
        if class_list:
            model_class = class_list[0]
            model_classes = class_list[1:]

    logger.info("Starting process id: {}".format(os.getpid()))
    ctr = cfg['try_start_idx']
    max_ctr = cfg['max_try_idx']
    path_params_names = config.PATH_CONFIG_PARAMS + ALL_PATH_CONFIG_PARAMS
    #  path_params = {param:algorithm_cfg[param] for param in path_params_names}

    if cfg['seed'] == 0:
        cfg['seed'] = int(time.time())
    log_dict(cfg, logger)

    logger.configure(folder=run_dir, format_strings=['csv', 'tensorboard'])
    plot_cols = cfg['plot_eval_cols']
    logger.Logger.CURRENT.output_formats.append(MatplotlibOutputFormat(run_dir, cfg['plot_at_most_every_secs'], cols_to_plot=plot_cols))
    logger.Logger.CURRENT.output_formats.append(FixedHumanOutputFormat(sys.stdout))
    logger.Logger.CURRENT.output_formats.append(FixedHumanOutputFormat(os.path.join(run_dir, f"log.txt")))

    logdir = logger.get_dir()
    logger.info("Data dir: {} ".format(logdir))

    # Prepare xmls for subgoal visualizations
    ideas_envs.wrappers.utils.goal_viz_for_gym_robotics()
    OmegaConf.save(config=cfg, f='params.yaml')
    launch(model_class, model_classes, cfg)


if __name__ == '__main__':
    main()