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
    # is_compatible = isinstance(model.action_space, type(env.action_space))
    # return is_compatible
    obs = env.reset()
    action, _states = model.predict(obs, deterministic=True)
    try:
        env.step(action)
        is_compatible = True
    except Exception as e:
        logger.info("Environment and algorithm not compatible, probably because of different action spaces. Exception: {}".format(e))
        logger.info(e)
        is_compatible = False
    return is_compatible

def train(model, train_env, eval_env, n_epochs, starting_epoch, **kwargs):
    # actions_per_episode = np.product([int(steps) for steps in kwargs['action_steps'].split(',')])
    # train_actions_per_epoch = steps_per_epoch * kwargs['n_train_rollouts']
    epochs_remaining = n_epochs - starting_epoch
    total_steps = kwargs['eval_after_n_steps'] * epochs_remaining

    checkpoint_callback = CheckpointCallback(save_freq=kwargs['save_model_freq'], save_path=logger.get_dir())
    if hasattr(model, 'time_scales'):
        eval_callback = HierarchicalEvalCallback(eval_env,
                                                 log_path=logger.get_dir(),
                                                 eval_freq=kwargs['eval_after_n_steps'],
                                                 n_eval_episodes=kwargs['n_test_rollouts'],
                                                 early_stop_last_n=kwargs['early_stop_last_n'],
                                                 early_stop_data_column=kwargs['early_stop_data_column'],
                                                 early_stop_threshold=kwargs['early_stop_threshold'],
                                                 top_level_model=model)
    else:
        eval_callback = CustomEvalCallback(eval_env,
                                           log_path=logger.get_dir(),
                                           eval_freq=kwargs['eval_after_n_steps'],
                                           n_eval_episodes=kwargs['n_test_rollouts'],
                                           early_stop_last_n=kwargs['early_stop_last_n'],
                                           early_stop_data_column=kwargs['early_stop_data_column'],
                                           early_stop_threshold=kwargs['early_stop_threshold'])

    # Create the callback list
    callback = CallbackList([checkpoint_callback, eval_callback])
    model.learn(total_timesteps=total_steps, callback=callback, log_interval=None)

    train_env.close()
    eval_env.close()
    logger.info("Training finished!")

def launch(starting_epoch, model_class, model_classes, cfg):
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
    try:
        train(model, train_env, eval_env, cfg.n_epochs, starting_epoch=starting_epoch, **cfg)
    except e:
        print(e)


@hydra.main(config_name="main", config_path="conf")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    run_dir = os.getcwd()
    root_dir = hydra.utils.get_original_cwd()
    algorithm_cfg = cfg.algorithm
    config = importlib.import_module('interface.' + cfg['algorithm'].name + '.config')
    #  all_cmd_kvs = {ctx.args[i][2:]: ctx.args[i+1] for i in range(0, len(ctx.args), 2)}
    #  policy_args = ctx.forward(main_linker.get_algorithm_click).copy()
    #  changed_policy_args = {k: type(policy_args[k])(v) for k,v in all_cmd_kvs.items() if k in policy_args.keys()}
    #  policy_args.update(changed_policy_args)

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

        # In policy args, exchange the string representing the model class with the actual class object.
        #  with open_dict(cfg):
        #      cfg.algorithm.model_class = class_list[0].__name__
        #      if len(class_list) > 1:
        #          cfg.algorithm.sub_model_classes = [cls.__name__ for cls in class_list[1:]]

    logger.info("Starting process id: {}".format(os.getpid()))
    ctr = cfg['try_start_idx']
    max_ctr = cfg['max_try_idx']
    path_params_names = config.PATH_CONFIG_PARAMS + ALL_PATH_CONFIG_PARAMS
    #  path_params = {param:algorithm_cfg[param] for param in path_params_names}
    starting_epoch = 0
    #  if cfg['restore_policy'] is not None:
    #      params_file = '/'.join(cfg['restore_policy'].split("/")[:-1]) + '/params.json'
    #      try:
    #          with open(params_file, 'r') as f:
    #              loaded_params = json.load(f)
    #              cfg['seed'] = loaded_params['seed']
    #              cfg['logdir'] = loaded_params['logdir']
    #          starting_epoch = int(get_last_epoch_from_logdir(cfg['logdir'])) + 1
    #          for k,v in loaded_params.items():
    #              if k in cfg.keys() and k != 'pid':
    #                  assert cfg[k] == v, "Error loaded parameter {} = {} does not match configuration: {} = {}".format(k,v,k,cfg[k])
    #      except Exception:
    #          logger.warn("Warning, could not determine random seed of loaded model because params.csv is missing.")
    #
    git_label = get_git_label()
    #  if git_label != '':
    #      data_basedir = os.path.join(cfg['base_logdir'], git_label, cfg['env'])
    #  else:
    #      data_basedir = os.path.join(cfg['base_logdir'], 'experiments', cfg['env'])
    #  logger.info("Data base dir: {} ".format(data_basedir))
    #  if 'logdir' not in cfg.keys():
    #      logdir = 'data'
    #      subdir_exists = True
    #      while subdir_exists:
    #          param_subdir = get_subdir_by_params(path_params, ctr)
    #          logdir = os.path.join(data_basedir, param_subdir)
    #          subdir_exists = os.path.exists(logdir)
    #          ctr += 1
    #      trial_no = ctr - 1
    #      cfg['logdir'] = logdir
    #      if trial_no > max_ctr:
    #          logger.info("Already collected enough data for this parameterization.")
    #          sys.exit()
    #      logger.info("Trying this config for {}th time. ".format(trial_no))

    if cfg['seed'] == 0:
        cfg['seed'] = int(time.time())
    log_dict(cfg, logger)

    logger.configure(folder=run_dir,
                     format_strings=['csv', 'tensorboard'])
    plot_cols = cfg['plot_eval_cols']
    logger.Logger.CURRENT.output_formats.append(MatplotlibOutputFormat(run_dir, cfg['plot_at_most_every_secs'], cols_to_plot=plot_cols))
    logger.Logger.CURRENT.output_formats.append(FixedHumanOutputFormat(sys.stdout))
    logger.Logger.CURRENT.output_formats.append(FixedHumanOutputFormat(os.path.join(run_dir, f"log.txt")))

    logdir = logger.get_dir()

    logger.info("Data dir: {} ".format(logdir))

    # Prepare xmls for subgoal visualizations
    ideas_envs.wrappers.utils.goal_viz_for_gym_robotics()

    OmegaConf.save(config=cfg, f='params.yaml')
    #  with open(os.path.join(logdir, 'params.'params.yaml'json'), 'w') as f:
    #      json.dump(cfg, f)
    launch(starting_epoch, model_class, model_classes, cfg)


if __name__ == '__main__':
    main()