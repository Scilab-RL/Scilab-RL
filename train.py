import os
import importlib
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import gym
from util.util import get_subdir_by_params,get_git_label,set_global_seeds,log_dict,get_last_epoch_from_logdir
# import logger
import time
import sys
import ideas_envs.register_envs
import ideas_envs.wrappers.utils
from stable_baselines3.common import logger
from util.custom_logger import MatplotlibOutputFormat, FixedHumanOutputFormat
from stable_baselines3.common.env_checker import check_env
from util.compat_wrappers import make_robustGoalConditionedHierarchicalEnv, make_robustGoalConditionedModel
from util.custom_eval_callback import CustomEvalCallback
from ideas_baselines.mbchac.hierarchical_eval_callback import HierarchicalEvalCallback
# from util.custom_train_callback import CustomTrainCallback
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv


def check_env_alg_compatibility(model, env):
    return isinstance(model.action_space, type(env.action_space)) \
        and isinstance(model.observation_space, type(env.observation_space['observation']))

def train(baseline, train_env, eval_env, cfg):
    total_steps = cfg.eval_after_n_steps * cfg.n_epochs
    checkpoint_callback = CheckpointCallback(save_freq=cfg.save_model_freq, save_path=logger.get_dir())
    if hasattr(baseline, 'time_scales'):
        eval_callback = HierarchicalEvalCallback(eval_env,
                                                 log_path=logger.get_dir(),
                                                 eval_freq=cfg.eval_after_n_steps,
                                                 n_eval_episodes=cfg.n_test_rollouts,
                                                 early_stop_last_n=cfg.early_stop_last_n,
                                                 early_stop_data_column=cfg.early_stop_data_column,
                                                 early_stop_threshold=cfg.early_stop_threshold,
                                                 top_level_model=baseline)
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
OmegaConf.register_resolver("git_label", lambda: get_git_label())

@hydra.main(config_name="main", config_path="conf")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    original_dir = os.getcwd()
    logger.info('Hydra dir', original_dir)
    path_dir_params = {key: cfg.algorithm[key] for key in cfg.algorithm.exp_path_params}
    subdir_exists = True

    ctr = cfg['try_start_idx']
    max_ctr = cfg['max_try_idx']

    while subdir_exists:
        param_dir = get_subdir_by_params(path_dir_params, ctr)
        run_dir = os.path.join(os.path.split(original_dir)[0], param_dir)
        subdir_exists = os.path.exists(run_dir)
        ctr += 1
    trial_no = ctr - 1
    logger.info('Renamed hydra dir to', run_dir)
    os.rename(original_dir, run_dir)
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

    logger.configure(folder=run_dir, format_strings=['csv', 'tensorboard'])
    plot_cols = cfg['plot_eval_cols']
    logger.Logger.CURRENT.output_formats.append(MatplotlibOutputFormat(run_dir, cfg['plot_at_most_every_secs'], cols_to_plot=plot_cols))
    logger.Logger.CURRENT.output_formats.append(FixedHumanOutputFormat(sys.stdout))
    logger.Logger.CURRENT.output_formats.append(FixedHumanOutputFormat(os.path.join(run_dir, "train.log")))

    logdir = logger.get_dir()
    logger.info("Data dir: {} ".format(logdir))

    # Prepare xmls for subgoal visualizations
    ideas_envs.wrappers.utils.goal_viz_for_gym_robotics()
    OmegaConf.save(config=cfg, f='params.yaml')
    launch(cfg, kwargs)


if __name__ == '__main__':
    main()