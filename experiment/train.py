import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import gym
import click_options as main_linker
from util.util import get_subdir_by_params,get_git_label,set_global_seeds,log_dict,get_last_epoch_from_logdir
import click
# import logger
import os
import time
import json
import sys
from queue import deque
import numpy as np
import importlib
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
from ideas_baselines.her2 import HER2

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

def get_model_class_args(model_class_str, all_cmd_kvs):
    model_click = importlib.import_module('interface.' + model_class_str + '.click_options')
    model_class_options = model_click.get_click_option.params
    model_class_args = {o.name: o.default for o in model_class_options}
    changed_mc_args = {k: type(model_class_args[k])(v) for k, v in all_cmd_kvs.items() if k in model_class_args.keys()}
    model_class_args.update(changed_mc_args)
    return model_class_args

def train(model, train_env, eval_env, n_epochs, starting_epoch, **kwargs):
    # actions_per_episode = np.product([int(steps) for steps in kwargs['action_steps'].split(',')])
    # train_actions_per_epoch = steps_per_epoch * kwargs['n_train_rollouts']
    epochs_remaining = n_epochs - starting_epoch
    total_steps = kwargs['eval_after_n_steps'] * epochs_remaining

    checkpoint_callback = CheckpointCallback(save_freq=kwargs['eval_after_n_steps'], save_path=logger.get_dir())
    if hasattr(model, 'time_scales'):
        eval_callback = HierarchicalEvalCallback(eval_env,
                                                 log_path=logger.get_dir(),
                                                 eval_freq=kwargs['eval_after_n_steps'],
                                                 n_eval_episodes=kwargs['n_test_rollouts'],
                                                 render=kwargs['render_test'],
                                                 early_stop_last_n=kwargs['early_stop_last_n'],
                                                 early_stop_data_column=kwargs['early_stop_data_column'],
                                                 early_stop_threshold=kwargs['early_stop_threshold'],
                                                 top_level_model=model
                                                 )
    else:
        eval_callback = CustomEvalCallback(eval_env,
                                           log_path=logger.get_dir(),
                                           eval_freq=kwargs['eval_after_n_steps'],
                                           n_eval_episodes=kwargs['n_test_rollouts'],
                                           render=kwargs['render_test'],
                                           early_stop_last_n=kwargs['early_stop_last_n'],
                                           early_stop_data_column=kwargs['early_stop_data_column'],
                                           early_stop_threshold=kwargs['early_stop_threshold'])

    # Create the callback list
    callback = CallbackList([checkpoint_callback, eval_callback])
    model.learn(total_timesteps=total_steps, callback=callback, log_interval=None, )

    train_env.close()
    eval_env.close()
    logger.info("Training finished!")

def launch(ctx, starting_epoch, policy_args, env, algorithm,  n_epochs, seed, restore_policy, logdir, **kwargs):
    set_global_seeds(seed)
    try:
        ModelClass = getattr(importlib.import_module('stable_baselines3.' + algorithm), algorithm.upper())
    except:
        ModelClass = getattr(importlib.import_module('ideas_baselines.' + algorithm), algorithm.upper())
    train_env = gym.make(env)
    eval_env = gym.make(env)
    if restore_policy is not None:
        model = ModelClass.load(restore_policy, env=train_env)
    else:
        model = ModelClass('MlpPolicy', train_env, **policy_args)
    env_alg_compatible = check_env_alg_compatibility(model, train_env)
    if not env_alg_compatible:
        sys.exit()
    logger.info("Launching training")
    train(model, train_env, eval_env, n_epochs, starting_epoch=starting_epoch, **kwargs)

@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@main_linker.click_main
@click.pass_context
def main(ctx, **kwargs):
    config = main_linker.import_creator(kwargs['algorithm'])
    all_cmd_kvs = {ctx.args[i][2:]: ctx.args[i+1] for i in range(0, len(ctx.args), 2)}
    policy_args = ctx.forward(main_linker.get_algorithm_click).copy()
    changed_policy_args = {k: type(policy_args[k])(v) for k,v in all_cmd_kvs.items() if k in policy_args.keys()}
    policy_args.update(changed_policy_args)

    # Get default options for model classes
    if 'model_class' in policy_args.keys():
        model_class_args = get_model_class_args(policy_args['model_class'], all_cmd_kvs)
        for k, v in model_class_args.items():
            if k in policy_args.keys():
                logger.warn(
                    "Warning, model class and algorithm have a common parameter {}. This will cause the model class to use the default parameter provided in its constructor.".format(
                        k))
            else:
                policy_args[k] = v
        kwargs.update(policy_args.copy())
        # In policy args, exchange the string representing the model class with the actual class object.
        policy_args['model_class'] = getattr(importlib.import_module('stable_baselines3.' + policy_args['model_class']),
                                             policy_args['model_class'].upper())

    if 'model_classes' in policy_args.keys():
        class_list = []
        for class_name in policy_args['model_classes'].split(','):
            model_class_args = get_model_class_args(class_name, all_cmd_kvs)
            for k, v in model_class_args.items():
                if k in policy_args.keys():
                    logger.warn(
                        "Warning, model class and algorithm have a common parameter {}. This will cause the model class to use the default parameter provided in its constructor.".format(
                            k))
                else:
                    policy_args[k] = v

            class_list.append(getattr(importlib.import_module('stable_baselines3.' + class_name),
                                      class_name.upper()))
        kwargs.update(policy_args.copy())
        # In policy args, exchange the string representing the model class with the actual class object.
        policy_args['model_class'] = class_list[0]
        if len(class_list) > 1:
            policy_args['sub_model_classes'] = class_list[1:]

    kwargs['pid'] = os.getpid()
    logger.info("Starting process id: {}".format(kwargs['pid']))
    ctr = kwargs['try_start_idx']
    max_ctr = kwargs['max_try_idx']
    path_params_names = config.PATH_CONFIG_PARAMS + ALL_PATH_CONFIG_PARAMS
    path_params = {param:kwargs[param] for param in path_params_names}
    starting_epoch = 0
    if kwargs['restore_policy'] is not None:
        params_file = '/'.join(kwargs['restore_policy'].split("/")[:-1]) + '/params.json'
        try:
            with open(params_file, 'r') as f:
                loaded_params = json.load(f)
                kwargs['seed'] = loaded_params['seed']
                kwargs['logdir'] = loaded_params['logdir']
            starting_epoch = int(get_last_epoch_from_logdir(kwargs['logdir'])) + 1
            for k,v in loaded_params.items():
                if k in kwargs.keys() and k != 'pid':
                    assert kwargs[k] == v, "Error loaded parameter {} = {} does not match configuration: {} = {}".format(k,v,k,kwargs[k])
        except Exception:
            logger.warn("Warning, could not determine random seed of loaded model because params.csv is missing.")

    git_label = get_git_label()
    if git_label != '':
        data_basedir = os.path.join(kwargs['base_logdir'], git_label, kwargs['env'])
    else:
        data_basedir = os.path.join(kwargs['base_logdir'], 'experiments', kwargs['env'])
    logger.info("Data base dir: {} ".format(data_basedir))
    if 'logdir' not in kwargs.keys():
        logdir = 'data'
        subdir_exists = True
        while subdir_exists:
            param_subdir = get_subdir_by_params(path_params, ctr)
            logdir = os.path.join(data_basedir, param_subdir)
            subdir_exists = os.path.exists(logdir)
            ctr += 1
        trial_no = ctr - 1
        kwargs['logdir'] = logdir
        if trial_no > max_ctr:
            logger.info("Already collected enough data for this parameterization.")
            sys.exit()
        logger.info("Trying this config for {}th time. ".format(trial_no))

    if kwargs['seed'] == 0:
        kwargs['seed'] = int(time.time())
    log_dict(kwargs, logger)

    logger.configure(folder=kwargs['logdir'],
                     format_strings=['csv', 'tensorboard'])
    plot_cols = kwargs['plot_eval_cols'].split(',')
    logger.Logger.CURRENT.output_formats.append(MatplotlibOutputFormat(kwargs['logdir'], kwargs['plot_at_most_every_secs'], cols_to_plot=plot_cols))
    logger.Logger.CURRENT.output_formats.append(FixedHumanOutputFormat(sys.stdout))
    logger.Logger.CURRENT.output_formats.append(FixedHumanOutputFormat(os.path.join(kwargs['logdir'], f"log.txt")))

    logdir = logger.get_dir()

    logger.info("Data dir: {} ".format(logdir))
    with open(os.path.join(logdir, 'params.json'), 'w') as f:
        json.dump(kwargs, f)
    launch(ctx, starting_epoch, policy_args, **kwargs)


if __name__ == '__main__':
    main()