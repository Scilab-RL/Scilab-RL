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
from util.custom_logger import MatplotlibOutputFormat
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from util.compat_wrappers import make_robustGoalConditionedHierarchicalEnv, make_robustGoalConditionedModel
from stable_baselines3.common.bit_flipping_env import BitFlippingEnv
from stable_baselines3 import HER, DDPG, DQN, SAC, TD3
from util.custom_eval_callback import CustomEvalCallback
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.her import HER

ALL_PATH_CONFIG_PARAMS = ['info', 'algorithm']

def check_env_alg_compatibility(model, env):
    obs = env.reset()
    try:
        action, _states = model.predict(obs, deterministic=True)
        env.step(action)
        is_compatible = True
    except Exception as e:
        logger.info("Environment and algorithm not compatible, probably because of different action spaces.")
        logger.info(e)
        is_compatible = False
    return is_compatible

def train(model, train_env, eval_env, n_epochs, steps_per_epoch, starting_epoch, **kwargs):
    # actions_per_episode = np.product([int(steps) for steps in kwargs['action_steps'].split(',')])
    train_actions_per_epoch = steps_per_epoch * kwargs['n_train_rollouts']
    epochs_remaining = n_epochs - starting_epoch
    total_actions = train_actions_per_epoch * epochs_remaining

    checkpoint_callback = CheckpointCallback(save_freq=train_actions_per_epoch, save_path=logger.get_dir())
    eval_callback = CustomEvalCallback(eval_env,
                                       log_path=logger.get_dir(),
                                       eval_freq=train_actions_per_epoch,
                                       n_eval_episodes=kwargs['n_test_rollouts'],
                                       render=kwargs['render_test'],
                                       early_stop_last_n=5,
                                       early_stop_data_column=kwargs['early_stop_data_column'],
                                       early_stop_threshold=kwargs['early_stop_threshold'],
                                       )

    # Create the callback list
    callback = CallbackList([checkpoint_callback, eval_callback])
    model.learn(total_timesteps=total_actions,callback=callback, log_interval=kwargs['n_train_rollouts'])

    train_env.close()
    eval_env.close()

def launch(ctx, starting_epoch, policy_args, env, algorithm,  n_epochs, seed, policy_save_interval, restore_policy, logdir, **kwargs):
    set_global_seeds(seed)

    # model_class = DDPG  # works also with SAC, DDPG and TD3
    # N_BITS = int(kwargs['action_steps'])
    # #
    # train_env = BitFlippingEnv(n_bits=N_BITS, continuous=model_class in [DDPG, SAC, TD3], max_steps=int(kwargs['action_steps']))
    # eval_env = BitFlippingEnv(n_bits=N_BITS, continuous=model_class in [DDPG, SAC, TD3], max_steps=int(kwargs['action_steps']))
    #
    # env_alg_compatible = True

    ModelClass = getattr(importlib.import_module('stable_baselines3.' + algorithm), algorithm.upper())
    train_env = gym.make(env)
    eval_env = gym.make(env)

    if hasattr(train_env, '_max_episode_steps'):
        policy_args['max_episode_length'] = train_env._max_episode_steps
    else:
        policy_args['max_episode_length'] = kwargs['action_steps']
    if restore_policy is not None:
        model = ModelClass.load(restore_policy, env=train_env)
    else:
        if algorithm == 'her':
            policy_args['model_class'] = getattr(importlib.import_module('stable_baselines3.' + policy_args['model_class']), policy_args['model_class'].upper())
        model = ModelClass('MlpPolicy', train_env, **policy_args)
    logger.info("Launching training")
    train(model, train_env, eval_env, n_epochs, steps_per_epoch=policy_args['max_episode_length'], starting_epoch=starting_epoch, **kwargs)

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

    # Get default options for model class
    if 'model_class' in policy_args.keys():
        model_click = importlib.import_module('interface.' + policy_args['model_class'] + '.click_options')
        model_class_options = model_click.get_click_option.params
        model_class_args = {o.name: o.default for o in model_class_options}
        changed_mc_args = {k: type(model_class_args[k])(v) for k, v in all_cmd_kvs.items() if k in model_class_args.keys()}
        model_class_args.update(changed_mc_args)
        for k,v in model_class_args.items():
            if k in policy_args.keys():
                logger.warn("Warning, model class and algorithm have a common parameter {}. This will cause the model class to use the default parameter provided in its constructor.".format(k))
            else:
                policy_args[k] = v

    kwargs.update(policy_args)
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
                     format_strings=['stdout', 'log', 'csv', 'tensorboard'])
    logger.Logger.CURRENT.output_formats.append(MatplotlibOutputFormat(kwargs['logdir']+'/plot.csv',cols_to_plot=['test/mean_reward', 'train/entropy_loss', 'train/loss']))
    logdir = logger.get_dir()

    logger.info("Data dir: {} ".format(logdir))
    with open(os.path.join(logdir, 'params.json'), 'w') as f:
        json.dump(kwargs, f)
    launch(ctx, starting_epoch, policy_args, **kwargs)

if __name__ == '__main__':
    main()