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
from evalLogCallback import EvalLogCallback
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

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

def train(model, train_env, eval_env, n_epochs, policy_save_interval, starting_epoch, **kwargs):
    best_early_stop_val = -np.inf
    n_epochs_avg_for_early_stop = 4
    early_stop_vals = deque(maxlen=n_epochs_avg_for_early_stop)
    total_action_steps = np.product([int(steps) for steps in kwargs['action_steps'].split(',')])
    actions_per_episode = np.product([int(steps) for steps in kwargs['action_steps'].split(',')])
    train_actions_per_epoch = actions_per_episode * kwargs['n_train_rollouts']
    # test_actions_per_epoch = actions_per_episode * kwargs['n_test_rollouts']
    epochs_remaining = n_epochs - starting_epoch
    total_actions = train_actions_per_epoch * epochs_remaining


    checkpoint_callback = CheckpointCallback(save_freq=train_actions_per_epoch, save_path=logger.get_dir())
    # Separate evaluation env

    eval_callback = EvalCallback(eval_env, best_model_save_path=logger.get_dir()+"/policy_best",
                                 log_path=logger.get_dir(), eval_freq=train_actions_per_epoch, n_eval_episodes=kwargs['n_test_rollouts'])
    
    # Create the callback list
    callback = CallbackList([checkpoint_callback, eval_callback])
    # callback = CallbackList([checkpoint_callback])

    # train_steps_per_epoch = kwargs['n_train_rollouts'] * total_action_steps
    # test_steps_per_epoch = kwargs['n_test_rollouts'] * total_action_steps
    # done_training = False
    # for epoch in range(starting_epoch, n_epochs):
        # logger_dump_fkt = logger.dump
        # logger.dump = lambda step: step
    model.learn(total_timesteps=total_actions,callback=callback)
        # logger.dump = logger_dump_fkt

        # epoch_info = {'reward': [], 'std_reward' : []}
        #
        #
        # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=kwargs['n_test_rollouts'], deterministic=True)
        # epoch_info['reward'].append(mean_reward)
        # epoch_info['std_reward'].append(std_reward)
        # epoch_avg = {"test/"+k:np.mean(epoch_info[k]) for k in epoch_info.keys()}
        # epoch_avg['epoch'] = epoch
        # epoch_avg['train/steps'] = (epoch + 1) * train_steps_per_epoch
        # for k,v in epoch_avg.items():
        #     logger.record_tabular(k, v)
        # logger.info("Data_dir: {}".format(logger.get_dir()))
        # if kwargs['early_stop_data_column'] in  logger.get_log_dict().keys():
        #     early_stop_current_val = logger.get_log_dict()[kwargs['early_stop_data_column']]
        #     early_stop_vals.append(early_stop_current_val)
        # else:
        #     logger.warn("Warning: Early stop data column \"{}\" not in logged values.".format(kwargs['early_stop_data_column']))
        #     early_stop_current_val = 0
        # logger.dump_tabular()
        #
        # latest_policy_path = os.path.join(logger.get_dir(), 'policy_latest')
        # best_policy_path = os.path.join(logger.get_dir(), 'policy_best')
        # periodic_policy_path = os.path.join(logger.get_dir(), 'policy_{}')
        # model.save(latest_policy_path)
        #
        # if policy_save_interval > 0 and epoch % policy_save_interval == 0:
        #     policy_path = periodic_policy_path.format(epoch)
        #     model.save(policy_path)
        #
        # # save the policy if it's better than the previous ones
        # if kwargs['early_stop_data_column'] in  logger.get_log_dict():
        #     assert early_stop_current_val is not None, "Early stopping value should not be none."
        #     if early_stop_current_val >= best_early_stop_val:
        #         best_early_stop_val = early_stop_current_val
        #         logger.info(
        #             'New best value for {}: {}. Saving policy to {}.'.format(kwargs['early_stop_data_column'],
        #                                                                         early_stop_current_val,
        #                                                                         best_policy_path))
        #         model.save(best_policy_path)
        #
        # if len(early_stop_vals) >= n_epochs_avg_for_early_stop:
        #     avg = np.mean(early_stop_vals)
        #     logger.info('Mean of {} of last {} epochs: {}'.format(kwargs['early_stop_data_column'],
        #                                                           n_epochs_avg_for_early_stop, avg))
        #
        #     if avg >= kwargs['early_stop_threshold'] and avg >= kwargs['early_stop_threshold'] != 0:
        #         logger.info('Policy is good enough now, early stopping')
        #         done_training = True
        # if done_training:
        #     break
    train_env.close()
    eval_env.close()

def launch(starting_epoch, policy_args, env, algorithm,  n_epochs, seed, policy_save_interval, restore_policy, logdir, **kwargs):
    set_global_seeds(seed)

    model_class = DQN  # works also with SAC, DDPG and TD3
    N_BITS = 15
    #
    train_env = BitFlippingEnv(n_bits=N_BITS, continuous=model_class in [DDPG, SAC, TD3], max_steps=N_BITS)
    eval_env = BitFlippingEnv(n_bits=N_BITS, continuous=model_class in [DDPG, SAC, TD3], max_steps=N_BITS)
    #
    # # Available strategies (cf paper): future, final, episode
    # goal_selection_strategy = 'future'  # equivalent to GoalSelectionStrategy.FUTURE
    #
    # # If True the HER transitions will get sampled online
    # online_sampling = True
    # # Time limit for the episodes
    # max_episode_length = N_BITS
    #
    # # Initialize the model
    # model = HER('MlpPolicy', env, model_class, n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy,
    #             online_sampling=online_sampling,
    #             verbose=1, max_episode_length=max_episode_length)

    env_alg_compatible = True

    ModelClass = getattr(importlib.import_module('stable_baselines3.' + algorithm), algorithm.upper())
    # train_env = gym.make(env)
    # eval_env = gym.make(env)
    ## env = make_robustGoalConditionedHierarchicalEnv(env)
    ## check_env(env)

    if restore_policy is not None:
        model = ModelClass.load(restore_policy, env=train_env)
    else:
        if algorithm == 'her':
            policy_args['model_class'] = getattr(importlib.import_module('stable_baselines3.' + policy_args['model_class']), policy_args['model_class'].upper())
        model = ModelClass('MlpPolicy', train_env, **policy_args)

    # model = make_robustGoalConditionedModel(model)
    #
    # env_alg_compatible = check_env_alg_compatibility(model, env)

    if env_alg_compatible:
        logger.info("Launching training")
        # model.learn(10)
        train(model, train_env, eval_env, n_epochs, policy_save_interval, starting_epoch, **kwargs)

@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@main_linker.click_main
@click.pass_context
def main(ctx, **kwargs):
    config = main_linker.import_creator(kwargs['algorithm'])
    policy_args = ctx.forward(main_linker.get_policy_click)
    cmd_line_update_args = {ctx.args[i][2:]: type(policy_args[ctx.args[i][2:]])(ctx.args[i + 1]) for i in
                            range(0, len(ctx.args), 2)}
    kwargs.update(cmd_line_update_args)
    kwargs.update(ctx.params)
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
    logger.Logger.CURRENT.output_formats.append(MatplotlibOutputFormat(kwargs['logdir']+'/plot.csv',cols_to_plot=['test/reward', 'train/entropy_loss', 'train/loss']))
    logdir = logger.get_dir()

    logger.info("Data dir: {} ".format(logdir))
    with open(os.path.join(logdir, 'params.json'), 'w') as f:
        json.dump(kwargs, f)
    launch(starting_epoch, policy_args, **kwargs)


if __name__ == '__main__':
    main()