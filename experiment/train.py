"""
This is the main script from which all training runs are started.
"""
import os
import sys
import time
import importlib
import mlflow
import hydra
import gym
import wandb
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.her.her import HerReplayBuffer
from omegaconf import DictConfig, OmegaConf, open_dict
from custom_envs.register_envs import register_custom_envs
from util.mlflow_util import setup_mlflow, get_hyperopt_score, log_params_from_omegaconf_dict
from util.util import get_git_label, set_global_seeds, flatten_dictConf
from util.custom_logger import FixedHumanOutputFormat, MLFlowOutputFormat, WandBOutputFormat
from util.custom_eval_callback import CustomEvalCallback

sys.path.append(os.getcwd())
# make git_label available in hydra
OmegaConf.register_new_resolver("git_label", get_git_label)


def train(baseline, train_env, eval_env, cfg, logger):
    total_steps = cfg.eval_after_n_steps * cfg.n_epochs
    callback = []
    if cfg.save_model_freq > 0:
        checkpoint_callback = CheckpointCallback(save_freq=cfg.save_model_freq, save_path=logger.get_dir())
        callback.append(checkpoint_callback)
    eval_callback = CustomEvalCallback(eval_env,
                                       agent=baseline,
                                       log_path=logger.get_dir(),
                                       render_args=cfg.render_args,
                                       eval_freq=cfg.eval_after_n_steps,
                                       n_eval_episodes=cfg.n_test_rollouts,
                                       early_stop_last_n=cfg.early_stop_last_n,
                                       early_stop_data_column=cfg.early_stop_data_column,
                                       early_stop_threshold=cfg.early_stop_threshold,
                                       logger=logger)
    # Create the callback list
    callback.append(eval_callback)
    training_finished = False
    try:
        baseline.learn(total_timesteps=total_steps, callback=callback, log_interval=None)
        training_finished = True
    except ValueError as e:
        logger.error(f"The experiment failed with error {e}")
        logger.error("If this error happened because of a tensor with NaNs in it, that is probably because the chosen "
                     "hyperparameters made the algorithm unstable.")
    train_env.close()
    eval_env.close()
    if training_finished:
        logger.info("Training finished!")
    return training_finished


def launch(cfg, logger, kwargs):
    set_global_seeds(cfg.seed)
    algo_name = cfg['algorithm'].name
    with open_dict(cfg):
        del cfg['algorithm']['name']  # remove name as we pass all arguments to the model constructor
    try:
        baseline_class = getattr(importlib.import_module('stable_baselines3.' + algo_name), algo_name.upper())
    except ModuleNotFoundError:
        baseline_class = getattr(importlib.import_module('custom_algorithms.' + algo_name), algo_name.upper())
    if cfg.env.endswith('-state-v0') or cfg.env.endswith('-vision-v0'):  # if the environment is an RLBench env
        from custom_envs.wrappers.rl_bench_wrapper import RLBenchWrapper
        render_mode = None
        # For RLBench envs, we can either not render at all, display train AND test, or record train or test or both
        # record will overwrite display
        # e.g. render_args=[['display',1],['record',1]] will have the same effect
        # as render_args=[['none',1],['record',1]]
        if cfg.render_args[0][0] == 'display' or cfg.render_args[1][0] == 'display':
            render_mode = 'human'
        if cfg.render_args[0][0] == 'record' or cfg.render_args[1][0] == 'record':
            render_mode = 'rgb_array'
        # there can be only one PyRep instance per process, therefore train_env == eval_env
        rlbench_env = gym.make(cfg.env, render_mode=render_mode)
        train_env = RLBenchWrapper(rlbench_env, "train")
        eval_env = RLBenchWrapper(rlbench_env, "eval")
    else:
        train_env = gym.make(cfg.env)
        eval_env = gym.make(cfg.env)
    alg_kwargs = OmegaConf.to_container(cfg.algorithm)
    if cfg.restore_policy is not None:
        baseline = baseline_class.load(cfg.restore_policy, **alg_kwargs, env=train_env, **kwargs)
    elif 'using_her' in cfg and cfg.using_her:  # enable with +replay_buffer=her
        # if learning_starts < max_episode_steps, learning starts before the first episode is stored
        if 'learning_starts' in alg_kwargs:
           alg_kwargs['learning_starts'] = max(alg_kwargs['learning_starts'], train_env._max_episode_steps)
        else:
           alg_kwargs['learning_starts'] = train_env._max_episode_steps
        baseline = baseline_class(policy='MultiInputPolicy', env=train_env, replay_buffer_class=HerReplayBuffer,
                                  **alg_kwargs, **kwargs)
    else:
        baseline = baseline_class(policy='MultiInputPolicy', env=train_env, **alg_kwargs, **kwargs)
    baseline.set_logger(logger)
    logger.info("Launching training")
    return train(baseline, train_env, eval_env, cfg, logger)


@hydra.main(config_name="main", config_path="../conf")
def main(cfg: DictConfig) -> (float, int):
    run_dir = os.getcwd()
    if cfg.restore_policy is not None:
        run_dir = os.path.split(cfg.restore_policy)[:-1][0]
        run_dir = run_dir + "_restored"

    if 'performance_testing_conditions' in cfg:
        cfg['n_epochs'] = int(cfg['performance_testing_conditions']['max_steps'] / cfg['eval_after_n_steps'])

    register_custom_envs()
    setup_mlflow(cfg)
    run_name = cfg['algorithm']['name'] + '_' + cfg['env']
    with mlflow.start_run(run_name=run_name) as mlflow_run:
        if run_dir is not None:
            mlflow.log_param('log_dir', run_dir)

        # Output will only be logged appropriately after configuring the logger in the following lines:
        logger = configure(folder=run_dir, format_strings=[])
        logger.output_formats.append(FixedHumanOutputFormat(sys.stdout))
        logger.output_formats.append(FixedHumanOutputFormat(os.path.join(run_dir, "train.log")))
        logger.output_formats.append(MLFlowOutputFormat())
        if cfg["wandb"]:
            non_nested_cfg = flatten_dictConf(cfg)
            os.environ["WANDB_START_METHOD"] = "thread"
            wandb_args = dict(project=cfg.project_name if cfg.project_name else run_name,
                    config=non_nested_cfg)
            if 'entity' in cfg:
                wandb_args['entity'] = cfg['entity']
            if 'group' in cfg:
                wandb_args['group'] = cfg['group']
            if 'tags' in cfg:
                wandb_args['tags'] = cfg['tags']
            wandb.init(**wandb_args)
            logger.output_formats.append(WandBOutputFormat())
        logger.info("Starting training with the following configuration:")
        logger.info(OmegaConf.to_yaml(cfg))
        logger.info(f"Log directory: {run_dir}")
        # End configure logger
        active_mlflow_run = mlflow.active_run()
        print(f"Active mlflow run_id: {active_mlflow_run.info.run_id}")

        logger.info(f"Starting process id: {os.getpid()}")

        if cfg['seed'] == 0:
            cfg['seed'] = int(time.time())

        log_params_from_omegaconf_dict(cfg)
        OmegaConf.save(config=cfg, f='params.yaml')

        kwargs = {}
        training_finished = launch(cfg, logger, kwargs)

        logger.info("Finishing main training function.")
        logger.info(f"MLflow run: {mlflow_run}.")
        if training_finished:
            hyperopt_score, n_epochs = get_hyperopt_score(cfg, mlflow_run)
        else:
            hyperopt_score, n_epochs = 0, cfg["n_epochs"]
        mlflow.log_metric("hyperopt_score", hyperopt_score)
        if cfg["wandb"]:
            wandb.log({"hyperopt_score": hyperopt_score})
            wandb.finish()
        logger.info(f"Hyperopt score: {hyperopt_score}, epochs: {n_epochs}.")
        try:
            with open(os.path.join(run_dir, 'train.log'), 'r') as logfile:
                log_text = logfile.read()
                mlflow.log_text(log_text, 'train.log')
        except:
            logger.info('Could not open logfile and log it in mlflow.')
        run_id = mlflow.active_run().info.run_id

    return hyperopt_score, n_epochs, run_id


if __name__ == '__main__':
    main()
