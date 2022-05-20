import os
import time
import importlib
import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import gym
import wandb
from stable_baselines3.her.her import HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from custom_envs.register_envs import register_custom_envs
from util.util import get_git_label, set_global_seeds
from util.mlflow_util import setup_mlflow, get_hyperopt_score, log_params_from_omegaconf_dict
from util.custom_logger import setup_logger
from util.custom_callbacks import EarlyStopCallback

# make git_label available in hydra
OmegaConf.register_new_resolver("git_label", get_git_label)

# DONE train with MuJoCo Envs
# TODO train with RLBench Envs
# TODO train with custom Envs
# train with SB3 algorithms
# - TODO on policy
# - DONE off policy
# TODO train with custom algorithms
# TODO configure with hydra
# DONE hyperopt-score
# render
# - TODO display
# - TODO record
# DONE restore
# TODO wandb
# TODO mlflow
# DONE early stopping
# DONE evaluation
#
# TODO hyperopt should not fail when experiment fails
#
# LATER
# adjust scripts
# adjust hyperopt
# adjust smoke and performance tests
# adjust wiki


def get_env_instance(cfg):
    if cfg.env.endswith('-state-v0') or cfg.env.endswith('-vision-v0'):  # if the environment is an RLBench env
        from custom_envs.wrappers.rl_bench_wrapper import RLBenchWrapper
        render_mode = None  # TODO change this whole render Extrawurst if possible
        # For RLBench envs, we can either not render at all, display train AND test, or record train or test or both
        # record will overwrite display
        # e.g. render_args=[['display',1],['record',1]] will have the same effect
        # as render_args=[['none',1],['record',1]]
        if cfg.render_args[0][0] == 'display' or cfg.render_args[1][0] == 'display':
            render_mode = 'human'
        if cfg.render_args[0][0] == 'record' or cfg.render_args[1][0] == 'record':
            render_mode = 'rgb_array'
        # there can be only one PyRep instance per process, therefore train_env == eval_env
        rlbench_env = gym.make(cfg.env, render_mode=render_mode, **cfg.env_kwargs)
        train_env = RLBenchWrapper(rlbench_env, "train")
        eval_env = RLBenchWrapper(rlbench_env, "eval")  # TODO maybe only train_env, worth a try as evaluation.py resets at the start.
    else:
        train_env = gym.make(cfg.env, **cfg.env_kwargs)
        eval_env = gym.make(cfg.env, **cfg.env_kwargs)
    return train_env, eval_env


def get_algo_instance(cfg, logger, env):
    algo_name = cfg['algorithm'].name
    alg_kwargs = OmegaConf.to_container(cfg.algorithm)
    del alg_kwargs['name']  # remove name as we pass all arguments to the model constructor
    try:
        baseline_class = getattr(importlib.import_module('stable_baselines3.' + algo_name), algo_name.upper())
    except ModuleNotFoundError:
        baseline_class = getattr(importlib.import_module('custom_algorithms.' + algo_name), algo_name.upper())
    if 'replay_buffer_class' in alg_kwargs and alg_kwargs['replay_buffer_class'] == 'HerReplayBuffer':
        alg_kwargs['replay_buffer_class'] = HerReplayBuffer
        # if learning_starts < max_episode_steps, learning starts before the first episode is stored
        if 'learning_starts' in alg_kwargs:
            alg_kwargs['learning_starts'] = max(alg_kwargs['learning_starts'], env._max_episode_steps)
        else:
            alg_kwargs['learning_starts'] = env._max_episode_steps
    if cfg.restore_policy is not None:
        baseline = baseline_class.load(cfg.restore_policy, env=env, **alg_kwargs)
    else:
        baseline = baseline_class(env=env, **alg_kwargs)
    baseline.set_logger(logger)
    return baseline


def create_callbacks(cfg, logger, eval_env):
    callback = []
    if cfg.save_model_freq > 0:
        checkpoint_callback = CheckpointCallback(save_freq=cfg.save_model_freq, save_path=logger.get_dir(), verbose=1)
        callback.append(checkpoint_callback)
    # Create the callback list
    eval_callback = EvalCallback(eval_env, n_eval_episodes=cfg.n_test_rollouts, eval_freq=cfg.eval_after_n_steps,
                                 log_path=logger.get_dir(), best_model_save_path=None)
    callback.append(eval_callback)
    early_stop_callback = EarlyStopCallback(metric=cfg.early_stop_data_column, eval_freq=cfg.eval_after_n_steps,
                                            threshold=cfg.early_stop_threshold, n_episodes=cfg.early_stop_last_n)
    callback.append(early_stop_callback)
    return callback


@hydra.main(config_name="main", config_path="conf")
def main(cfg: DictConfig) -> (float, int):
    run_dir = os.getcwd()
    if cfg.restore_policy is not None:
        run_dir = os.path.split(cfg.restore_policy)[:-1][0]
        run_dir = run_dir + "_restored"
    run_name = cfg['algorithm']['name'] + '_' + cfg['env']

    register_custom_envs()
    setup_mlflow(cfg)

    with mlflow.start_run(run_name=run_name) as mlflow_run:
        mlflow.log_param('log_dir', run_dir)
        logger = setup_logger(run_dir, run_name, cfg)
        run_id = mlflow.active_run().info.run_id  # TODO delete if not used in hyperopt
        print(f"Active mlflow run_id: {run_id}")  # this too
        log_params_from_omegaconf_dict(cfg)
        OmegaConf.save(config=cfg, f='params.yaml')

        if cfg['seed'] == 0:
            cfg['seed'] = int(time.time())
        set_global_seeds(cfg.seed)

        train_env, eval_env = get_env_instance(cfg)

        baseline = get_algo_instance(cfg, logger, train_env)

        logger.info("Launching training")
        total_steps = cfg.eval_after_n_steps * cfg.n_epochs

        callback = create_callbacks(cfg, logger, eval_env)

        training_finished = False
        try:
            baseline.learn(total_timesteps=total_steps, callback=callback, log_interval=None)
            training_finished = True
            logger.info("Training finished!")
        except ValueError as e:
            if e.args[0].startswith("Expected parameter loc"):
                logger.error(f"The experiment failed with error {e}")
                logger.error("If this error happened because of a tensor with NaNs in it, "
                             "that is probably because the chosen hyperparameters made the algorithm unstable.")
            else:
                raise e
        train_env.close()
        eval_env.close()

        # TODO Use different wrappers and callbacks
        # custom wrapper for early stopping
        # vec_video_recorder for recording
        # what for display?

        # after training
        if training_finished:
            hyperopt_score, n_epochs = get_hyperopt_score(cfg, mlflow_run)
        else:
            hyperopt_score, n_epochs = 0, cfg["n_epochs"]
        logger.info(f"Hyperopt score: {hyperopt_score}, epochs: {n_epochs}.")
        mlflow.log_metric("hyperopt_score", hyperopt_score)
        with open(os.path.join(run_dir, 'train.log'), 'r') as logfile:
            log_text = logfile.read()
            mlflow.log_text(log_text, 'train.log')
        if cfg["wandb"]:
            wandb.log({"hyperopt_score": hyperopt_score})
            wandb.finish()

    return hyperopt_score, n_epochs, run_id  # TODO run_id probably not needed


if __name__ == '__main__':
    main()
