import os
import time
import importlib
import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import gym
import wandb

from stable_baselines3.her.her import HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

from custom_envs.register_envs import register_custom_envs
from util.util import get_git_label, set_global_seeds, get_train_video_schedule, get_eval_video_schedule, \
    avoid_start_learn_before_first_episode_finishes
from util.mlflow_util import setup_mlflow, get_hyperopt_score, log_params_from_omegaconf_dict
from util.custom_logger import setup_logger
from util.custom_callbacks import EarlyStopCallback, DisplayMetricCallBack, EvalCallback
from util.custom_wrappers import DisplayWrapper

# make git_label available in hydra
OmegaConf.register_new_resolver("git_label", get_git_label)


def get_env_instance(cfg, logger):
    def is_rlbench_env(env_name):
        return env_name.endswith('-state-v0') or cfg.env.endswith('-vision-v0')

    def is_coppelia_env(env_name):
        return env_name.startswith('Cop')

    if is_rlbench_env(cfg.env) or is_coppelia_env(cfg.env):
        # For envs based on CoppeliaSim, we can either not render at all, display train AND test,
        # or record train or test or both. 'record' will overwrite 'display'
        # e.g. render_args=[['display',1],['record',1]] will have the same effect
        # as render_args=[['none',1],['record',1]]
        render_mode = None
        if cfg.render_args[0][0] == 'display' or cfg.render_args[1][0] == 'display':
            render_mode = 'human'
        if cfg.render_args[0][0] == 'record' or cfg.render_args[1][0] == 'record':
            render_mode = 'rgb_array'
        # there can be only one PyRep instance per process, therefore train_env == eval_env
        if is_rlbench_env(cfg.env):
            from custom_envs.wrappers.rl_bench_wrapper import RLBenchWrapper
            rlbench_env = gym.make(cfg.env, render_mode=render_mode, **cfg.env_kwargs)
            train_env = RLBenchWrapper(rlbench_env, "train")
            eval_env = RLBenchWrapper(rlbench_env, "eval")
        else:
            train_env = gym.make(cfg.env, render_mode=render_mode, **cfg.env_kwargs)
            eval_env = gym.make(cfg.env, render_mode=render_mode, **cfg.env_kwargs)
    else:
        train_env = gym.make(cfg.env, **cfg.env_kwargs)
        eval_env = gym.make(cfg.env, **cfg.env_kwargs)

    # wrappers for rendering
    if cfg.render_args[0][0] == 'display':
        train_env = DisplayWrapper(train_env, cfg.render_args[0][1], epoch_steps=cfg.eval_after_n_steps)
    if cfg.render_args[1][0] == 'display':
        eval_env = DisplayWrapper(eval_env, cfg.render_args[1][1], epoch_episodes=cfg.n_test_rollouts)
    if cfg.render_args[0][0] == 'record':
        train_env = gym.wrappers.RecordVideo(env=train_env,
                                             video_folder=logger.get_dir() + "/videos",
                                             name_prefix="train",
                                             step_trigger=get_train_video_schedule(cfg.eval_after_n_steps
                                                                                   * cfg.render_args[0][1]))
    if cfg.render_args[1][0] == 'record':
        eval_env = gym.wrappers.RecordVideo(env=eval_env,
                                            video_folder=logger.get_dir() + "/videos",
                                            name_prefix="eval",
                                            episode_trigger=get_eval_video_schedule(cfg.render_args[1][1],
                                                                                    cfg.n_test_rollouts))

    # The following gym wrappers can be added via commandline parameters,
    # e.g. use +flatten_obs to use the FlattenObservation wrapper
    if 'flatten_obs' in cfg and cfg.flatten_obs:
        train_env = gym.wrappers.FlattenObservation(train_env)
        eval_env = gym.wrappers.FlattenObservation(eval_env)

    if 'clip_action' in cfg and cfg.clip_action:
        train_env = gym.wrappers.ClipAction(train_env)
        eval_env = gym.wrappers.ClipAction(eval_env)

    if 'normalize_obs' in cfg and cfg.normalize_obs:
        train_env = gym.wrappers.NormalizeObservation(train_env)
        eval_env = gym.wrappers.NormalizeReward(eval_env)

    if 'normalize_reward' in cfg and cfg.normalize_reward:
        train_env = gym.wrappers.NormalizeReward(train_env)
        eval_env = gym.wrappers.NormalizeReward(eval_env)

    if 'time_aware_observation' in cfg and cfg.time_aware_observation:
        train_env = gym.wrappers.TimeAwareObservation(train_env)
        eval_env = gym.wrappers.TimeAwareObservation(eval_env)

    # At last, wrap in DummyVecEnv. This has to be the last wrapper, because it breaks the .unwrapped attribute.
    train_env = DummyVecEnv([lambda: train_env])
    eval_env = DummyVecEnv([lambda: eval_env])

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
        alg_kwargs = avoid_start_learn_before_first_episode_finishes(alg_kwargs, env)
    if cfg.restore_policy is not None:
        baseline = baseline_class.load(cfg.restore_policy, env=env, **alg_kwargs)
    else:
        baseline = baseline_class(env=env, **alg_kwargs)
    baseline.set_logger(logger)
    return baseline


def create_callbacks(cfg, logger, eval_env):
    callback = []

    # cfg.render_args[0][2][k][0] == 1 -> episodic animation
    # cfg.render_args[0][2][k][1] == 2 -> one animation

    # for training
    metrics = []
    if len(cfg.render_args[0]) > 2:
        if (cfg.render_args[0][0] == 'display' or cfg.render_args[0][0] == 'record') and cfg.render_args[0][2][-1] != 0:
            v_episodic = True
            v_save_anim = False
            if cfg.render_args[0][0] == 'record':
                v_save_anim = True
            if cfg.render_args[0][2][-1] == 2:
                v_episodic = False
            display_metric_callback_train = DisplayMetricCallBack(cfg.render_args[0][2][:len(cfg.render_args[0][2])-1], logger,
                                                                  episodic=v_episodic,
                                                                  save_anim=v_save_anim)
            callback.append(display_metric_callback_train)

    # for testing

    # custom callback necessary for eval metric viz
    # If display_metric_callback_test stays None --> no metric visualization
    display_metric_callback_test = None

    if len(cfg.render_args[1]) > 2:
        if (cfg.render_args[1][0] == 'display' or cfg.render_args[1][0] == 'record') and cfg.render_args[1][2][-1] != 0:
            v_episodic = True
            v_save_anim = False
            if cfg.render_args[1][0] == 'record':
                v_save_anim = True
            if cfg.render_args[1][2][-1] == 2:
                v_episodic = False
            display_metric_callback_test = DisplayMetricCallBack(cfg.render_args[1][2][:len(cfg.render_args[1][2])-1], logger, episodic=v_episodic,
                                                                 save_anim=v_save_anim)

    if cfg.save_model_freq > 0:
        checkpoint_callback = CheckpointCallback(save_freq=cfg.save_model_freq, save_path=logger.get_dir(), verbose=1)
        callback.append(checkpoint_callback)

    eval_callback = EvalCallback(eval_env, n_eval_episodes=cfg.n_test_rollouts, eval_freq=cfg.eval_after_n_steps,
                                 log_path=logger.get_dir(), best_model_save_path=None, render=False, warn=False,
                                 callback_metric_viz=display_metric_callback_test)
    callback.append(eval_callback)
    early_stop_callback = EarlyStopCallback(metric=cfg.early_stop_data_column, eval_freq=cfg.eval_after_n_steps,
                                            threshold=cfg.early_stop_threshold, n_episodes=cfg.early_stop_last_n)
    callback.append(early_stop_callback)
    callback = CallbackList(callback)
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
        run_id = mlflow.active_run().info.run_id
        print(f"Active mlflow run_id: {run_id}")
        log_params_from_omegaconf_dict(cfg)
        OmegaConf.save(config=cfg, f='params.yaml')
        if cfg['seed'] == 0:
            cfg['seed'] = int(time.time())
        set_global_seeds(cfg.seed)

        train_env, eval_env = get_env_instance(cfg, logger)

        baseline = get_algo_instance(cfg, logger, train_env)

        callback = create_callbacks(cfg, logger, eval_env)

        logger.info("Launching training")
        training_finished = False
        total_steps = cfg.eval_after_n_steps * cfg.n_epochs
        try:
            baseline.learn(total_timesteps=total_steps, callback=callback, log_interval=None)
            training_finished = True
            logger.info("Training finished!")
            # Save model when training is finished
            p = logger.get_dir() + "/rl_model_finished"
            logger.info(f"Saving policy to {p}")
            baseline.save(path=p)
        except ValueError as e:
            if e.args[0].startswith("Expected parameter loc"):
                logger.error(f"The experiment failed with error {e}")
                logger.error("If this error happened because of a tensor with NaNs in it, "
                             "that is probably because the chosen hyperparameters made the algorithm unstable.")
            else:
                raise e
        train_env.close()
        eval_env.close()

        # after training
        if training_finished:
            hyperopt_score, n_epochs = get_hyperopt_score(cfg, mlflow_run)
        else:
            hyperopt_score, n_epochs = -1, cfg["n_epochs"]
        logger.info(f"Hyperopt score: {hyperopt_score}, epochs: {n_epochs}.")
        mlflow.log_metric("hyperopt_score", hyperopt_score)
        with open(os.path.join(run_dir, 'train.log'), 'r') as logfile:
            log_text = logfile.read()
            mlflow.log_text(log_text, 'train.log')
        if cfg["wandb"]:
            wandb.log({"hyperopt_score": hyperopt_score})
            wandb.finish()

    return hyperopt_score, n_epochs, run_id


if __name__ == '__main__':
    main()
