import hydra
from stable_baselines3.common import logger
import mlflow
import omegaconf
from omegaconf import DictConfig, ListConfig
import numpy as np

def get_hyperopt_score(cfg, current_run):
    data_key = cfg.early_stop_data_column
    client = mlflow.tracking.MlflowClient()
    hist = client.get_metric_history(current_run.info.run_id, data_key)
    data_val_hist  = [h.value for h in hist]
    epochs = len(hist)
    hyperopt_score = 0
    if epochs > 0:
        avg_data_val = np.mean(data_val_hist) / epochs

        smooth_last_n = min(len(data_val_hist), cfg.early_stop_last_n)
        last_n_avg_val = np.mean(data_val_hist[-smooth_last_n:])
        smoothed_data_val_growth = last_n_avg_val / epochs

        # hyperopt score is the average data value divided by epochs + last data values divided by epochs.
        hyperopt_score = avg_data_val + smoothed_data_val_growth

    return hyperopt_score, epochs

def setup_mlflow(cfg, logdir=None):
    orig_path = hydra.utils.get_original_cwd()
    mlflow.set_tracking_uri('file://' + orig_path + '/mlruns')
    tracking_uri = mlflow.get_tracking_uri()
    logger.info("Current tracking uri: {}".format(tracking_uri))
    study_name = omegaconf.OmegaConf.load(f'{orig_path}/conf/main.yaml').hydra.sweeper.study_name
    mlflow.set_experiment(study_name)
    mlflow.start_run()
    log_params_from_omegaconf_dict(cfg)
    if logdir is not None:
        mlflow.log_param(f'log_dir', logdir)


def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        explore_recursive(param_name, element)


def explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                explore_recursive(f'{parent_name}.{k}', v)
            else:
                mlflow.log_param(f'{parent_name}.{k}', v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlflow.log_param(f'{parent_name}.{i}', v)