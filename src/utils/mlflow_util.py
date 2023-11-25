import hydra
import mlflow
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig
import numpy as np


def get_hyperopt_score(cfg, current_run):
    data_key = cfg.hyperopt_criterion
    client = mlflow.tracking.MlflowClient()
    hist = client.get_metric_history(current_run.info.run_id, data_key)
    data_val_hist = [h.value for h in hist]
    epochs = len(hist)
    hyperopt_score = 0
    if epochs > 0:
        dval_mean = np.mean(data_val_hist)
        if dval_mean >= 0:
            avg_data_val = np.mean(data_val_hist) / epochs
            # smooth_last_n values is 20% of data_val_hist
            smooth_last_n = max(1, len(data_val_hist)//5)
            last_n_avg_val = np.mean(data_val_hist[-smooth_last_n:])
            smoothed_data_val_growth = last_n_avg_val / epochs

            # hyperopt score is the average data value divided by epochs + last data values divided by epochs.
            hyperopt_score = avg_data_val + smoothed_data_val_growth
        else: # if data is not positive, we have to multiply with number of epochs not divide by epochs.
            data_val_epochs = np.mean(data_val_hist) * epochs
            # smooth_last_n values is 20% of data_val_hist
            smooth_last_n = max(1, len(data_val_hist) // 5)
            last_n_avg_val = np.mean(data_val_hist[-smooth_last_n:])
            smoothed_data_val_epochs = last_n_avg_val * epochs

            # hyperopt score is here negative.
            hyperopt_score = data_val_epochs + smoothed_data_val_epochs

    return hyperopt_score, epochs


def setup_mlflow(cfg: DictConfig):
    orig_path = hydra.utils.get_original_cwd()
    mlflow.set_tracking_uri('file://' + orig_path + '/mlruns')
    experiment_name = 'Default'
    # if multirun with sweeper
    if HydraConfig.get().sweeper.study_name:
        experiment_name = f"{HydraConfig.get().sweeper.study_name}"
    elif 'defaults' in cfg.keys() and cfg.defaults == 'smoke_test':
        experiment_name = 'smoke_test'
    print('MLFlow experiment name', experiment_name)
    mlflow.set_experiment(experiment_name)


def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        explore_recursive(param_name, element)


def explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, (DictConfig, ListConfig)):
                explore_recursive(f'{parent_name}.{k}', v)
            else:
                mlflow.log_param(f'{parent_name}.{k}', v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlflow.log_param(f'{parent_name}.{i}', v)
