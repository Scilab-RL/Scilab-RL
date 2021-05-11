import hydra
from stable_baselines3.common import logger
import mlflow
import omegaconf
from omegaconf import DictConfig, ListConfig

def get_avg_metric_val_epochs(data_key, current_run):
    client = mlflow.tracking.MlflowClient()
    hist = client.get_metric_history(current_run.info.run_id, data_key)
    epochs = len(hist)
    if epochs > 0:
        score = sum([h.value for h in hist]) / epochs
    else:
        score = 0
    return score, epochs

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