# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Sequence, Optional
import time

from hydra.core.config_loader import ConfigLoader
from hydra.core.hydra_config import HydraConfig
from hydra.core.singleton import Singleton
from hydra.core.utils import (
    JobReturn,
    JobStatus,
    JobRuntime,
    configure_log,
    filter_overrides,
    env_override,
    setup_globals,
    _flush_loggers,
    _save_config
)
from hydra.types import TaskFunction
from joblib import Parallel, delayed  # type: ignore
from omegaconf import DictConfig, open_dict, OmegaConf, read_write

from .joblib_launcher import JoblibLauncher

log = logging.getLogger(__name__)


def run_job(
    config: DictConfig,
    task_function: TaskFunction,
    job_dir_key: str,
    job_subdir_key: Optional[str],
    configure_logging: bool = True,
) -> "JobReturn":
    from hydra._internal.callbacks import Callbacks

    callbacks = Callbacks(config)

    old_cwd = os.getcwd()
    orig_hydra_cfg = HydraConfig.instance().cfg
    HydraConfig.instance().set_config(config)
    working_dir = str(OmegaConf.select(config, job_dir_key))
    if job_subdir_key is not None:
        # evaluate job_subdir_key lazily.
        # this is running on the client side in sweep and contains things such as job:id which
        # are only available there.
        subdir = str(OmegaConf.select(config, job_subdir_key))
        working_dir = os.path.join(working_dir, subdir)

    with read_write(config.hydra.runtime):
        with open_dict(config.hydra.runtime):
            config.hydra.runtime.output_dir = os.path.abspath(working_dir)

    # update Hydra config
    HydraConfig.instance().set_config(config)
    try:
        ret = JobReturn()
        task_cfg = copy.deepcopy(config)
        with read_write(task_cfg):
            with open_dict(task_cfg):
                del task_cfg["hydra"]

        ret.cfg = task_cfg
        hydra_cfg = copy.deepcopy(HydraConfig.instance().cfg)
        assert isinstance(hydra_cfg, DictConfig)
        ret.hydra_cfg = hydra_cfg
        overrides = OmegaConf.to_container(config.hydra.overrides.task)
        assert isinstance(overrides, list)
        ret.overrides = overrides
        # handle output directories here
        Path(str(working_dir)).mkdir(parents=True, exist_ok=True)
        os.chdir(working_dir)

        if configure_logging:
            configure_log(config.hydra.job_logging, config.hydra.verbose)

        if config.hydra.output_subdir is not None:
            hydra_output = Path(config.hydra.output_subdir)
            _save_config(task_cfg, "config.yaml", hydra_output)
            _save_config(hydra_cfg, "hydra.yaml", hydra_output)
            _save_config(config.hydra.overrides.task, "overrides.yaml", hydra_output)

        with env_override(hydra_cfg.hydra.job.env_set):
            callbacks.on_job_start(config=config, task_function=task_function)
            try:
                ret.return_value = task_function(task_cfg)
                ret.status = JobStatus.COMPLETED
            except Exception:
                ret.return_value = Exception(traceback.format_exc())
                ret.status = JobStatus.FAILED

        ret.task_name = JobRuntime.instance().get("name")

        _flush_loggers()

        callbacks.on_job_end(config=config, job_return=ret)

        return ret
    finally:
        HydraConfig.instance().cfg = orig_hydra_cfg
        os.chdir(old_cwd)


def execute_job(
    idx: int,
    overrides: Sequence[str],
    config_loader: ConfigLoader,
    config: DictConfig,
    task_function: TaskFunction,
    singleton_state: Dict[Any, Any],
) -> JobReturn:
    """Calls `run_job` in parallel"""
    setup_globals()
    Singleton.set_state(singleton_state)

    sweep_config = config_loader.load_sweep_config(config, list(overrides))
    with open_dict(sweep_config):
        sweep_config.hydra.job.id = "{}_{}".format(sweep_config.hydra.job.name, idx)
        sweep_config.hydra.job.num = idx
    HydraConfig.instance().set_config(sweep_config)
    try:
        ret = run_job(
            config=sweep_config,
            task_function=task_function,
            job_dir_key="hydra.sweep.dir",
            job_subdir_key="hydra.sweep.subdir",
        )
    except Exception as e:
        log.error(f"Error running job. Exception: {e}")
        ret = JobReturn(overrides=overrides, return_value=None, cfg=sweep_config)

    return ret


def process_joblib_cfg(joblib_cfg: Dict[str, Any]) -> None:
    for k in ["pre_dispatch", "batch_size", "max_nbytes"]:
        if k in joblib_cfg.keys():
            try:
                val = joblib_cfg.get(k)
                if val:
                    joblib_cfg[k] = int(val)
            except ValueError:
                pass


def launch(
    launcher: JoblibLauncher,
    job_overrides: Sequence[Sequence[str]],
    initial_job_idx: int,
) -> Sequence[JobReturn]:
    """
    :param job_overrides: a List of List<String>, where each inner list is the arguments for one job run.
    :param initial_job_idx: Initial job idx in batch.
    :return: an array of return values from run_job with indexes corresponding to the input list indexes.
    """
    setup_globals()
    assert launcher.config is not None
    assert launcher.config_loader is not None
    assert launcher.task_function is not None

    configure_log(launcher.config.hydra.hydra_logging, launcher.config.hydra.verbose)
    sweep_dir = Path(str(launcher.config.hydra.sweep.dir))
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # Joblib's backend is hard-coded to loky since the threading
    # backend is incompatible with Hydra
    joblib_cfg = launcher.joblib
    joblib_cfg["backend"] = "loky"
    process_joblib_cfg(joblib_cfg)

    log.info(
        "Joblib.Parallel({}) is launching {} jobs".format(
            ",".join([f"{k}={v}" for k, v in joblib_cfg.items()]),
            len(job_overrides),
        )
    )
    log.info("Launching jobs, sweep output dir : {}".format(sweep_dir))
    for idx, overrides in enumerate(job_overrides):
        log.info("\t#{} : {}".format(idx, " ".join(filter_overrides(overrides))))

    singleton_state = Singleton.get_state()

    runs = Parallel(**joblib_cfg)(
        delayed(get_staggered_func(execute_job))(
            initial_job_idx + idx,
            overrides,
            launcher.config_loader,
            launcher.config,
            launcher.task_function,
            singleton_state,
        )
        for idx, overrides in enumerate(job_overrides)
    )

    assert isinstance(runs, List)
    for run in runs:
        assert isinstance(run, JobReturn)
    return runs


def get_staggered_func(func):
    """
    Wrapper to start all the processes with a slight temporal difference to avoid
    problems with WandB server communication.
    """
    def staggered_func(_id, *args, **kwargs):
        time.sleep(_id*0.1)
        return func(_id, *args, **kwargs)
    return staggered_func
