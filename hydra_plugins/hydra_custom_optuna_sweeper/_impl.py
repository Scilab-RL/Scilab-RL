# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import sys
from typing import Any, Dict, List, MutableMapping, MutableSequence, Optional
from optuna.trial import TrialState
import optuna
import time
import datetime
from optuna import pruners
from hydra_plugins.hydra_custom_optuna_sweeper.param_repeat_pruner import ParamRepeatPruner
from hydra.core.config_loader import ConfigLoader
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.override_parser.types import (
    ChoiceSweep,
    IntervalSweep,
    Override,
    RangeSweep,
    Transformer,
)
import multiprocessing as mp
from hydra.core.plugins import Plugins
from hydra.plugins.sweeper import Sweeper
from hydra.types import TaskFunction
from omegaconf import DictConfig, OmegaConf
from optuna.distributions import (
    BaseDistribution,
    CategoricalChoiceType,
    CategoricalDistribution,
    DiscreteUniformDistribution,
    IntLogUniformDistribution,
    IntUniformDistribution,
    LogUniformDistribution,
    UniformDistribution,
)

from .config import Direction, DistributionConfig, DistributionType

log = logging.getLogger(__name__)


def create_optuna_distribution_from_config(
    config: MutableMapping[str, Any]
) -> BaseDistribution:
    kwargs = dict(config)
    if isinstance(config["type"], str):
        kwargs["type"] = DistributionType[config["type"]]
    param = DistributionConfig(**kwargs)
    if param.type == DistributionType.categorical:
        assert param.choices is not None
        return CategoricalDistribution(param.choices)
    if param.type == DistributionType.int:
        assert param.low is not None
        assert param.high is not None
        if param.log:
            return IntLogUniformDistribution(int(param.low), int(param.high))
        step = int(param.step) if param.step is not None else 1
        return IntUniformDistribution(int(param.low), int(param.high), step=step)
    if param.type == DistributionType.float:
        assert param.low is not None
        assert param.high is not None
        if param.log:
            return LogUniformDistribution(param.low, param.high)
        if param.step is not None:
            return DiscreteUniformDistribution(param.low, param.high, param.step)
        return UniformDistribution(param.low, param.high)
    raise NotImplementedError(f"{param.type} is not supported by Optuna sweeper.")


def create_optuna_distribution_from_override(override: Override) -> Any:
    value = override.value()
    if not override.is_sweep_override():
        return value

    choices: List[CategoricalChoiceType] = []
    if override.is_choice_sweep():
        assert isinstance(value, ChoiceSweep)
        for x in override.sweep_iterator(transformer=Transformer.encode):
            assert isinstance(
                x, (str, int, float, bool)
            ), f"A choice sweep expects str, int, float, or bool type. Got {type(x)}."
            choices.append(x)
        return CategoricalDistribution(choices)

    if override.is_range_sweep():
        assert isinstance(value, RangeSweep)
        assert value.start is not None
        assert value.stop is not None
        if value.shuffle:
            for x in override.sweep_iterator(transformer=Transformer.encode):
                assert isinstance(
                    x, (str, int, float, bool)
                ), f"A choice sweep expects str, int, float, or bool type. Got {type(x)}."
                choices.append(x)
            return CategoricalDistribution(choices)
        return IntUniformDistribution(
            int(value.start), int(value.stop), step=int(value.step)
        )

    if override.is_interval_sweep():
        assert isinstance(value, IntervalSweep)
        assert value.start is not None
        assert value.end is not None
        if "log" in value.tags:
            if isinstance(value.start, int) and isinstance(value.end, int):
                return IntLogUniformDistribution(int(value.start), int(value.end))
            return LogUniformDistribution(value.start, value.end)
        else:
            if isinstance(value.start, int) and isinstance(value.end, int):
                return IntUniformDistribution(value.start, value.end)
            return UniformDistribution(value.start, value.end)

    raise NotImplementedError(f"{override} is not supported by Optuna sweeper.")


class CustomOptunaSweeperImpl(Sweeper):
    def __init__(
        self,
        sampler: Any,
        direction: Any,
        storage: Optional[str],
        study_name: Optional[str],
        max_trials: int,
        n_jobs: int,
        max_duration_minutes: int,
        # max_repeats_prune: int,
        min_trials_per_param: int,
        max_trials_per_param: int,
        search_space: Optional[DictConfig],
    ) -> None:
        self.sampler = sampler
        self.direction = direction
        self.storage = storage
        self.study_name = study_name
        self.max_trials = max_trials
        self.n_jobs = n_jobs
        self.max_duration_minutes = max_duration_minutes
        self.max_duration_seconds = max_duration_minutes * 60
        # self.max_repeats_prune = max_repeats_prune
        self.min_trials_per_param = min_trials_per_param
        self.max_trials_per_param = max_trials_per_param
        self.search_space = {}
        # self.percentile_pruner = pruners.PercentilePruner(25.0, n_startup_trials=5, n_warmup_steps=30, interval_steps=10)
        self.param_repeat_pruner = None
        # base_pruner = pruners.BasePruner
        self.pruners = []
        if search_space:
            assert isinstance(search_space, DictConfig)
            self.search_space = {
                str(x): create_optuna_distribution_from_config(y)
                for x, y in search_space.items()
            }
        self.job_idx: int = 0
        self.jobs_running = 0
        # if self.n_jobs > 1:
        # if True:
        #     self.proc_pool = mp.Pool(processes=self.n_jobs)

    def setup(
        self,
        config: DictConfig,
        config_loader: ConfigLoader,
        task_function: TaskFunction,
    ) -> None:
        self.job_idx = 0
        self.config = config
        self.config_loader = config_loader
        self.launcher = Plugins.instance().instantiate_launcher(
            config=config, config_loader=config_loader, task_function=task_function
        )
        self.sweep_dir = config.hydra.sweep.dir

    def run_trial(self, trial, params):
        launcher_arg = [tuple(f"{name}={val}" for name, val in params.items())]
        ret = self.launcher.launch(launcher_arg, initial_job_idx=self.job_idx)[0]
        try:
            value = float(ret.return_value)
        except (ValueError, TypeError):
            raise ValueError(
                f"Return value must be float-castable. Got '{ret.return_value}'."
            ).with_traceback(sys.exc_info()[2])
        return value

    def cb_async_trial(self, arg):
        print(arg)
        self.jobs_running -= 1
        return True

    def start_trial_async(self, trial, params):
        self.jobs_running += 1
        launcher_arg = [tuple(f"{name}={val}" for name, val in params.items())]
        ret = self.launcher.launch(launcher_arg, initial_job_idx=self.job_idx)[0]
        # ret = self.proc_pool.apply_async(self.launcher.launch, (launcher_arg), {'initial_job_idx': self.job_idx}, callback=self.cb_async_trial)
        # ret = ret.get()
        # ret = self.proc_pool.apply(self.launcher.launch, (launcher_arg), {'initial_job_idx': self.job_idx})
        return True

    def sweep_single_proc(self, arguments: List[str]) -> None:
        assert self.config is not None
        assert self.launcher is not None
        assert self.job_idx is not None

        parser = OverridesParser.create()
        parsed = parser.parse_overrides(arguments)

        search_space = dict(self.search_space)
        fixed_params = dict()
        for override in parsed:
            value = create_optuna_distribution_from_override(override)
            if isinstance(value, BaseDistribution):
                search_space[override.get_key_element()] = value
            else:
                fixed_params[override.get_key_element()] = value
        # Remove fixed parameters from Optuna search space.
        for param_name in fixed_params:
            if param_name in search_space:
                del search_space[param_name]

        directions: List[str]
        if isinstance(self.direction, MutableSequence):
            assert False, "Multi objectives optimization not implemented / tested"
        else:
            if isinstance(self.direction, str):
                directions = [self.direction]
            else:
                directions = [self.direction.name]

        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            sampler=self.sampler,
            directions=directions,
            load_if_exists=True
        )

        self.param_repeat_pruner = ParamRepeatPruner(study, max_runs=self.max_trials_per_param,
                                                     should_compare_states=[TrialState.COMPLETE, TrialState.FAIL,
                                                                            TrialState.PRUNED])
        log.info(f"Study name: {study.study_name}")
        log.info(f"Storage: {self.storage}")
        log.info(f"Sampler: {type(self.sampler).__name__}")
        log.info(f"Directions: {directions}")

        n_trials_to_go = self.max_trials
        start_time = time.time()
        current_time = start_time
        while n_trials_to_go > 0 and (start_time + self.max_duration_seconds) > current_time:
            running_duration = (current_time - start_time)
            log.info(f"Hyperparameter optimization is now running for {str(datetime.timedelta(seconds=running_duration))} of {str(datetime.timedelta(seconds=self.max_duration_seconds))}. Max. {n_trials_to_go} trials left.")
            trial = study._ask()
            for param_name, distribution in search_space.items():
                trial._suggest(param_name, distribution)
            params = dict(trial.params)
            params.update(fixed_params)
            total_param_runs, repeated_trial_idx = self.param_repeat_pruner.check_params(trial, prune_existing=False)
            if total_param_runs > self.max_trials_per_param:
                log.info(
                    f"Parameters {params} have been tested or pruned {total_param_runs} times in trial {repeated_trial_idx} already, pruning this trial.")
                state = optuna.trial.TrialState.PRUNED
                study._tell(trial, state, None)
                continue
            try:
                value = self.run_trial(trial, params)
                state = optuna.trial.TrialState.COMPLETE
            except Exception as e:
                log.info(f"Could not run trial {params}, returning FAIL.")
                value = None
                state = optuna.trial.TrialState.FAIL

            study._tell(trial, state, [value])

            if total_param_runs < self.min_trials_per_param: # Add repetition of the same trial for next study._ask()
                study.enqueue_trial(params)

            self.job_idx += 1
            current_time = time.time()
            n_trials_to_go -= 1

        results_to_serialize: Dict[str, Any]
        assert len(directions) < 2, "Multi objective optimization is not implemented"
        best_trial = study.best_trial
        results_to_serialize = {
            "name": "optuna",
            "best_params": best_trial.params,
            "best_value": best_trial.value,
        }
        log.info(f"Best parameters: {best_trial.params}")
        log.info(f"Best value: {best_trial.value}")
        OmegaConf.save(
            OmegaConf.create(results_to_serialize),
            f"{self.config.hydra.sweep.dir}/optimization_results.yaml",
        )
        df = study.trials_dataframe()
        df.to_csv("tmp_trials.csv", index=False)

    def sweep(self, arguments: List[str]) -> None:
        assert self.config is not None
        assert self.launcher is not None
        assert self.job_idx is not None

        parser = OverridesParser.create()
        parsed = parser.parse_overrides(arguments)

        search_space = dict(self.search_space)
        fixed_params = dict()
        for override in parsed:
            value = create_optuna_distribution_from_override(override)
            if isinstance(value, BaseDistribution):
                search_space[override.get_key_element()] = value
            else:
                fixed_params[override.get_key_element()] = value
        # Remove fixed parameters from Optuna search space.
        for param_name in fixed_params:
            if param_name in search_space:
                del search_space[param_name]

        directions: List[str]
        if isinstance(self.direction, MutableSequence):
            assert False, "Multi objectives optimization not implemented / tested"
        else:
            if isinstance(self.direction, str):
                directions = [self.direction]
            else:
                directions = [self.direction.name]

        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            sampler=self.sampler,
            directions=directions,
            load_if_exists=True
        )

        self.param_repeat_pruner = ParamRepeatPruner(study, max_runs=self.max_trials_per_param,
                                                     should_compare_states=[TrialState.COMPLETE, TrialState.FAIL,
                                                                            TrialState.PRUNED])
        log.info(f"Study name: {study.study_name}")
        log.info(f"Storage: {self.storage}")
        log.info(f"Sampler: {type(self.sampler).__name__}")
        log.info(f"Directions: {directions}")

        n_trials_to_go = self.max_trials
        start_time = time.time()
        current_time = start_time
        while n_trials_to_go > 0 and (start_time + self.max_duration_seconds) > current_time:
            running_duration = (current_time - start_time)
            log.info(f"Hyperparameter optimization is now running for {str(datetime.timedelta(seconds=running_duration))} of {str(datetime.timedelta(seconds=self.max_duration_seconds))}. Max. {n_trials_to_go} trials left.")
            enqueued_param_runs = 0
            batch_size = min(n_trials_to_go, self.n_jobs)
            overrides = []
            trials = []
            while len(overrides) < batch_size:
                trial = study._ask()
                for param_name, distribution in search_space.items():
                    trial._suggest(param_name, distribution)
                params = dict(trial.params)
                params.update(fixed_params)
                total_param_runs, repeated_trial_idx = self.param_repeat_pruner.check_params(trial, prune_existing=False)
                total_and_enqueued_param_runs = total_param_runs + enqueued_param_runs
                if total_and_enqueued_param_runs > self.max_trials_per_param:
                    log.info(
                        f"Parameters {params} have been tested or pruned {total_param_runs} times in trial {repeated_trial_idx} already, pruning this trial.")
                    state = optuna.trial.TrialState.PRUNED
                    study._tell(trial, state, None)
                    continue
                overrides.append(tuple(f"{name}={val}" for name, val in params.items()))
                trials.append(trial)

                if total_and_enqueued_param_runs < self.min_trials_per_param: # Add repetition of the same trial for next study._ask()
                    study.enqueue_trial(params)
                    enqueued_param_runs += 1
                else:
                    enqueued_param_runs = 0

            returns = self.launcher.launch(overrides, initial_job_idx=self.job_idx)
            self.job_idx += len(returns)
            for trial, ret in zip(trials, returns):
                values: Optional[List[float]] = None
                state: optuna.trial.TrialState = optuna.trial.TrialState.COMPLETE
                try:
                    if len(directions) == 1:
                        try:
                            values = [float(ret.return_value)]
                        except (ValueError, TypeError):
                            raise ValueError(
                                f"Return value must be float-castable. Got '{ret.return_value}'."
                            ).with_traceback(sys.exc_info()[2])
                    else:
                        try:
                            values = [float(v) for v in ret.return_value]
                        except (ValueError, TypeError):
                            raise ValueError(
                                "Return value must be a list or tuple of float-castable values."
                                f" Got '{ret.return_value}'."
                            ).with_traceback(sys.exc_info()[2])
                        if len(values) != len(directions):
                            raise ValueError(
                                "The number of the values and the number of the objectives are"
                                f" mismatched. Expect {len(directions)}, but actually {len(values)}."
                            )
                    study._tell(trial, state, values)
                except Exception as e:
                    state = optuna.trial.TrialState.FAIL
                    study._tell(trial, state, values)
                    raise e

            n_trials_to_go -= batch_size
            current_time = time.time()


        results_to_serialize: Dict[str, Any]
        assert len(directions) < 2, "Multi objective optimization is not implemented"
        best_trial = study.best_trial
        results_to_serialize = {
            "name": "optuna",
            "best_params": best_trial.params,
            "best_value": best_trial.value,
        }
        log.info(f"Best parameters: {best_trial.params}")
        log.info(f"Best value: {best_trial.value}")
        OmegaConf.save(
            OmegaConf.create(results_to_serialize),
            f"{self.config.hydra.sweep.dir}/optimization_results.yaml",
        )
        df = study.trials_dataframe()
        df.to_csv("tmp_trials.csv", index=False)



    def sweep_multiproc_async(self, arguments: List[str]) -> None:
        assert self.config is not None
        assert self.launcher is not None
        assert self.job_idx is not None

        parser = OverridesParser.create()
        parsed = parser.parse_overrides(arguments)

        search_space = dict(self.search_space)
        fixed_params = dict()
        for override in parsed:
            value = create_optuna_distribution_from_override(override)
            if isinstance(value, BaseDistribution):
                search_space[override.get_key_element()] = value
            else:
                fixed_params[override.get_key_element()] = value
        # Remove fixed parameters from Optuna search space.
        for param_name in fixed_params:
            if param_name in search_space:
                del search_space[param_name]

        directions: List[str]
        if isinstance(self.direction, MutableSequence):
            assert False, "Multi objectives optimization not implemented / tested"
        else:
            if isinstance(self.direction, str):
                directions = [self.direction]
            else:
                directions = [self.direction.name]

        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            sampler=self.sampler,
            directions=directions,
            load_if_exists=True
        )

        self.param_repeat_pruner = ParamRepeatPruner(study, max_runs=self.max_trials_per_param,
                                                     should_compare_states=[TrialState.COMPLETE, TrialState.FAIL,
                                                                            TrialState.PRUNED])
        log.info(f"Study name: {study.study_name}")
        log.info(f"Storage: {self.storage}")
        log.info(f"Sampler: {type(self.sampler).__name__}")
        log.info(f"Directions: {directions}")

        n_trials_to_go = self.max_trials
        start_time = time.time()
        current_time = start_time
        self.jobs_running = 0
        while n_trials_to_go > 0 and (start_time + self.max_duration_seconds) > current_time:
            running_duration = (current_time - start_time)
            log.info(f"Hyperparameter optimization is now running for {str(datetime.timedelta(seconds=running_duration))} of {str(datetime.timedelta(seconds=self.max_duration_seconds))}. Max. {n_trials_to_go} trials left.")
            trial = study._ask()
            for param_name, distribution in search_space.items():
                trial._suggest(param_name, distribution)
            params = dict(trial.params)
            params.update(fixed_params)
            total_param_runs, repeated_trial_idx = self.param_repeat_pruner.check_params(trial, prune_existing=False)
            if total_param_runs > self.max_trials_per_param:
                log.info(
                    f"Parameters {params} have been tested or pruned {total_param_runs} times in trial {repeated_trial_idx} already, pruning this trial.")
                state = optuna.trial.TrialState.PRUNED
                study._tell(trial, state, None)
                continue
            # if self.n_jobs == 1:
            if False:
                try:
                    self.run_trial(trial,params)
                    value = self.run_trial(trial, params)
                    state = optuna.trial.TrialState.COMPLETE
                except Exception as e:
                    log.info(f"Could not start trial {params}, returning FAIL.")
                    value = None
                    state = optuna.trial.TrialState.FAIL

                study._tell(trial, state, [value])
            else:
                while self.jobs_running >= self.n_jobs:
                    time.sleep(10)
                try:
                    self.start_trial_async(trial, params) # will tell the study automatically if it has started.
                except Exception as e:
                    log.info(f"Could not start trial {params}, returning FAIL. \n{e}")
                    value = None
                    state = optuna.trial.TrialState.FAIL
                    study._tell(trial, state, [value])


            if total_param_runs < self.min_trials_per_param: # Add repetition of the same trial for next study._ask()
                study.enqueue_trial(params)

            self.job_idx += 1
            current_time = time.time()
            n_trials_to_go -= 1

        results_to_serialize: Dict[str, Any]
        assert len(directions) < 2, "Multi objective optimization is not implemented"
        best_trial = study.best_trial
        results_to_serialize = {
            "name": "optuna",
            "best_params": best_trial.params,
            "best_value": best_trial.value,
        }
        log.info(f"Best parameters: {best_trial.params}")
        log.info(f"Best value: {best_trial.value}")
        OmegaConf.save(
            OmegaConf.create(results_to_serialize),
            f"{self.config.hydra.sweep.dir}/optimization_results.yaml",
        )
        df = study.trials_dataframe()
        df.to_csv("tmp_trials.csv", index=False)
