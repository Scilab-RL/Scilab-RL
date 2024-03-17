# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import datetime
import logging
import os
import time
from typing import Any, Dict, List, MutableMapping, MutableSequence, Optional

import optuna
from hydra.core.utils import JobStatus
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.override_parser.types import (
    ChoiceSweep,
    IntervalSweep,
    Override,
    RangeSweep,
    Transformer,
)
from hydra.core.plugins import Plugins
from hydra.plugins.sweeper import Sweeper
from hydra.types import TaskFunction, HydraContext
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
from optuna.trial import TrialState
from optuna.visualization import plot_contour
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances

from hydra_plugins.hydra_custom_optuna_sweeper.param_repeat_pruner import ParamRepeatPruner
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
        min_trials_per_param: int,
        max_trials_per_param: int,
        search_space: Optional[DictConfig],
    ) -> None:
        self.sampler = sampler
        self.direction = direction
        if storage is None:
            self.storage = f"sqlite:///{study_name}.db"
        else:
            self.storage = storage
        self.study_name = study_name
        self.max_trials = max_trials
        self.n_jobs = n_jobs
        self.max_duration_minutes = max_duration_minutes
        self.max_duration_seconds = max_duration_minutes * 60
        self.min_trials_per_param = min_trials_per_param
        self.max_trials_per_param = max_trials_per_param
        self.search_space = {}
        self.param_repeat_pruner = None
        self.pruners = []
        if search_space:
            search_space = OmegaConf.to_container(search_space)
            self.search_space = {
                str(x): create_optuna_distribution_from_config(y)
                for x, y in search_space.items()
            }
        self.job_idx: int = 0
        self.jobs_running = 0
        self.del_to_stop_fname = "delete_me_to_stop_hyperopt"
        with open(self.del_to_stop_fname, 'w') as f:
            f.write("Delete this file to stop the hyperparameter optimization after the current batch of jobs.")

        self.metric_to_check_for_early_stop = 'time/total timesteps'

    def setup(
        self,
        config: DictConfig,
        hydra_context: HydraContext,
        task_function: TaskFunction,
    ) -> None:
        self.job_idx = 0
        self.config = config
        self.config_loader = hydra_context.config_loader
        self.launcher = Plugins.instance().instantiate_launcher(
            config=config, hydra_context=hydra_context, task_function=task_function
        )
        self.sweep_dir = config.hydra.sweep.dir

    def sweep(self, arguments: List[str]) -> None:
        assert self.config is not None
        assert self.launcher is not None
        assert self.job_idx is not None

        if 'smoke_test' in self.config and self.config.smoke_test:
            return self.smoke_test(arguments)

        parser = OverridesParser.create()
        parsed = parser.parse_overrides(arguments)

        search_space = dict(self.search_space)
        fixed_params = {}
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
            directions = [
                d.name if isinstance(d, Direction) else d for d in self.direction
            ]
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

        if 'max_n_epochs' not in study.user_attrs.keys():
            study.set_user_attr("max_n_epochs", self.config.n_epochs)

        n_trials_to_go = self.max_trials
        start_time = time.perf_counter()
        current_time = start_time
        while n_trials_to_go > 0 and (start_time + self.max_duration_seconds) > current_time and os.path.isfile(self.del_to_stop_fname):
            running_duration = (current_time - start_time)
            log.info(f"Hyperparameter optimization is now running for {str(datetime.timedelta(seconds=running_duration))} "
                     f"of {str(datetime.timedelta(seconds=self.max_duration_seconds))}. Max. {n_trials_to_go} trials left.")
            enqueued_param_runs = 0
            batch_size = min(n_trials_to_go, self.n_jobs)
            overrides = []
            runs_in_batch = 0
            trials = []
            max_n_epochs = min(study.user_attrs['max_n_epochs'], self.config.n_epochs)
            if max_n_epochs is not None:
                fixed_params['n_epochs'] = max_n_epochs
            while runs_in_batch < batch_size:
                trial = study.ask()
                for param_name, distribution in search_space.items():
                    trial._suggest(param_name, distribution)
                params = dict(trial.params)
                params.update(fixed_params)

                total_param_runs, repeated_trial_idx = self.param_repeat_pruner.check_params(trial, prune_existing=False)
                total_and_enqueued_param_runs = total_param_runs + enqueued_param_runs
                if total_and_enqueued_param_runs > self.max_trials_per_param:
                    log.info(
                        f"Parameters {params} have been tested or pruned {total_param_runs} times in "
                        f"trial {repeated_trial_idx} already, pruning this trial.")
                    state = optuna.trial.TrialState.PRUNED
                    study.tell(trial, None, state)
                    continue
                overrides.append(tuple(f"{name}={val}" for name, val in params.items()))
                runs_in_batch = len(overrides)
                trials.append(trial)

                # Add repetition of the same trial for next study.ask()
                if total_and_enqueued_param_runs < self.min_trials_per_param:
                    study.enqueue_trial(params)
                    enqueued_param_runs += 1
                else:
                    enqueued_param_runs = 0
                # Wait until starting next study. This helps if e.g. a temporary file is generated with a study, and the next study uses the same file name.
                time.sleep(0.2)

            returns = self.launcher.launch(overrides, initial_job_idx=self.job_idx)
            self.job_idx += len(returns)
            for trial, ret in zip(trials, returns):
                state: optuna.trial.TrialState = optuna.trial.TrialState.COMPLETE
                # try:
                assert len(
                    ret.return_value) == 3, "The return value of main() should be a triple where the first element " \
                                            "is the hyperopt score, the second value is the number of epochs the " \
                                            "script ran and the third is the mlflow run_id."
                assert len(directions) == 1, "We currently support only one optimization objective and direction."
                # try:
                values = [float(ret.return_value[0])]
                if len(ret.return_value) > 1:
                    n_epochs = int(ret.return_value[1])
                    new_max_epochs = int(n_epochs * 1.5)
                    if new_max_epochs <= study.user_attrs['max_n_epochs']:
                        log.info(f"This trial had only {n_epochs} epochs. "
                                 f"New upper limit for max. epochs is now {new_max_epochs}. ")
                        study.set_user_attr("max_n_epochs", new_max_epochs)

                study.tell(trial, values, state)

            self.plot_study_summary(study)
            n_trials_to_go -= batch_size
            current_time = time.perf_counter()

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

    def smoke_test(self, args):
        algos, envs, other_args = [], [], []
        for arg in args:
            if arg.startswith('algorithm='):
                algos = arg[10:].split(',')
            elif arg.startswith('env='):
                envs = arg[4:].split(',')
            else:
                other_args.append(arg)

        if not algos:
            algos = ['sac']
        if not envs:
            envs = ['FetchReach-v2']
        configs = []
        for a in algos:
            for e in envs:
                args_for_this_conf = other_args.copy()
                args_for_this_conf.append('algorithm='+a)
                args_for_this_conf.append('env='+e)
                configs.append(tuple(args_for_this_conf))

        job_idx = 0
        while configs:
            batch_size = min(len(configs), self.n_jobs)
            results = self.launcher.launch(configs[:batch_size], initial_job_idx=job_idx)
            for r in results:
                if r.status == JobStatus.FAILED:
                    assert False, f"Experiment with overrides {r.overrides.__str__()} " \
                                  f"failed with {r._return_value}"
            job_idx += batch_size
            configs = configs[batch_size:]

    def plot_study_summary(self, study):
        try:
            fig = plot_optimization_history(study)
            fig.write_image(f"{self.sweep_dir}/hyperopt_history.png")
        except:
            pass
        try:
            fig = plot_contour(study)
            fig.update_layout(width=3072, height=2048)
            fig.write_image(f"{self.sweep_dir}/hyperopt_contour.png")
        except:
            pass
        try:
            fig = plot_param_importances(study)
            fig.write_image(f"{self.sweep_dir}/hyperopt_param_importances.png")
        except:
            pass
