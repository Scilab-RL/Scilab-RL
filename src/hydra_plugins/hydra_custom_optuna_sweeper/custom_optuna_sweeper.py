# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Any, List, Optional

from hydra.core.config_loader import ConfigLoader
from hydra.plugins.sweeper import Sweeper
from hydra.types import TaskFunction, HydraContext
from omegaconf import DictConfig

from .config import SamplerConfig
from ._impl import CustomOptunaSweeperImpl


class CustomOptunaSweeper(Sweeper):
    """Class to interface with Optuna"""

    def __init__(
        self,
        sampler: SamplerConfig,
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

        self.sweeper = CustomOptunaSweeperImpl(
            sampler, direction, storage, study_name, max_trials, n_jobs, max_duration_minutes,
            min_trials_per_param, max_trials_per_param, search_space
        )

    def setup(
        self,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.sweeper.setup(config, hydra_context, task_function)

    def sweep(self, arguments: List[str]) -> None:
        return self.sweeper.sweep(arguments)
