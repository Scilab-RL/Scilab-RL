# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


class DistributionType(Enum):
    int = 1
    float = 2
    categorical = 3


class Direction(Enum):
    minimize = 1
    maximize = 2


@dataclass
class SamplerConfig:
    _target_: str = MISSING
    seed: Optional[int] = None


@dataclass
class TPESamplerConfig(SamplerConfig):
    """
    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html
    """

    _target_: str = "optuna.samplers.TPESampler"

    consider_prior: bool = True
    prior_weight: float = 1.0
    consider_magic_clip: bool = True
    consider_endpoints: bool = False
    n_startup_trials: int = 10
    n_ei_candidates: int = 24
    multivariate: bool = False
    warn_independent_sampling: bool = True


@dataclass
class RandomSamplerConfig(SamplerConfig):
    """
    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.RandomSampler.html
    """

    _target_: str = "optuna.samplers.RandomSampler"


@dataclass
class CmaEsSamplerConfig(SamplerConfig):
    """
    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.CmaEsSampler.html
    """

    _target_: str = "optuna.samplers.CmaEsSampler"

    x0: Optional[Dict[str, Any]] = None
    sigma0: Optional[float] = None
    independent_sampler: Optional[Any] = None
    warn_independent_sampling: bool = True
    consider_pruned_trials: bool = False
    restart_strategy: Optional[Any] = None
    inc_popsize: int = 2
    use_separable_cma: bool = False
    source_trials: Optional[Any] = None


@dataclass
class NSGAIISamplerConfig(SamplerConfig):
    """
    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.NSGAIISampler.html
    """

    _target_: str = "optuna.samplers.NSGAIISampler"

    population_size: int = 50
    mutation_prob: Optional[float] = None
    crossover_prob: float = 0.9
    swapping_prob: float = 0.5
    constraints_func: Optional[Any] = None


@dataclass
class MOTPESamplerConfig(SamplerConfig):
    """
    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.MOTPESampler.html
    """

    _target_: str = "optuna.samplers.MOTPESampler"

    consider_prior: bool = True
    prior_weight: float = 1.0
    consider_magic_clip: bool = True
    consider_endpoints: bool = False
    n_startup_trials: int = 10
    n_ehvi_candidates: int = 24


@dataclass
class DistributionConfig:

    # Type of distribution. "int", "float" or "categorical"
    type: DistributionType

    # Choices of categorical distribution
    # List element type should be Union[str, int, float, bool]
    choices: Optional[List[Any]] = None

    # Lower bound of int or float distribution
    low: Optional[float] = None

    # Upper bound of int or float distribution
    high: Optional[float] = None

    # If True, space is converted to the log domain
    # Valid for int or float distribution
    log: bool = False

    # Discretization step
    # Valid for int or float distribution
    step: Optional[float] = None


defaults = [{"sampler": "tpe"}]


@dataclass
class OptunaSweeperConf:
    _target_: str = "hydra_plugins.hydra_custom_optuna_sweeper.custom_optuna_sweeper.CustomOptunaSweeper"
    defaults: List[Any] = field(default_factory=lambda: defaults)

    # Sampling algorithm
    # Please refer to the reference for further details
    # https://optuna.readthedocs.io/en/stable/reference/samplers.html
    sampler: SamplerConfig = MISSING

    # Direction of optimization
    # Union[Direction, List[Direction]]
    direction: Any = Direction.minimize

    # Storage URL to persist optimization results
    # For example, you can use SQLite if you set 'sqlite:///example.db'
    # Please refer to the reference for further details
    # https://optuna.readthedocs.io/en/stable/reference/storages.html
    # if no storage name is provided, it defaults to 'sqlite:///study_name.db'
    storage: Optional[str] = None

    # Name of study to persist optimization results
    study_name: Optional[str] = None

    # Max number of function evaluations. There may be less function evaluations when trials are pruned
    # because the parameterization has already been tested (see max_trials_per_param below).
    max_trials: int = 20

    # Number of parallel workers
    n_jobs: int = 1

    # Max. duration in minutes for hyperopt
    max_duration_minutes: int = 1440

    # Min and max. number of trials per parameterization. min_trials is important for non-deterministic processes,
    # max_trials is to prune parameterizations if they occur too often.
    min_trials_per_param: int = 1
    max_trials_per_param: int = 3

    search_space: Dict[str, Any] = field(default_factory=dict)


ConfigStore.instance().store(
    group="hydra/sweeper",
    name="optuna",
    node=OptunaSweeperConf,
    provider="custom_optuna_sweeper",
)

ConfigStore.instance().store(
    group="hydra/sweeper/sampler",
    name="tpe",
    node=TPESamplerConfig,
    provider="custom_optuna_sweeper",
)

ConfigStore.instance().store(
    group="hydra/sweeper/sampler",
    name="random",
    node=RandomSamplerConfig,
    provider="custom_optuna_sweeper",
)

ConfigStore.instance().store(
    group="hydra/sweeper/sampler",
    name="cmaes",
    node=CmaEsSamplerConfig,
    provider="custom_optuna_sweeper",
)

ConfigStore.instance().store(
    group="hydra/sweeper/sampler",
    name="nsgaii",
    node=NSGAIISamplerConfig,
    provider="custom_optuna_sweeper",
)

ConfigStore.instance().store(
    group="hydra/sweeper/sampler",
    name="motpe",
    node=MOTPESamplerConfig,
    provider="custom_optuna_sweeper",
)
