import optuna
from optuna.trial import TrialState
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict


class ParamRepeatPruner:
    """Prunes repeated trials, which means trials with the same parameters won't waste time/resources."""

    def __init__(
        self,
        study: optuna.study.Study,
        max_runs: int = 1,
        should_compare_states: List[TrialState] = [TrialState.COMPLETE],
        compare_unfinished: bool = True,
    ):
        """
        Args:
            study (optuna.study.Study): Study of the trials.

            max_runs (int, optional): Max. number of allowed runs per parameterization.
            (max_runs = 0 prunes all parameterizations, even if they have never been tested).

            should_compare_states (List[TrialState], optional): By default it only skips the trial if the parameters
            are equal to existing COMPLETE trials, so it repeats possible existing FAILed and PRUNED trials.
            If you also want to skip these trials then use [TrialState.COMPLETE,TrialState.FAIL,TrialState.PRUNED]
            for example. Defaults to [TrialState.COMPLETE].

            compare_unfinished (bool, optional): Unfinished trials (e.g. `RUNNING`) are treated like COMPLETE ones,
            if you don't want this behavior change this to False. Defaults to True.
        """
        self.should_compare_states = should_compare_states
        self.max_runs = max_runs
        self.repeats: Dict[int, List[int]] = defaultdict(lambda: [], {})
        self.unfinished_repeats: Dict[int, List[int]] = defaultdict(lambda: [], {})
        self.compare_unfinished = compare_unfinished
        self.study = study
        self.register_existing_trials()

    def register_existing_trials(self):
        """In case of studies with existing trials, it counts existing repeats"""
        trials = self.study.trials
        trial_n = len(trials)
        for trial_idx, trial_past in enumerate(trials[1:]):
            self.check_params(trial_past, False, -trial_n + trial_idx)

    def prune(self):
        self.check_params()

    def should_compare(self, state):
        return any(state == state_comp for state_comp in self.should_compare_states)

    def clean_unfinised_trials(self):
        trials = self.study.trials
        finished = []
        for key, value in self.unfinished_repeats.items():
            if self.should_compare(trials[key].state):
                for t in value:
                    self.repeats[key].append(t)
                finished.append(key)

        for f in finished:
            del self.unfinished_repeats[f]

    def check_params(
        self,
        trial: Optional[optuna.trial.BaseTrial] = None,
        prune_existing=True,
        ignore_last_trial: Optional[int] = None,
    ):
        """
        Check if parameterization has been executed already. If so, return number of previous executions
        and a trial id of a previous exeuction.
        Args:
            trial:
            prune_existing:
            ignore_last_trial:

        Returns:

        """
        if self.study is None:
            return
        trials = self.study.trials
        if trial is None:
            trial = trials[-1]
            ignore_last_trial = -1

        self.clean_unfinised_trials()

        self.repeated_number = -1
        for trial_past in trials[:ignore_last_trial]:
            should_compare = self.should_compare(trial_past.state)
            should_compare |= (
                self.compare_unfinished and not trial_past.state.is_finished()
            )
            if should_compare and trial.params == trial_past.params:
                if not trial_past.state.is_finished():
                    self.unfinished_repeats[trial_past.number].append(trial.number)
                    continue
                self.repeated_number = trial_past.number
                break

        past_param_runs = len(
            self.repeats[self.repeated_number])
        now_param_runs = past_param_runs + 1
        if self.repeated_number > -1:
            self.repeats[self.repeated_number].append(trial.number)
        if now_param_runs > self.max_runs:
            if prune_existing:
                raise optuna.exceptions.TrialPruned()

        return now_param_runs, self.repeated_number

    def get_value_of_repeats(
        self, repeated_number: int, func=lambda value_list: np.mean(value_list)
    ):
        if self.study is None:
            raise ValueError("No study registered.")
        trials = self.study.trials
        values = (
            trials[repeated_number].value,
            *(
                trials[tn].value
                for tn in self.repeats[repeated_number]
                if trials[tn].value is not None
            ),
        )
        return func(values)


if __name__ == "__main__":
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=42), direction="minimize"
    )
    # Create "Pruner"
    prune_params = ParamRepeatPruner(study)
    # By default, it only skips the trial if the parameters are equal to existing COMPLETE trials, so it repeats
    # possible existing FAILed and PRUNED trials. If you also want to skip these trials then just declare it like so:
    # prune_params = ParamRepeatPruner(study, should_compare_states=[TrialState.COMPLETE,TrialState.FAIL,TrialState.PRUNED])
    # Check the constructor docstring for more information

    def dummy_objective(trial: optuna.trial.Trial):
        trial.suggest_int("dummy_param-0", 1, 20)
        return trial.params["dummy_param-0"]

    study.optimize(dummy_objective, n_trials=40)

    df = study.trials_dataframe()
    df.to_csv("tmp_trials.csv", index=False)
