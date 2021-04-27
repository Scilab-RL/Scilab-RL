import optuna
from optuna.trial import TrialState
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict


class ParamRepeatPruner:
    """Prunes reapeated trials, which means trials with the same paramters won't waste time/resources."""

    def __init__(
        self,
        study: optuna.study.Study,
        repeats_max: int = 0,
        should_compare_states: List[TrialState] = [TrialState.COMPLETE],
        compare_unfinished: bool = True,
    ):
        """
        Args:
            study (optuna.study.Study): Study of the trials.

            repeats_max (int, optional): Instead of prunning all of them (not repeating trials at all, repeats_max=0) you can choose to repeat them up to a certain number of times, useful if your optimization function is not deterministic and gives slightly different results for the same params. Defaults to 0.

            should_compare_states (List[TrialState], optional): By default it only skips the trial if the paremeters are equal to existing COMPLETE trials, so it repeats possible existing FAILed and PRUNED trials. If you also want to skip these trials then use [TrialState.COMPLETE,TrialState.FAIL,TrialState.PRUNED] for example. Defaults to [TrialState.COMPLETE].

            compare_unfinished (bool, optional): Unfinished trials (e.g. `RUNNING`) are treated like COMPLETE ones, if you don't want this behavior change this to False. Defaults to True.
        """
        self.should_compare_states = should_compare_states
        self.repeats_max = repeats_max
        self.repeats: Dict[int, List[int]] = defaultdict(lambda: [], {})
        self.unfinished_repeats: Dict[int, List[int]] = defaultdict(lambda: [], {})
        self.compare_unfinished = compare_unfinished
        self.study = study

    @property
    def study(self) -> Optional[optuna.study.Study]:
        return self._study

    @study.setter
    def study(self, study):
        self._study = study
        if self.study is not None:
            self.register_existing_trials()

    def register_existing_trials(self):
        """In case of studies with existing trials, it counts existing repeats"""
        trials = study.trials
        trial_n = len(trials)
        for trial_idx, trial_past in enumerate(study.trials[1:]):
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
        if self.study is None:
            return
        trials = self.study.trials
        if trial is None:
            trial = trials[-1]
            ignore_last_trial = -1

        self.clean_unfinised_trials()

        self.repeated_idx = -1
        self.repeated_number = -1
        for idx_p, trial_past in enumerate(trials[:ignore_last_trial]):
            should_compare = self.should_compare(trial_past.state)
            should_compare |= (
                self.compare_unfinished and not trial_past.state.is_finished()
            )
            if should_compare and trial.params == trial_past.params:
                if not trial_past.state.is_finished():
                    self.unfinished_repeats[trial_past.number].append(trial.number)
                    continue
                self.repeated_idx = idx_p
                self.repeated_number = trial_past.number
                break

        if self.repeated_number > -1:
            self.repeats[self.repeated_number].append(trial.number)
        if len(self.repeats[self.repeated_number]) > self.repeats_max:
            if prune_existing:
                raise optuna.exceptions.TrialPruned()

        return self.repeated_number

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
    # By default it only skips the trial if the paremeters are equal to existing COMPLETE trials, so it repeats possible existing FAILed and PRUNED trials. If you also want to skip these trials then just declare it like so:
    # prune_params = ParamRepeatPruner(study, should_compare_states=[TrialState.COMPLETE,TrialState.FAIL,TrialState.PRUNED])
    # Check the constructor docstring for more information

    def dummy_objective(trial: optuna.trial.Trial):
        trial.suggest_int("dummy_param-0", 1, 20)
        # Check parameters with the pruner
        repeated = prune_params.check_params()
        # # Instead of prunning you can return a mean of previous values, useful if allowing some repeats to happen in non deterministic objective functions
        # repeated = prune_params.check_params(prune_existing=False)
        # if repeated > -1:
        #     print("repeated")
        #     return prune_params.get_value_of_repeats(repeated)

        return trial.params["dummy_param-0"]

    study.optimize(dummy_objective, n_trials=40)

    df = study.trials_dataframe()
    df.to_csv("tmp_trials.csv", index=False)