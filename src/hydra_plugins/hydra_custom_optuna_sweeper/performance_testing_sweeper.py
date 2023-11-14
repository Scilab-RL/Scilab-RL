import datetime
import time
import logging
from typing import List
from hydra.core.plugins import Plugins
from hydra.plugins.sweeper import Sweeper
from hydra.types import TaskFunction, HydraContext
from omegaconf import DictConfig
from mlflow.tracking import MlflowClient


log = logging.getLogger(__name__)


class PerformanceTestingSweeper(Sweeper):
    """
    Run performance testing for an environment-algorithm combination. The conditions for a performance test are stored
    in conf/performance/<ENV>/<OPTIONAL_ENV_SUBTYPE>/<OPTIONAL_ENV_CONFIG>-<ALGO>-test.yaml
    You can for example run: "python main.py +performance=FetchReach/her-test --multirun"
    The joblib launcher allows to run n_jobs in parallel.
    YOU CANNOT run multiple performance tests by simply providing multiple configs separated by commas, for example:
    "python main.py +performance=FetchReach/her-test,FetchSlide/sac_her-test --multirun" doesn't work.
    In that case, just call main.py twice with the different performance test configs.
    """
    def __init__(self, study_name, n_jobs, **kwargs):
        self.study_name = study_name
        self.n_jobs = n_jobs
        self.mlflow_client = MlflowClient()
        if 'max_duration_minutes' in kwargs:
            self.max_duration = 60*kwargs['max_duration_minutes']
        else:
            self.max_duration = 60*24*7  # one week

    def setup(
        self,
        config: DictConfig,
        hydra_context: HydraContext,
        task_function: TaskFunction,
    ) -> None:
        self.config = config
        self.hydra_context = hydra_context
        self.task_function = task_function
        self.launcher = Plugins.instance().instantiate_launcher(
            config=config, hydra_context=hydra_context, task_function=task_function)

    def sweep(self, arguments: List[str]) -> None:
        for arg in arguments:
            if arg.startswith('+performance='):
                p_test_name = arg[13:]
        p_test_cond = self.config['performance_testing_conditions']
        n_runs_left = p_test_cond['total_runs']
        needed_succ_runs = p_test_cond['succ_runs']
        succ_runs = 0
        eval_col = p_test_cond['eval_columns']
        eval_val = p_test_cond['eval_value']
        arguments.append(f"n_epochs={int(p_test_cond['max_steps'] / self.config['eval_after_n_steps'])}")

        job_idx = 0
        start_time = time.perf_counter()
        current_time = start_time

        # The following is the main loop that starts all runs
        while n_runs_left > 0 and (start_time + self.max_duration) > current_time:
            running_duration = (current_time - start_time)
            log.info(
                f"Performance testing runs for {str(datetime.timedelta(seconds=running_duration))} of "
                f"{str(datetime.timedelta(seconds=self.max_duration))}. Max. {n_runs_left} runs left.")

            batch_size = min(n_runs_left, self.n_jobs)

            returns = self.launcher.launch([tuple(arguments)]*batch_size, initial_job_idx=job_idx)

            job_idx += batch_size
            n_runs_left -= batch_size

            # Use mlflow to get the value in eval_col and check whether the test succeeded.
            for ret in returns:
                run_id = ret.return_value[2]
                run = self.mlflow_client.get_run(run_id)
                succ_runs += int(run.data.metrics[eval_col] >= eval_val)

            if succ_runs >= needed_succ_runs:
                log.info(f"Performance test {p_test_name} successful! The value for {eval_col} "
                         f"was at least {eval_val} in {succ_runs} runs.")
                return
            if n_runs_left < needed_succ_runs - succ_runs:
                message = f"Performance test {p_test_name} failed. "\
                          "The needed number of successful runs could not be achieved."
                log.info(message)
                assert False, message

            current_time = time.perf_counter()

        if needed_succ_runs > succ_runs and (start_time + self.max_duration) < current_time:
            message = "The needed number of successful runs could not be achieved within "\
                      f"{int(self.max_duration/60)} minutes for performance test {p_test_name}."
            log.info(message)
            assert False, message
