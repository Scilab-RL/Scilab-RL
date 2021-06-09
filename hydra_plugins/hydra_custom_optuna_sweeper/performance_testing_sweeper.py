import datetime
import time
import logging
from typing import List
from hydra.core.config_loader import ConfigLoader
from hydra.core.plugins import Plugins
from hydra.plugins.sweeper import Sweeper
from hydra.types import TaskFunction
from omegaconf import DictConfig
from mlflow.tracking import MlflowClient


log = logging.getLogger(__name__)


"""
only command that calls this so far: python experiment/train.py +performance=FetchReach/her-test --multirun
"""
class PerformanceTestingSweeper(Sweeper):
    def __init__(self, study_name, n_jobs, max_duration_minutes, **kwargs):
        self.study_name = study_name
        self.n_jobs = n_jobs
        self.max_duration = 60*max_duration_minutes
        self.mlflow_client = MlflowClient()

    def setup(
        self,
        config: DictConfig,
        config_loader: ConfigLoader,
        task_function: TaskFunction,
    ) -> None:
        self.config = config
        self.config_loader = config_loader
        self.task_function = task_function
        self.launcher = Plugins.instance().instantiate_launcher(
            config=config, config_loader=config_loader, task_function=task_function)

    def sweep(self, arguments: List[str]) -> None:
        # TODO add support for running multiple performance tests
        p_test_cond = self.config['performance_testing_conditions']
        n_runs_left = p_test_cond['total_runs']
        needed_succ_runs = p_test_cond['succ_runs']
        succ_runs = 0
        eval_col = p_test_cond['eval_columns']
        eval_val = p_test_cond['eval_value']

        job_idx = 0
        start_time = time.time()
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
                log.info(f"Performance test successful! The value for {eval_col} "
                         f"was at least {eval_val} in {succ_runs}")
                break
            if n_runs_left < needed_succ_runs - succ_runs:
                log.info(f"Performance test failed. The needed number of successful runs could not be achieved.")
                break

            current_time = time.time()
