import getpass
import json
from experiment.testing_envs import TestingEnvs
from experiment.testing_algos import TestingAlgos

"""
This script is used for testing backwards compatibility after adding a new feature.
If you want to merge your development branch with the overall devel branch, please proceed as described in README.md file
For each algorithm, please add a few testing environments to the respective get_<alg_name>_params() function and specify the following five values in the performance_params dictionary for one specific hyperparameterization:
n_runs: number of runs to train
epochs: numer of epochs to train per run
performance_measure: the data column in that measures the performance. This is usually test/success_rate, but you can also select other data columns.
min_performance_value: the value to achieve on the data column for the run to be considered successful.
min_success_runs: The minimal number of runs to be successful.

Note that you can have multiple performance and hyperparameterizations for each algorithm-environment combination.
"""

def write_performance_params_json():
    algorithms = TestingAlgos.algo_names
    env_names = TestingEnvs.env_names

    env_alg_performance = {}
    for env in env_names:
        env_alg_performance[env] = []
        for alg in algorithms:
            params_list = eval("TestingAlgos.get_{}_performance_params".format(alg))(env)
            for params in params_list:
                if params is not None:
                    performance_params, hyper_params = params
                    hyper_params['n_epochs'] = performance_params['epochs']
                    env_alg_performance[env].append({'alg': alg, 'performance_params': performance_params.copy(),
                                                     'hyper_params': hyper_params.copy()})
    with open('./performance_test_logs/performance_params.json', 'w') as outfile:
        json.dump(env_alg_performance, outfile, indent=4, sort_keys=True)
    outfile.close()

def main(**kwargs):
    cmds = []

    n_train_rollouts = 100
    n_test_rollouts = 20
    n_epochs = 70
    rollout_batch_size = 1
    n_cpu = 1

    whoami = getpass.getuser()

    default_opts_values = {}
    default_opts_values['num_cpu'] = n_cpu
    default_opts_values['n_epochs'] = n_epochs
    default_opts_values['n_train_rollouts'] = n_train_rollouts
    default_opts_values['n_test_rollouts'] = n_test_rollouts
    default_opts_values['base_logdir'] = "/data/" + whoami + "/baselines"
    default_opts_values['render'] = 0
    default_opts_values['try_start_idx'] = 100

    write_performance_params_json ()

    base_cmd = "python3 experiment/train.py"

    get_params_functions = {}
    for alg in TestingAlgos.algo_names:
        get_params_functions[alg] = eval("TestingAlgos.get_{}_performance_params".format(alg))

    for env in TestingEnvs.env_names:
        env_base_cmd = base_cmd + " --env {}".format(env)
        extra = ''
        if 'CopReacherEnv' in env:
            # necessary for running on servers
            extra = 'xvfb-run -a '
        for alg in TestingAlgos.algo_names:
            params_list = get_params_functions[alg](env)
            for performance_params, hyper_params in params_list:
                # if hyper_params is None:
                #     continue
                hyper_params['n_epochs'] = performance_params['epochs']
                cmd = env_base_cmd
                for k, v in sorted(default_opts_values.items()):
                    if k not in hyper_params.keys():
                        cmd += " --{}".format(k) + " {}".format(str(v))
                cmd += " --algorithm " + str(alg)
                for param,val in hyper_params.items():
                    cmd += ' --{} {}'.format(param, val)
                cmd += ' --max_try_idx {}'.format(default_opts_values['try_start_idx'] + performance_params['n_runs'] - 1)
                for _ in range(performance_params['n_runs']):
                    cmds.append(extra + cmd)

    cmd_file_name = "performance_test_cmds.txt"
    with open(cmd_file_name, "w") as cmd_file:
        for cmd in cmds:
            cmd_file.write(cmd + "\n")
        cmd_file.close()
    print("Done generating performance testing commands")

if __name__ == "__main__":
    main()