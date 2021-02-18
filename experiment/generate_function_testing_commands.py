import getpass
from testing_algos import TestingAlgos
from testing_envs import TestingEnvs

"""
This script is used for testing backwards compatibility after adding a new feature.
If you want to merge your development branch with the overall devel branch, please proceed as described in README.md file
"""

if __name__ == "__main__":
    cmds = []

    eval_after_n_actions = 100
    n_test_rollouts = 2
    n_epochs = 2
    rollout_batch_size = 1
    n_cpu = 1

    whoami = getpass.getuser()

    opts_values = {"general": {}}
    opts_values["general"]['n_epochs'] = n_epochs
    opts_values["general"]['eval_after_n_actions'] = eval_after_n_actions
    opts_values["general"]['n_test_rollouts'] = n_test_rollouts
    opts_values["general"]['base_logdir'] = "/data/" + whoami + "/baselines"
    opts_values["general"]['render'] = 0
    opts_values["general"]['try_start_idx'] = 100
    opts_values["general"]['max_try_idx'] = 500

    base_cmd = "python3 experiment/train.py"
    for k, v in sorted(opts_values["general"].items()):
        base_cmd += " --{}".format(k) + " {}".format(str(v))

    get_alg_cmd_functions = {}
    for alg in TestingAlgos.algo_names:
        get_alg_cmd_functions[alg] = eval("TestingAlgos.get_{}_function_testing_cmds".format(alg))

    for env in TestingEnvs.env_names:
        env_base_cmd = base_cmd + " --env {}".format(env)
        extra = ''
        if 'CopReacherEnv' in env:
            # necessary for running on servers
            extra = 'xvfb-run -a '
        for alg in TestingAlgos.algo_names:
            cmds += [extra + env_base_cmd + _cmd for _cmd in get_alg_cmd_functions[alg]()]

    cmd_file_name = "function_test_cmds.txt"
    with open(cmd_file_name, "w") as cmd_file:
        for cmd in cmds:
            cmd_file.write(cmd + "\n")
        cmd_file.close()
    print("Done generating debug commands")