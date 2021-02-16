import sys
import os
import csv
import json
from check_error_logs import test_error_logs_in_dir

def locate_data_dirs(number_files, result_dir):
    data_dirs = []
    for i in range(number_files):
        log_dir = result_dir + str(i + 1) + ".log"
        file = open(log_dir, "r")
        content = str(file.read())
        file.close()
        try:
            start = content.index('Data base dir: ') + len('Data base dir: ')
        except ValueError:
            continue
        end = content.index('\n', start)
        data_base_dir = content[start:end].rstrip()
        sub_dirs = os.listdir(data_base_dir)
        for file in sub_dirs:
            d = os.path.join(data_base_dir, file)
            if os.path.isdir(d):
                if d not in data_dirs:
                    data_dirs.append(d)
    return data_dirs

def find_data_dirs_for_params(env_name, alg_name, hyper_params, data_dirs):
    target_dirs = []
    for dir_counter, current_dir in enumerate(data_dirs):
        with open(current_dir+'/params.json', 'r') as jsonparamsfile:
            dir_params = json.load(jsonparamsfile)
        alg = dir_params['algorithm'].split('.')[-1]
        env = dir_params['env']

        if not (env_name == env and alg_name == alg):
            continue

        not_equal = False
        if len(hyper_params.keys()) > 0:
            for key in hyper_params.keys():
                if not str(hyper_params[key]) == str(dir_params[key]):
                    not_equal = True
                    break
        if not_equal:
            continue
        target_dirs.append(current_dir)
    return target_dirs


def main():
    with open('./performance_test_logs/performance_params.json', encoding='utf-8') as param_input:
        performance_json = json.load(param_input, strict=False)
    result_dir = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-1]) + "/performance_test_logs/"

    number_files = int((len(os.listdir(result_dir)) - 1) / 2)
    data_dirs = locate_data_dirs(number_files, result_dir)

    # print("--- data directories:")
    # for entry in data_dirs:
    #     print(entry)
    # print("---")
    failed_ctr = 0
    err_ctr = 0
    for env_counter, env in enumerate(performance_json.keys()):

        for param_config in performance_json[env]:
            alg_name = param_config['alg']
            hyper_params = param_config['hyper_params']
            performance_params = param_config['performance_params']
            performance_measure = performance_params['performance_measure']
            epoch = performance_params['epochs']
            assigned_success_rate = performance_params['min_performance_value']
            min_success_runs = performance_params['min_success_runs']
            n_runs = performance_params['n_runs']

            target_dirs = find_data_dirs_for_params(env, alg_name, hyper_params, data_dirs)

            if target_dirs is not None and len(target_dirs) > 0:

                number_of_runs = len(target_dirs)


                real_success_rate = 0
                successful = 0

                for data_dir in target_dirs:
                    with open(data_dir + "/progress.csv") as csvfile:
                        csv_content = csv.DictReader(csvfile, delimiter=',')
                        for row in csv_content:
                            # print(float(row[performance_measure]) == float(assigned_success_rate))
                            # print("Defined Success: " + str(float(assigned_success_rate)))
                            # print("Actual Success: " + str(row[performance_measure]))
                            if (int(row["epoch"]) + 1) == epoch and float(
                                    row[performance_measure]) >= float(assigned_success_rate):
                                successful += 1
                                real_success_rate = float(row[performance_measure])
                            elif (int(row["epoch"]) + 1) == epoch:
                                real_success_rate = float(row[performance_measure])
                #print(successful)
                if successful >= min_success_runs:
                    print("Success for alg: " + str(alg_name) + " in env: " + str(env))
                else:
                    failed_ctr += 1
                    print("Failed for alg: " + str(alg_name) + " in env: " + str(env) + " only " + str(successful)
                          + " out of " + str(number_of_runs) + " experiments successful" +
                          " with hyperparameters: " + str(hyper_params))
            else:
                print("Warning!!! Could not find data for alg: " + str(alg_name) + " in env: " + str(env) +
                      " with hyperparameters: " + str(hyper_params))
                print("Maybe delete the data folders?")
                failed_ctr += len(performance_json[env])
    print("\n\n\n")
    if failed_ctr > 0:
        print("Performance test of {} environment-algorithm combinations failed".format(failed_ctr))
    if err_ctr > 0:
        print("Performance test was erroneous for {} environment-algorithm combinations".format(err_ctr))
    if failed_ctr == 0 and err_ctr == 0:
        print("Performance tests for all environment-algorithm combinations were successful")

    test_error_logs_in_dir('./performance_test_logs')

if __name__ == "__main__":
    main()