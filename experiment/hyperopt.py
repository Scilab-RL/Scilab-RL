import matplotlib
# matplotlib.use("TkAgg")
from matplotlib import  pyplot as plt
# DB_NAME = 'ideas_hrl_hyperopt_database'
# DB_USER = 'hyperopt_user'
# DB_PW = 'Ideas21!'
# DB_HOST = 'wtmpc165'

#import comet_ml
import hydra
import mlflow
from omegaconf import DictConfig, ListConfig
from torchvision.datasets import MNIST
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader, random_split
import os
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
from multiprocessing import Process, Queue
import multiprocessing as mp
import psutil
import numpy as np
from util.hyper_param_utils import ParamRepeatPruner
import optuna
from mlflow.tracking import MlflowClient
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice


PROCS_RUNNING = 0
MINUTES_TO_RUN = 0.5
HOURS_TO_RUN = MINUTES_TO_RUN / 60
RUNS_PER_PARAM = 3


def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        _explore_recursive(param_name, element)


def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f'{parent_name}.{k}', v)
            else:
                mlflow.log_param(f'{parent_name}.{k}', v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlflow.log_param(f'{parent_name}.{i}', v)


class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, cfg.conv1.n_kernels, cfg.conv1.kernel_size)
        self.conv2 = nn.Conv2d(cfg.conv1.n_kernels, cfg.conv2.n_kernels, cfg.conv2.kernel_size)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(cfg.conv2.n_kernels * (cfg.conv1.n_kernels-1) * (cfg.conv1.n_kernels-1), cfg.fc1_units)  # 6*6 from image dimension
        self.fc2 = nn.Linear(cfg.fc1_units, cfg.fc2_units)
        self.fc3 = nn.Linear(cfg.fc2_units, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # If the size is a square, you can specify with a single number
        x = self.conv2(x)
        x = F.max_pool2d(F.relu(x), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=0)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def check_resources_free(free_ram=1000, free_gpu_ram=2000, free_cpus=2):
    global PROCS_RUNNING
    cpu_used = psutil.cpu_percent()
    # print(f"Current cpu used: {cpu_used}%")
    # total_cpu = psutil.cpu_count(logical=True) * 100
    # cpus_free = (total_cpu - cpu_used) >= (free_cpus * 100)
    cpus_free = PROCS_RUNNING < free_cpus

    gpu_free = True

    ram_free = True

    return cpus_free and gpu_free and ram_free

def do_train(cfg: DictConfig) -> float:

    data_path = hydra.utils.get_original_cwd()
    dataset = MNIST(data_path, download=True, transform=transforms.ToTensor())
    train, val = random_split(dataset, [55000, 5000])

    trainloader = DataLoader(train, batch_size=cfg.train.batch_size, shuffle=True)
    testloader = DataLoader(val, batch_size=cfg.test.batch_size, shuffle=False)

    # load model
    model = Net(cfg.model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.optimizer.lr,
                          momentum=cfg.optimizer.momentum)

    # start new run
    mlflow.set_tracking_uri('file://' + hydra.utils.get_original_cwd() + '/mlruns')
    tracking_uri = mlflow.get_tracking_uri()
    print("Current tracking uri: {}".format(tracking_uri))
    mlflow.set_experiment(cfg.mlflow.runname)
    steps = 0
    test_acc = 0
    with mlflow.start_run():
        for epoch in range(cfg.train.epoch):
            running_loss = 0.0

            # log param
            log_params_from_omegaconf_dict(cfg)
            mlflow.log_param("cfg.model.fc1_units", cfg.model.fc1_units)
            mlflow.log_param("cfg.model.fc2_units", cfg.model.fc2_units)
            for i, (x, y) in enumerate(trainloader):
                steps = epoch * len(trainloader) + i
                optimizer.zero_grad()

                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                # log metric
                mlflow.log_metric("train_loss", loss.item(), step=steps)
                mlflow.log_metric("steps", steps, step=steps)

            test_loss = 0
            test_acc = 0
            with torch.set_grad_enabled(False):
                for i, (x, y) in enumerate(testloader):
                    # steps = epoch * len(trainloader) + i
                    # optimizer.zero_grad()

                    outputs = model(x)
                    test_loss += criterion(outputs, y)

                    out_idxs = outputs.topk(1, dim=1)[1]
                    out_idxs = out_idxs.view(y.shape)
                    correct = out_idxs == y
                    acc = torch.mean(correct.type(torch.DoubleTensor))
                    test_acc += acc

                test_acc /= len(testloader)
                test_loss /= len(testloader)
                mlflow.log_metric("test_acc", float(test_acc), step=steps)
                mlflow.log_metric("test_loss", float(test_loss), step=steps)
                print(f"Test accuracy of epoch {epoch}: {test_acc}")
    return test_acc

def check_cfg_duplicate(cfg, metric='', max_duplicates=1):
    experiment = mlflow.get_experiment_by_name(cfg.mlflow.runname)

    try:
        runs = mlflow.list_run_infos(experiment.experiment_id)
    except:
        runs = []
    n_duplicates = 0
    duplicate_val = 0
    for r in runs:
        r_info = mlflow.get_run(r.run_id)
        cfg_duplicate = True
        for k,v in r_info.data.params.items():
            k_items = k.split(".")[1:]
            cfg_item = cfg
            for k_item in k_items:
                cfg_item = cfg_item[k_item]
            this_run_v = str(cfg_item)
            if this_run_v != v:
                cfg_duplicate = False
                break
        if cfg_duplicate is True:
            n_duplicates += 1
            if metric != '':
                metric_val = r_info.data.metrics[metric]
                duplicate_val += metric_val
        if n_duplicates > max_duplicates: # One duplicate is fine because the current run is already registered. But no more than one.
            break
    is_duplicate = n_duplicates >= max_duplicates
    if metric != '':
        if n_duplicates > 0:
            avg_metric_val = duplicate_val / n_duplicates
        else:
            avg_metric_val = None
        return is_duplicate, avg_metric_val
    else:
        return is_duplicate

def my_func(cfg: DictConfig, queue) -> float:
    # mlflow.set_tracking_uri('file://' + hydra.utils.get_original_cwd() + '/mlruns')
    # tracking_uri = mlflow.get_tracking_uri()
    # print("Current tracking uri: {}".format(tracking_uri))
    np.random.seed()
    print(f"current proc: {mp.current_process()}")
    print(f"fc1: {cfg.model.fc1_units}")
    print(f"fc2: {cfg.model.fc2_units}")
    with mlflow.start_run():
        quotient = cfg.model.fc1_units / cfg.model.fc2_units
        rndn = np.random.normal()
        t_sleep = quotient * 4 + quotient * rndn
        t_sleep /= 10
        # t_sleep = 6
        print(f"sleeping for {t_sleep} secs.")
        n_loops = int(t_sleep * 10000)
        a = 1
        for _ in range(n_loops):
            a *= 1.000001
        # time.sleep(t_sleep) # time.sleep triggers a KeyboardInterrupt when running in multiprocess mode
        # start = time.process_time()
        # while (start + t_sleep) > time.process_time():
        #   var_a = 1
        mlflow.log_param("cfg.model.fc1_units", cfg.model.fc1_units)
        mlflow.log_param("cfg.model.fc2_units", cfg.model.fc2_units)
        mlflow.log_metric("t_sleep", float(t_sleep))
        mlflow.log_metric("hyperopt_score", float(t_sleep))
    queue.put(t_sleep)
    return t_sleep

def proc_finished_cb(result):
    global PROCS_RUNNING, RUNS_PER_PARAM
    print(f"Proc done with result {result}")
    PROCS_RUNNING -= RUNS_PER_PARAM

@hydra.main(config_name='config.yaml')
def main(cfg: DictConfig) -> float:
    global RUNS_PER_PARAM
    mlflow.set_tracking_uri('file://' + hydra.utils.get_original_cwd() + '/mlruns')
    tracking_uri = mlflow.get_tracking_uri()
    print("Current tracking uri: {}".format(tracking_uri))
    mlflow.set_experiment(cfg.mlflow.runname)
    # time.sleep(1)
    is_duplicate, duplicate_score = check_cfg_duplicate(cfg, metric='hyperopt_score', max_duplicates=RUNS_PER_PARAM)
    if is_duplicate:
        print("This parameterization has been tried before, will just return the results from last time.")
        return duplicate_score

        # inner_pool.apply_async(my_func, cfg, callback=proc_finished_cb)

    ### Parallel runs with pool. # NOT WORKING
    # inner_pool = mp.Pool(processes=RUNS_PER_PARAM)
    # for _ in range(RUNS_PER_PARAM):
    #     inner_pool.map(my_func, cfg)
    # # print(res)
    # result = 0

    ### Parallel runs with multiprocess.
    value_queue = mp.Queue()
    processes = [Process(target=my_func, args=(cfg,value_queue)) for x in range(RUNS_PER_PARAM)]
    for p in processes:

        p.start()

    result = 0
    for p in processes:
        p.join()
    while not value_queue.empty():
        result += value_queue.get()
    result /= RUNS_PER_PARAM

    ### SINGLE RUN:
    # result = my_func(cfg)

    ### AVG over consecutive runs:
    # result = 0
    # for _ in range(RUNS_PER_PARAM):
    #     res = my_func(cfg)
    #     result += res
    #
    # result /= RUNS_PER_PARAM

    return result

if __name__ == "__main__":
    # main()
    # mlflow.set_tracking_uri('file://' + hydra.utils.get_original_cwd() + '/mlruns')
    # tracking_uri = mlflow.get_tracking_uri()
    # print("Current tracking uri: {}".format(tracking_uri))
    # mlflow.set_experiment(cfg.mlflow.runname)

    n_cpus = psutil.cpu_count(logical=True) - 1

    # prune_params = ParamRepeatPruner(study)
    # runs_per_param = 3
    # print(f"We will use {n_cpus} cpus")
    # # n_cpus = mp.cpu_count()-1 # leave one cpu free for other tasks the machine has to do.
    n_outer_pool_procs = n_cpus // RUNS_PER_PARAM
    # n_outer_pool_procs = 1
    outer_pool = mp.Pool(n_outer_pool_procs)
    #
    sec_to_run = HOURS_TO_RUN * 60 * 60
    start_time = time.time()
    while True:
        runtime = time.time() - start_time
        runtime_hours = runtime / 3600
        if runtime >= sec_to_run:
            break
        while True:
            res_free = check_resources_free(free_cpus=RUNS_PER_PARAM + 1)
            if res_free:
                break
            time.sleep(5)
        print(f"Starting next processes. Running for {runtime_hours:.2f} of {HOURS_TO_RUN:.2f} hours now.")
        # outer_pool.apply(main)
        main()

    study = optuna.load_study("simple_mnist", "sqlite:///simple_mnist_hyperopt.db")
    plot_optimization_history(study)
    plt.show()
    # plot_intermediate_values(study)
    plot_contour(study)
    #     # outer_pool.apply_async(main, callback=proc_finished_cb)
    #     PROCS_RUNNING += runs_per_param

    plot_param_importances(study)