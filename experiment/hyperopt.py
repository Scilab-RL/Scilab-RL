import matplotlib
matplotlib.use("pdf")
from matplotlib import pyplot as plt
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
from matplotlib.backends.backend_pdf import PdfPages


PROCS_RUNNING = 0
MINUTES_TO_RUN = 0.5
HOURS_TO_RUN = MINUTES_TO_RUN / 60
MAX_RUNS_PER_PARAM = 3


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

def do_train(cfg: DictConfig, queue=None) -> float:
    np.random.seed()
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

    if queue is not None:
        queue.put(test_acc)
    return test_acc

def check_cfg_duplicate(cfg, metric='', max_duplicates=1):
    experiment = mlflow.get_experiment_by_name(cfg.mlflow.runname)
    try:
        runs = mlflow.list_run_infos(experiment.experiment_id)
    except:
        runs = []
    n_duplicate_vals = 0
    duplicate_val = 0
    # n_duplicate_runs = 0
    for r in runs:
        r_info = mlflow.get_run(r.run_id)
        cfg_duplicate = True
        for k,v in r_info.data.params.items():
            k_items = k.split(".")[1:]
            cfg_item = cfg
            invalid_param = False
            for k_item in k_items:
                if k_item in cfg_item.keys():
                    cfg_item = cfg_item[k_item]
                else:
                    invalid_param = True
            if invalid_param:
                continue
            this_run_v = str(cfg_item)
            if this_run_v != v:
                cfg_duplicate = False
                break
        if cfg_duplicate is True:
            # n_duplicate_runs += 1
            if metric != '' and metric in r_info.data.metrics.keys():
                metric_val = r_info.data.metrics[metric]
                duplicate_val += metric_val
                n_duplicate_vals += 1

        if n_duplicate_vals > max_duplicates: # One duplicate is fine because the current run is already registered. But no more than one.
            break
    is_duplicate = n_duplicate_vals >= max_duplicates
    if metric != '':
        if n_duplicate_vals > 0:
            avg_metric_val = duplicate_val / n_duplicate_vals
        else:
            avg_metric_val = None
            # is_duplicate = False # In this case there are duplicate runs but these are not yet finished so we cannot get their value. If this is the case we just claim that this is not a duplicate and evaluate the run normally.
        return is_duplicate, avg_metric_val
    else:
        return is_duplicate

def my_func(cfg: DictConfig, queue=None) -> float:
    np.random.seed()
    print(f"current proc: {mp.current_process()}")
    print(f"fc1: {cfg.model.fc1_units}")
    print(f"fc2: {cfg.model.fc2_units}")
    with mlflow.start_run():
        quotient = cfg.model.fc1_units / cfg.model.fc2_units
        rndn = np.random.normal()
        t_sleep = quotient * 4 + quotient * rndn
        t_sleep /= 10
        print(f"sleeping for {t_sleep} secs.")
        n_loops = int(t_sleep * 100000)
        a = 1
        for _ in range(n_loops):
            a *= 1.000001
        # time.sleep(t_sleep) # time.sleep triggers a KeyboardInterrupt when running in multiprocess mode
        mlflow.log_param("cfg.model.fc1_units", cfg.model.fc1_units)
        mlflow.log_param("cfg.model.fc2_units", cfg.model.fc2_units)
        mlflow.log_metric("t_sleep", float(t_sleep))
        mlflow.log_metric("hyperopt_score", float(t_sleep))
    if queue is not None:
        queue.put(t_sleep)
    return t_sleep

def proc_finished_cb(result=None):
    global PROCS_RUNNING, MAX_RUNS_PER_PARAM
    print(f"Proc done with result {result}")
    PROCS_RUNNING -= MAX_RUNS_PER_PARAM

@hydra.main(config_name='config.yaml')
def main(cfg: DictConfig, *args) -> float:
    global MAX_RUNS_PER_PARAM, PROCS_RUNNING
    mlflow.set_tracking_uri('file://' + hydra.utils.get_original_cwd() + '/mlruns')
    tracking_uri = mlflow.get_tracking_uri()
    print("Current tracking uri: {}".format(tracking_uri))
    mlflow.set_experiment(cfg.mlflow.runname)
    # time.sleep(1)
    is_duplicate, duplicate_score = check_cfg_duplicate(cfg, metric='hyperopt_score', max_duplicates=MAX_RUNS_PER_PARAM)
    if is_duplicate:
        print("This parameterization has been tried before, will just return the results from last time.")
        return duplicate_score

    PROCS_RUNNING += MAX_RUNS_PER_PARAM

    func_to_train = do_train

    ### Parallel runs with pool, sync and async version. # NOT WORKING
    # inner_pool = mp.Pool(processes=MAX_RUNS_PER_PARAM)
    # for _ in range(MAX_RUNS_PER_PARAM):
    #     inner_pool.map(func_to_train, cfg) # sync execution
    #     inner_pool.apply_async(func_to_train, cfg, callback=proc_finished_cb) # or async execution
    # # print(res)
    # result = 0

    ### Parallel runs with multiprocess.
    # value_queue = mp.Queue()
    # processes = [Process(target=func_to_train, args=(cfg,value_queue)) for x in range(MAX_RUNS_PER_PARAM)]
    # for p in processes:
    #     p.start()
    #
    # result = 0
    # for p in processes:
    #     p.join()
    # while not value_queue.empty():
    #     result += value_queue.get()
    # result /= MAX_RUNS_PER_PARAM
    # proc_finished_cb()

    ### SINGLE RUN:
    result = func_to_train(cfg)
    proc_finished_cb()

    ### AVG over consecutive runs:
    # result = 0
    # for _ in range(MAX_RUNS_PER_PARAM):
    #     res = func_to_train(cfg)
    #     result += res
    # result /= MAX_RUNS_PER_PARAM
    # proc_finished_cb()

    return result

if __name__ == "__main__":
    # score = main()
    # n_cpus = psutil.cpu_count(logical=True) - 1
    #
    # # n_outer_pool_procs = n_cpus // MAX_RUNS_PER_PARAM
    # n_outer_pool_procs = 1
    # outer_pool = mp.Pool(n_outer_pool_procs)
    #
    main()

    # sec_to_run = HOURS_TO_RUN * 60 * 60
    # start_time = time.time()
    # while True:
    #     runtime = time.time() - start_time
    #     runtime_hours = runtime / 3600
    #     if runtime >= sec_to_run:
    #         break
    #     # while True:
    #     #     res_free = check_resources_free(free_cpus=MAX_RUNS_PER_PARAM + 1)
    #     #     if res_free:
    #     #         break
    #     #     else:
    #     #         print("Waiting for free resources")
    #     # time.sleep(5)
    #     print(f"Starting next processes. Running for {runtime_hours:.2f} of {HOURS_TO_RUN:.2f} hours now.")
    #     # outer_pool.apply_async(main, callback=proc_finished_cb)
    #     # outer_pool.apply(main, range(n_outer_pool_procs))
    #     main()

    study = optuna.load_study("simple_mnist", "sqlite:///simple_mnist_hyperopt.db")
    # with PdfPages('hyperopt_pdf.pdf') as pdf:
    if not os.path.exists("hyperopt_logs"):
        os.mkdir("hyperopt_logs")
    fig = plot_optimization_history(study)
    fig.write_image("hyperopt_logs/plot_optimization_history.png")
    fig = plot_contour(study)
    fig.write_image("hyperopt_logs/plot_contour.png")
    fig = plot_param_importances(study)
    fig.write_image("hyperopt_logs/plot_param_importances.png")
    fig = plot_intermediate_values(study)
    fig.write_image("hyperopt_logs/plot_intermediate_values.png")
