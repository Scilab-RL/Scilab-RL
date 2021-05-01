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
import omegaconf
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
import random


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
        # self.input_data_size = 28 * 28
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.device = 'cuda:0'
        self.conv1 = nn.Conv2d(1, cfg.conv1.n_kernels, cfg.conv1.kernel_size)
        # conv1_out_dim = self.input_data_size - (cfg.conv1.kernel_size - 1) - 1
        # pool1_out_dim = conv1_out_dim
        self.conv2 = nn.Conv2d(cfg.conv1.n_kernels, cfg.conv2.n_kernels, cfg.conv2.kernel_size)
        # an affine operation: y = Wx + b
        # fc1_in_dim = cfg.conv2.n_kernels * (cfg.conv1.n_kernels) * (cfg.conv1.n_kernels)
        mp2_out_dim = 28
        fc1_in_dim = mp2_out_dim * mp2_out_dim * cfg.conv2.n_kernels
        self.mp2 = nn.AdaptiveMaxPool2d(mp2_out_dim)
        self.fc1 = nn.Linear(fc1_in_dim, cfg.fc1_units)  # 6*6 from image dimension
        self.fc2 = nn.Linear(cfg.fc1_units, cfg.fc2_units)
        self.fc3 = nn.Linear(cfg.fc2_units, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x.to(self.device)
        datasize = self.num_flat_features(x)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # If the size is a square, you can specify with a single number
        x = self.conv2(x)
        # x = F.max_pool2d(F.relu(x), 2)
        x = self.mp2(x)
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


def set_rnd_seed():
    rnd_seed = random.randint(0, 100000000)
    np.random.seed(rnd_seed)
    random.seed(rnd_seed)
    torch.manual_seed(rnd_seed)
    return rnd_seed


def do_train(cfg: DictConfig, queue=None) -> float:
    device = torch.device("cuda:0")
    torch.cuda.device(device)
    seed = set_rnd_seed()
    data_path = hydra.utils.get_original_cwd()
    dataset = MNIST(data_path, download=True, transform=transforms.ToTensor())
    train, val = random_split(dataset, [55000, 5000])
    small_dataset = torch.utils.data.Subset(dataset, list(range(6000)))
    train, val = random_split(small_dataset, [5000, 1000])


    trainloader = DataLoader(train, batch_size=cfg.train.batch_size, shuffle=True)
    testloader = DataLoader(val, batch_size=cfg.test.batch_size, shuffle=False)

    # load model
    model = Net(cfg.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.optimizer.lr,
                          momentum=cfg.optimizer.momentum)

    # start new run
    mlflow.set_tracking_uri('file://' + hydra.utils.get_original_cwd() + '/mlruns')
    tracking_uri = mlflow.get_tracking_uri()
    print("Current tracking uri: {}".format(tracking_uri))
    mlflow.set_experiment(cfg.hydra.sweeper.study_name)
    steps = 0
    test_acc = 0
    with mlflow.start_run():
        mlflow.log_param("rnd_seed", seed)
        for epoch in range(cfg.train.epoch):
            running_loss = 0.0

            # log param
            log_params_from_omegaconf_dict(cfg)
            mlflow.log_param("cfg.model.fc1_units", cfg.model.fc1_units)
            mlflow.log_param("cfg.model.fc2_units", cfg.model.fc2_units)
            for i, (x, y) in enumerate(trainloader):
                steps = epoch * len(trainloader) + i
                optimizer.zero_grad()
                x = x.to(device)
                y = y.to(device)
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
                    x = x.to(device)
                    y = y.to(device)
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
                mlflow.log_metric("hyperopt_score", float(test_acc), step=steps)

    if queue is not None:
        queue.put(test_acc)
    return test_acc


@hydra.main(config_name='config.yaml')
def main(cfg: DictConfig, *args) -> float:

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print("Cuda is available!")

    mlflow.set_tracking_uri('file://' + hydra.utils.get_original_cwd() + '/mlruns')
    tracking_uri = mlflow.get_tracking_uri()
    print("Current tracking uri: {}".format(tracking_uri))
    study_name = omegaconf.OmegaConf.load('experiment/config.yaml').hydra.sweeper.study_name
    mlflow.set_experiment(study_name)
    # mlflow.set_experiment("mlflow-study")

    func_to_train = do_train

    result = func_to_train(cfg)

    return result

if __name__ == "__main__":
    print(f"Running hyperopt and using torch device {torch.cuda.current_device()}.")
    main()

    cfg = omegaconf.OmegaConf.load('experiment/config.yaml')
    study_name = cfg.hydra.sweeper.study_name
    study = optuna.load_study(study_name, cfg.hydra.sweeper.storage)
    imgdir = f"hyperopt_logs/{study_name}"
    if not os.path.exists("hyperopt_logs"):
        os.mkdir("hyperopt_logs")
    if not os.path.exists(imgdir):
        os.mkdir(imgdir)

    try:
        fig = plot_optimization_history(study)
        fig.write_image(f"{imgdir}/plot_optimization_history.png")
    except:
        pass
    try:
        fig = plot_contour(study)
        fig.write_image(f"{imgdir}//plot_contour.png")
    except:
        pass
    try:
        fig = plot_param_importances(study)
        fig.write_image(f"{imgdir}//plot_param_importances.png")
    except:
        pass
    # try:
    #     fig = plot_intermediate_values(study)
    #     fig.write_image(f"{imgdir}//plot_intermediate_values.png")
    # except:
    #     pass
