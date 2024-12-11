import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from gymnasium import spaces
import os
from collections import OrderedDict
from utils.fw_utils import Fwd_Training_Data

LOG_STD_MAX = 2
LOG_STD_MIN = -5

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')

class ProbabilisticForwardNet(nn.Module):
    def __init__(self, config, env):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.n_hidden_layers = config['n_hidden_layers']  # Number of hidden layers

        # Activation function setup
        if config['activation_func'] == 'relu':
            self.activation_func = nn.ReLU
        elif config['activation_func'] == 'logistic':
            self.activation_func = nn.Sigmoid
        elif config['activation_func'] == 'tanh':
            self.activation_func = nn.Tanh
        else:
            assert False, "Error, activation function not valid"

        if isinstance(env.observation_space, spaces.Dict):
            self.obs_shape = env.observation_space.spaces['observation'].shape[0]
        else:
            self.obs_shape = np.array(env.observation_space.shape).prod()

        if isinstance(env.action_space, spaces.Discrete):
            self.action_shape = env.action_space.n.size
        else:
            self.action_shape = np.prod(env.action_space.shape)

        self.input_shape = self.obs_shape + self.action_shape

        # Loss function setup
        if config['loss_func'] == 'l2':
            self.loss_func = self.l2_loss_delta
        elif config['loss_func'] == 'nll':
            self.loss_func = self.nll_loss_delta
        else:
            assert False, "Error: loss function invalid"

        self.cfg = config

    def build_hidden_layers(self, input_size, output_size):
        layers = [nn.Linear(input_size, self.hidden_size), self.activation_func()]
        for _ in range(self.n_hidden_layers - 1):  # Add additional hidden layers
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(self.activation_func())
        layers.append(nn.Linear(self.hidden_size, output_size))
        return nn.Sequential(*layers)

    def l2_loss_delta(self, obs, action, next_obs):
        next_obs_prediction = self(obs, action).loc
        loss = (next_obs_prediction - (next_obs - obs)) ** 2
        return loss

    def nll_loss_delta(self, obs, action, next_obs):
        forward_normal = self(obs, action)
        fwd_normal_loss = -forward_normal.log_prob(next_obs - obs)
        return fwd_normal_loss

    def predict(self, obs, action):
        assert (self.obs_shape == obs.shape[-1])
        assert (self.action_shape == action.shape[-1])

        if isinstance(obs, OrderedDict):
            obs = torch.tensor(obs['observation'], dtype = torch.float32)
            action = torch.tensor(action, dtype = torch.float32)
        else:
            obs = torch.tensor(obs, dtype=torch.float32)
            action = torch.tensor(action, dtype=torch.float32)

        if len(action.shape) != len(obs.shape):
            action = torch.unsqueeze(action, dim=0)

        return self.forward(obs, action).loc.detach() + obs.detach()

    def train(self, optimizer, dataloader):
        for obs, action, next_obs in dataloader:
            obs, action, next_obs = obs.to(device), action.to(device), next_obs.to(device)
            loss = self.loss_func(obs, action, next_obs)
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

    def get_average_loss(self, dataloader):
        losses = []
        for obs, action, next_obs in dataloader:
           losses.append(self.loss_func(obs, action, next_obs).mean().detach().item())
        return np.mean(losses)

    def collect_training_data(self, training_data:Fwd_Training_Data, last_obs, action, new_obs):
        training_data.collect_training_data(last_obs, action, new_obs)

    def save_model(self, model_name):
        torch.save(self, os.path.join(self.cfg['model_save_path'], f'{model_name}.pt'))

    def save_state_dict(self, state_dict_name):
        torch.save(self.state_dict(), os.path.join(self.cfg['model_save_path'], f'{state_dict_name}.pt'))

class ProbabilisticForwardMLENetwork(ProbabilisticForwardNet):
    def __init__(self, config, env):
        super().__init__(config, env)
        # Building the state-action encoder and output layers dynamically based on n_hidden_layers
        self.state_action_encoder = self.build_hidden_layers(self.input_shape, self.hidden_size)
        self.fw_mu = self.build_hidden_layers(self.hidden_size, self.obs_shape)
        self.fw_log_std = self.build_hidden_layers(self.hidden_size, self.obs_shape)

    def forward(self, obs, action):
        assert (self.obs_shape == obs.shape[-1])
        assert (self.action_shape == action.shape[-1])
        hx = torch.cat([obs, action], dim=-1).float()
        hx = self.state_action_encoder(hx)
        fw_mu, fw_log_std = self.fw_mu(hx), self.fw_log_std(hx)
        fw_log_std = torch.clamp(fw_log_std, LOG_STD_MIN, LOG_STD_MAX)
        return Normal(fw_mu, fw_log_std.exp())


class DeterministicForwardNetwork(ProbabilisticForwardNet):
    def __init__(self, config, env):
        super().__init__(config, env)
        # Building the state-action model dynamically based on n_hidden_layers
        self.state_action_model = self.build_hidden_layers(self.input_shape, self.obs_shape)

    def forward(self, obs, action):
        assert (self.obs_shape == obs.shape[-1])
        assert (self.action_shape == action.shape[-1])
        hx = torch.cat([obs, action], dim=-1)
        hx = self.state_action_model(hx)
        return Normal(hx, torch.zeros_like(hx)+10**(-10))

class ForwardNetEnsamble(nn.Module):
    def __init__(self, config, env, ensamble_size: int, fw_class):
        super().__init__()
        self._ensamble = nn.ModuleList(
            [
                fw_class(config, env)
                for _ in range(ensamble_size)
            ]
        )

    def forward(self, obs, action):
        return torch.stack([model(obs, action) for model in self._ensamble])
