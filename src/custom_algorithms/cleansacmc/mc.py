import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class MorphologicalNetworks(nn.Module):
    def __init__(self, env, cfg):
        super().__init__()
        hidden_size = cfg["hidden_size"]
        self.obs_shape = np.sum(env.observation_space.shape)
        self.action_shape = np.prod(env.action_space.shape)


        self.state_encoder = nn.Sequential(
            nn.Linear(self.obs_shape + self.action_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.fw_mu = nn.Linear(hidden_size, self.obs_shape)
        self.fw_std = nn.Linear(hidden_size, self.obs_shape)

        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.w_mu = nn.Linear(hidden_size, self.obs_shape)
        self.w_std = nn.Linear(hidden_size, self.obs_shape)

    def forward(self, obs, action):
        hx = self.state_encoder(torch.cat([obs, action], dim=-1))
        mu, std = self.fw_mu(hx), F.softplus(self.fw_std(hx))

        hw = self.action_encoder(action)
        w_mu, w_std = self.w_mu(hw), F.softplus(self.w_std(hw))

        return Normal(mu, std), Normal(w_mu, w_std)
