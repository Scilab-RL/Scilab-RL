from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.torch_layers import create_mlp
from torch import nn
import torch
from stable_baselines3.common.preprocessing import get_action_dim

class MultiplyLayerModule(nn.Module):
    def __init__(self, multi_val):
        self.multi_val = multi_val
        super(MultiplyLayerModule, self).__init__()

    def forward(self, tensors):
        result = tensors * self.multi_val
        return result

class AddLayerModule(nn.Module):
    def __init__(self, add_val):
        self.add_val = add_val
        super(AddLayerModule, self).__init__()

    def forward(self, tensors):
        result = tensors + torch.Tensor(self.add_val).to(tensors.device)
        return result

class ContinuousOptimisticCritic(ContinuousCritic):
    def __init__(self, **kwargs):
        # Same init as Continuous Critic, but overwrite the q-networks with optimistic and clipped ones ones.
        if 'lower_q_limit' in kwargs:
            self.q_limit = kwargs['lower_q_limit']
            del kwargs['lower_q_limit']
            super().__init__(**kwargs)
            self.q_init = -0.067
            self.q_offset = -torch.tensor([self.q_limit / self.q_init - 1]).log().to(self.device)
            self.q_networks = []
            features_dim = kwargs['features_dim']
            action_dim = get_action_dim(kwargs['action_space'])
            net_arch = kwargs['net_arch']
            activation_fn = kwargs['activation_fn']
            for idx in range(self.n_critics):
                q_net = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn)
                q_net.append(AddLayerModule(self.q_offset))
                q_net.append(nn.Sigmoid())
                q_net.append(MultiplyLayerModule(self.q_limit))
                q_net = nn.Sequential(*q_net)
                self.add_module(f"qf{idx}", q_net)
                self.q_networks.append(q_net)
        else:
            super().__init__(**kwargs)


