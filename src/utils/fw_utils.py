from numpy.ma.extras import apply_along_axis
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict
import torch

class Fwd_Training_Dataset(Dataset):
    """
    This class is not used for collecting training data. It is just a necessary intermediary for preparing the dataloader.
    """
    def __init__(self, data):
        self.obs = data['observation']
        self.action = data['action']
        self.next_obs = data['next_observation']
        self.reward = data['reward']

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.action[idx], self.next_obs[idx], self.reward[idx]

class Fwd_Training_Data():
    """
    Used to collect training data and to prepare it for model training.
    """
    def __init__(self):
        self.raw_data = {
            'observation' : [],
            'action' : [],
            'next_observation' : [],
            'reward' : []
        }
        #self.data_set = Training_Data()
        #self.data_loader = DataLoader(self.raw_data)

    def collect_training_data(self, obs, action, next_obs, reward):
        if isinstance(obs, OrderedDict):
            obs = torch.tensor(obs['observation'], dtype = torch.float32)
            action = torch.tensor(action, dtype = torch.float32)
            next_obs = torch.tensor(next_obs['observation'], dtype = torch.float32)
            reward = torch.tensor(reward, dtype = torch.float32)
        else:
            obs = torch.tensor(obs, dtype=torch.float32)
            action = torch.tensor(action, dtype=torch.float32)
            next_obs = torch.tensor(next_obs, dtype=torch.float32)
            reward = torch.tensor(reward, dtype=torch.float32)

        if len(action.shape) != len(obs.shape):
            action = torch.unsqueeze(action, dim=0)

        self.raw_data['observation'].append(obs)
        self.raw_data['action'].append(action)
        self.raw_data['next_observation'].append(next_obs)
        self.raw_data['reward'].append(reward)

    def prepare_dataloader(self, batch_size = 256, shuffle = True):
        fwd_training_dataset = Fwd_Training_Dataset(self.raw_data)
        self.dataloader = DataLoader(fwd_training_dataset, batch_size = batch_size, shuffle = shuffle)

    def get_dataloader(self):
        self.prepare_dataloader()
        return self.dataloader