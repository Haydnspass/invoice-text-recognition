import argparse
import dill
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms, utils
from torch.autograd import Variable

from model import ConvolutionalNN

class Args:
    def __init__(self):
        self.batch_size = 64
        self.epochs = 10
        self.lr = 0.001
        self.momentum = 0.5
        self.seed = np.random.randint(32000)
        self.log_interval = 1


class Chardata(Dataset):
    def __init__(self, data, target, label=None, transform=None):
        self.data_tensor = torch.from_numpy(data)
        self.target_tensor = torch.from_numpy(target)
        self.label = label
        self.transform = transform

    def __len__(self):
        return self.target_tensor.shape[0]

    def __getitem__(self, idx):

        data_sample = self.data_tensor[idx]
        if self.transform:
            data_sample = self.transform(sample)

        target_sample = self.target_tensor[idx]

        return data_sample, target_sample


def train(data_iterator):
    pass


if __name__ == '__main__':
    args = Args()
    torch.manual_seed(args.seed)

    # load the dataset
    data = np.load('data.npy')

    target = np.load('target.npy')

    data_character = Chardata(data=data, target=target)
    data_loader = DataLoader(data_character,
                             batch_size=args.batch_size,
                             shuffle=True,
                             drop_last=True)

    img_dim = 32 * 32

    # Save results.
    # torch.save(gen.state_dict(), 'torch_save/gen_state_dict.pth')
