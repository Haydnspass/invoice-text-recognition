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
        self.batch_size = 128
        self.epochs = 10
        self.lr = 0.001
        self.momentum = 0.5
        self.seed = np.random.randint(32000)
        self.log_interval = 1


class Chardata(Dataset):
    def __init__(self, data, target, label=None, transform=None):
        self.data_tensor = torch.from_numpy(data).type(torch.FloatTensor)
        self.target_tensor = torch.from_numpy(target).type(torch.LongTensor)
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


def train(epoch, data_iterator, criterion, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(data_iterator):
        data, target = Variable(data.view(data.shape[0],1,32,32)), Variable(target)
        optimizer.zero_grad
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_iterator.dataset),
                100. * batch_idx / len(data_iterator), loss.data[0]))


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

    model = ConvolutionalNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        train(epoch, data_loader, criterion, optimizer)
    # Save results.
    # torch.save(gen.state_dict(), 'torch_save/gen_state_dict.pth')
