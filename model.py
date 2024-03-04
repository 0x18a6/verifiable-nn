import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import logging
from scipy.ndimage import zoom
from giza_actions.action import action
from giza_actions.task import task
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 196 #14*14
hidden_size = 10
num_classes = 10
num_epochs = 10
batch_size = 256
learning_rate = 0.001

class Net(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

        