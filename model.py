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

def resize_images(images):
    return np.array([zoom(images[0], (0.5, 0.5)) for image in images])

@task(name=f'Prepare Datasets')
def prepare_datasets():
    print("Prepare datasets...")
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False)

    x_train = resize_images(train_dataset)
    y_train = resize_images(test_dataset)

    x_train = torch.tensor(x_train.reshape(-1, 14*14).astype('float32') / 255)
    y_train = torch.tensor([label for _, label in train_dataset], dtype=torch.long) 

    x_test = torch.tensor(x_test.reshape(-1, 14*14).astype('float32') / 255)
    y_test = torch.tensor([label for _, label in test_dataset], dtype=torch.long)

    print("✅ Datasets prepared successfully")

    return x_train, y_train, x_test, y_test
