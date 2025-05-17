import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tranforms
from torch.utils.data import DataLoader
import time

# Setting the variables for convolutional layers
convolutional_kernel_size = 3    
convolutional_stride = 1
convolutional_padding = 1
convolutional_activation = nn.ReLU()
convolutional_pool = nn.MaxPool2d(kernel_size=2, stride=2)