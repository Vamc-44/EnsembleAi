import torch.nn as nn
import math
from ndlinear import NdLinear

class NdLinearMNISTNet(nn.Module):
    def __init__(self, input_shape, hidden_size):
        super(NdLinearMNISTNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.ndlinear = NdLinear(input_shape, hidden_size)
        final_dim = math.prod(hidden_size)
        self.fc_out = nn.Linear(final_dim, 100)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.ndlinear(x)
        x = x.view(x.shape[0], -1)
        x = self.fc_out(self.relu(x))
        return x