import torch
import torch.nn as nn


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.sigmod = nn.Sigmoid()
    def forward(self, x):
        return self.sigmod(self.conv(x))
