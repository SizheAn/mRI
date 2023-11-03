# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 22:27:33 2021

@author: sizhean
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

torch.set_default_tensor_type(torch.FloatTensor)

class CNN(nn.Module):
    def __init__(self, channels, neurons, fc_input_size):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=3, padding=1)
        self.drop1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.drop2 = nn.Dropout(0.3)
        self.batchnorm1 = nn.BatchNorm2d(32, momentum=0.05, eps = 0.001)
        
        self.fc = nn.Linear(fc_input_size, 512)
        self.batchnorm2 = nn.BatchNorm1d(512, momentum=0.05, eps = 0.001)
        self.drop3 = nn.Dropout(0.4)
        self.regression = nn.Linear(512, neurons)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.drop1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.drop2(x)
        x = F.relu(x)
        x = self.batchnorm1(x)
        
        # flatten
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = F.relu(x)
        x = self.batchnorm2(x)
        x = self.drop3(x)
        
        x = self.regression(x)
        
        return x

