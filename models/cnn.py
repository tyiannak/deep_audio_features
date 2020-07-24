import copy
import re
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, recall_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler, DataLoader


class CNN1(nn.Module):
    def __init__(self, output_dim=7):
        super(CNN1, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.linear_layer1 = nn.Sequential(
            nn.Dropout2d(0.75),
            nn.Linear(5376, 1024),
            nn.LeakyReLU(),

            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.LeakyReLU(),

            # nn.Dropout(0.2),
            nn.Linear(256, output_dim),
            nn.LeakyReLU()
        )

    def forward(self, x):
        # input: (batch_size,1,max_seq,features)
        # Each layer applies the following matrix tranformation
        # recursively: (batch_size,conv_output,max_seq/2 -1,features/2 -1)
        # CNN
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(len(out), -1)

        # DNN
        out = self.linear_layer1(out)

        return out
