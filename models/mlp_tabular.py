import os

import torch.nn as nn
import torch
import math
from torch import autocast

class MLPNet(nn.Module):
    def __init__(self, num_features, num_classes, num_hidden=512, init_method='kaiming', dropout=0.0):
        super(MLPNet, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.encoder = nn.Sequential(
            nn.Linear(num_features, num_hidden),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_hidden, num_hidden),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_hidden, num_hidden),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_hidden, num_hidden),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_hidden, num_hidden),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_hidden, num_hidden),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_hidden, num_hidden),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_hidden, num_hidden),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
        )
        self.init_method = init_method

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_hidden, num_hidden),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_classes),
        )

        # To prevent pruning
        self.classifier.is_classifier = True
        self.init_method = init_method

    def features(self, input):
        # input = input.view(input.size(0), -1)
        x = self.encoder(input)
        return x

    def forward(self, input):
        x = self.features(input)
        # device = \
        #     f'cuda:{os.environ["CUDA_VISIBLE_DEVICE"]}' if \
        #     "CUDA_VISIBLE_DEVICE" in os.environ else 'cuda'
        # with autocast(device_type=device, dtype=torch.float16):
        x = self.classifier(x)
        return x

    def reset_weights(self):
        def init_weights(module):
            if isinstance(module, nn.Linear):
                if self.init_method == 'kaiming':
                    nn.init.kaiming_uniform_(module.weight)
                elif self.init_method == 'xavier':
                    nn.init.xavier_uniform(module.weight)
                else:
                    assert False
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
        self.apply(init_weights)

def mlpnet_debug(num_features, num_classes, pretrained=None):
    # For CIFAR 100
    model = MLPNet(num_features=num_features, num_classes=num_classes, num_hidden=64)
    return model

def mlpnet(num_features, num_classes, pretrained=None):
    # For CIFAR 100
    model = MLPNet(num_features=num_features, num_classes=num_classes)
    return model

def mlpnet_dropout(num_features, num_classes, pretrained=None):
    # For CIFAR 100
    model = MLPNet(num_features=num_features, num_classes=num_classes, dropout=0.5)
    return model

def mlpnet_xavier(num_features, num_classes, pretrained=None):
    # For CIFAR 100
    model = MLPNet(num_features=num_features, num_classes=num_classes, init_method='xavier')
    return model
