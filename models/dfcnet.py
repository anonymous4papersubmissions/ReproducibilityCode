import torch.nn as nn
import torch
import os
import math
from torch import autocast

class CNN(nn.Module):
    def __init__(self, num_classes=10, s=32, alpha=8, init_method='kaiming'):
        super(CNN, self).__init__()

        self.num_classes = num_classes
        self.encoder = nn.Sequential(
            nn.Conv2d(3, alpha, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(alpha),
            nn.ReLU(),
            nn.Conv2d(alpha, 2*alpha, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(2*alpha),
            nn.ReLU(),
            nn.Conv2d(2*alpha, 2*alpha, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2*alpha),
            nn.ReLU(),
            nn.Conv2d(2*alpha, 4*alpha, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4*alpha),
            nn.ReLU(),
            nn.Conv2d(4*alpha, 4*alpha, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4*alpha),
            nn.ReLU(),
            nn.Conv2d(4*alpha, 8*alpha, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8*alpha),
            nn.ReLU(),
            nn.Conv2d(8*alpha, 8*alpha, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8*alpha),
            nn.ReLU(),
            nn.Conv2d(8*alpha, 16*alpha, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16*alpha),
            nn.ReLU(),
        )
        self.s = s
        self.alpha = alpha
        self.init_method = init_method

        self.classifier = nn.Sequential(
            nn.Linear(int(alpha/16*s**2), 64*alpha),
            nn.BatchNorm1d(int(alpha/16*s**2)),
            nn.ReLU(),
            nn.Linear(64*alpha, self.num_classes),
        )

        # To prevent pruning
        self.classifier[-1].is_classifier = True
        self.init_method = init_method

    def features(self, input):
        # device = (
        #     f'cuda:{os.environ["CUDA_VISIBLE_DEVICE"]}' if \
        #     "CUDA_VISIBLE_DEVICE" in os.environ else 'cuda'
        # ) if input.device is not 'cpu' else 'cpu'
        # with autocast(device_type=device, dtype=torch.float16):
        x = self.encoder(input)
        x = x.view(x.shape[0], -1)
        return x

    def forward(self, input):
        x = self.features(input)
        # device = (
        #     f'cuda:{os.environ["CUDA_VISIBLE_DEVICE"]}' if \
        #     "CUDA_VISIBLE_DEVICE" in os.environ else 'cuda'
        # ) if input.device is not 'cpu' else 'cpu'
        # with autocast(device_type=device, dtype=torch.float16):
        x = self.classifier(x)
        return x


class DFCNet(nn.Module):
    def __init__(self, num_classes=10, s=32, alpha=8, init_method='kaiming'):
        super(DFCNet, self).__init__()

        self.num_classes = num_classes
        self.encoder = nn.Sequential(
            nn.Linear(int(3*s**2), int(alpha*s**2)),
            nn.BatchNorm1d(int(alpha*s**2)),
            nn.ReLU(),
            nn.Linear(int(alpha*s**2), int(alpha/2*s**2)),
            nn.BatchNorm1d(int(alpha/2*s**2)),
            nn.ReLU(),
            nn.Linear(int(alpha/2*s**2), int(alpha/2*s**2)),
            nn.BatchNorm1d(int(alpha/2*s**2)),
            nn.ReLU(),
            nn.Linear(int(alpha/2*s**2), int(alpha/4*s**2)),
            nn.BatchNorm1d(int(alpha/4*s**2)),
            nn.ReLU(),
            nn.Linear(int(alpha/4*s**2), int(alpha/4*s**2)),
            nn.BatchNorm1d(int(alpha/4*s**2)),
            nn.ReLU(),
            nn.Linear(int(alpha/4*s**2), int(alpha/8*s**2)),
            nn.BatchNorm1d(int(alpha/8*s**2)),
            nn.ReLU(),
            nn.Linear(int(alpha/8*s**2), int(alpha/8*s**2)),
            nn.BatchNorm1d(int(alpha/8*s**2)),
            nn.ReLU(),
            nn.Linear(int(alpha/8*s**2), int(alpha/16*s**2)),
            nn.BatchNorm1d(int(alpha/16*s**2)),
            nn.ReLU(),
        )
        self.s = s
        self.alpha = alpha
        self.init_method = init_method

        self.classifier = nn.Sequential(
            nn.Linear(int(alpha/16*s**2), 64*alpha),
            nn.BatchNorm1d(int(alpha/16*s**2)),
            nn.ReLU(),
            nn.Linear(64*alpha, self.num_classes),
        )

        # To prevent pruning
        self.classifier[-1].is_classifier = True
        self.init_method = init_method

    def features(self, input):
        # device = (
        #     f'cuda:{os.environ["CUDA_VISIBLE_DEVICE"]}' if \
        #     "CUDA_VISIBLE_DEVICE" in os.environ else 'cuda'
        # ) if input.device is not 'cpu' else 'cpu'
        # with autocast(device_type=device, dtype=torch.float16):
        input = input.reshape(input.size(0), -1)
        x = self.encoder(input)
        return x

    def forward(self, input):
        x = self.features(input)
        device = (
            f'cuda:{os.environ["CUDA_VISIBLE_DEVICE"]}' if \
            "CUDA_VISIBLE_DEVICE" in os.environ else 'cuda'
        ) if input.device != 'cpu' else 'cpu'
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
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(module.bias, -bound, bound)
        self.apply(init_weights)

def dfcnet_debug(num_classes, pretrained=None):
    # For CIFAR 100
    model = DFCNet(num_classes=num_classes, alpha=1)
    return model

def dfcnet(num_classes, pretrained=None):
    # For CIFAR 100
    model = DFCNet(num_classes=num_classes)
    return model

# def dfcnet_100(pretrained=None):
#     # For CIFAR 100
#     model = DFCNet(num_classes=100)
#     return model

def dfcnet_xavier(num_classes, pretrained=None):
    # For CIFAR 100
    model = DFCNet(num_classes=num_classes, init_method='xavier')
    return model


def cnn(num_classes, pretrained=None):
    model = CNN(num_classes=num_classes)
    return model

# def dfcnet_xavier_100(pretrained=None):
#     # For CIFAR 100
#     model = DFCNet(num_classes=100, init_method='xavier')
#     return model
