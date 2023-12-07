# SE-Alexnet, perfrom badly

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .AlexNet import AlexNet

class SEBlock(nn.Module):
    def __init__(self, input_channels, ratio, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.excitation = nn.Sequential(
            nn.Linear(input_channels, input_channels // ratio),
            nn.ReLU(),
            nn.Linear(input_channels // ratio, input_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        y = self.squeeze(x).squeeze(dim = [-1, -2])
        y = self.excitation(y)
        y = y.unsqueeze(-1).unsqueeze(-1)
        # y = y.expand_as(x)
        return x * y

class SEAlexNet(AlexNet):
    def __init__(self, ratio=2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # self.se = SEBlock(256, 2)
        self.feature = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            SEBlock(384, ratio), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), 
            SEBlock(384, ratio), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), 
            SEBlock(256, ratio), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten()
        )
        pass
    
    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        # x = self.se(x)
        x = self.adaptor(x)
        return x
    
    def train_model(self, train_loader: torch.utils.data.DataLoader, epochs:float, lr:float, **kwargs):
        super().train_model(train_loader, epochs, lr, **kwargs)
        pass
    pass