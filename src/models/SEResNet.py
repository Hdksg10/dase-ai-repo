# SE-ResNet18

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .ResNet import ResdiaulBlock

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
        # z = y
        # shape = z.shape
        # shape = shape[1]
        # # z = F.softmax(z, dim=1)
        # # entropy = -(z * torch.log2(z)).sum() / shape
        # var = torch.var(z, dim=1)
        # range_tensor = torch.max(z) - torch.min(z)
        # print(range_tensor)
        # y = y.expand_as(x)
        return x * y

class SEResidualBlock(ResdiaulBlock):
    def __init__(self, input_channels, output_channels, ratio = 2, stride = 1, *args, **kwargs) -> None:
        super().__init__(input_channels, output_channels, stride=stride, *args, **kwargs)
        self.se = SEBlock(output_channels, ratio)
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        x = self.conv3(x)
        return F.relu(self.se(y) + x)

class SENetBlock(nn.Module):
    def __init__(self, input_channels:int, output_channels:int, num_blocks:int, stride = 1, ratio = 2,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.blocks = nn.Sequential(
            SEResidualBlock(input_channels, output_channels, stride=stride, ratio=ratio),
            *[SEResidualBlock(output_channels, output_channels, ratio=ratio) for _ in range(num_blocks-1)]
        )
        pass
    def forward(self, x):
        return self.blocks(x)

class SEResNet(nn.Module):
    def __init__(self, num_blocks_list:list, ratio=2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert len(num_blocks_list) == 4
        self.pre_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU (),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.blocks = nn.Sequential(
            SENetBlock(64, 64, num_blocks_list[0], ratio=ratio),
            SENetBlock(64, 128, num_blocks_list[1], stride=2, ratio=ratio),
            SENetBlock(128, 256, num_blocks_list[2], stride=2, ratio=ratio),
            SENetBlock(256, 512, num_blocks_list[3], stride=2, ratio=ratio)
        )
        
        self.post_block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 1000)
        )
        
        self.adaptor = nn.Sequential(
            nn.Linear(1000, 10),
        )
        
        for net in self.pre_block:
            if isinstance(net, nn.Conv2d) or isinstance(net, nn.Linear):
                nn.init.xavier_uniform_(net.weight)
                nn.init.zeros_(net.bias)

        for net in self.post_block:
            if isinstance(net, nn.Conv2d) or isinstance(net, nn.Linear):
                nn.init.xavier_uniform_(net.weight)
                nn.init.zeros_(net.bias)
        
        for net in self.adaptor:
            if isinstance(net, nn.Conv2d) or isinstance(net, nn.Linear):
                nn.init.xavier_uniform_(net.weight)
                nn.init.zeros_(net.bias)
        pass
    
    def forward(self, x):
        return self.adaptor(self.post_block(self.blocks(self.pre_block(x))))
        pass
    
    def train_model(self, train_loader: torch.utils.data.DataLoader, epochs:float, lr:float, **kwargs):
        device = kwargs.get("device", torch.device("cpu"))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=5e-4)
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = len(train_loader)
            for input, labels in train_loader:
                batch_size = input.shape[0]
                input = input.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                output = self(input)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                pass
            epoch_loss /= num_batches
            print(f"Epoch {epoch+1} / {epochs} finished! Loss: {epoch_loss}")
        pass
    
    def validate(self, valid_loader: torch.utils.data.DataLoader, **kwargs):
        device = kwargs.get("device", torch.device("cpu"))
        correct = 0
        total = 0
        with torch.no_grad():
            for input, labels in valid_loader:
                input = input.to(device)
                labels = labels.to(device)
                output = self(input)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pass
        
        accuracy = 100 * correct / total
        return accuracy
        pass

### test
# x = torch.randn(1, 64, 32, 32)
# se = SEBlock(64, 16)
# print(se(x).shape) 