# implemention of ResNet-18, but can be easily modified to other ResNet models

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import tqdm

class ResdiaulBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride = 1, residual = "add", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.residual = residual
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, stride=stride),
            # nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(output_channels),
        )
        # we need to change the size of input to match the output using 1x1 conv
        if input_channels != output_channels:
            self.conv3 = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride)
            nn.init.xavier_uniform_(self.conv3.weight)
            nn.init.zeros_(self.conv3.bias)
        else:
            self.conv3 = nn.Identity()
            
        # init weights
        for net in self.conv1:
            if isinstance(net, nn.Conv2d) or isinstance(net, nn.Linear):
                nn.init.xavier_uniform_(net.weight)
                nn.init.zeros_(net.bias)
        
        for net in self.conv2:
            if isinstance(net, nn.Conv2d) or isinstance(net, nn.Linear):
                nn.init.xavier_uniform_(net.weight)
                nn.init.zeros_(net.bias)    
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        x = self.conv3(x)
        if self.residual == "add":
            out = F.relu(x + y)
        elif self.residual == "concat":
            out = F.relu(torch.cat([x, y], dim=1))
        elif self.residual == "minus":
            out = F.relu(x - y)
        elif self.residual == "mul":
            out = F.relu(x * y)
        elif self.residual == "identity":
            out = F.relu(y)
        return out

class ResNetBlock(nn.Module):
    def __init__(self, input_channels:int, output_channels:int, num_blocks:int, stride = 1, residual = "add", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.blocks = nn.Sequential(
            ResdiaulBlock(input_channels, output_channels, stride=stride, residual=residual),
            *[ResdiaulBlock(output_channels, output_channels, residual=residual) for _ in range(num_blocks-1)]
        )
        pass
    def forward(self, x):
        return self.blocks(x)

class ResNetBN(nn.Module):
    def __init__(self, num_blocks_list:list, residual = "add", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert len(num_blocks_list) == 4
        self.pre_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU (),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.blocks = nn.Sequential(
            ResNetBlock(64, 64, num_blocks_list[0], residual=residual),
            ResNetBlock(64, 128, num_blocks_list[1], stride=2, residual=residual),
            ResNetBlock(128, 256, num_blocks_list[2], stride=2, residual=residual),
            ResNetBlock(256, 512, num_blocks_list[3], stride=2, residual=residual)
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
            pbar = tqdm.tqdm(train_loader, position=0, leave=False)
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
                pbar.set_description(f"Epoch {epoch+1} / {epochs}")
                pbar.update()
                pass
            # epoch_loss /= num_batches
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
