import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import tqdm

class LeNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16*5*5, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, 10)
        )
        for net in self.net:
            if isinstance(net, nn.Conv2d) or isinstance(net, nn.Linear):
                nn.init.xavier_uniform_(net.weight)
                nn.init.zeros_(net.bias)
        pass
    
    def forward(self, x):
        return self.net(x)
    
    def train_model(self, train_loader: torch.utils.data.DataLoader, epochs:float, lr:float, **kwargs):
        momentum = kwargs.get("momentum", 0.9)
        device = kwargs.get("device", torch.device("cpu"))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)
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
            epoch_loss /= num_batches
            # print(f"Epoch {epoch+1} / {epochs} finished! Loss: {epoch_loss}")
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
