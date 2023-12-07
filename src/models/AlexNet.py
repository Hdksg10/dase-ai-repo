import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import tqdm

class AlexNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.feature = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 1000)
        )
        self.adaptor = nn.Sequential(
            nn.Linear(1000, 10),
            nn.Softmax(dim = 1)
        )
        for net in self.feature:
            if isinstance(net, nn.Conv2d) or isinstance(net, nn.Linear):
                nn.init.xavier_uniform_(net.weight)
                nn.init.zeros_(net.bias)
        
        for net in self.classifier:
            if isinstance(net, nn.Conv2d) or isinstance(net, nn.Linear):
                nn.init.xavier_uniform_(net.weight)
                nn.init.zeros_(net.bias)
        
        for net in self.adaptor:
            if isinstance(net, nn.Conv2d) or isinstance(net, nn.Linear):
                nn.init.xavier_uniform_(net.weight)
                nn.init.zeros_(net.bias)
        pass
    
    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        x = self.adaptor(x)
        return x
    
    def train_model(self, train_loader: torch.utils.data.DataLoader, epochs:float, lr:float, **kwargs):
        momentum = kwargs.get("momentum", 0.9)
        device = kwargs.get("device", torch.device("cpu"))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=lr, weight_decay=5e-4, momentum=momentum)
        scheuler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10, verbose=True)
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
            scheuler.step(epoch_loss)
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
