import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

class DenseBlock(nn.Module):
    def __init__(self, input_channels, growth_rate, num_layers, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(input_channels + i*growth_rate),
                nn.ReLU(),
                nn.Conv2d(input_channels + i*growth_rate, growth_rate, kernel_size=3, padding=1, stride=1),
            ))  
        pass
    def forward(self, x):
        for layer in self.layers:
            x = torch.cat([x, layer(x)], dim=1)
        return x
    
class TransitionLayer(nn.Module):
    def __init__(self, input_channels, output_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        pass
    def forward(self, x):
        return self.pool(self.conv(x))
    
class DenseNet(nn.Module):
    def __init__(self, num_block_convs_list, growth_rate, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pre_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU (),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.dense_transition_layers = nn.ModuleList()
        num_channels = 64
        for i in range(len(num_block_convs_list)-1):
            new_num_channels = num_channels + growth_rate * num_block_convs_list[i]
            num_block_convs = num_block_convs_list[i]
            self.dense_transition_layers.append(nn.Sequential(
                DenseBlock(num_channels, growth_rate, num_block_convs),
                TransitionLayer(new_num_channels, (new_num_channels) // 2)
            )) 
            new_num_channels = (new_num_channels) // 2
            num_channels = new_num_channels
        
        self.dense_transition_layers.append(DenseBlock(num_channels, growth_rate, num_block_convs_list[-1]))
        num_channels = num_channels + growth_rate * num_block_convs_list[-1]
        
        self.post_block = nn.Sequential(
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_channels, 1000)
        )
        
        self.adaptor = nn.Sequential(
            nn.Linear(1000, 10),
        )
    
    def forward(self, x):
        x = self.pre_block(x)
        for layer in self.dense_transition_layers:
            x = layer(x)
        x = self.post_block(x)
        return self.adaptor(x)
    
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
        
### test code
# test_tensor = torch.randn(1, 3, 224, 224)
# model = DenseNet([2, 2, 2, 2], 32)
# # print(model(test_tensor).shape)
