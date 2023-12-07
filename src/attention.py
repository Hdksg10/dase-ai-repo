import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from models.SEResNet import SEResNet

def load_mnist(data_path: str, batch_size: int, transform: transforms.Compose, valid_ratio: float = 0.2):
    # Download MNIST dataset and normalize it
    mnist_train = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
    validation_size = int(len(mnist_train) * valid_ratio)
    train_size = len(mnist_train) - validation_size
    generator = torch.Generator().manual_seed(411)
    mnist_train, mnist_valid = torch.utils.data.random_split(mnist_train, [train_size, validation_size], generator=generator)
    # Create dataloader
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(mnist_valid, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True)
    return train_loader, valid_loader, test_loader

data_path = "../data"
config_path = "../config"
ckpt_path = "../ckpt"
transform_others = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.Grayscale(num_output_channels=3),  
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
train_loader, valid_loader, test_loader = load_mnist(data_path, batch_size=1, transform=transform_others)

state = torch.load('../ckpt/SEResNet34.pth')
# seresnet18 = SEResNet([2, 2, 2, 2], ratio=2)
seresnet18 = SEResNet([3, 4, 6, 3], ratio=2)
seresnet18.load_state_dict(state)

seresnet18.eval()
data, target = next(iter(train_loader))
x = data
print(x.shape)

y = seresnet18(x)
