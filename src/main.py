import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from models.LeNet import LeNet
from models.AlexNet import AlexNet
from models.ResNet import ResNet
from models.DenseNet import DenseNet
from models.SEResNet import SEResNet
from models.SEAlexNet import SEAlexNet
from models.SELeNet import SELeNet
from models.ResNet50 import ResNet50
from models.ResNetBN import ResNetBN
from models.ResNet50BN import ResNet50BN
import argparse


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
# lenet
transform_lenet = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# image net
transform_others = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.Grayscale(num_output_channels=3),  
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="LeNet", help="model name")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--epochs", type=int, default=10, help="epochs")
parser.add_argument("--residual", type=str, default="add", help="residual type")
parser.add_argument("--gr", type=int, default=32, help="growth rate(DenseNet)")
parser.add_argument("--ratio", type=int, default=16, help="SE ratio")
parser.add_argument("--noresize", action="store_true", help="resize image")
parser.add_argument("--test", action="store_true", help="use test set")
args = parser.parse_args()
print(args.noresize)
config = {}
config["model"] = args.model
config["lr"] = args.lr
config["epochs"] = args.epochs
if args.model == "LeNet":
    args.lr = 1e-2
    model = LeNet()
    model.to(device)
    transform = transform_lenet
elif args.model == "AlexNet":
    model = AlexNet()
    model.to(device)
    transform = transform_others
    print(transform)
# For resnet if args.residual == 'idenetity', then residual block does not have any residual connection, which is equivalent to a plain block.
elif args.model == "ResNet18":
    config["residual"] = args.residual
    model = ResNet([2, 2, 2, 2], args.residual)
    model.to(device)
    transform = transform_others
elif args.model == "ResNet34":
    config["residual"] = args.residual
    model = ResNet([3, 4, 6, 3], args.residual)
    model.to(device)
    transform = transform_others
elif args.model == "ResNet50":
    config["residual"] = args.residual
    model = ResNet50([3, 4, 6, 3], args.residual)
    model.to(device)
    transform = transform_others
elif args.model == "ResNet101":
    config["residual"] = args.residual
    model = ResNet50([3, 4, 23, 3], args.residual)
    model.to(device)
    transform = transform_others
elif args.model == "ResNet18BN":
    config["residual"] = args.residual
    model = ResNetBN([2, 2, 2, 2], args.residual)
    model.to(device)
    transform = transform_others
elif args.model == "ResNet34BN":
    config["residual"] = args.residual
    model = ResNetBN([3, 4, 6, 3], args.residual)
    model.to(device)
    transform = transform_others
elif args.model == "ResNet50BN":
    config["residual"] = args.residual
    model = ResNet50BN([3, 4, 6, 3], args.residual)
    model.to(device)
    transform = transform_others
elif args.model == "ResNet101BN":
    config["residual"] = args.residual
    model = ResNet50BN([3, 4, 23, 3], args.residual)
    model.to(device)
    transform = transform_others
elif args.model == "DenseNet":
    config["gr"] = args.gr
    model = DenseNet([2, 2, 2, 2], args.gr)
    model.to(device)
    transform = transform_others
elif args.model == "SEResNet18":
    config["ratio"] = args.ratio
    model = SEResNet([2, 2, 2, 2], ratio=args.ratio)
    model.to(device)
    transform = transform_others
elif args.model == "SEResNet34":
    config["ratio"] = args.ratio
    model = SEResNet([3, 4, 6, 3], ratio=args.ratio)
    model.to(device)
    transform = transform_others
elif args.model == "SEAlexNet":
    config["ratio"] = args.ratio
    model = SEAlexNet(ratio=args.ratio)
    model.to(device)
    transform = transform_others

else:
    raise ValueError("Unsupported model name!")

train_loader, valid_loader, test_loader = load_mnist(data_path, batch_size=128, transform=transform)
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')
model.train_model(train_loader, epochs=args.epochs, lr=args.lr, device=device)

if args.test:
    valid_loader = test_loader
accuracy = model.validate(valid_loader, device=device)
print("Config:")
print(config)
print(f"Validation accuracy: {accuracy}")
torch.save(model.state_dict(), f"{ckpt_path}/{args.model}.pth")