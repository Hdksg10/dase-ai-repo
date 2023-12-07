import torch
from torchinfo import summary
from models.LeNet import LeNet
from models.AlexNet import AlexNet
from models.ResNet import ResNet
from models.ResNet50 import ResNet50
from models.DenseNet import DenseNet
from models.SEResNet import SEResNet
from models.SEAlexNet import SEAlexNet
from models.SELeNet import SELeNet

input_lenet_sz = (1, 1, 28, 28)
input_others_sz = (4, 3, 224, 224)
input_lenet = torch.randn(input_lenet_sz)
input_others = torch.randn(input_others_sz)
# model = AlexNet()
# macs, params = profile(LeNet(), inputs=(input_lenet, ))
# macs, params = clever_format([macs, params], "%.3f")
# print(f"macs: {macs}, params: {params}")
summary(LeNet(), input_size=input_lenet_sz, device="cpu")
# output = ResNet50([3, 4, 6, 3])(input_others)
# print(output)
# macs, params = profile(ResNet50([3, 4, 6, 3]), inputs=(input_others, ))
# macs, params = clever_format([macs, params], "%.3f")
# print(f"macs: {macs}, params: {params}")
summary(AlexNet(), input_size=input_others_sz, device="cpu")
summary(ResNet([2, 2, 2, 2]), input_size=input_others_sz, device="cpu")
summary(ResNet([3, 4, 6, 3]), input_size=input_others_sz, device="cpu")
summary(ResNet50([3, 4, 6, 3]), input_size=input_others_sz, device="cpu")
summary(ResNet50([3, 4, 23, 3]), input_size=input_others_sz, device="cpu")
summary(DenseNet([2, 2, 2, 2], 32), input_size=input_others_sz, device="cpu")
summary(SEResNet([2, 2, 2, 2]), input_size=input_others_sz, device="cpu")
summary(SEResNet([3, 4, 6, 3]), input_size=input_others_sz, device="cpu")
summary(SEAlexNet(), input_size=input_others_sz, device="cpu")
