import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from PIL import Image
import matplotlib.pyplot as plt

data_path = "../../data"

target_size = (224, 224)

mnist = MNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())

index = 0  
image, _ = mnist[index]
image_pil = transforms.ToPILImage()(image)

transform = transforms.Compose([
    transforms.Resize(target_size),
])

resized_image = transform(image_pil)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(image_pil, cmap='gray')
axs[0].set_title('Original Image')
axs[0].axis('off')

axs[1].imshow(resized_image, cmap='gray')
axs[1].set_title('Resized Image')
axs[1].axis('off')

plt.tight_layout()
plt.show()
plt.savefig("resize.png")
