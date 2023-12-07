import matplotlib.pyplot as plt
import numpy as np

x_ticks = np.linspace(1, 21, num=20)
loss_18 = [78.9662,23.9144,22.0282,20.9600,18.5381,18.7033,16.5203,15.8900,15.0076,13.0906,12.7733,11.4877,10.6299,10.1920,9.2607,9.7954,8.9278,7.6931,8.1058,8.0052]
loss_34 = [183.3912,30.7912,26.3757,24.1056,22.6118,20.6849,18.6926,17.7086,15.6896,14.5987,13.2664,12.1660,12.0657,11.3636,10.2648,10.1608,9.7113,8.5469,9.2779,8.9679]

plt.plot(x_ticks, loss_18, label="ResNet-18")
plt.plot(x_ticks, loss_34, label = "ResNet-34")
plt.title('Train Loss Curve')
plt.xlabel('epochs')
plt.ylabel('Training loss')
plt.legend()

plt.savefig("residual.png")