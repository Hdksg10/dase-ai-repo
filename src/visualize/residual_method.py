import matplotlib.pyplot as plt

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
x_ticks = [1, 2, 3 ,4]
acc_add = [98.82, 98.82, 98.20, 98.56]
acc_minus = [98.77, 98.67, 98.25, 98.61]
acc_plain = [98.74, 98.39, 97.82, 96.77]
acc_mul_bn = [87.75, 11.11, 9.68]
acc_mul_noBN = [11.18, 11.18, 11.18, 11.18]
train_loss_18 = [863.1561,863.0347,863.0251,863.0274,863.0077,862.9847,862.9842,862.9845,863.0260,863.0262]
train_loss_34 = [863.1941,863.0280,863.0480,862.9973,862.9743,862.9998,862.9939,862.9918,863.0064,863.0116]
train_loss_50 = [863.1221,863.0368,862.9989,862.9884,863.0152,862.9728,862.9916,863.0470,862.9804,863.0007]
train_loss_101 = [863.1371,863.0210,863.0160,863.0107,863.0185,863.0143,862.9601,863.0013,862.9690,863.0112]
epochs = [1,2,3,4,5,6,7,8,9,10]


ax[0].set_xticks([1, 2, 3, 4], ['ResNet-18', 'ResNet-34', 'ResNet-50', 'ResNet-101'])
ax[0].plot(x_ticks, acc_add, label='add', linestyle='--', marker='o')
ax[0].plot(x_ticks, acc_minus, label='minus', linestyle='--', marker='o')
ax[0].plot(x_ticks, acc_plain, label='plain', linestyle='--', marker='o')
ax0 = ax[0].twinx()
ax0.plot(x_ticks[:3], acc_mul_bn, label='multiply', linestyle='--', marker='o')
ax0.plot(x_ticks, acc_mul_noBN, label='mutiply_without_BN', linestyle='--', marker='o')
ax[0].set_ylabel('Add, Minus, Plain')
ax[0].legend(loc = 'upper right')
ax[0].set_title('Accuracy(%, validation set)')
ax0.legend(loc = 'center left')
ax0.set_ylabel('mutiply, mutiply_without_BN')

ax[1].plot(epochs, train_loss_18, label='ResNet-18')
ax[1].plot(epochs, train_loss_34, label='ResNet-34')
ax[1].plot(epochs, train_loss_50, label='ResNet-50')
ax[1].plot(epochs, train_loss_101, label='ResNet-101')
ax[1].set_title('Training Loss without BN')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Training Loss')
ax[1].legend()
ax[1].set_ylim(862.5, 863.5)
plt.tight_layout()
plt.savefig('residual_method.png')

