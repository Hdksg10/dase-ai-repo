import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

x = [1, 2, 3, 4, 5, 6]
y = [89.03, 97.52, 98.74, 98.39, 97.81, 96.77]



ax1.plot(x, y, linestyle='--', marker='o')
ax1.set_xticks([1, 2, 3, 4, 5, 6], ['LeNet', 'AlexNet', 'PlainNet-18', 'PlainNet-34', 'PlainNet-50', 'PlainNet-101'])
ax1.set_title('Deeper networks')
# plt.xlabel('X 轴')
ax1.set_ylabel('Acc(%, vaidation set)')

x_epoch = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
loss_18 = [76.21, 29.04, 27.68, 25.21, 24.61, 23.17, 20.54, 18.98, 17.27, 15.68]
loss_34 = [241.09, 65.10, 47.63, 40.71, 40.25, 37.92, 33.30, 30.73, 30.08, 29.49]
loss_50 = [313.32, 63.56, 49.66, 43.04, 37.65, 37.65, 33.41, 31.57, 30.91, 29.63]
loss_101 = [889.44, 774.51, 349.19, 143.64, 90.91, 66.95, 48.996, 39.45, 36.37, 31.41]
ax2.plot(x_epoch, loss_18, marker='o', label='PlainNet-18', markersize=4)
ax2.plot(x_epoch, loss_34,  marker='o', label='PlainNet-34', markersize=4)
ax2.plot(x_epoch, loss_50,  marker='o', label='PlainNet-50', markersize=4)
ax2.plot(x_epoch, loss_101,  marker='o', label='PlainNet-101', markersize=4)
ax2.set_title('Deeper networks')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Training Loss')
ax2.legend()
plt.tight_layout()
plt.savefig('Deepernetworks.png')