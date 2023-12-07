import matplotlib.pyplot as plt
import numpy as np

se18 = [0.0128,0.0050,0.0155,0.0157,0.0240,0.0276,0.0252,0.4912]
se34 = [0.0103,0.0033,0.0041,0.0104,0.0132,0.0084,0.0097,0.0287,0.0306,0.0107,0.0079,0.0053,0.0052,0.0255,0.0465,0.0253]
x1 = np.arange(len(se18))
x2 = np.arange(len(se34))

plt.scatter(x1, se18, label='SEResNet-18')  
plt.scatter(x2, se34, label='SEResNet-34')  
plt.xlabel('SE Blocks Depth')
plt.ylabel('Range')
plt.legend()  

plt.savefig('se.png')