import numpy as np


a = [2,4,6]

array = np.linspace(1,10,10)

array2 = array.reshape(2,5)

array2[1,3]

array2[:,0:3]

array2[1,:]


x = np.arange(1, 11)

y = x**2
print(y)

import matplotlib.pyplot as plt

plt.plot(x, y, color='red')