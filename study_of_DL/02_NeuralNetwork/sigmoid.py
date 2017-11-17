import numpy as np
import matplotlib.pylab as plt
from activation import activation

act = activation()

X = np.arange(-5.0, 5.0, 0.1)
Y = act.sigmoid(X)
plt.plot(X, Y)
plt.ylim(-0.1, 1.1)
plt.show()
