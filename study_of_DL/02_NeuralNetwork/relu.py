import numpy as np
import matplotlib.pylab as plt
from activation import activation

act = activation()

X = np.arange(-5.0, 5.0, 0.1)
Y = act.relu(X)
plt.plot(X, Y)
plt.ylim(-1.0, 5.5)
plt.show()
