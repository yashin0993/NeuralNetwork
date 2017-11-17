import numpy as np
import matplotlib.pylab as plt
from activation import activation

act = activation()

# -5～5の間を間隔0.1の配列を生成
X = np.arange(-5.0, 5.0, 0.1)
Y = act.step(X)
plt.plot(X, Y)
plt.ylim(-0.1, 1.1)
plt.show()
