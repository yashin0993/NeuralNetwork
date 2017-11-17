import numpy as np

class activation:
    def __init__(self):
        pass

    def step(self, x):
        return np.array(x > 0, np.int)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def identity(self, x):
        return x

    def softmax(self, x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T

        x = x-np.max(x) # オーバーフロー対策
        return np.exp(x) / np.sum(np.exp(x), axis=0)