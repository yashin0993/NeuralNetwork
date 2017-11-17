import numpy as np

class gate:
    def __init__(self):
        pass

    def OR(self, x1, x2):
        x = np.array([x1, x2])
        w = np.array([0.5, 0.5])
        b = -0.2
        tmp = np.sum(w*x) + b
        if tmp <= 0:
            return 0
        else:
            return 1

    def AND(self, x1, x2):
        x = np.array([x1, x2])
        w = ???
        b = ???
        tmp = np.sum(w*x) + b
        if tmp <= 0:
            return 0
        else:
            return 1

    def NAND(self, x1, x2):
        x = np.array([x1, x2])
        w = ???
        b = ???
        tmp = np.sum(w*x) + b
        if tmp <= 0:
            return 0
        else:
            return 1

    def XOR(self, x1, x2):
        pass