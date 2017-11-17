import numpy as np

def cross_entropy_error(y, t):
    delta = 1e-7    # log(0)は-無限大なので微小値を加算する
    return -np.sum(t * np.log(y + delta))

T = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
Y1 = np.array([0.01, 0.02, 0.8, 0.01, 0.01, 0.02, 0.01, 0.04, 0.03, 0.05])  # 正解
Y2 = np.array([0.01, 0.02, 0.04, 0.01, 0.01, 0.02, 0.01, 0.8, 0.03, 0.05])  # 不正解

E1 = cross_entropy_error(Y1, T)
E2 = cross_entropy_error(Y2, T)

print("error(correct)   = " + str(E1))
print("error(incorrect) = " + str(E2))
