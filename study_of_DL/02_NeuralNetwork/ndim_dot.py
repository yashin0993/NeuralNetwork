import numpy as np

# 多次元配列の内積計算
X = np.array([1.0, 0.5])
print("X = ")
print(X)
print("dim : " + str(np.ndim(X)))
print("shape : " + str(X.shape))

W = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
print("\nW = ")
print(W)
print("dim : " + str(np.ndim(W)))
print("shape : " + str(W.shape))

B = np.array([0.1, 0.2, 0.3])
print("\nB = ")
print(B)
print("dim : " + str(np.ndim(B)))
print("shape : " + str(B.shape))

print("\nX・W + B = ")
print(np.dot(X, W) + B)

