import numpy as np
from activation import activation

act = activation()

def init_network():
    network = {}
    network['w1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['w2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])

    return network

def forward(network, x):
    w1, w2 = network['w1'], network['w2']
    b1, b2 = network['b1'], network['b2']

    a1 = np.dot(x, w1) + b1
    z1 = act.sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    y = act.softmax(a2)

    return y

# --------- メイン処理 -----------
network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print("output = " + str(y))
print("sum(y1, y2) = " + str(y[0]+y[1]))

