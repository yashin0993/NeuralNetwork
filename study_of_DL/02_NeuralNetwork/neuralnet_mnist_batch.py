# coding: utf-8
import sys, os
root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
print("root : " + root)
sys.path.append(root)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pickle
from tqdm import tqdm
from dataset.mnist import load_mnist
from activation import activation

act = activation()

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open(os.path.join(os.path.dirname(__file__), "sample_weight.pkl"), 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = act.sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = act.sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = act.softmax(a3)

    return y


x, t = get_data()
network = init_network()
print("image size = " + str(len(x)))

batch_size = 100 # バッチの数
accuracy_cnt = 0

for i in tqdm(range(0, len(x), batch_size)):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
