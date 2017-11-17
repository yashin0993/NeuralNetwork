# coding: utf-8
import sys, os
root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
print("root : " + root)
sys.path.append(root)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from tqdm import tqdm
import pickle
from dataset.mnist import load_mnist
from activation import activation

act = activation()

# 教師データを取得する
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

# 学習済みのパラメータを読み込む
def init_network():
    with open(os.path.join(os.path.dirname(__file__), "sample_weight.pkl"), 'rb') as f:
        network = pickle.load(f)
    return network

# 入力画像がどのラベルに該当するか推定する
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = act.sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = act.sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = act.softmax(a3)

    return y


x, t = get_data()
network = init_network()
print("image size = " + str(len(x)))

accuracy_cnt = 0    # 正解数
for i in tqdm(range(len(x))):
    y = predict(network, x[i])
    # predictの戻り値は確率。最も確率の高い要素のインデックスが正解のラベルなので、
    # 教師データtと同じであれば問題に正解したとして、正解数をインクリメント
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
