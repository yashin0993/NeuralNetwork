# coding: utf-8
import sys, os
root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
print("root : " + root)
sys.path.append(root)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000               # 繰り返しの回数を適宜設定する
train_size = x_train.shape[0]   # 全データ数
batch_size = 100                # 1回の学習でまとめて入力するデータの数
learning_rate = 0.1             # 学習率

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

# パラメータの更新を1万回繰り返す
for i in range(iters_num):
    # 学習データの中からバッチサイズ分ランダムな数字(index)を取得する
    batch_mask = np.random.choice(train_size, batch_size)

    # indexに該当する学習データを取得
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配の計算
    grad = network.numerical_gradient(x_batch, t_batch)
    # grad = network.gradient(x_batch, t_batch)

    # パラメータの更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1エポックが終わるたびに精度を確認
    if i % iter_per_epoch == 0:
        # 学習に使用したtrainデータでの精度と、使用しなかったtestデータの精度を比較し、
        # 未知のデータにも対応できているかチェック
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + '{:0<15}'.format(str(train_acc)) + ", " + str(test_acc))

# グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()