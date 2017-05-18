import os
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

root = os.path.dirname(__file__)

#mnistデータ取得
mnist = input_data.read_data_sets(os.path.join(root, "data"), one_hot=True)

# sessionの作成
sess = tf.InteractiveSession()

# 入力データ整形
x = tf.placeholder(tf.float32, [None, 784])
# 重みパラメータ取得
W = tf.Variable(tf.zeros([784, 10]))
# バイアス取得
b = tf.Variable(tf.zeros([10]))
# 出力層
y = tf.nn.softmax(tf.matmul(x,W) + b)

# 全結合層を定義
y_ = tf.placeholder(tf.float32, [None, 10])
# 損失率取得関数
cross_entropy = -tf.reduce_sum(y_*tf.log(y)) 
# 勾配の最小値を取得
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# initialize
tf.global_variables_initializer().run()

batch_size = 100


# 学習開始
for i in range(1000):
    print
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    train_step.run({x: batch_xs, y_:batch_ys})

# 学習済みモデルにテストデータを入れてテスト
print(sess.run(tf.argmax(y, 1), feed_dict={x: [mnist.test.images[0]]}))
print(mnist.test.labels[0])

# 学習済みモデルの正解率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x:mnist.test.images, y_:mnist.test.labels}))