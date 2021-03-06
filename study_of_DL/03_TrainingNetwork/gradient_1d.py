# coding: utf-8
import numpy as np
import matplotlib.pylab as plt

# y = 0.01x^2 + 0.1x　についての数値微分

def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)


def function_1(x):
    return 0.01*x**2 + 0.1*x 


def tangent_line(f, x):
    d = numerical_diff(f, x)    #数値微分した値 = 傾き
    print(d)
    y = f(x) - d*x  # 接戦の高さを取得
    return lambda t: d*t + y

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function_1, 5) # 微分した直線を取得
y2 = tf(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.show()
