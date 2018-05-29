#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = Linear regression   简单线性回归
__author__ = LEI
__mtime__ = '2018/4/25'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃      ☃      ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""

"""简单线性回归实现
    假设函数h(x) = Oo + O1X
    模型参数 Oo + O1
    思考问题:
        如何拟合模型参数 使得假设函数不断逼近实际值
        
    问题转化:
        即预测值与实际值的差距最小 -- 代价函数最小化问题
"""
import numpy as np



"""梯度下降法
    ---方法实现1 比实现2精度更高?
    ---为啥子
"""
def createData():
    x = [1, 2, 3, 4, 5, 6]
    y = [13, 14, 20, 21, 25, 30]

    return np.array(x), np.array(y)


# 简单线性回归 实现
def linearRegression(x, y):
    """
    cost function min
    同时更新theta0 theta1
    目标:代价函数最小化
    :param x: 特征值x
    :param y: 目标变量y 实际值y
    :return: h(x)
    """
    epsilon = 0.01
    error = 0   #
    alpha = 0.001     # 步长
    maxCycle = 20   # 最大迭代次数
    count = 0
    theta0 = 0
    theta1 = 0

    m = len(x)
    while True:
        diff = [0, 0]
        count += 1
        for i in range(m):
            diff[0] += theta0 + theta1 * x[i] -y[i]
            diff[1] += (theta0 + theta1 * x[i] -y[i]) * x[i]
        print('diff---------------------', diff)
        theta0 = theta0 - alpha/m * diff[0]
        theta1 = theta1 - alpha/m * diff[1]
        print('theta0', theta0)
        print('theta1', theta1)
        error1 = 0
        for i in range(m):
            error1 += (theta0 + theta1 * x[i] - y[i]) ** 2
        if abs(error1 - error) < epsilon:
            break

        if count > 200:
            break

    print(theta0, theta1, error1)
    return theta0, theta1, error1



def linearRegression2(x, y):
    x = np.mat(x)
    y = np.mat(y)

    m, n = np.shape(x)
    print(m, n)
    weights = np.zeros((1, m+1))
    print(np.shape(weights))

    alpha = 0.0013
    esplion = 70
    maxCycle= 2000
    count = 0
    while True:
        count += 1
        diff = [0, 0]
        diff[0] = np.sum(x.T * weights - y.T)
        print(diff[0])
        diff[1] = np.sum((x.T * weights - y.T).T * x.T)
        print(diff[1])
        print(diff)
        weights[0, 0] = weights[0, 0] - alpha/(2*n) * diff[0]
        print(weights)
        weights[0, 1] = weights[0, 1] - alpha/(2*n) * diff[1]


        error = np.sum(np.array(x.T * weights - y.T) ** 2)
        if error < esplion:
            break
        if count > maxCycle:
            break

        print(weights, '-------------------', error)
    return weights, error


def linearRegression3(x, y):
    xMat = np.mat(x).transpose()

    print(xMat)
    yMat = np.mat(y).transpose()
    print(yMat)

    m, n = np.shape(xMat)
    print(m, n)

    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    print(weights)

    for i in range(maxCycles):
        hx = x * weights
        error = np.sum(y - hx, axis=1) ** 2
        print('error---------------------', error)

        weights = weights - alpha * xMat.transpose() * error.tolist()[0]

        print('weight--------------------', weights)



if __name__ == '__main__':
    x, y = createData()
    # print(x) [1, 2, 3, 4, 5, 6]
    # print(y) [13, 14, 20, 21, 25, 30]

    # weights = linearRegression(x, y)
    # print(weights)

    # print(linearRegression2(x, y))
    # print(x)
    # print(np.shape(np.mat(x).T)) (6, 1)

    linearRegression3(x, y)

