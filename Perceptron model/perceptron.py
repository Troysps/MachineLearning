# -*- coding: utf-8 -*-

"""
__title__ = "Perceptron model"
__author__ = "Lei"
__mtime__ = "2018/08/03"
"""
import numpy as np

# 感知机学习算法的原始形式实现
def load_data():
    data_set = [[3, 3], [4, 3], [1, 1]]
    label_set = [1, 1, -1]
    return np.mat(data_set), label_set

def sign(point, w, b):
    """
    实际上 numpy中已经封装了该函数
    :param point:   测试数据点
    :param w: w
    :param b: b
    :return: {+1, -1}
    """
    if np.dot(point, w) + b > 0:
        return 1
    elif np.dot(point, w) + b < 0:
        return -1


def original_perceptron_model(data_set, label_set, max_cycle, alpha):
    """
    感知机学习算法的原始形式
    :param data_set: 输入空间
    :param label_set: 输出空间
    :param max_cycle:  最大迭代次数
    :param alpha:   学习速率(0, 1]
    :returns
        w   向量空间参数
        b   残差项
    """
    x = data_set
    y = label_set
    m, n = np.shape(data_set)
    w = np.zeros((n, 1))
    b = 0
    # for _ in range(max_cycle):
    count = 0
    while count < max_cycle:
        count += 1
        for i in range(m):
            # print(y[i] * (x[i] * w + b))
            # 随机选取一个误分类点 使其梯度下降
            if y[i] * (x[i] * w + b) <= 0:
                w = w + alpha*y[i]*x[i].T
                b = b + alpha*y[i]
            # 判断训练数据中是否存在误分类点
            error_set = np.multiply(np.mat(y).T, (x * w + b))
            if (error_set > 0).all():
                print(w, b, count)
                return w, b

def calc_gram(data_set):
    m = np.shape(data_set)[0]
    gram_mat = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            gram_mat[i][j] = data_set[i] * data_set[j].T
    return np.mat(gram_mat)


def have_no_any_error(data_set, label_set, a, b):
    print("np.shape a", a, np.shape(a))
    m, n = np.shape(data_set)
    w = np.sum(np.multiply(np.multiply(a, np.mat(label_set)).T, data_set), axis=0)
    if (np.multiply(np.mat(label_set).T, (data_set * w.T + b)) > 0).all():
        return True
    else:
        return False



def dual_perceptron_model(data_set ,label_set, max_cycle, alpha):
    """
    感知机学习的对偶形式
    :param data_set:
    :param label_set:
    :param max_cycle:
    :param alpha:
    :return:
    """
    m = np.shape(data_set)[0]
    a = np.zeros((1, m))
    x = data_set
    y = label_set
    b = 0
    alpha = 1
    gram_mat = calc_gram(data_set)
    # while True:
    for _ in range(10):
        for i in range(m):
            # print(a * np.mat(y).T * gram_mat[i, i] + b)
            if y[i] * (np.multiply(a, np.mat(y)) * gram_mat[:, i] + b) <= 0:
                # print(a[:, i])
                a[:, i] = a[:, i] + 1
                b = b + y[i]
            # print(a, b)
            if have_no_any_error(data_set, label_set, a, b):
                return a, b
        # if :
        #     return a, b





def main():
    data_set, label_set = load_data()
    # print('data_set\n', data_set, '\n', 'label_set\n', label_set)
    w, b = original_perceptron_model(data_set, label_set, max_cycle=10, alpha=1)
    gram_mat = calc_gram(data_set)
    # print(gram_mat)
    dual_perceptron_model(data_set, label_set, max_cycle=10, alpha=1)


if __name__ == '__main__':
    main()