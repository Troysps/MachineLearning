#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 疝气病症预测病马的死亡率 --- logistics classify
__author__ = 'LEI'
__mtime__ = '2018/5/14'
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
"""
    训练集:HorseColicTraining.txt
    测试集:HorseColicTest.txt
"""
import codecs
import pandas as pd
import numpy as np


def loadDataSet(filename):

    data = []
    with codecs.open(filename, 'r', 'utf-8') as f:
        for line in f.readlines():
            data.append(line.strip().split('\t'))
        f.close()
    print(np.shape(data))

    data = pd.DataFrame(data, dtype='float') # 转化数据为浮点数
    # print(data) # 查看数据
    # print(data.info())  # 数据类型 float 行列 299x22 有无空值
    # print(data.iloc[:, -1])  # 索引最后一行 # 最后一列为输出空间
    data['x0'] = 1.0   # 设置x0
    # print(data.info)  # 数据类型 float 行列 299x23 有无空值

    # print(data.columns)
    """
        查看数据列名 
        Index([   0,    1,    2,    3,    4,    5,    6,    7,    8,    9,   10,   11,
         12,   13,   14,   15,   16,   17,   18,   19,   20,   21, 'x0'],
      dtype='object')
    """
    data = data.reindex(columns=['x0',  0,    1,    2,    3,    4,    5,    6,    7,    8,    9,   10,   11, 12,   13,  14,   15,   16,   17,   18,   19,   20,   21])
    # print(data.columns)
    # print(data.describe())
    # print(data.info())
    dataSet = data.iloc[:, :-1]  # 输入空间X
    classLabel = data.iloc[:, -1:]  # 输出空间Y

    # print(dataSet)
    # print(dataSet.info())
    # print(classLabel)
    # print(classLabel.info)
    return dataSet, classLabel

def sigmoid(inX):
    """ 监督学习 模型
            二分类问题
            设置阈值: >0.5  return 1
                     <=0.5 return 0
    :param inX: 输入数据集
    :return g(z):  跃阶函数 --- 实际上就是概率计算
    """
    return 1.0 / (1 + np.exp(-inX))

# 批量梯度上升法
def gradientUp(dataSet, classLabel, alpha=0.001, maxCycle=1000):
    """
        批量梯度上升法:计算量较大 -- 优化为
    :param dataSet: 训练集 输入空间
    :param classLabel: 输出空间
    :param alpha: 学习速率
    :param maxCycle: 最大迭代次数
    :return: tehta 值集合
    """
    dataSet = np.mat(dataSet)
    classLabel = np.mat(classLabel)

    # print(dataSet)
    # print('------dataSet shape------')
    # print(np.shape(dataSet))  #(299, 22)
    # print(classLabel)
    # print('-------classLabel shape----')
    # print(np.shape(classLabel))  #(299, 1)

    m, n = np.shape(dataSet)
    print('m, n', m, n)
    theta = np.ones((n, 1))

    # alpha = 0.0001
    # maxCycle = 1000


    for i in range(maxCycle):
        # 梯度上升法
        hx = sigmoid(dataSet * theta)
        error = (classLabel - hx)
        # print('error------------------------------------\n', error)
        theta = theta + alpha * dataSet.transpose() * error


    return theta

def stocGradientUp(dataSet, classLabel, alpha=0.001, maxCycle=1000):
    """
    随机梯度上升法 一次只使用一个样本点来更新回归系数
    :param dataSet:  训练数据集 输入空间X
    :param classLabel: 训练数据集 输出空间 Y
    :param alpha: 学习速率
    :param maxCycle: 迭代次数
    :return: 模型参数theta集合
    """
    dataSet = np.mat(dataSet)
    classLabel = np.mat(classLabel)

    m, n = np.shape(dataSet)
    print('m, n', m, n)
    print(np.shape(dataSet[0]))
    theta = np.ones((n, 1))

    # alpha = 0.0001
    # maxCycle = 1000
    for i in range(m):
        # 梯度上升法
        hx = sigmoid(dataSet[i] * theta)
        error = (classLabel[i] - hx)
        # print('error------------------------------------\n', error)
        theta = theta + alpha * dataSet[i].transpose() * error

    return theta

def stocGradientUp1(dataSet, classLabel, numIter=150):
    """
    随机梯度上升法(随机化)
    :param dataSet:  训练数据集 输入空间X
    :param classLabel: 训练数据集 输出空间 Y
    :param alpha: 学习速率
    :param maxCycle: 迭代次数
    :return: 模型参数theta集合
    """
    dataSet = np.matrix(dataSet)
    classLabel = np.matrix(classLabel)

    m, n = np.shape(dataSet)
    print('m, n', m, n)
    print(np.shape(dataSet[0]))
    theta = np.ones((n, 1))

    # alpha = 0.0001
    # maxCycle = 1000
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            # i和j的不断增大，导致alpha的值不断减少，但是不为0
            alpha = 4 / (1.0 + j + i) + 0.0001  # alpha 会随着迭代不断减小，但永远不会减小到0，因为后边还有一个常数项0.0001
            # 梯度上升法
            # 随机选取
            randIndex = int(np.random.randint(0, len(dataIndex)))

            hx = sigmoid(dataSet[randIndex] * theta)
            error = classLabel[randIndex] - hx
            # print('error------------------------------------\n', error)
            # print('xxxxxxxxxxxx')
            # print('theta shape', np.shape(theta))  # (22, 1)
            # print(dataSet[randIndex])  # (1, 22)
            # print('dataSet shape', np.shape(dataSet[randIndex]))  # (1, 22)
            # print('error', error)
            # print('error shape', np.shape(error))
            # print('alpha', alpha)
            theta = theta + alpha * dataSet[randIndex].transpose() * error
            del(dataIndex[randIndex])

    return theta


# def stocGradAscent1(dataMatrix, classLabels, numIter=150):
#     m, n = np.shape(dataMatrix)
#     print('m, n', m, n)
#     weights = np.ones(n)  # 创建与列数相同的矩阵的系数矩阵，所有的元素都是1
#     # 随机梯度, 循环150,观察是否收敛
#     print(dataMatrix[0])
#     print(np.shape(dataMatrix[0]))
#     for j in range(numIter):
#         # [0, 1, 2 .. m-1]
#         dataIndex = list(range(m))
#         for i in range(m):
#             # i和j的不断增大，导致alpha的值不断减少，但是不为0
#             alpha = 4 / (1.0 + j + i) + 0.0001  # alpha 会随着迭代不断减小，但永远不会减小到0，因为后边还有一个常数项0.0001
#             # 随机产生一个 0～len()之间的一个值
#             # random.uniform(x, y) 方法将随机生成下一个实数，它在[x,y]范围内,x是这个范围内的最小值，y是这个范围内的最大值。
#             randIndex = int(np.random.randint(0, len(dataIndex)))
#             # sum(dataMatrix[i]*weights)为了求 f(x)的值， f(x)=a1*x1+b2*x2+..+nn*xn
#             h = sigmoid(sum(dataMatrix[randIndex] * weights))
#             error = classLabels[randIndex] - h
#             # print weights, '__h=%s' % h, '__'*20, alpha, '__'*20, error, '__'*20, dataMatrix[randIndex]
#
#             weights = weights + alpha * error * dataMatrix[randIndex]
#             del dataIndex[randIndex]
#     return weights

def logisticsClassifier(testData, theta):
    """

    :param testData: 测试数据
    :param theta: 训练出的 theta集合
    :return: 预测分类
    """
    prob = sigmoid(testData * theta)

    if prob > 0.5:
        return 1.0
    return 0.0



def test(theta, testData, testClass):
    """

    :param theta: 训练出来的tehta参数集合
    :param testData: 测试数据 输入空间
    :param testClass: 测试数据 输出空间
    :return: rightRation 正确率
             count       分类错误数
    """
    testData = np.mat(testData)
    testClass = np.mat(testClass)
    count = 0
    for index in range(len(testData)):
        # print(index)
        # print(testData[index])
        predict = logisticsClassifier(testData[index], theta)
        right_result = testClass[index]
        if predict != right_result:
            print('predict error, predict:{}, right_result:{}'.format(predict, right_result))
            count += 1
    rightRatio = float(len(testData) - count) / float(len(testData))

    print('正确率为', rightRatio)
    print('错误数为', count)

    return rightRatio


def main():
    trainingFilename = r'C:\Users\Administrator\Documents\数据挖掘常用算法\logistic regreesion\疝气病症预测病马的死亡率\HorseColicTraining.txt'
    testFilename = r'C:\Users\Administrator\Documents\数据挖掘常用算法\logistic regreesion\疝气病症预测病马的死亡率\HorseColicTest.txt'
    trainingData, trainingClass = loadDataSet(trainingFilename)
    testData, testClass = loadDataSet(testFilename)
    print(np.shape(trainingData), np.shape(trainingClass))
    print(np.shape(testData), np.shape(testClass))

    #theta = gradientUp(trainingData, trainingClass)
    # print(theta)
    # print(np.shape(theta))
    # test(theta, testData, testClass)

    numTests = 10
    rightSum = 0.0
    for k in range(numTests):
        theta = gradientUp(trainingData, trainingClass) #正确率 72% 28% 错误
        # theta = stocGradientUp(trainingData, trainingClass) #正确率 52%  48% 错误
        # theta = stocGradientUp1(trainingData, trainingClass, numIter=150) # 正确率 63% 37% 错误
        rightSum += test(theta, testData, testClass)
        print("after %d iterations the average right rate is: %f" % (numTests, rightSum / float(numTests)))

if __name__ == '__main__':
    main()