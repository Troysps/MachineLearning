#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = KNN REVIREW
__author__ = 'Administrator'
__mtime__ = '2018/4/23'
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
import numpy as np
import math
import operator
"""knn算法优化
    :param  dataSet 无量纲化
    :func   knn weight
    :return 系统体系
    other
    :knn logistic regression
"""


def createDataSet():
    group = np.array([[10, 9.2], [8.8, 7.9], [0.1, 1.4], [0.3, 3.5]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def normDatafunc(dataSet):
    print(dataSet)
    print(np.max(dataSet))
    print(np.min(dataSet))
    dataMax = np.max(dataSet)
    dataMin = np.min(dataSet)
    dataRange = dataMax - dataMin
    normData = (dataSet-dataMin)/dataRange
    return dataRange, dataMin, dataMax, normData

def normTest(dataMin, dataRange, testSet):
    test = (testSet-dataMin)/dataRange
    return test

def gaussian(dist, a=1, b=0, c=0.3):
    return a * np.math.e ** (-(dist - b) ** 2 / (2 * c ** 2))


def knnClassify(dataSet, Labels, testSet, k):
    nrow = np.shape(dataSet)[0]
    cdiff = np.tile(testSet, (nrow, 1)) - dataSet
    sqdiff = cdiff ** 2
    sumdiff = np.sum(sqdiff, axis=1)
    dist = np.sqrt(sumdiff)
    print('dist', dist)

    # knn加权 越近权重越大 越远权重越小 与dist排序相反
    weightDis = []
    for i in dist:
        weightDis.append(gaussian(i, a=1, b=0, c=0.3))
    print('weightDis', weightDis)
    weightDis = np.array(weightDis)
    # weightDis = []
    # for i in dist:
    #     weightDis.append(gaussian(dist=i, a=1, b=0, c=0.3))

    indexSorted = np.argsort(-weightDis) # 降序排序
    print(indexSorted)

    classCount = {}
    for i in range(k):
        vote = Labels[indexSorted[i]]
        classCount[vote] = classCount.get(vote, 0) + 1

    # maxCount = 0
    # for k, v in classCount.items():
    #     if v > maxCount:
    #         maxCount = v
    #         classes = k
    sortedLabel = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedLabel[0][0]

def main():
    dataSet, Labels = createDataSet()

    dataRange, dataMin, dataMax, normData = normDatafunc(dataSet)

    testSet = [0.0, 0.0]
    test = normTest(dataMin, dataRange, testSet)

    k = 3

    lis = {
        'A': 'big',
        'B': 'smaller'
    }
    print(lis)
    result = knnClassify(normData, Labels, test, k)
    print('result is:'+lis[result])



if __name__ == '__main__':
    main()
    # dataSet, Labels = createDataSet()
    #
    # dataRange, dataMin, dataMax, normData = normData(dataSet)
    #
    # testSet = [0.0, 0.0]
    # test = normTest(dataMin, dataRange, testSet)
    #
    # k = 3
    # print(knnClassify(normData, Labels, test, k))
    # nrow = np.shape(dataSet)[0]
    # cdiff = np.tile(testSet, (nrow, 1)) - dataSet
    # sqdiff = cdiff ** 2
    # sumdiff = np.sum(sqdiff, axis=1)
    # dist = np.sqrt(sumdiff)
    #
    # print(dist)
    # indexSorted = np.argsort(dist)
    # print(indexSorted)
    #
    # classCount = {}
    # for i in range(k):
    #     vote = Labels[indexSorted[i]]
    #     classCount[vote] = classCount.get(vote, 0) + 1
    #
    # maxCount = 0
    # for k, v in classCount.items():
    #     if v > maxCount:
    #         maxCount = v
    #         classes = k
    # return classes
