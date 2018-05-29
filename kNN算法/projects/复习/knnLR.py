#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = knn logistic regression
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
"""KNN LR回归预测
"""

import numpy as np
import operator

def creatDataSet():
    groups = np.array([[1, 1.1], [1, 0.99], [0.1, 0.02], [0.2, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return groups, labels

def classify(test, dataSet, labels, k):
    dataSize = dataSet.shape[0]
    diff = np.tile(test, (dataSize, 1)) - dataSet
    stdDiff = diff ** 2
    stdDis = np.sum(stdDiff, axis=1)
    sortedIndex = np.argsort(stdDis)

    classCount = {}
    for n in range(k):
        voteLabel = labels[sortedIndex[n]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def test():
    dataSet, labels = creatDataSet()
    test = [0.9, 1]
    result = knnLR(test, dataSet, labels, 2)
    print(result)

def weight(dist, a=1, b=0, c=0.3):
    return a * np.math.e ** (-(dist - b) ** 2 / (2 * c ** 2))

# 回归预测 -- 使用权重
def knnLR(inX,dataset,labels,k):
    #1,计算距离
    dataset_size = dataset.shape[0]
    diff_mat = np.tile(inX,(dataset_size,1))-dataset
    sqdiff_mat = diff_mat**2
    sq_distances = sqdiff_mat.sum(axis=1)
    distances = sq_distances**0.5
    #2，按递增排序
    print(distances, 'distances')
    lis = []
    for i in distances:
        lis.append(weight(i, a=1, b=0, c=0.3))
    print(lis, 'lis')

    sorted_distances_index = distances.argsort()


    #3，选择距离最近的前k个点, 取其均值
    knnpredict = int()
    for i in range(k):
       knnpredict += dataset[sorted_distances_index[i]] * lis[sorted_distances_index[i]]

    knnpredict = knnpredict/k

    return knnpredict


    #4,返回前k个里面统计的最高次类别作为预测数值
    # return sorted_class_count[0][0]


if __name__ == '__main__':
    test()