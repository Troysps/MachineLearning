#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'Administrator'
__mtime__ = '2018/5/3'
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

def createDataSet():
    dataSet = [[1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def calcShonnonEntropy(dataSet):
    m = len(dataSet)
    classCount = dict()
    for i in dataSet:
        vote = i[-1]
        classCount[vote] = classCount.get(vote, 0) + 1

    print(classCount)
    entropy = 0
    for i in classCount:
        prob = classCount[i] / m
        entropy -= prob * np.log2(prob)
    print('entropy', entropy)
    return entropy

def splitDataSet(dataSet, index, value):
    splitdataSet = []
    for i in dataSet:
        if i[index] == value:
            reduceFeatVec = i[:index]
            reduceFeatVec.extend(i[index+1:])
            splitdataSet.append(reduceFeatVec)
    print('splitdataSet', splitdataSet)
    return splitdataSet


if __name__ == '__main__':
    dataSet, labels = createDataSet()
    print(dataSet)
    print(labels)
    # 计算信息熵
    calcShonnonEntropy(dataSet)
    # 拆分数据集
    set1 = splitDataSet(dataSet, 0, 1)
    # 计算信息增益
    calcShonnonEntropy(set1)