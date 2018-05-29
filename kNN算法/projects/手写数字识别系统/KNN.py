# !/usr/bin/env python3
# -*- coding: utf-8 -*-
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
    result = classify(test, dataSet, labels, 2)
    print(result)

def classify0(inX,dataset,labels,k):
    #1,计算距离
    dataset_size = dataset.shape[0]
    diff_mat = np.tile(inX,(dataset_size,1))-dataset
    sqdiff_mat = diff_mat**2
    sq_distances = sqdiff_mat.sum(axis=1)
    distances = sq_distances**0.5 
    #2，按递增排序
    sorted_distances_index = distances.argsort()

    #3，选择距离最近的前k个点,并且计算它们类别的次数排序
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_distances_index[i]]
        class_count[vote_label] = class_count.get(vote_label,0) + 1
        sorted_class_count = sorted(class_count.items(),key=operator.itemgetter(1),reverse=True)

    #4,返回前k个里面统计的最高次类别作为预测类别
    return sorted_class_count[0][0]

    
