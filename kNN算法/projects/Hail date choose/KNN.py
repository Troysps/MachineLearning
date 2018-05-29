# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Projects: KNN classify
"""
import numpy as np
import operator

def creatDataSet():
    dataSet = np.array([[1, 1.1], [1, 0.99], [0.1, 0.1], [0.2, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return dataSet, labels

def classify(test, dataSet, labels, k):
    dataSize = dataSet.shape[0]
    diffMat = np.tile(test, (dataSize, 1)) - dataSet
    stdDiff = diffMat ** 2
    stdDis = np.sum(stdDiff, axis=1)
    sortedIndex = np.argsort(stdDis)

    classCount = {}
    for n in range(k):
        voteLabel = labels[sortedIndex[n]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    sortedclassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedclassCount[0][0]

def test():
    test = [1.1, 0.88]
    dataSet, labels = creatDataSet()
    knn_result = classify(test, dataSet, labels, 1)
    return knn_result

if __name__ == '__main__':
    classify_result = test()
    print(classify_result)

