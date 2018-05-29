# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projects: 优化约会网站相亲配对结果
@ author: Troy
@ email: ots239ltfok@gmail.com
"""

import numpy as np
import pandas as pd
import operator
import KNN
import matplotlib.pyplot as plt


def get_data(filename):
    data = pd.read_csv(filename, sep='\t', header=None)
    returnMat = data.iloc[:,:-1]
    classLabelVector = data.iloc[:,-1:]
    return returnMat, classLabelVector
filename = 'datingTestSet2.txt'
datingDataMat, datingLabels = get_data(filename)

def autoNorm(dataSet):
    """
    Desc:
        归一化特征值，消除特征之间量级不同导致的影响
    parameter:
        dataSet: 数据集
    return:
        归一化后的数据集 normDataSet. ranges和minVals即最小值与范围，并没有用到

    归一化公式：
        Y = (X-Xmin)/(Xmax-Xmin)
        其中的 min 和 max 分别是数据集中的最小特征值和最大特征值。该函数可以自动将数字特征值转化为0到1的区间。
    """
    # 取出数据集中 最大值 最小值 极差范围
    minVal = dataSet.min(0)
    maxVal = dataSet.max(0)
    ranges = maxVal - minVal
     
    #dataSize = dataSet.shape[0]
    
    #normDataSet = dataSet - np.tile(minVal, (dataSize, 1))
    #normDataSet = normDataSet / np.tile(ranges, (dataSize, 1))

    normDataSet = dataSet.apply(lambda x: (x-minVal)/ranges, axis=1)
    return normDataSet, ranges, minVal



def datingClassTest():
    """
    Desc:
        对约会网站的测试方法
    parameters:
        none
    return:
        错误数
    """
    # 设置测试数据的比列
    hoRatio = 0.1 # 测试范围 一部分测试 一部分作为样本
    # 从文件中加载数据
    datingDataMat, datingLabels = get_data(filename)
    # 归一化数据
    normDataSet, ranges, minVals = autoNorm(datingDataMat)
    normDataSet = np.array(normDataSet)
    print('$'*100, normDataSet, len(normDataSet))
    datingLabels = datingLabels.iloc[:,0].tolist()
    # 表示数据的行数
    dataSize = normDataSet.shape[0]
    # 设置测试样本的数据
    numTestVecs = int(dataSize * hoRatio)
    print(numTestVecs)
    print('NumTestVecs:', numTestVecs)
    print(normDataSet[numTestVecs:])
    errorCount = 0
    for n in range(numTestVecs):
        # 对数据进行测试
        classifierResult = KNN.classify(normDataSet[n], normDataSet[numTestVecs: ], datingLabels[numTestVecs : dataSize], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[n]))
        if (classifierResult != datingLabels[n]):
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print(errorCount)

# 使用算法构建完整可用系统
def classifyPerson():
    percentTats = float(input("percentage of time spent playing video games? "))
    ffMiles = float(input("frequent flier miles earned per year? "))
    iceCream = float(input("liters of ice cream consumed per year? "))
    data = [ffMiles, percentTats, iceCream]
    filename = 'datingTestSet2.txt'
    dataSet, labels = get_data(filename)
    normDataSet, ranges, minVals = autoNorm(dataSet)
    normDataSet = np.array(normDataSet)
    labels = labels.iloc[:, 0].tolist()
    classifierResult = KNN.classify(data, normDataSet, labels, 3)
    resultList = ['not at all', 'in small doses', 'in large doses']
    print("You will probably like this person: ", resultList[classifierResult - 1])
    
if __name__ == '__main__':
    datingClassTest()
    classifyPerson()