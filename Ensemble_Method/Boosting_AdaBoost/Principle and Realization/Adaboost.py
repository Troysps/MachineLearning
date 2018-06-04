# !/usr/bin/env python3
# -*- utf-8 -*-
"""
__title__ = ''
__author__ = LEI
__mtime__ = '2018/6/4'

  we are drowning in information,but starving for knowledge
"""
import numpy as np

def loadDataSet(filename):
    dataSet, labelMat = list(), list()
    with open(filename, 'r') as fr:
        for line in fr.readlines():
            dataSet.append([int(line.split(',')[0])])
            labelMat.append([int(line.split(',')[1])])

    return np.mat(dataSet), np.mat(labelMat)

# 三个弱分类器
def weekClassify1(x):
    if x < 2.5:
        return 1
    elif x > 2.5:
        return -1

def weekClassify2(x):
    if x < 8.5:
        return 1
    elif x > 8.5:
        return -1

def weekClassify3(x):
    if x > 5.5:
        return 1
    elif x < 5.5:
        return -1

def calcErrorRate(dataSet, labelMat, func):
    dataSize = np.shape(labelMat)[0]

    errorIndex = list()
    for index in range(len(dataSet)):
        predict = func(dataSet[index])
        if predict * labelMat[index] < 0:
            errorIndex.append(index)
    return errorIndex

def adaBoost(dataSet, labelMat, funcList):
    """

    :param dataSet: 训练数据集 input  特征值
    :param labelMat: 训练数据集 output 类标记
    :param funcList: 弱分类器的列表集合
    :return
        fx 字典形式的基本分类器的线性组合
    """
    # step1 初始化训练数据的权值分布
    dataSize = np.shape(labelMat)[0]
    D = np.ones((dataSize, 1)) / dataSize
    print(D)  # [[0.1], [0.1], [0.1].....[0.1]]
    # step2 对m=1,2,....,M
    # (a) 使用具有权值分布D的训练数据集学习 得到基本分类器 Gm(x)
    # funcList = [weekClassify1, weekClassify2, weekClassify3]
    # print(funcList)

    # (b) 计算Gm(x)在训练数据集上的分类误差率 = 错分类数据权值之和
    min_errorRate = 1

    # fx 基本分类器的线性组合
    fx = dict()
    for funcIndex in range(len(funcList)):
        errorIndex = calcErrorRate(dataSet, labelMat, funcList[funcIndex])

        errorRate = 0

        for index in errorIndex:
            errorRate += float(D[index])
        print('errorRate:', errorRate)
        # print('min_errorRate', min_errorRate)
        # (c) 计算Gm(x)的系数   选择弱分类器中错分率最低的分类器 优先计算系数
        if errorRate < min_errorRate:
            min_errorRate = errorRate
            print('错分率最低的分类器索引', funcIndex)
            alpha = (1/2)*np.log((1-errorRate)/errorRate)
            print('计算Gm(x)的系数', alpha)
            # (d) 更新训练数据的权值分布
            print(errorIndex)
            print('更新权重')
            for indexD in range(len(D)):
                if indexD in errorIndex:
                    # print('D indexD', D[indexD])
                    D[indexD] = D[indexD] / (2*errorRate)
                else:
                    D[indexD] = D[indexD] / (2*(1-errorRate))
            print(D)
            # step 3 构建基本分类器的线性组合
            print('构建基本分类器的线性组合')
            fx[alpha] = funcList[funcIndex]
    return fx

# 最终分类器
def sign(fx, testData):
    result = 0
    for key, value in fx.items():
        result += key*value(testData)

    if result > 0:
        result = 1
    else:
        result = -1

    return result
#
def strongClassify(fx, testData, labelMat):
    errorCount = 0
    for index in range(len(testData)):
        predict = sign(fx, testData[index])
        # print(predict)
        if predict != float(labelMat[index]):
            errorCount += 1

    print('strongClassify errorCount:', errorCount)
    correctRate = (1 - (errorCount/(len(labelMat))))*100

    return correctRate

def main():
    filename = 'test.txt'
    dataSet, labelMat = loadDataSet(filename)
    funcList = [weekClassify1, weekClassify2, weekClassify3]
    fx = adaBoost(dataSet, labelMat, funcList)
    print(fx)
    correctRate = strongClassify(fx, dataSet, labelMat)
    print('AdaBoost StrongClassify CorrectRate:%.2f %%' % correctRate)

if __name__ == '__main__':
    main()


"""目前算法的问题:
    1.并不是真正选择 误分类率最低的弱分类器
    2.fx 应该更新并带入运算中
    因此: 目前来说该算法 需要优化
"""