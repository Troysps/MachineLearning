# !/usr/bin/env python3
# -*- utf-8 -*-
"""
__title__ = Adaboost better
__author__ = LEI
__mtime__ = '2018/6/4'

  we are drowning in information,but starving for knowledge
"""

import numpy as np

D = None
fx = dict()


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
    global D
    global fx
    if D is None:
        dataSize = np.shape(labelMat)[0]
        D = np.ones((dataSize, 1)) / dataSize
        print(D)  # [[0.1], [0.1], [0.1].....[0.1]]
    else:

        # step2 对m=1,2,....,M
        # (a) 使用具有权值分布D的训练数据集学习 得到基本分类器 Gm(x)
        # funcList = [weekClassify1, weekClassify2, weekClassify3]
        # print(funcList)

        # (b) 计算Gm(x)在训练数据集上的分类误差率 = 错分类数据权值之和

        # fx 基本分类器的线性组合

        errorRateList = list()
        errorIndexList = list()
        for funcIndex in range(len(funcList)):
            errorIndex = calcErrorRate(dataSet, labelMat, funcList[funcIndex])
            errorIndexList.append(errorIndex)
            errorRate = 0

            for index in errorIndex:
                errorRate += float(D[index])
            print('errorRate:', errorRate)
            errorRateList.append(errorRate)


            # print('min_errorRate', min_errorRate)
            # (c) 计算Gm(x)的系数   选择弱分类器中错分率最低的分类器 优先计算系数
        min_errorRate = min(errorRateList)
        min_funcIndex = errorRateList.index(min_errorRate)
        print('min_errorRate', min_errorRate)
        print('min_funcIndex', min_funcIndex)
        print('错分率最低的分类器索引', min_funcIndex)
        alpha = (1 / 2) * np.log((1 - min_errorRate) / min_errorRate)
        # print('alpha1', alpha)


        # print('alpha2', alpha)
        print('计算Gm(x)的系数', alpha)
        # (d) 更新训练数据的权值分布
        print(errorIndexList[min_funcIndex])
        print('更新权重')
        for indexD in range(len(D)):
            if indexD in errorIndexList[min_funcIndex]:
                # print('D indexD', D[indexD])
                D[indexD] = D[indexD] / (2 * min_errorRate)
            else:
                D[indexD] = D[indexD] / (2 * (1 - min_errorRate))
        print(D)
        # step 3 构建基本分类器的线性组合
        print('构建基本分类器的线性组合')
        fx[alpha] = funcList[min_funcIndex]
        # print('fx', fx)
        sign_errorIndex = strongClassify(fx, dataSet, labelMat)
        sign_errorRate = (1 - (float(len(sign_errorIndex)) / len(labelMat))) * 100
        if sign_errorRate > 90.00:
            print("最终分类器正确率率大于0.9, 正确率为%.2f %%" % sign_errorRate)
            # print('fx:', fx)
            return fx
        else:
            print("当前最终分类器正确率为%.2f %%" % sign_errorRate)
            print('当前最终分类器误分类个数为: %d' % len(sign_errorIndex))
            print('继续优化最终分类器fx:', fx)



    return adaBoost(dataSet, labelMat, funcList)



# 最终分类器
def sign(fx, testData):
    result = 0
    for key, value in fx.items():
        result += key * value(testData)
    if result > 0:
        result = 1
    else:
        result = -1

    return result


# 强分类器验证
def strongClassify(fx, testData, labelMat):

    errorIndex = list()
    for index in range(len(testData)):
        predict = sign(fx, testData[index])
        # print(predict)
        if predict != float(labelMat[index]):

            errorIndex.append(index)

    print('strongClassify errorIndex', errorIndex)
    return errorIndex


def main():
    filename = 'test.txt'
    dataSet, labelMat = loadDataSet(filename)
    funcList = [weekClassify1, weekClassify2, weekClassify3]
    fx = adaBoost(dataSet, labelMat, funcList)
    print(fx)
    # correctRate = strongClassify(fx, dataSet, labelMat)
    # print('AdaBoost StrongClassify CorrectRate:%.2f %%' % correctRate)


if __name__ == '__main__':
    main()

"""目前算法的问题:
    1.并不是真正选择 误分类率最低的弱分类器
    2.fx 应该更新并带入运算中
    因此: 目前来说该算法 需要优化
    
    
    问题解决:
        达到《李航统计学习》一书要求 
"""