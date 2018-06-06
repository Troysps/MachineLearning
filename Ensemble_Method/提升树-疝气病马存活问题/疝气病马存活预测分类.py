# !/usr/bin/env python3
# -*- utf-8 -*-
"""
__title__ = 疝气病马存活预测分类
__author__ = LEI
__mtime__ = '2018/6/5'

  we are drowning in information,but starving for knowledge
"""


import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(filename):
    dataSet = list()
    labelMat = list()
    with open(filename, 'r') as f:
        for lines in f.readlines():
            lis = lines.split('\t')
            lineArr = []
            for i in lis[:-1]:
                lineArr.append(float(i))
            # print(lineArr)
            # print(len(lineArr))
            dataSet.append(lineArr)
            labelMat.append(float(lis[-1]))
    print(np.shape(dataSet), type(dataSet[0][0]))  # (299, 21) <class 'float'>
    print(np.shape(labelMat), type(dataSet[0][0]))  # (299, ) <class 'float'>
    return dataSet, labelMat


# 建立单层决策树
def stumpTree(dataSet, labelMat, D):
    """
    建立单层决策树
    :param dataSet: 训练数据集   特征值
    :param labelMat: 训练数据集  类别标签
    :param D: 权重值
    :return
        bestStumpTree  最佳单层决策树
    """
    dataMat = np.mat(dataSet)
    labelMat = np.mat(labelMat).T
    m, n = np.shape(dataMat)
    # print('m, n', m, n)
    # print('dataMat:', dataMat)
    # print('labelMat:', labelMat)
    # step1 设置最小误差
    minError = np.inf  # 最小误差率设置为正无穷

    numSteps = 10.0
    bestStump = dict()
    bestClasEst = np.mat(np.zeros((m, 1)))
    # step2 对每一个特征进行循环 计算步长

    # print('训练数据集特征', featureNums)
    for index in range(n):
        # 连续性数据 分类  需要计算步长
        rangeMin = dataMat[:, index].min()
        rangeMax = dataMat[:, index].max()
        delta = (rangeMax - rangeMin) / numSteps
        # step3 第二层循环
        for j in range(-1, int(numSteps)+1):
            # step4 第三层循环 建立一颗单层决策树并利用加权数据对它进行测试(错误率)
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + j * delta)
                predictedVals = stumpClassify(dataMat, index, threshVal, inequal)
                # print(predictedVals)  结果集

                # 计算加权错误率 weightedError
                errorMat = np.mat(np.ones((m, 1)))
                errorMat[predictedVals == labelMat] = 0
                weightedError = D.T * errorMat
                # print('index', index)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = index
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


def stumpClassify(dataMat, index, threshVal, threshIneq):
    """
    单层决策树分类
    在阈值一边的数据会分到类别-1
    另一边的阈值分到类别1
    首先将全部数组元素设置为1 然后将不满足不等式要求的元素设置为-1
    :param dataMat: 训练数据集 输入空间
    :param index:   训练数据集 特征index
    :param threshVal: 阈值
    :param threshIneq: 小于或大于
    :return
        retArray: 结果集
    """
    # print('try', (np.shape(dataMat))[0])
    dimen = (np.shape(dataMat))[0]
    retArray = np.ones((((np.shape(dataMat))[0]), 1))  # 5x1
    if threshIneq == 'lt':
        retArray[dataMat[:, index] <= threshVal] = -1.0
    else:
        retArray[dataMat[:, index] > threshVal] = -1.0
    return retArray


"""基于单层决策树的AdaBoost算法的实现
伪代码
    对每次迭代:
        利用stumpTree()函数找到最佳的单层决策树
        将最佳的单层决策树加入到决策树组
        计算alpha
        计算新的权重向量D
        更新累计类别估计值
        如果错误率==0, 退出循环
"""
def adaBoostTrainDT(dataMat, labelMat, maxCycle):
    """
    基于单层决策树的AdaBoost算法实现
    :param dataMat: 训练数据集   输入空间
    :param labelMat: 训练数据集  输出空间
    :param maxCycle: 最大迭代次数
    :return
        strongDtree 决策数组
    """
    strongDtree = list()
    m, n = np.shape(dataMat)
    # 初始化权重向量D
    D = np.ones((m, 1)) / m
    # print('初始化权重向量D', D)
    upClassEst = np.mat(np.zeros((m, 1)))
    for cycle in range(maxCycle):
        bestStump, minError, bestClasEst = stumpTree(dataMat, labelMat, D)
        # 计算 alpha 系数
        alpha = float(0.5 * np.log((1-minError)/max(minError, 0.001)))
        bestStump['alpha'] = alpha
        strongDtree.append(bestStump)

        # 更新D 权重向量
        # 错误分类的加重权重 正确分类的减少权重
        expon = np.multiply(-1*alpha*np.mat(labelMat).T, bestClasEst)
        D = np.multiply(D, np.exp(expon))
        D = D/D.sum()

        # 更新累计类别估计值
        upClassEst += alpha * bestClasEst
        aggError = np.multiply(np.sign(upClassEst)!=np.mat(labelMat).T, np.ones((m, 1)))
        errorRate = aggError.sum() / m
        print('errorRate:', errorRate)
        if errorRate == 0.0:
            break

    return strongDtree, upClassEst

# 测试算法: 基于AdaBoost的分类
def adaClassify(dataToClass, classifierArr):
    """
    基于AdaBoost的强分类器的分类
    :param dataToClass: 输入变量
    :param classifierArr: 强分类器
    :return
        np.sign(aggClassEst) 分类结果
    """
    # do stuff similar to last aggClassEst in adaBoostTrainDS
    dataMat = np.mat(dataToClass)
    m = np.shape(dataMat)[0]

    aggClassEst = np.mat(np.zeros((m, 1)))


    # 循环 多个分类器
    for i in range(len(classifierArr)):
        # 前提： 我们已经知道了最佳的分类器的实例
        # 通过分类器来核算每一次的分类结果，然后通过alpha*每一次的结果 得到最后的权重加和的值。
        classEst = stumpClassify(dataMat, classifierArr[i]['dim'], classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        # print(aggClassEst)
    return np.sign(aggClassEst)


def plotROC(predStrengths, classlabels):
    """
    ROC曲线
    :param predStrengths: 分类器的预测强度
    :param classlabels: 实际分类
    :return
        ROC
    """

    # (1,1) 以及原点 0.0
    cur = (1.0, 1.0)
    ySum = 0.0
    # 步长
    positiveNum = np.sum(np.array(classlabels)==1.0)
    yStep = 1/float(positiveNum)
    xStep = 1/float(len(classlabels) - positiveNum)
    # 预测强度排序索引
    sortedIndeicies = predStrengths.argsort()

    fig = plt.figure()
    fig.clf()

    ax = plt.subplot(111)
    for index in sortedIndeicies.tolist()[0]:
        if classlabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1]-delY], c='b')
        cur = (cur[0]-delX, cur[1]-delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('FP Rate'), plt.ylabel('TP Rate')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print('AUC:', ySum * xStep)

def main():
    filename = 'horseColicTraining2.txt'
    dataSet, labelMat = loadDataSet(filename)

    strongDtree, aggClassEst = adaBoostTrainDT(dataSet, labelMat, maxCycle=10)
    print('strongDtree', strongDtree)
    print('aggClassEst', aggClassEst)
    plotROC(aggClassEst.T, labelMat)
    filename = 'horseColicTest2.txt'
    testSet, testlabel = loadDataSet(filename)
    predictList = adaClassify(testSet, strongDtree)

    # 测试：计算总样本数，错误样本数，错误率
    m = np.shape(testSet)[0]
    error = np.mat(np.ones((m, 1)))

    print('测试样本个数%s 错误个数%s 错误率%.2f%%' % (m, error[predictList != np.mat(testlabel).T].sum(), error[predictList != np.mat(testlabel).T].sum()/m *100))


if __name__ == '__main__':
    main()