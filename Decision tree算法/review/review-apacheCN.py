#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = Decision tree (DT algorithm)
__author__ = 'Administrator'
__mtime__ = '2018/4/27'
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

"""决策树原理:信息论
    --树形模型  特征对实例的分类
        --节点
            --内部节点  判断
            --叶节点    类别
        --有向边
    ID3算法:
        --信息熵和信息增益
    决策树算法优点:
        --易于理解
        --人类思维契合
        --模型树的形式可视化
        --可以处理非数值型数据
        --速度快
        --准确性高
        --可以处理连续和种类字段
        --适合高维数据
        
    缺点:
        --处理不好连续变量
        --不好处理变量之间存在复杂关系
        --决定分类的因素取决于更多变量的复杂组合
        --可规模性一般
        --易于过拟合
        --对于各类别样本数量不一致的数据 信息增益偏向于具有更多数值特征的
        --忽略属性之间的相关性
        
    决策树实现的三个步骤：
        --特征选择
        --决策树生产
        --决策树的修剪
        
         
"""

"""基本实现原理
    --1.计算样本信息熵(香农熵)
    --2.根据特征属性拆分数据
    --3.计算信息增益 = 信息熵 - 条件熵
    
    :return 选出信息增益最大的特征
"""
import numpy as np
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def calcShonnonEntropy(dataSet):
    classCount = {}
    for i in dataSet:
        cls = i[-1]
        classCount[cls] = classCount.get(cls, 0) + 1
    print(classCount)

    m, n = np.shape(dataSet)
    print(m, n)
    entropy = 0
    for key, value in classCount.items():
        prob = value / m
        entropy -= prob * np.log2(prob)
    print(entropy)
    return entropy

def splitData(dataSet, index, value):
    splitData = []
    for i in dataSet:
        if i[index] == value:
            reduceFeatVec = i[:index]
            reduceFeatVec.extend(i[index+1:])
            splitData.append(reduceFeatVec)

    return splitData



def chooseBestFeatureTosplit_Gain(dataSet):
    """ID3算法
        信息增益计算
        ID3算法不足:
        1.使用ID3算法构建决策树 若出现各属性值取值数分布偏差大的情况下 分类精度会大打折扣
        2.ID3算法本身并未给出处理连续数据的方法
        3.ID3算法不能处理带有缺失值的数据集 需要在算法挖掘之前对数据集中的缺失值进行预处理
        4.ID3算法只有树的生成 该算法生成的树容易过拟合

    :param dataSet:
    :return: gain
    """
    entropy = calcShonnonEntropy(dataSet)
    baseinfoGain = 0.0
    bestFeat = -1
    featVecNum = len(dataSet[0]) - 1
    for i in range(featVecNum):
        attrs = [example[i] for example in dataSet]
        uniqueAttrs = set(attrs)
        newEntropy = 0
        for value in uniqueAttrs:
            splitdataSet = splitData(dataSet, i, value)
            prob = len(splitdataSet) / len(dataSet)
            newEntropy += prob * calcShonnonEntropy(splitdataSet)
        infoGain = entropy - newEntropy
        if infoGain > baseinfoGain:
            baseinfoGain = infoGain
            bestFeat = i
    print('baseinfoGain', baseinfoGain)
    print('bestFeat', bestFeat)
    return bestFeat
"""ID3算法 C4.5算法
    ID3算法:  基于信息增益
    C4.5算法: 基于信息增益率
    
    ID3算法缺点:
        对于样本数据集中可数数目较多属性有所偏好
        例子:考虑一个特殊情况 若分支数目与样本数目相等 此时信息增益等于样本数据集的信息熵
            信息增益如此大 但是对于样本并没有泛化能力
            
    C4.5算法 --对于ID3算法的优化
        使用信息增益的比例 而非信息增益率进行比较
        
"""

def chooseBestFeatureTosplit_GainRatio(dataSet):
    """C4.5算法
        基于信息增益率计算
        1.属性选择度量 C4.5使用信息增益比选择最佳特征
        2.信息增益比率度量
        3.把连续分布特征的处理

    :param dataSet:
    :return:
    """
    baseinfoGainRatio = 0
    bestFeat = -1
    entropy = calcShonnonEntropy(dataSet)

    featNum = len(dataSet[0]) - 1

    for index in range(featNum):
        attrs = [i[index] for i in dataSet]
        uniqueAttrs = set(attrs)
        newentropy = 0
        iv = 0
        for value in uniqueAttrs:
            splitDataSet = splitData(dataSet, index, value)
            prob = len(splitDataSet) / len(dataSet)
            iv -= prob * np.log2(prob)
            newentropy += prob * calcShonnonEntropy(splitDataSet)
        infoGain = entropy - newentropy
        GainRatio = infoGain / iv
        if infoGain > baseinfoGainRatio:
            baseinfoGain = infoGain
            bestFeat = index

        return bestFeat

"""CART (classification and regression tree) 分类与回归树算法
     给定输入随机变量X条件下输出随机变量Y的条件概率分布的学习方法
     假设决策树是二叉树 内部节点特征的取值为'是'和'否'
     这样的决策树等同于递归地二分每个特征 将输入控件即特征控件划分为有限个单元
     
     Gini指数计算:
     Gini(p) = 1 - sigma (Pk**2)
     在分类问题中假设有K个类 样本点属于K类的概率为Pk 
     
     对于给定的样本集合 其基尼指数为Gini(D) = 1 - sigma{|Ck|/|D|)**2
     其中Ck是D中属于第K类的样本子集 K是类的个数
     
     样本集合D根据特征
"""
def calcGini(dataSet):
    gini = 1
    labelCount = {}
    labelList = [i[-1] for i in dataSet]
    for i in labelList:
        vote = i
        labelCount[vote] = labelCount.get(vote, 0) + 1

    print(labelCount)

    for key in labelCount:
        prob = float(labelCount[key]) / float(len(dataSet))
        gini -= (prob ** 2)

    print('gini', gini)
    return gini


def chooseBestFeatureTosplit_Gini(dataSet):
    baseGiniIndex = 1000000
    bestFeatindex = -1

    featNum = len(dataSet[0]) - 1

    for index in range(featNum):
        attrs = [example[index] for example in dataSet]
        uniqueAttrs = set(attrs)
        newGiniIndex = 0
        for value in uniqueAttrs:
            splitdataSet = splitData(dataSet, index, value)
            prob = len(splitdataSet) / len(dataSet)
            newGiniIndex += prob * calcGini(splitdataSet)
        if newGiniIndex < baseGiniIndex:
            baseGiniIndex = newGiniIndex
            bestFeatindex = index
    print('bestFeatindex', bestFeatindex)
    return bestFeatindex


def majorityCnt(classList):
    """majorityCnt 选择出现次数最多的一个结果

    :param classList: -- 包含数据集最后一列的列表
    :return: 最优特征列
    """
    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labelSet):
    """
    创建树

    停止条件:
        1. 所有类标签都相同
        2. 使用完了所有特征 但是仍然不存在唯一元素

    :param dataSet:
    :param labelSet:
    :return: tree dict
    """
    classList = [example[-1] for example in dataSet]
    print(classList)
    # 停止条件1：所有的类标签完全相同 则直接返回该类标签
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 停止条件2：使用完了所有特征 仍然不能将数据集划分为仅仅包含唯一类别的分组
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    # 两个停止条件
    # 选择最优的特征的index
    bestFeat = chooseBestFeatureTosplit_Gain(dataSet)
    # 获取最优的特征
    bestFeatLabel = labelSet[bestFeat]
    # 初始化myTree
    myTree = {bestFeatLabel: {}}
    del (labelSet[bestFeat])

    # 取出最优列 然后用它的branch做分类
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 求出剩余的标签label
        subLabels = labelSet[:]
        # 遍历当前选择特征包含的所有属性值 在每个数据集划分上递归调用函数createTree()
        myTree[bestFeatLabel][value] = createTree(splitData(dataSet, bestFeat, value), subLabels)
    return myTree

def classify(inputTree, featLabel, testVec):
    """

    :param inputTree:   决策树模型
    :param featLabel:   输入特征
    :param testVec:     输入数据
    :return:
    """
    # 获取根节点
    firstStr = list(inputTree.keys())[0]
    print(firstStr)

    # 获取该节点对应的分类或节点
    secondDict = inputTree[firstStr]
    print(secondDict)

    featIndex = featLabel.index(firstStr)

    # 测试数据
    key = testVec[featIndex]
    value = secondDict[key]

    if isinstance(value, dict):
        classLabel = classify(value, featLabel, testVec)
    else:
        classLabel = value
    print(classLabel)
    return classLabel

"""决策树的存储与提取
    存储：store
    提取: grabTree
"""
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()

def grapTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


def main():
    dataSet, labelSet = createDataSet()
    # calcShonnonEntropy(dataSet)
    # set1 = splitData(dataSet, 0, 1)
    # print(set1)
    # calcShonnonEntropy(set1)
    # print(chooseBestFeatureTosplit_Gain(dataSet))
    # # 计算信息增益率 遵循条件 不断的迭代
    # tree = createTree(dataSet, labelSet)
    # print(tree)
    # classify(tree, ['no surfacing', 'flippers'], [1, 1])
    calcGini(dataSet)
    chooseBestFeatureTosplit_Gini(dataSet)


if __name__ == '__main__':
    main()