#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 社区留言板侮辱性言论屏蔽 -- 基于贝叶斯估计
__author__ = Lei
__mtime__ = '2018/5/8'
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

def loadDataSet():
    """
    训练数据集
    :return: 单词列表postingList, 所属类别classVec
    """
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], #[0,0,1,1,1......]
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec

# step1 分词集合
def CreatewordsSet(dataSet):
    """

    :param dataSet: 训练数据集
    :return: wordsSet 分词集合
    """
    wordsSet = set([])
    for words in dataSet:
        wordsSet = wordsSet | set(words)

    return list(wordsSet)

# step2 转化训练集以向量形式表示
def transferToVec(data, wordsSet):
    """

    :param data: 训练数据[0], 训练数据[1] .... 训练数据[i]
    :param wordsSet: 分词集合
    :return: vec 训练数据的向量形式
    """
    dimensionJ = len(wordsSet)

    vec = np.zeros(dimensionJ)

    for word in data:
        if word in wordsSet:
            print("该词汇存在于分词集合中：%s"% word)
            # += 1 词袋模型
            # = 1 词集模型
            vec[wordsSet.index(word)] += 1
        else:
            print("该词汇不存在于分词集合中：%s"% word)
    return vec

# 先验概率计算及条件概率计算
def tranNBBayes(vecDataSet, classLabel):
    """

    :param vecDataSet: 输入数据集
    :param classLabel: 输出空间类标记
    :return: p1Con 条件概率
    :return: p0Con 条件概率
    :return: probDict 先验概率
    """
    # 先验概率计算
    classCount = dict()
    for vote in classLabel:
        classCount[vote] = classCount.get(vote, 0) + 1
    # print(classCount)

    classNums = len(classLabel)
    probDict = {}
    for key, value in classCount.items():
        probDict[key] = value / classNums

    # print(probDict)

    # 条件概率计算
    # 在类别n条件下, 即侮辱性文档的[P(F1|Cn),P(F2|Cn),P(F3|Cn),P(F4|Cn),P(F5|Cn)....]列表
    classStastic = list(i for i in probDict.keys()) # 就两个类别
    p1Vec = np.ones(len(vecDataSet[0]))
    p0vec = np.ones(len(vecDataSet[0]))

    p1Nums = len(classStastic)
    p0Nums = len(classStastic)

    for i in range(len(classLabel)):
        if classLabel[i] == classStastic[0]:
            p1Vec += vecDataSet[i]
            p1Nums += sum(vecDataSet[i])
        else:
            p0vec += vecDataSet[i]
            p0Nums += sum(vecDataSet[i])

    # 条件概率计算
    p1Con = np.log(p1Vec / p1Nums)
    p0Con = np.log(p0vec / p0Nums)

    print(p1Con)
    print(p0Con)
    print(probDict)
    return p1Con, p0Con, probDict

def classify(testVec, p1Con, p0Con, probDict):
    """

    :param testVec: 数据转向量
    :param p1Con: 条件概率 类别1 即侮辱性文档的[P(F1|Cn),P(F2|Cn),P(F3|Cn),P(F4|Cn),P(F5|Cn)....]列表
    :param p0Con: 条件概率 类别0 即侮辱性文档的[P(F1|Cn),P(F2|Cn),P(F3|Cn),P(F4|Cn),P(F5|Cn)....]列表
    :param probDict: 先验概率
    :return: 分类 0 or 1
    """

    p1 = np.abs(np.sum(np.array(testVec) * p1Con) + np.log(probDict[1]))
    p0 = np.abs(np.sum(np.array(testVec) * p0Con) + np.log(probDict[0]))
    print(p1)
    print(p0)
    if p1 > p0:
        return 1
    return 0



def main():
    dataSet, classLabel = loadDataSet()
    wordsSet = CreatewordsSet(dataSet)
    print(wordsSet, '\n', len(wordsSet))
    vecDataSet = []
    for data in dataSet:
        vecDataSet.append(transferToVec(data, wordsSet))
    # print('vec', vecDataSet)
    p1Con, p0Con, probDict = tranNBBayes(vecDataSet, classLabel)
    test1 = ['love', 'my', 'dalmation']  # 0
    test2 = ['stupid', 'garbage']  # 1
    testVec1 = transferToVec(test1, wordsSet)
    result1 = classify(testVec1, p1Con, p0Con, probDict)

    testVec2 = transferToVec(test2, wordsSet)
    result2 = classify(testVec2, p1Con, p0Con, probDict)


    print('{} classify : {}'.format(test1, result1))
    print('{} classify : {}'.format(test2, result2))


if __name__ == '__main__':
    main()