#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 垃圾邮件过滤 -- 基于贝叶斯估计
__author__ = Lei
__mtime__ = '2018/5/10'
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
import os
import numpy as np
import re
import codecs

"""
    非垃圾邮件数据集:os.listdir(r'./Naive Bayes Classifier/垃圾邮件过滤/email/ham')
    垃圾邮件数据集:os.listdir(r'./Naive Bayes Classifier/垃圾邮件过滤/email/spam')
"""
# step1 读取数据集 - 切分为列表
def dataToList(data):
    dataList = re.split(r"\W*", data)
    return [word.lower() for word in dataList if len(word) > 2]

def loadDataSet():
    """
    dataSet load
    :return dataSet: 输入空间
    :return classLabel: 输出空间
    """
    dataSet = []
    classLabel = []
    # 非垃圾邮件统计
    try:
        hamList = os.listdir(r'./email/ham')
        for i in hamList:
            # print(i)
            # 处理文本中遇到非法字符 使用codecs 模块 设置 codecs.open(filename, 'r', 'utf-8', 'ignore')
            dataList = dataToList(codecs.open(r'./email/ham/'+i, 'r', 'utf-8', 'ignore').read())
            dataSet.append(dataList)
            classLabel.append(0)

        # 垃圾邮件统计
        spamList = os.listdir(r'./email/spam')
        for i in spamList:
            dataList = dataToList(codecs.open(r'./email/spam/' + i, 'r', 'utf-8', 'ignore').read())
            dataSet.append(dataList)
            classLabel.append(1)

        # print(len(dataSet))  #
        #  print(len(classLabel))
        # print(dataSet[0])
        # print(dataSet[49])
        #
        # print(classLabel)
        return dataSet, classLabel

    except BaseException as bec:
        print('error', bec.args)

# step2 构建分词集合
def createVocableList(dataSet):
    """

    :param dataSet: 输入空间
    :return vocableList: 分词集合
    """
    vocableList = set([])
    for words in dataSet:
        vocableList = vocableList | set(words)

    print(len(vocableList))
    # 去重检验

    """
        三种方式
        
    """
    # 第一种 查看是否为 set
    if isinstance(vocableList, set):
        print("True")
    # 第二种 双重循环检验去重
    vocableList = list(vocableList)
    count1 = []
    for i in range(len(vocableList)):
        for m in vocableList:
            if vocableList.index(m) == i:
                count1.append(True)
            else:
                count1.append(False)
    print('count1', sum(count1))
    # 第三种
    count2 = []
    for i in range(len(vocableList)):
        if vocableList.index(vocableList[i]) == i:
            count2.append(True)

    print('count2', sum(count2))

    return vocableList


# step3 构建词集模型 || 词袋模型
def transferToVec(data, vocableList):
    """

    :param data: 输入空间
    :param vocableList:
    :return:
    """
    modelSize = len(vocableList)
    vec = np.zeros(modelSize)
    for word in data:
        if word in vocableList:
            print("word is in vocableList")
            vec[vocableList.index(word)] += 1
        else:
            print("word is not in vocableList")

    return vec

# step4 随机的抽取部分样本集 转为向量 并计算先验概率 后验概率
def trainNB(trainMat, trainClass):
    """
    计算先验概率和条件概率
    :param trainMat: 训练集输入空间 X
    :param trainClass: 训练集输出空间 Y
    :return p0:先验概率 非垃圾邮件概率
    :return p1:先验概率 垃圾邮件概率
    :return p0Con: 条件概率 P(X=x|Y=ck)  k=0 在事件 非垃圾邮箱 发生条件下 变量X发生的概率 P(F1|Y=0)P(F2|Y=0)P(F3|Y=0)...P(Fn|Y=0)
    :return p1Con: 条件概率 P(X=x|Y=ck)  k=1 在事件 非垃圾邮箱 发生条件下 变量X发生的概率 P(F1|Y=1)P(F2|Y=1)P(F3|Y=1)...P(Fn|Y=1)
    """
    # 计算先验概率
    sampleSize = int(len(trainClass))
    p1Spam = np.sum(trainClass) / sampleSize
    p0Ham = 1 - p1Spam

    # 条件概率计算
    p1Vec = np.ones(len(trainMat[0]))
    p0Vec = np.ones(len(trainMat[0]))

    p0Nums = 2
    p1Nums = 2

    for index in range(sampleSize):
        if trainClass[index] == 0:
            p0Vec += trainMat[index]
            p0Nums += np.sum(trainMat)


        elif trainClass[index] == 1:
            p1Vec += trainMat[index]
            p1Nums += np.sum(trainMat)

    # p1Con p0Con
    p0Con = np.log10(p0Vec / p0Nums)
    p1Con = np.log10(p1Vec / p1Nums)

    print('p0Con---------------', p0Con)
    print('p1Con---------------', p1Con)

    return p0Con, p1Con, p0Ham, p1Spam




# step5 贝叶斯估计分类
def classify(testVec, p0Con, p1Con, p0Ham, p1Spam):
    # print(testVec)
    # print(p0Con)
    # print(p1Con)
    p0 = np.sum(testVec * p0Con) + np.log10(p0Ham)
    p1 = np.sum(testVec * p1Con) + np.log10(p1Spam)

    print('p0-------', p0)
    print('p1-------', p1)
    if p1 > p0:
        return 1
    return 0


# step5 将剩下部分的数据集 进行分类验证

def test(testMat, testClass, p0Con, p1Con, p0Ham, p1Spam):
    testSetCount = len(testMat)
    count = 0
    for index in range(testSetCount):
        predict = classify(testMat[index], p0Con, p1Con, p0Ham, p1Spam)
        rightResult = testClass[index]
        if predict == rightResult:
            count += 1
            print("dataSet {} classify is right".format(testMat))
        else:
            print("dataSet {} classify is error, right result is {}".format(testMat[index], rightResult))

    rightRatio = float(count / testSetCount)
    return rightRatio


def main():
    # content = 'Hi Peter,\n\nWith Jose out of town, do you want to\nmeet once in a while to keep things\ngoing and do some interesting stuff?\n\nLet me know\nEugene'
    # print(dataToList(content))
    dataSet, classLabel = loadDataSet()
    vocableList = createVocableList(dataSet)


    # vecList0 = transferToVec(dataSet[0], vocableList)
    # print(vecList0)

    """
    # 留存交叉验证
    # 随机抽取部分数据集 作为训练集
    # 留存部分数据集 作为测试数据
    """
    # 随机抽取40个样本数据集作为训练集 10个样本数据集作为测试数据集
    print('dataSet Size:', len(dataSet))
    traningIndex = list(range(50))

    testIndex = []
    for i in range(10):
        randomIndex = int(np.random.uniform(0, len(traningIndex)))
        print(randomIndex)
        testIndex.append(traningIndex[randomIndex])
        # testClass.append(classLabel[randomIndex])
        del(traningIndex[randomIndex])


    print('trainIndex------', traningIndex)   # 40个样本训练集
    print('testIndex-------', testIndex)      # 10个样本测试集


    # 使用训练集训练 分类器
    trainMat = []
    trainClass = []
    for index in traningIndex:
        print('traningIndex', index)
        trainMat.append(transferToVec(dataSet[index], vocableList))
        trainClass.append(classLabel[index])


    print('--'*50 + '训练集'+'--'*50)
    print(np.shape(trainMat))
    # print(trainMat)

    print('--'*50 + '训练集类标记' + '--'*50)
    print(np.shape(trainClass))
    print(trainClass)
    print(sum(trainClass))

    # 计算先验概率 条件概率
    p0Con, p1Con, p0Ham, p1Spam = trainNB(trainMat, trainClass)

    # 留存数据作为测试数据集
    testMat = []
    testClass = []
    for index in testIndex:
        print('testIndex', index)
        testMat.append(transferToVec(dataSet[index], vocableList))
        testClass.append(classLabel[index])


    # 对测试数据集测试

    rightRatio = test(testMat, testClass, p0Con, p1Con, p0Ham, p1Spam)
    # print('Right Ratio ', rightRatio)
    print('Rigth Ratio %.2f%%' % (rightRatio * 100))

if __name__ == '__main__':
    main()