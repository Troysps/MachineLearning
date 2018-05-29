#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 个人广告中获取区域倾向
__author__ = 'LEI'
__mtime__ = '2018/5/14'
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

"""
    出现错误 rss源数据无法获取
    假装有数据
"""
import feedparser
import operator
import re
import numpy as np

def textParse(data):
    """
        正则表达式--将字符串拆分为list 小写
    :param data: example data = 'asfsad S Dfads'
    :return dataList: example dataList = ['asfsad', 'Dfads']
    """
    dataList = re.split(r"\w*", data)
    return [word.lower() for word in dataList if len(word) > 2]

# step1 数据加载
def localWords(feed1,feed0):
    """

    :param feed1: 来自newyork的数据源1
    :param feed0: 来自sfbay的数据源0
    :return docList  : 输入空间X 输入变量x集合 nxj
            classList: 输出空间Y 类标签y集合
            fullText:  记录所有输入变量x的列表
    """
    import feedparser
    docList=[]; classList = []; fullText =[]
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    print('load data')
    print(minLen)
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    return docList, classList, fullText

# step2 创建分词集合
def createVocableList(docList):
    vocableList = set([])
    for data in docList:
        vocableList = vocableList | set(data)

    return list(vocableList)


# step3 构建词集模型 | 词袋模型
def transferToVec(data, vocableList):
    sampleSize = len(vocableList)
    vec = np.zeros(sampleSize)
    for word in data:
        vec[vocableList.index(word)] += 1

    return vec

# step4 贝叶斯估计 计算先验概率 条件概率
def transNB(trainMat, trainClass):
    """

    :param trainMat:
    :param trainClass:
    :return p0:先验概率 SF class:0
    :return p1:先验概率 NY class:1
    :return p0Con: 条件概率 P(X=x|Y=ck)  k=0 在事件 非垃圾邮箱 发生条件下 变量X发生的概率 P(F1|Y=0)P(F2|Y=0)P(F3|Y=0)...P(Fn|Y=0)
    :return p1Con: 条件概率 P(X=x|Y=ck)  k=1 在事件 非垃圾邮箱 发生条件下 变量X发生的概率 P(F1|Y=1)P(F2|Y=1)P(F3|Y=1)...P(Fn|Y=1)
    """
    sampleSize = len(trainClass)
    p1NY = np.sum(trainClass) / sampleSize
    p0SF = 1 - p1

    # 条件概率计算
    p1Vec = np.ones(len(trainMat[0]))
    p0Vec = np.ones(len(trainMat[0]))

    p0Nums = 2.0
    p1Nums = 2.0

    for index in range(sampleSize):
        if trainClass[index] == 1:
            p1Vec += trainMat[index]
            p1Nums += np.sum(trainMat[index])
        elif trainClass[index] == 0:
            p0Vec += trainMat[index]
            p0Nums += np.sum(trainMat[index])

    p1Con = np.log10(p1Vec / p1Nums)
    p0Con = np.log10(p0Vec / p0Nums)

    return p1NY, p0SF, p1Con, p0Con

def classify(testVec, p1NY, p0SF, p1Con, p0Con):
    p1 = np.sum(testVec * p1Con) + np.log10(p1NY)
    p0 = np.sum(testVec * p0Con) + np.log10(p0SF)
    if p1 > p0:
        return 1
    return 0

def main():
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')

    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    print('newyork\n', ny)
    print('sfbay\n', sf)
    # localWords(ny,sf)
    docList, classList, fullText = localWords(ny, sf)
    vocableList = createVocableList(docList)
    trainSet = []
    for data in docList:
        trainSet.append(transferToVec(data, vocableList))
    p1NY, p0SF, p1Con, p0Con = transNB(trainSet, classList)




if __name__ == "__main__":
    main()
