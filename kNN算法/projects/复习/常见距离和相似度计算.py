#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ML 常见的距离和相似度计算
__author__ = LEI
__mtime__ = '2018/4/24'
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

"""常见的距离和相似度计算
    欧式距离
    
    闵可夫斯基距离
    
    马氏距离
    
    
    互信息
    
    余弦相似度
    
    皮尔逊相关系数
    
    Jaccard相关系数    
"""
import numpy as np
import pandas as pd
def createDataSet():
    dataSet = np.matrix([[0, 0.5, 0.7],[0.1, 0.4, 0.3], [8, 9, 10], [10, 6, 7]])
    label = ['A', 'A', 'B', 'B']

    return dataSet, label



# 欧式距离
def euclidean(dataSet, testSet):
    nrow = np.shape(dataSet)[0]
    test = np.matrix(np.tile(testSet, (nrow, 1)))
    print(testSet)
    # 欧式距离公式

    m = np.array(test - dataSet) ** 2
    diff = np.sqrt(np.sum(m, axis=1))
    return diff

# 闵可夫斯基距离 两个向量(点)的p阶段距离 p=1 or 2
"""闵可夫斯基距离
    计算公式:
        minkowski = (|x -y|^p)^(1/p)
    当p=1是曼哈顿距离 当p=2时就是欧式距离
"""
def minkowskiDistance(dataSet, testSet, p):
    # n组数据
    nrow = np.shape(dataSet)[0]
    # 将test数据转为n组的test数据
    test = np.tile(testSet, (nrow, 1))
    # 计算差值绝对值
    diffAbs = np.abs(test - dataSet)
    # 计算绝对值的 p次
    sqdiff = np.power(diffAbs, p)
    # 对p次diff 计算其1/p次
    diff = np.power(sqdiff, 1/p)
    # 对距离排序
    distance = np.sum(diff, axis=1)
    print(distance)
    # count = 0
    # for i in range(len(distance)):
    #     if distance[i, 0] > count:
    #         count = distance[i, 0]
    #         max = count
    """matrix get value 
        :np.matrix[m, n]
        :m row n col
    """
    return distance


# 马氏距离
"""马氏距离：首要条件在同一个分布中
    定义在两个向量上 这两个点在同一个分布里 点x和点y的马氏距离为
    np.sqrt((x-y)^T  cov^-1 (x-y))
    cov 是这个分布的协方差矩阵
    当cov=I时 马氏距离退化为欧式距离
    马氏距离是计算两个未知样本集的相似度的方法
    与欧式距离的区别:
        考虑各种特性之间的联系 并且与尺度无关
    优点:
        不受量纲的影响 两点之间的马氏距离与原始数据的测量单位无关
        由标准化数据和中心化数据(即原数据与均值之差)计算出的两点之间的马氏距离相同
        还可怕排除变量之间的相关性的干扰
        
    缺点:
        夸大了变化微小的变量的作用

"""

def mahalanobisDist(dataSet, testSet, labelSet):
    # print('dataSet', dataSet)
    # print('testSet', testSet)
    #
    # # setLabel = list(set(labelSet))
    # # labelSet = np.array(labelSet)
    # #
    # # classLabel = {}
    # # for i in range(len(setLabel)):
    # #     index = np.where(labelSet==setLabel[i])
    # #     for m in index:
    # #         print('index', m[i][m])
    # data = pd.DataFrame(dataSet, labelSet, columns=['data1', 'data2', 'data3'])
    # data1 = np.matrix(np.array(data.ix['A']))
    # data2 = np.matrix(np.array(data.ix['B']))
    #
    # print('data1', data1)
    # print('data2', data2)
    # X = np.vstack([data2, testSet])
    # print('X', X)
    # V = np.cov(X.T)
    # print('X.T', V)
    # VI = np.linalg.inv(V)
    # print('VI', VI)
    #
    # lis = []
    # for i in range(len(X)):
    #     print('dataSet i', X[i])
    #     delta = testSet - X[i]
    #     print('delta:', delta)
    #     # 构造对角矩阵
    #     #print(np.sqrt(np.dot(delta), VI))
    #     #D = np.sqrt(np.einsum('nj,jk,nk->n', delta, VI, delta))
    #     print(np.diag(np.sqrt(np.dot(np.dot(delta, VI), delta.T))))
    # print(lis)

    # 同一分布中计算两个点的相似度
    distances = []
    dataSize = np.shape(dataSet)[0]
    for i in range(dataSize):
        x = np.array(dataSet) # 链接
        #print(x)
        xt = x.T # 转置
        cov = np.cov(xt) # 协方差
        #print(cov)
        s = np.linalg.inv(cov)
        tp = testSet - dataSet[i]
        distances.append(np.sqrt(np.dot(np.dot(tp, s), tp.T)))
    print(distances)



if __name__ == '__main__':

    dataSet, labelSet = createDataSet()
    print(dataSet)
    testSet = [10, 10, 10]
    # 欧式距离
    # euclidean(dataSet, testSet)
    # 闵可夫斯基距离
    # minkowskiDistance(dataSet, testSet, p=1)
    mahalanobisDist(dataSet, testSet, labelSet)