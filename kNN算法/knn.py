import numpy as np
from sklearn import neighbors
knn = neighbors.KNeighborsClassifier() #取得 knn 分类器
data = np.array([[3,104],[2,100],[1,81],[101,10],[99,5],[98,2]]) # <span style="font-family:Arial, Helvetica, sans-serif;">data 对应着
#打斗次数和接吻次数</span>
labels = np.array([2,2,2,3,3,3]) #<span style="font-family:Arial, Helvetica, sans-serif;">labels 则是对应 Romance 和 Action</span>
knn.fit(data,labels) #导入数据进行训练'''
#Out：KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
#metric_params=None, n_jobs=1, n_neighbors=5, p=2,weights='uniform')
knn.predict([[18,90]])

# !/usr/bin/env python3
# -*- coding: utf-8
import numpy as np
import operator

# 给出训练集以及对应类别
def creatDataSet():
    group = np.array([[1.0, 2.0], [1.2, 0.1], [0.1, 1.4], [0.3, 3.5]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# 设置kNN算法进行分类
def classify(test, dataSet, label, k):
    dataSize = dataSet.shape[0]
    #shape是numpy函数库中的方法，用于查看矩阵或者数组的维度
    #shape(array)若矩阵有m行n列，则返回(m,n)
    #array.shape[0]返回矩阵的行数m，array.shape[1]返回矩阵的列数n
    # 计算欧式距离
    diff = np.tile(test, (dataSize, 1)) - dataSet
    sqDiff = diff ** 2
    sqDiffSum = np.sum(sqDiff, axis=1) # 行向量分别相加 从而得到一个新的行向量
    dist = sqDiffSum ** 0.5
    
    # 对距离进行排序 从小到大的规则
    sortedDistIndex = np.argsort(dist)
    
    classCount = {}
    for i in range(k):
        voteLabel = label[sortedDistIndex[i]]
        # 对选取的K个样本所属的类别个数进行统计
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
        
    # 选取出现类别次数最多的类别
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            classes = key
    return classes
    
def test():
    dataSet, labels = creatDataSet()
    test = np.array([1.1, 0.3])
    k = 3
    output = classify(test, dataSet, labels, k)
    print('Test data:', test, 'classify result:', output)
test()
