# !/usr/bin/env python3
# -*- utf-8 -*-
"""
__title__ = ROC receiver operating characteristic 接受者操作特征
__author__ = LEI
__mtime__ = '2018/6/5'

  we are drowning in information,but starving for knowledge
"""

"""ROC
    什么是ROC是对分类器性能评分的一种指标
    ROC曲线中给出了两条线 一条虚线 一条实线
    横轴为: 阳率--伪正例 FP/(FP+TN) 
    纵轴为: 真阳率 --真正例 TP / (TP + FN) 
    原点(坐下角):表示将所有样例判断为反例
    (1,1):表示将所有样例判断为正例
    因此:分类器性能越好 曲线越在左上角
    对不同的ROC曲线进行比较的一个指标是曲线下的面积AUC(area unser the Curve)给出的分类器的平均性能
    一个完美的分类器AUC面积为1 随机猜测为0.5
    
"""
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
fig.clf()  #  clear current fig

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