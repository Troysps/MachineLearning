#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 马氏距离
__author__ = 'Administrator'
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

import numpy as np
from scipy.spatial.distance import mahalanobis
x = np.array([[[1,2,3,4,5],
               [5,6,7,8,5],
               [5,6,7,8,5]],
              [[11,22,23,24,5],
               [25,26,27,28,5],
               [5,6,7,8,5]]])
i,j,k = x.shape

xx = x.reshape(i,j*k).T


y = np.array([[[31,32,33,34,5],
               [35,36,37,38,5],
               [5,6,7,8,5]],
              [[41,42,43,44,5],
               [45,46,47,48,5],
               [5,6,7,8,5]]])


yy = y.reshape(i,j*k).T
print(yy)
print(len(y))

X = np.vstack([xx, yy])
print(X)
V = np.cov(X.T)
print(V)
VI = np.linalg.inv(V)

delta = xx - yy
print('xx\n', xx)
print('yy\n', yy)
D = np.sqrt(np.einsum('nj,jk,nk->n', delta, VI, delta))
print(D)
for i in range(len(xx)):
    print(xx[i], '>>', yy[i], '>>', D[i])


# 计算马氏距离
""" 两个数据xx, yy
    step1:
        数据出于同一分布中
        合并数据X = np.vstack([xx, yy])
    step2:
        计算X.T协方差矩阵 并且求可逆矩阵体现各个特征对数值的影响
        V = np.cov(X.T)
        # np.cov 求样本的协方差
        # np.linalg.inv 求得逆矩阵 体现不同特征对于结果体现的(权重)
        VI = np.linalg.inv(V)
    step3:
        根据公式进行计算
        np.diag 对角矩阵
            dd = [1,2,3]
            dilogg = diag(dd)
            print 'diag=',dilogg
        np.diag(np.sqrt(np.dot(np.dot(xx-yy), VI), (xx-yy).T
        
"""