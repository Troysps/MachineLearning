# !/usr/bin/env python3
# -*- utf-8 -*-
"""
__title__ = Principal Component Analysis  主成分分析降维
__author__ = LEI
__time__ = '2018/6/20'

  we are drowning in information,but starving for knowledge
"""

import numpy as np

print(__doc__)

"""
    算法实现 PCA 主成分分析
在数据转化为前N个主成分的伪代码大致如下:
    去除平均值
    计算协方差矩阵
    计算协方差矩阵的特征值和特征向量
    将特征值从大到小排序
    保留最上面的N个特征向量
    将数据转化到上述N个特征向量构建的新空间中

"""
# def loadDataSet(fileName, delim='\t'):
#     fr = open(fileName)
#     stringArr = [line.strip().split(delim) for line in fr.readlines()]
#     datArr = [map(float, line) for line in stringArr]
#     return mat(datArr)


def load_data(filename, delim='\t'):
    fr = open(filename)
    data_set = list()
    string_set = [line.strip().split(delim) for line in fr.readlines()]
    # print('string_set', string_set)
    for line in string_set:
        data_set.append(list(map(float, line)))
    return np.mat(data_set)


def pca(data_set, top_features=999999999):
    """
    principal component analysis 主成分降维
    :param data_set: 数据集
    :param top_features: 去N个特征向量
    :return:
        low_data_mat   降维后的数据集
        recon_mat      新的数据集空间

    """
    # step1 去除均值并且计算协方差矩阵
    data_mean = np.mean(data_set, axis=0)
    # print('data_mean', data_mean)
    mean_removed = data_set - data_mean

    cov_mat = np.cov(mean_removed, rowvar=0)
    # print('cov mat', cov_mat)
    # step2 计算协方差矩阵的特征值和特征向量
    eig_vals, eig_vects = np.linalg.eig(np.mat(cov_mat))
    print('eig vals', eig_vals)
    print('eig_vects', eig_vects)
    # step3 将特征值从大到小排序 保留最上面的N个特征向量
    eig_vals_index = np.argsort(eig_vals)   # 从小到大
    # print('eig_vals_index', eig_vals_index)
    eig_vals_index = eig_vals_index[:-(top_features + 1): -1]   # 从大到小
    print('eig_vals_index', eig_vals_index)
    read_eig_vects = eig_vects[:, eig_vals_index]    # 保留最上面的N个特征向量
    # step4 将数据转换到新空间
    low_data_mat = mean_removed * read_eig_vects   # 1000x2 * 2x1 = 1000x1
    recon_mat = (low_data_mat * read_eig_vects.T) + data_mean
    # print('low_data_mat', low_data_mat)
    # print('recon_mat', recon_mat)
    return low_data_mat, recon_mat


def mat_fig(data_set, recon_mat):
    import matplotlib.pyplot as plt
    figure = plt.figure()
    ax = figure.add_subplot(111)
    ax.scatter(data_set[:, 0].flatten().A[0], data_set[:, 1].flatten().A[0], marker='^', s=90)
    ax.scatter(recon_mat[:, 0].flatten().A[0], data_set[:, 1].flatten().A[0], marker='o', s=50, c='r')
    # figure.show()
    plt.show()


def replace_nan_with_mean():
    data_set = load_data('secom.data', ' ')
    num_feature = np.shape(data_set)[1]
    for i in range(num_feature):
        mean_val = np.mean(data_set[np.nonzero(~np.isnan(data_set[:, i].A))[0], i])
        data_set[np.nonzero(np.isnan(data_set[:, i].A))[0], i] = mean_val
    return data_set


def main():
    # filename = 'testSet.txt'
    # data_set = load_data(filename)
    # print('data_set', data_set)
    # low_data_mat, recon_mat = pca(data_set, top_features=1)
    # print('降维后的数据集', np.shape(low_data_mat))
    # print('low_data_mat', low_data_mat)
    # print('新的数据集空间 recon_mat', recon_mat)
    # mat_fig(data_set, recon_mat)
    data_set = replace_nan_with_mean()
    print('data_set', data_set)


if __name__ == '__main__':
    main()
