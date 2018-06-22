# !/usr/bin/env python3
# -*- utf-8 -*-
"""
__title__ = Singular Value Decomposition   奇异值分解
__author__ = LEI
__time__ = '2018/6/21'

  we are drowning in information,but starving for knowledge
"""
import numpy as np

# print(__doc__)


def load_exData():
    return [[1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1]]


def calc_singular_value(sigma):
    """
    奇异值   保留矩阵中90%的能量信息
    :param sigma:
    :return:
    """
    sigma_squre = np.power(sigma, 2)
    sigma_sum = np.sum(sigma_squre)
    print('sigma_sum', sigma_sum)
    print(len(sigma))
    for i in range(len(sigma)):
        # print('sigma_squre', i, sigma_squre[i])
        result = np.sum(sigma_squre[:i])

        # result += np.power(sigma[i], 2) + np.power(sigma[i+1], 2)
        # print(result)
        if result > (sigma_sum*0.90):
            return i


def dimensionality_reduction(u, sigma, vt, sigma_info):
    """
    实质上只需要保留三个奇异值就能保留矩阵的重要信息
    :param u:
    :param sigma:
    :param vt:
    :param sigma_info:
    :return:
    """
    sigma_reduce = np.mat(np.zeros((sigma_info+1, sigma_info+1)))
    for i in range(sigma_info+1):
        sigma_reduce[i, i] = sigma[i]
    print('sigma_reduce\n', sigma_reduce)

    data_reduce = u[:, : sigma_info+1] * sigma_reduce * vt[:sigma_info+1, :]
    print('data_reduce', data_reduce)
    print('shape data_reduce', np.shape(data_reduce))
    return data_reduce



def eur_distance_sim(mat_a, mat_b):
    """
    欧式距离相似度
    :param mat_a:
    :param mat_b:
    :return:
    """
    distance = np.linalg.norm(mat_a-mat_b, 2)
    return 1 + 1/(1+distance)


def pears_sim(inA, inB):
    """
    皮尔逊相关系数
    :param inA:
    :param inB:
    :return:
    """
    # 如果不存在，该函数返回1.0，此时两个向量完全相关。
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * np.corrcoef(inA, inB, rowvar=0)[0][1]


def cos_sim(mat_a, mat_b):
    """
    余弦相似度 相似度度量
    :param mat_a:
    :param mat_b:
    :return:
    """
    num = mat_a.T * mat_b
    demon = np.linalg.norm(mat_a) * np.linalg.norm(mat_b)
    return 0.5 + 0.5 * num / demon


def main():
    data = load_exData()
    print('shape data', np.shape(data))
    print('before reduce', data)
    u, sigma, vt = np.linalg.svd(data)
    print('sigma', sigma)
    sigma_info = calc_singular_value(sigma)
    print('sigma', type(sigma), dir(sigma))
    print('sigma info', sigma_info)
    data_reduce = dimensionality_reduction(u, sigma, vt, sigma_info)
    a = np.mat([[1, 1, 1, 1], [2, 2, 2, 2]]).T
    print('a', a)
    eur_similar = eur_distance_sim(a[:, 0], a[:, 1])
    print('欧式距离度量相似度', eur_similar)   # 欧式距离
    correlation_similar = pears_sim(a[:, 0], a[:, 1])
    print('皮尔逊相关系数度量相似度', correlation_similar)   # 相关系数
    cos_similar = cos_sim(a[:, 0], a[:, 1])
    print('余弦相似度度量相似度', cos_similar)   # 余弦相似度


if __name__ == '__main__':
    main()
