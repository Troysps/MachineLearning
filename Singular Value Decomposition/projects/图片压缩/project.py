# !/usr/bin/env python3
# -*- utf-8 -*-
"""
__title__ = SVD 图像压缩
__author__ = LEI
__time__ = '2018/6/22'

  we are drowning in information,but starving for knowledge
"""
import numpy as np

# 图像压缩函数
# 加载并转换数据


def imgLoadData(filename):
    myl = []
    # 打开文本文件，并从文件以数组方式读入字符
    for line in open(filename).readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    # 矩阵调入后，就可以在屏幕上输出该矩阵
    myMat = np.mat(myl)
    return myMat


# 打印矩阵
def printMat(inMat, thresh=0.8):
    # 由于矩阵保护了浮点数，因此定义浅色和深色，遍历所有矩阵元素，当元素大于阀值时打印1，否则打印0
    for i in range(32):
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                print(1, end=''),
            else:
                print(0, end=''),
        print('')


# 实现图像压缩，允许基于任意给定的奇异值数目来重构图像
def imgCompress(numSV=3, thresh=0.8):
    """imgCompress( )
    Args:
        numSV       Sigma长度
        thresh      判断的阈值
    """
    # 构建一个列表
    myMat = imgLoadData('0_5.txt')
    print('myMat shape', np.shape(myMat))

    print("****original matrix****")
    # 对原始图像进行SVD分解并重构图像e
    printMat(myMat, thresh)

    # 通过Sigma 重新构成SigRecom来实现
    # Sigma是一个对角矩阵，因此需要建立一个全0矩阵，然后将前面的那些奇异值填充到对角线上。
    U, Sigma, VT = np.linalg.svd(myMat)
    # SigRecon = mat(zeros((numSV, numSV)))
    # for k in range(numSV):
    #     SigRecon[k, k] = Sigma[k]

    # 分析插入的 Sigma 长度
    # analyse_data(Sigma, 20)

    SigRecon = np.mat(np.eye(numSV) * Sigma[: numSV])
    reconMat = U[:, :numSV] * SigRecon * VT[:numSV, :]
    print('reconMat shape', np.shape(reconMat))
    print("****reconstructed matrix using %d singular values *****" % numSV)
    printMat(reconMat, thresh)


def main():
    imgCompress(2)

if __name__ == '__main__':
    main()