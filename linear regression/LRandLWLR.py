# !/usr/bin/env python3
# -*- utf-8 -*-
"""
__title__ = 线性回归 linear regression
__author__ = LEI
__mtime__ = '2018/6/9'

  we are drowning in information, but starving for knowledge
"""


import numpy as np
import matplotlib.pyplot as plt

def loadData(filename):
    fr = open(filename)
    featureNum = len(fr.readline().strip().split('\t')) - 1
    dataSet = list()
    labelMat = list()
    for lines in fr.readlines():
        lineArr = list()
        lines = lines.split('\t')
        for i in range(featureNum):
            lineArr.append(float(lines[i]))
        dataSet.append(lineArr)
        labelMat.append(float(lines[-1]))
    # print(dataSet)
    # print(labelMat)
    return dataSet, labelMat


def lr(dataSet, labelMat):
    """
    标准方程法解线性回归
    :param dataSet: 样本数据集 特征向量 X
    :param labelMat: 样本数据集 输入变量 Y
    :return
        w    模型系数
    """
    xMat = np.mat(dataSet)
    # print(xMat)
    yMat = np.mat(labelMat).T
    # print(yMat)

    xMatx  = xMat.T * xMat
    if np.linalg.det(xMatx) == 0:
        print('行列式为0 为不可逆矩阵')
        return None
    w = xMatx.I * xMat.T*yMat
    # print(w)
    """
    [[3.00681047]
     [1.69667188]]
    """
    return w


def lr_stand(dataSet, labelMat):
    """
    标准方程法解线性回归
    :param dataSet: 样本数据集 特征向量 X
    :param labelMat: 样本数据集 输入变量 Y
    :return
        w    模型系数
    """
    dataSet, labelMat = regularize(dataSet, labelMat)
    xMat = np.mat(dataSet)
    # print(xMat)
    yMat = np.mat(labelMat)
    # print(yMat)

    xMatx  = xMat.T * xMat
    if np.linalg.det(xMatx) == 0:
        print('行列式为0 为不可逆矩阵')
        return None
    w = xMatx.I * xMat.T*yMat
    # print(w)
    """
    [[3.00681047]
     [1.69667188]]
    """
    return w

def lwlr(weightPoint, dataSet, labelMat, k):
    """
    局部加权线性回归 lwlr
    :param weightPoint: 预测数据点 给预测数据点赋予权重
    :param dataSet: 样本数据集 输入空间 X
    :param labelMat: 样本数据集 输出空间 Y
    :return
        hat_w
    """
    m, n = np.shape(dataSet)
    w = np.mat(np.eye(m))

    # 检查是否可逆
    xMat = np.mat(dataSet)
    yMat = np.mat(labelMat).T

    for i in range(m):
        diff = weightPoint - dataSet[i, :]
        w[i, i] = np.exp((diff*diff.T)/(-2*k**2))
    xTx = xMat.T * (w * xMat)
    if np.linalg.det(xTx) == 0:
        print('行列式为0 该矩阵不可逆')
        return None
    # print(np.shape(hat_w))
    ws = xTx.I * (xMat.T * (w * yMat))
    return weightPoint * ws


def lwlrTest(testSet, dataSet, labelMat, k):
    """
    局部加权线性回归 返回fit_y 测试结果
    :param testSet: 预测数据集
    :param dataSet: 样本数据集 输入空间
    :param labelMat: 样本数据集 输出空间
    :param k: 权重计算 高斯核系数
    :return
        fit_y
    """
    dataSet = np.mat(dataSet)
    m, n = np.shape(dataSet)
    fit_y = np.zeros(m)
    # fit_y = list()
    for i in range(m):
        # print(dataSet[i, :])
        # fit_y[i] = copy_lwlr(testSet[i], dataSet, labelMat, k)
        fit_y[i] = lwlr(testSet[i], dataSet, labelMat, k)

    # print('局部线性加权回归-(fit_y):', fit_y)
    print('局部线性加权回归-(fit_y):', np.shape(fit_y))
    return fit_y

def lwlr_plot(dataSet, labelMat, fit_y):
    """
    局部加权线性回归图像
    """
    yHat = fit_y
    xMat = np.mat(dataSet)
    srtInd = xMat[:, 1].argsort(0)           # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出
    xSort = xMat[srtInd][:, 0, :]
    # print('xSort', xSort)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    print(yHat[srtInd])  # 从小到大排序
    ax.plot(xSort[:, 1], yHat[srtInd])
    ax.scatter(xMat[:, 1].flatten().A[0], np.mat(labelMat).T.flatten().A[0], s=2, c='red')
    plt.show()


def lr_plot(dataSet, labelMat, w):
    x = list()
    for i in dataSet:
        x.append(i[-1])
    y = labelMat

    """失败尝试"""
    # w = np.mat(w).T
    # dataSet = np.mat(dataSet).T
    # print(np.shape(w))
    # print(np.shape(dataSet))
    # print(np.array(w * dataSet))
    # for i in (w * dataSet):
    #     print('i:', i)
    # fit_y = [float(i) for i in np.array(w * dataSet)]

    fit_y = list()
    for i in dataSet:
        fit_y.append(float(np.array(i*w)))

    print(np.shape(fit_y))
    print('fit_y', fit_y)

    fig = plt.subplot(111)
    print(np.shape(x))
    print(np.shape(y))
    rSqure = calcR(y, fit_y)
    fig.scatter(x, y, s=12, c='b', alpha=0.5, edgecolors=None, marker='o')
    fig.plot(x, fit_y, c='r')
    plt.title(rSqure)
    plt.show()


def calcR(y, fit_y):
    """
    计算R的平方  R^2 = 回归平方和 - 总平方和
    总平方和 = \sum (y的实际值 - 平均值) ^ 2
    回归平方和 = 总平方和 - 残差平方和
    残差平方和 = \sum (y的估计值 - 实际值) ^ 2
    :param y: 实际值
    :param fit_y: 估计值
    :return
        R^2 决定系数 表示回归关系可以解释应变量80%的变异
    """
    y = np.mat(y)
    fit = np.mat(fit_y)
    yMean = np.mean(y)

    # print(yMean)
    # print(y - yMean)
    # print(np.sum(np.power((y-yMean), 2)))

    sumSqu = np.sum(np.power((y-yMean), 2))
    # print('总平方和', sumSqu)
    residual_squareSum = np.sum(np.power((fit_y - y), 2))
    # print('残差平方和', residual_squareSum)

    rSqure = (sumSqu - residual_squareSum) / sumSqu
    print(rSqure)
    print('R^2 %.2f %%' % (rSqure*100))
    return rSqure


def rss_error(labelMat, fit_y):
    """平方误差和: 实际值与预测值之差"""
    print('平方误差和:', np.sum(np.power((np.mat(labelMat) - fit_y), 2)))
    return np.sum(np.power((np.mat(labelMat) - fit_y), 2))

def load_abalone_data(filename):
    fr = open(filename)
    data_Set = list()
    label_mat = list()

    featureNum = int(len(fr.readline().strip().split('\t'))-1)

    for lines in fr.readlines():
        line = lines.strip().split('\t')
        # print(line)
        lineArr = list()
        for index in range(featureNum):
            # print('index', index)
            lineArr.append(float(line[index]))
        data_Set.append(lineArr)

        label_mat.append(float(line[-1]))
    # print('xxxxx')
    # print(data_Set)
    return data_Set, label_mat

def ridge_regression(dataSet, labelMat, lamb=10):
    """
    岭回归 求解模型参数
    优缺点: 损失了无偏性,得到较高的计算精度
    :param dataSet: 数据集 输入空间 X
    :param labelMat:数据集 输出空间 Y
    :param lamb: lambda 系数
    :return
        w 模型参数
    """
    xMat = np.mat(dataSet)
    yMat = np.mat(labelMat)
    yMat = np.mat(labelMat).T
    i = np.eye(np.shape(dataSet)[1])
    demo = xMat.T * xMat + lamb * i
    if np.linalg.det(demo) == 0:
        print('行列式为0 无法计算逆矩阵')
    ws = demo.I * (xMat.T * yMat)
    # print('ws---', np.shape(ws))
    return ws


def ridge_test(dataSet, labelMat):
    """
    转化数据集为均值为0,方差为1的数据集
    :param dataSet: 输入空间
    :param labelMat: 输出空间
    :param lamb: lambda 系数
    :return
        w_list 模型参数集合
    """
    # dataSet = np.mat(dataSet)
    # labelMat = np.mat(labelMat).T
    # # 计算均值
    # xMean = np.mean(dataSet, 0)
    # yMean = np.mean(labelMat, 0)
    #
    # # print(xMean)
    # # print(yMean)
    # xVar = np.var(dataSet, 0)
    # dataSet = (dataSet - xMean) / xVar
    # labelMat = labelMat - yMean
    dataSet, labelMat = regularize(dataSet, labelMat)
    # print('dataSet', dataSet)
    # print('labelMat', labelMat)

    lamb = 30
    wMat = np.zeros((lamb, np.shape(dataSet)[1]))
    # print('wMat', np.shape(wMat))
    for i in range(lamb):
        ws = ridge_regression(dataSet, labelMat.T, lamb=np.exp(i-10))

        wMat[i, :] = ws.T
    return wMat

def regularize(dataSet, labelMat):
    """
    按列标准化数据
    :param dataSet:
    :param labelMat:
    :return:
    """
    dataSet = np.mat(dataSet)
    labelMat = np.mat(labelMat).T
    xMean = np.mean(dataSet, 0)
    yMean = np.mean(labelMat, 0)
    xVar = np.var(dataSet, 0)

    dataSet = (dataSet - xMean) / xVar
    labelMat = labelMat - yMean

    return dataSet, labelMat

def ridge_regress_plot(wMat):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(wMat)
    plt.show()


def stepwise_regression(dataSet, labelMat, alpha, numCycle):
    """
    前向逐步算法
    伪代码
        1.数据标准化 分布满足0均值 方差为1
        2.对每次迭代过程
            3.设置当前最小误差 lowestError 为无穷大
            4.对每个特征
                5.增大或者缩小
                    6.  改变一个系数得到一个新的w
                        计算新w下的误差
                        如果误差小于当前最小误差, 设置为wbest
                    将w设置为新的wbest
    :param dataSet: 数据集 输入空间
    :param labelMat:数据集 输出空间
    :param alpha: 学习速率
    :param numCycle: 最大迭代次数
    :return
        wMat  每次的wbest组成的列表
    """
    dataSet = np.mat(dataSet)
    labelMat = np.mat(labelMat).T
    xMean = np.mean(dataSet, 0)
    yMean = np.mean(labelMat, 0)
    xVar = np.var(dataSet, 0)

    dataSet = (dataSet - xMean) / xVar
    labelMat = labelMat - yMean

    print(np.shape(dataSet))  # (4176, 8)
    print(np.shape(labelMat))  #  (4176, 1)
    feature_num = np.shape(dataSet)[1]
    print(feature_num)  # 8
    w = np.zeros((1, feature_num))
    w_test = w.copy()
    wbest = None
    wMat = np.zeros((numCycle, feature_num))
    print(np.shape(w))  # (1, 8)
    for cycle in range(numCycle):
        lowestError = np.inf
        for i in range(feature_num):
            for sign in [-1, 1]:
                w_test = w.copy()
                w_test[:, i] += alpha * sign
                error = np.sum(np.power((dataSet * w_test.T - labelMat), 2))
                if error < lowestError:
                    lowestError = error
                    wbest = w_test
        w = wbest.copy()
        wMat[cycle, :] = wbest
    # print(wMat)
    return wMat


def abalone_stepwise_regression():
    filename = 'abalone.txt'
    dataSet, labelMat = load_abalone_data(filename)
    result_stepwise = stepwise_regression(dataSet, labelMat, alpha=0.01, numCycle=200)
    print('result', result_stepwise[-1, :])
    # dataSet, labelMat = regularize(dataSet, labelMat)
    result_lr = lr_stand(dataSet, labelMat)
    print('result_lr:', result_lr.T)

def abalone_predict_ridge_regress():
    filename = 'abalone.txt'
    dataSet, labelMat = load_abalone_data(filename)
    # w = ridge_regression(dataSet, labelMat)
    # print(w)
    # lr_plot(dataSet, labelMat, w)
    wMat = ridge_test(dataSet, labelMat)
    print('wMat,', wMat)
    ridge_regress_plot(wMat)


def abalone_predict_project():
    filename = 'abalone.txt'
    # fr = open(filename)
    dataSet, labelMat = load_abalone_data("abalone.txt")
    # print('abX,', np.shape(dataSet), type(labelMat[0][0]))
    # print('abY,', np.shape(dataSet), type(labelMat[0]))
    """
    abX, (4177, 8) <class 'float'>
    abY, (4177,) <class 'float'>
    """
    # 使用不同的核进行预测
    fit_y01 = lwlrTest(dataSet[0:99], dataSet[0:99], labelMat[0:99], 0.1)
    fit_y1 = lwlrTest(dataSet[0:99], dataSet[0:99], labelMat[0:99], 1)
    fit_y10 = lwlrTest(dataSet[0:99], dataSet[0:99], labelMat[0:99], 10)


    # 打印出不同的核预测值与训练数据集上的真实值之间的误差大小
    error01 = rss_error(labelMat[0:99], fit_y01)
    error1 = rss_error(labelMat[0:99], fit_y1)
    error10 = rss_error(labelMat[0:99], fit_y10)

    # 打印出不同的核预测值与训练数据集上的r^2 拟合度
    r_square01 = calcR(labelMat[0:99], fit_y01)
    r_square1 = calcR(labelMat[0:99], fit_y1)
    r_square10 = calcR(labelMat[0:99], fit_y10)

    # 打印出 不同的核预测值 与 新数据集（测试数据集）上的真实值之间的误差大小
    new_fit_y01 = lwlrTest(dataSet[100:199], dataSet[0:99], labelMat[0:99], 0.1)
    new_fit_y1 = lwlrTest(dataSet[100:199], dataSet[0:99], labelMat[0:99], 1)
    new_fit_y10 = lwlrTest(dataSet[100:199], dataSet[0:99], labelMat[0:99], 10)

    new_error01 = rss_error(labelMat[100:199], new_fit_y01)
    new_error1 = rss_error(labelMat[100:199], new_fit_y1)
    new_error10 = rss_error(labelMat[100:199], new_fit_y10)


def main():
    filename = 'data.txt'
    dataSet, labelMat = loadData(filename)
    # w = lr(dataSet, labelMat)
    # lr_plot(dataSet, labelMat, w)
    # fit_y = lwlrTest(dataSet, dataSet, labelMat, k=1)
    # fit_y = lwlrTest(dataSet, dataSet, labelMat, k=0.01)
    fit_y = lwlrTest(dataSet, dataSet, labelMat, k=0.03)
    lwlr_plot(dataSet, labelMat, fit_y)
    error = rss_error(labelMat, fit_y)
    # 查看平方误差和
    print('error', error)  # 0.03 0.0678212359601 # 0.01 1.16751327518 # 1 1.3520374286
    r_square = calcR(labelMat, fit_y)
    # 查看r^2 相关系数
    print('r^2: %.2f %%' % (r_square*100))   # 0.03 99.45 %   # 0.01 99.70 % #  1 97.30 %

if __name__ == '__main__':

    # main()  # 线性回归与局部线性回归
    # abalone_predict_project()  # 局部线性回归 鲍鱼年龄预测项目
    # abalone_predict_ridge_regress()  # 岭回归  鲍鱼年龄预测项目
    abalone_stepwise_regression()  # 前向逐步算法 鲍鱼年龄预测项目