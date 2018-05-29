#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'Administrator'
__mtime__ = '2018/5/21'
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
import matplotlib.pyplot as plt


def loadDataSet(fileName):
    """
    对文件进行逐行解析，从而得到第行的类标签和整个特征矩阵
    Args:
        fileName 文件名
    Returns:
        dataMat  特征矩阵
        labelMat 类标签
    """
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def selectJrand(i, m):
    """

    :param i:  alpha i index
    :param m:  dataMat m dimension
    :return: j

    """
    while True:
        j = int(np.random.uniform(m))
        if j != i:
            return j

def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj

def smoSimple(dataMat, labelMat, C, toler, maxCycle):
    """

    :param dataMat:    数据的输入空间
    :param labelMat:   数据的输出空间
    :param C:          常量值 松弛变量
    :param toler:      容错率
    :param maxCycle:   最大迭代次数
    :returns
        b        模型的常量值
        alphas   拉格朗日乘子法
    """

    # 输入空间 矩阵化
    dataMatrix = np.mat(dataMat)
    # 输出空间 矩阵化
    labelMat = np.mat(labelMat).transpose()

    # 获取样本数据集 样本数m 维度n
    m, n = np.shape(dataMatrix)

    # 初始化b 以及 拉格朗日参数alphas

    b = 0
    # 对每一输入数据 都有一个拉格朗日参数alpha 初始化alphas 为mx1维度的矩阵
    alphas = np.mat(np.zeros((m, 1)))
    print(np.shape(alphas))

    # 记录循环
    iter = 0
    while (iter < maxCycle):
        # 记录alpha参数是否优化
        alphaPairsChanged = 0
        for i in range(m):

            # 计算alpha[i] 的预测值 与 误差值
            # 将负责的问题转化为 二阶问题 抽取 alphas_i alphas_j 进行优化 将大问题转为小问题
            predXi = float(np.multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[i, :].T)) + b
            # print(predXi)
            Ei = predXi - float(labelMat[i])
            # print(Ei)

            """kkt 详解"""
            # 约束条件 (KKT条件是解决最优化问题的时用到的一种方法。我们这里提到的最优化问题通常是指对于给定的某一函数，求其在指定作用域上的全局最小值)
            # 0<=alphas[i]<=C，但由于0和C是边界值，我们无法进行优化，因为需要增加一个alphas和降低一个alphas。
            # 表示发生错误的概率：labelMat[i]*Ei 如果超出了 toler， 才需要优化。至于正负号，我们考虑绝对值就对了。
            '''
            # 检验训练样本(xi, yi)是否满足KKT条件
            yi*f(i) >= 1 and alpha = 0 (outside the boundary)
            yi*f(i) == 1 and 0<alpha< C (on the boundary)
            yi*f(i) <= 1 and alpha = C (between the boundary)
            '''
            # 不满足kkt条件进行优化
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                # print('需要进行优化')

                # 随机抽取非i 的alphas j
                j = selectJrand(i, m)

                # print('i:{} j:{}'.format(i, j))

                # 计算alphas j 的预测值与误差值
                predXj = float(np.multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[j, :].T)) + b
                print('predXj', predXj)
                Ej = predXj - float(labelMat[j])
                # 记录未优化前的alphas[i] 与 alphas[j] 值
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                # 二变量优化问题
                if (labelMat[j] != labelMat[i]):
                    # \sum ai * yi = k
                    # 如果是异侧 相减 ai-aj=k   那么 定义域为 [k, C + k]
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    # 同侧 相加 ai + aj=k 那么定义域为 [k-c, k]
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])

                # 定义与确定就没有优化空间了 跳出循环
                if L == H:
                    print("没有优化空间 定义域确定")
                    continue

                # 对alphas[j] 进行优化

                # 首先计算其 eta值 eta=2*ab - a^2 - b^2  如果eta>=0那么跳出循环 是正确的
                eta = 2.0*dataMatrix[i, :]*dataMatrix[j, :].T - dataMatrix[i, :]*dataMatrix[i, :].T - dataMatrix[j, :]*dataMatrix[j,:].T
                if eta>=0:
                    print('eta>=0')
                    continue
                # 计算出一个新的alphas[j]值

                print('eta', eta)
                print('Ei', Ei)
                print('Ej', Ej)

                print(labelMat[j]*(Ei - Ej)/eta)
                print(alphas[j])
                # 优化新aj值 aj = yj * (ei-ej)/eta
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta

                # 使用辅助函数调整alphas[j] aj在定义域中
                alphas[j] = clipAlpha(alphas[j], H, L)

                # 检查alphaJ 调整幅度对比 比较小就退出
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving enough")
                    continue

                # 同时优化alpha[i] 优化了aj 那么同样优化 ai += yj*yi(aJ_old - aj)
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                # 分别计算模型的常量值
                # bi = b - Ei - yi*(ai - ai_old)*xi*xi.T - yj(aj-aj_old)*xi*xj.T
                # bj = b - Ej - yi*(ai - ai_old)*xi*xj.T - yj(aj-aj_old)*xj*xj.T
                b1 = b - Ei - labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i, :]*dataMatrix[i, :].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i, :]*dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i, :]*dataMatrix[j, :].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j, :]*dataMatrix[j, :].T
                # 判断哪个模型常量值符合 定义域规则 不满足就 暂时赋予 b = (bi+bj)/2.0
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
        # 在for循环外，检查alpha值是否做了更新，如果在更新则将iter设为0后继续运行程序
        # 知道更新完毕后，iter次循环无变化，才推出循环。
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)
    return b, alphas


def calcWs(alphas, dataArr, classLabels):
    """
    基于alpha计算w值
    Args:
        alphas        拉格朗日乘子
        dataArr       feature数据集
        classLabels   目标变量数据集
    Returns:
        wc  回归系数
    """
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * labelMat[i], X[i, :].T)
    print('w', w)
    return w


def plotfig_SVM(xMat, yMat, ws, b, alphas):
    """
    参考地址：
       http://blog.csdn.net/maoersong/article/details/24315633
       http://www.cnblogs.com/JustForCS/p/5283489.html
       http://blog.csdn.net/kkxgx/article/details/6951959
    """

    xMat = np.mat(xMat)
    yMat = np.mat(yMat)

    # b原来是矩阵，先转为数组类型后其数组大小为（1,1），所以后面加[0]，变为(1,)
    b = np.array(b)[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 注意flatten的用法
    ax.scatter(xMat[:, 0].flatten().A[0], xMat[:, 1].flatten().A[0])

    # x最大值，最小值根据原数据集dataArr[:, 0]的大小而定
    x = np.arange(-1.0, 10.0, 0.1)

    # 根据x.w + b = 0 得到，其式子展开为w0.x1 + w1.x2 + b = 0, x2就是y值
    y = (-b-ws[0, 0]*x)/ws[1, 0]
    ax.plot(x, y)

    for i in range(np.shape(yMat[0, :])[1]):
        if yMat[0, i] > 0:
            ax.plot(xMat[i, 0], xMat[i, 1], 'cx')
        else:
            ax.plot(xMat[i, 0], xMat[i, 1], 'kp')

    # 找到支持向量，并在图中标红
    for i in range(100):
        if alphas[i] > 0.0:
            ax.plot(xMat[i, 0], xMat[i, 1], 'ro')
    plt.show()

def main():
    filename = 'testSet.txt'
    dataMat, labelMat = loadDataSet(filename)


    b, alphas = smoSimple(dataMat, labelMat, 0.6, 0.001, 40)


    ws = calcWs(alphas, dataMat, labelMat)
    plotfig_SVM(dataMat, labelMat, ws, b, alphas)

if __name__ == '__main__':
    main()