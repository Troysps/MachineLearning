#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = Simplify SMO linear classify SVM support vector machine
__author__ = 'Administrator'
__mtime__ = '2018/5/20'
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
    SMO 实现 support vector machine
    用途: 用于训练 SVM
    思想: 将大优化问题转化分解为多个小优化问题来求解
    原理: 每次循环选择两个alpha进行优化处理 一旦找出一对合适的alpha 就增大一个的同时减少另一个
            这里的合适需要符合一定的条件
            1.这两个alpha必须要在间隔边界之外
            2.这两个alpha还没有进行过区间化处理或者不在边界上
            之所以要同时改变两个alpha:
                原因:需要满足约束条件 $\(\sum_{i=1}^{m} a_i·label_i=0\)$
    
    伪代码:
        
        创建一个 alpha 向量并将其初始化为0向量
        当迭代次数小于最大迭代次数时(外循环)
            对数据集中的每个数据向量(内循环)：
                如果该数据向量可以被优化
                    随机选择另外一个数据向量
                    同时优化这两个向量
                    如果两个向量都不能被优化，退出内循环
            如果所有向量都没被优化，增加迭代数目，继续下一次循环
"""

import numpy as np
import matplotlib.pyplot as plt



def loadDataSet(filename):
    """
    对文件进行逐行解析 得到类标签和整个特征矩阵
    :param filename: 文件名字
    :returns
        dataMat 特征矩阵
        labelMat 类标签
    """
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def smoSimple(dataMatIn, classLabel, C, toler, maxIter):
    """
    SMO Simple
    :param dataMatIn: 特征集合
    :param classLabel: 类别集合
    :param C: 松弛变量(常量值) 允许有些数据点可以处于分割面的错误一侧
              控制最大化间隔和保证大部分的函数间隔小于1.0 这两个目标的权重
              可以通过调节该参数达到不同的结果
    :param toler: 容错率 是指在某个体系中能减少一些因素或选择对某个系统产生不稳定的概率
    :param maxIter: 退出前最大的循环次数
    :returns
        b 模型的常量值
        alphas 拉格朗日乘子
    """
    # 输入空间 矩阵化
    dataMatrix = np.mat(dataMatIn)
    # 输出空间 矩阵化 转置
    classLabel = np.mat(classLabel).transpose()
    m, n = np.shape(dataMatrix)

    print(m, n)

    # 初始化 b 和 alphas
    b = 0
    alphas = np.mat(np.zeros((m, 1)))
    print("alphas shape", np.shape(alphas))

    # 记录在没有alpha改变的情况下遍历数据的次数
    iter = 0
    while (iter < maxIter):
        # 记录alpha是否已经优化 每次循环时 设置为0 然后再对整个集合顺序遍历
        alphaPairsChanged = 0
        for i in range(m):
            # 已知alphas 根据公式(1) y[i] = w ^T X[i] + b
            # 由拉格朗日乘子法 可得公式(2) w = \sum\limits_{i=1}^N alpha_i * y_i * x_i 带入公式2
            # print(np.multiply(classLabel, alphas))
            fXi = float(np.multiply(classLabel, alphas).T * (dataMatrix * dataMatrix[i, :].T)) + b
            print('predict fXi:\n', fXi)
            # 计算误差error
            Ei = fXi - float(classLabel[i])
            print('predict error:\n', Ei)

            # 满足KKT条件 进行优化

            # 约束条件 (KKT条件是解决最优化问题的时用到的一种方法。我们这里提到的最优化问题通常是指对于给定的某一函数，求其在指定作用域上的全局最小值)
            # 0<=alphas[i]<=C，但由于0和C是边界值，我们无法进行优化，因为需要增加一个alphas和降低一个alphas。
            # 表示发生错误的概率：labelMat[i]*Ei 如果超出了 toler， 才需要优化。至于正负号，我们考虑绝对值就对了。
            '''
            # 检验训练样本(xi, yi)是否满足KKT条件
            yi*f(i) >= 1 and alpha = 0 (outside the boundary) 在边界外
            yi*f(i) == 1 and 0<alpha< C (on the boundary) 在边界上
            yi*f(i) <= 1 and alpha = C (between the boundary) 在边界内
            '''
            if ((classLabel[i]*Ei < -toler) and (alphas[i] < C)) or ((classLabel[i] * Ei > toler) and (alphas[i] > 0)):
                print("超出容错率 需要优化")
                # 如果该向量可以被优化 那么随机选取一个非i的点 进行优化比较
                # 选取非i点 点j
                print('选取非i点 点j 且与i点 满足约束条件 aiyi + ajyj = 0')
                j = selectJrand(i, m)
                # 预测j的结果 与 误差
                # j = 88
                print('计算j的结果与误差')
                fXj = float(np.multiply(alphas, classLabel).T * (dataMatrix*dataMatrix[j, :].T)) + b
                Ej = fXj - float(classLabel[j])
                print('计算j的结果与误差 误差为:', Ej)

                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                print('记录未更新时候的 i j alpha值')
                # L和H用于调整alphas[j]的取值 使得其取值范围为[0, C]之间
                # 分情况设置Low High --- 情况1 classLabel[i] 与 classLabel[j]同侧 相加 情况2 异侧 相减
                print('二变量优化 同侧或异侧时 定义域不同')

                if (classLabel[i] != classLabel[j]):
                    # 异侧
                    high = min(C, C + alphas[j] - alphas[i])
                    low = max(0, alphas[j] - alphas[i])
                    print('异侧\n high %s low %s' % (high, low))
                else:
                    # 同侧
                    high = min(C, alphas[j] + alphas[i] -C)
                    low = max(0, alphas[j] + alphas[i])
                    print('同侧\n high:{} low:{}'.format(high, low))

                if high == low:
                    # 如果相同就没有办法优化了
                    print("high == low")
                    continue

                # eta是alphas[j]的最优修改量 如果eta==0 需要退出for循环的当前迭代过程
                # 《统计学习方法》 <序列最小最优算法>
                # eta = 2xi*xj - xi^2 - xj^2

                print('dataMatrix i:', dataMatrix[i])
                print('dataMatrix j:', dataMatrix[j])
                eta = 2.0 * dataMatrix[i, :]*dataMatrix[j, :].T - dataMatrix[i, :]*dataMatrix[i, :].T - dataMatrix[j, :]*dataMatrix[j, :].T
                print('计算最优修改量 eta:', eta)
                if eta >= 0:
                    print("eta>=0")
                    continue

                # 计算出一个新的alphas[j]值

                alphas[j] -= classLabel[j]*(Ei - Ej)/eta

                # 并使用辅助函数 使用L H对其进行调整
                alphas[j] = clipAlpha(alphas[j], high=high, low=low)
                print('使用辅助函数调整 alphas j :', alphas[j])


                # 检查alphas[j] 是否只是轻微改变 如果是的 退出for循环
                if (np.abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving enough")
                    continue

                # 为了满足约束条件 \sum yi * ai = 0 在alphas[j]改变的同时alphas[i]同样进行改变 方向相反
                alphas[i] += classLabel[j] * classLabel[i] * (alphaJold - alphas[j])

                # 在对alpha[i] 进行优化之后 给这两个alpha值计算常数b
                # w= Σ[1~n] ai*yi*xi => b = yj- Σ[1~n] ai*yi(xi*xj)
                # 所以：  b1 - b = (y1-y) - Σ[1~n] yi*(a1-a)*(xi*x1)
                # 为什么减2遍？ 因为是 减去Σ[1~n]，正好2个变量i和j，所以减2遍

                b1 = b - Ei - classLabel[i]*(alphas[i]-alphaIold)*dataMatrix[i, :]*dataMatrix[i, :].T - classLabel[j]*(alphas[j]-alphaJold)*dataMatrix[i, :]*dataMatrix[j, :].T
                b2 = b - Ej - classLabel[i]*(alphas[i]-alphaIold)*dataMatrix[i, :]*dataMatrix[j, :].T - classLabel[j]*(alphas[j]-alphaJold)*dataMatrix[j, :]*dataMatrix[j, :].T

                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0

                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))

            # 在 for 循环外 检查 alpha值是否做了更新 如果更新则将iter设置为0后继续运行程序
            # 直到到跟新完毕后 iter次循环无变化 才推出循环

        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)
        return b, alphas


def selectJrand(i, m):
    """
    随机选择一个正数
    :param i: 优化的第一个alpha的下标
    :param m: 所有alpha的数目
    :return
        j 返回一个不为i的随机数 在0~m之间取值
    """
    print('selectJrand i:', i)
    # while True:
    #     j = int(np.random.uniform(0, m))
    #     if j != i:
    #         print('selectJrand j:', j)
    #         return j

    j = i
    while j == i:
        j = int(np.random.uniform(0, m))
    return j


def clipAlpha(alphas_j, high, low):
    if alphas_j > high:
        alphas_j = high
    if low > alphas_j:
        alphas_j = low

    return alphas_j

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
    print('----dataMat----\n', dataMat, '\n------labelMat-----\n', labelMat)
    b, alphas = smoSimple(dataMat, labelMat, 0.6, 0.001, 40)
    print('b\n', b)
    print('alphas\n', alphas)
    print(alphas[alphas > 0])

    ws = calcWs(alphas, dataMat, labelMat)
    plotfig_SVM(dataMat, labelMat, ws, b, alphas)

if __name__ == '__main__':
    main()