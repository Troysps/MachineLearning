# !/usr/bin/env python3
# -*- utf-8 -*-
"""
__title__ = ''
__author__ = LEI
__mtime__ = '2018/5/29'

  we are drowning in information,but starving for knowledge
"""
import numpy as np
import matplotlib.pyplot as plt
import os


class optStruct(object):
    """建立数据结构保存所有重要的值"""
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        """

        :param dataMatIn: 训练数据集输入空间
        :param classLabels: 训练数据集输出空间
        :param C: 松弛变量
        :param toler: 容错率
        :param kTup: 核函数信息
        """
        self.X = dataMatIn
        self.labelMat = classLabels
        self.m = np.shape(dataMatIn)[0]

        self.C = C
        self.tol = toler

        # init model alphas and b
        self.alphas = np.mat(np.zeros((self.m , 1)))
        self.b = 0

        # 误差缓存 第一列给出的是eCache是否有效的标志位,第二列给出的是实际的E值
        self.eCache = np.mat(np.zeros((self.m, 2)))

        # m行m列的矩阵
        # m行m列的矩阵
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


def kernelTrans(X, A, kTup):
    """kernel transfer function

    :param X: training data input data
    :param A: training data input data i index
    :param kTup: kernel function select
    :return:
        K
    """
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        # linear kernel:   m*n * n*1 = m*1
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        # 径向基函数的高斯版本
        K = np.exp(K / (-1 * kTup[1] ** 2))  # divide in NumPy is element-wise not matrix like Matlab
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K

def img2vector(filename):
    """dirname/filenameStr transfer Vector

    :param filename: dirname/filenameStr
    :return
        returnVect
    """
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i + j] = int(lineStr[j])

    return returnVect

def loadImages(dirName):
    """load handle writing training dataSet and transfer Vec
    :param dirName: dir of handle writing training dataSet
    :returns
        dataSet: training dataSet matrix
        labelMat: training dataSet label matrix {-1, 1}
                    -1:handle writing == 9
                    1: handle writing != 9
    """
    labelMat = []
    print(dirName)
    trainingFileList = os.listdir(dirName)
    m = len(trainingFileList)
    dataSet = np.zeros((m, 1024))

    for i in range(m):
        filenameStr = trainingFileList[i]
        fileStr = filenameStr.split('.')[0]
        classNumstr = int(fileStr.split('_')[0])
        if classNumstr == 9:
            labelMat.append(-1)
        else:
            labelMat.append(1)
        # '%s/%s' % (dirName, filenameStr) == dirname/filenameStr
        dataSet[i, :] = img2vector('%s/%s' % (dirName, filenameStr))

    return dataSet, labelMat

def calcEk(oS, i):
    """
    Calculation Ek(计算预测误差 Ek=预测值-真实值)
    :param i: 训练数据集中输入空间具体的某一行
    :param oS: optStruct对象
    :return:
        Ek 预测结果与真实结构比对 计算误差Ek
    """
    # 考虑到采用核函数 实质上求预测值为 与转化后的特征空间k求解
    # predictEk = float(np.multiply(oS.alphas, oS.classLabels).T * (oS.X * oS.X[i, :].T)) + oS.b
    # multiply(mx1, mx1)= mx1 (mx1).T * (m,1) = 1x1
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * oS.K[:, i] + oS.b)
    Ek = fXk - float(oS.labelMat[i])
    return Ek

def selectJrand(i, m):
    """
    随机选择一个整数
    Args:
        i  第一个alpha的下标
        m  所有alpha的数目
    Returns:
        j  返回一个不为i的随机数，在0~m之间的整数值
    """
    j = i
    while j == i:
        j = int(np.random.uniform(0, m))
    return j

def selectJ(i, oS, Ei):
    # 实质上就是在选定ai的情况下, 遍历循环alpha 选取alphaj 计算deltaE 并且记录alphaj的index取值 以及Ej取值

    """
    选择最优的j和Ej
    什么情况下是最优的j和Ej?即:选择最大的误差对应的j进行优化

    内循环的启发式方法
    选择第二个(内循环)alphas中的alpha值
    目标:选择合适的第二个alpha值以保证每次优化中采用最大步长
    该函数的误差与第一个alpha值Ei和下标i有关
    :param i: 输入空间数据集中的第i行
    :param oS: optStruct对象
    :param Ei: 预测结果与真实结果比对 计算误差Ei
    :returns
        j: 随机选出的第j行
        Ej: 预测结果与真实结果比对 计算误差Ej
    """
    maxK = -1  # 设Ei-Ek对应的 最大误差值的index
    maxDeltaE = 0  # 设最大误差=0
    Ej = 0

    # 首先ai的误差值存储在缓存设置中
    oS.eCache[i] = [i, Ei]

    # 非零E值的行(index)的list列表, 所对应的alpha值
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]  # 有效的缓存值列表
    # print('validEcacheList:', validEcacheList)
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:  # don't calc for i
                continue

            # 求Ek误差:预测值-真实值的差
            Ek = calcEk(oS, k)
            # 求Ei-Ek 误差值之差
            deltaE = np.abs(Ei - Ek)

            if (deltaE > maxDeltaE):
                # 选择具有最大步长的j
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    # 如果是第一次循环 则随机选择一个alpha值
    else:
        j = selectJrand(i, oS.m)

        # 求Ek误差: 预测值-真实值的差
        Ej = calcEk(oS, j)

    return j, Ej

def clipAlpha(aj, H, L):
    """调整aj的值 使得aj处于 L <= aj <= H

    :param aj: 目标值
    :param H: 最大值
    :param L: 最小值
    :return
        aj 目标值优化后
    """
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def updateEk(oS, k):
    """计算误差值并存入到这个缓存中

    :param oS:
    :param K:
    :return:
    """
    # 求 误差：预测值-真实值的差
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]
def innerL(i, oS):
    """
    SMO算法内循环
    :param i:
    :param oS:
    :return
        0 or 1
        if (存在alpha对改变):return 1
        else :return 0
    """
    # step1 calc Ei
    Ei = calcEk(oS, i)
    # print('i, Ei', i, Ei)

    # step2 kkt condition
    '''
    # 检验训练样本(xi, yi)是否满足KKT条件
    yi*f(i) >= 1 and alpha = 0 (outside the boundary)
    yi*f(i) == 1 and 0<alpha< C (on the boundary)
    yi*f(i) <= 1 and alpha = C (between the boundary)
    '''
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] < 0)):
        # step3 选择最大的误差对应的j进行优化 效果更明显
        j, Ej = selectJ(i, oS, Ei)

        # print('j, Ej', j, Ej)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()

        # step4 L和H用于将alpha[j]调整定义域[0,C]之间, 如果L==H,则不做任何改变
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            # print("L==H")
            return 0

        # print('L and H:', L, H)

        # step 5 计算eta值 优化alpha[j] 并使用辅助函数优alpha[j] 同时更新误差缓存
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]  # changed for kernel
        if eta >= 0:
            print('eta >= 0')
            return 0
        # print('eta:', eta)
        # 计算出一个新的alpha[j]值 并使用辅助函数优化alpha[j]
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta

        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        # print('优化后的alpha[j]:', oS.alphas[j])
        # 更新误差缓存
        updateEk(oS, j)


        # step7 检查alpha[j]是否只是轻微改变 不是就同幅度改变alpha[i] 并且更新误差缓存
        # 检查alpha[j]是否只是轻微的改变，如果是的话，就退出for循环。
        if (np.abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("j not moving enough")
            return 0

        # 然后alphas[i]和alphas[j]同样进行改变，虽然改变的大小一样，但是改变的方向正好相反
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        # 更新误差缓存
        updateEk(oS, i)

        # step8 带入Ei Ej 模型常量值b
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0

        # print('oS.b', oS.b)
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('linear', 0)):
    """
    完整SMO算法外循环 与smoSimple类似 但是这里的循环退出条件更多
    :param dataMatIn:  训练数据集
    :param classLabels: 类别标签
    :param C: 松弛变量(常量值)
    :param toler: 容错率
    :param maxIter: 推出前的最大循环次数
    :param kTup: 包含核函数信息的元组
    :return: 
        b 模型的常量值
        alphas 拉格朗日乘子
    """
    # 使用class optStruct 保存数据
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True  # 设置值 循环遍历所有数据一次之后调整为flase
    alphaPairsChanged = 0

    # 循环遍历条件: 循环maxIter次并且(alphaPairsChanged存在可改变的值) or 将所有数据遍历一次
    while (((iter < maxIter) and (alphaPairsChanged > 0 )) or (entireSet)):
        alphaPairsChanged = 0

        # entireSet=True or 非边界alpha对没有了, 就开始寻找alpha对,然后决定是否要进行else
        if entireSet:
            for i in range(oS.m):
                # 是否存在alpha对, 存在就+1
                alphaPairsChanged += innerL(i, oS)
                # print(alphaPairsChanged)
            iter += 1
        # 对以及存在的alpha对 选出非边界的alpha值 进行优化
        else:
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
            iter += 1

        # 如果找到alpha对 就优化非边界alpha值 否则继续讯号
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True

    return oS.b, oS.alphas

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


def testDigits(kTup=('rbf', 10)):

    # 1. 导入训练数据
    trainingFile = r'C:\Users\Administrator\Documents\数据挖掘常用算法\Support vector machine\手写数字识别_基于支持向量机\trainingDigits'
    dataArr, labelArr = loadImages(trainingFile)
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    # print('b, alphas', b, alphas[alphas >0])
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    # print("there are %d Support Vectors" % shape(sVs)[0])
    m, n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        # 1*m * m*1 = 1*1 单个预测结果
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1
    print('errorCount', errorCount)
    print("the training error rate is: %f" % (float(errorCount) / m))

    # 2. 导入测试数据
    testFile = r'C:\Users\Administrator\Documents\数据挖掘常用算法\Support vector machine\手写数字识别_基于支持向量机\testDigits'

    dataArr, labelArr = loadImages(testFile)
    errorCount = 0
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    m, n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1
    print('errorCount:', errorCount)
    print("the test error rate is: %f" % (float(errorCount) / m))

def plotfig_SVM(xArr, yArr, ws, b, alphas):
    """
    参考地址：
       http://blog.csdn.net/maoersong/article/details/24315633
       http://www.cnblogs.com/JustForCS/p/5283489.html
       http://blog.csdn.net/kkxgx/article/details/6951959
    """

    xMat = np.mat(xArr)
    yMat = np.mat(yArr)

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


if __name__ == '__main__':
    # # 示例：手写识别问题回顾
    # testDigits(('rbf', 0.1))
    testDigits(('rbf', 5))
    # testDigits(('rbf', 10))
    # testDigits(('rbf', 50))
    # testDigits(('rbf', 100))
    # testDigits(('linear'))