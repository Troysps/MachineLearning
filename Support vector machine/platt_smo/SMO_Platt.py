#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = PLATT SMO 完整smo算法
__author__ = LEI
__mtime__ = '2018/5/23'
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
    Platt SMO 算法 是通过一个外循环来选择第一个alpha值 并且对其选择过程中会在两种方式之间交替
    一种方式就是在所有数据集上进行单遍扫描 另一种方式则是在非边界alpha中实现单遍扫描
    
    非边界alpha指的是:不等于边界0或者C的alpha值 对整个数据集的扫描相当容易 而实现非边界alpha值得扫描时
    首先需要建立这些alpha值得列表 然后对这个表进行遍历 同时 该步骤会跳过哪些已知的不会改变的alpha值
    
    在选择第一个alpha值后 算法会通过一个内循环来选择第二个alpha值 在优化过程中 会通过最大化步长的方式来获得第二个alpha值
    smo算法：在简化版smo算法中 我们会选择j之后计算错误率Ej 
    platt smo算法：但在这里 我们会建立一个全局的缓存用于保存误差值 从而使得步长或者说Ei-Ej最大的alpha值
"""
import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(fileName):
    """loadDataSet（对文件进行逐行解析，从而得到第行的类标签和整个数据矩阵）
    Args:
        fileName 文件名
    Returns:
        dataMat  数据矩阵
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


class optStructK(object):
    def __init__(self, dataMatIn, classLabels, C, toler):
        """初始化参数结构"""
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))


def calcEk(oS, k):
    """calcEk 求Ek误差： 预测值-真实值的差
    在smo算法中出现次数较多 因此将其作为一个单独的方法
    :param oS: optStruct对象
    :param k:具体的某一行
    :return: Ei 预测结果与真实结果比对 计算误差Ek
    """

    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i, oS, Ei):
    """selectJ 返回最优的j和Ej
    内循环的启发式方法
    选择第二个内循环alpha的alpha值
    这里的目标是选择合适的第二个alpha值以保证每次优化中采用最大部长
    该函数的误差与第一个alpha值Ei与下标i有关

    :param i:  具体的第i行 index
    :param oS: optStruct对象
    :param Ei: 预测结果与真实结果比对
    :return:
        j      随机选出的第j行
        Ej     预测结果与真实结果比对 计算误差Ej
    """
    maxK = -1
    maxDelteE = 0
    Ej = 0
    # 首先将输入值Ei在缓存中设置成为有效的 这里的有效意为这它已经计算好了
    oS.eCache[i] = [1, Ei]
    """
        np.nonzero 返回非零数组的索引
        a = np.arange(-10, 10)  array([-10,  -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1,   0,   1,   2, 3,   4,   5,   6,   7,   8,   9])
        np.nonzero(a)  (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=int64),)
        return list number not==zero index

        
    """
    """
        np.matrix.A means transfer matrix to array
        x = np.matrix(np.arange(12).reshape((3,4)))
        print(type(x)) ====> matrix
        xi = x.A
        print(type(xi)) ====> array
    """
    # 返回非0的：行列值
    vaildEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
    if (len(vaildEcacheList)) > 1:
        # 在所有值上进行循环 并选择使其改变最大的那个值
        for k in vaildEcacheList:
            if k == i:
                # don't calc for i, waste of time
                continue

            # 求 Ek误差: 预测值-真实值
            Ek = calcEk(oS, k)
            deltaE = np.abs(Ei - Ek)
            if (deltaE > maxDelteE):
                maxK = k
                maxDelteE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        # 如果是第一次循环 则随机选择一个alpha值
        j = selectJrand(i, oS.m)

        # 求 Ek误差: 预测值-真实值的差
        Ej = calcEk(oS, j)

    return j, Ej

def updateEk(oS, k):
    # after any alpha has changed update the new value in the cache
    """ updateEk 计算误差值并存入缓存中

    在对alpha值进行优化之后会用到这个值
    :param oS:  optStruct对象
    :param k: 某一列的行号
    :return:
    """
    # 求 误差: 预测值-真实值的差

    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

def innerL(i, oS):
    """innerL
    内循环代码
    :param i: 具体的某一行
    :param oS: optStruct对象
    :returns:
        0 找不到最优的值
        1 找到了最优的值 并且oS.Cache到缓存中

    """
    # 求Ek误差: 预测值-真实值的差
    Ei = calcEk(oS, i)
    """
    约束条件 
    (KKT条件是解决最优化问题的时用到的一种方法 我们这里提到的最优化问题通常是指对于给定的某一函数, 求其在指定作用域上的全局最小值)
    0<=alphas[i]<=C 但是由于0和C是边界值 我们无法进行优化 因为需要增加一个alphas和降低一个alphas
    toler表示发生错误的概率 labelMat[i]*Ei 如果超出了toler 才需要优化 至于正负号 简单的考虑绝对值就对了
    """
    """检验训练样本(xi, yi)是否满足KKT条件
    yi*f(i) >= 1 and alpha = 0 (outside the boundary)
    yi*f(i) == 1 and 0<alpha<C (on the boundary)
    yi*f(i) <= 1 and alpha = C (between the boundary)
    """
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        # 选择最大的误差对应的j进行优化 效果更佳明显
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()

        # L和H 用于将alphas[j]调整到0-C之间 如果L==H就不做任何改变 直接return 0
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = max(oS.C, oS.alphas[j] + oS.alphas[i])

        if L == H:
            print('L==H')
            return 0

        # eta 是alphas[j]的最优修改量 如果eta==0 需要退出for循环的当前迭代过程
        eta = 2.0 * oS.X[i, :]*oS.X[j, :].T - oS.X[i, :]*oS.X[i, :].T - oS.X[j, :]*oS.X[j, :].T
        if eta >= 0:
            print('eta>=0')
            return 0

        # 计算出一个新的alphas[j]值
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej) / eta
        # 并使用辅助函数 以及L和H对其进行调整
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        # 更新误差缓存
        updateEk(oS, j)

        # 检查alphas[j]是否只是轻微的改变 如果是的话 就退出for循环
        if (np.abs(oS.alphas[j] - alphaJold) < 0.00001):
            print('j not moving enough')
            return 0

        # 然后alphas[i]和alphas[j]同样进行改变 虽然改变的大小一样 但是改变的方向正好相反
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold-oS.alphas[j])
        # 更新误差缓存
        updateEk(oS, i)

        # 在对alpha[i], alpha[j] 进行优化之后，给这两个alpha值设置一个常数b。
        # w= Σ[1~n] ai*yi*xi => b = yj Σ[1~n] ai*yi(xi*xj)
        # 所以：  b1 - b = (y1-y) - Σ[1~n] yi*(a1-a)*(xi*x1)
        # 为什么减2遍？ 因为是 减去Σ[1~n]，正好2个变量i和j，所以减2遍
        b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i] - alphaIold)*oS.X[i, :]*oS.X[i, :].T - oS.labelMat[j]*(oS.alphas[j] - alphaJold)*oS.X[i, :]*oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMat[i]*(oS.alphas[i] - alphaIold)*oS.X[i, :]*oS.X[j, :].T - oS.labelMat[j]*(oS.alphas[j] - alphaJold)*oS.X[j, :]*oS.X[j, :].T
        if ((0 < oS.alphas[i]) and (oS.C > oS.alphas[i])):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 0
    else:
        return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter):
    """
    完整smo算法外循环 与smoSimple有些类似 但是这里的循环退出条件更多一些
    :param dataMatIn:   数据集
    :param classLabels: 类别标签
    :param C:           松弛变量(常量值) 允许有些数据点可以处理分割面的错误一侧
    :param toler:       容错率
    :param maxIter:     退出前最大的循环次数
    :return
        b               模型的常量值
        alphas          拉格朗日乘子
    """
    # 创建一个 optStruct 对象
    oS = optStructK(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0

    # 循环遍历: 循环maxIter次 并且(alphaPairsChanged存在可以改变 or 所有行遍历一遍)
    # 循环迭代结束 或者 循环遍历所有alpha后 alphaPairs还没有变化
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        # 当entireSet=true or 非边界alpha对没有了 就开始寻找alpha对 然后决定是否要进行else
        if entireSet:
            # 在数据集上遍历所有可能的alpha
            for i in range(oS.m):
                # 是否存在alpha对 存在就+1
                alphaPairsChanged += innerL(i, oS)
                print('fullSet, iter:%d i:%d, pairs changed %d' % (iter, i, alphaPairsChanged))
            iter += 1

        # 对已存在 alpha对 选出非边界的alpha值 进行优化
        else:
            # 遍历所有的非边界alpha值 也就是不再边界0或者C以上的值
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < 0))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter:%d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1

        # 如果找到alpha对 就优化非边界alpha值 否则 就重新进行寻找 如果寻找一遍 遍历所有的行还没找到 就退出循环
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        print('iteration number:%d' % iter)
    return oS.b, oS.alphas

def calcWs(alphas, dataArr, classLabels):
    """
    基于alpha计算w值
    :param alphas: 拉格朗日乘子
    :param dataArr:数据集--输入空间
    :param classLabels:输出空间
    :return:
        ws  回归系数
    """
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(X)
    w = np.zeros((n, 1))

    for i in range(m):
        w += np.multiply(alphas[i] * labelMat[i], X[i, :].T)
    print('=====计算出w为======', w)
    return w

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


def smoSimple(dataMat, labelMat, C, toler, maxIter):
    """

    :param dataMat: 输入空间
    :param labelMat:  输出空间
    :param C: 常量值 惩罚参数的常量值
    :param toler: 容错率
    :param maxIter: 最大迭代次数
    :returns
        b   模型常量值
        alphas  拉格朗日参数
    """

    dataMatrix = np.mat(dataMat)
    labelMat = np.mat(labelMat).transpose()

    m, n = np.shape(dataMatrix)

    # 初始化 模型常量值 以及 alphas参数
    b = 0
    alphas = np.mat(np.zeros((m, 1)))

    # 设置迭代次数计数值
    iter = 0
    while (iter < maxIter):
        # 设置迭代是否改变的计数值
        alphasChangedCount = 0

        for i in range(m):
            # 计算ai的预测值与误差值
            fXi = float(np.multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[i, :].T)) + b
            Ei = fXi - labelMat[i]

            # 不满足ktt条件就继续优化
            if((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                # 需要进行优化 选取随机的非i点 J点
                j = selectJrand(i, m)

                # 计算aj的预测值 与 误差值
                fXj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - labelMat[j]

                # 记录 ai aj 之后会进行优化
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                # 优化 aj
                # 首先 计算二变量问题
                if (labelMat[i] != labelMat[j]):
                    #   如果是异侧 相减 ai-aj=k   那么 定义域为 [k, C + k]
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    # 同侧 相加 ai + aj=k 那么定义域为 [k-c, k]
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print('L === H')
                    print('没有优化空间')
                    continue


                # 计算eta值优化aj
                eta = 2.0*dataMatrix[j, :]*dataMatrix[i, :].T - dataMatrix[i, :]*dataMatrix[j, :].T - dataMatrix[j, :]*dataMatrix[j, :].T
                if eta >= 0:
                    print('eta >= 0')
                    continue


                alphas[j] -= labelMat[j] * (Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j], H, L)


                # 检查alphaJ 调整幅度对比 比较小就退出
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving enough")
                    continue

                # 优化完了aj 同时优化ai 保持满足优化条件
                alphas[i] += labelMat[i]*labelMat[j]*(alphaJold - alphas[j])

                # 分别计算模型的常量值
                # bi = b - Ei - yi*(ai - ai_old)*xi*xi.T - yj(aj-aj_old)*xi*xj.T
                # bj = b - Ej - yi*(ai - ai_old)*xi*xj.T - yj(aj-aj_old)*xj*xj.T
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - labelMat[
                    j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - labelMat[
                    j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                # 判断哪个模型常量值符合 定义域规则 不满足就 暂时赋予 b = (bi+bj)/2.0
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphasChangedCount += 1
                print("iter: %d i:%d, pairs changed %d" % (iter, i, alphasChangedCount))
                # 在for循环外，检查alpha值是否做了更新，如果在更新则将iter设为0后继续运行程序
                #  知道更新完毕后，iter次循环无变化，才推出循环。
            if (alphasChangedCount == 0):
                iter += 1
            else:
                iter = 0
            print("iteration number: %d" % iter)
            return b, alphas


def selectJrand(i, m):
    """

    :param i: i index
    :param m: 数据样本数
    :return
        j 非i的 alpha[j]值
    """
    while True:
        j = int(np.random.uniform(0, m))
        if j != i:
            return j




def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj


def main():
    dataArr, labelArr = loadDataSet('testSet.txt')
    # print labelArr

    # b是常量值， alphas是拉格朗日乘子
    b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
    # b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    # print('/n/n/n')
    # print('b=', b)
    # print('alphas[alphas>0]=', alphas[alphas > 0])
    # print('np.shape(alphas[alphas > 0])=', np.shape(alphas[alphas > 0]))
    for i in range(100):
        if alphas[i] > 0:
            print(dataArr[i], labelArr[i])
    # 画图
    ws = calcWs(alphas, dataArr, labelArr)
    plotfig_SVM(dataArr, labelArr, ws, b, alphas)

if __name__ == '__main__':
    main()

