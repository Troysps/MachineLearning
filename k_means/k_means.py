# !/usr/bin/env python3
# -*- utf-8 -*-
"""
__title__ = K means
__author__ = LEI
__time__ = '2018/6/12'

  we are drowning in information,but starving for knowledge
"""
import numpy as np


def load_data(filename):
    fr = open(filename)
    data_set = list()
    for lines in fr.readlines():
        line = lines.strip().split('\t')
        line_list = list()
        for i in line:
            line_list.append(float(i))
        data_set.append(line_list)
    data_set = np.mat(data_set)
    return data_set


def calc_dist_sse(data_set, centroids):
    """
    k-means 距离度量方式  sum of square error
    :param data_set:
    :param centroids:
    :return:
    """
    # print('data set', type(data_set))
    # print('centriods', type(centroids))
    return np.sqrt(np.sum(np.power(data_set - centroids, 2)))


def create_centroids(data_set, k):
    """
    k means 算法 随机选取质心
    :param data_set: 数据集
    :param k: k个质心
    :return
        centroids
    """
    m, n = np.shape(data_set)
    # print('shape data set', np.shape(data_set))
    centroids = np.mat(np.zeros((k, n)))
    # print('centroids', np.shape(centroids))
    for row in range(k):
        for index in range(n):
            range_min = np.min(data_set[:, index])
            range_of = np.max(data_set[:, index]) - range_min
            centroids[row, index] = range_min + np.random.uniform(range_of)
    # print('centroids\n', centroids)
    return centroids


def k_means(data_set, k, calc_dist, create_centroids=create_centroids):
    """
    k means 算法主体
    计算质心——分配——重新计算
    实现重点:
        part1: 通过对每个点遍历所有质心并计算点到每个置信的距离
        part2: 遍历所有质心并更新它们的取值
    :param data_set:数据集
    :param k:k个簇
    :param calc_dist: 指向计算距离的函数
    :param create_centroids: 指向创建质心的函数
    :return:
        centroids: 最终确定的质心
        cluster_ment: 聚类的情况(class, dist)  分配到哪个质心附近 以及dist距离
    """
    m, n = np.shape(data_set)
    cluster_ment = np.mat(np.zeros((m, 2)))
    centroids = create_centroids(data_set, k)
    print('init centriods', centroids)
    cluster_changed = True
    count = 0
    while cluster_changed:
        count += 1
        cluster_changed = False
        # step1 通过对每个点遍历所有质心并计算每个点到每个质心的距离
        for i in range(m):  # 循环每一个数据点并分配到最近的质心中去
            # print(np.inf, -1)
            min_dist = np.inf
            min_index = -1
            for j in range(k):
                dist_ji = calc_dist(data_set[i, :], centroids[j, :])     # 计算数据点到质心的最小距离
                if dist_ji < min_dist:      # 如果距离比最小距离还小, 就更新最小距离和最小质心的index
                    min_dist = dist_ji
                    min_index = j
            if cluster_ment[i, 0] != min_index:     # 簇分配结果改变
                cluster_changed = True
                cluster_ment[i, :] = min_index, min_dist**2     # 更新簇分配结果为最小质心的索引,最小距离
        # step2 遍历所有质心并更新它们的取值
        for cent in range(k):
            pts_in_cluster = data_set[np.nonzero(cluster_ment[:, 0].A == cent)[0]]   # 获取该簇中的所有点
            centroids[cent, :] = np.mean(pts_in_cluster, axis=0)        # 将质心修改为簇中所有点的平均值
            print('centriods changed', centroids)
    print('count', count)
    return centroids, cluster_ment


def biKMeans(data_set, k, calc_dist=calc_dist_sse):
    m = np.shape(data_set)[0]
    clusterAssment = np.mat(np.zeros((m,2)))  # 保存每个数据点的簇分配结果和平方误差
    centroid0 = np.mean(data_set, axis=0).tolist()[0]  # 质心初始化为所有数据点的均值
    centList =[centroid0]  # 初始化只有 1 个质心的 list
    for j in range(m):  # 计算所有数据点到初始质心的距离平方误差
        clusterAssment[j,1] = calc_dist(np.mat(centroid0), data_set[j,:])**2
    while (len(centList) < k):  # 当质心数量小于 k 时
        lowestSSE = np.inf
        for i in range(len(centList)):  # 对每一个质心
            ptsInCurrCluster = data_set[np.nonzero(clusterAssment[:,0].A==i)[0],:] # 获取当前簇 i 下的所有数据点
            centroidMat, splitClustAss = k_means(ptsInCurrCluster, 2, calc_dist) # 将当前簇 i 进行二分 kMeans 处理
            sseSplit = sum(splitClustAss[:,1]) # 将二分 kMeans 结果中的平方和的距离进行求和
            sseNotSplit = sum(clusterAssment[np.nonzero(clusterAssment[:,0].A!=i)[0],1]) # 将未参与二分 kMeans 分配结果中的平方和的距离进行求和
            print("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE: # 总的（未拆分和已拆分）误差和越小，越相似，效果越优化，划分的结果更好（注意：这里的理解很重要，不明白的地方可以和我们一起讨论）
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        # 找出最好的簇分配结果
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) # 调用二分 kMeans 的结果，默认簇是 0,1. 当然也可以改成其它的数字
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit # 更新为最佳质心
        print('the bestCentToSplit is: ',bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        # 更新质心列表
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0] # 更新原质心 list 中的第 i 个质心为使用二分 kMeans 后 bestNewCents 的第一个质心
        centList.append(bestNewCents[1,:].tolist()[0]) # 添加 bestNewCents 的第二个质心
        clusterAssment[np.nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss # 重新分配最好簇下的数据（质心）以及SSE
    return np.mat(centList), clusterAssment


def main():
    filename = 'testSet.txt'
    data_set = load_data(filename)
    print('mean data set ', np.mean(data_set, axis=0).tolist()[0])
    # print('*****data set*****\n', data_set)
    # create_centroids(data_set, k=3)
    #
    # centriods, cluster_ment = k_means(data_set, k=3, calc_dist=calc_dist_sse, create_centroids=create_centroids)
    # print('result centriods,', centriods)
    # print("result cluster_ment", cluster_ment)
    centroids, cluster_ment = biKMeans(data_set, 3, calc_dist=calc_dist_sse)
    print('centroids', centroids)
    print('cluster ment', cluster_ment)

if __name__ == '__main__':
    main()
