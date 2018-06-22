# !/usr/bin/env python3
# -*- utf-8 -*-
"""
__title__ = 餐馆菜肴推荐系统
__author__ = LEI
__time__ = '2018/6/21'

  we are drowning in information,but starving for knowledge
"""

import numpy as np

"""构建一个基本的推荐引擎
    目标:
        1.寻找用于没有尝过的菜肴
        2.然后通过svd来减少特征空间并提高推荐的效果
        3.程序打包并通过用户可读的人机界面提供给人机界面提供给人们使用
    系统工作流程:
        给定一个用户, 系统会为此用户返回N个最好的推荐菜
        将此步骤细分:
            (1) 寻找用户没有评级的菜肴  即在用户 —— 物品 矩阵中的0值
            (2) 在用户没有评级的菜肴中, 对每个物品预计一个可能的评级分数 这就是说 我们认为用户可能会对物品的打分
            (3) 对这些物品的评分从高到低进行排序  返回前N个物品
            
"""


def load_data1():
    """
    用户row 物品column
    :return:
    """
    return np.mat([[4, 4, 0, 2, 2],
                   [4, 0, 0, 3, 3],
                   [4, 0, 0, 1, 1],
                   [1, 1, 1, 2, 0],
                   [2, 2, 2, 0, 0],
                   [1, 1, 1, 0, 0],
                   [5, 5, 5, 0, 0]])


# 核心思想 对用户未评分的物品 通过计算相似度 来补填数据

def load_Data2():
    # 利用SVD提高推荐效果，菜肴矩阵
    return[[2, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0],
           [5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
           [4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5],
           [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
           [0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0],
           [1, 1, 2, 1, 1, 2, 1, 0, 4, 5, 0]]

# step1 计算相似度的几种方式 -- 欧式距离 相关系数--不在乎评分的量级  余弦相似度
def eur_sim(input_a, input_b):
    """
    欧式距离 相似度度量
    :param input_a:
    :param input_b:
    :return:
    """
    # np.divide(1, 1+1+np.linalg.norm(input_a - input_b))
    # print(np.linalg.norm(input_a - input_b, 2))
    return np.divide(1, 1+np.linalg.norm(input_a - input_b, 2))


def pear_sim(input_a, input_b):
    """
    相关系数 相似度度量
    :param input_a:
    :param input_b:
    :return:
    """
    return 0.5 + 0.5 * np.corrcoef(input_a, input_b, rowvar=0)[0][1]


def cosine_sim(input_a, input_b):
    """
    余弦相似度 相似度度量
    :param input_a:
    :param input_b:
    :return:
    """
    up = input_a.T * input_b
    down = np.linalg.norm(input_a) * np.linalg.norm(input_b)
    return 0.5 + 0.5 * (up / down)


# step2 已知矩阵中 行为用户 列为食品  分解步骤 找到用户没有评级的菜肴
def non_review_list(data_set, user):
    """
    找到用户没有评级的菜肴
    :param data_set: 用户-食物数据集  row: 用户  column：食物
    :param user: 用户
    :return:
        non_review_list 用户没有评级的菜肴索引集合 list[column1, ..., columnN]
        nonzero_list    评级的菜肴索引集合 list[column1, ..., columnN]
    """
    n = np.shape(data_set)[1]
    # print('n', n)
    _, nonzero_list = np.nonzero(data_set[user, :])
    non_review_column = [i for i in list(range(n)) if i not in nonzero_list]
    print('nonzero_list', nonzero_list, len(nonzero_list))
    print('non_review_column', non_review_column)
    return non_review_column, nonzero_list


# step3 已知用户评级和未评级的食物索引集合 那么遍历计算相似度
def compare_others(data_set, user):
    """
    遍历计算用户相似度(基于用户相似度) --- 采用欧式距离 从大到小排序 越大的相似度越高
    :param data_set: 用户-食物数据集
    :param user: 用户
    :return:
        relative_count dict{user1:similar_value1, ..., userN: similar_valueN}
    """
    non_review_column, nonzero_list = non_review_list(data_set, user)
    data_copy = np.delete(data_set, user, axis=0)   # 删除user对应的数据行
    print('data_copy', data_copy)
    m, n = np.shape(data_copy)
    relative_count = dict()
    for i in range(m):
        # print('xxxx', data_copy[i, :][:, nonzero_list]) relative_count[frozenset(data_copy[i, :])]
        # if (data_copy[i, :][:, non_review_column]).all():
        dist_similar = eur_sim(data_copy[i, :][:, nonzero_list], data_set[user, :][:, nonzero_list])
        print('dist_similar', dist_similar)
        # print(type(lis), frozenset(lis))
        relative_count[i] = dist_similar

    print('基于用户相似度', relative_count)    # 这里看出基于用户相似度
    return relative_count


"""思考:
    1.选择关于欧式距离的相似度计算是否合理
    2.衡量两个用户之间的相似度 是不是选取两方都进行评分的物品进行比较比较好
    3.计算相似度之后该如何进行评分估计  怎么度量  为什么
"""


"""基于物品相似度的推荐引擎伪代码
    part1-推荐引擎
        step1: 寻找用户未评级的物品
        step2: 如果不存在, 就直接退出; 存在就在未评分物品集合上变量 给出预估评级
    part2-预估评级
        step1: 对每个物品进行遍历 如果此用户该物品的值为0 跳出函数
        step2: 基于物品--列 进行相似度比较 存在相似元素 就使用余弦相似度计算其相似度
        step3: 相似度会不断累加 每次计算是还考虑相似度和当前用户评分的乘积
        step4：返回估计值

"""


# 《机器学习实战》一书上 选择了基于物品相似度的推荐引擎
def stand_est(data_mat, user, sim_meas, item):
    """
    stand est 计算某用户未评分物品中, 以该物品和其他物品评分的用户的物品相似度  然后进行综合评分
    :param data_mat:    数据集
    :param user:        用户编号
    :param sim_meas:    相似度计算方法
    :param item:        未评分的物品编号
    :return:
        rat_simtotal / sim_total  评分(0~5)之间
    """
    # 得到数据集中的物品数目
    n = np.shape(data_mat)[1]
    # 初始化两个评分
    sim_total = 0.0
    rat_simtotal = 0.0
    # 遍历行中的每个物品(对用户评过分的物品进行遍历, 并将它与其他物品进行比较)
    for j in range(n):
        print(user, j, data_mat[user, j])
        user_rating = data_mat[user, j]
        # 如果某个物品的评分值为0 则跳过这个物品
        if user_rating == 0:
            continue
        # 寻找两个用户都评级的物品
        # 变量overlap给出的是两个物品当中已经被评分的那个元素的索引ID
        # logical_and 计算x1和x2元素的真值
        overlap = np.nonzero(np.logical_and(data_mat[:, item].A > 0, data_mat[:, j].A > 0))[0]
        print('overlap', overlap)
        # 如果相似度为0 则两者没有任何重合元素,终止本次循环
        if len(overlap) == 0:
            similarity = 0
        # 如果存在重合的物品 则基于这些重合物重新计算相似度
        else:

            similarity = sim_meas(data_mat[overlap, item], data_mat[overlap, j])
        # 相似度会不断累加 每次计算是还考虑相似度和当前用户评分的乘积
        # similarity 用户相似度  user_rating 用户评分
        print('相似度', similarity)
        sim_total += similarity
        print('累计相似度', sim_total)
        rat_simtotal += similarity * user_rating
        print('用户评分乘积', user_rating)
    if sim_total == 0:
        return 0
    else:
        return rat_simtotal / sim_total

def analyse_data(sigma, loop=20):
    """
    分析sigma的长度取值
    :param sigma:  sigma的值
    :param loop:   循环次数
    :return:

    """
    # 总方差的集合
    sig_square_list = sigma**2
    sig_sum = sum(sig_square_list)
    for i in range(loop):
        sigma_calc = sum(sig_square_list[:i+1])
    # 根据实际业务 进行处理 设置对应的sigma次数
        print('主成分:%s, 方差占比:%s%%' % (format(i+1, '2.0f'), format(sigma_calc/sig_sum*100, '4.2f')))


# 基于svd的评分估计
def svd_est(data_mat, user, sim_meas, item):
    """
    基于svd的评分估计
    :param data_mat:  训练数据集
    :param user:      用户编号
    :param sim_meas:  相似度计算方法
    :param item:      未评分的物品编号
    :return:
        rat_sim_total/sim_total   评分(0~5之间的值)
    """
    # 物品数目
    n = np.shape(data_mat)[1]
    # 对数据集进行SVD分解
    sim_total = 0.0
    rate_sim_total = 0.0
    # 奇异值分解
    # 在SVD分解之后, 我们只利用包含90%能量的奇异值
    u, sigma, vt = np.linalg.svd(data_mat)
    # 分析sigma的长度
    # analyse_data(sigma, 20)

    # 构建奇异值 对角矩阵
    sig_sigma = np.mat(np.eye(4) * sigma[:4])

    # 利用u矩阵将物品转换到低维度空间中, 构建转换后的物品(物品+4个主要的特征)
    xformed_items = data_mat.T * u[:, :4] * sig_sigma.I
    print('data_mat', np.shape(data_mat))
    print('u[:, :4]', np.shape(u[:, :4]))
    print('sig_sigma.I', np.shape(sig_sigma.I))
    print('vt[:4, :]', np.shape(vt[:4, :]))
    print('xformed_items', np.shape(xformed_items))

    # 在低维空间进行相似度计算
    for j in range(n):
        user_rating = data_mat[user, j]
        if user_rating == 0 or j == item:
            continue
        # 相似度计算
        similarity = sim_meas(xformed_items[item, :].T, xformed_items[j, :].T)
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        # 相似度累加求和
        sim_total += similarity
        # 相似度及对应的评分值的乘积和
        rate_sim_total += similarity * user_rating
    if sim_total == 0:
        return 0
    else:
        # 计算估计评分
        return rate_sim_total / sim_total


# recommend()函数, 就是推荐引擎, 默认调用stand_est()函数 产生了最高的N个推荐结果
# 如果不指定N的大小, 则默认值为3. 该函数另外的参数还包括相似度计算和估计方法
def recommend(data_mat, user, N=3, sim_meas=cosine_sim, est_method=svd_est):
    """
    推荐引擎
    :param data_mat:训练数据集
    :param user: 用户编号
    :param N:
    :param sim_meas: 相似度计算方法 --- 这里选用了余弦相似度
    :param est_method: 使用的推荐算法
    :return:
        返回最终的N个推荐结果
    """
    # 寻找未评级的物品
    # 对给定的用户建立一个未评分的物品列表
    unrated_items = np.nonzero(data_mat[user, :].A == 0)[1]
    print('未评分列表', unrated_items)
    # 如果不存在未评分物品, 那么就退出函数
    if len(unrated_items) == 0:
        return 'user rated everything'
    # 物品的编号和评分值
    item_scores = list()
    # 在未评分物品上进行循环
    for item in unrated_items:
        # 获取item该物品的评分
        estimated_score = est_method(data_mat, user, sim_meas, item)
        item_scores.append((item, estimated_score))
    # 按照评分得分  进行逆排序 获取前N个未评级物品进行推荐
    print(item_scores)
    return sorted(item_scores, key=lambda jj: jj[1], reverse=True)[: N]


def main():
    data_set = load_data1()
    print(data_set)
    # # non_review_list(data_set, 2)
    # compare_others(data_set, 2)
    result_recommend = recommend(data_set, 2)
    print('最终推荐结果:', result_recommend)


if __name__ == '__main__':
    main()


"""总结: 
    考虑到用户人数远远大于物品数   --- 应当使用基于物品的相似度计算
    推荐引擎系统的不足与可优化:
        1. svd 分解可以单独抽离
        2. 面临大量数据时, 是否可以只存储非零元素来节省内存和计算
        3. 相似计算应该抽离 离线计算并保存相似度
        4. 如何在缺乏数据时给出好的推荐
            冷启动问题?---把推荐问题看做搜索问题 基于内容的推荐
"""
