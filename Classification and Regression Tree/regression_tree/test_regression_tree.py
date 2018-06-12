# !/usr/bin/env python3
# -*- utf-8 -*-
"""
__title__ = test regression tree and pruning
__author__ = LEI
__time__ = '2018/6/11'

  we are drowning in information,but starving for knowledge
"""
import numpy as np

"""建立回归树模型
    step1: load data
    step2: 二元切分法
    step3: create tree
"""


def load_data(filename):
    """
    step1 load data
    :param filename:
    :return
        data_set
    """
    fr = open(filename)
    data_set = list()
    for lines in fr.readlines():
        line = lines.strip().split('\t')
        # print('line', line)
        line_set = list()
        for i in line:
            line_set.append(float(i))
        data_set.append(line_set)
    return data_set


def bin_split_data(data_set, feature, thresh):
    """
    step2 二元切分法
    :param data_set:
    :param feature:
    :param thresh:
    :return:
    """
    data_set = np.mat(data_set)
    # print((data_set[:, feature] > thresh)[0])

    l_mat = data_set[np.nonzero(data_set[:, feature] > thresh)[0], :]
    # print('l_mat', l_mat)
    r_mat = data_set[np.nonzero(data_set[:, feature] <= thresh)[0], :]
    # print('r_mat', r_mat)
    return l_mat, r_mat


def reg_leaf(data_set):
    # data_set = np.mat(data_set)
    return np.mean(data_set[:, -1])


def reg_error(data_set):
    # data_set = np.mat(data_set)
    return np.var(data_set[:, -1]) * (np.shape(data_set)[0])


def create_tree(data_set, reg_leaf, reg_error, ops=(1, 4)):
    """
    构建树
    :param data_set: 训练数据集
    :param reg_leaf: 叶子节点计算函数
    :param reg_error: 误差函数
    :param ops: 预剪枝
    :return
        reg_tree 回归树
    """
    feature, val = choose_best_feature(data_set, reg_leaf, reg_error, ops)
    if feature is None:
        return val
    reg_tree = dict()
    reg_tree['feature'] = feature
    reg_tree['val'] = val
    l_mat, r_mat = bin_split_data(data_set, feature, val)
    reg_tree['left'] = create_tree(l_mat, reg_leaf, reg_error, ops)
    reg_tree['right'] = create_tree(r_mat, reg_leaf, reg_error, ops)
    return reg_tree


def choose_best_feature(data_set, reg_leaf, reg_error, ops):
    """
    选择最优特征及其阈值
    :param data_set:
    :param reg_leaf:
    :param reg_error:
    :param ops:
    :return:
    """
    data_set = np.mat(data_set)
    m, n = np.shape(data_set)

    limit_erorr = ops[0]
    length = ops[1]

    best_feature = None
    val = None
    min_error = np.inf

    if len(set(data_set[:, -1].T.tolist()[0])) == 1:
        return None, reg_leaf(data_set)
    var_s = np.var(data_set[:, -1]) * (np.shape(data_set)[0])
    # print('data_set[:, i].T.tolist()[0]', data_set[:, 0].T.tolist()[0])
    for feature in range(n - 1):
        # print('xxxxxxx', data_set[:, feature].T.tolist()[0])
        for thresh_val in set(data_set[:, feature].T.tolist()[0]):
            l_mat, r_mat = bin_split_data(data_set, feature, thresh_val)
            if (np.shape(l_mat)[0] < length) or (np.shape(r_mat)[0] < length):
                continue
            bin_error = reg_error(l_mat) + reg_error(r_mat)
            if bin_error < min_error:
                min_error = bin_error
                best_feature = feature
                val = thresh_val
    if (var_s - min_error) < limit_erorr:
        return None, reg_leaf(data_set)

    l_mat, r_mat = bin_split_data(data_set, best_feature, val)
    if (np.shape(l_mat)[0] < length) or (np.shape(r_mat)[0] < length):
        return None, reg_leaf(data_set)
    return best_feature, val


# 后剪枝
""" 后剪枝
    遍历所有的叶子节点
    计算拆分的数据集与未拆分的数据之间误差
    如果合并之后的数据之间误差降低 就合并
    伪代码:
        基于已有的树切分测试集
            如果存在任一子集是一棵树,则在该子集递归剪枝过程
            计算将当前的叶节点合并后的误差
            计算不合并的误差
            如果合并会降低误差的话 将叶节点合并
"""


def is_tree(obj):
    return type(obj).__name__ == 'dict'


def get_mean(tree):
    if is_tree(tree['left']):
        tree['left'] = get_mean(tree['left'])
    if is_tree(tree['right']):
        tree['right'] = get_mean(tree['right'])

    return (tree['left'] + tree['right']) / 2.0


def postpruning(tree, test_data):
    if np.shape(test_data)[0] == 0:
        return get_mean(tree)
    if is_tree(tree['left']) or is_tree(tree['right']):
        l_mat, r_mat = bin_split_data(test_data, tree['feature'], tree['val'])
    if is_tree(tree['left']):
        tree['left'] = postpruning(tree['left'], l_mat)
    if is_tree(tree['right']):
        tree['right'] = postpruning(tree['right'], r_mat)
    if not is_tree(tree['left']) and not is_tree(tree['right']):
        l_mat, r_mat = bin_split_data(test_data, tree['feature'], tree['val'])
        error_no_merge = np.sum(np.power(l_mat[:, -1] - tree['left'], 2)) + \
                         np.sum(np.power(r_mat[:, -1] - tree['right'], 2))
        tree_mean = (tree['left'] + tree['right']) / 2.0
        # print('tree_mean\n', tree_mean)
        # print('test_data\n', test_data)
        # print('index\n', (test_data[:, -1])[0])
        error_merge = np.sum(np.power(test_data[:, -1] - tree_mean, 2))
        if error_merge < error_no_merge:
            print('merge')
            return tree_mean
        else:
            return tree
    else:
        return tree


def main():
    filename = 'data3.txt'
    data_set = load_data(filename)
    # print(data_set)
    reg_tree = create_tree(data_set, reg_leaf, reg_error, ops=(0, 1))
    print('reg_tree:\n', reg_tree)
    # bin_split_data(data_set, 0, 0.5)
    test_file = 'data3test.txt'
    test_set = load_data(test_file)
    after_prune = postpruning(tree=reg_tree, test_data=test_set)
    print('after prune:\n', after_prune)


if __name__ == '__main__':
    main()
