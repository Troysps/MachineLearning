# !/usr/bin/env python3
# -*- utf-8 -*-
"""
__title__ = Model Tree
__author__ = LEI
__time__ = '2018/6/11'

  we are drowning in information,but starving for knowledge
"""
import numpy as np

"""如何判断模型树 回归树 以及线性回归机器衍生算法的优劣
    :一般来说使用相关系数进行判断 np.corrcoef()
"""
"""模型树与回归树
    实质上:
        叶节点函数规则不同
        误差函数规则不同
        
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
    二元切分法
    :param data:    训练数据集
    :param feature: 特征
    :param thresh:  阈值
    :return
        l_mat   r_mat
    """
    data_set = np.mat(data_set)
    l_mat = data_set[np.nonzero(data_set[:, feature] > thresh)[0], :]
    r_mat = data_set[np.nonzero(data_set[:, feature] <= thresh)[0], :]
    return l_mat, r_mat


def linear_model(data_set):
    """
    线性模型构建
    :param data_set: 数据集
    :return:
        ws, x , y  模型参数 输入空间 输出空间
    """
    data_set = np.mat(data_set)
    m, n = np.shape(data_set)
    x = np.mat(np.ones((m, n)))
    # print('x\n', x)
    x[:, 1:n] = data_set[:, 0:n - 1]
    y = np.mat(data_set[:, -1])
    # print('x'*10, x)
    # print('y'*10, y)
    x_tx = x.T * x
    if np.linalg.det(x_tx) == 0:
        raise NameError('det == 0, try increase second ops')
    ws = x_tx.I * (x.T * y)
    return ws, x, y


def linear_leaf(data_set):
    ws, x, y = linear_model(data_set)
    return ws


def linear_err(data_set):
    ws, x, y = linear_model(data_set)
    y_predict = x * ws
    # print('y_predict', y_predict)
    # print('y', y)
    return np.sum(np.power(y - y_predict, 2))


def choose_best_featrue(data_set, type_leaf, type_err, ops):
    """
    选择最优的特征 以及阈值
    :param data_set:    训练数据集
    :param type_leaf:   叶节点
    :param type_err:    误差计算
    :param ops:         预剪枝
    :return
        feature val
    """
    data_set = np.mat(data_set)
    m, n = np.shape(data_set)
    min_error = np.inf
    limit_error = ops[0]
    limit_branch = ops[1]
    best_feature = None
    val = None
    sum_error = type_err(data_set)
    if len(set(data_set[:, -1].T.tolist()[0])) == 1:
        return None, type_leaf(data_set)
    for feature_index in range(n - 1):
        for thresh_val in set(data_set[:, -1].T.tolist()[0]):
            l_mat, r_mat = bin_split_data(data_set, feature_index, thresh_val)
            if (np.shape(l_mat)[0] < limit_branch) or (np.shape(r_mat)[0] < limit_branch):
                continue
            error = type_err(l_mat) + type_err(r_mat)

            if error < min_error:
                min_error = error
                best_feature = feature_index
                val = thresh_val
    if (sum_error - min_error) < limit_error:
        return None, type_leaf(data_set)
    l_mat, r_mat = bin_split_data(data_set, best_feature, val)
    if (np.shape(l_mat)[0] < limit_branch) or (np.shape(r_mat)[0] < limit_branch):
        return None, type_leaf(data_set)
    return best_feature, val


def create_model_tree(data_set, type_leaf, type_err, ops=(1, 4)):
    """
    构建树
    :param data_set:
    :param type_leaf:
    :param ops:
    :return:
    """
    feature, val = choose_best_featrue(data_set, type_leaf, type_err, ops)
    if feature is None:
        return val
    model_tree = dict()
    l_mat, r_mat = bin_split_data(data_set, feature, val)
    model_tree['feature'] = feature
    model_tree['val'] = val
    model_tree['left'] = create_model_tree(l_mat, type_leaf, type_err, ops)
    model_tree['right'] = create_model_tree(r_mat, type_leaf, type_err, ops)
    return model_tree


# 后剪枝
def is_tree(obj):
    return type(obj).__name__ == 'dict'


def get_mean(tree):
    if is_tree(tree['left']):
        tree['left'] = get_mean(tree['left'])
    if is_tree(tree['right']):
        tree['right'] = get_mean(tree['right'])
    return (tree['left'] + tree['right']) / 2.0


def prune(tree, test_data):
    """
    后剪枝
    :param test_data: 测试数据集
    :param tree: 待剪枝的树
    :return
        prune_tree
    """
    # step1: 检查训练数据是否存在 如果不存在就做塌陷处理
    if np.shape(test_data)[0] == 0:
        return get_mean(tree)
    # step2: 如果存在任意子集是一棵树 则在该子集上递归剪枝过程
    if (is_tree(tree['left'])) or (is_tree(tree['right'])):
        l_mat, r_mat = bin_split_data(test_data, tree['fea_index'], tree['val'])
    if is_tree(tree['left']):
        tree['left'] = prune(tree['left'], l_mat)
    if is_tree(tree['right']):
        tree['right'] = prune(tree['right'], r_mat)
    # step3: 递归到叶节点 计算当前两个叶节点合并前后的误差
    if not is_tree(tree['left']) and not is_tree(tree['right']):
        l_mat, r_mat = bin_split_data(test_data, tree['fea_index'], tree['val'])
        error_no_merge = np.sum(np.power(l_mat[:, -1] - tree['left'], 2)) + \
                         np.sum(np.power(r_mat[:, -1] - tree['right'], 2))
        tree_mean = (tree['left'] + tree['right']) / 2.0
        error_merge = np.sum(np.power(test_data[:, -1] - tree_mean, 2))
        if error_merge < error_no_merge:
            print('merge')
            return tree_mean
        else:
            return tree
    else:
        return tree


def regression_tree_estimate_val(tree_model, input_data):
    return float(tree_model)


def model_tree_estimate_val(tree_model, input_data):
    """
    模型树 预测数据
    :param tree_model:
    :param input_data:
    :return:
    """
    n = np.shape(input_data)[0]
    x = np.mat(np.ones((1, n + 1)))
    x[:, 1:n + 1] = input_data
    print('x: %s , tree model: %s' % (x, tree_model))
    return float(x * tree_model)


def predict_val(tree, input_data, model_val):
    """
    使用树模型预测数据
    :param tree: 训练好的模型树
    :param input_data:input 需要预测的数据数据
    :param model_val:针对不同的构建树 采用不同的算法输出预测数据
    :return:
    """
    input_data = np.mat(input_data)
    if not is_tree(tree):
        print('tree********************', tree)
        return model_val(tree, input_data)
    print(tree['feature'])
    if input_data[0, tree['feature']] > tree['val']:
        if is_tree([tree['left']]):
            print('is tree: tree["left"]', tree['left'])

            return predict_val(tree['left'], input_data, model_val)
        else:
            print('not is tree: tree["left"]', tree['left'])

            return model_val(tree['left'], input_data)
    else:
        if is_tree([tree['right']]):
            print('is tree: tree["right"]', tree['right'])
            return predict_val(tree['right']['right'], input_data, model_val)
        else:
            print('not is tree: tree["right"]', tree['right'])
            return model_val(tree['right']['right'], input_data)


def main():
    filename = 'data1.txt'
    data_set = load_data(filename)
    linear_model(data_set)
    model_tree = create_model_tree(data_set, type_leaf=linear_leaf, type_err=linear_err, ops=(1, 4))
    print('model tree:', model_tree)
    estimate = predict_val(model_tree, 0.530897, model_tree_estimate_val)
    print('estimate', estimate)


if __name__ == '__main__':
    main()
