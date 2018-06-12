# !/usr/bin/env python3
# -*- utf-8 -*-
"""
__title__ = regression tree
__author__ = LEI
__time__ = '2018/6/11'

  we are drowning in information,but starving for knowledge
"""
import numpy as np


# load data 数据读取
def load_data(filename):
    """
    :param filename:文件名
    :return
        data_set
    """
    fr = open(filename)
    data_set = list()
    for lines in fr.readlines():

        data = lines.strip().split('\t')
        line_data = list()
        for i in data:
            line_data.append(float(i))
        data_set.append(line_data)
    return data_set


# 二元切分法
def bin_split_data(data_set, feature, threshold):
    """
    :param data_set: 训练数据集
    :param feature: 特征   index
    :param threshold: 阈值
    :return
        l_mat  大于阈值
        r_mat   小于阈值
    """
    data_set = np.mat(data_set)
    # print('data_set, feature', data_set[:, feature])
    # print('ssss', data_set[:, feature] > threshold)
    l_mat = data_set[np.nonzero(data_set[:, feature] > threshold)[0], :]
    r_mat = data_set[np.nonzero(data_set[:, feature] <= threshold)[0], :]
    return l_mat, r_mat


def reg_leaf(data_set):
    return np.mean(data_set[:, -1])


def reg_err(data_set):
    # data_set = np.mat(data_set)
    return np.var(data_set[:, -1]) * np.shape(data_set)[0]


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


# 构建树
def create_tree(data_set, leaf_type=reg_leaf, err_type=reg_err, ops=(1, 4)):
    """

    :param data_set: 训练数据集
    :param leaf_type: 建立叶节点的函数
    :param err_type: 代表计算误差
    :param ops: 包含数构建所需其他参数的元组(误差,切分数据集数量)
            当误差很小时, 可以停止切分
            当切分出来的数据集数量已经很小的时候, 停止切分
    :return
        ret_tree
    """
    feature, val = choose_best_feature(data_set, leaf_type, err_type, ops)
    print('feature %s val %s' % (feature, val))
    if feature is None:
        return val
    ret_tree = dict()
    ret_tree['fea_index'] = feature
    ret_tree['val'] = val
    l_mat, r_mat = bin_split_data(data_set, feature, val)
    ret_tree['left'] = create_tree(l_mat, leaf_type, err_type, ops)
    ret_tree['right'] = create_tree(r_mat, leaf_type, err_type, ops)
    # print('ret_tree', ret_tree)
    return ret_tree


# def reg_leaf(data_set):
#     return np.mean(data_set[:, -1])
#
#
# def reg_err(data_set):
#     return np.var(data_set[:, -1]) * (np.shape(data_set)[0])


def choose_best_feature(data_set, leaf_type, err_type, ops=(1, 4)):
    """
    伪代码流程:

    对每个特征
        对每个特征值
            将数据集切分为两份
            计算切分的误差
            如果当前误差小于当前的最小误差,那么将当前切分设定为最佳切分并更新最小误差

    切分前条件:
        所有Y值(data_set[:,-1])    相等的情况下则退出
    切分后条件:
        切分处理的数据集太小就跳出

        当误差很小的时候可以停止切分 返回阈值
        当切分出来的数据集很小就停止切分 返回阈值

    :param data_set: 训练数据集
    :param leaf_type: 叶子节点 -- 切分出来的数据集对应的data_set[:, -1]均值
    :param err_type: 计算切分误差 每次误差最小的同一分布
    :param ops: 包含数构建所需其他参数的元组(误差,切分数据集数量)
            当误差很小时, 可以停止切分
            当切分出来的数据集数量已经很小的时候, 停止切分
    :return
        feature  val
    """
    data_set = np.mat(data_set)
    error = ops[0]  # 误差计算
    length = ops[1]  # 切分数据集样本数
    m, n = np.shape(data_set)
    min_error = np.inf  # 最小误差
    best_feature = None  # 最优特征
    val = None  # 最优特征对应的阈值
    var_s = err_type(data_set)
    print('********\n', data_set[:, -1].T.tolist()[0])
    if len(set(data_set[:, -1].T.tolist()[0])) == 1:
        return None, leaf_type(data_set)
    for feature in range(n - 1):
        # print('xxxxxxx', data_set[:, feature].T.tolist()[0])
        for thresh_val in set(data_set[:, feature].T.tolist()[0]):
            l_mat, r_mat = bin_split_data(data_set, feature, thresh_val)
            if (np.shape(l_mat)[0] < length) or (np.shape(r_mat)[0] < length):
                continue
            bin_error = err_type(l_mat) + err_type(r_mat)
            if bin_error < min_error:
                min_error = bin_error
                best_feature = feature
                val = thresh_val
    if (var_s - min_error) < error:
        return None, leaf_type(data_set)

    l_mat, r_mat = bin_split_data(data_set, best_feature, val)
    if (np.shape(l_mat)[0] < length) or (np.shape(r_mat)[0] < length):
        return None, leaf_type(data_set)

    return best_feature, val


def is_tree(obj):
    """
    判断是否为'树' 树以字典形式保存
    :param obj:
    :return
        bool
    """
    return type(obj).__name__ == 'dict'


def get_mean(tree):
    """
    递归直到叶节点为止  如果找到两个叶节点 则计算他们的平均值
    :param obj:
    :return:
    """
    if is_tree(tree['left']):
        tree['left'] = get_mean(tree['left'])
    if is_tree(tree['right']):
        tree['right'] = get_mean(tree['right'])

    return (tree['left'] + tree['right']) / 2


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
            return predict_val(tree['left'], input_data, model_val)
        else:
            return model_val(tree['left'], input_data)
    else:
        if is_tree([tree['right']]):
            print('is tree: tree["right"]', tree['right'])
            return predict_val(tree['right']['right'], input_data, model_val)
        else:
            print('not is tree: tree["right"]', tree['right'])
            return model_val(tree['right']['right'], input_data)


def main():
    file_name = r'data3.txt'
    data_set = load_data(file_name)
    print(data_set)
    l_mat, r_mat = bin_split_data(data_set, 0, 0.5)
    print('l_mat\n', l_mat)
    print('r_mat\n', r_mat)
    reg_tree = create_tree(data_set, reg_leaf, reg_err, ops=(0, 1))
    print('reg_tree', reg_tree)
    test_file = r'data3test.txt'
    test_data = load_data(test_file)
    prune_tree = prune(reg_tree, test_data)
    print('prune_tree', prune_tree)
    estimate = predict_val(prune_tree, 0.993349, model_tree_estimate_val)


if __name__ == '__main__':
    main()
