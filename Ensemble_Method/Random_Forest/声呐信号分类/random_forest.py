# !/usr/bin/env python3
# -*- utf-8 -*-
"""
__title__ = Sonar classify base on Random Forest
__author__ = LEI
__mtime__ = '2018/5/30'

  we are drowning in information,but starving for knowledge
"""

import numpy as np


def loadDataSet(filename):
    """
    读取文件数据
    :param filename:
    :return
        dataSet
        labelMat
    """
    dataset = []
    with open(filename, 'r') as fr:
        for line in fr.readlines():
            if not line:
                continue
            lineArr = []
            for feature in line.split(','):
                # strip()返回移除字符串头尾指定的字符生成的新字符串
                str_f = feature.strip()
                if str_f.isdigit():  # 判断是否是数字
                    # 将数据集的第column列转换成float形式
                    lineArr.append(float(str_f))
                else:
                    # 添加分类标签
                    lineArr.append(str_f)
            dataset.append(lineArr)
    return dataset

def cross_validation_split(dataSet, n_folds):
    """样本数据随机化
    对数据集进行重抽样 n_folds份 数据可以重复抽取--用于交叉验证
    :param dataSet:原始数据集
    :param n_fold:拆分为n_folds份数据集
    :return
        dataSet_split: 拆分数据集
    """
    dataSet_split = []
    dataSet_copy = dataSet.copy()

    dataSet_Num = len(dataSet_copy)
    fold_size = float(dataSet_Num) / n_folds

    for i in range(n_folds):
        fold = []
        while len(fold) < fold_size:
            index = np.random.randint(dataSet_Num)
            fold.append(dataSet_copy[index])
        dataSet_split.append(fold)
    return dataSet_split

def subsample(dataSet, ratio):
    """训练数据随机化

    :param dataSet: 训练数据集
    :param ratio: 训练数据集的样本比例
    :return
        sample: 随机抽样的训练样本
    """
    sample = list()
    sample_num = int(len(dataSet) * ratio)
    while len(sample) < sample_num:
        index = np.random.randint(len(dataSet))
        sample.append(dataSet[index])
    return sample

def get_split(dataSet, n_features):
    """特征随机化
    找出分隔数据集的最优特征 得到最优的特征index 特征值 row[index] 以及分隔完的数据 groups(left, right)
    :param dataSet:  原始数据集
    :param n_features: 选取特征的个数
    :return
        b_index: 最优特征的index
        b_value: 最优特征的值
        b_score: 最优特征的gini指数
        b_groups: 选取最优特征后的分隔完的数据
    """
    class_value = list(set(row[-1] for row in dataSet))  # class_value[0,1]
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    features = list()
    # 在features中 随机 添加n_features个特征
    # print(len(dataSet[0]))
    while len(features) < n_features:
        index = np.random.randint(len(dataSet[0])-1)
        if index not in features:
            features.append(index)
    # print('features', features)
    for index in features:

        # print(index)
        for row in dataSet:
            # 遍历每一行index索引下的特征值作为分类值value 找出最优的分类特征
            groups = test_split(index, row[index], dataSet)
            gini = gini_index(groups, class_value)
            # 左右两边的数量越一样 说明数据区分度不高 gini系数越大
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups  # 最后得到最优的分类特征 b_index,分类特征值 b_value,分类结果 b_groups。b_value 为分错的代价成本
    return {'index': b_index, 'value': b_value, 'groups': b_groups}

def test_split(index, value, dataSet):
    """根据特征和特征值分割数据集
    :param index: 特征索引---index
    :param value: 特征值---row[index]
    :param dataSet: 数据集---dataSet
    :return
        right 小于特征值的列表
        left 大于特征值的列表
    """
    right, left = list(), list()
    for row in dataSet:
        # print('value', value)
        # print('row[index]', row[index])
        if float(row[index]) < float(value):
            left.append(row)
        else:
            right.append(row)
    return left, right

def gini_index(groups, class_values):
    """

    :param groups:
    :param class_values:
    :return:
    """
    gini = 0.0
    D = len(groups[0]) + len(groups[1])
    for class_value in class_values:
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            proportion = [row[-1] for row in group].count(class_value) / float(size)
            gini += float(size)/D * (proportion * (1.0 - proportion))   # 计算代价 分类越准确 则gini越小
    return gini


def to_terminal(group):
    """输出group中出现次数较多的标签
        该函数参考决策树停止的两个条件
    :param group:
    :return
        group中出现次数较多的标签
    """

    outcomes = [row[-1] for row in group]
    # 输出group中出现次数较多的标签
    # max()函数中,当key参数不为空时 就以key的函数对象为判断的标准
    return max(set(outcomes), key=outcomes.count)

# create child splits for a node or make terminal
def split(node, max_depth, min_size, n_features, depth):
    """
    创建子分割器 递归分类 直到分类结束
    :param node:        节点
    :param max_depth:   最大深度
    :param min_size:    最小
    :param n_features:  特征量
    :param depth:       深度
    :return:
    """
    # print('node', node)
    left, right = node['groups']
    del (node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:  # max_depth=10 表示递归十次，若分类还未结束，则选取数据中分类标签较多的作为结果，使分类提前结束，防止过拟合
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left,
                                 n_features)  # node['left']是一个字典，形式为{'index':b_index, 'value':b_value, 'groups':b_groups}，所以node是一个多层字典
        split(node['left'], max_depth, min_size, n_features, depth + 1)  # 递归，depth+1计算递归层数
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth + 1)


def bulid_tree(train, max_depth, min_size, n_features):
    """
    创建一个决策树
    :param train:       训练数据集
    :param max_depth:   决策树深度不能太深 不然容易导致过拟合
    :param min_size:    叶子节点的大小
    :param n_features:  选择的特征的个数
    :return
        root    返回决策树
    """
    root = get_split(train, n_features)

    split(root, max_depth, min_size, n_features, 1)
    return root

def predict(node, row):
    """
    预测模型分类结果
    :param node:
    :param row:
    :return:
    """
    if float(row[node['index']]) < float(node['value']):
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

def bagging_predict(trees, row):
    """
    bagging 预测
    :param trees: 决策树集合
    :param row: 测试数据集的每一行数据
    :return
        返回随机森林中,决策树结果出现次数最多的
    """
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)

def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    """
    random_forest(评估算法性能, 返回模型得分)
    :param train: 训练数据集
    :param test: 测试数据集
    :param max_depth: 决策树深度 不能太深 容易过拟合
    :param min_size: 叶子节点的大小
    :param sample_size: 训练数据集的样本比例
    :param n_trees: 决策树的个数
    :param n_features: 选取的特征的个数
    :return
        predictions 每一行的预测结果 bagging 预测最后的分类结果
    """
    trees = list()
    # n_trees 表示决策树的数量
    for i in range(n_trees):
        # 随机抽样的训练样本 随机采样保证了每颗决策树训练集的差异
        sample = subsample(train, sample_size)
        # 创建一个决策树
        tree = bulid_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)

    # 每一行的预测结果 bagging预测最后的分类结果
    predictions = [bagging_predict(trees, row) for row in test]
    return predictions

def accuracy_metric(actual, predicted):
    """
    计算精确度 导入实际值和预测值
    :param actual:
    :param predicted:
    :return:
    """
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

def evaluate_algorithm(dataSet, algorithm, n_folds, *args):
    """
    评估算法性能 返回模型得分
    :param dataset:     原始数据集
    :param algorithm:   使用的算法
    :param n_folds:     数据的份数
    :param args:        其他参数
    :return
        scores          模型得分
    """
    # 将数据集进行抽重抽样
    folds = cross_validation_split(dataSet, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            row_copy[-1] = None
            test_set.append(row_copy)
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]

        # 计算随机森林的预测结果的正确率
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


def main():
    filename = r"C:\\Users\\Administrator\\Documents\\数据挖掘常用算法\\Ensemble_Method\\Random_Forest\\声呐信号分类\\sonar_all_data.txt"

    dataSet = loadDataSet(filename)

    n_folds = 5
    max_depth = 20
    min_size = 1
    sample_size = 1.0
    n_features = 15
    for n_trees in [1, 10, 20, 30, 40, 50]:
        scores = evaluate_algorithm(dataSet, random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
        # 每一次执行本文件时都能产生同一个随机数
        np.random.seed(1)
        print('random=', np.random.random())
        print('Trees: %d' % n_trees)
        print('Scores: %s' % scores)
        print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

if __name__ == '__main__':
    main()