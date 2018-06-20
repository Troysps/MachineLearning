# !/usr/bin/env python3
# -*- utf-8 -*-
"""
__title__ = ''
__author__ = LEI
__time__ = '2018/6/19'

  we are drowning in information,but starving for knowledge
"""

"""FP-growth 算法: frequent pattern
    FP: frequent pattern
    建立FP tree 相对Apriori算法而言只需要遍历两次数据集
    算法思路:
        使用Tree模型存储数据
        第一次遍历数据, 以字典形式存储满足最小支持度的频繁项集
        第二次遍历数据, 根据频繁项集对数据进行排序并且更新FP tree 
"""


def load_data():

    data_set = [['r', 'z', 'h', 'j', 'p'],
                ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
                ['z'],
                ['r', 'x', 'n', 'o', 's'],
                ['y', 'r', 'x', 'z', 'q', 't', 'p'],
                ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return data_set


def init_data(data_set):
    """
    数据初始化, 对同一事务数据集尽量避免重复遍历
    tips: list ---> frozenset  will allow to be dict key
    :param data_set:
    :return:
        init_data
    """
    init_data = dict()
    for data in data_set:
        if frozenset(data) not in init_data.keys():
            init_data[frozenset(data)] = init_data.get(frozenset(data), 0) + 1
    print('init_data:', init_data)
    return init_data


def dict_support(data_set, min_sup):
    """step1：第一次遍历数据, 寻找满足数据的频繁项集
    """
    frequent_dict = dict()
    for data in data_set:
        for i in data:
            frequent_dict[i] = frequent_dict.get(i, 0) + data_set[data]
    more_than_sup_list = list(filter(lambda p: frequent_dict[p] < min_sup, frequent_dict.keys()))
    print('more_than_sup_list', more_than_sup_list)
    for i in more_than_sup_list:
        del frequent_dict[i]
    print('frequent_dict', frequent_dict)

    return frequent_dict


def sorted_data(data_set, frequent_dict):
    """
    step2：第二次遍历数据, 对数据集进行排序
    :param data_set:
    :param frequent_dict:
    :return:
    """
    sorted_list = list()
    for data in data_set:
        line_dict = dict()
        if len(data) > 0:
            for i in data:
                if i in frequent_dict.keys():
                    line_dict[i] = frequent_dict[i]
        print('line_dict', line_dict)
        sorted_list.append([v[0] for v in sorted(line_dict, reverse=True)])
    print('sorted_list', sorted_list)

    return sorted_list



def build_tree():
    pass


def main():
    data_set = load_data()
    data_set = init_data(data_set)
    frequent_dict = dict_support(data_set, min_sup=1)
    sorted_data(data_set, frequent_dict)

if __name__ == '__main__':
    main()
