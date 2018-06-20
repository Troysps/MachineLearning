# !/usr/bin/env python3
# -*- utf-8 -*-
"""
__title__ = ''
__author__ = LEI
__time__ = '2018/6/15'

  we are drowning in information,but starving for knowledge
"""


def load_data():

    data_set = [['r', 'z', 'h', 'j', 'p'],
                ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
                ['z'],
                ['r', 'x', 'n', 'o', 's'],
                ['y', 'r', 'x', 'z', 'q', 't', 'p'],
                ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]

    return data_set


"""step1:
    1.遍历所有的数据集集合,计算所有项的支持度
    2.丢弃非频繁项
    3.基于支持度 降序排序所有的项
    4.所有数据集合按照得到的顺序重新整理
    5.重新整理完成后,丢弃每个集合末尾非频繁的项
"""


def filter_data(data_set):
    """
    遍历所有的数据集集合,筛选出所有的单个项
    :param data_set:
    :return:
    """
    items = list()
    # items_support = dict()
    for i in data_set:
        for j in i:
            if j not in items:
                items.append([j])
    # print(items)
    return map(frozenset, items)


def calc_support(data_set, items, min_support=0.5):
    """
    遍历所有的数据集集合,计算所有项的支持度,并丢弃非频繁项
    :param data_set:
    :param items:
    :param min_support:
    :return:
    """
    m = len(data_set)
    # print('m', m)
    frequent_list = list()
    frequent_support = dict()
    for item in items:
        count = 0
        for i in data_set:
            if item.issubset(i):
                count += 1
        # print(item, count/m)
        if ((count/m) >= min_support) and (item not in frequent_list):
            frequent_support[item] = count
            frequent_list.append(item)

    return frequent_list, frequent_support


def sorted_data_set(data_set, frequent_support):
    """基于支持度 降序排序所有的项"""
    sorted_data_set = list()
    for i in data_set:
        print(sorted(i))

    return sorted_data_set


def main():
    data_set = load_data()
    print('data_set', data_set)
    items = filter_data(data_set)
    print('items', items)
    frequent_list, frequent_support = calc_support(data_set, items, min_support=0.5)
    print('frequent_list', frequent_list)
    print('frequent_support', frequent_support)
    sorted_data_set(data_set, frequent_support)


if __name__ == '__main__':
    main()
