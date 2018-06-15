# !/usr/bin/env python3
# -*- utf-8 -*-
"""
__title__ = 关联分析
__author__ = LEI
__time__ = '2018/6/14'

  we are drowning in information,but starving for knowledge
"""

"""关联分析
    关联分析: 从大规模数据集中寻找物品间的隐含关系
    优点: 易编码
    缺点: 在大数据集上可能比较慢
    适用数据类型: 数值型或者标称型
    关联分析两个要点:   
                    频繁项集: 经常一起出现
                    关联规则：按时两种及两种以上物品之间可能存在很强的关系
    
    如何量化关联关系?
        支持度: 数据集中包含该项集的记录所占的比例
        可信度: 是针对一条关联规则来进行定义的 例如{尿布}-->{葡萄酒} 这条规则的可信度被定义为'支持度({尿布,葡萄酒})/支持度({葡萄酒})'
        
"""


"""Apriori算法思路:
    step1: 首先生成单个物品的项集列表
    step2: 接着扫描数据集来查看哪些项集满足最小支持度要求,同时将不满足最小支持度的集合会被去掉
    step3: 然后对剩下来的项集组合以生成包含两个元素的项集,重新扫描数据集,去掉不满足最小支持度的项集
    step4: 不断重复上述过程, 直到所有项集都被去掉
    
"""


def load_data():
    """
    加载数据集
    :return:
    """

    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def single_set(data_set):
    """
    生成单个物品的频繁项集
    :param data_set:
    :return:
    """
    single_list = list()
    for i in data_set:
        for j in i:
            if [j] not in single_list:
                single_list.append([j])
    single_list.sort()
    return map(frozenset, single_list)


def calc_support(data_set, single_list, limit_support=0.5):
    """
    计算项集的支持度 去掉不满足最小支持度的项集
    :param data_set: 数据集
    :param single_list: 单个项集'物品'
    :param limit_support: 最小支持度
    :return
        frequent_to_rate_dict
        frequent_list
    """
    frequent_to_rate_dict = dict()
    frequent_list = []
    m = len(data_set)
    for i in single_list:
        # print(i)
        count = 0
        for j in data_set:
            # print(j)
            if i.issubset(j):
                count += 1
        # print('count', count)
        # print('divide count m', count/m)
        if (count/m) >= limit_support:
            frequent_to_rate_dict[i] = count/m
            frequent_list.append(i)
    # print('frequent_to_rate_dict', frequent_to_rate_dict)
    # print('frequent_list', frequent_list)
    return frequent_to_rate_dict, frequent_list


def build_items(support_list, k):
    """
    构建一个k项的候选项集
    :param support_list: 频繁项集
    :param k: k个项
    :return:
        candidate_list 候选项集
    """

    range_list = len(support_list)
    # print('range_list', range_list)
    candidate_list = list()
    # print('support_list', support_list)
    # print('xxxxx', support_list[0])
    for i in range(range_list):
        # print('i', i)
        for j in range(i+1, range_list):
            if type(support_list[i]).__name__ == 'frozenset' and type(support_list[j]).__name__ == 'frozenset':
                # print('support_list[i]', support_list[i])
                # print('support_list[j]', support_list[j])
                union_set = support_list[i] | (support_list[j])   # | 并集
                # print('union_set', union_set, len(set(union_set)))
                if union_set not in candidate_list and len(union_set) == k:
                    candidate_list.append(union_set)
                    # print('candidate_list', candidate_list)
        # print('result candidate list', candidate_list)
    return candidate_list


def apriori(data_set, min_support=0.5):
    """
    Apriori 算法---频繁项集
        检查数据以确认每个项集都是频繁的
        保留频繁项集并构建k+1项组成的候选项集的列表
    :param data_set: 数据集
    :param min_support: 最小支持度
    :return:
        frequent_to_rate_dict   满足支持度的所有频繁项集与其支持度
        frequent_list           满足支持度的所有频繁项集
    """
    base_list = single_set(data_set)
    frequent_to_rate_dict, frequent_list = calc_support(data_set, base_list, min_support)
    frequent_lists = [frequent_list]
    k = 2
    # print('frequent_lists[k-2]', frequent_lists[k-2])
    while len(frequent_lists[-1]) > 0:
        candidate_list = build_items(frequent_lists[-1], k)
        frequent_to_rate_dict1, frequent_list1 = calc_support(data_set, candidate_list, min_support)
        # print('frequent_to_rate_dict1', frequent_to_rate_dict1)
        # print('frequent_list1', frequent_list1)
        frequent_to_rate_dict.update(frequent_to_rate_dict1)
        if len(frequent_list1) == 0:
            break
        frequent_lists.append(frequent_list1)
        k += 1
    return frequent_to_rate_dict, frequent_lists


def generate_rules(frequent_to_rate_dict, frequent_lists, min_conf=0.7):
    """
    关联规则生成函数
    :param frequent_to_rate_dict:  满足支持度的所有频繁项集与其支持度
    :param frequent_lists:   满足支持度的所有频繁项集
    :param min_conf:  最小可信度设置
    :return:
        rule_lists   关联规则列表
    """
    rule_lists = list()
    for i in range(1, len(frequent_lists)):
        for frequent_set in frequent_lists[i]:
            print('frequent_set', frequent_set)
            items = [frozenset([i]) for i in frequent_set]      # 从频繁项集中拆解出单个的项
            print('items', items)
            if i > 1:
                rules_from_consequent(frequent_set, items, frequent_to_rate_dict, rule_lists, min_conf)
            else:
                calc_conf(frequent_set, items, frequent_to_rate_dict, rule_lists, min_conf)
    print('rule_lists func', rule_lists)
    return rule_lists


def calc_conf(frequent_set, items, frequent_to_rate_dict, rule_lists, min_conf):
    """
    可信度计算
    :param frequent_set:   频繁项集中的元素
    :param items:          频繁项集中的元素的集合
    :param frequent_to_rate_dict:   所有元素的支持度字典
    :param rule_lists:    关联规则列表的空数组
    :param min_conf:       最小可信度
    :return:
        conf_list 记录 可信度大于阈值的集合
    """
    conf_list = list()
    for conseq in items:
        # print('conseq', conseq, 'frequent_set', frequent_set)
        conf = frequent_to_rate_dict[frequent_set] / frequent_to_rate_dict[conseq]
        # print('conf', conf)
        if conf >= min_conf:
            print('frequent_set', frequent_set, '>>>>', 'conseq', conseq, conf)
            print('conf', conf)
            rule_lists.append((frequent_set, conseq, conf))
            conf_list.append(conseq)
    return conf_list


def rules_from_consequent(frequent_set, items, frequent_to_rate_dict, rule_lists, min_conf=0.7):
    """
    生成候选规则集合
    :param frequent_set:      频繁项集中的元素
    :param items:              频繁项集中的元素的集合
    :param frequent_to_rate_dict: 所有元素的支持度的字典
    :param rule_lists: 关联规则列表的数组
    :param min_conf: 最小可信度
    """
    m = len(items[0])
    # print('m', m)
    # print('len(frequent_set)', len(frequent_set))
    if len(frequent_set) > (m + 1):
        hmp1 = build_items(items, m+1)
        print('hmp1', hmp1)
        hmp1 = calc_conf(frequent_set, hmp1, frequent_to_rate_dict, rule_lists, min_conf)
        if len(hmp1) > 1:
            print('应该继续迭代')
            rules_from_consequent(frequent_set, hmp1, frequent_to_rate_dict, rule_lists, min_conf)


def main():
    data_set = load_data()
    # single_list = single_set(data_set)
    # print('single list', single_list)
    # for i in single_list:
    #     print(i, type(i))
    #     if i.issubset([1, 2, 3]):
    #         print('True')
    #     print('False')
    # frequent_to_rate_dict, frequent_list = calc_support(data_set, single_list)

    # candidate_list = build_items(frequent_list, k=2)
    # print('候选集项', candidate_list)
    frequent_to_rate_dict, frequent_list = apriori(data_set, min_support=0.5)   # 至此找出了频繁项集
    print('frequent_to_rate_dict', frequent_to_rate_dict, 'length', len(frequent_to_rate_dict))
    print('frequent_list', frequent_list)

    rule_list = generate_rules(frequent_to_rate_dict, frequent_list, min_conf=0.7)
    print('rule_list', rule_list)


if __name__ == '__main__':
    main()
