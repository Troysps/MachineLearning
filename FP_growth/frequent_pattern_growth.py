# !/usr/bin/env python3
# -*- utf-8 -*-
"""
__title__ = ''
__author__ = LEI
__time__ = '2018/6/15'

  we are drowning in information,but starving for knowledge
"""


class tree_node(object):
    def __init__(self, name_value, num_occur, parent_node):
        self.name = name_value      # 节点名称
        self.count = num_occur      # 节点出现次数
        self.node_link = None       # 不同项集的相同项通过node_link连接在一起
        # needs to be updated
        self.parent = parent_node   # 指向父节点
        self.children = {}          # 存储叶子节点

    def increase(self, num_occur):
        """
        increase 对count变量增加给定定值
        :param num_occur:
        :return:
        """
        self.count += num_occur

    def display(self, ind=1):
        """
        display 用于将树以文本形式显示
        :param ind:
        :return:
        """
        print(' '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.display(ind+1)


def load_data():

    data_set = [['r', 'z', 'h', 'j', 'p'],
                ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
                ['z'],
                ['r', 'x', 'n', 'o', 's'],
                ['y', 'r', 'x', 'z', 'q', 't', 'p'],
                ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return data_set


def create_init_set(data_set):
    """
    初始化数据集 dic{事务:出现的频次}
    :param data_set: 数据集
    :return:
        ret_dict 以字典形式存储单个项集及其对应的支持度
    """
    ret_dict = dict()
    for data in data_set:
        if frozenset(data) not in ret_dict:
            ret_dict[frozenset(data)] = ret_dict.get(frozenset(data), 0) + 1
    return ret_dict


def update_header(node_to_test, target_node):
    """
    更新头指针,建立相同元素之间的关系,例如: 左边的r指向右边的r值,就是后出现的相同元素,指向已经出现的元素
    从头指针的node_link开始,一直沿着node link直到到达链表末尾, 这就是链表
    性能: 如果链表很长可能会遇到迭代调用的次数限制
    :param node_to_test:    满足min support {所有元素+(value, tree node)}
    :param target_node:     tree对象的子节点
    :return:
    """
    # 建立相同元素之间的关系, 例如左边的r指向右边的r值
    while node_to_test.node_link is not None:
        node_to_test = node_to_test.node_link
    node_to_test.node_link = target_node


def update_tree(items, inTree, header_table, count):
    """
    update tree (更新FP-tree,第二次遍历)
    :param items: 满足min support 排序后的元素key的数组 (从大到小排序)
    :param inTree:  空的tree对下岗
    :param header_table: 满足min support {所有元素+(value, treeNone)}
    :param count: 原数据集中每一组事务出现的次数
    :return:
    """
    # 如果该元素在 inTree.children 这个字典中,就进行累加
    # 如果该元素不存在就inTree.children 字典中新增key,value 为初始化的tree_node 对象
    if items[0] in inTree.children:
        # 更新最大元素, 对应的tree node 对象的count进行叠加
        inTree.children[items[0]].increase(count)
    else:
        # 如果不存在子节点,就为该intree添加子节点
        inTree.children[items[0]] = tree_node(items[0], count, inTree)
        # 如果满足min support的dict字典的value值第二位为null, 我们就设置该元素为本节点对应的tree节点
        # 如果元素第二位不为null 我们就更新header节点
        if header_table[items[0]][1] is None:
            # header_table 只记录第一次节点出现的位置
            header_table[items[0]][1] = inTree.children[items[0]]
        else:
            # 本质是修改header_table的key对应的tree的node link值
            update_header(header_table[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:
        # 递归调用,在items[0]的基础上, 添加items[1]做子节点,count只要循环的进行累计加和而已, 统计出节点的最后的统计值
        update_tree(items[1:], inTree.children[items[0]], header_table, count)


def create_tree(data_set, min_sup=1):
    """
    创建FP树
    :param data_set:数据集 dict{行:出现次数}的样本数据
    :param min_sup: 最小支持度
    :return:
        tree FP-tree
        headerTable   头指针表 存储频繁项集与其对应的支持度
    """
    # step1 第一次遍历数据集,创建头指针表
    header_table = dict()
    for trans in data_set:
        for item in trans:
            header_table[item] = header_table.get(item, 0) + data_set[trans]

    print('第一个遍历数据集创建header_table', header_table)
    # 根据最小支持度过滤
    less_than_min_sup = list(filter(lambda k: header_table[k] < min_sup, header_table.keys()))
    print('less than min support', less_than_min_sup)
    for k in less_than_min_sup:
        del header_table[k]
    print('过滤后的header table--头指针表', header_table)

    # 如果所有数据都不满足最小支持度,返回None, None
    frequent_item_set = set(header_table.keys())
    if len(frequent_item_set) == 0:
        return None, None

    for k in header_table:
        header_table[k] = [header_table[k], None]

    print('再次优化的header_table', header_table)

    # 第二次遍历数据集,构建树FP-TREE
    ret_tree = tree_node('NULL Set', 1, None)
    # 循环dic{行:出现次数}的样本数据
    for trans, count in data_set.items():
        print('trans', trans, 'count', count)
        # local_d = dict{元素key:元素总出现次数}
        local_d = dict()
        for item in trans:
            # 判断是否在满足min support的集合中
            if item in frequent_item_set:
                print('header_table[item][0]=', header_table[item][0], header_table[item])
                local_d[item] = header_table[item][0]

        print('local_d', local_d)
        # 判断local_d dic 是否存在
        if len(local_d) > 0:
            # p = key, value; 所以是通过value值的大小, 进行从大到小的排序
            # ordered_items表示取出元组的key值,也就是字母本身, 但是字母本身是从大到小的顺序
            ordered_items = [v[0] for v in sorted(local_d.items(), key=lambda p: p[1], reverse=True)]
            print('ordered_items', ordered_items)

            # 填充树, 通过有序的orderItems的第一位,进行顺序填充,第一层的子节点
            update_tree(ordered_items, ret_tree, header_table, count)
    return ret_tree, header_table


"""FP tree --- 发现频繁项集 接下来需要从FP tree中构建关联规则
    概念：
        条件模式基:头部链表中的某一点的前缀路径组合就是条件模式基  条件模式基的值取决于末尾节点的值
        条件FP树: 以条件模式基为数据集构造的FP树叫做条件FP树
    原理与实现: 得到FP树后 需要对每一个频繁项集 逐个挖掘频繁项集
        具体过程为:
            首先获得频繁项的前缀路径
            然后将前缀路径作为新的数据集 以此构建前缀路径的条件FP树
            然后对条件FP树中的每一个频繁项 获得前缀路径并以此构建新的条件FP树 不断迭代 直到条件FP树中只包含一个频繁项为止    
"""


def ascend_tree(leaf_node, prefix_path):
    """
    ascend_tree 如果存在父节点 就记录当前节点的name值
    :param leaf_node: 查询的的节点对于的node_tree
    :param prefix_path: 要查询的节点值
    :return:
        prefix_path   递归该节点上所有要查询的节点值(条件模式基)
    """
    if leaf_node.parent is not None:
        prefix_path.append(leaf_node.name)
        ascend_tree(leaf_node.parent, prefix_path)


def find_prefix_path(base_pat, tree_node):
    """
    基础数据集
    step1：递归FP tree 寻找该节点的父节点 ----> 实质上就是找到该节点的频繁项集
    step2：对递归计算而出的频繁项集 计数 -----> 相当于Apriori算法中的频繁项集组合
    :param base_pat: 要查询的节点值
    :param tree_node: 查询的节点所在的当前node_tree
    :return:
        cond_pats 对非base_pat的倒叙值作为key 赋值为count数

    """
    cond_pats = dict()
    while tree_node is not None:
        prefix_path = list()
        # 寻找该节点的父节点, 相当于找到了该节点的频繁项集
        ascend_tree(tree_node, prefix_path)
        print('prefix_path', prefix_path)
        # 避免 单独 'z'一个元素 添加了空节点
        if len(prefix_path) > 1:
            # 对非base_pat的倒叙值作为key 赋值为count数
            # prefix_path[1:] 变frozenset后 字母就变无序了
            print(prefix_path[1:])
            cond_pats[frozenset(prefix_path[1:])] = tree_node.count
        # 递归 寻找该节点的下一个 相同值的链接节点
        tree_node = tree_node.node_link
    return cond_pats

def mine_tree(in_tree, header_table, min_sup, prefix, frequent_list):
    """
    mine_tree 创建条件FP tree
    构建条件FP tree -----> 实质上就是构建关联规则
    :param in_tree:  FP TREE
    :param header_table:  满足最小支持项集{所有元素+{value, tree_node}}
    :param min_sup:     最小支持项集
    :param prefix:     prefix 为newFreqset 上一次的存储记录 一旦没有myhead 就不会更新
    :param frequent_list: 用来存储频繁子项的列表
    :return:
    """
    # 通过value进行从小到大的排序 得到频繁项集的key
    # 最小支持项集的key的list集合
    # print('sorted header table', sorted(header_table.items()))
    big_list = [v[0] for v in sorted(header_table.items(), key=lambda p:p[1][0])]
    print()
    print('big_list', big_list)
    # 循环遍历 最频繁项集的key  从小到大的递归寻找对应的频繁项集
    for base_pat in big_list:
        # prefix 为newFreqset 上一次的存储记录 一旦没有  myhead 就不会更新
        new_frequent_set = prefix.copy()
        new_frequent_set.add(base_pat)
        print('new_frequent_set= ', new_frequent_set, prefix)

        frequent_list.append(new_frequent_set)
        print('frequent_list', frequent_list)

        cond_pattern_bases = find_prefix_path(base_pat, header_table[base_pat][1])
        print('cond_pattern_bases', base_pat, '*'*8,  cond_pattern_bases)

        # 构建FP tree
        cond_tree, cond_head = create_tree(cond_pattern_bases, min_sup)
        print('cond_head', cond_head)
        if cond_head is not None:
            cond_tree.display(1)
            print('\n\n\n')
            # 递归 cond_head 找出频繁项集
            mine_tree(cond_tree, cond_head, min_sup, new_frequent_set, frequent_list)
        print('\n\n\n\n')



    pass

def main():
    data_set = load_data()
    init_data_set = create_init_set(data_set)
    print(init_data_set)
    fp_tree, header_table = create_tree(init_data_set, min_sup=2)
    print('header table', header_table)
    fp_tree.display()

    # 抽取条件模式基
    # 查询树节点的 频繁子项
    print('x --->', find_prefix_path('x', header_table['x'][1]))
    print('z --->', find_prefix_path('z', header_table['z'][1]))
    print('r --->', find_prefix_path('r', header_table['r'][1]))


    # 创建条件模式基
    frequent_list = []
    mine_tree(fp_tree, header_table, 3, set([]), frequent_list)
    print(frequent_list)

if __name__ == '__main__':
    main()
