# -*- coding: utf-8 -*-
"""
__title__ = Build Balance KD Tree
__author__ = 'LEI'
__mtime__ = '2018/08/02'
"""

import numpy as np
from math import sqrt
from collections import namedtuple

"""
构造平衡KD tree
    (1) 开始: 构造根节点, 根节点对应与包含T的k维度空间的超矩形区域
    选择x^{(1)}为坐标轴, 以T中所有实例的x^{(1)}坐标的中位数为切分点 
    将根节点对应的超矩形区域切分为两个子区域, 切分由通过切分点并与坐标轴x^{(1)}垂直的超平面实现
    由根节点生成深度为1的左、右子节点: 左子节点对应与坐标x^{(1)}小于切分点的子区域, 右子节点对应于坐标x^{(1)}大于切分点的子区域
    将落在切分超平面上的实例点保存在根节点中
    (2) 重复
    (3) 直到两个子区域没有实例存在时停止 从而形成kd树的区域划分
"""
def load_data_set():
    data_set = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
    return data_set

class KdNode(object):
    def __init__(self, node, dimension, left_node, right_node):
        self.node = node           # 节点
        self.dimension = dimension # 对应维度
        self.left_node = left_node  # 小
        self.right_node = right_node # 大


class KdTree(object):
    def __init__(self, data_set):
        k = np.shape(data_set)[1]

        def create_node(dimension, data_set):
            if not data_set:
                return
            # print('dimension', dimension)
            data_set.sort(key=lambda x: x[dimension])
            split = len(data_set) // 2
            node = data_set[split]
            dimension_next = (dimension + 1 ) % k
            # print("dimension next", dimension_next)
            return KdNode(node,
                          dimension,
                          create_node(dimension_next, data_set[:split]),
                          create_node(dimension_next, data_set[split+1:]))

        self.root = create_node(0, data_set)

def preorder(root):
    """
    前序遍历 从根节点开始 先遍历左节点 再遍历右节点

    """
    print(root.node)
    if root.left_node:
        preorder(root.left_node)
    if root.right_node:
        preorder(root.right_node)


def Kd_tree_search_path(data, tree):
    k = len(data)
    # print('search kd tree', k)
    search_path = list()


    def kd_nearest_search_path(dimension, tree=tree, data=data):
        """
        查找最近邻点
        """
        if not tree:
            return search_path
        # print('tree', tree.node)
        search_path.append(tree.node)
        # print('search_node', search_path)
        dimension_next = (dimension + 1) % k
        # print(data[dimension], tree.node[dimension])
        if data[dimension] <= tree.node[dimension]:
            return kd_nearest_search_path(dimension_next, tree.left_node, data)
        elif data[dimension] > tree.node[dimension]:
            return kd_nearest_search_path(dimension_next, tree.right_node, data)

    return kd_nearest_search_path(0, tree, data)

result = namedtuple("Result_tuple", "nearest_point nearest_dist nodes_visited")

def find_nearest(tree, point):
    k = len(point)

    def travel(kd_node, target, max_dist):
        if kd_node is None:
            print('kd_node', kd_node)
            return result([0] * k, float("inf"), 0)

        nodes_visited = 1

        s = kd_node.dimension  # 进行分割的维度
        pivot = kd_node.node    # 进行分割的“轴”

        if target[s] <= pivot[s]:
            nearer_node = kd_node.left_node  # 如果目标点第s维小于分割轴的对应值(目标离左子树更近)
            further_node = kd_node.right_node   # 下一个访问节点为左子树根节点
        else:                                   # 同时记录下右子树
            nearer_node = kd_node.right_node    # 目标离右子树更近
            further_node = kd_node.left_node    # 下一个访问节点为右子树根节点

        temp1 = travel(nearer_node, target, max_dist)   # 进行遍历找到包含目标点的区域
        print('temp1', temp1)

        nearest = temp1.nearest_point   # 以此叶结点作为“当前最近点”
        dist = temp1.nearest_dist   # 更新最近距离

        nodes_visited += temp1.nodes_visited

        if dist < max_dist:
            max_dist = dist # 最近点将在以目标点为球心，max_dist为半径的超球体内

        temp_dist = abs(pivot[s] - target[s])    # 第s维上目标点与分割超平面的距离
        if max_dist < temp_dist:                # 判断超球体是否与超平面相交
            print('temp_dist', temp_dist)
            return result(nearest, dist, nodes_visited) # 不相交则可以直接返回，不用继续判断

        # 计算目标点与分割点的欧式距离
        temp_dist = sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pivot, target)))

        # 计算目标点与分割点的欧氏距离
        if temp_dist < dist:
            nearest = pivot
            dist = temp_dist
            max_dist = dist

        temp2 = travel(further_node, target, max_dist)
        print("temp2", temp2)

        nodes_visited += temp2.nodes_visited
        if temp2.nearest_dist < dist:
            nearest = temp2.nearest_point
            dist = temp2.nearest_dist
        return result(nearest, dist, nodes_visited)
    return travel(tree.root, point, float("inf"))


def main():
    data_set = load_data_set()
    # print(data_set)
    kd_tree = KdTree(data_set)
    # print(kd_tree.root.node)
    # print(kd_tree.root.left_node.node)
    # print(kd_tree.root.right_node.node)
    # print("----前序遍历----")
    # preorder(kd_tree.root)
    # # 寻找点(3, 4.5) 的最近邻点
    point = [3, 4.5]
    search_path = Kd_tree_search_path(data=point, tree=kd_tree.root)
    print(search_path)
    ret = find_nearest(kd_tree, [3,4.5])
    print(ret)


if __name__ == '__main__':
    main()
    # test = {1: {2: {5: 6}, 3: 4}}
    # print(len(test.keys()))
    # print(len(test[1].keys()))

    # # 如何递归查找所有的值
    # def is_tree(cls):
    #     if isinstance(cls, dict):
    #         return False
    #     elif isinstance(cls, int):
    #         return True
    #
    # def get_value(test):
    #     key_list = list(test.keys())
    #     print('key list', key_list)
    #     for key in key_list:
    #         print('key', key)
    #         next = test.get(key)
    #         if is_tree(next):
    #             print('result:', next)
    #         else:
    #             get_value(next)
    # get_value(test)
