# !/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
Project:社区留言板侮辱性言论屏蔽
@author:Lei
@datetime:2018/5/7
@email:1173682167@qq.com
"""
"""
Project Readme: 构建一个快速过滤器来屏蔽在线社区留言板上的侮辱性言论.
如果某条留言使用了负面或者侮辱性的语言，那么就将该留言标识为内容不当。
对此问题建立两个类别: 侮辱类和非侮辱类，使用 1 和 0 分别表示
"""
import numpy as np
import logging  

# np.seterr(divide='ignore', invalid='ignore')
logging.basicConfig(level=logging.INFO,  
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',  
                    datefmt='%a, %d %b %Y %H:%M:%S',  
                    filename='./Naive Bayes Classifier/tmp/test.log',  
                    filemode='w')

def loadDataSet():
    """
    创建数据集
    :return: 单词列表postingList, 所属类别classVec
    """
    logging.info("read dataSet")
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], #[0,0,1,1,1......]
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec

# step1 建立分词集合
def createWordsSet(dataSet):
    logging.info("create words Set")
    # print(np.shape(dataSet)) (6,)

    wordsSet = set([])

    for words in dataSet:
        wordsSet = wordsSet | set(words)
    return list(wordsSet)

# step2 转化数据为向量
def transferToVec(data, wordsSet):

    logging.info("transfer words to vec")
    # print(len(wordsSet)) output 32
    vec = [0] * len(wordsSet)
    print(vec, len(vec))

    for i in data:
        if i in wordsSet:
            print("词汇存在于集合中")
            vec[wordsSet.index(i)] += 1
        else:
            print("词汇不存在与集合中:%s"% i)
        
    # print(vec)
    return vec

# 先验概率及其条件概率计算
def tranNBMaxLike(vecDataSet, classLable):
    """
        param vecDataSet: 输入空间的向量表示形式
        param classLable: 输出空间的类标记集合

        return p1Abusive: 先验概率
        return p0Abusive: 先验概率
        return p1Con: 条件概率 1类别 每个单词出现次数的占比
        return p0Con: 条件概率 0类别 每个单词出现次数的占比
    """
    logging.info("Maximum likelihood estimate p1Con p0Con")
    classLen = len(classLable)
    vecLen = len(vecDataSet[0])

    print(classLen, vecLen)
    # 先验概率
    p1Abusive = sum(classLable) / float(classLen)
    print('侮辱言论概率', p1Abusive)
    p0Abusive = 1 - p1Abusive
    print('非侮辱言论概率', p0Abusive)

    # 针对nxj维度的数据集 遵循条件独立性假设
    # 构造记录单词出现次数的列表
    p1Vec = np.zeros(vecLen)
    p0Vec = np.zeros(vecLen)

    # 条件概率下 每个类别出现单词的频数
    p0Nums = 0
    p1Nums = 0

    # 条件概率
    for i in range(classLen):
        # 在侮辱言论发生的条件下
        if classLable[i] == 1:
            # 单词出现的频数 -- 以向量形式记录
            p1Vec += vecDataSet[i]
            # 该类别时 出现的单词总频数
            p1Nums += sum(vecDataSet[i])
        # 在非侮辱言论发生的条件下
        if classLable[i] == 0:
            p0Vec += vecDataSet[i]
            p0Nums += sum(vecDataSet[i])

    # print(p1Vec)
    # print(p0Vec)
    # print(sum(p1Vec), p1Nums)
    # print(sum(p0Vec), p0Nums)

    # 计算条件概率
    # 类别1，即侮辱性文档的[P(F1|C1),P(F2|C1),P(F3|C1),P(F4|C1),P(F5|C1)....]列表
    # 即 在1类别下，每个
    # 单词出现次数的占比
    p1Con = p1Vec / p1Nums
    # print('p1Con', p1Con)
    # 类别0，即正常文档的[P(F1|C0),P(F2|C0),P(F3|C0),P(F4|C0),P(F5|C0)....]列表
    # 即 在1类别下，每个单词出现次数的占比
    p0Con = p0Vec / p0Nums
    # print('p0Con', p1Con)

    # print(p1)
    # print(p0)

    return p1Con, p0Con, p1Abusive, p0Abusive



def classifyNB(testVec, p1Con, p0Con, p1Abusive, p0Abusive):
    """
        param: testVec 待测数据集的向量表示形式
        param p1Con: 类别1，即侮辱性文档的[P(F1|C1),P(F2|C1),P(F3|C1),P(F4|C1),P(F5|C1)....]列表
        param p0Con: 类别1，即侮辱性文档的[P(F1|C1),P(F2|C1),P(F3|C1),P(F4|C1),P(F5|C1)....]列表
        param p1Abusive: 即 在1类别下，每个单词出现次数的占比
        param p0Abusive: 即 在0类别下，每个单词出现次数的占比

        return: 0 or 1 类别

    """
    """
        计算方法:解决下溢出的问题 将乘积问题转化为相加
                对乘积去自然对数 ln(a * b) = ln(a) + ln(b)
                求对数可以避免下溢出或者浮点数舍入导致的错误。同时，采用自然对数进行处理不会有任何损失
    """
    logging.info("Bayes classify")
    testVec = np.array(testVec)
    p1Con = np.array(p1Con)
    p0Con = np.array(p0Con)

    """
        注意:numpy array进行向量运算时 若存在无效值将会使得 运算结果为nan
            需要设置 np.seterr(divide='ignore', invalid='ignore')
            即使设置之后 计算np.sum 还是保存 弃用此方法
    """
    pp = np.log(p1Con)
    print('pp', pp)
    print(sum(testVec * pp))
    print(sum(testVec * p1Con))
    p1 = sum(testVec * p1Con) + p1Abusive
    p0 = sum(testVec * p0Con) + p0Abusive

    print('p1', p1)
    print('p0', p0)

    if p1 > p0:
        return 1
    return 0

def main():
    dataSet, classLable = loadDataSet()
    # print(dataSet)
    # print(classLable)
    wordsSet = createWordsSet(
        dataSet)
    vecDataSet = []
    for i in dataSet:
        vecDataSet.append(transferToVec(i, wordsSet))
    print(vecDataSet)
    p1Con, p0Con, p1Abusive, p0Abusive = tranNBMaxLike(vecDataSet, classLable)
    test = ['love', 'my', 'dalmation']  # 0
    # test = ['stupid', 'garbage']  # 1
    testVec = transferToVec(test, wordsSet)
    print(testVec)

    result = classifyNB(testVec, p1Con, p0Con, p1Abusive, p0Abusive)
    print(result)


if __name__ == '__main__':
    main()