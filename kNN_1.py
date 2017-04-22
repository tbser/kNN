#!usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import operator as op   # 运算符模块,k-近邻算法 执行排序操作时 将使用这个模块提供的函数。


# 为了确保输入相同的数据集
def create_dataset():
    dataset = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return dataset, labels


# kNN
def classify0(inX, dataset, labels, k):
    datasetSize = dataset.shape[0]     # dataset的行数(样本数目); dataset.shape[1]为dataset的列数

    # 计算输入向量inX与dataset里面数据点的距离 此处为计算欧式距离
    # step 1: calculate Euclidean distance
    # tile(A, reps): Construct an array by repeating A reps times
    diffMat = np.tile(inX, (datasetSize, 1)) - dataset   # 将inX向量扩展成和dataset一样的矩阵 相减
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)   # sum is performed by row
    distances = sqDistance ** 0.5        # 开根号

    # step 2: sort the distance
    sortedDistanceIndices = distances.argsort()    # 按距离的从小到大进行排列, 返回的是下标

    # 确定前k个距离最小元素所在的主要分类
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistanceIndices[i]]    # 第i个距离的下标 对应的标签
        # D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None.
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1   # 如:voteIlabel为'B',则'B'对应的key值加1。
    # print(classCount)   # {'B': 2, 'A': 1}

    # 将classCount字典分解为元组列表( Python 3 renamed dict.iteritems -> dict.items ),
    # 然后用itemgetter方法, 按照第二个元素的次序对元组进行排序( 此处为逆序,即按照从大到小次序排序 )。
    # 最后返回发生频率最高的元素标签。
    sortedClassCount = sorted(classCount.items(), key=op.itemgetter(1), reverse=True)
    print(sortedClassCount)   # [('B', 2), ('A', 1)]
    print(sortedClassCount[0][0])
    return sortedClassCount[0][0]


if __name__ == '__main__':
    dataset, labels = create_dataset()
    classify0([0, 0], dataset, labels, 3)
