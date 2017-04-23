#!usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import operator as op
import matplotlib.pyplot as plt


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

    # 将classCount字典分解为元组列表( Python 3 renamed dict.iteritems -> dict.items ),
    # 然后用itemgetter方法, 按照第二个元素的次序对元组进行排序( 此处为逆序,即按照从大到小次序排序 )。
    # 最后返回发生频率最高的元素标签。
    sortedClassCount = sorted(classCount.items(), key=op.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]


# 为了确保输入相同的数据集
def create_dataset():
    dataset = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return dataset, labels


# 在将特征数据(3种特征)输入到分类器之前,必须将待处理数据的格式改变为分类器可以接受的格式。
# 处理输入格式 得到特征矩阵、标签向量
def file2matrix(filename):
    fr = open(filename, 'r')
    numberOfLines = len(fr.readlines())        # 打开文件,得到文件的行数

    returnMat = np.zeros((numberOfLines, 3))   # 创建以零填充的numpy矩阵 特征矩阵
    classLabelVector = []                      # 创建返回的label向量
    index = 0                                  # 特征矩阵行数索引

    # 解析文件数据到列表
    for line in fr.readlines():
        line = line.strip()                     # 截取掉所有的回车字符
        listFromLine = line.split('\t')         # 使用tab字符\t将上一步得到的 整行数据 分割成一个元素列表
        returnMat[index, :] = listFromLine[0:3]   # 选取前3个元素, 存储到特征矩阵中
        classLabelVector.append(int(listFromLine[-1]))  # 必须明确地通知解释器,列表中存储的元素值为整型,否则python会将这些元素当字符串处理。
        index += 1
    print(returnMat)
    print(classLabelVector)

    return returnMat, classLabelVector


# 归一化特征值    newValue = (oldValue - min)/(max - min)
def autoNorm(dataset):
    minVals = dataset.min(0)    # 参数0使得函数可以从列中选取最小值,而不是选取当前行的最小值
    maxVals = dataset.max(0)
    ranges = maxVals - minVals      # 取值范围
    normDataset = np.zeros(np.shape(dataset))   # 创建新的返回矩阵
    datasetSize = dataset.shape[0]      # dataset的行数(样本数目); dataset.shape[1]为dataset的列数
    normDataset = dataset - np.tile(minVals, (datasetSize, 1))      # oldValue - min
    # 注意这里是具体特征值相除, 而对于某些数值处理软件包, /可能意味着矩阵除法, 但在numpy库中, 矩阵除法需要使用函数linalg.solve(matA, matB)
    normDataset = normDataset / np.tile(ranges, (datasetSize, 1))   # (oldValue - min)/(max - min)

    return normDataset, ranges, minVals


# 测试代码
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    normMatSize = normMat.shape[0]              # 样本数目
    numTestVecs = int(normMatSize * hoRatio)    # 测试样本数
    errorCount = 0.0
    for i in range(numTestVecs):
        # dataset为: 测试样本数到归一化后的样本数之间的样本
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:normMatSize, :],
                                     datingLabels[numTestVecs:normMatSize], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))


# 约会网站预测函数
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    # raw_input in Python 2 is just input in Python 3
    # input in Python 2 is eval(input()) in Python 3
    # 对象的三个特征
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))

    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])  # 需要预测的对象的三个特征
    classifierResult = classify0((inArr - minVals)/ranges, normMat, datingLabels, 3)
    print("You will probably like this person: ", resultList[classifierResult - 1])


# if __name__ == '__main__':
#     datingClassTest()

# if __name__ == '__main__':
#     datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#
#     # 散点图使用datingDataMat矩阵的第二、第三列数据,分别表示特征值"玩视频游戏所耗时间百分比"和"每周所消费的冰淇淋公升数"
#     # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
#     # 利用变量datingLabels存储的类标签属性,在散点图上绘制了色彩不等、尺寸不同的点。
#     ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * np.array(datingLabels), 15.0 * np.array(datingLabels))
#     plt.show()
