import numpy as np
from matplotlib import pyplot as plt

def fileOperation(filename):
    file = open(filename)
    dataSet = file.readlines()
    dataList = []
    labelList = []
    for line in dataSet:
        dataList.append([float(i) for i in line.strip().split('\t')[:-1]])
        labelList.append([int(i) for i in line.strip().split('\t')[-1:]])
    return np.asarray(dataList), np.asarray(labelList)

data, label = fileOperation("Implementation Files/Ch06/testSet.txt")
empty = []
x, y = np.hsplit(data, 2)
np.reshape(x, [len(x[0]) * len(x), 1])
print(x)