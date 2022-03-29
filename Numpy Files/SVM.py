import numpy as np
from matplotlib import pyplot as plt
import random

def fileOperation(filename):
    file = open(filename)
    dataSet = file.readlines()
    dataList = []
    labelList = []
    for line in dataSet:
        dataList.append([float(i) for i in line.strip().split('\t')[:-1]])
        labelList.append([int(i) for i in line.strip().split('\t')[-1:]])
    return np.asarray(dataList), np.asarray(labelList)

def selectRandOpt(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j

def clipAlpha(aj, lowerBound, upperBound):
    if aj > upperBound:
        aj = upperBound
    if aj < lowerBound:
        aj = lowerBound
    return aj

