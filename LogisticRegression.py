import numpy as np
import matplotlib.pyplot as plt

def loadDataset(filename):
    dataMat = []; labelMat = []
    file = open(filename)
    fileString = file.readlines()
    for line in fileString:
        lineArray = line.strip().split()
        dataMat.append([1.0] + [float(i) for i in lineArray[:-1]])
        labelMat.append([int(float(i)) for i in lineArray[-1:]])
    dataMat = np.asarray(dataMat)
    labelMat = np.asarray(labelMat)
    return dataMat, labelMat

def sigmoid(X):
    return 1.00/(1+np.exp(-X))

def sigmoidClassify(X):
    if 1 / (1 + float(np.exp(-X))) > 0.5:
        return 1
    else:
        return 0

def gradAscent(dataMat, labelMat, iterTimes, stepLength):
    row, column = np.shape(dataMat)
    weights = np.ones((column, 1))
    for i in range(iterTimes):
        h = sigmoid(dataMat.dot(weights))
        error = labelMat - h
        weights = weights + stepLength * (dataMat.transpose().dot(error))
    return weights

def plotFit(weights, filename): # Plot the two-dimensional model 
    dataMat, labelMat = loadDataset(filename)
    col = np.shape(dataMat)[0]
    x1, y1, x2, y2 = [], [], [], []
    for i in range(col):
        if labelMat[i] == 1:
            x1.append(dataMat[i, 1])
            y1.append(dataMat[i, 2])
        else:
            x2.append(dataMat[i, 1])
            y2.append(dataMat[i, 2])
    figure = plt.figure()
    plt.scatter(x1, y1, c = 'red')
    plt.scatter(x2, y2, c = 'green')
    x = np.arange(-4, 4, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    plt.plot(x, y)
    plt.xlabel('Type 1')
    plt.ylabel('Type 2')
    plt.show()