import numpy as np
from numpy import *

def fileToMatrix(filename):
    file = open(filename)
    arrayOfLines = file.readlines()
    numOfLines = len(arrayOfLines)
    returnMat = np.zeros([numOfLines, 3], dtype = double)
    labelVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        lineList = line.split('\t')
        returnMat[index,...] = lineList[0:3]
        labelVector.append(int(lineList[-1]))
        index += 1
    return returnMat, labelVector

def normalize(dataMat): #normalize the dataset
    colMinVal = dataMat.min(0)
    colMaxVal = dataMat.max(0)
    interval = colMaxVal - colMinVal
    normDataMat = np.zeros(shape(dataMat))
    colLength = shape(dataMat)[0]
    normDataMat = dataMat - np.tile(colMinVal, (colLength, 1))
    np.seterr(invalid = 'ignore')
    normDataMat = np.divide(normDataMat, np.tile(interval, (colLength, 1)))
    return normDataMat

def classify(sampleVec, dataSet, labelVec, K):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(sampleVec, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1) #calculate the distance to each element in the dataset
    distances = sqDistances ** (1/2)
    disIndices = distances.argsort() #replace the distances with their ranks
    totalDistance = 0
    dis, vote = {}, {}
    for i in range(dataSetSize):
        if disIndices[i] < K:
            dis[disIndices[i]] = (distances[i], i)
    for d in dis:
        totalDistance += dis[d][0]
    for i in dis:
        weight = dis[i][0] / totalDistance
        if labelVec[dis[i][-1]] in vote:
            vote[labelVec[dis[i][-1]]] += weight
        else:
            vote[labelVec[dis[i][-1]]] = weight
    sortedVoteCount = sorted({v : k for k, v in vote.items()}.items(), reverse = True)
    return sortedVoteCount[0][1]

def K_NN(sampleVec, K, dataSet, labelVec):
    return classify(normalize(sampleVec), normalize(dataSet), labelVec, K)