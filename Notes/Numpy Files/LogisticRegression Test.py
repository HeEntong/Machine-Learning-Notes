import LogisticRegression
from LogisticRegression import *


dataTrain = loadDataset("Implementation Files/Ch05/horseColicTraining.txt")
dataTest, labelTest = loadDataset("Implementation Files/Ch05/horseColicTest.txt")
weights = gradAscent(dataTrain[0], dataTrain[1], 400, 0.02)
for i in range(1, 100, 2):
    for j in range(400, 1200, 10):
        weights = gradAscent(dataTrain[0], dataTrain[1], j, float(i) / 1000)
        tempSet = dataTest.dot(weights)
        tempSet = sigmoid(tempSet)
        for i in range(len(tempSet)):
            tempSet[i, 0] = sigmoidClassify(tempSet[i, 0])
        setLength = len(tempSet)
        output = tempSet - labelTest
        error = 0
        for i in range(setLength):
            if output[i, 0] != 0:
                error += 1