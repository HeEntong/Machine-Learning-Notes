from decisionTree import *
def getData(filename):
    dataset = []
    file = open(filename)
    dataLines = file.readlines()
    for line in dataLines:
        dataLine = line.split('\n')[0]
        modifiedVec = dataLine.split('\t')
        dataset.append(modifiedVec)
    return dataset

dataset= getData("./machinelearninginaction/Ch03/lenses.txt")
labels = ['age','prescript','astigmatic','tearRate']
p = createTree(dataset, labels)
p.visualize(p, 1)
