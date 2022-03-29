from math import log

class decisionNode(object): # tree organized by decision tree data structure
    def __init__(self, label):
        self.label = label
        self.branches = []
        self.decision = ""
    def assignDecision(self, decision):
        self.decision += decision
    def addBranch(self, newNode):
        newNode.assignDecision(self.decision)
        self.branches.append(newNode)
    def visualize(self, treeNode, layer): # Visualize manipulation displays the layer an label belongs to and its anterior choice
        print("({}){}: {}".format(layer, treeNode.decision, treeNode.label))
        for s in treeNode.branches:
            self.visualize(s, layer + 1)

def shannonEntropy(dataset):
    entriesNum = len(dataset)
    labelCount = {}
    for dataVec in dataset:
        dataLabel = dataVec[-1] # The last element in the vector is our decision
        if dataLabel not in labelCount:
            labelCount[dataLabel] = 0
        labelCount[dataLabel] += 1
    shannonEntropy = 0.000
    for label in labelCount:
        probability = labelCount[label] / entriesNum
        shannonEntropy -= probability * log(probability, 2)
    return shannonEntropy

def splitDataset(dataset, index, expectValue): # Search the dataset with specified index and return the reduced dataset
    retDataset = []
    for lineVec in dataset:
        if lineVec[index] == expectValue:
            reducedVec = lineVec[:index]
            reducedVec += lineVec[index+1:]
            retDataset.append(reducedVec)
    return retDataset

def optimalPartition(dataset): # ID3 Algotihm
    featureNums = len(dataset[0]) - 1
    originalEntropy = shannonEntropy(dataset)
    bestFeature = -1
    maxInfoGain = 0.00
    for featureIndex in range(featureNums):
        labelVec = set([dataset[i][featureIndex] for i in range(len(dataset))])
        extraEntropy = 0.00
        for label in labelVec:
            reducedSet = splitDataset(dataset, featureIndex, label)
            extraEntropy += len(reducedSet) / float(len(dataset)) * shannonEntropy(reducedSet)
        if originalEntropy - extraEntropy > maxInfoGain:
            maxInfoGain = originalEntropy - extraEntropy
            bestFeature = featureIndex
    return bestFeature

def majorityCount(classList):
    classNum = {}
    for member in classList:
        if member not in classNum:
            classNum[member] = 0
        classNum[member] += 1
    reverseDict = {v:k for k, v in classNum.items()}
    orderList = sorted(reverseDict)
    return reverseDict[max(orderList)]
        
    
def createTree(dataset, labels):
    classList = [data[-1] for data in dataset]
    if classList.count(classList[0]) == len(classList):
        return decisionNode(classList[0])
    if len(dataset[0]) == 1:
        return decisionNode(majorityCount(classList))
    bestPartition = optimalPartition(dataset)
    bestPartitionLabel = labels[bestPartition]
    newTree = decisionNode(bestPartitionLabel)
    uniqueVal = set([data[bestPartition] for data in dataset])
    del(labels[bestPartition])
    for value in uniqueVal:
        subLabels = labels[:]
        branchNode = createTree(splitDataset(dataset, bestPartition, value), subLabels)
        branchNode.assignDecision(value)
        newTree.addBranch(branchNode)
    return newTree