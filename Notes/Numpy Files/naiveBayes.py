def posterioriProb(inputVec, trainData, smoothCoef): # Laplacian smooth coefficient is used to prevent possibility annihilation
    labels = trainData[-1]
    uniqueLabels = set(labels)
    labelClasses = len(set(labels))
    trainData = trainData[:-1]
    totalSample = len(trainData[0])
    characterList = []
    labelDict = {}
    posterioriProbList = []
    for i in labels:
        if i not in labelDict:
            labelDict[i] = 1
        else:
            labelDict[i] += 1
    for character in trainData:
        singleCharacter = len(set(character))
        characterDict = {}
        for j in range(totalSample):
            if (character[j], labels[j]) not in characterDict:
                characterDict[(character[j], labels[j])] = 1
            else:
                characterDict[(character[j], labels[j])] += 1
        for j in characterDict:
            characterDict[j] = (characterDict[j] + smoothCoef) / float(labelDict[j[-1]] + singleCharacter*smoothCoef)
        characterList.append(characterDict)
    for label in labelDict:
        probability = (labelDict[label] + smoothCoef) / (totalSample + smoothCoef * labelClasses)
        for i in range(len(trainData)):
            layerCharacter = characterList[i]
            probability *= layerCharacter[(inputVec[i], label)]
        posterioriProbList.append(probability)
    posterioriIndex = posterioriProbList.index(max(posterioriProbList))
    return list(uniqueLabels)[posterioriIndex]