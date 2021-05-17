import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import neural_network
from sklearn.metrics import accuracy_score

def readImages(fileName):

    imgs = []
    labels = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            rgb = mpimg.imread(row[0])
            img = []
            for elem in rgb:
                lista = []
                for x in elem:
                    lista.append(sum(x) // 3) #rgb to grayscale
                img.append(lista)
            imgs.append(img)

            labels.append(row[1])

    return imgs,labels
    



def splitData(inputs,outputs):
    np.random.seed(5)
    indexes = [i for i in range(len(outputs))]
    trainSample = np.random.choice(indexes, int(0.8 * len(outputs)), replace = False)
    testSample = [i for i in indexes  if not i in trainSample]

    trainInputs = [inputs[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]
    testInputs = [inputs[i] for i in testSample]
    testOutputs = [outputs[i] for i in testSample]

    return trainInputs,trainOutputs,testInputs,testOutputs

def flatten1(mat):
    x = []
    for line in mat:
        for el in line:
            x.append(el)
    return x 

def normalisation(trainData, testData):
    scaler = StandardScaler()
    if not isinstance(trainData[0], list):
        #encode each sample into a list
        trainData = [[d] for d in trainData]
        testData = [[d] for d in testData]
        
        scaler.fit(trainData)  #  fit only on training data
        normalisedTrainData = scaler.transform(trainData) # apply same transformation to train data
        normalisedTestData = scaler.transform(testData)  # apply same transformation to test data
        
        #decode from list to raw values
        normalisedTrainData = [el[0] for el in normalisedTrainData]
        normalisedTestData = [el[0] for el in normalisedTestData]
    else:
        scaler.fit(trainData)  #  fit only on training data
        normalisedTrainData = scaler.transform(trainData) # apply same transformation to train data
        normalisedTestData = scaler.transform(testData)  # apply same transformation to test data
    return normalisedTrainData, normalisedTestData



def main():
    inputs,outputs = readImages("date.txt")
    readImages("date.txt")


#---------split the data ---------------
    trainInputs,trainOutputs,testInputs,testOutputs = splitData(inputs,outputs)

#--------flaten the data---------------
    trainInputsFlatten = [flatten1(elem) for elem in trainInputs ]
    testInputsFlatten = [flatten1(elem) for elem in testInputs ]
 
#----------normalize the data-----------
    trainInputsNormalised,testInputsNormalized = normalisation(trainInputsFlatten, testInputsFlatten)

#-----------clasiffier the data----------
    classifier = neural_network.MLPClassifier(hidden_layer_sizes=(5, ),activation='relu', max_iter=100, solver='sgd', verbose=10, random_state=1, learning_rate_init=0.01)
    classifier.fit(trainInputsNormalised,trainOutputs)
    predictedLabels = classifier.predict(testInputsNormalized)
    acc = accuracy_score(testOutputs,predictedLabels)

#---------print the data-----------------
    print('\n\naccuracy: ', acc)
    print("Real labels: ", testOutputs)
    print("Computed labels: ", predictedLabels)
    print('\n\n')






main()