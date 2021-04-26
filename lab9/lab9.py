import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from math import exp
from numpy.linalg import inv



def loadData(fileName):
    dataInputs = []
    dataOutputs = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            dataInputs.append([float(row[0]),float(row[1]),float(row[2]),float(row[3])])
            dataOutputs.append(row[4])
       
    return dataInputs,dataOutputs

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



def regresieLogisticaCuTool(trainInputs,trainOutputs,testInputs,testOutput):
    clasificator = linear_model.LogisticRegression()
    clasificator.fit(trainInputs,trainOutputs)
    w0,w1,w2,w3 = clasificator.intercept_,clasificator.coef_[0],clasificator.coef_[1],clasificator.coef_[2]
    computedTestOutputs = clasificator.predict(testInputs)
    print('computedTestOutputs: ',computedTestOutputs,'\n')
    print('testOutputs: ',testOutput,'\n')

    eroareaCuTool(computedTestOutputs,testOutput)

def eroareaCuTool(computedTestOutputs,testOutput):
    error = 1 - accuracy_score(testOutput,computedTestOutputs)
    print("classification error (tool): ", error,'\n')

def transformareaOutputs(testOutputs):
    testOutputsTransformat = []
    for i in testOutputs:
        if i == 'Iris-setosa':
            testOutputsTransformat.append(0)
        if i == 'Iris-versicolor':
            testOutputsTransformat.append(1)
        if i == 'Iris-virginica':
            testOutputsTransformat.append(2)
    return testOutputsTransformat


def sigmoid(x):
    return 1 / (1 + exp(-x))
    
# use the gradient descent method
# simple stochastic GD
def fit(x, y, learningRate = 0.001, noEpochs = 1000):
    coef_ = [0.0 for _ in range(1 + len(x[0]))]    
    for epoch in range(noEpochs):
        for i in range(len(x)): # for each sample from the training data
            ycomputed = sigmoid(eval(x[i], coef_))     # estimate the output
            crtError = ycomputed - y[i]     # compute the error for the current sample
            for j in range(0, len(x[0])):   # update the coefficients
                coef_[j + 1] = coef_[j + 1] - learningRate * crtError * x[i][j]
            coef_[0] = coef_[0] - learningRate * crtError * 1

    intercept_ = coef_[0]
    coef_ = coef_[1:]
    return intercept_,coef_[0],coef_[1],coef_[2],coef_[3]
 
def eval( xi, coef):
    yi = coef[0]
    for j in range(len(xi)):
        yi += coef[j + 1] * xi[j]
    return yi

def predictOneSample(intercept_,coef_, sampleFeatures):
    coefficients = [intercept_] + [c for c in coef_]
    computedFloatValue = eval(sampleFeatures, coefficients)
    return sigmoid(computedFloatValue)
    

def predict(intercept_,coef_,inTest):
    
    computedLabels = [predictOneSample(intercept_,coef_, sample) for sample in inTest]
    return computedLabels

def regresieLogisticaFaraTool(trainInputsNormalizat,trainOutputs,testInputsNormalizat,testOutputs):
    w00,w01,w02,w03,w04 = fit(trainInputsNormalizat,trainOutputs[0])
    w10,w11,w12,w13,w14 = fit(trainInputsNormalizat,trainOutputs[1])
    w20,w21,w22,w23,w24 = fit(trainInputsNormalizat,trainOutputs[2])

    computedTestOutputs0 = predict(w00,[w01,w02,w03,w04],testInputsNormalizat)
    computedTestOutputs1 = predict(w10,[w11,w12,w13,w14],testInputsNormalizat)
    computedTestOutputs2 = predict(w20,[w21,w22,w23,w24],testInputsNormalizat)

    computedTestOutputs = [[computedTestOutputs0[i] , computedTestOutputs1[i], computedTestOutputs2[i]] for i in range(len(computedTestOutputs0))]
    labels = [0,1,2]
    computedLabels = []
    for i in range(len(computedTestOutputs0)):
        computedLabels.append(labels[computedTestOutputs[i].index(max(computedTestOutputs[i]))])

    print('computedTestOutputs: ',computedLabels,'\n')
    print('testOutputs: ',testOutputs,'\n')
    eroareFaraTool(computedLabels,testOutputs)
    eroareaCuTool(computedLabels,testOutputs)

def eroareFaraTool(computedTestOutputs,testOutputs):
    error = 0.0
    for t1, t2 in zip(computedTestOutputs, testOutputs):
        if (t1 != t2):
            error += 1
    error = error / len(testOutputs)
    print("classification error (manual): ", error)

def minMax(trainInputs,testInputs):
    minn,maxx = min(trainInputs), max(trainInputs)
    dateNormalizateTrain = [ (x - minn) / (maxx - minn) for x in trainInputs]
    dateNormalizateTest = [ (x - minn) / (maxx - minn) for x in testInputs]
    return dateNormalizateTrain,dateNormalizateTest

def myNormalisation(trainInputs, testInputs):
    list0Train = [x[0] for x in trainInputs]
    list1Train = [x[1] for x in trainInputs]
    list2Train = [x[2] for x in trainInputs]
    list3Train = [x[3] for x in trainInputs]

    list0Test = [x[0] for x in testInputs]
    list1Test = [x[1] for x in testInputs]
    list2Test = [x[2] for x in testInputs]
    list3Test = [x[3] for x in testInputs]

    normalizare0train,normalizare0Test = minMax(list0Train,list0Test)
    normalizare1train,normalizare1Test = minMax(list1Train,list1Test)
    normalizare2train,normalizare2Test = minMax(list2Train,list2Test)
    normalizare3train,normalizare3Test = minMax(list3Train,list3Test)

    dateNormalizateTrain = [[normalizare0train[i],normalizare1train[i],normalizare2train[i],normalizare3train[i]] for i in range(len(normalizare1train)) ]
    dateNormalizateTest= [[normalizare0Test[i],normalizare1Test[i],normalizare2Test[i],normalizare3Test[i]] for i in range(len(normalizare1Test)) ]
    
    return dateNormalizateTrain,dateNormalizateTest


def main():
    inputs, outputs = loadData('iris.data')
    trainInputs,trainOutputs,testInputs,testOutputs = splitData(inputs,outputs)
    # trainInputsNormalizat,testInputsNormalizat = normalisation(trainInputs,testInputs)

    trainInputsNormalizat,testInputsNormalizat = myNormalisation(trainInputs,testInputs)
    testOutputs = transformareaOutputs(testOutputs)
    trainOutputs = transformareaOutputs(trainOutputs)

    print('\n ------------------REGRESIE CU TOOL ----------\n')
    regresieLogisticaCuTool(trainInputsNormalizat,trainOutputs,testInputsNormalizat,testOutputs)
    print('\n-------------------REGRESIE FARA TOOL---------\n')
    trainOutputs0 = [1 if trainOutputs[i] == 0 else 0 for i in range(len(trainOutputs))]
    trainOutputs1 = [1 if trainOutputs[i] == 1 else 0 for i in range(len(trainOutputs))]
    trainOutputs2 = [1 if trainOutputs[i] == 2 else 0 for i in range(len(trainOutputs))]
    xx = [trainOutputs0,trainOutputs1,trainOutputs2]
    regresieLogisticaFaraTool(trainInputsNormalizat,xx,testInputsNormalizat,testOutputs)

main()