import csv
from sklearn import linear_model
import numpy as np
from sklearn.metrics import mean_squared_error
import random

def loadData(fileName,inputName1,inputName2,outputName):
    data = []
    dataNames = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                dataNames = row
            else:
                data.append(row)
            line_count += 1
    selectedVariable1 = dataNames.index(inputName1)
    selectedVariable2 = dataNames.index(inputName2)
    inputs1 = [float(data[i][selectedVariable1]) for i in range(len(data))] #happiness
    inputs2 = [float(data[i][selectedVariable2]) for i in range(len(data))] #freedom
    inputs=[inputs1,inputs2]
    selectedOutput = dataNames.index(outputName)
    outputs = [float(data[i][selectedOutput]) for i in range(len(data))]
    
    return inputs, outputs




def splitData(inputs,outputs):
    np.random.seed(5)
    indexes = [i for i in range(len(outputs))]
    trainSample = np.random.choice(indexes, int(0.8 * len(outputs)), replace = False)
    testSample = [i for i in indexes  if not i in trainSample]

    trainInputs1 = [inputs[0][i] for i in trainSample]
    trainInputs2 = [inputs[1][i] for i in trainSample]
    
    trainInputs = [trainInputs1,trainInputs2]

    trainOutputs = [outputs[i] for i in trainSample]

    testInputs1 = [inputs[0][i] for i in testSample]
    testInputs2 = [inputs[1][i] for i in testSample]

    testInputs = [testInputs1,testInputs2]

    testOutputs = [outputs[i] for i in testSample]

    return trainInputs,trainOutputs,testInputs,testOutputs



def gradientCuTool(trainInputs,trainOutputs,testOutputs,testInputs):

    xx = [[elem1,elem2] for elem1,elem2 in zip(trainInputs[0],trainInputs[1])]
    regressor = linear_model.SGDRegressor(alpha = 0.01,max_iter=100)
    for i in range(1000):
        regressor.partial_fit(xx,trainOutputs)
    w0, w1, w2 = regressor.intercept_[0], regressor.coef_[0], regressor.coef_[1]
    print("f(x) = " , w0, ' + ',w1,'* x1',' + ',+w2,'* x2')
    prezicereaEroriiManual(w0,w1,w2,testOutputs,testInputs)


def prezicereaEroriiManual(w0,w1,w2,testOutputs,testInputs):
    computedTestOutputs = [w0 + w1 * el1 + w2 * el2 for el1,el2 in zip(testInputs[0],testInputs[1])]
    error = 0.0
    for t1, t2 in zip(computedTestOutputs, testOutputs):
        error += (t1 - t2) ** 2
    error = error / len(testOutputs)
    print("prediction error (manual): ", error)

def prezicereaEroriiManualUnivariate(w0,w1,testOutputs,testInputs):
    computedTestOutputs = [w0 + w1 * el1  for el1 in testInputs]
    error = 0.0
    for t1, t2 in zip(computedTestOutputs, testOutputs):
        error += (t1 - t2) ** 2
    error = error / len(testOutputs)
    print("prediction error (manual): ", error)

def regresieUnivariate(x,y,testInputs,testOutputs,learningRate = 0.001, nrEpoci = 1000):
    # f(x) = w0 + w1 * x1 
    coef = [random.random() for _ in range(len(x[0]) + 1 )]
    
    #EROAREA se calculeaza pt fiecare exemplu de antrenament
    for epoca in range(nrEpoci):
        suma = 0
        for i in range(len(x)):
            computedTestOutputs = eval(coef,x[i])
            err = computedTestOutputs - y[i]
            suma = suma + err
        suma = suma / len(x)
        for i in range(len(x)):
            for j in range(0, len(x[0])):
                coef[j] = coef[j] - learningRate * suma * x[i][j]
            coef[len(x[0])] = coef[len(x[0])] - learningRate * suma * 1
        
    w0 = coef[1]
    w1 = coef[0]

    print("f(x) = " , w0, ' + ',w1,'* x1')
    prezicereaEroriiManualUnivariate(w0,w1,testOutputs,testInputs)


def regresieMultivariata(x,y,testInputs,testOutputs,learningRate = 0.001, nrEpoci = 1000):
    # f(x) = w0 + w1 * x1 + w2 * x2
    coef = [random.random() for _ in range(len(x[0]) + 1 )]
    
    for epoca in range(nrEpoci):
        suma = 0
        for i in range(len(x)):
            computedTestOutputs = eval(coef,x[i])
            err = computedTestOutputs - y[i]
            suma = suma + err
        suma = suma / len(x)
        for i in range(len(x)):
            for j in range(0, len(x[0])):
                coef[j] = coef[j] - learningRate * suma * x[i][j]
            coef[len(x[0])] = coef[len(x[0])] - learningRate * suma * 1
        
    w0 = coef[2]
    w1 = coef[0]
    w2 = coef[1]

    print("f(x) = " , w0, ' + ',w1,'* x1',' + ',w2,'* x2')
    prezicereaEroriiManual(w0,w1,w2,testOutputs,testInputs)

def eval(coef, x):
    y = coef[-1]
    for j in range(len(x)):
        y += coef[j] * x[j]
    return y 


def normalizareaDatelor(inputs):
    minn,maxx = min(inputs), max(inputs)
    dateNormalizate = [ (x - min(inputs)) / (max(inputs) - min(inputs)) for x in inputs]
    
    print(dateNormalizate)
    return minn,maxx

def normalizareaDatelorDeTest(inputs,minn,maxx):
    dateNormalizate = [ (x - minn) / (maxx - minn) for x in inputs]
    
    print(dateNormalizate)

def main():
    
    inputs, outputs = loadData("world-happiness-report-2017.csv",'Economy..GDP.per.Capita.','Freedom','Happiness.Score')
    trainInputs,trainOutputs,testInputs,testOutputs = splitData(inputs,outputs)
    print('\n ---------------Gradient cu tool-----------------')
    gradientCuTool(trainInputs,trainOutputs,testOutputs,testInputs)
    print('\n---------------Regresie univariata-------------')
    xx = [[x] for x in trainInputs[0]]
    xTestInputsUnivariate = [x for x in testInputs[0]]
    regresieUnivariate(xx,trainOutputs,xTestInputsUnivariate,testOutputs)
    print('\n--------------Regresie multi-variata---------')
    xx = [[elem1,elem2] for elem1,elem2 in zip(trainInputs[0],trainInputs[1])]
    xTestInputsMultivariate= [[elem1,elem2] for elem1,elem2 in zip(testInputs[0],testInputs[1])]
    regresieMultivariata(xx,trainOutputs,xTestInputsMultivariate,testOutputs)
    print('\n-------------------Normalizarea datelor------------------')
    print("\n------------Freedom--------------\n")
    minn , maxx = normalizareaDatelor(trainInputs[0])
    print('\n')
    normalizareaDatelorDeTest(testInputs[0],minn,maxx)
    print('\n------------PIB----------------\n')
    normalizareaDatelor(trainInputs[1])
    print('\n')
    normalizareaDatelorDeTest(testInputs[1],minn,maxx)



    print('\n')



main()