import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

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

def plotDataHistogram(x, variableName):
    plt.hist(x, 10)
    plt.title('Histogram of ' + variableName)
    plt.show()

def plots(inputs,outputs):
    plotDataHistogram(inputs[0],'PIB')
    plotDataHistogram(inputs[1],'freedom')
    plotDataHistogram(outputs,'happiness')

    ax = plt.axes(projection='3d')
    ax.scatter3D(inputs[0], inputs[1], outputs, c=outputs, cmap='Blues')
    plt.title('PIB & freedom vs. happiness')
    plt.xlabel('PIB')
    plt.ylabel('freedom')
    ax.set_zlabel('happiness')
    plt.show()


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

def trainingStep(trainInputs,trainOutputs):
    # training the model by using the training inputs and known training outputs
    xx = [[elem1,elem2] for elem1,elem2 in zip(trainInputs[0],trainInputs[1])]
    regressor = linear_model.LinearRegression()
    regressor.fit(xx, trainOutputs)
    w0, w1, w2 = regressor.intercept_, regressor.coef_[0], regressor.coef_[1]
    print('w0 = ',w0,'w1=',w1,'w2=',w2)
    return w0,w1,w2

def plotTestAndTrainData(trainInputs,trainOutputs,testInputs,testOutputs):
    ax = plt.axes(projection='3d')
    ax.scatter3D(trainInputs[0], trainInputs[1], trainOutputs, c=trainOutputs, cmap='Greens',label="Training data")
    ax.scatter3D(testInputs[0], testInputs[1], testOutputs, c=testOutputs, cmap='Blues',label="Testing data")
    plt.title('train and test data')
    plt.xlabel('PIB')
    plt.ylabel('freedom')
    ax.set_zlabel('happiness')
    plt.legend()
    plt.show()

def plotTrainDataAndModel(xref,yref,zref,trainInputs,trainOutputs):
    ax = plt.axes(projection='3d')
    ax.scatter3D(xref, yref, zref, c=zref, cmap='Reds',label="model")
    ax.scatter3D(trainInputs[0], trainInputs[1], trainOutputs, c=trainOutputs, cmap='Blues',label="Training data")
    plt.title('train data and model')
    plt.xlabel('PIB')
    plt.ylabel('freedom')
    ax.set_zlabel('happiness')
    plt.legend()
    plt.show()

def plotComputedAndRealData(testInputs,testOutputs,computedTestOutputs):

    ax = plt.axes(projection='3d')
    ax.scatter3D(testInputs[0], testInputs[1], computedTestOutputs, c=computedTestOutputs, cmap='Greens',label="Computed")
    ax.scatter3D(testInputs[0], testInputs[1], testOutputs, c=testOutputs, cmap='Blues',label="Real")
    plt.title('computed and real ')
    plt.xlabel('PIB')
    plt.ylabel('freedom')
    ax.set_zlabel('happiness')
    plt.legend()
    plt.show()

def main():
    print("\n")
    #load data
    inputs, outputs = loadData("world-happiness-report-2017.csv",'Economy..GDP.per.Capita.','Freedom','Happiness.Score')

    #show plots of a data
    plots(inputs,outputs)

    # split data into training data (80%) and testing data (20%)
    trainInputs,trainOutputs,testInputs,testOutputs = splitData(inputs,outputs)

    #plot the train and test data
    plotTestAndTrainData(trainInputs,trainOutputs,testInputs,testOutputs)   

    #plot the model
    #datele sintetice
    noOfPoints = 1000
    xref = []
    val = min(trainInputs[0])
    step = (max(trainInputs[0]) - min(trainInputs[0])) / noOfPoints
    for i in range(1, noOfPoints):
        xref.append(val)
        val += step

    yref = []
    val = min(trainInputs[1])
    step = (max(trainInputs[1]) - min(trainInputs[1])) / noOfPoints
    for i in range(1, noOfPoints):
        yref.append(val)
        val += step


    # training/learning step
    w0,w1,w2 = trainingStep(trainInputs,trainOutputs)
    # w0,w1,w2 = trainingStepWithoutTool(trainInputs,trainOutputs)

    zref = [w0 + w1*elem1 + w2*elem2 for elem1,elem2 in zip(xref,yref)]

    plotTrainDataAndModel(xref,yref,zref,trainInputs,trainOutputs)


    #makes predictions for test data
    #makes predictions for test data (by tool)
    computedTestOutputs = [w0 + w1 * elem1 + w2 * elem2 for elem1,elem2 in zip(testInputs[0],testInputs[1])]
    
    #plot the computedTestOutputs
    plotComputedAndRealData(testInputs,testOutputs,computedTestOutputs)

    # compute the differences between the predictions and real outputs
    error = 0.0
    for t1, t2 in zip(computedTestOutputs, testOutputs):
        error += (t1 - t2) ** 2
    error = error / len(testOutputs)
    print("prediction error (manual): ", error)

  
    print('\n')

        
        
        

    


main()

