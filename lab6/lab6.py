# compute the prediction error

from math import sqrt
import numpy as np

#Suma diferențelor absolute între valorile reale și cele calculate
def sumMAE(r1, c1):
    suma = 0.0
    for i in range(len(r1)):
    #realOutputs-computedOutputs
        suma += abs(r1[i] - c1[i])
    return suma


#Suma pătratelor diferențelor între valorile reale și cele calculate
def sumRMSE(r1, c1):
    suma = 0.0
    for i in range(len(r1)):
        #realOutputs-computedOutputs
        suma += (r1[i] - c1[i]) ** 2
    return suma

def predictionError(realOutputs,computedOutputs):
    # MAE (mean absolute error)
    errorL1 = sum(sumMAE(r,c) for r, c in zip(realOutputs, computedOutputs)) / len(realOutputs)
    print('Error (L1): ', errorL1)

    # RMSE(root mean square error)
    
    errorL2 = sqrt(sum(sumRMSE(r,c) ** 2 for r, c in zip(realOutputs, computedOutputs)) / len(realOutputs))
    print('Error (L2): ', errorL2)




# version 1 - using the sklearn functions
def evalClassificationV1(realLabels, computedLabels, labelNames):
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

    cm = confusion_matrix(realLabels, computedLabels, labels = labelNames)
    acc = accuracy_score(realLabels, computedLabels)
    precision = precision_score(realLabels, computedLabels, average = None, labels = labelNames)
    recall = recall_score(realLabels, computedLabels, average = None, labels = labelNames)
    return acc, precision, recall 



def multiTarget():
    realOutputs = [[1, 2, 3], [6, 7, 8], [10, 20, 30], [5, 15, 25]]
    computedOutputs = [[1.5, 2, 2.8], [3, 7.4, 8], [10.2, 19, 30.1], [4.9, 15, 25]]
    print('Prediction error:')
    predictionError(realOutputs, computedOutputs)
    print('')

def multiClasa():
    realLabels = ["apple", "pear", "pear", "grape", "apple", "grape"]
    labels = ["apple", "pear", "grape"]
    computedLabels = ["apple", "pear", "grape", "pear", "apple", "grape"]
    acc, precision, recall = evalClassificationV1(realLabels, computedLabels, labels)

    print("Accuracy: " + str(acc) + " Precision: " + str(precision) + " Recall: " + str(recall))




def main():
    print('\n')
    multiTarget()
   
    print('\n')
    multiClasa()
    
   

main()

