from utils import *
from GA import GA

import math

def readFromFile(filename):
    matrix = []
    numberOfCities = 0
  
    f = open(filename,"r")
    numberOfCities = int(f.readline())
    
    for i in range(numberOfCities):
        line = f.readline().split(",")
        list = []
        for city in line:
            list.append(int(city))
        matrix.append(list)

    
    f.close()
    return numberOfCities,matrix

def distanta(Xa,Xb,Ya,Yb):
    return math.sqrt(pow((Xb - Xa),2) + pow((Yb - Ya),2))

def readFromTsp(filename):
    matrix = []
    
    numberOfCities = 0
    f = open(filename,"r")
    x =[]
    y =[]
    n=[]
    line = f.readline()
    while (line):
        
        numberOfCities += 1
        
        lin2 = line.split(",")
        
        n.append(float(lin2[0]))
        x.append(float(lin2[1]))
        y.append(float(lin2[2]))
        line = f.readline()
        
    for i in range(len(x)):
        linii = []
        for j in range(len(y)):
            linii.append(int(distanta(x[i],x[j],y[i],y[j])))
        matrix.append(linii)


    return numberOfCities,matrix
      
    
       




numberOfCities,net = readFromTsp('bays29.tsp')

#parametri genetic algorithm
#300 de cromozomi in populatie 
gaParam = {"popSize": 600, "noGen": 400, "network": net}
problParam = {'function': fitness, 'retea': net, 'noNodes': numberOfCities}


def main():
    
    ga = GA(gaParam, problParam)
    ga.initialisation()
    ga.evaluation()

    seed(1)
    i = 1
    while ( i <= gaParam['noGen']):
        
        ga.oneGeneration()
        bestChromo = ga.bestChromosome()
        l=[]
        for c in bestChromo.repres:
            l.append(c + 1)
        
        print("path= "+str(l))
        print(" f(x)= " +str(bestChromo.fitness))

        i += 1
    

 


main()
