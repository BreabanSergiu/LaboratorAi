import math
from plot import plot
from classes import ACO, Graph

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
        lin2 = line.split(" ")
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

rank,cost_matrix = readFromTsp("./testU.txt")

def main():
    
    
    number_of_ant = 10  #numarul de furnici
    generations = 100  #numarul de generatii
    alpha = 1.0 #controleaza importanta urmei (cate furnici au mai trecut pe muchia respectiva)
    beta = 10.0 #controleaza importanaa vizibilitatii (cat de aproape se afla următorul oras)
    rho = 0.5 #coeficient de degradare a feronomului
    q = 10 #intensitatea(cantitatea) unui feromon lasata de o furnica

    aco = ACO(number_of_ant, generations, alpha, beta, rho, q)
    graph = Graph(cost_matrix, rank)

    path, cost = aco.solve(graph)
    print('cost: {}, path: {}'.format(cost, path))
    


main()