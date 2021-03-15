from GA import GA
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt 
import warnings 
import math


from utils import readDataFromFile, modularity, printData

#Pun aceeasi valoare intr un vector pe pozitia nodurilor care fac parte dintr o comunitate



#reteaua mea din care citesc
net = readDataFromFile('polbooks.gml')

#parametri genetic algorithm
#300 de cromozomi in populatie 
gaParam = {"popSize": 300, "noGen": 15, "network": net}
problParam = {'function': modularity, 'retea': net}


def afisare_graf(network):
    warnings.simplefilter('ignore')

    A=np.matrix(network["matrix"])
    G=nx.from_numpy_matrix(A)
    pos = nx.spring_layout(G)  # compute graph layout
    plt.figure(figsize=(8, 8))  # image is 8 x 8 inches 
    nx.draw_networkx_nodes(G, pos, node_size=600, cmap=plt.cm.RdYlBu)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    plt.show()

def afisare_comunitati(network,bestChromo):
    communities = [1,2,1,2,1, 1]

    A=np.matrix(network["matrix"])
    partition=bestChromo.repres
    G=nx.from_numpy_matrix(A)
    pos = nx.spring_layout(G)  # compute graph layout
    plt.figure(figsize=(8, 8))  # image is 8 x 8 inches 
    nx.draw_networkx_nodes(G, pos, node_size = 600, cmap = plt.cm.RdYlBu, node_color=list(bestChromo.repres))
    nx.draw_networkx_edges(G, pos, alpha = 0.3)
    plt.show()

def main():
    ga = GA(gaParam, problParam)
    ga.initialisation()
    ga.evaluation()

  
    i = 1
    while ( i <= gaParam['noGen']):
        
        ga.oneGeneration()
        bestChromo = ga.bestChromosome()
     
        print('Cea mai buna solutie in generatia: ' + str(i) + ' cu fitnessul f(x) = ' + str(
            bestChromo.fitness) + ' ' + 'numarul comunitatilor fiind:' + str(len(printData(bestChromo.repres))))
        i += 1
    print(bestChromo.repres)

    afisare_comunitati(gaParam['network'],bestChromo)
    # afisare_graf(gaParam['network'])

main()
