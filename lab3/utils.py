from random import randint
import networkx as nx

def randomValue(number1, number2):
    return randint(number1, number2)

#la vectorul generat aleator pe pozitia y pune valoarea din vector de pe poz y
#x si y sunt doua noduri din matricea de adiacenta intre care exista muchie
#practic vectorul sa aiba aceeasi valoare pe indexul x si y daca exista muchie intre x si y
def reprezentare(network):
    #generam numere random 0 si numarul de noduri din retea pentru fiecare nod 
    repres = [randomValue(0, network['noNodes'] - 1) for _ in range(network['noNodes'])] 
    
    for i in range(len(repres)):
        for j in range(len(repres)):
            if network['matrix'].item(i, j) == 1:
                repres[j] = repres[i]
    return repres


#citeste data din fisierul gml cu ajutorul networks
def readDataFromFile(filename):
    G = nx.read_gml(filename, label='id')
    network = {}
    
    network['degrees'] = [val for (node, val) in G.degree()]#gradele fiecarui nod din retea
    network['graph'] = G
    network['noEdges'] = G.number_of_edges()
    network['noNodes'] = G.number_of_nodes()
    network['matrix'] = nx.adjacency_matrix(G).todense()#returneaza o matrice de adiacenta a grafului
    
    return network


#aici calculex fitnessul fiecarui cromozon
#fitnesul reprezinta calitatea cromozomului
def modularity(repres, network):
    noNodes = network['noNodes']
    noEdges = network['noEdges']
    degrees = network['degrees']
    matrix = network['matrix']

    M = 2 * noEdges
    Q = 0.0
    for index in range(0, noNodes):
        for secondIndex in range(0, noNodes):
            if (repres[index] == repres[secondIndex]):
                Q += (matrix.item(index, secondIndex) - degrees[index] * degrees[secondIndex] / M)
               
    return Q * 1 / M

net = readDataFromFile('polbooks.gml')
problParam = {'function': modularity, 'retea': net}

def printData(x):
    vector = []
    for i in range(0, problParam['retea']['noNodes']+1):
        vector.append([])
    for i in range(0, net['noNodes']):
        vector[x[i]].append(i + 1)
    j = 0
    #elimin toate listele in care nu sunt adaugate noduri
    while j < len(vector):
        if vector[j] == []:
            vector.pop(j)
        else:
            j += 1
    return vector