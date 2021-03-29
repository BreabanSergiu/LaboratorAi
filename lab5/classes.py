import random


class Graph(object):
    def __init__(self, cost_matrix, rank):
        """
        :param cost_matrix: matricea de costuri
        :param rank: numarul de orase 
        initializezi feromonul cu 1/nrOrase^2
        """
        self.matrix = cost_matrix
        self.rank = rank
        self.pheromone = [[1 / (rank * rank) for j in range(rank)] for i in range(rank)]


class ACO(object): #colonie
    def __init__(self, number_of_ant, generations, alpha, beta, rho, q):
        
        self.Q = q
        self.rho = rho
        self.beta = beta
        self.alpha = alpha
        self.number_of_ant = number_of_ant
        self.generations = generations
        

    def _update_pheromone(self, graph: Graph, ants: list):
        for i, row in enumerate(graph.pheromone):
            for j, col in enumerate(row):
                graph.pheromone[i][j] *= self.rho # degradam feromonul inmultim cu coeficientul de degradare
                for ant in ants:
                    #se calculeaza cantitatea totala de feromon a tuturor furnicilor  (pag 38)
                    graph.pheromone[i][j] += ant.pheromone_delta[i][j] 

   
    def solve(self, graph):
        
        best_cost = float('inf')
        best_solution = []
        for gen in range(self.generations):
            ants = [_Ant(self, graph) for i in range(self.number_of_ant)]
            for ant in ants:
                for i in range(graph.rank - 1):
                    ant._select_next()
                ant.total_cost += graph.matrix[ant.tabu[-1]][ant.tabu[0]] #adauga si costul de la ultimul oras la primul oras
                if ant.total_cost < best_cost:
                    best_cost = ant.total_cost
                    best_solution = ant.tabu #tine minte pathul
                
                ant._update_pheromone_delta() # face update la feromonul pentru o furnicuta

            self._update_pheromone(graph, ants) #face update la feromon pentrul o generatie
            
        return best_solution, best_cost


class _Ant(object):
    def __init__(self, aco, graph):
        self.colony = aco 
        self.graph = graph
        self.total_cost = 0.0 #lungimea(costul) turului efectuat furnica
        self.tabu = []  # o lista cu orasele care au fost vizitate de furnica
        self.pheromone_delta = []  # cantitatea unitate de feromon lasata de furnica pe un drum
        self.allowed = [i for i in range(graph.rank)]  # (permise) o lista cu orase nevizitate, in care poate merge furnica 
        self.eta = [[0 if i == j else 1 / graph.matrix[i][j] for j in range(graph.rank)] for i in
                    range(graph.rank)]  #vizibilitatea din oraşul i spre oraşul j
        
        start = 0
        self.tabu.append(start)
        self.current = start
        self.allowed.remove(start)

    def _select_next(self):
        #pagina 37
        denominator = 0
        for i in self.allowed:
            denominator += self.graph.pheromone[self.current][i] ** self.colony.alpha * self.eta[self.current][i] ** self.colony.beta

        probabilities = [0 for i in range(self.graph.rank)]  # initializam toate probabilitatile cu 0
        for i in range(self.graph.rank):
            if i in self.allowed: #daca i este permis
                probabilities[i] = self.graph.pheromone[self.current][i] ** self.colony.alpha * self.eta[self.current][i] ** self.colony.beta / denominator
           
        # selectam urmatorul oras cu ruleta probabilitatii
        selected = 0
        rand = random.random()
        for i, probability in enumerate(probabilities):
            rand -= probability
            if rand <= 0:
                selected = i
                break

        self.allowed.remove(selected)
        self.tabu.append(selected)
        self.total_cost += self.graph.matrix[self.current][selected] #costul total curent
        self.current = selected

    
    def _update_pheromone_delta(self):
        self.pheromone_delta = [[0 for j in range(self.graph.rank)] for i in range(self.graph.rank)]
        for orase in range(1, len(self.tabu)):
            i = self.tabu[orase - 1]
            j = self.tabu[orase]
            #Se calculează cantitatea unitară de feromoni lăsată de a k-a furnică  pagina 40
            self.pheromone_delta[i][j] = self.colony.Q / self.total_cost