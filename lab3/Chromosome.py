from random import randint
from utils import reprezentare, randomValue


#reprezinta o solutie candidat
class Chromosome:
    def __init__(self, problParam = None):
        self.__problParam = problParam
        self.__repres = reprezentare(problParam['retea'])
        self.__fitness = 0.0

    @property
    def repres(self):
        return self.__repres

    
    @property
    def fitness(self):
        return self.__fitness

    @repres.setter
    def repres(self, l = []):
        self.__repres = l

    @fitness.setter
    def fitness(self, fit = 0.0):
        self.__fitness = fit

    #pun valoarea lui p1 daca in p2 nu se afla pe pozitia curenta valoarea "value" altfel 
    # pun valoarea random din p2

    def crossover(self, c):
        r = randint(0, len(self.__repres) - 1)
        newrepres = []
        for i in range(r):
            newrepres.append(self.__repres[i])
        for i in range(r, len(self.__repres)):
            newrepres.append(c.__repres[i])
        offspring = Chromosome(c.__problParam)
        offspring.repres = newrepres
        return offspring


    def mutation(self):
        position = randint(0, self.__problParam['retea']['noNodes']-1)#aleg o pozitie random
        #pun in repres pe acea pozitie random o valoare random intre 0 si nrDeNoduri din retea
        self.__repres[position] = self.__repres[randomValue(0, self.__problParam['retea']['noNodes'] - 1)]

    def __str__(self):
        return '\nChromo: ' + str(self.__repres) + ' has fit: ' + str(self.__fitness)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, c):
        return self.__repres == c.__repres and self.__fitness == c.__fitness