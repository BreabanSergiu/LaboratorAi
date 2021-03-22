from random import randint, seed


def generateARandomPermutation(n):
    perm = [i for i in range(n)]
    pos1 = randint(0, n - 1)
    pos2 = randint(0, n - 1)
    perm[pos1], perm[pos2] = perm[pos2], perm[pos1]
    return perm



def fitness(reprezentar,network):
    costPath = 0
  
    for i in range(len(reprezentar) - 1):
        costPath += network[reprezentar[i]][reprezentar[i+1]]

    costPath += network[reprezentar[len(reprezentar) - 1]][reprezentar[0]] # calculez si drumul care se intoarce de unde a pornit
    return costPath



   