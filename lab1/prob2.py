
import math



#input Xb,Xa,Ya,Yb 
#output - distanta euclidiana 
#complexitate - theta(1)
def distanta(Xa,Xb,Ya,Yb):
    return math.sqrt(pow((Xb - Xa),2) + pow((Yb - Ya),2))


def test():
    assert(distanta(1,5,4,1) == 5)
    assert(distanta(4,5,4,1) == 4)



test()
