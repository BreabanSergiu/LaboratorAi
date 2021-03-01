
import numpy


#input l - list , n - int
#complexity - O(n)
def function(l,n):
    caracteristic = [0]*(n+1)
    for i in range(n):
        caracteristic[l[i]] += 1
        if caracteristic[l[i]] == 2:
            return l[i]
    return -1


#input l - list , n - int
#complexity - theta(n)
def function2(l,n):
    return sum(l) - (n-1)*n/2


def test():
    assert(function([1,2,3,4,2],5) == 2)
    assert(function([1,2,3,4,3],5) == 3)
    assert(function2([1,2,3,3],4) == 3)
    assert(function2([1,2,1,3],4) == 1)


test()

