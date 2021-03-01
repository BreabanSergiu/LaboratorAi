
#input - vector -> list , k -> int
#complexity - O(n * log n)
def Kelement(vector,k):
    vector.sort(reverse=True)
    return vector[k-1]



def test():
    assert(Kelement([1,2,3],2) ==2 )
    assert(Kelement([1,2,3,4],2) == 3 )
    assert(Kelement([7,4,6,3,9,1],2) == 7)

test()
    