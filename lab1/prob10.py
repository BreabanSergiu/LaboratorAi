#input l:list, n,m -> int
#complexity: O(n^2)
def function(l,n,m):
    for j in range(m):
        for i in range(n):
            if(l[i][j] == 1):
                return i
    return -1


        
def test():
    assert(function([[0,0,0,1,1], [0,0,1,1,1] , [0,0,0,1,1]],3,5) == 1)
    assert(function([[0,0,0,1,1], [0,1,1,1,1] , [0,0,1,1,1]],3,5) == 1)
    assert(function([[1,1,1,1,1], [0,0,1,1,1] , [0,0,0,1,1]],3,5) == 0)



test()