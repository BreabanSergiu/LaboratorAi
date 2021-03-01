
#input - l -> list
#complexity - theta(n)
def function(l):
    vectorCaracteristic = [0] * (len(l) + 1)
    maxx = 0
    element = 0
    for i in range(len(l)):
        vectorCaracteristic[l[i]] += 1
        if vectorCaracteristic[l[i]] > maxx:
            maxx = vectorCaracteristic[l[i]]
            element = l[i]

    if maxx > len(l)/2:
        return element
    else:
        return -1

def test():
    assert(function([2,8,7,2,2,5,2,3,1,2,2]) == 2)
    assert(function([2,2,3,3,3]) == 3)
    assert(function([2,3,1,1]) == -1)

test()
