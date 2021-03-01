
#input l -> str
#complexity O(n * log n )
def function(l):
    rez = l.split(" ")  
    rez.sort()
    words = []
    n = len(rez)
    for i in range(n):
        if i > 0 and i < n - 1:
            if rez[i] != rez[i + 1] and rez[i] != rez[i - 1]:
                words.append(rez[i])
        if i == 0:
            if rez[i] != rez[i + 1]:
                words.append(rez[i])
        if i == n - 1 :
            if rez[i] != rez[i - 1]:
                words.append(rez[i])
  
    return words


#input l -> str
#complexity theta(n)
def function2(l):
    dict = {}
    words = []
    rez = l.split(" ") 
    for i in range(len(rez)):
        dict[rez[i]] = 0
    for i in range(len(rez)):
        dict[rez[i]] += 1
    for i in range(len(rez)):
        if dict[rez[i]] == 1:
            words.append(rez[i])

    return words



def test():
    assert(function("ana are ana are mere rosii ana") == ["mere","rosii"])
    assert(function("ana are ana ana") == ["are"])
    assert(function("ana ana ana") == [])
    assert(function2("ana are ana are mere rosii ana") == ["mere","rosii"])
    assert(function2("ana are ana ana") == ["are"])
    assert(function2("ana ana ana") == [])



test()
