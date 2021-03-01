
#input - n -> int
#output - l -> list
#complexity - theta(n)
def zecimalToBinary(n):
 
    l=[]
    nr = "1"
    l.append(nr)
    i = 1

    while(len(l)<n):
      
        l.append(nr+"0")
        l.append(nr+"1")

        nr = l[i]
        i += 1
       

    return l
 
def test():
    assert(zecimalToBinary(7)[0] == "1")
    assert(zecimalToBinary(7)[1] == "10")
    assert(zecimalToBinary(7)[2] == "11")
    assert(zecimalToBinary(7)[3] == "100")



test()
