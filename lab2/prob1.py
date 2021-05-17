
#read from file the date
def readFromFile(matrix,filename):
    
    numberOfCities = 0
    sourceCity = 0
    destinationCity = 0
    f = open(filename,"r")
    numberOfCities = int(f.readline())
    
    for i in range(numberOfCities):
        line = f.readline().split(",")
        list = []
        for city in line:
            list.append(int(city))
        matrix.append(list)

    sourceCity = int(f.readline())
    destinationCity = int(f.readline())
    f.close()
    return numberOfCities,sourceCity,destinationCity


#output - number of visited cities : int
#       -suma costurilor : int
#       - path : list
def nearestNeighbor(matrix, numberOfCities,sourceCity,destinationCity):
    visited = [0]*numberOfCities
    visited[sourceCity]=1
    path = []
    currentCity = sourceCity 
    numberOfVisitedCity = 1
    suma = 0
    path.append(currentCity+1)

    for j in range(numberOfCities-1):
        min = 99999
        index = sourceCity
        for i in range(numberOfCities):
            if min > matrix[currentCity][i] and visited[i] == 0 and matrix[currentCity][i] != 0:
                min = matrix[currentCity][i]
                index = i
    
        suma = suma + min
        numberOfVisitedCity += 1
        visited[currentCity] = 1
        
        currentCity = index
        path.append(currentCity+1)

        if currentCity == destinationCity:# daca am ajuns la destinatia dorita iesim din for
            break

    
    if(sourceCity == destinationCity):
        suma += matrix[currentCity][sourceCity]
        
    
    return numberOfVisitedCity,path,suma

    
   
def writeToFile(filename,numberOfVisitedCitiesAll,pathAll,sumaAll,numberOfVisitedCities,path,suma):
    f = open(filename, "w",encoding="utf-8")
    f.write(str(numberOfVisitedCitiesAll)+"\n")
    f.write(str(pathAll)+"\n")
    f.write(str(sumaAll)+"\n")
    f.write(str(numberOfVisitedCities)+"\n")
    f.write(str(path)+"\n")
    f.write(str(suma)+"\n")

    f.close()

   
    

def main():
    matrix = []
    filenameIn = "easy_01_tsp.txt"
    filenameOut = "out.txt"
    numberOfCities,sourceCity,destinationCity =  readFromFile(matrix,filenameIn)
    minn=32000
    pathVisited = []
    numberOfVisitedCity,path,suma = nearestNeighbor(matrix,numberOfCities,sourceCity-1,destinationCity-1)
    
    
    for i in range(1,numberOfCities+1):
        numberOfVisitedCityAll,pathAll,sumaAll = nearestNeighbor(matrix,numberOfCities,i-1,i-1)
        if( minn > suma ):
            minn = suma
            pathVisitedAll = pathAll

    writeToFile(filenameOut,numberOfVisitedCityAll,pathVisitedAll,sumaAll,numberOfVisitedCity,path,suma)

        
  
    

main()