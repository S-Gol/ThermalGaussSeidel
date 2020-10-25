import numpy as np
import plotly.graph_objects as go
import sys
def GS(a,x,b): #Gauss-Seidel iteration function for the equation a*x=b
    n = len(a)
    
    for j in range(0,n):
        v=b[j]
        
        for i in range(0,n):
            if(j!=i):
                v-=a[i,j]*x[i]
        x[j]=v/a[j,j]
    return x
def linToSq(a, stride): #Convert the 1-D array from the Gauss-Seidel solution into a 2-d array, for image use
    height = int(len(a)/stride)
    print(str(stride)+" by "+ str(height))
    n = np.zeros([stride, height])
    for x in range(0, stride):
        for y in range (0, height):
            n[x,y]=a[getN(x,y,stride)]
    return n


#Temperatures for each of the 3 materials
knownTemps = np.array([1720,0,355])

#Grid of the physical layour
#TODO: Rework import system for ease of use
#0 exhaust, 1 blade, 2 air
baseMaterialArray = np.array([[0,0,0,0,0,0,0,0,0,0],
                          [1,1,1,1,1,1,1,1,1,1],
                          [1,1,1,1,1,1,1,1,1,1],
                          [1,1,2,2,2,2,2,2,1,1],
                          [1,1,2,2,2,2,2,2,1,1],
                          [1,1,1,1,1,1,1,1,1,1],
                          [1,1,1,1,1,1,1,1,1,1],
                          [0,0,0,0,0,0,0,0,0,0]
                         ])

ds = 0.001 #The spatial step, 1mm per block

#Subdivide 
nSub = 3
ds /= nSub
materialArray = np.zeros([nSub*baseMaterialArray.shape[0],nSub*baseMaterialArray.shape[1]])
for x in range(0,baseMaterialArray.shape[0]):
    for y in range(0,baseMaterialArray.shape[1]):
        for i in range(0,nSub):
            for j in range(0,nSub):
                materialArray[x*nSub+i, y*nSub+j]=baseMaterialArray[x,y]
materialArray = materialArray.astype(int)
#TODO: Rework these methods to allow for greater flexibility in boundary conditions
#R-values for each of the 3 materials
rCoeffs = np.array([1/(1200*ds),
                    1/25,
                    1/(250*ds)])

#Calculate the size for easier reference later
nodes = materialArray.size
shape = materialArray.shape

#Matrix of T's coefficients 
eqnArray = np.zeros([nodes,nodes])
#Matrix of constants to set the other matrix equal to 
bArray = np.zeros(nodes)
#4 directions to iterate when solving
offsets = [(1,0),(-1,0),(0,1),(0,-1)]
#Set the T-array to a default value
T=np.full(nodes,1000)
#Get a 1-D position from a 2-d set of coordinates
def getN(x,y, stride=shape[0]):
    return x+y*shape[0]

#Set up matrix in preparation for gauss-siedel
for y in np.arange(0,shape[1]):
    for x in np.arange(0,shape[0]):
        n=getN(x,y)
        #If it's a fixed temperature, set the [n,n] = 1 and b[n] = the known T, to indicate a constant 
        if(knownTemps[materialArray[x,y]]!=0):
            eqnArray[n,n]=1
            bArray[n] = knownTemps[materialArray[x,y]]
        else: # T is not known at this point
            #Sum(Q)=Sum(dT/R)=0
            #Add this equation to the matrix for each node
            #At the sides, apply adiabatic condition
            for d in offsets:
                nX = x + d[0]
                nY = y + d[1]
                if nX >= 0 and nX < shape[0] and nY >= 0 and nY < shape[1]:
                    r=rCoeffs[materialArray[x,y]]+rCoeffs[materialArray[x+d[0],y+d[1]]]
                    eqnArray[n,n]+=1/r
                    eqnArray[getN(nX, nY),n]-= 1.00/r
#Apply the gauss-seidel iteration 
nGS=50
for i in range(0,nGS):
    print ("\r"+str(100*i/nGS)+"%,", end='', flush=True)
    x = GS(eqnArray, T, bArray)
t=linToSq(x,shape[0])

fig = go.Figure(data = go.Contour(z=t))
fig.show()