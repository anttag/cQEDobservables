import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

#read the fundamental state component
xfile = open('psi_free.txt','r')
#pippo
psi = []
for line in xfile:
    a,b= line.strip().strip('()').split(',')
    psi.append(complex(float(a),float(b)))
 
#unpack the value of momentum for the two electrons and the number of photon in the basis
j , k1x , k1y , k2x, k2y, npp, npm = np.loadtxt('fort.61',unpack=True)

#dimel = 45 #read the dimension of electrons' Hilbert space
#dimph = 16 #read the dimension of the photons' hilbert space

s = 1 #variable for the spin of the electron system
m = 3

if (s==0):
    dimel = int(0.5*((2*m+1)**2)*((2*m+1)**2+1)) #read the dimension of electrons' Hilbert space
if (s==1):
    dimel = int(0.5*((2*m+1)**2)*((2*m+1)**2-1))


nph = 1 #read the dimension of the photons' hilbert space
dimph = nph**2

dimtot = dimel*dimph
k1 = np.array([k1x,k1y]) #construct the first electron momentum array
k2 = np.array([k2x,k2y]) #construct the second electron momentum array
#construct the array contining the number of photons in the basis

xi = np.zeros(dimtot)
xi[155] = 1.
def nump(psi,num,dim):
    n_avg = 0
    for i in range (dim):
        n_avg = n_avg + num[i]*np.conj(psi[i])*psi[i]
        
    return n_avg    
print(nump(psi,npp,dimtot),nump(psi,npm,dimtot))



def g2(psi,num,dim):
    g2 = 0
    for i in range (dim):
        g2 = g2 + (num[i]**2)*psi[i]*np.conj(psi[i]) - (num[i])*psi[i]*np.conj(psi[i]) 
    return g2

    
g = g2(xi,npp,dimtot)/nump(xi,npp,dimtot)**2

print(g)        

