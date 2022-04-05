import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


#read the ground state component
xfile = open('psi_coul.txt','r')
psi = []
for line in xfile:
    a,b= line.strip().strip('()').split(',')
    psi.append(complex(float(a),float(b)))
 

psi = np.asarray(psi)  

print(psi)
#unpack the value of momentum for the two electrons and the number of photon in the basis
j , k1x , k1y , k2x, k2y, npp, npm = np.loadtxt('fort.61',unpack=True) 



s = 0 #variable for the spin of the electron system
m = 3

if (s==0):
    dimel = int(0.5*((2*m+1)**2)*((2*m+1)**2+1)) #read the dimension of electrons' Hilbert space
if (s==1):
    dimel = int(0.5*((2*m+1)**2)*((2*m+1)**2-1))


nph = 1 #read the dimension of the photons' hilbert space
dimph = nph**2

k1 = np.array([k1x,k1y]) #construct the first electron momentum array
k2 = np.array([k2x,k2y]) #construct the second electron momentum array


def rho_mat(psi0,dimph,dimel): 

    """
    reduced density matrix for the two electrons
    """

    rho = np.zeros((dimel,dimel)) #reduced density matrix of the form dimel *dimel
    for k in range (0,dimph):
        for i in range (0,dimel):
            s = k*dimel + i
            for j in range (0,dimel):
                l = k*dimel + j
                print(s,l,i,j)
                rho[i][j] = rho[i][j] + psi0[s]*np.conj(psi0[l])

    return rho



print(rho_mat(psi,dimph,dimel))
eig = np.zeros(dimel) #array that contains eigenvalue of the reduced density matrix
vect = np.zeros((dimel,dimel)) #matrix for the eigenvector of the reduced density matric
eig,vect = np.linalg.eigh(rho_mat(psi,dimph,dimel)) #exact diagonalization of the density matrix


print(eig)
def entanglement_entropy(eig):
    s = 0
    for i in range (0,len(eig)):
        k = eig[i]
        print(k)
        if(k != 0):
            s = s - np.log(eig[i])*eig[i]
    return s        

s = entanglement_entropy(eig)
