import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


#read the ground state component
xfile = open('psi_free.txt','r')
psi = []
for line in xfile:
    a,b= line.strip().strip('()').split(',')
    psi.append(complex(float(a),float(b)))
 

psi = np.asarray(psi)  

#print(psi)
#unpack the value of momentum for the two electrons and the number of photon in the basis
j , k1x , k1y , k2x, k2y, npp, npm = np.loadtxt('fort.61',unpack=True) 



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



def delta(k,p):
    d = 0
    if(np.array_equal(k,p)):
        d = 1
    return d


vec1 = np.array([3,0]) #k1
vec2 = np.array([3,0]) #k1'
vec3 = np.array([3,0]) #k2
vec4 = np.array([3,0]) #k2'


def skew_density(v1,v2,v3,v4,x,y):

    n = 0
    n = (n + delta(v1,v2)*np.exp(1j*(v3[0]-v4[0])*x)*np.exp(1j*(v3[1]-v4[1])*y) + 
        delta(v3,v4)*np.exp(1j*(v1[0]-v2[0])*x)*np.exp(1j*(v1[1]-v2[1])*y) - 
        delta(v1,v4)*np.exp(1j*(v3[0]-v2[0])*x)*np.exp(1j*(v3[1]-v2[1])*y) - 
        delta(v2,v3)*np.exp(1j*(v1[0]-v4[0])*x)*np.exp(1j*(v1[1]-v4[1])*y))
    

    return n




def sym_density(v1,v2,v3,v4,x,y):
    n= 0
    if(not(np.array_equal(v1,v3)) and not(np.array_equal(v2,v4))): 
        
        n = delta(v1,v2)*np.exp(1j*(v3[0]-v4[0])*x)*np.exp(1j*(v3[1]-v4[1])*y) + \
        delta(v3,v4)*np.exp(1j*(v1[0]-v2[0])*x)*np.exp(1j*(v1[1]-v2[1])*y) - \
        delta(v1,v4)*np.exp(1j*(v3[0]-v2[0])*x)*np.exp(1j*(v3[1]-v2[1])*y) - \
            delta(v2,v3)*np.exp(1j*(v1[0]-v4[0])*x)*np.exp(1j*(v1[1]-v4[1])*y)

    if( np.array_equal(v1,v3) and not(np.array_equal(v2,v4))): #checked
        
        
        n = delta(v1,v2)*np.sqrt(2)*np.exp(1j*(v1[0]-v4[0])*x)*np.exp(1j*(v1[1]-v4[1])*y) +\
            delta(v4,v1)*np.sqrt(2)*np.exp(1j*(v1[0]-v2[0])*x)*np.exp(1j*(v1[1]-v2[1])*y)  

    if(np.array_equal(v2,v4) and not(np.array_equal(v1,v3))):#checked
       
        n = delta(v2,v4)*np.sqrt(2)*np.exp(1j*(v1[0]-v4[0])*x)*np.exp(1j*(v1[1]-v4[1])*y) +\
            delta(v1,v4)*np.sqrt(2)*np.exp(1j*(v1[0]-v3[0])*x)*np.exp(1j*(v1[1]-v3[1])*y)

    if((np.array_equal(v1,v3)) and  np.array_equal(v2,v4)):
        print('yes')
        n = 2*delta(v1,v2)*np.exp(1j*(v3[0]-v4[0])*x)*np.exp(1j*(v3[1]-v4[1])*y)        
    return n

'''
definition of the axis
'''
x = np.linspace(0,3,1000) # value of the coordinate along 1 axis
y = np.zeros(1000)

######################################################
#    density function for the 2 electron problem     #   
######################################################

def density(psi0,k1x,k2x,k1y,k2y,dimtot,x,y):


    """
    density of the two electrons 
    
    depending on their spin

    """
    if(s==1):
            
        n = np.zeros(len(x))
        for i in range (dimtot):
            for j in range (dimtot):
                k1 = np.array([k1x[i],k1y[i]])

                k2 = np.array([k2x[i],k2y[i]])

                q1 = np.array([k1x[j],k1y[j]])

                q2 = np.array([k2x[j],k2y[j]])


                #if(npp[i]==npp[j] and  nmm[i] == nmm[j]): #the g function is diagonal in the photon number

                n = n + sym_density(k1,q1,k2,q2,x,y)*np.conj(psi0[j])*psi0[i]
        return n
    if(s==0):
        n = np.zeros(len(x))
        for i in range (dimtot):
            for j in range (dimtot):
                k1 = np.array([k1x[i],k1y[i]])

                k2 = np.array([k2x[i],k2y[i]])

                q1 = np.array([k1x[j],k1y[j]])

                q2 = np.array([k2x[j],k2y[j]])


                #if(npp[i]==npp[j] and  nmm[i] == nmm[j]): #the g function is diagonal in the photon number

                n = n + skew_density(k1,q1,k2,q2,x,y)*np.conj(psi0[j])*psi0[i]
        return n


################################################
###          plot of density function        ###
################################################

fig = plt.figure(figsize = (8, 6), dpi=92)
ax = plt.subplot(1, 1, 1)
ax.set_title('density')
ax.grid(True, color = 'silver')

#ax.legend()
ax.title.set_fontsize(18)
ax.xaxis.label.set_fontsize(18)
ax.yaxis.label.set_fontsize(18)
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.xaxis.set_minor_locator(AutoMinorLocator(5))

ax.set_xlabel(r'$2\pi r/L$ ')
ax.set_ylabel(r'n(r)')


ax.plot(x,density(psi,k1x,k1y,k2x,k2y,dimtot,x,y))
plt.show()
#fig.savefig('pair_correlation.pdf')