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

#print(psi)
#unpack the value of momentum for the two electrons and the number of photon in the basis
j , k1x , k1y , k2x, k2y, npp, npm = np.loadtxt('fort.61',unpack=True) 



s = 1 #variable for the spin of the electron system
m = 2

if (s==0):
    dimel = int(0.5*((2*m+1)**2)*((2*m+1)**2+1)) #read the dimension of electrons' Hilbert space
if (s==1):
    dimel = int(0.5*((2*m+1)**2)*((2*m+1)**2-1))


nph = 1 #read the dimension of the photons' hilbert space
dimph = nph**2
dimtot = dimel*dimph
k1 = np.array([k1x,k1y]) #construct the first electron momentum array
k2 = np.array([k2x,k2y]) #construct the second electron momentum array


def skew_g(vec1,vec2,vec3,vec4,x,y):
    g = np.zeros(len(x))
    if(np.array_equal(vec1+vec3,vec2+vec4)):
        g = g + 2*(np.cos((vec1[0]-vec2[0])*x + (vec1[1]-vec2[1])*y) - np.cos((vec3[0]-vec2[0])*x + (vec3[1]-vec2[1])*y))

    return g

def sym_g(v1,v2,v3,v4,x,y):
    g = np.zeros(len(x))
    
    if(np.array_equal(v1+v3,v2+v4)):
        
        if(not(np.array_equal(v1,v3)) and not(np.array_equal(v2,v4))):
            
            g =  g + 2*(np.cos((v1[0]-v2[0])*x + (v1[1]-v2[1])*y) +\
                 np.cos((v3[0]-v2[0])*x + (v3[1]-v2[1])*y))

        if( np.array_equal(v1,v3) and not(np.array_equal(v2,v4))):
            
            g = g +  2*np.sqrt(2)*(np.cos((v1[0]-v2[0])*x + (v1[1]-v2[1])*y))  

        if(np.array_equal(v2,v4) and not(np.array_equal(v1,v3))): 
             
            g = g +  2*np.sqrt(2)*(np.cos((v1[0]-v2[0])*x + (v1[1]-v2[1])*y))
        if(np.array_equal(v1,v3) and  np.array_equal(v2,v4)):
            
            g = g + 2*np.exp(1j*(v1[0]-v2[0])*x)*np.exp(1j*(v1[1]-v2[1])*y) 
             
    return g


def averaged_pair(psi0,k1x,k2x,k1y,k2y,dimtot,x,y):


    """
    pair correlation function of the two electrons 
    
    depending on their spin

    """
    if(s==1):
            
        g2 = 0
        for i in range (dimtot):
            for j in range (dimtot):
                k1 = np.array([k1x[i],k1y[i]])

                k2 = np.array([k2x[i],k2y[i]])

                q1 = np.array([k1x[j],k1y[j]])

                q2 = np.array([k2x[j],k2y[j]])


                #if(npp[i]==npp[j] and  nmm[i] == nmm[j]): #the g function is diagonal in the photon number

                g2 = g2 + skew_g(k1,q1,k2,q2,x,y)*np.conj(psi0[j])*psi0[i]
        return g2
    if(s==0):
        g2 = 0
        for i in range (dimtot):
            for j in range (dimtot):
                k1 = np.array([k1x[i],k1y[i]])

                k2 = np.array([k2x[i],k2y[i]])

                q1 = np.array([k1x[j],k1y[j]])

                q2 = np.array([k2x[j],k2y[j]])


                #if(npp[i]==npp[j] and  nmm[i] == nmm[j]): #the g function is diagonal in the photon number

                g2 = g2 + sym_g(k1,q1,k2,q2,x,y)*np.conj(psi0[j])*psi0[i]
        return g2


#def sym_g(vec1,vec2,vec3,vec4,x,y):
#    return 0.5*(g_elem(vec1,vec2,vec3,vec4,x,y) + g_elem(vec3,vec2,vec1,vec4,x,y) + g_elem(vec1,vec4,vec3,vec2,x,y) + g_elem(vec3,vec4,vec1,vec2,x,y))


#print(np.array_equal(vec1,vec3) and  np.array_equal(vec2,vec4))


x = np.linspace(0,7,1000) # value of the coordinate along 1 axis
y = np.zeros(1000)



#######################################
#       plot of averaged pair         #
#######################################

fig = plt.figure(figsize = (8, 6), dpi=92)
ax = plt.subplot(1, 1, 1)
ax.set_title('averaged pair correlation function')
ax.grid(True, color = 'silver')

#ax.legend()
ax.title.set_fontsize(18)
ax.xaxis.label.set_fontsize(18)
ax.yaxis.label.set_fontsize(18)
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.xaxis.set_minor_locator(AutoMinorLocator(5))

ax.set_xlabel(r'$2\pi r/L$ ')
ax.set_ylabel(r'$ g(r)$')


ax.plot(x,averaged_pair(psi,k1x,k1y,k2x,k2y,dimtot,y,x))
plt.show()
fig.savefig('avg_pair_1m2.pdf')
