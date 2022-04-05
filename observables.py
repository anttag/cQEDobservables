#####################################################
#####################################################

###   observables for the two electron problem    ###

#####################################################
#####################################################


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

norm = np.sum(np.conj(psi)*psi)
print(norm)
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

k1 = np.array([k1x,k1y]) #construct the first electron momentum array
k2 = np.array([k2x,k2y]) #construct the second electron momentum array

#print(k1)
x = np.linspace(0,3,1000) # value of the coordinate along 1 axis
y = np.zeros(1000)
r = np.array([x,x]) # coordinate of the two dimensional position array
#print(r)



#s = exp(1j*(vec1[0]-vec2[0]))*exp(1j*(vec1[1]-vec2[1]))
#print(s)
#temp1 = (vec1[0]-vec2[0])
#temp2 = (vec1[1]-vec2[1])
#print(type(temp1))
 
def density_elem(vec1,vec2,vec3,vec4,x,y):
    '''
        matrix element for the density operator
    '''

    # vec1 = k1 , vec2 = k1', vec3 = k2 , vec4 = k2' (using the convenction in matrix_element.pdf)
    n = np.zeros(len(x))

    temp1 = (vec1[0]-vec2[0])

    temp2 = (vec1[1]-vec2[1])

    #print(temp1,temp2)
    if (np.array_equal(vec3,vec4)): #check if .any() works fine
        n = n + np.exp(1j*temp1*x)*np.exp(1j*temp2*y)
    if (np.array_equal(vec1,vec2)):

        n = n + np.exp(1j*(vec3[0]-vec4[0])*x)*np.exp(1j*(vec3[1]-vec4[1])*y)
    

    return n  
   
def sym_density(vec1,vec2,vec3,vec4,x,y) :
    if(np.array_equal(vec1,vec3) and np.array_equal(vec2,vec4)):
        return density_elem(vec1,vec2,vec3,vec4,x,y)
    if(np.array_equal(vec1,vec3) and not np.array_equal(vec2,vec4)):
        return  np.sqrt(2)*(density_elem(vec1,vec2,vec3,vec4,x,y)+density_elem(vec3,vec2,vec1,vec4,x,y))

    if(np.array_equal(vec2,vec4) and not  np.array_equal(vec1,vec3) ):
        return np.sqrt(2)*(density_elem(vec1,vec2,vec3,vec4,x,y) + density_elem(vec1,vec4,vec3,vec2,x,y))
    else:  
        return 0.5*(density_elem(vec1,vec2,vec3,vec4,x,y)+density_elem(vec3,vec2,vec1,vec4,x,y) + density_elem(vec1,vec4,vec3,vec2,x,y) + density_elem(vec3,vec4,vec1,vec2,x,y))

#print(density_elem(vec1,vec2,vec3,vec4,x,0).real)

#print(density_elem(vec1,vec2,vec3,vec4,x,y))

def skew_density(vec1,vec2,vec3,vec4,x,y) :
    return 0.5*(density_elem(vec1,vec2,vec3,vec4,x,y)- density_elem(vec3,vec2,vec1,vec4,x,y) - density_elem(vec1,vec4,vec3,vec2,x,y) + density_elem(vec3,vec4,vec1,vec2,x,y))


def g_elem(vec1,vec2,vec3,vec4,x,y):

    '''
    matrix element for the averaged pair correlation function
    '''

    g = np.zeros(len(x))
    if(np.array_equal(vec1+vec3,vec2+vec4)):
        g = g + np.exp(1j*(vec1[0]-vec2[0])*x)*np.exp(1j*(vec1[1]-vec2[1])*y) + np.exp(1j*(vec3[0]-vec4[0])*x)*np.exp(1j*(vec3[1]-vec4[1])*y)
    return g

#def f(x,y):
#    return np.exp(1j*temp1*x)*np.exp(1j*temp2*y)


def sym_g(vec1,vec2,vec3,vec4,x,y):
    return 0.5*(g_elem(vec1,vec2,vec3,vec4,x,y)+ g_elem(vec3,vec2,vec1,vec4,x,y) + g_elem(vec1,vec4,vec3,vec2,x,y) + g_elem(vec3,vec4,vec1,vec2,x,y))


def skew_g(vec1,vec2,vec3,vec4,x,y):
    return 0.5*(g_elem(vec1,vec2,vec3,vec4,x,y) - g_elem(vec3,vec2,vec1,vec4,x,y) - g_elem(vec1,vec4,vec3,vec2,x,y) + g_elem(vec3,vec4,vec1,vec2,x,y))
######################

## density function ##

######################



def density(psi0,k1x,k2x,k1y,k2y,npp,nmm,dimtot,x,y):
    """
    density function for the two electrons depending on their spin
    """
    if(s==1):
        d = 0
        for i in range (dimtot):
            for j in range (dimtot):
                k1 = np.array([k1x[i],k1y[i]])

                k2 = np.array([k2x[i],k2y[i]])

                q1 = np.array([k1x[j],k1y[j]])

                q2 = np.array([k2x[j],k2y[j]])

                if(npp[i]==npp[j] and  nmm[i] == nmm[j]): #the density function in diagonal in the photon number
                    d = d + skew_density(k1,q1,k2,q2,x,y)*np.conj(psi0[i])*psi0[j]

        return d
    if(s==0):
        d = 0
        for i in range (dimtot):
            for j in range (dimtot):
                k1 = np.array([k1x[i],k1y[i]])

                k2 = np.array([k2x[i],k2y[i]])

                q1 = np.array([k1x[j],k1y[j]])

                q2 = np.array([k2x[j],k2y[j]])

                if(npp[i]==npp[j] and  nmm[i] == nmm[j]): #the density function in diagonal in the photon number
                    d = d + sym_density(k1,q1,k2,q2,x,y)*np.conj(psi0[i])*psi0[j]
        return d

    ########################################

##### pair correlation function ########

########################################

def averaged_pair(psi0,k1x,k2x,k1y,k2y,npp,nmm,dimtot,x,y):
    """
    pair correlation function of the two electrons 
    
    depending on their spin

    """
    if(s==1):
            
        g = 0
        for i in range (dimtot):
            for j in range (dimtot):
                k1 = np.array([k1x[i],k1y[i]])

                k2 = np.array([k2x[i],k2y[i]])

                q1 = np.array([k1x[j],k1y[j]])

                q2 = np.array([k2x[j],k2y[j]])


                if(npp[i]==npp[j] and  nmm[i] == nmm[j]): #the g function is diagonal in the photon number

                    g = g + skew_g(k1,q1,k2,q2,x,y)*np.conj(psi0[i])*psi0[j]
        return g
    if(s==0):
        g = 0
        for i in range (dimtot):
            for j in range (dimtot):
                k1 = np.array([k1x[i],k1y[i]])

                k2 = np.array([k2x[i],k2y[i]])

                q1 = np.array([k1x[j],k1y[j]])

                q2 = np.array([k2x[j],k2y[j]])

                if(npp[i]==npp[j] and  nmm[i] == nmm[j]): #the g function is diagonal in the photon number
                    g = g + sym_g(k1,q1,k2,q2,x,y)*np.conj(psi0[i])*psi0[j]
        return g




def rho_mat(psi0,dimph,dimel): 

    """
    reduced density matrix for the two electrons
    """

    rho = np.zeros((dimel,dimel)) #reduced density matrix of the form dimel *dimel
    for k in range (0,dimph):
        for i in range (k*dimel,(k+1)*dimel):
            for j in range (k*dimel,(k+1)*dimel):
                rho[i][j] = rho[i][j] + psi0[i]*np.conj(psi0[j])

    return rho

eig = np.zeros(dimel) #array that contains eigenvalue of the reduced density matrix
vect = np.zeros((dimel,dimel)) #matrix for the eigenvector of the reduced density matric
vect, eig = np.linalg.eigh(rho_mat(psi,dimph,dimel)) #exact diagonalization of the density matrix



S = - np.sum((np.log(eig)*eig)) #entanglement entropy

#plt.rc('text', usetex=True)
#plt.rc('text.latex', preamble=r'\usepackage{amsmath}' r' \usepackage{physics}')

#plt.rcParams["text.latex.preamble"].join([ r"\usepackage{physics}" r"\setmainfont{xcolor}"])

############################################

######    plot \hat{\bar {g}}(r)     #######

 ###########################################
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

ax.set_xlabel('r')
ax.set_ylabel(r'$ \langle\hat{\bar {g}}(r)\rangle$')

ax.plot(x,averaged_pair(psi,k1x,k2x,k1y,k2y,npp,npm,dimel,y,x).real)

#fig.savefig('pair_correlation.pdf')
plt.show()

##################################################

################    plot n(r)   ##################       

##################################################

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

ax.set_xlabel('r')
ax.set_ylabel(r'$ \langle\hat{n}(r)\rangle $')

ax.plot(x,density(psi,k1x,k2x,k1y,k2y,npp,npm,dimel,y,x).real)

#fig.savefig('pair_correlation.pdf')
plt.show()
