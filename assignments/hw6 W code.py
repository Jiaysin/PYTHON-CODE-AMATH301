import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
 
def f(t,y): #y' = f(t,y)
    return y*(1-y)*(y-1/2)

dt = 1
dp = 0.1
tvec = np.arange(0,11,dt)
yvec = np.arange(0,1.1,dp)


#MAKE QUIVER PLOT---------------------------------------

Tmat,Ymat = np.meshgrid(tvec,yvec) #converts vectors to matrices. necessary for
#quiver plot

U = np.ones([len(yvec),len(tvec)]) #horizontal component of each vector
V = f(Tmat,Ymat)*(tvec[-1]-tvec[0])/(yvec[-1]-yvec[0])*3/4 #vertical component
#of each vector; factor needed to scale arrows with axes correctly


plt.quiver(tvec,yvec, U, V, width=0.003) #plot vectors
plt.plot(Tmat,Ymat,'ro',markersize=2) #plot red dots
plt.show()