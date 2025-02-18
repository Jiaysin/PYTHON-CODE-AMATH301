import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.integrate
# question 1

#approximate area of exp(tan(x)) on -1 <= x <= 1

def f(x):
    return np.exp(np.tan(x))


#from 3.2 code
a = -1
b = 1
N = 100

xvec = np.linspace(a,b,N+1) #N subintervals = N+1 points
h = (b-a)/N

#midpoint rule
integral_mid = 0
for k in range(0,N):
    integral_mid += f((xvec[k]+xvec[k+1])/2) #just every point inthe sum
integral_mid = integral_mid*h

#note: error cannot be found with exp(tan(x)) because integral calculator deadass breaks on you

#trapezoid rule
integral_trapz = f(xvec[0])+f(xvec[N]) #first and last point
for k in range(1,N):
    integral_trapz += 2*f(xvec[k]) #every point in between
integral_trapz = integral_trapz*h/2 #multiply by h/2 at the end

#simpson's rule
xvec = np.linspace(a,b,2*N+1) #N subintervals = 2N+1 points
h = (b-a)/(2*N)

integral_simp = f(xvec[0])+f(xvec[2*N]) #first and last point
for k in range(1,2*N,2): #half-integer subscripts
    integral_simp += 4*f(xvec[k])
for k in range(2,2*N-1,2): #integer subscripts
    integral_simp += 2*f(xvec[k])
integral_simp = integral_simp*h/3








#question 2
#2a
bitcoin = np.load('assignments/bitcoin.npy')


# delta t found using difference bwt two points? 

N = len(bitcoin)

cpvec = np.zeros(N)


#forward for first
cpvec[0] = (bitcoin[1] - bitcoin[0])/1

#forward finite diff 
for i in range(1, N-1) : #get the second value and the second to last value (ie only the middle points)
    cpvec[i] = (bitcoin[i+1]-bitcoin[i])/1
    
#forward for end
cpvec[-1] = (bitcoin[-2] - bitcoin[-1])/1

# print(cpvec)


#2b 
cp2vec = np.zeros(N)

#forward for first
cp2vec[0] = (cpvec[1] - cpvec[0])/1

#forward finite diff 
for i in range(1, N-1) : #get the second value and the second to last value (ie only the middle points)
    cp2vec[i] = (cpvec[i+1] - cpvec[i])/1
    
#forward for end
cp2vec[-1] = (cpvec[-2] - cpvec[-1])/1

# print(cp2vec)

#2c
# x_values = np.linspace(1, N, N)
# plt.plot(x_values,cpvec,'-ro',label='c\'(t) [$/month]')
# plt.plot(x_values,cp2vec,'-bo',label='c\'\'(t) [$/month^2]')
# plt.legend()
# plt.xlabel('month of 2024')
# plt.show()






#question 3
def f(x, y):
    if (x - 1) ** 2 + (y - 1) ** 2 < 1:  # Exclude the cylinder region
        return 0
    elif x**2 + y**2 <= 1:  # Inside the paraboloid
        return 1 - x**2 - y**2
    return 0  # Outside the shape


volume, error = scipy.integrate.dblquad(f, -2, 2, -2, 2)
# print(volume)




#question 4

def f(t):
    return np.exp(1/t)

#4a
derivtrue = -0.412180317675

#4b
dtvec = np.array([2**(-i) for i in range(11)])

#4c

# second order
logrelerr2o = np.zeros(len(dtvec))
for i in range(len(dtvec)):
    dt = dtvec[i]
    fapprox = (f(2+dt) - f(2-dt))/ (2*dt)
    logrelerr2o[i] = np.abs((fapprox - derivtrue)/derivtrue)
# print(logrelerr2o)

#4d

#fourth order

logrelerr4o = np.zeros(len(dtvec))
for i in range(len(dtvec)):
    dt = dtvec[i]
    fapprox = (-f(2 + 2*dt) + 8*f(2 + dt) - 8*f(2 - dt) + f(2 - 2*dt)) / (12*dt)
    logrelerr4o[i] = np.abs((fapprox - derivtrue)/derivtrue)
# print(logrelerr4o)

#4e

# plt.plot(np.log(dtvec),logrelerr2o,'-bo',label='2nd order FD')
# plt.plot(np.log(dtvec),logrelerr4o,'-go',label='4th order FD')
# plt.legend()
# plt.xlabel('log(dt)')
# plt.ylabel('log(relative error)')
# plt.show()


