#SYNC THIS INTO THE OTHER ASSIGNMENT LATER PLS
#QUESTION 3

import numpy as np
import matplotlib.pyplot as plt

#3A
xnewton = np.linspace(-5,5,201)
numiters = np.array([])


def f(x):
    y = x**3 - 2*x + 2
    return y

def fp(x):
    y = 3*x**2 - 2
    return y
    
for xn in xnewton: #initial guesses
    for j in range(1,101):
        xn = xn-f(xn)/fp(xn)
        # print("xn = ", xn)
    
        fc = f(xn)
    
        if abs(fc) < 1e-5:
            # print('number of iterations: ', j)

            break
    numiters = np.append(numiters,j)
    
print(len(numiters))
print("numiters: ",numiters)

#QUESTION 4

# q must be => 1
# p must be => 3
#unit circle

def xpoint(angle): #make x points
    return np.cos(angle)

def ypoint(angle): #make y points
    return np.sin(angle)

def xstarpoints(p,q,xstar):
    theta = np.linspace(0, 2*np.pi, p, endpoint = False) # goes around full circle in even intervals of p. endpoint is unnecessary bc (1,0) is already a given point
    
    current_index = 0
    
    for j in range(p+1):
        angle = theta[current_index] #gets the angle at that point 
        
        xstar = np.append(xstar,xpoint(angle))
        
        current_index = (current_index + q) % p #prevent it from being larger than actual p value, wrapping it around
        
    return xstar

def ystarpoints(p,q,ystar):
    theta = np.linspace(0, 2*np.pi, p, endpoint = False) # goes around full circle in even intervals of p. endpoint is unnecessary bc (1,0) is already a given point
    
    current_index = 0
    
    for j in range(p+1):
        angle = theta[current_index] #gets the angle at that point 
        
        ystar = np.append(ystar,ypoint(angle))
        
        current_index = (current_index + q) % p #prevent it from being larger than actual p value, wrapping it around
    
    return ystar

#QUESTION 4A
p = 11
q = 2

xstar_11_2 = np.array ([])
ystar_11_2 = np.array ([])

xstar_11_2 = xstarpoints(p,q,xstar_11_2)
ystar_11_2 = ystarpoints(p,q,ystar_11_2)


plt.plot(xstar_11_2, ystar_11_2, '-or', markeredgecolor='k', markerfacecolor='k')
plt.axis('scaled')
plt.show()

#QUESTION 4B

p = 11
q = 4

xstar_11_4 = np.array ([])
ystar_11_4 = np.array ([])

xstar_11_4 = xstarpoints(p,q,xstar_11_4)
ystar_11_4 = ystarpoints(p,q,ystar_11_4)

plt.plot(xstar_11_4, ystar_11_4, '-or', markeredgecolor='k', markerfacecolor='k')
plt.axis('scaled')
plt.show()

