import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#Problem 1

#dy/dt = (1-y)y, y(0) = y0

def f(t,y):
    return (1-y)*y

tspan = np.array([0,10]) #for the solve_ivp
tt = np.arange(0,10.1,0.1) #for the graph
# print(tt)

y0 = np.array([0,0.01]) # when t = 0, y = 0.01

sol = solve_ivp(f,tspan,y0,method="RK45", t_eval = tt)

logistic1 = sol.y[1]

# (b)

y0 = np.array([0,0.5])

sol = solve_ivp(f,tspan,y0,method="RK45", t_eval = tt)

logistic2 = sol.y[1]

# (c)

y0 = np.array([0, 1.5])

sol = solve_ivp(f,tspan,y0,method="RK45", t_eval = tt)

logistic3 = sol.y[1]

# plt.plot(tt,logistic1,'-m',label='y_0=0.01')
# plt.plot(tt,logistic2,'-b',label='y_0=0.5')
# plt.plot(tt,logistic3,'-g',label='y_0=1.5')
# plt.xlabel('t')
# plt.ylabel('y')
# plt.legend(loc='lower right')

#plt.show()

#problem 2

# y'' = -g , y0 = y0 , y'0 = y'0

# with air drag, y'' = -g - k(y')**2, y0 = y0 , y'0 = y'0

#(a) Eulers method

g = 9.81 #gravity
def f(t,v): #v' = f(t,v) where v = [ y, z ]^T
    output = np.array ([v[1], -g])
    return output

dt = 0.1
tvec = np.arange(0,2.01,dt)


yold = np.array([10,5]) #y0 and y'0 initial conditions
eulermat1 = yold
for t in tvec[1:len(tvec)]:
    ynew = yold+dt*f(t,yold) #eulers method yn+1 = yn +dt*f(tn,yn)
    eulermat1 = np.vstack((eulermat1,ynew)) #vertical stacking of all values?
    yold = ynew
    
# print(eulermat1)

#(b)
k = (0.1)
def f_air(t,v): #WITH AIR DRAG
    output = np.array (v[1], -g-k*(v[1])**2)
    return output

eulermat2 = yold
for t in tvec[1:len(tvec)]:
    ynew = yold+dt*f_air(t,yold) #eulers method yn+1 = yn +dt*f(tn,yn)
    eulermat2 = np.vstack((eulermat2,ynew)) #vertical stacking of all values?
    yold = ynew
    
#print(eulermat2)

# plt.plot(tvec,eulermat1[:,0],'-m',label='no drag')
# plt.plot(tvec,eulermat2[:,0],'-g',label='drag coeff = 0.1')
# plt.xlabel('t')
# plt.ylabel('y')
# plt.legend(loc='lower left')
# plt.show()

#problem 3
# d2x/d2t - mu*(1-x**2)*dx/dt + x = 0
# mu = 1 , x(0) = 0 , x(5) = 5
mu = 1
def f(x,v):
    output = np.array([v[1], mu*(1-x**2)*v[1]+x])
    return output

tspan = ([0,5])

#(a)
alpha = 20.

tt3 = np.arange(tspan[0], tspan[1]+0.05, 0.05)

ic = np.array([0, alpha])
sol = solve_ivp(f, tspan, ic, method="RK45", t_eval = tt3)
ysolve = sol.y[0]
endpt = ysolve[-1]
# print (endpt)

alphainterval = np.array([0,0])
for i in range(100):
    alpha = 19 - i
    ic = np.array([0, alpha])
    sol = solve_ivp(f, tspan, ic, method="RK45", t_eval = tt3)
    ysolve = sol.y[0]
    endpt = ysolve[-1]
    if endpt < 0:
        alphainterval[0] = endpt
        print(alpha)
        break
    
for i in range(100):
    alpha = 19 - i
    ic = np.array([0, alpha])
    sol = solve_ivp(f, tspan, ic, method="RK45", t_eval = tt3)
    ysolve = sol.y[0]
    endpt = ysolve[-1]
    if endpt > 0:
        alphainterval[1] = endpt
        print(alpha)
        break

print(alphainterval)

a_left = alphainterval[0]
a_right = alphainterval [1]
for i in range(101):
    a_mid = (a_left+a_right)/2
    ic = np.array([0, a_mid])
    sol = solve_ivp(f, tspan, ic, method="RK45", t_eval = tt3)
    ysolve = sol.y[0]
    endpt = ysolve[-1]
    
    if np.abs(endpt) < 1e-5:
        alphatrue = a_mid
        print(alphatrue)
        break
        
    if endpt > 0:
        a_right = a_mid
    else:
        a_left = a_mid

    
    
