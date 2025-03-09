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

def f(t,v): # v is dy and dz
    return np.array([v[1], -9.8])
    
dt = 0.1
vold = np.array([10, 5])
tvec = np.arange(0, 2.01, 0.1)
eulermat1 = vold
for t in tvec[0:len(tvec)-1]:
    vnew = vold + dt * f(t,vold)
    eulermat1 = np.vstack((eulermat1, vnew))
    vold = vnew

print(np.shape(eulermat1))
    
# print(eulermat1)

#(b)
k = 0.1
def g(t,v):
    return np.array([v[1], -9.8-k*(v[1])**2])

vold = np.array([10, 5])
tvec = np.arange(0, 2.01, 0.1)
eulermat2 = vold
for t in tvec[0:len(tvec)-1]:
    vnew = vold + dt*g(t,vold)
    eulermat2 = np.vstack((eulermat2, vnew))
    vold = vnew

print(np.shape(eulermat2))
    
#print(eulermat2)

plt.plot(tvec,eulermat1[:,0],'-m',label='no drag')
plt.plot(tvec,eulermat2[:,0],'-g',label='drag coeff = 0.1')
plt.xlabel('t')
plt.ylabel('y')
plt.legend(loc='lower left')
# plt.show()

#problem 3
# d2x/d2t - mu*(1-x**2)*dx/dt + x = 0
# mu = 1 , x(0) = 0 , x(5) = 5

#(a)

mu = 1
def h(t,v): 
    return np.array([v[1], mu*(1 - v[0]**2) * v[1] - v[0]])
    
alpha = 20
tt3 = np.arange(0,5.01, 0.05)
tspan = np.array([0,5])
ic = np.array([0, alpha])

sol = solve_ivp(h, tspan, ic, method = "RK45", t_eval = tt3)
xsol = sol.y[0]
endpt = xsol[-1]
print(endpt)

#(b)

#all green doesnt work lmao 

#------------------------------------------------------------------


# for i in range(100):
#     alpha = 19 - i
#     ic = np.array([0, alpha])
#     sol = solve_ivp(f, tspan, ic, method="RK45", t_eval = tt3)
#     ysolve = sol.y[0]
#     endpt = ysolve[-1]
#     if endpt < 0:
#         alphainterval[0] = endpt
#         print(alpha)
#         break
    
# for i in range(100):
#     alpha = 19 - i
#     ic = np.array([0, alpha])
#     sol = solve_ivp(f, tspan, ic, method="RK45", t_eval = tt3)
#     ysolve = sol.y[0]
#     endpt = ysolve[-1]
#     if endpt > 0:
#         alphainterval[1] = endpt
#         print(alpha)
#         break

#----------------------------------------------------------------
alphainterval = np.array([0,0])

for i in range(100):
    alpha = 19 - i
    ic = np.array([0, alpha])
    sol = solve_ivp(h, tspan, ic, method="RK45", t_eval = tt3)
    ysolve = sol.y[0]
    endval = ysolve[-1]
    if endval < 0:
        alphainterval[0] = alpha
        alphainterval[1] = prev_alpha
    
        break
    prev_alpha = alpha
    
print(alphainterval)


a_left = alphainterval[0]
a_right = alphainterval [1]
for i in range(101):
    a_mid = (a_left+a_right)/2
    ic = np.array([0, a_mid])
    sol = solve_ivp(h, tspan, ic, method="RK45", t_eval = tt3)
    ysolve = sol.y[0]
    endval = ysolve[-1]
    
    if np.abs(endval) < 1e-5:
        alphatrue = a_mid
        #print(alphatrue)
        break
        
    if endval > 0:
        a_right = a_mid
    else:
        a_left = a_mid

    
    
