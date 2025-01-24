import numpy as np
import matplotlib.pyplot as plt
import math
#QUESTION 1 ---------------------

#1A
#use np.linspace to create a vector x with 201 equally spaced values on the interval [-10,10]. call this variable xvals

xvals = np.linspace(-10,10,201) #interval from -10 to 10, with 201 spaced values
#print(xvals)


#1B

def func(x):
    return abs(1/2 - abs(2 * np.cos(x) - b)) - 1

#when b = 0

yvals0 = np.array([])
b = 0

for xn in xvals:
    y0_xn = func(xn)
    
    yvals0 = np.append(yvals0, y0_xn)

#when b = 1

yvals1 = np.array([])
b = 1

for xn in xvals:
    y1_xn = func(xn)
    
    yvals1 = np.append(yvals1, y1_xn)
    
#when b = 2

yvals2 = np.array([])

b = 2

for xn in xvals:
    y2_xn = func(xn)
    
    yvals2 = np.append(yvals2, y2_xn)
    
    
#1C ----- WRITTEN STUFF

# fig = plt.subplot(311)
# plt.plot(xvals,yvals0)
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.title("Plot for b = 0")

# fig = plt.subplot(312)
# plt.plot(xvals,yvals1)
# plt.title("Plot for b = 1")
# plt.xlabel('x')
# plt.ylabel('f(x)')

# fig = plt.subplot(313)
# plt.plot(xvals,yvals2)
# plt.title("Plot for b = 2")
# plt.xlabel('x')
# plt.ylabel('f(x)')

#plt.show()


#QUESTION 2

#2B

def T(x,n):
    taylor_sum = 0
    for k in range(n + 1): #ranges from 0 to n
        term = ((-1)**k / math.factorial(2*k)) * x**(2*k)
        taylor_sum = (taylor_sum + term)
        
    return taylor_sum

#2C
# find T0(2), T1(3), and T2(-1) by plugging the appropriate values of x and n into your T(x,n) function. call these values T02, T13, and T2m1 respectively

T02 = T(2,0)
T13 = T(3,1)
T2m1 = T(-1,2)

# print("T02: ", T02)
# print("T13: ", T13)
# print("T2m1: ", T2m1)

#2D
xtaylor = np.linspace(-np.pi, np.pi, 51)
print (xtaylor)

#2E
rows = []
for n in range(6): #goes from 0 to 5
    row = T(xtaylor, n) #compute all T_n(x) values for x in the xtaylor
    rows.append(row) #add the row to the list
    
Mtaylor = np.array(rows)
#print("Mtaylor: ", Mtaylor)

#2F ---------- WRITTEN STUFF
# x = np.linspace(-10,10,1001)
# plt.plot(x, T(x, 0), "g-", label="T\\_0(x)")
# plt.plot(x, T(x, 2), "r:", label="T\\_2(x)")
# plt.plot(x, T(x, 4), "b--", label="T\\_4(x)")
# plt.plot(x, np.cos(x), "k-.", label="cos(x)")

# plt.xlabel("x values")
# plt.ylabel("y values")
# plt.title("Functions cos(x) and its taylor series'")
# plt.legend()

# plt.ylim(-np.pi,np.pi)
# plt.show()


    
    
