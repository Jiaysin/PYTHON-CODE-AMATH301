import numpy as np
import matplotlib.pyplot as plt
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
    
    
#1C

fig = plt.subplot(311)
plt.plot(xvals,yvals0)
plt.title("Plot for b = 0")
plt.xlabel('x')
plt.ylabel('f(x)')


fig = plt.subplot(312)
plt.plot(xvals,yvals1)
plt.title("Plot for b = 1")
plt.xlabel('x')
plt.ylabel('f(x)')



fig = plt.subplot(313)
plt.plot(xvals,yvals2)
plt.title("Plot for b = 2")
plt.xlabel('x')
plt.ylabel('f(x)')

plt.show()