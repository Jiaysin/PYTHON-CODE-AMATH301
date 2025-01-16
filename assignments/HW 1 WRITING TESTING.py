import numpy as np

#QUESITON 1
A = np.array([[30, 15, 21, 16, 18],[20, 11, 26, 37, 15],
[35, 33, 10, 14, 19],[21, 18, 17, 31, 32]])
#print(A)
#lines below to be changed---
for i in range(0,4):
    for j in range(0,5):
        if A[i,j]<=19:
            A[i,j]=0
        elif A[i,j]>=30:
            A[i,j]=0
#lines above to be changed---
#print(A)

#using logical operations instead (boolean)
B = np.array([[30, 15, 21, 16, 18],[20, 11, 26, 37, 15],
[35, 33, 10, 14, 19],[21, 18, 17, 31, 32]])

B[(B <= 19)|(B >= 30)] = 0
#print(B)

#QUESITON 2

xl = 0 # left bound
xr = 2 # right bound

for j in range(0,3): # 2 intervals w the final one being the answer
    xc =(xr + xl)/2 #bisection
    value_xc = (xc)**3 + (xc)**2 + (xc) - 2
    
    if value_xc > 0:
        xr = xc
    else:
        xl = xc
#print(xc)

#QUESTION 3
xr = -2.8; xl = -4 #old code
xr = 4; xl = 0
for j in range(0, 100):
    xc = (xr + xl)/2
#    fc = np.exp(xc) - np.tan(xc) #old code
    fc = xc**2-2
#    display(fc) #debug
    if ( fc > 0 ):
        xr = xc 
        #xl = xc #xl and xr have to change in the if statement
    else:
        xl = xc
        #xr = xc

    if ( abs(fc) < 1e-5 ):
        #print(xc); print(j)
        break

#using sin(x) instead for our example
xl = 1 ; xr = 4
# assignment allows for zero being one of the starting interval values so use that
for j in range(0,100):
    xc = (xr + xl)/2

    fc = np.sin(xc)
    fxl = np.sin(xl)

    if fc * fxl > 0: #checking if the values between the left interval and the center interval are above/below the x-axis
        xl = xc #sets the new interval to have the centerpoint as the left interval, as the right side will contain the negative, telling us the root is in that bound.
    else:
        xr = xc #sets the new interval to have the centerpoint as teh right interval

    if ( abs(fc) < 1e-5):
        #print(xl)
        #print(xr)
        #print(j)
        break
