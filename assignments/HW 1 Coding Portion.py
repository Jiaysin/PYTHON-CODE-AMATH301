# HW 1 Coding Portion
# AMATH 301
# Jayson Nguyen #ID 2420780

import numpy as np
# question 1a ----------------------------

# solving for the step gap between 3 and 4
# we need [3, 3.2, 3.4, 3.6, 3.6, 4]
u = np.arange(3,4.1,.2) #use 4.1 bc we want to include 4 and not 4.2

#print('u: ', u) # gives [3, 3.2, 3.4, 3.6, 3.8, 4]

# question 1b -------------------------
v = np.arange(0,4,0.75)

#print('v: ',v) # gives [0, 0.75, 1.5, 2.25, 3, 3.75]

# question 1c ----------------------
# save the vector w = v + 2u to the variable w
w = v + 2*u

#print ('w = ',w) # gives us [6, 7.15, 8.3, 9.45, 10.6, 11.75]

# question 1d ----------------
# create a vector alpha whose entries are the entries of u cubed, component wise.
alpha = u**3

#print('alpha = ', alpha) #gives [27, 32.768, 39.304, 46.656, 54.872, 64. ]

# question 1e ---------------------
# create a vector z where z is the componentwize product of u and v
z = u*v

#print('z = ',z) # gives [0, 2.4, 5.1, 8.1, 11.4, 15]

#question 1f ----------------------
# compute the dot product d = u^T * v = ∑uv. ans is a single number
d = (u @ v)

#print('d = ',d) # gives us 42

#---------------------------------------------------------------

# question 2 ---------------------
# use logical operations, not for loops or if statements. 
exponents = np.arange(-10,11) #goes from -10 to 10
a = np.array(2.0**exponents) #applies the exponent to 2, since its a matrix itll work.

# logical sorting creating a bound
b = a[( a >= 0.1 ) & ( a <= 10.0 )] #boolean array looking for true and false within bound

#print(b) #gives [0.125 0.25  0.5   1.    2.    4.    8.   ]

#question 3 --------------------------------------
r = np.array(np.arange(-3,3.1,0.1)) #array is to set the vector, arange is to quickly put it together
#print(r)

y=np.array([]) #empty matrix to be used

set

#using if statement
#append adds the value to the table as the value of rn follows along r
for rn in r:
    if rn <= -1:
        y = np.append(y,(1 / rn)) #function 1/x if x <= -1
    elif (-1 < rn) & (rn <= 1):
        y = np.append(y,np.exp(-rn)) #function e^(-x) if -1 < x <= 1
    else: #use else instead bc every other condition is written in if and elif
        y = np.append(y,np.cos(rn))

#print(y) #the function y = f(r)

#question 4a ------------------
# using loop
q = np.arange(1,101)
#print(q)
sum_q = 0 # initial value for sum
for j in q:
    sum_q = (sum_q + j**2)
normq_loop = np.sqrt(sum_q)

#print(normq_loop) #gives 581.6786054171153

#question 4b ------------------- 
# using vector manipulation
q_squared_sum = q @ q # dot product
normq_vec = np.sqrt(q_squared_sum) #sqrt the dot product to get magnitude

#print(normq_loop) #gives 581.6786054171153

#question 5 ----------------------
# find the smallest positive int value of N such that SN > 7
# use a loop
SN = 0 # initial start before summing
for N in range(1,100000): # must start at 1 bc 1/0 is DNE
    SN = SN + (1 / N)
    
    if SN > 7:
        #print(SN) #displays the value thats > 7 
        # gives us the value 7.00127
        #print(N) #displays how many iterations it took
        # gives us the value 616
        break




    