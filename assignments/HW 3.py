import numpy as np
import matplotlib.pyplot as plt

# implements Naive Gauss elimination without partial pivoting. may encounter a zero pivot, which breaks the algorithm.
def naivegauss(A,b):
    # combine A and b into an augmented matrix
    augmat = np.hstack([A,b]) #horizontal concatenation. b must be column vector

    n = len(b)
    
    # forward elimination
    for i in range(0,n-1): #iterate over columns
        for j in range(i+1,n): #iterate over rows in this column, below the pivot
            factor = augmat[j,i]/augmat[i,i]
            augmat[j,:] = augmat[j,:] - factor*augmat[i,:]

    # back substitution
    for i in range(n-1,-1,-1): #iterate over columns
        for j in range(i-1,-1,-1): #iterate over rows in this column, above the pivot
            factor = augmat[j,i]/augmat[i,i]
            augmat[j,:] = augmat[j,:] - factor*augmat[i,:]

    # scale diagonal elements to 1
    for i in range(0,n):
        augmat[i,:] = augmat[i,:]/augmat[i,i]

    return augmat[:,n]

# implements Gauss elimination with partial pivoting

def gauss(A,b):
    
    rowflips = 0 # for question 2
    
    # combine A and b into an augmented matrix
    augmat = np.hstack([A,b]) #horizontal concatenation. b must be column vector
    n = len(b)

    # forward elimination
    for i in range(0,n-1): #iterate over columns
        #partial pivoting steps below-------------------------------------------
        rowtobepivot = i+np.argmax(np.abs(augmat[i:,i]))
        if rowtobepivot!=i: #swap rows
            # print("row {1} swapped with row {tobepivot}")
            augmat[[rowtobepivot,i]] = augmat[[i,rowtobepivot]]
            rowflips += 1
        #partial pivoting steps above-------------------------------------------

        for j in range(i+1,n): #iterate over rows in this column, below the pivot
            factor = augmat[j,i]/augmat[i,i]
            augmat[j,:] = augmat[j,:] - factor*augmat[i,:]

    # back substitution
    for i in range(n-1,-1,-1): #iterate over columns
        for j in range(i-1,-1,-1): #iterate over rows in this column, above the pivot
            factor = augmat[j,i]/augmat[i,i]
            augmat[j,:] = augmat[j,:] - factor*augmat[i,:]

    # scale diagonal elements to 1
    for i in range(0,n):
        augmat[i,:] = augmat[i,:]/augmat[i,i]
        

    return augmat[:,n], rowflips


#----------------------------------------------------------------------------------
epsvec = np.logspace (0, -15, 16) #does the log from 0 to -15. log(0) = 1, etc etc.
# print(epsvec)
errorvec = np.array([])
for epsvalue in epsvec:
    A = [[epsvalue, 2., 1, 10, 5],[4, 6, 10, 4, 6],[5, 10, 8, 0, 7],[8, 0, 8, 2, 4],[9, 8, 1, 0, 4]]
    b = [[-13], [6], [8], [-8], [-21]]
    # xnaive = np.array(naivegauss(A,b) [0]) #the 0 indicates the index and what value i want. since my thing is return matrix AND rowflip, i ned to do this
    # print("xnaive is: ", naivegauss(A,b))
    # print("actual solution is: ", np.linalg.solve(A,b))
    # xfull = gauss(A,b) [0]
    
    error = np.linalg.norm(naivegauss(A,b) - gauss(A,b)[0], 2)
    errorvec = np.append(errorvec,error)
    
logeps = np.log10(epsvec)
logerror = np.log10(errorvec)
plt.plot(logeps,logerror,"-or")
plt.xlabel(r"$\log_{10}(\epsilon)$")
plt.ylabel(r"$\log_{10}(||x_{\text{naive}} - x_{\text{full}}||_2)$")


# plt.show()
    
#QUESTION 2 

nvec = np.array([4, 8, 16, 32, 64, 128, 256])
percentflips = np.zeros(len(nvec))

for j in range(len(nvec)):
    
    xn = nvec[j]
    
    np.random.seed(1) #ensures your random numbers are the same as the autograder’s
    A=np.random.randint(0,100,[xn,xn])
    A=A.astype(np.float32) #convert integers to floating-point objects (decimals)
    b=np.random.randint(0,100,[xn,1])
    if xn==256:
        bseedcheck = b  
    
    # print(A,b)
        
    rowflips = gauss(A,b)[1]
    
    percentflips[j] = float(rowflips / xn)
    
# print("percentage of flips in each n value is: ", percentflips)

#PERCENTAGES ARENT RIGHT (?) i get [0.5       0.625     0.75      0.6875    0.890625  0.953125  0.9765625]

#QUESTION 3


np.random.seed(1) #ensures your random numbers are the same as the autograder’s
Ajacobi = 0.01*np.random.randint(-50,51,[10,10])+10*np.eye(10)
bjacobi = np.random.randint(-50,51,[10,1])

# Jacobi iterations
stepvec = []

x0 = np.zeros(10) #initial guesses 

xn = x0.copy()

for j in range(100):
    xn1 = np.zeros_like(xn) # sets initial guesses to 0
    
    for i in range(10): #looking at the rows
        
        sum_Ax = 0 #sum of non-diagonal terms to the dominant diagonal
        
        for j_col in range(10): #looking at the columns
            
            if j_col != i: #skips the diagonal dominant
                
                sum_Ax += Ajacobi[i, j_col] * xn[j_col]
                
        #computing xn1 values
        xn1[i] = (bjacobi[i] - sum_Ax) / Ajacobi[i,i]
            
    
    step = np.linalg.norm(xn1 - xn, 2)
    stepvec = np.append(stepvec, step)
    
    if step < 1e-5:
        print(step)
        break
    
    xn = xn1.copy() # for next iteration
    
stepvec = np.array(stepvec)
    
print("stepvec is: ", stepvec)