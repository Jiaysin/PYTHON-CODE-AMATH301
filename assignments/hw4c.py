import numpy as np
import matplotlib.pyplot as plt
# number 1

Apower = np.array([[1/3, 1/2, 1/2, 1/2, 1],[1/3, 0, 0, 1/2, 0],[0, 1/2, 1/2, 0, 0],[1/3, 0, 0, 0, 0],[0, 0, 0, 0, 0]])
x0 = np.array ([[1], [0], [0], [0] ,[0]])
 
eigvecpower = np.array([]) #for B
for iterspower in range (1, 101):
    
    product = Apower@x0 #dot product between large Apower and initial guess, (Ax)
    xnew = product/np.linalg.norm(product,2) #X_{n+1} = Ax_n / ||Ax_n||
    step = np.linalg.norm(xnew - x0)
    
    if step < 1e-5:
        print(iterspower)
        eigvecpower = np.abs(xnew) #for B
        eigvalpower = (xnew.T @ Apower @ xnew) / (xnew.T @ xnew) #for C
        print(eigvalpower)
        break
    
    x0 = xnew
    
# number 2

#2a)

def f(v):
    x = v[0]; y = v[1]; z = v[2] #setting v variables to x y z specifically.
    f1 = z - np.exp(x)
    f2 = z - 1 + np.exp(x+y)
    f3 = y - x*z - z**3
    fv = np.array([f1, f2, f3]) #[f1], [f2], [f3] no work bc it breaks the dimension calculation ig idk
    return fv

def J(v):
    x = v[0]; y = v[1]; z = v[2]
    
    f1dx = -np.exp(x)
    f1dy = 0
    f1dz = 1
    
    f2dx = np.exp(x+y)
    f2dy = np.exp(x+y)
    f2dz = 1
    
    f3dx = -z
    f3dy = 1
    f3dz = -x-3*z**2
    
    Jv = np.array ([[f1dx, f1dy, f1dz],
                    [f2dx, f2dy, f2dz],
                    [f3dx, f3dy, f3dz]])
    
    return Jv

v = np.array([0,
              0,
              0]) #initial guess
fv0 = f(v)
Jv0 = J(v)

#2b)

#vn+1 = vn + deltavn
#J(vn) deltavn = -f(vn)

for i in range(0,100):
    deltavn = -np.linalg.solve(J(v),f(v)) #its like Ax, it must be a square times a one column vector
    v = v + deltavn
    
    if np.linalg.norm(f(v), 2) < 1e-5:
        vnonlin = v
        numiternonlin = i
        break
print(vnonlin)
print(numiternonlin)

# number 3

imggray = np.load('assignments/imggray.npy')

U, S, VT = np.linalg.svd(imggray)
Smat = np.zeros(np.shape(imggray)) #makes a matrix in the shape (dimension) of imggray
Smat[0:np.min(np.shape(imggray)), 0:np.min(np.shape(imggray))] = np.diag(S)
#--> makes a matrix w min imggray dimensino (200x200) and makes it a diagonal matrix


# 3a)
u1 = U[:, 0] # 200 x 1
v1 = VT[0, :] # 1 x 300
σ1 = Smat[0,0] # 1 x 1

Arank1 = np.outer(u1, v1) * σ1

print(np.shape(Arank1))


# 3b)
u10 = U[:, 0:10] # 200 x 10
v10 = VT[0:10, :] # 10 x 300
σ10 = Smat[0:10,0:10] # 10 x 10

Arank10 = (u10 @ σ10) @ v10

print(np.shape(Arank10))

#3c)

sum = 0
r = 0
for sn in S: #does all eigenvalues
    sum += sn
    r += 1
    
    if sum >= 0.8*np.sum(S): #we only want 80% of the sum
        break
    

rsvd = r - 1 #54
usvd = U[:, 0:rsvd] # 200 x 54
vsvd = VT[0:rsvd, :] # 54 x 300
Σrsvd = Smat[0:rsvd,0:rsvd] # 54 x 54 (only counts diagonal units)

# Arankr = (usvd @ vsvd) @ Σrsvd doesnt work bc its ( 200 x 54 ) x (54 x 300) x (54 x 54) and that doesnt work
Arankr = (usvd @ Σrsvd) @ vsvd #gives us 200 x 300
print(np.shape(Arankr))

#3d)

#compression is rsvd + usvd + Σrsvd
compressionfraction = (200 + 300 + 1) * rsvd / 60000

fig, ax = plt.subplots(2,2)
ax[0,0].imshow(Arank1, cmap='gray')
ax[0,1].imshow(Arank10, cmap='gray')
ax[1,0].imshow(Arankr, cmap='gray')
ax[1,1].imshow(imggray, cmap='gray')
plt.show()