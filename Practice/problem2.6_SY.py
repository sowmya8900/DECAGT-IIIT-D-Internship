import numpy as np
from scipy.linalg import hilbert, lu_factor, lu_solve
from numpy.linalg import cond
from scipy.linalg import cho_factor, cho_solve
from numpy import linalg
# input numpy.linalg as npla

n = 5

H = hilbert(n) # Generate Hilbert's Matrix
x = np.ones((n, 1))
#b = np.matmul(H,x)

b = np.dot(H, x) # Generate n-vector, b

print("Hilbert Matrix:\n", H) 
print("n-vector b:\n", b) 

c, low = cho_factor(H) # Cholesky factorization
xcap = cho_solve((c, low), x)
print("Cholesky factorization:\n", xcap)

residue = np.subtract(b, np.dot(H, xcap))
error = np.subtract(xcap, x)
# print(residue)
# print(error)

print("Infinity norm of the residual r: ", np.linalg.norm(residue, np.inf))
print("Infinity norm of the error: ", np.linalg.norm(error, np.inf))

'''
condition_number = []
for i in range (0, n):
    condition_number.append(np.linalg.cond(H, np.inf))
'''
condition_number = np.linalg.cond(H, np.inf)
print("condition_number: ", condition_number)

'''
cH = np.linalg.cholesky(H) # Cholesky factorization
appx = np.matmul(cH,b)
invH = np.linalg.inv(H)
ix = np.matmul(invH,b)

print(ix)
print(invH)
print(cH)

Hdash = np.abs(H).sum(axis = 1) 
print(np.max(Hdash, axis=0))
'''
# infnorm = linalg.norm(H, ord=1, axis=1)
# print(infnorm)

# error = x_cap - x
# x_cap = calc error


