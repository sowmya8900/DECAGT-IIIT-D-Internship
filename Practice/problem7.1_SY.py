from numpy.lib.polynomial import poly1d
import numpy as np
from scipy.integrate import quad

def horner(x, t):

    sol = x[0]

    for i in range(0, len(x)-1): # i = 0,1,2
        sol = sol*t + x[i+1]  # x[1,2,3]
    return sol


x = [4, 3, 2, 1] # 4x**3 + 3x**2 + 2x + 1 would be seen as: x(x(x(1) + 2) + 3) + 4
# x = [1, 2, 3, 4]
t = 2

no_of_coeff = len(x)

degree = no_of_coeff - 1

pt = poly1d(x)
print("Polynomial given:\n", pt)
print("degree of the polynomial is: ", degree)

print("Solution for the given polynomial at t =", t, "is:", horner(x, t)) # (a) part: solution for general equation

derivative = np.polyder(pt, 1) # derivative of order 1 of pt

derToPol = np.array(derivative)

print("Derivative for the given polynomial is:\n", derivative)
print("Solution for the derivative at t =", t, "is:", horner(derToPol, t)) # (b).(i) part: first derivative of pt

a = 1
b = 2
I = quad(pt, a, b)
integration = sum(list(I))

print("Solution for the integration is:", round(integration, 2)) # (b).(ii) part: first integral of pt

