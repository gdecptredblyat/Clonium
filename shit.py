import numpy as np
import math
"""def mngs(A, b, x0, eps) :
    xk = np.array(x0)
    qk = A @ xk + b
    return qk
A = np.array([[1,1], [0,1]])
b = np.array([1,0])
x0 = np.array([1,0])
result = mngs(A, b, x0, 1e-6)
print(result)


"""
A = np.array([[4, 1, 1], [1, 7.4, -1], [1, -1, 9.4]])
b = np.array([1, -2, 3])
def f(xk):
    return 2*xk[0]*xk[0] + 3.7*xk[1]*xk[1]+4.7*xk[2]*xk[2] + xk[0]*xk[1] - xk[1]*xk[2] + xk[0]*xk[2] + xk[0]-2*xk[1]+3*xk[2]+7
def mnps(A, b, x0, eps) :
    j=0
    xk = np.array([1, 0, 0])
    xk1 = np.array([0,0,0])
    while np.linalg.norm(xk1-xk) > eps :
        ort = choose_ort(j)
        xk = xk1
        mk = (-ort @ (A @ xk + b))/(ort @ (A @ ort))
        xk1 = xk + mk * ort
        print(f(xk1))
        j+=1
    return xk1
def choose_ort(j):
    ort = np.array([0, 0, 0])
    ort[j%3] += 1
    return ort
c = mnps (A, b, np.array([0,0,0]), 1e-6)
print(c, f(c))
result = np.linalg.solve(A, -b)
print(result, f(result))

"""
j = 0
def choose_ort():
    ort = np.array([0, 0, 0])
    n=j%3
    ort[n] += 1
    j+=1
    print(ort,j)

choose_ort()
choose_ort()
choose_ort()
choose_ort()
choose_ort()
choose_ort()
choose_ort()
"""
