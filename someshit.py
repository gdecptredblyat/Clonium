"""import math
k = 256275
z = 0
j = 0
f = 19
for i in range (2, 2020):
    z = i
    while z > 0.99:
        j = j + 1
        z = int(z/f)
print(j)

import numpy as np
import math
from matplotlib import pyplot as plt
def p(x):
    return x/(1-x)
def q(x):
    return math.exp(x)/(1-x)
def f(x):
    return x

def progonka(p, q, f, a, b, c1, c2, c, d1, d2, d, n):
    u = []
    v = []
    h = (b - a) / n
    alp = 0
    bet = c1 * h - c2
    gam = c2
    phi = h * c
    x = np.zeros(n + 1)
    x[0] = a
    y = []
    v.append(-gam / (bet + alp * (-gam / bet)))
    u.append((phi - alp * phi / bet) / (bet + alp * v[0]))
    for i in range(1, n):
        x[i] = x[0] + i * h
        alp = 1 - (1 / 2) * p(x[i]) * h
        bet = q(x[i]) * (h ** 2) - 2
        gam = 1 + (1 / 2) * p(x[i]) * h
        phi = f(x[i]) * (h ** 2)
        v.append(-gam / (bet + alp * v[i - 1]))
        u.append((phi - alp * u[i - 1]) / (bet + alp * v[i]))
    alp = -d2
    bet = h * d1 + d2
    phi = h * d
    v.append(0)
    u.append((phi - alp * u[n - 1]) / (bet + alp * v[n]))
    y.append(u[n])
    x[n] = b
    for i in range(n - 1, -1, -1):
        y.insert(0, u[i] + v[i] * y[0])
    return ([x, y])

result = progonka(p, q, f, 2, 4, 0, 1, 1, 0.28, 0.34, 0.14, 100)
plt.grid()
plt.plot(result[0], result[1])
plt.show()

import numpy as np
from matplotlib import pyplot as plt
p = lambda x: 1/x
q = lambda x: -1/x**2
f = lambda x: np.sin(x)


def prog(a, b, c1, c2, c, d1, d2, d, n):
    v = []
    u = []
    h = (b-a)/n
    alfa = 0
    bet = c1*h - c2
    gam = c2
    fi = h*c
    x = np.zeros(n+1)
    x[0] = a
    y = []
    v.append(-gam/(bet+alfa*(-gam/bet)))
    u.append((fi-alfa*fi/bet)/(bet+alfa*v[0]))
    for i in range(1,n):
        x[i] = x[0]+i*h
        alfa = 1-(1/2)*p(x[i])*h
        bet = q(x[i])*(h**2)-2
        gam = 1+(1/2)*p(x[i])*h
        fi = f(x[i])*(h**2)
        v.append(-gam/(bet+alfa*v[i-1]))
        u.append((fi - alfa * u[i-1]) / (bet + alfa * v[i]))
    alfa = -d2
    bet = h*d1 + d2
    gam = 0
    fi = h*d
    v.append(0)
    u.append((fi - alfa * u[n - 1]) / (bet + alfa * v[n]))
    y.append(u[n])
    x[n] = b
    for i in range(n-1,-1,-1):
        y.insert(0, u[i]+v[i]*y[0])
    return([x,y])

ans = prog(1., 2., 1., 0., 0., 0.4, 0., 0, 100)

plt.grid()
plt.plot(ans[0], ans[1],)
plt.show()

import numpy as np
def f(x):
    return np.array([np.sin(x[1])/2 - 0.8, 0.8-np.cos(x[0]+0.5)])
def SITERS(f, x0, eps):
    xk = np.array(x0)
    xk1 = f(xk)
    k = 1
    while(np.linalg.norm(xk1-xk)>eps):
        k+=1
        xk = xk1
        xk1 = f(xk)
    return [xk1, k]
result,iter = SITERS(f, np.array([1,1]),1e-6)[0], SITERS(f, np.array([1,1]),1e-6)[1]
print(result, iter)
print([np.sin(result[1]) - 1.6-2*result[0], 0.8-np.cos(result[0]+0.5)-result[1]])

import numpy as np
A = np.array([[0.92, -0.83, 0.62], [0.24, -0.54, 0.43], [0.73, -0.81, -0.67]])
b = np.array([2.15, 0.62, 0.88])
M = []
result = []
for i in range(3):
    m = 0
    mi = 0
    C = []
    d = []
    for k in range(3):
        if abs(A[i][k]) > m:
            m = abs(A[i][k])
            mi = k
    b[i] /= m
    A[i] /= m
    M.append(mi)
    for j in range(3):
        if not(j == i):
            b[j] -= b[i]*A[j][mi]
            A[j] -= A[i]*A[j][mi]
    d = b[mi]
    b[mi] = b[i]
    b[i] = d
for i in range(3):
    result.append(b[M[i]])
print(result)









import numpy as np
import math
def do_L(A):
    L = np.ndarray(A.shape)
    n = 4
    for i in range(n):
        for j in range(i):
            temp = 0
            for k in range(j):
                temp += L[i][k] * L[j][k]
            L[i][j] = (A[i][j] - temp) / L[j][j]
        for j in range(i + 1, n):
            L[i][j] = 0
        temp = A[i][i]
        for k in range(i):
            temp -= L[i][k] * L[i][k]
        L[i][i] = math.sqrt(temp)
    return L

A = np.array([[1.15, 0.62, -0.83, 0.92],
              [0.82, -0.54, 0.43, -0.25],
              [0.24, 1.15, -0.33, 1.42],
              [0.73, -0.81, 1.27, -0.67]])
b = np.array([2.15, 0.62, -0.62, 0.88])
result = np.linalg.cholesky(A)

print(result)



z = 0
j = 0
q = 10
n = 2039
u=0
for i in range(1, 10001):
    z = i
    while z > 0:
        j = j + 1
        if j==n:
            z=i
            z=str(z)
            print(z[u], i)
            break
        z = int(z / q)
        u += 1
    u=0

n=2019
u=0
while n > 0:
    u+=1
    n=int(n/10)
while n>0:
    if u!=1:
        n=n-9*(10**u)
    for i in range (10)
        if n-i*(10**u) == 0:
            
    for k in range(u):
        if n%(10**k)==0:
            for i in range(1,10):
                n = n - i*(10**u)
        u+=1
        mnps(A, b, 1e-6)
        if(n == 0)
            """""


import numpy as np
import math



A = np.array([[4, 1, 1], [1, 7.4, -1], [1, -1, 9.4]])
b = np.array([1, -2, 3])
def f(xk):
    return 2*xk[0]*xk[0] + 3.7*xk[1]*xk[1]+4.7*xk[2]*xk[2] + xk[0]*xk[1] - xk[1]*xk[2] + xk[0]*xk[2] + xk[0]-2*xk[1]+3*xk[2]+7
def mngs(A, b, x0, eps) :
    xk = np.array([0, 0, 0])
    xk1 = x0

    while np.linalg.norm(xk1-xk) > eps :
        xk = xk1
        qk = A @ xk + b
        mk = (-qk @ qk)/(qk @ (A @ qk))
        xk1 = xk + mk * qk
    return xk1
"""def mnps(A, b, x0, eps) :
    j=0
    xk = np.array([0, 0, 0])
    xk1 = x0
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
b = mnps (A, b, np.array([0,0,0]), 1e-6)
"""







k = mngs(A, b,np.array([1,0,0]), 1e-6)

print(k, f(k))
result = np.linalg.solve(A, -b)
print(result, f(result))



