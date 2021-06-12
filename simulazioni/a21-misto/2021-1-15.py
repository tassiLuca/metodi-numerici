#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 11:13:29 2021

@author: lucatassi
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from sympy.utilities import lambdify

f1 = lambda i : (15 * ((3 / 5)**i + 1)) / (5 * (3 / 5)**i + 3)
f2 = lambda prv : 8 - (15 / prv)
f3 = lambda prvprv, prv : 108 - (815 / prv) + (1500 / (prv * prvprv))

start = 1
stop = 35
step = 1

sequence1 = []

sequence2 = []
sequence2.append(4.0)

sequence3 = sequence2.copy()
sequence3.append(17 / 4)

for i in range(start, stop + step, step):
    sequence1.append(f1(i))
    
for i in range(start, stop, step):
    sequence2.append(f2(sequence2[-1]))

for i in range(start, stop - step, step):
    sequence3.append(f3(sequence3[-2], sequence3[-1]))
    
plt.plot(sequence1, np.zeros_like(sequence1), '*', sequence2, np.zeros_like(sequence2), 'o')
plt.show()
plt.plot(sequence3, np.zeros_like(sequence3), 'x')
plt.show()

sequence1 = np.asarray(sequence1)
sequence2 = np.asarray(sequence2)
sequence3 = np.asarray(sequence3)

err2 = np.abs(sequence2 - sequence1) / np.abs(sequence1)
err3 = np.abs(sequence3 - sequence1) / np.abs(sequence1)
plt.semilogy(range(stop), err2, 'o', range(stop), err3, 'x')
plt.show()

'''

'''

x = sym.symbols('x')

g1_name = 8 - 15 / x
dg1_name = sym.diff(g1_name, x, 1)
g1 = lambdify(x, g1_name, np)
dg1 = lambdify(x, dg1_name, np)

g2_name = 108 - (815 / x) + (1500 / (x ** 2))
dg2_name = sym.diff(g2_name, x, 1)
g2 = lambdify(x, g2_name, np)
dg2 = lambdify(x, dg2_name, np)

x_axis = np.linspace(2, 100, 100)
plt.plot(x_axis, g1(x_axis), x_axis, x_axis, x_axis, g2(x_axis))
plt.plot([3], [3], 'rx', [5], [5], 'rx', [100], [100], 'rx') # punti fissi
plt.legend(["g1(x) = " + str(g1_name), "y = x", "g2(x) = " + str(g2_name)])
plt.grid(True)
plt.show()

# calcolo i punti fissi
def iterazione(gname, trigger, tolx, max_it):
    approx = []
    it = 0
    
    prv = trigger
    while True:
        it += 1
        nxt = gname(prv)
        approx.append(nxt)
        
        if it >= max_it or abs(nxt - prv) < tolx * abs(nxt):
            if it >= max_it:
                print("Max iterations reached")
            break
        else:
            prv = nxt
            
    return nxt, approx, it


toll = 1e-8
max_it = 500
trigger = 1
fp1, approx1, it1 = iterazione(g1, trigger, toll, max_it)
fp2, approx2, it2 = iterazione(g2, trigger, toll, max_it)
print("Punto fisso di g1(x) =", fp1)
print("Punto fisso di g2(x) =", fp2)

x_axis = np.linspace(3, 6)
plt.plot(x_axis, dg1(x_axis), x_axis, dg2(x_axis))
plt.plot([3], [0], 'rx', [5], [0], 'rx') # punti fissi
plt.grid(True)
plt.fill_between(x_axis, np.ones_like(x_axis), - np.ones_like(x_axis), facecolor = 'yellow', alpha = 0.4)
plt.legend(['dg1(x)', 'dg2(x)'])
plt.show()

x_axis = np.linspace(90, 110)
plt.plot(x_axis, dg2(x_axis))
plt.plot([100], [0], 'rx') # punti fissi
plt.grid(True)
plt.fill_between(x_axis, np.ones_like(x_axis), - np.ones_like(x_axis), facecolor = 'yellow', alpha = 0.4)
plt.legend(['dg2(x)'])
plt.show()