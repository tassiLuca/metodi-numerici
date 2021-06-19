#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Appello 12 Aprile 2020 - Esercizio 1
------------------------------------
INTEGRAZIONE NUMERICA E STABILITÀ.
"""

import numpy as np
import matplotlib.pyplot as plt

def trapezi_composita(fname, a, b, n):
    h = (b - a) / n
    nodes = np.arange(a, b + h, h)
    f = fname(nodes)
    return h / 2 * (f[0] + 2 * np.sum(f[1:n]) + f[n])

def trapezi_auto(fname, a, b, toll):
    max_steps = 2048
    steps = 1
    err = 1
    
    integral = trapezi_composita(fname, a, b, steps)
    while steps <= max_steps and err > toll:
        steps *= 2
        integral_double_steps = trapezi_composita(fname, a, b, steps)
        err = abs(integral - integral_double_steps) / 3
        integral = integral_double_steps
    
    return integral, steps

fname = lambda x, n : x**n / (x + 10)
a = 0
b = 1
toll = 1.e-6
n = np.arange(1, 31, 1)
integral = []

yn = np.zeros((30, ), dtype = float)
yn[0] = np.log(11) - np.log(10)
for k in range(1, 30, 1):
    yn[k] = 1 / k - 10 * yn[k - 1]


zn = np.zeros((31, ), dtype = float)
zn[30] = 0.0
for k in range(30, 0, -1):
    zn[k - 1] = (1 / 10) * (1 / k - zn[k])

for n_i in n:
    sol, steps = trapezi_auto(lambda x : fname(x, n_i), a, b, toll)
    integral.append(sol)
    
plt.plot(integral, 'r-o')
plt.show()
print(integral)

err_yn = np.abs(yn - integral) / np.abs(integral)
err_zn = np.abs(zn[0:30] - integral) / np.abs(integral)

plt.semilogy(range(0, 30), err_yn, 'g-.', range(0, 30), err_zn, 'b--')
plt.grid(True)
plt.xlabel("n")
plt.legend(["Errore relativo yn", "Errrore relativo zn"])
plt.show()