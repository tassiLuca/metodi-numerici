#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Appello 4 Settembre 2020 - Esercizio 1
--------------------------------------
ZERI DI FUNZIONI
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from sympy.utilities import lambdify

x = sym.symbols('x')
fname  = sym.log(x**2 + 1) + 3 * x
dfname = sym.diff(fname, x, 1)

f  = lambdify(x, fname, np)
df = lambdify(x, dfname, np)
a = -1
b = 2

# grafico la funzione
x_axis = np.linspace(a, b)
plt.plot(x_axis, f(x_axis))
plt.legend(["f(x) = " + str(fname)])
plt.grid(True)
plt.show()

def corde(f, x0, m, tolx, tolf, nmax):
    it = 0
    xk = []
    prv = x0
    f_prv = f(prv)
    
    while True:
        it += 1
        nxt = prv - (f_prv / m)
        f_nxt = f(nxt)
        xk.append(nxt)
        
        if it >= nmax or abs(nxt - prv) < tolx * abs(nxt) or abs(f_nxt) < tolf:
            if it >= nmax:
                print("Raggiunto numero massimo di iterazioni.")
            break 
        else:
            prv = nxt
            f_prv = f_nxt
                
    return nxt, xk, it


def plot_xk(sol, xk, it, m, x0):
    ordine = stima_ordine(xk, it)
    plt.semilogy(range(0, it), np.abs(xk), 'x')
    plt.title("m = " + str(m) + ", x0 =" + str(x0) + "\n" +
              "Soluzione =" + str(sol) + " raggiunta con " + str(it) + " iterazioni" + "\n" +
              "Ordine = " + str(ordine))
    plt.xlabel("#it")
    plt.ylabel("abs(xk)")
    plt.grid(True)
    plt.show()
    
def stima_ordine(xk, it):
    p = []
    for k in range(it - 3):
        p.append(np.log(abs(xk[k + 2] - xk[k + 3]) / abs(xk[k + 1] - xk[k + 2])) / 
                 np.log(abs(xk[k + 1] - xk[k + 2]) / abs(xk[k] - xk[k + 1])))
        
    return p[-1]

toll = 1.e-12
nmax = 500

# prima scelta
m = np.arange(1.5, 3.5, 0.5)
x0 = -0.5
for k in range(len(m)):
    sol, xk, it = corde(f, x0, m[k], toll, toll, nmax)
    plot_xk(sol, xk, it, m[k], x0)
    
    
# seconda scelta
m = 3
x0 = np.array([0.25, 0.5, 1.0])
for k in range(len(x0)):
    sol, xk, it = corde(f, x0[k], m, toll, toll, nmax)
    plot_xk(sol, xk, it, m, x0[k])
    
# terza scelta
x0 = np.array([-1.0, -0.5, -0.25, 0.25, 0.5, 1.0])
for k in range(len(x0)):
    m = df(x0[k])
    sol, xk, it = corde(f, x0[k], m, toll, toll, nmax)
    plot_xk(sol, xk, it, m, x0[k])
    
    
