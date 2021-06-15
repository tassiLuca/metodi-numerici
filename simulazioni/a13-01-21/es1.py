#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Appello del 13 Gennaio 2021 - Esercizio 1
-----------------------------------------
ZERI DI FUNZIONI.
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from sympy.utilities import lambdify
from scipy.optimize import fsolve

x = sym.symbols('x')
fname = sym.log(x + 1) + sym.sqrt(sym.cos(x**2) + 2)
dfname = sym.diff(fname, x, 1)
f  = lambdify(x, fname, np)
df = lambdify(x, dfname, np)
a = -1
b = 1

root = fsolve(f, -0.5)
x_axis = np.linspace(a, b, 100)
plt.plot(x_axis, f(x_axis), root, f(root), 'o')
plt.grid(True)
plt.show()

def corde(f, x0, m, tolx, tolf, max_it):
    xk = []
    it = 0
    
    prv = x0
    f_prv = f(prv)
    while True:
        it += 1
        nxt = prv - (f_prv / m)
        f_nxt = f(nxt)
        xk.append(nxt)
        
        if it > max_it or abs(nxt - prv) < tolx * abs(nxt) or abs(f_nxt) < tolf:
            if it > max_it:
                print("Raggiunto massimo numero di iterazioni!")
            break
        else:
            prv = nxt
            f_prv = f_nxt
    
    return nxt, xk, it

    
def newton(fname, dfname, trigger, tolx, tolf, max_it):
    it = 0
    approx = []
    
    prv = trigger
    f_prv = fname(prv)
    df_prv = dfname(prv)
    while True:
        it += 1
        nxt = prv - (f_prv / df_prv)
        f_nxt = fname(nxt)
        df_nxt = dfname(nxt)
        approx.append(nxt)
        
        if it > max_it or abs(nxt - prv) < tolx * abs(nxt) or abs(f_nxt) < tolf:
            if it > max_it:
                print("Raggiunto massimo numero di iterazioni.")
            break
        else:
            prv = nxt
            f_prv = f_nxt
            df_prv = df_nxt
    
    return nxt, approx, it
        

def stima_ordine(xk, it):
    p = []
    
    for k in range(it - 3):
        p.append(np.log(abs(xk[k + 2] - xk[k + 3]) / abs(xk[k + 1] - xk[k + 2])) / 
                 np.log(abs(xk[k + 1] - xk[k + 2]) / abs(xk[k] - xk[k + 1])))

    return p[-1]

def display_results(root, it, ordine, x0, m = None):
    plt.semilogy(range(0, it), np.abs(xk))
    if m != None:
        plt.title("[METODO DELLE CORDE] x0 = " + str(x0) + ", m = " + str(m) + "\n" + 
                  "Radice = " + str(root) + " raggiunta con " + str(it) + " iterazioni. \n" + 
                  "Ordine = " + str(ordine))
    else:
        plt.title("[METODO DI NEWTON]: x0 = " + str(x0_i) + "\n" + 
                  "Radice = " + str(root) + " raggiunta con " + str(it) + " iterazioni. \n" + 
                  "Ordine = " + str(ordine))
    plt.grid(True)
    plt.show()
        
toll = 1.e-12
max_it = 500

# ================================= CORDE =================================
x0 = 0
m = [2, 3, 4, 5]
for m_i in m:
    root, xk, it = corde(f, x0, m_i, toll, toll, max_it)
    ordine = stima_ordine(xk, it)
    display_results(root, it, ordine, x0, m_i)

# ================================ NEWTON =================================
x0 = [0, -1/4, -1/2, -3/4]
for x0_i in x0:
    root, xk, it = newton(f, df, x0_i, toll, toll, max_it)
    ordine = stima_ordine(xk, it)
    display_results(root, it, ordine, x0_i)

    