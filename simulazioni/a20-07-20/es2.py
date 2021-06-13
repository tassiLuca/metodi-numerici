#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Appello 20 Luglio 2020 - Esercizio 2
------------------------------------
ZERI DI FUNZIONI
"""

import numpy as np
import matplotlib.pyplot as plt

f = lambda x : 20 / (1 + np.log(x**2)) - 5 * np.sin(np.exp(x)) - 16
a = 1
b = 2

x_axis = np.linspace(a, b, 100)
plt.plot(x_axis, f(x_axis))
plt.grid(True)
plt.show()

'''
a)  Scrivere la function secanti che implementa il metodo delle secanti per calcolare lo zero di f.
'''

def secanti(fname, xm1, x0, tolx, toly, max_it):
    approx = []
    it = 0
    
    xim1 = xm1
    f_xim1 = fname(xim1)
    xi = x0
    f_xi = fname(xi)
    
    while True:
        it += 1
        xip1 = xi - f_xi * ((xi - xim1) / (f_xi - f_xim1))
        f_xip1 = fname(xip1)
        approx.append(xip1)
        
        if it > max_it or abs(xip1 - xi) < tolx * abs(xip1) or abs(f_xip1) < toly:
            break
        else:
            xim1 = xi
            f_xim1 = f_xi
            xi = xip1
            f_xi = f_xip1
    
    return xip1, approx, it

'''
b)  Scrivere la function falsi che implementa il metodo di falsa posizione per calcolare lo zero di f.
'''

def sign(x):
    return np.copysign(1, x)

def regula_falsi(fname, a, b, tolx, toly, max_it):
    f_a = fname(a)
    f_b = fname(b)
    if sign(f_a) == sign(f_b):
        print("ERRORE: la funzione deve assumere valori opposti agli estremi")
        return [], [], 0
    
    eps = np.spacing(1)
    it = 0
    approx = []
    f_nxt = f_a
    
    while it <= max_it and abs(b - a) >= tolx + eps * max(abs(a), abs(b)) and abs(f_nxt) >= toly:
        it += 1
        nxt = a - f_a * ((b - a) / (f_b - f_a))
        f_nxt = fname(nxt)
        approx.append(nxt)
        
        if f_nxt == 0:
            break
        elif sign(f_nxt) == sign(f_a):
            a = nxt
            f_a = f_nxt
        elif sign(f_nxt) == sign(f_b):
            b = nxt
            f_b = f_nxt
        
    return nxt, approx, it

max_it = 500
toll = 1.e-6
xm1 = 1.0
x0 = 1.6

root_sec, approx_sec, it_sec = secanti(f, xm1, x0, toll, toll, max_it)
print(root_sec, approx_sec, it_sec)

root_falsi, approx_falsi, it_falsi = regula_falsi(f, a, b, toll, toll, max_it)
print(root_falsi, approx_falsi, it_falsi)

print("-------------------------------------------")

xm1 = 1.0
x0 = 2

root_sec, approx_sec, it_sec = secanti(f, xm1, x0, toll, toll, max_it)
print(root_sec, it_sec)

root_falsi, approx_falsi, it_falsi = regula_falsi(f, a, b, toll, toll, max_it)
print(root_falsi, it_falsi)