#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Appello 20 Febbraio 2020 - Esercizio 2
"""

import numpy as np
import matplotlib.pyplot as plt

fname = lambda x : x**5 + 23 / 12 * x**4 - 95 / 12 * x**3 - 173 / 16 * x**2 + 115 / 24 * x + 325 / 48
a, b = -2, 2

x_axis = np.linspace(a, b, 100)
plt.plot(x_axis, fname(x_axis))
plt.legend(["f(x)"])
plt.grid(True)

def sign(x):
    return np.copysign(1, x)

def bisection(fname, a, b, toll):
    f_a = fname(a)
    f_b = fname(b)
    if sign(f_a) == sign(f_b):
        print("ERROR: sign(f(a)) = sign(f(b))")
        return [], [], 1
    
    eps = np.spacing(1)
    approx = []
    it = 0
    max_it = np.log((b - a) / toll) / np.log(2)
    
    while it <= max_it and abs(b - a) >= toll + eps * max(abs(a), abs(b)):
        it += 1
        m = a + (b - a) / 2
        f_m = fname(m)
        approx.append(m)
        
        if f_m == 0:
            break
        elif sign(f_m) == sign(f_a):
            a = m
            f_a = f_m
        else:
            b = m
            f_b = f_m
        
    return m, approx, it

toll = 1.e-8
root, approx, it = bisection(fname, a, b, toll)

plt.plot(root, fname(root), 'ro')
plt.show()

def composite_simpson(fname, a, b, n):
    h = (b - a) / (2 * n)
    nodes = np.arange(a, b + h, h)
    f = fname(nodes)
    return h / 3 * (f[0] + 2 * np.sum(f[2:2*n:2]) + 4 * np.sum(f[1:2*n:2]) + f[2*n])

def simpson_auto(fname, a, b, toll):
    steps = 1
    err = 1
    max_steps = 2048
    
    integral = composite_simpson(fname, a, b, steps)
    while steps <= max_steps and err >= toll:
        steps *= 2
        integral_double_steps = composite_simpson(fname, a, b, steps)
        err = abs(integral - integral_double_steps) / 15
        integral = integral_double_steps
    
    return integral, steps
    
sol = - 121 / 20
toll = 1.e-5
integral, steps = simpson_auto(fname, a, b, toll)
print("Integrale esatto =", sol)
print("Integrale approssimato con la formula di Simpson automatica =", integral, "--> ", steps, " sottointervalli")