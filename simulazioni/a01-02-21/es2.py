#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Appello 1 Febbraio 2021 - Esercizio 2
-------------------------------------
INTEGRAZIONE ED INTERPOLAZIONE
"""

import numpy as np
import sympy as sym
from sympy.utilities import lambdify
import math
import matplotlib.pyplot as plt

import scipy.integrate as integrate


def pol_lagrange(nodes, j):
    zeros = np.zeros_like(nodes)
    
    if j == 0:
        zeros = nodes[1:]
    else:
        zeros = np.append(nodes[:j], nodes[j+1:])
        
    num = np.poly(zeros)
    den = np.polyval(num, nodes[j])
    return num / den

def lagrange_intrpl(nodes, ordinates, points):
    n = nodes.size
    m = points.size
    L = np.zeros((n, m))

    for k in range(n):
        k_pol = pol_lagrange(nodes, k)
        L[k, :] = np.polyval(k_pol, points)
        
    return np.dot(ordinates, L)

def trapezi_composita(fname, a, b, n):
    h = (b - a) / n
    nodes = np.arange(a, b + h, h)
    f = fname(nodes)
    return h / 2 * (f[0] + 2 * np.sum(f[1:n]) + f[n])


def trapezi_toll(fname, a, b, toll):
    max_steps = 4096
    err = 1
    steps = 1
    
    integral = trapezi_composita(fname, a, b, steps)
    while steps <= max_steps and abs(err) >= toll:
        steps *= 2
        integral_double_steps = trapezi_composita(fname, a, b, steps)
        err = abs(integral - integral_double_steps) / 3
        integral = integral_double_steps
    
    return integral, steps

t = sym.symbols('t')
fname = lambda x, v : math.pi * sym.cos(x * sym.sin(t) - (v * t))

a = 0
b = math.pi
toll = 1.e-3
x = np.array([1, 13/6, 10/3, 9/2, 17/3, 41/6, 8], dtype = float)

start = 1
stop = 5
step = 2

integrals_v = []
integrals_s = []

vis = np.linspace(1, 8, 100)

for v in range(start, stop + step, step):
    for i in range(7):
        f_val = fname(x[i], v)
        print("f(t) =", f_val)
        f = lambdify(t, f_val, np)
        
        integral, steps = trapezi_toll(f, a, b, toll)
        #sol = float(sym.integrate(f_val, (t, a, b)))
        sol, ecc = integrate.quad(f, a, b)
        err = abs(integral - sol) / abs(sol)
        print("Integrale con trapezi =", integral, "raggiunto con", steps, "steps", "Errore relativo = ", err, "\n")
        
        integrals_v.append(integral)
        integrals_s.append(sol)

        # grafico funzione
        x_axis = np.linspace(a, b, 100, False)
        plt.plot(x_axis, f(x_axis))
        plt.title("f(t) = " + str(f_val) + 
                  "\n Integrale = " + str(integral) + " raggiunto con " + str(steps) + "steps" + 
                  "\n Errore relativo = " + str(err))
        plt.grid(True)
        plt.fill_between(x_axis, f(x_axis))
        plt.show()
        
    print("---------------------------")
    pol_intrpl = lagrange_intrpl(x, integrals_v, vis)
    plt.plot(x, integrals_v, 'o', vis, pol_intrpl)
    plt.show()
    
    integrals_v = []
        
    print("+++++++++++++++++++++++++++++")
    
    
