#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Appello 20 Febbraio 2020 - Esercizio 1
"""

import numpy as np
import matplotlib.pyplot as plt

def interpN(x, y):
    n = x.size
    c = y.copy()
    for k in range(1, n):
        c[k:n] = (c[k:n] - c[k-1:n-1]) / (x[k:n] - x[0:n-k])
    return c

def hornerN(c, x, z):
    n = c.size
    pval = c[n-1] * np.ones_like(z)
    for k in range(n-2, -1, -1):
        pval = (z - x[k]) * pval + c[k]
    return pval
 
def lagrange_pol(nodes, j):
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
        k_pol = lagrange_pol(nodes, k)
        L[k, :] = np.polyval(k_pol, points)
        
    return np.dot(ordinates, L)

fname = lambda x : np.log(x + 3) - 1 / 2 * np.sin(x - 1)
a, b, num = -1, 3, 50
start, stop, step = -1, 3, 1

points = np.linspace(a, b, num)
nodes = np.arange(start, stop + step, step)
ordinates = fname(nodes)

c = interpN(nodes, ordinates)
yy = hornerN(c, nodes, points)
nerror = np.abs(fname(points) - yy)
max_nerror = np.linalg.norm(nerror, np.inf)
max_nabscissa = np.argmax(nerror) * (b - a) / num + a

lpol_intrpl = lagrange_intrpl(nodes, ordinates, points)
lerror = np.abs(fname(points) - lpol_intrpl)
max_lerror = np.linalg.norm(lerror, np.inf)
max_labscissa = np.argmax(lerror) * (b - a) / num + a

# ================================ grafici ===================================
rows, cols = 1, 2

fig  = 1
plt.subplot(rows, cols, fig)
plt.plot(nodes, ordinates, '*', points, fname(points), points, lpol_intrpl, points, yy)
plt.legend(["Nodi", "f(x)", "lagrange", "netwon"])
plt.subplots_adjust(left = 0, right = 1)

fig += 1
plt.subplot(rows, cols, fig)
plt.plot(points, lerror, max_labscissa, max_lerror, 'x', points, nerror, max_nabscissa, max_nerror, 'x')
plt.legend(["lagrange", "max_lerror", "newton", "max_nerror"])


plt.show()
