#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 17:10:53 2021

@author: lucatassi
"""

import numpy as np
import matplotlib.pyplot as plt

fname = lambda x : 2**x + x**2 - 8
a = -1
b = 3

def lagrange_pol(nodes, j):
    zeros = np.zeros_like(nodes)
    
    if j == 0:
        zeros = nodes[1:]
    else:
        zeros = np.append(nodes[0:j], nodes[j+1:])
        
    num = np.poly(zeros)
    den = np.polyval(num, nodes[j])
    return num / den

def lagrange_interpl(nodes, ordinates, points):
    n = nodes.size
    m = points.size
    L = np.zeros((n, m))
    
    for k in range(n):
        k_pol = lagrange_pol(nodes, k)
        L[k, :] = np.polyval(k_pol, points)

    return np.dot(ordinates, L)

points = np.linspace(a, b)
nodes = np.array([-1, 1, 2, 3], dtype = float)
pol = lagrange_interpl(nodes, fname(nodes), points)

plt.plot(nodes, fname(nodes), '*', points, fname(points), points, pol)
plt.legend(["Nodi di interpolazione", "f(x)", "Polinomio interpolante"])
plt.grid(True)
plt.show()