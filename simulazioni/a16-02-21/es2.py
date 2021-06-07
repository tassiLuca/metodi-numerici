#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESAME 16 Febbraio 2021 - Esercizio 1.
INTEPOLAZIONE POLINOMIALE
"""

import numpy as np
import math
import matplotlib.pyplot as plt

def pol_lagrange(nodes, j):
    zeros = np.zeros_like(nodes)
    
    if j == 0:
        zeros = nodes[1:]
    else:
        zeros = np.append(nodes[:j], nodes[j+1:])
    
    num = np.poly(zeros)
    den = np.polyval(num, nodes[j])
    return num / den

def lagrange_interpl(nodes, ordinates, points):
    n = nodes.size
    m = points.size
    L = np.empty((n, m))
    
    for k in range(n):
        k_pol = pol_lagrange(nodes, k)
        L[k, :] = np.polyval(k_pol, points)
        
    return np.dot(ordinates, L)

a = 0
b = 2
x_axis = np.linspace(a, b)
function = lambda x : np.cos(math.pi * x) + np.sin(math.pi * x)
nodes = np.array([1, 1.5, 1.75])

pol_intrpl = lagrange_interpl(nodes, function(nodes), x_axis)

plt.plot(x_axis, function(x_axis), x_axis, pol_intrpl, nodes, function(nodes), '*')
plt.legend(["f(x)", "Polinomio interpolante", "Nodi equispaziati"])
plt.show()

'''
Osservando che l'errore di interpolazione in x = 0.75 è pari a 0, si può facilmente intuire che x = 0.75 è 
un nodo interpolatorio per il polinomio calcolato. Quindi ricalcolare il polinomio interpolante dandodogli in 
input anche quest'ultimo non causerà nessun effetto.
'''
x = 0.75
resto = abs(function(x) - lagrange_interpl(nodes, function(nodes), np.array([x])))
print(resto)

np.append(nodes, x)
pol_intrpl2 = lagrange_interpl(nodes, function(nodes), x_axis)

plt.plot(x_axis, function(x_axis), x_axis, pol_intrpl2, nodes, function(nodes), '*')
plt.legend(["f(x)", "Polinomio interpolante 2", "Nodi equispaziati"])
plt.show()