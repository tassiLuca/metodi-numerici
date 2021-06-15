#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APPELLO DELL'8-10-20 - ESERCIZIO 2
----------------------------------
INTERPOLAZIONE POLINOMIALE
"""

import numpy as np
import math
import matplotlib.pyplot as plt

fname = lambda x : 1 / (1 + 900 * x**2)
a, b = -1, 1

'''
a)  si determinano i due polinomi di interpolazione di grado n = 5 : 30 : 5 della funzione f.
'''

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

def chebyshev_nodes(a, b, n):
    t1 = (a + b) / 2
    t2 = (b - a) / 2
    x = np.zeros((n + 1, ))
    for i in range(n + 1):
        x[i] = t1 + t2 * np.cos(((2 * i + 1) * math.pi) / (2 * (n + 1)))
    return x

'''
b)  dopo aver creato la Figura 1 e suddiviso la finestra grafica in 3 × 2 sottofinestre, si disegnano nelle 
    6 sottofinestre i grafici degli errori del polinomio interpolatorio su nodi equispaziati al variare di n.
c)  dopo aver creato la Figura 1 e suddiviso la finestra grafica in 3 × 2 sottofinestre, si disegnano nelle 
    6 sottofinestre i grafici degli errori del polinomio interpolatorio sui nodi chebyshev al variare di n.
d)  si calcolano le approssimazioni della costante di Lebesgue sia nel caso di nodi equispaziati che di Chebyshev,
    e si rappresentano in un grafico in scala semilogaritmica su y al variare di n (Figura 3).
'''   

def lebesgue(nodes, points):
    lebesgue_acc = np.zeros_like((points, ))
    
    for k in range(nodes.size):
        k_pol = lagrange_pol(nodes, k)
        lebesgue_acc = lebesgue_acc + np.abs(np.polyval(k_pol, points))
        
    return np.max(lebesgue_acc)

start, stop, step = 5, 30, 5
points = np.linspace(a, b, 100)

size = int((stop + step - start) / step)
equi_err = np.zeros((size, ))
equi_lebesgue = np.zeros((size, ))
cheb_err = np.zeros((size, ))
cheb_lebesgue = np.zeros((size, ))

# =============================================================================
# NODI EQUISPAZIATI
# =============================================================================
i = 1
for n in range(start, stop + step, step):
    # calcolo polinomio interpolante su nodi equispaziati
    equi_nodes = np.linspace(a, b, n + 1)
    equi_ordinates = fname(equi_nodes)
    equi_pol = lagrange_intrpl(equi_nodes, equi_ordinates, points)
    # calcolo errore 
    equi_err[i - 1] = np.abs(fname(points) - equi_pol)
    # calcolo costante di lebesgue
    equi_lebesgue[i - 1] = lebesgue(equi_nodes, points)
    
    # grafico gli errori
    plt.subplot(3, 2, i)
    plt.subplots_adjust(left = 0, right = 1, wspace = 0.3, hspace = 0.5)
    plt.plot(points, equi_err)
    plt.legend(["n =" + str(n)])
    i += 1
plt.show()

# =============================================================================
# NODI DI CHEBYSHEV
# =============================================================================
i = 1
for n in range(start, stop + step, step):
    # calcolo polinomio interpolante su nodi di chebyshev
    cheb_nodes = chebyshev_nodes(a, b, n)
    cheb_ordinates = fname(cheb_nodes)
    cheb_pol = lagrange_intrpl(cheb_nodes, cheb_ordinates, points)
    # calcolo errore 
    cheb_err = np.abs(fname(points) - cheb_pol)
    # calcolo costante di lebesgue
    cheb_lebesgue[i - 1] = lebesgue(cheb_nodes, points)
    
    # grafico gli errori
    plt.subplot(3, 2, i)
    plt.subplots_adjust(left = 0, right = 1, wspace = 0.3, hspace = 0.5)
    plt.plot(points, cheb_err)
    plt.legend(["n =" + str(n)])
    i += 1
plt.show()

# grafico le costanti di Lebesgue
n = np.arange(start, stop + step, step)
plt.semilogy(n, cheb_lebesgue, '-o', n, equi_lebesgue, '-o')
plt.legend(["Chebyshev", "Equi"])
plt.grid(True)
plt.show()
