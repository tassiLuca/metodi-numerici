#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APPELLO DELL'8-10-20 - ESERCIZIO 2
----------------------------------
INTERPOLAZIONE
"""

import numpy as np
import math
import matplotlib.pyplot as plt

def pol_lagrange(nodes, j):
    zeros = np.zeros_like(nodes)
    
    if j == 0:
        zeros = nodes[1:]
    else:
        zeros = np.append(nodes[0:j], nodes[j+1:])

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

def chebychev_nodes(a, b, n):
    t1 = (a + b) / 2
    t2 = (b - a) / 2
    nodes = np.zeros((n + 1,))
    for i in range(n + 1):
        nodes[i] = t1 + t2 * np.cos(((2 * i + 1) * math.pi) / (2 * (n + 1)))
        
    return nodes

fname = lambda x : 1 / (1 + 900 * x**2)
a = -1
b = +1

points = np.linspace(a, b, 200)

equi_lebesgue = np.zeros(6)
cheb_lebesgue = np.zeros(6)
equi_lebesgue_acc = np.zeros(200)
cheb_lebesgue_acc = np.zeros(200)

fig = 1
i = 0
for n in range(5, 31, 5):
    # nodi equispaziati
    equi_nodes = np.linspace(a, b, n + 1)
    equi_ordinates = fname(equi_nodes)
    equi_pol = lagrange_intrpl(equi_nodes, equi_ordinates, points)
    # calcolo errore
    equi_err = abs(fname(points) - equi_pol)
    # calcolo lebesgue
    for k in range(n + 1):
        pol = pol_lagrange(equi_nodes, k)
        equi_lebsgue_acc = equi_lebesgue_acc + np.abs(np.polyval(pol, points))
    equi_lebesgue[i] = np.max(equi_lebsgue_acc)    
    
    plt.subplots_adjust(hspace = 0.7, wspace = 0.3)
    plt.subplot(3, 2, fig)
    plt.title("n = " + str(n))
    plt.plot(points, equi_err)
    fig += 1
    i += 1
    
plt.show()  

fig = 1
i = 0
for n in range(5, 31, 5):
    # nodi di Chebychev
    cheb_nodes = chebychev_nodes(a, b, n)
    cheb_ordinates = fname(cheb_nodes)
    cheb_pol = lagrange_intrpl(cheb_nodes, cheb_ordinates, points)
    # calcolo errore
    cheb_err = abs(fname(points) - cheb_pol)
    # calcolo lebesgue
    for k in range(n + 1):
        pol = pol_lagrange(cheb_nodes, k)
        cheb_lebsgue_acc = cheb_lebesgue_acc + np.abs(np.polyval(pol, points))
    cheb_lebesgue[i] = np.max(cheb_lebsgue_acc)
    
    plt.subplots_adjust(hspace = 0.7, wspace = 0.3)
    plt.subplot(3, 2, fig)
    plt.plot(points, cheb_err)
    plt.title("n = " + str(n))
    fig += 1
    i += 1
   
plt.show()

# grafico costanti di lebegue
plt.semilogy(range(5, 31, 5), equi_lebesgue, 'o-', range(5, 31, 5), cheb_lebesgue, '*-')
plt.legend(["Nodi equispaziati", "Nodi di Chebychev"])
plt.legend()
plt.show()
