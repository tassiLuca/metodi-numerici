#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Appello del 4 Settembre 2020 - Esercizio 2
------------------------------------------
INTERPOLAZIONE POLINOMIALE
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import sympy as sym
from sympy.utilities import lambdify

x = sym.symbols('x')
fname = 2**x + x**2 - 8
f = lambdify(x, fname, np)
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


def chebychev_nodes(a, b, n):
    t1 = (a + b) / 2
    t2 = (b - a) / 2
    
    nodes = np.zeros((n + 1, ))
    for i in range(n + 1):
        nodes[i] =  t1 + t2 * np.cos(((2 * i + 1) * math.pi) / (2 * (n + 1)))
        
    return nodes

'''
a) si determina il polinomio di interpolazione p della funzione f che si ottiene dalla formula di Lagrange
   relativa ai nodi -1, 1, 2, 3;
   
b) si disegnano, in una stessa finestra, il grafico della funzione f (x) e del polinomio interpolante p(x)
   insieme ai punti di interpolazione;
'''

points = np.linspace(a, b)
nodes = np.array([-1, 1, 2, 3], dtype = float)
pol = lagrange_interpl(nodes, f(nodes), points)

plt.plot(nodes, f(nodes), '*', points, f(points), points, pol)
plt.legend(["Nodi di interpolazione", "f(x)", "Polinomio interpolante"])
plt.grid(True)
plt.show()

'''
c) si calcola il valore assoluto dell’errore commesso approssimando f(0) con p(0) e si confronta tale valore
   assoluto con la stima teorica dell’errore di interpolazione in 0;

   Si ricordi che: 
       
                             r_{n+1}(x) = omega_{n+1}(x) * df^{(n+1)}(c) / (n+1)! 
                              
   dove c è un punto appartente ad (a, b) e omega = (x - x0) * (x - x1) * ... * (x - xn) (vd slide 22).
   Visto che c non è noto a priori, conoscendo mu_{n+1} = max_{x in [a, b]} df^{(n+1)}(x) si uò stimare con:
       
                           |r_{n+1}(x)| <= |omega_{n+1}(x)| * mu_{n+1} / (n+1)! 
   (vedi slide 25).
'''

err = abs(f(0) - lagrange_interpl(nodes, f(nodes), np.array([0.])))
print("Errore = ", err)

# N.B. qui si è interpolato con 4 nodi => n + 1 = 4 
# qui calcolo la derivata quarta di fname, omega e il massimo della derivata quarta di f
df4_name = sym.diff(fname, x, 4)
df4 = lambdify(x, df4_name, np)     
omega  = lambda x : (x - 1) * (x + 1) * (x - 2) * (x - 3)
# calcolo df^4 in [1, 3] => array di valutazione di df^4(x) con x in [1, 3] => calcolo il valore massimo
mu = np.max(df4(np.arange(a, b + 1, 1, dtype = float)))
stima_err = abs(omega(0.) * mu) / math.factorial(4)
print("Stima errore = ", stima_err)

'''
d) si calcola un’approssimazione della costante di Lebesgue nel caso si scelgano i nodi equispaziati -1,
   0, 1, 2, 3 e la si confronti con l’approssimazione che si otterrebbe utilizzando i 5 nodi di Chebyshev
   sull’intervallo [-1, 3].
'''

equi_nodes = np.arange(-1, 4, 1, dtype = float)
cheb_nodes = chebychev_nodes(-1, 3, 4)
equi_pol = lagrange_interpl(equi_nodes, f(equi_nodes), points)
cheb_pol = lagrange_interpl(cheb_nodes, f(cheb_nodes), points)

plt.plot(equi_nodes, f(equi_nodes), '*', cheb_nodes, f(cheb_nodes), 'o', points, f(points), points, equi_pol, points, cheb_pol)
plt.legend(["equispaziati", "chebychev", "f(x)", "interpolante su equispaziati", "interpolante su chebychev"])
plt.grid(True)
plt.show()

# calcolo costanti di Lebesgue
equi_lebesgue_acc = np.zeros_like(points)
cheb_lebesgue_acc = np.zeros_like(points)

for i in range(5):
    equi_lebesgue_acc = equi_lebesgue_acc + np.abs(np.polyval(lagrange_pol(equi_nodes, i), points))
    cheb_lebesgue_acc = cheb_lebesgue_acc + np.abs(np.polyval(lagrange_pol(cheb_nodes, i), points))
    
equi_lebesgue = np.max(equi_lebesgue_acc)
cheb_lebesgue = np.max(cheb_lebesgue_acc)
print("Costante di Lebesgue per nodi equispaziati = ", equi_lebesgue)
print("Costante di Lebesgue per nodi di Chebychev = ", cheb_lebesgue)

