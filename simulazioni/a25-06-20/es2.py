#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESAME 26 Giugno 2020 - Esercizio 1.
INTERPOLAZIONE POLINOMIALE ED INTEGRAZIONE NUMERICA.
"""

import numpy as np
import matplotlib.pyplot as plt

'''
(a) Scrivere il proprio codice Matlab per determinare il polinomio p(x) di grado 3, in
    forma di Lagrange, che interpola la funzione f(x) su nodi equispaziati.
    
(b) Disegnare in una stessa figura i punti di interpolazione, il grafico di f e del polinomio
    di interpolazione p ottenuto al punto a).
'''

def lagrange_pol(nodes, j):
    zeros = np.zeros_like(nodes)
    
    if j == 0 :
        zeros = nodes[1:]
    else :
        zeros = np.append(nodes[:j], nodes[j+1:])
    
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

fname = lambda x : x - np.sqrt(x - 1)
a = 1
b = 3
n = 3
equi_nodes = np.linspace(a, b, n + 1)   # DA NOTARE n + 1 ANZICHÈ n!!
val_points = np.linspace(a, b, 100)
ordinates = fname(equi_nodes)
intrpl = lagrange_interpl(equi_nodes, ordinates, val_points)

# grafico f e il polinomio interpolante
plt.plot(val_points, fname(val_points), val_points, intrpl, equi_nodes, fname(equi_nodes), '*')
plt.legend(["f(x)", "Intepolante di Lagrange", "Nodi equispaziati"])
plt.show()

'''
(c) Scrivere il proprio codice Python per calcolare con la formula di Simpson composita su N sottointervalli 
    equispaziati, i valori approssimati print("INTEGRALE f(x)") degli integrali di I1 = f(x) e I2 = p(x).
'''

def simpson(fname, a, b, n):
    h = (b - a) / (2 * n)
    nodes = np.arange(a, b + h, h)
    f = fname(nodes)
    return h / 3 * (f[0] + 2 * np.sum(f[2:n*2:2]) + 4 * np.sum(f[1:n*2:2]) + f[2*n])

def simpson_auto(fname, a, b, toll):
    max_steps = 2048
    steps = 1
    err = 1
    
    integral = simpson(fname, a, b, steps)
    while steps <= max_steps and err > toll:
        steps *= 2
        integral_double_steps = simpson(fname, a, b, steps)
        err = abs(integral_double_steps - integral) / 15
        integral = integral_double_steps
        
    return integral, steps
  
'''
(d) Si stimi il numero N di sottointervalli equispaziati che servono per approssimare con la formula di Simpson 
    composita i due integrali (il cui valore esatto è rispettivamente I1 = 2.114381916835873 e I2 = 2.168048769926493) 
    nel rispetto della tolleranza 10−5. 
    Quanto vale N nei due casi? Quanto valgono gli errori assoluti? Motivare i risultati ottenuti.

    Il numero di suddivisioni per il calcolo dell'integrale del polinomio interpolatore di grado 3 è uguale a 2 e 
    l'errore commesso è dell'ordine di 1e-14. Questo perchè l'errore della formula di integrazione di Simpson dipende  
    dalla derivata quarta della funzione integranda e poichè nel calcolo del secondo integrale la funzione integranda è 
    un polinomio di grado 3, che ha derivata quarta nulla, non è necessario suddividere ulteriormente l'intervallo di 
    integrazione per ottenere l'approssimazione dell'integrale con la precisione richiesta, basta la formula base con 
    solo due suddivisioni!
    (Infatti siccome ha derivata quarta nulla, l'errore di quadratura per la formula di simpson è 0).
     
(slide 28)
'''

i1_exact = 2.114381916835873
i1, n1 = simpson_auto(fname, a, b, 1e-5)
i2_exact = 2.168048769926493
i2, n2 = simpson_auto(lambda points : lagrange_interpl(equi_nodes, ordinates, points), a, b, 1e-5)

err_i1 = abs(i1_exact - i1)
err_i2 = abs(i2_exact - i2)

print("INTEGRALE f(x)")
print("\t Numero N di sottointervalli equispaziati per approssimare = ", n1)
print("\t Errore Assoluto = ", err_i1)

print("INTEGRALE p(x)")
print("\t Numero N di sottointervalli equispaziati per approssimare = ", n2)
print("\t Errore Assoluto = ", err_i2)
