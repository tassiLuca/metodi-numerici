#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Appello del 25 Giugno 2020 - Esercizio 2
----------------------------------------
INTERPOLAZIONE POLINOMIALE ED INTEGRAZIONE.
"""

import numpy as np
import matplotlib.pyplot as plt

f = lambda x : x - np.sqrt(x - 1)
a = 1
b = 3
n = 3

'''
a)  Scrivere il proprio codice Matlab per determinare il polinomio p(x) di grado 3 che interpola la funzione f(x) su 
    nodi equispaziati.
b)  Disegnare in una stessa figura i punti di interpolazione, il grafico di f e del polinomio
    di interpolazione p ottenuto al punto a).
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

points = np.linspace(a, b, 100)
nodes  = np.linspace(a, b, n + 1)
ordinates = f(nodes)
pol = lagrange_intrpl(nodes, ordinates, points)

plt.plot(points, f(points), 'orange', nodes, ordinates, 'r*', points, pol, 'green')
plt.legend(["f(x)", "Punti di interpolazione", "Polinomio interpolante p(x)"])
plt.grid(True)
plt.show()

'''
c)  Scrivere il proprio codice per calcolare con la formula di Simpson composita
    su N sottointervalli equispaziati, i valori approssimati degli integrali di f(x) e p(x).
    
d)  Si stimi il numero N di sottointervalli equispaziati che servono per approssimare con
    la formula di Simpson composita i due integrali (il cui valore esatto è rispettivamente
    I1 = 2.114381916835873 e I2 = 2.168048769926493) nel rispetto della tolleranza
    10−5. Quanto vale N nei due casi? Quanto valgono gli errori assoluti? Motivare i risultati ottenuti.

    Il fatto che la formula di Simpson per calcolare l'integrale del polinomio approssimante di grado 2
    raggiunga il risultato in soli 2 passi commettendo un errore (assoluto) pari a circa 1.3e-14, dipende dal fatto che 
    l'errore della formula di NC composita con n = 2 (=> s = 4) dipende dalla derivata quarta della funzione integranda. 
    Nella fattispecie, essendo la funzione integranda il polinomio di grado 4, la sua derivata quarta si annulla, 
    di fatto rendendo zero l'errore sin dalla prima suddivisione in 2 passi.
'''

def simpson_composita(fname, a, b, n):
    h = (b - a) / (2 * n)
    nodes = np.arange(a, b + h, h)
    f = fname(nodes)
    return h / 3 * (f[0] + 2 * np.sum(f[2:2*n:2]) + 4 * np.sum(f[1:2*n:2]) + f[2*n])

def simpson_toll(fname, a, b, toll):
    steps = 1
    max_steps = 2048
    err = 1
    
    integral = simpson_composita(fname, a, b, steps)
    while err > toll and steps <= max_steps:
        steps *= 2
        integral_double_steps = simpson_composita(fname, a, b, steps)
        err = abs(integral - integral_double_steps) / 15
        integral = integral_double_steps
    
    return integral, steps

toll = 1.e-5

print("Integrale f(x)")
i1_esatto = 2.114381916835873
print("\t Integrale esatto = ", i1_esatto)
i1_simpson, i1_steps = simpson_toll(f, a, b, toll)
print("\t Integrale f(x) = ", i1_simpson, "raggiunto con ", i1_steps, "passi")
err_i1 = abs(i1_esatto - i1_simpson)
print("\t Errore assoluto = ", err_i1)

print("Integrale p(x)")
i2_esatto = 2.168048769926493
print("\t Integrale esatto = ", i2_esatto)
i2_simpson, i2_steps = simpson_toll(lambda x : lagrange_intrpl(nodes, ordinates, x), a, b, toll)
err_i2 = abs(i2_esatto - i2_simpson)
print("\t Integrale p(x) = ", i2_simpson, "raggiunto con ", i2_steps, "passi")
print("\t Errore assoluto = ", err_i2)

plt.plot(points, f(points), 'orange')
plt.legend(["f(x)"])
plt.fill_between(points, f(points), color = 'orange', alpha = 0.2)
plt.text(1, 0.4, " ∫ f(x) dx = "+ str(i1_simpson) + " raggiunto con " + str(i1_steps) + " passi")
plt.text(1, 0.2, "Errore = " + str(err_i1))
plt.grid(True)
plt.show()

plt.plot(points, pol, "green")
plt.legend(["Polinomio interpolante"])
plt.fill_between(points, pol, color = "green", alpha = 0.2)
plt.text(1, 0.4, " ∫ p(x) dx = "+ str(i2_simpson) + " raggiunto con " + str(i2_steps) + " passi")
plt.text(1, 0.2, "Errore = " + str(err_i2))
plt.grid(True)
plt.show()