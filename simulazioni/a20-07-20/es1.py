#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 21:10:36 2021

@author: lucatassi
"""

import numpy as np
import scipy.linalg as spl
import math
import matplotlib.pyplot as plt

pi = math.pi

f = lambda x : np.sin(x) + np.sin(5 * x)
a = 0
b = 2 * pi
n = 2

x1 = np.arange(a, b, pi / 3, dtype = float)
y1 = f(x1)

x2 = np.array([pi / 6, 2 * pi / 5, 4 * pi / 5, 8 * pi / 5, 11 * pi / 6], dtype = float)
y2 = f(x2)

'''
a)  Per ciascuna sequenza di punti si costruisca il polinomio di grado 2 di approssimazione nel senso dei
    minimi quadrati.
b)  Si rappresentino in una stessa figura la funzione f , le due sequenze di punti in a.1) e a.2), e i corri-
    spondenti polinomi di approssimazione nel senso dei minimi quadrati (rispettivamente p1(x) e p2 (x)).
    Quale dei due approssima meglio f ?

    pol1 risulta essere il polinomio identicamente nullo. Quindi pol2 approssima meglio f(x).
'''

def backward(A, b):
    n, m = A.shape 
    if n != m:
        print("ERRORE: matrici non quadrata")
        return [], 1
    elif np.all(np.diag(A)) != True:
        print("ERRORE: det(A) = 0")
        return [], 1
    
    x = np.zeros((n, 1))
    for i in range(n - 1, -1, -1):
        s = np.dot(A[i, i+1:], x[i+1:])
        x[i] = (b[i] - s) / A[i, i]
    
    return x, 0

def metodoQR(x, y, n):
    A = np.vander(x, n + 1)
    Q, R = spl.qr(A)
    y1 = np.dot(Q.T, y)
    a, flag = backward(R[:n+1, :], y1[:n+1])
    return a

points = np.linspace(a, b, 100)

a1 = metodoQR(x1, y1, n)
pol1 = np.polyval(a1, points)
a2 = metodoQR(x2, y2, n)
pol2 = np.polyval(a2, points)

plt.plot(points, f(points), x1, y1, '*', x2, y2, 'o', points, pol1, points, pol2)
plt.legend(["f(x)", "(x1, y1)", "(x2, y2)", "p1(x)", "p2(x)"])
plt.title("Grafico polinomi quadratici")
plt.grid(True)
plt.show()

'''
c)  Scrivere il proprio codice per calcolare, con la formula dei Trapezi Composita su N sottointervalli 
    equispaziati, i valori approssimati degli integrali dei due polinomi quadratici determinati precedentemente.
'''

def trapezi_composita(fname, a, b, n):
    h = (b - a) / n
    nodes = np.arange(a, b + h, h)
    f = fname(nodes)
    return h / 2 * (f[0] + 2 * np.sum(f[1:n]) + f[n])

'''
d)  Utilizzando la tecnica del raddoppio degli intervalli, scrivere la function traptoll per stimare il numero
    N di sottointervalli equispaziati che servono per approssimare con la formula dei Trapezi Composita
    gli integrali I1 e I2 nel rispetto della tolleranza 10−4. Quanto vale N nei due casi? Quanto valgono gli 
    intergrali? Quale dei due integrali approssimati risulta essere una miglior approssimazione dell’integrale
    esatto di f in [0, 2π]? Motivare la risposta.
    
    Poichè l'integrale di f(x) in [0, 2π] è 0, è più vicina all'integrale esatto di f(x) l'approssimazione 
    data da pol1.
'''

def trap_toll(fname, a, b, toll):
    max_steps = 2048
    steps = 1
    err = 1
    
    integral = trapezi_composita(fname, a, b, steps)
    while err > toll and steps <= max_steps:
        steps *= 2
        integral_double_steps = trapezi_composita(fname, a, b, steps)
        err = abs(integral - integral_double_steps) / 3
        integral = integral_double_steps
    
    return integral, steps

toll = 1.e-4

int1, steps1 = trap_toll(lambda x : np.polyval(a1, x), a, b, toll)
print("Integrale di p1(x) = ", int1, " --> steps =", steps1)

int2, steps2 = trap_toll(lambda x : np.polyval(a2, x), a, b, toll)
print("Integrale di p2(x) = ", int2, " --> steps =", steps2)

plt.plot(points, pol1, points, pol2)
plt.fill_between(points, pol1, alpha = 0.3)
plt.fill_between(points, pol2, alpha = 0.3)
plt.plot(points, f(points))
plt.fill_between(points, f(points), alpha = 0.3)
plt.legend(["p1(x)", "p2(x)", "f(x)", "∫ p1(x) dx = ", "∫ p2(x) dx", "∫ f(x) dx"])
plt.title("Grafico integrali")
plt.grid(True)
plt.show()
