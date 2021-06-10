#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Appello 1 Febbraio 2021 - Esercizio 1
-------------------------------------
APPROSSIMAZIONE AI MINIMI QUADRATI
"""

import numpy as np
import scipy.linalg as spl
import matplotlib.pyplot as plt

'''
a)  Scrivere la function Usolve che implementa il metodo delle sostituzioni all’indietro per risolvere 
    un sistema lineare con matrice dei coefficienti triangolare superiore.
'''
def Usolve(U, b):
    n, m = U.shape
    if n != m:
        print("ERROR: matrice non quadrata")
        return [], 1
    elif np.all(np.diag(U)) != True:
        print("ERROR: det(A) = 0")
        return [], 1
    
    x = np.zeros((n, 1))
    for i in range(n - 1, -1, -1):
        s = np.dot(U[i, i+1:], x[i+1:])
        x[i] = (b[i] - s) / U[i, i]
        
    return x, 0

'''
b)  Scrivere la function metodoQR che, presi in input due vettori contenenti rispettivamente le ascisse
    e le ordinate dei punti da approssimare ai minimi quadrati, determini i coefficienti del polinomio di 
    approssimazione di grado n risolvendo un opportuno sistema lineare tramite chiamata della function Usolve.
'''

def metodoQR(x, y, n):
    A = np.vander(x, n + 1)
    Q, R = spl.qr(A)
    y1 = np.dot(Q.T, y)
    a, flag = Usolve(R[:n+1, :], y1[:n+1])
    return a

'''
c)  Si utilizzi la function metodoQR per determinare i polinomi di approssimazione ai minimi quadrati di 
    grado 1, 2 e 3 dei dati assegnati in tabella, e si rappresentino in uno stesso grafico i dati
    (xi, yi), i = 1, ..., 12 e i tre polinomi determinati.
d)  Quale tra le tre approssimazioni ottenute al punto precedente risulta migliore? Confrontare gli errori
                                E_j = \sum_{i = 1}^{12} (f_j(x_i) - y_i)^2, j = 1, 2, 3
    dove f_1, f_2 e f_3 denotano i polinomi di approssimazione di grado 1, 2 e 3 determinati al punto c).
    
    L'errore è minimo per il polinomio di approssimazione di grado 3, il quale si discosta di pochissimo dal 
    polinomio di grado 2.
'''

start = 1900
stop = 2010
step = 12
x = np.linspace(start, stop, 12, False)
y = np.array([76, 92, 106, 123, 132, 151, 179, 203, 226, 249, 281, 305])

val_points = np.linspace(start, stop, 100)

err = []
legend = []

# grafico i punti di valutazione
plt.plot(x, y, 'ko')
legend.append("Punti di valutazione")

for n in range(1, 4):
    # calcolo il polinomio di approssimazione
    pol = metodoQR(x, y, n)
    ordinates = np.polyval(pol, val_points)
    
    # calcolo l'errore e lo stampo
    err.append(np.sum((y - np.polyval(pol, x)) ** 2))
    # equivalentemente: np.linalg.norm(y - np.polyval(pol, x))**2
    print("Errore polinomio di approssimazione di grado", n, " = ", err[-1])
        
    # grafico il polinomio
    plt.plot(val_points, ordinates)
    legend.append("Polinomio grado " + str(n))
    plt.xlabel("x")
    plt.ylabel("y")

plt.legend(legend)
plt.show()
    