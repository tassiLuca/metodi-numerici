#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esercizio 3
-------------------
Scrivere uno script Python per approssimare la seguente configurazione di punti
    x = numpy.arange(10, 10.6, 0.1);
    y = numpy.array([11.0320, 11.1263, 11.1339, 11.1339, 11.1993, 11.1844]);
mediante un polinomio ai minimi quadrati di grado 4 costruito con il metodo QRLS. 
Perturbare poi il secondo punto nel seguente modo
    x(2) = x(2) + 0.013;
    y(2) = y(2) − 0.001;
e calcolare il polinomio ai minimi quadrati relativo alla configurazione perturbata. 
Commentare e motivare i risultati ottenuti.

NOTE: Nonostante il secondo punto sia stato perturbato, il polinomio di approssimazione non viene inficiato.
      Questo dipende dal fatto che se la matrice di Vandermonde B è ben condizionata anche R1 (cioè la parte
      triangolare superiore di R) lo è, e la soluzione del sistema lineare non produce soluzioni inaccurate.
      (Vd. slide 15 teoria.)
"""

import numpy as np
import minimi_quadrati as mq
import matplotlib.pyplot as plt

def _pol_regression(x, y, n, info = None):
    '''
    Calcola il polinomio di regressione di grado n, lo grafica e stampa 
    su console la norma al quadrato del residuo.
    
    Parametri
    ----------
    x: vettore colonna con le ascisse dei punti
    y: vettore colonna con le ordinate dei punti
    n: grado del polinomio approssimante
    '''
    a = mq.QR(x, y, n)
    residuo = np.linalg.norm(y - np.polyval(a, x))**2
    print("[ Risultati: ", info, " ]") if info != None else print("[ Risultati ]")
    print("n = ", n, "--> Norma al quadrato del residuo =", residuo)
    x_val = np.linspace(np.min(x), np.max(x), 100)
    p_x = np.polyval(a, x_val)
    plt.plot(x_val, p_x, x, y, 'o')

n = 4
x = np.arange(10, 10.6, 0.1)
y = np.array([11.0320, 11.1263, 11.1339, 11.1339, 11.1993, 11.1844])

_pol_regression(x, y, n, "DATI REALI")

# Perturbo i dati
x_delta = x.copy()
y_delta = y.copy()
x_delta[1] = x_delta[1] + 0.013
y_delta[1] = y_delta[1] - 0.001

_pol_regression(x_delta, y_delta, n, "DATI PERTURBATI")

plt.show()

