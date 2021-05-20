#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esercizio 2
-------------------

Per i dati (x_i, y_i) riportati nei seguenti array:
    x = [0.0004, 0.2507, 0.5008, 2.0007, 8.0013];
    y = [0.0007, 0.0162, 0.0288, 0.0309, 0.0310];
-- costruire la retta di regressione;
-- costruire la parabola approssimante i dati nel senso dei minimi quadrati;
-- costruire la cubica approssimante i dati nel senso dei minimi quadrati;

Quale tra le tre approssimazioni risulta migliore? 
Confrontare i grafici e la norma euclidea al quadrato del vettore dei residui. 
"""

import numpy as np
import matplotlib.pyplot as plt
import minimi_quadrati as mq

x = np.array([0.0004, 0.2507, 0.5008, 2.0007, 8.0013])
y = np.array([0.0007, 0.0162, 0.0288, 0.0309, 0.0310])
x_val = np.linspace(np.min(x), np.max(x), 100)
y_val = []

for n in range(1, 4):
    a = mq.QR(x, y, n)
    '''
    Calcolo il residuo: ||y - Ba||^2 dove B è la matrice di Vandermonde, y il vettore colonna 
    contenente le ascisse dei punti dati e a è il vettore colonna incognito. 
    Dalla teoria (pg. 17) si ha che:
                   ||y - Ba||^2 = Σ(y_i - p_n(x_i))^2
    '''
    residuo = np.linalg.norm(y - np.polyval(a, x))**2
    print("n = ", n, "--> Norma al quadrato del residuo =", residuo)
    '''
    Per graficare la curva valuto il polinomio nei 100 punti equidistanti compresi tra min(x) e max(x).
    In particolare polyval ritorna, detta N la lunghezza di a:
            p[i] = a[0]*x_val[i]**(N-1) + a[1]*x_val[i]**(N-2) + ... + a[N-2]*x_val[i] + a[N-1]
    '''
    p = np.polyval(a, x_val)
    plt.plot(x_val, p)

plt.plot(x, y, 'ro')
plt.legend(['Retta di regressione', 'Parabola', 'Cubica', 'Dati'])
plt.show()
    
