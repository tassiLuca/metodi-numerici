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
    p = np.polyval(a, x_val)
    plt.plot(x_val, p)

plt.plot(x, y, 'go')
plt.legend(['Retta di regressione', 'Parabola', 'Cubica'])
plt.show()
    
