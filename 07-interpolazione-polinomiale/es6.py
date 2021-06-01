#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esercizio 6
-----------
Si interpolino mediante il polinomio p_{21}(x) i 22 punti (xi, yi) con x_i equispaziati in [−1, 1] 
e y_i = sin(2π*x_i). Si considerino poi le ordinate ỹi = yi + εi , dove εi denota l’i-esima componente 
del vettore 0.0002∗numpy.random.randn(22, ), e si calcoli il corrispondente polinomio interpolante 
p̃_{21}(x). Si visualizzino e si commentino i risultati ottenuti, calcolando anche l’errore relativo sul
polinomio interpolante e sui dati.
"""

import numpy as np
import math
import interpolazione as intrpl
import matplotlib.pyplot as plt
import numpy.linalg as npl

n = 22
nodes = np.linspace(-1, 1, n)
points = np.linspace(-1, 1, 300)
function = lambda x : np.sin(2 * math.pi * x)

# dati esatti
y1 = function(nodes)
pol1 = intrpl.lagrange_interp(nodes, y1, points)

# dati perturbati
y2 = y1.copy()
y2 += 0.0002 * np.random.randn(22, )
pol2 = intrpl.lagrange_interp(nodes, y2, points)

# grafico i risultati esatti
plt.plot(points, pol1, nodes, y1, '*', points, function(points))
plt.legend(["Interpolante di Lagrange", "Punti di interpolazione", "Funzione"])
plt.show()

# grafico i risultati perturbati
plt.plot(points, pol2, nodes, y2, '*', points, function(points))
plt.legend(["Interpolante di Lagrange", "Punti di interpolazione perturbati", "Funzione"])
plt.show()

# errore relativo 
err_dati = npl.norm(y2 - y1, np.inf) / npl.norm(y1, np.inf)
err_risultati = npl.norm(pol2 - pol1, np.inf) / npl.norm(pol1, np.inf)
print("Errore relativo sui dati: ", err_dati)
print("Errore relativo sui risultati: ", err_risultati)
