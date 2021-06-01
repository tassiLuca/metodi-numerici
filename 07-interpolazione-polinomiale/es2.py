#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esercizio 2
-----------
Realizzare uno script che calcoli nella forma di Lagrange i polinomi che interpolano le funzioni 
test sin(x) e cos(x) nei punti x_k = kπ/2, con k = 0, 1, 2, 3, 4. 
Visualizzare graficamente i polinomi ottenuti insieme alle funzioni assegnate.
"""

import numpy as np
import math
import interpolazione as intrpl
import matplotlib.pyplot as plt

# Come al solito np.arange() esclude lo stop, mentre noi vogliamo che 2*pi sia incluso.
nodes = np.arange(0, 2 * math.pi + 0.1, math.pi / 2)
# I punti in cui vado a valutare il polinomio sono equispaziati.
points_values = np.arange(0, 2 * math.pi + 0.1, math.pi / 40)

sin_intrpl = intrpl.lagrange_interp(nodes, np.sin(nodes), points_values)
plt.plot(nodes, np.sin(nodes), '*', points_values, sin_intrpl, '--', points_values, np.sin(points_values), '-')
plt.legend(['Punti di interpolazionee', 'Interpolante di lagrange', 'y = sin(x)'])
plt.show()

cos_intrpl = intrpl.lagrange_interp(nodes, np.cos(nodes), points_values)
plt.plot(nodes, np.cos(nodes), '*', points_values, cos_intrpl, '--', points_values, np.cos(points_values), '-')
plt.legend(['Punti di interpolazionee', 'Interpolante di lagrange', 'y = cos(x)'])
plt.show()