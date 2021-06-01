#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esercizio 1
-----------
Si disegnino i grafici dei polinomi di Lagrange associati ai nodi {0, 1/4, 1/2, 3/4, 1}
e ai nodi {−1, −0.7, 0.5, 2}.

NOTE: Questo esercizio mostra che quando si valutano i polinomi fondamentali di Lagrange in 
      un'ascissa che ha lostesso indice dei polinomio fondamentali si ottiene 1, altrimenti 0.
      Cioè:
          --> L_j(x_i) = 0 per i != j
          --> L_j(x_j) = 1
      (vedi slide 9)
"""

import numpy as np
import matplotlib.pyplot as plt
import interpolazione as intrpl

nodes = [np.arange(0, 1.1, 1/4), np.array([-1, -0.7, 0.5, 2])]

for node in nodes:
    n = node.size
    values = np.linspace(node[0], node[n-1], 200)
    for j in range(n):
        p, flag = intrpl.plagrange(node, j)
        if flag == 1:
            break
        L = np.polyval(p, values)
        plt.plot(node, np.zeros((n, )), 'o')
        plt.plot(node[j], 1, '*')
        plt.plot(values, L)
    plt.show()

