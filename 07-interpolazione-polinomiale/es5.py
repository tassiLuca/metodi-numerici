#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esercizio 5
-----------
Per n = 5, 10, 15, 20 fornire un’approssimazione della costante di Lebesgue scegliendo x_1, x_2 , ..., x_n+1 
equispaziati in [−1, 1] oppure coincidenti con i nodi di Chebyshev x_i = cos(((2i + 1)π) / (2(n + 1))) i = 0, ..., n
"""

import numpy as np
import matplotlib.pyplot as plt
import interpolazione as intrpl

start = 5
stop = 25
step = 5
a = -1      # estremo sinistro dell'intervallo di interpolazione
b = 1       # estremo destro dell'intervallo di interpolazione
points_values = np.linspace(a, b, 200)

# Vettore contenente le costanti di Lebesgue per n = 5, 10, 15, 20 nel caso di nodi equispaziati e di Chebyshev
equi_lebegue = np.zeros((4, 1))
cheby_lebegue = np.zeros((4, 1))

for n in range(start, stop, step):
    equi_nodes = np.linspace(a, b, n + 1)
    cheby_nodes = intrpl.chebyshev_nodes(a, b, n)
    
    equi_lebesgue_acc = np.zeros((200, 1))
    cheby_lebesgue_acc = np.zeros((200, 1))
    for i in range(n + 1):
        equi_pol = intrpl.plagrange(equi_nodes, i)
        print(equi_pol)
        print(np.polyval(equi_pol, points_values))
        # Accumulo i valori assoluti di tutti gli n + 1 polinomi di Lagrange sui nodi equispaziati
        equi_lebesgue_acc = equi_lebesgue_acc + np.abs(np.polyval(equi_pol, points_values))
        
        cheby_pol = intrpl.plagrange(cheby_nodes, i)
        # Accumulo i valori assoluti di tutti gli n + 1 polinomi di Lagrange sui nodi di Chebyshev
        cheby_lebesgue_acc = cheby_lebesgue_acc + np.abs(np.polyval(cheby_pol, points_values))
        
    # equi_lebegue[i] = np.max(equi_lebesgue_acc)
    # cheby_lebegue[i] = np.max(cheby_lebesgue_acc)
  
    