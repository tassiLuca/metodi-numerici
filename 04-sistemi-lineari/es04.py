#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04 - Esercizio 4
-----------
Sia assegnato il sistema lineare Ax = b con 

            | ε  1 |        | 2 + ε |
        A = | 1  1 |,   b = |   4   |

dove ε > 0. Dopo aver osservato che quando ε = 0 tale sistema ha soluzione [2, 2]^T, risolvere il sistema lineare
senza far uso della strategia pivotale per valori di ε pari a 10^-k con k = 2 : 2 : 18. Confrontare i risultati
ottenuti con le soluzioni trovate per i medesimi valori di ε quando viene applicata la strategia pivotale.
"""

import numpy as np
import Sistemi_lineari as sl

x_esatta = np.array([[2], [2]])

for k in range(2, 19, 2):
    eps = 10 ** - k
    A = np.array([[eps, 1], [1, 1]])
    b = np.array([[2 + eps], [4]])
    
    P, L, U, flag = sl.LU_nopivot(A)
    if flag == 0:
        x, flag = sl.LUsolve(L, U, P, b)
        print("Exit status: ", flag, "\nSoluzione x = \n", x)
    
