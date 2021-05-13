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

NOTE: 
"""

import numpy as np
import Sistemi_lineari as sl
import matplotlib.pyplot as plt

x_esatta = np.array([[2], [2]])
err_rel_nopivot = []
err_rel_pivot = []

start = 2
stop = 19
step = 2

def _stime_errori(P, L, U, flag) :
    x, flag = sl.LUsolve(L, U, P, b)
    err = np.linalg.norm(x - x_esatta) / np.linalg.norm(x_esatta)
    return err

for k in range(start, stop, step):
    eps = 10 ** (-k)
    A = np.array([[eps, 1], [1, 1]])
    b = np.array([[2 + eps], [4]])
    
    # Calcolo senza strategia pivotale
    P, L, U, flag = sl.LU_nopivot(A)
    if flag == 0:
        err_rel_nopivot.append(_stime_errori(P, L, U, flag))
    else :
        print("Sistema non risolubile senza strategia pivotale!")
    
    # Calcolo con strategia pivotale
    P, L, U, flag = sl.LU_pivot(A)
    if flag == 0:
        err_rel_pivot.append(_stime_errori(P, L, U, flag))
    else :
        print("Sistema non risolubile con strategia pivotale!")

plt.semilogy(range(start, stop, step), err_rel_nopivot, 
             range(start, stop, step), err_rel_pivot)
plt.legend(['Errore relativo senza strategia pivotale', 'Errore relativo con strategia pivotale'])
plt.show()