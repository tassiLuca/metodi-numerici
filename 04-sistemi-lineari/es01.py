#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 18:11:06 2021

@author: lucatassi
"""

'''
Esercizio 1
-----------

Al variare di n = 100, ..., 200 costruire la matrice:
            A[i, j] = sqrt(2 / (n + 1)) * sin((i + 1) * (j - 1) * pi / (n + 1))
i, j = 0, ..., n - 1 di dimensione n * n, definire la soluzione x_esatta = (1:n)^T e calcolare
il termine noto come b = A * x_esatta. Utilizzando le tre diverse function implementate calcolare
x_nopivot, x_parziale, e confrontarli con x_esatta usando i grafici in scala semilogaritmica 
dell'errore relativo al variare di n. 

Inoltre, per verificare che il numero di operazioni è proporzionale a n^3, con n dimensione del sistema, 
riportare in un grafico in scala semilogaritmica il tempo impiegato a risolvere il sistema al variare di n. 
Il grafico dovrebbe essere *asintoticamente* una retta con pendenza 3. Per verificarne la pendenza, 
disegnare contemporaneamente anche la curva n^3 e controllare che siano parallele. La strategia di pivot
non dovrebbe influenzare il risultato.
'''

import numpy as np
import numpy.linalg as npl
import math 
import Sistemi_lineari as sl

start = 100
stop = 200

errore_rel_nopivot = []
errore_rel_pivot = []
errore_rel_solve = []

for n in range(start, stop):
    A = np.empty((n, n), dtype = float)
    for i in range(n):
        for j in range(n):
            A[i, j] = math.sqrt(2 / (n + 1)) * math.sin((i + 1) * (j + 1) * math.pi / (n + 1))
            
    # Siccome se imponessi un termine noto non avrei modo di calcolare la soluzione esatta, 
    # ma solo una soluzione approssimata ottenuta mediante i metodi diretti studiati, impongo 
    # che la soluzione del sistema sia x_esatta e calcolo conseguentemente il vettore dei 
    # termini noti b come il prodotto A * x_esatta.
    x_esatta = np.arange(1, n + 1).reshape((n, 1))
    b = np.dot(A, x_esatta)
    
    # Calcolo la soluzione con il metodo di Gauss senza pivot
    P, L, U, flag = sl.LU_nopivot(A)
    x, flag = sl.LUsolve(L, U, P, b)
    errore_rel_nopivot.append(npl.norm(x - x_esatta, 1) / npl.norm(x_esatta, 1))
    
    
    