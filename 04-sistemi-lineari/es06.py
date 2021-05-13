#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04 - Esercizio 6
-----------
Dato n = 100, calcolare Q = I - 2 v v^T, dove I è la matrice identità di ordine n e v è un vettore colonna di n 
componenti formato da numeri casuali ed avente norma 2 unitaria. Quindi, per k = 1, ..., 20, porre:
                D = I
                D[n - 1, n - 1] = 10^k
                A = Q D
Tale costruzione produce una matrice A che abbia esattamente numero di condizionamento 10^k in norma 2.
Al variare di k e quindi del numero di condizionamento di A, studiare l'errore del metodo di eliminazione gaussiana
senza pivoting e con pivoting parziale nella risoluzione del sistema lineare Ax = b, dove b = A x_esatta e 
x_esatta = (1, 1, ..., 1)^T.
"""

import numpy as np
import Sistemi_lineari as sl
import matplotlib.pyplot as plt

start = 1
stop = 21

n = 100
v = np.random.rand(n, 1)
v = v / np.linalg.norm(v, 2)
Q = np.eye(n) - 2 * np.outer(v, v.T)    # v.T è il vettore trasposto di v
D = np.eye(n)
x_esatta = np.ones((n, 1))

indice_condizionamento = []
err_rel_nopivot = []
err_rel_pivot = []

for k in range(start, stop):
    D[n - 1, n - 1] = 10.0 ** k
    A = np.dot(Q, D)
    indice_condizionamento.append(np.linalg.cond(A, 2)) # calcolo il numero di condizionamento di A
    b = np.dot(A, x_esatta)
    
    # Senza pivoting 
    P, L, U, flag = sl.LU_nopivot(A)
    if flag == 0:
        x_nopivot, flag = sl.LUsolve(L, U, P, b)
    else :
        print("Sistema non risolubile senza strategia pivotale.")
    
    err_nopivot = np.linalg.norm(x_nopivot - x_esatta, 2) / np.linalg.norm(x_esatta, 2)
    err_rel_nopivot.append(err_nopivot)
    
    # Con pivoting 
    P, L, U, flag = sl.LU_pivot(A)
    if flag == 0:
        x_pivot, flag = sl.LUsolve(L, U, P, b)
    else :
        print("Sistema non risolubile con strategia pivotale parziale.")
    
    err_pivot = np.linalg.norm(x_pivot - x_esatta, 2) / np.linalg.norm(x_esatta, 2)
    err_rel_pivot.append(err_pivot)
        
plt.loglog(indice_condizionamento, err_rel_nopivot, 'ro-', 
           indice_condizionamento, err_rel_pivot, 'bo-')
plt.legend(['Not pivot', 'Pivot'])
plt.xlabel('Indice di condizionamento')
plt.ylabel('Errore relativo sulla soluzione')
plt.show()