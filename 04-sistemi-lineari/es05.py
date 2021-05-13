#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04 - Esercizio 5
-----------
Sia assegnato il sistema lineare Ax = b con 

            | 3  1  1  1 |        | 4 |
        A = | 2  1  0  0 |,   b = | 1 |
            | 2  0  1  0 |        | 2 |
            | 2  0  0  1 |        | 4 |
            
la cui soluzione esatta è x = [1, -1, 0, 2]^T. Calcolare la fattorizzazione LU di A e osservare la presenza di fill-in
nei fattori L e U. Osservare come l'utilizzo di pivot parziale determini moltiplicatori m_ik tali che |m_ik| <= 1 ed
una minor crescita del modulo degli elementi della matrice triangolare superiore U. Analizzare inoltre la 
fattorizzazione LU risultante nel caso del sistema lineare ottenuto permutando nella matrice A la prima riga con
l'ultima, la prima colonna con l'ultima e nel termine noto la prima componente con l'ultima:
    
            | 1  0  0  2 |        | 4 |
        A = | 0  1  0  2 |,   b = | 1 |
            | 0  0  1  2 |        | 2 |
            | 1  1  1  3 |        | 4 |
            
La soluzione esatta di quest'ultimo è x = [2, -1, 0, 1]^T come atteso considerando la permutazione effettuata sulle
colonne della matrice A.
"""

import numpy as np
import Sistemi_lineari as sl

choice = input("Scegli Matrice: ")

matrices = {
    # Lista con [matrice dei coefficienti A, vettore termini noti b, soluzione esatta x]
    '1': [np.array([[3, 1, 1, 1], [2, 1, 0, 0], [2, 0, 1, 0], [2, 0, 0, 1]], dtype = float), 
          np.array([4, 1, 2, 4], dtype = float), 
          np.array([1, -1, 0, 2], dtype = float)],
    
    '2': [np.array([[1, 0, 0, 2], [0, 1, 0, 2], [0, 0, 1, 2], [1, 1, 1, 3]], dtype = float), 
          np.array([4, 1, 2, 4], dtype = float), 
          np.array([2, -1, 0, 2], dtype = float)]
    }

A, b, x_esatta = matrices.get(choice)
m, n = A.shape

# SENZA STRATEGIA PIVOTALE
P, L, U, flag = sl.LU_nopivot(A)
if flag == 0:
    x_nopivot, flag = sl.LUsolve(L, U, P, b)
    print("Soluzione senza strategia pivotale \n", x_nopivot)
else :
    print("Soluzione NON risolubile senza strategia pivotale")
    
max_L_nopivot = np.max(np.abs(L))
max_U_nopivot = np.max(np.abs(U))

# CON STRATEGIA PIVOTALE
P, L, U, flag = sl.LU_pivot(A)
if flag == 0:
    x_pivot, flag = sl.LUsolve(L, U, P, b)
    print("Soluzione con strategia pivotale \n", x_nopivot)
else :
    print("Soluzione NON risolubile con strategia pivotale")

max_L_pivot = np.max(np.abs(L))
max_U_pivot = np.max(np.abs(U))

print("Fattorizzazione no pivot \n ---------- \n Massimo matrice L = ", max_L_nopivot,
      "\n Massimo matrice U = ", max_U_nopivot)
print("Fattorizzazione con pivot \n --------- \n Massimo matrice L = ", max_L_pivot,
      "\n Massimo matrice U = ", max_U_pivot)
