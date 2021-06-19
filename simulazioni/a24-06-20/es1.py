#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Appello 24 Giugno 2020 - Esercizio 1
------------------------------------
SISTEMI LINEARI.
Traccia molto simile a quella del 26 Giugno 2020 (esercizio 1).
Si faccia riferimento a quella per maggiori spiegazioni.
"""

import numpy as np
import numpy.linalg as npl

A = np.array([[5, 0, 0, 0, 0], 
              [0, 1, 0, 0, 0],
              [0, 0, 1, 2, 0], 
              [0, 0, 2, 0, 2], 
              [0, 0, 0, 2, 1]], dtype = float)

'''
a) Spiegare se A ammette fattorizzazione LU senza pivoting o fattorizzazione di Cholesky.
'''

def is_LU_applicable(A):
    n, m = A.shape
    for k in range(1, n):
        if npl.det(A[:k, :k]) == 0:
            return False, k
    return True, 0

def is_Cholesky_applicable(A):
    return np.all(A) == np.all(A.T) and np.all(npl.eigvals(A) > 0)

print("Fattorizzazione LU? --> ", is_LU_applicable(A))
print("Fattorizzazione di Cholesky? --> ", is_Cholesky_applicable(A))

'''
b)  Scrivere il proprio codice matlab per calcolare le fattorizzazioni di A che nel punto a) risultano possibili.
'''

def LU(A):
    n, m = A.shape
    if n != m:
        print("ERRORE: matrice non quadrata")
        return [], [], 1
    
    U = A.copy()
    for k in range(n - 1):
        if U[k, k] == 0:
            print("ERRORE: elemento pivotale nullo")
            return [], [], 1
        
        U[k+1:n, k] = U[k+1:n, k] / U[k, k]
        U[k+1:n, k+1:n] = U[k+1:n, k+1:n] - np.outer(U[k+1:n, k], U[k, k+1:n])
    
    L = np.tril(U, -1) + np.eye(n)
    U = np.triu(U)
    return L, U, 0

'''
c)  Scrivere il proprio codice matlab per calcolare la fattorizzazione LU di A con pivoting parziale.
'''

def swap(A, k, r):
    A[[k, r], :] = A[[r, k], :]

def LU_pivot(A):
    n, m = A.shape
    if n != m:
        print("ERRORE: matrice non quadrata")
        return [], [], 1
    
    P = np.eye(n)
    U = A.copy()
    for k in range(n - 1):
        r = np.argmax(U[k:n, k]) + k
        if r != k:
            swap(A, r, k)
            swap(P, r, k)
        
        U[k+1:n, k] = U[k+1:n, k] / U[k, k]
        U[k+1:n, k+1:n] = U[k+1:n, k+1:n] - np.outer(U[k+1:n, k], U[k, k+1:n])
    
    L = np.tril(U, -1) + np.eye(n)
    U = np.triu(U)
    return P, L, U, 0


print("SENZA PIVOTING")
L, U, flag = LU(A)
print("L (triangolare inferiore con moltiplicatori): \n", L)
print("U (triangolare superiore): \n", U)

print("CON PIVOTING")
P_pivot, L_pivot, U_pivot, flag_pivot = LU_pivot(A)
print("P: \n", P_pivot)
print("L (triangolare inferiore con moltiplicatori): \n", L_pivot)
print("U (triangolare superiore): \n", U_pivot)

'''
d)  Sfruttare la più conveniente delle fattorizzazioni precedentemente implementate ai punti b) e c) per
    calcolare det(A) in maniera efficiente (ovvero con minor costo computazionale).
'''

detA = np.prod(np.diag(U))
print("Determinante di A = ", detA)
print("Verifica del detA (con numpy.linalg.det(A)) = ", npl.det(A))
