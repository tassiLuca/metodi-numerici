#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESAME 26 Giugno 2020 - Esercizio 1
SISTEMI LINEARI
"""

import numpy as np
# import scipy

'''
N.B. Non dimenticarsi il dtype = float, altrimenti i risultati delle operazioni a valle vengono trattate come interi.
     In alternativa mettere i .0: 10.0, -4.0, ...
'''

A = np.array([[10, -4, 4, 0],
              [-4, 10, 0, 2],
              [4, 0, 10, 2],
              [0, 2, 2, 0]], dtype = float)

B = np.array([[5, -2, 2, 0], 
              [-2, 5, 0, 1], 
              [2, 0, 5, 1], 
              [0, 1, 1, 5]], dtype = float)

'''
(a) Stabilire se le matrici A e B ammettono la fattorizzazione di Cholesky, motivandone la risposta.
    
    Sia A una matrice quadrata n x n a coefficienti reali. Se A è simmetrica e definita positiva allora esiste
    un'unica matrice singolare inferiore L con elementi positivi sulla diagonale tale che A = L L.T.
    
    Una matrice è simmetrica se A = A.T. Inoltre sappiamo che ogni matrice simmetrica definita positiva ha tutti gli
    autovalori strettamente positivi. 
    
(slide 54)
'''

def is_Cholesky_applicable(A):
    return np.all(A) == np.all(A.T) and np.all(np.linalg.eigvals(A) > 0)

if is_Cholesky_applicable(A):
    print("Matrice A ammette fattorizzazione Cholesky")
else :
    print("Matrice A NON ammette fattorizzazione Cholesky")
    
if is_Cholesky_applicable(B):
    print("Matrice B ammette fattorizzazione Cholesky")
else :
    print("Matrice B NON ammette fattorizzazione Cholesky")

'''
(b) Stabilire se le matrici A e B ammettono la fattorizzazione LU senza pivoting, motivandone la risposta.

    TEOREMA
    Sia A una matrice quadrata n x n.
    La fattorizzazione A = LU esiste ed è unica se le k sottomatrici principali di testa di A sono non singolari, 
    con k = 1 : n - 1.
    
(slide 20)
'''

def is_LU_applicable(A):
    for i in range(A.size):
        if np.linalg.det(A[:i, :i]) == 0:
            return False
    
    return True

if is_LU_applicable(A):
    print("Matrice A ammette fattorizzazione LU")
else :
    print("Matrice A NON ammette fattorizzazione LU")
    
if is_LU_applicable(B):
    print("Matrice B ammette fattorizzazione LU")
else :
    print("Matrice B NON ammette fattorizzazione LU")

'''
(c) Scrivere una function che, presa in input una matrice M che ammette fattorizzazione LU senza
    pivoting, calcoli e restituisca in output le matrici di tale fattorizzazione.
'''

def LU_decomposition(A):
    n, m = A.shape
    flag = 0
    if n != m:
        flag = 1
        print("ERRORE: matrice non quadrata")
        return [], [], flag
    
    U = A.copy()
    for k in range(n-1):
        if U[k, k] == 0:
            flag = 1
            print("ERRORE: elemento pivotale nullo")
            return [], [], flag
        
        U[k+1:n, k] = U[k+1:n, k] / U[k, k]
        U[k+1:n, k+1:n] = U[k+1:n, k+1:n] - np.outer(U[k+1:n, k], U[k, k+1:n])
        
    L = np.tril(U, -1) + np.eye(n)
    U = np.triu(U)
    return L, U, flag

L_A, U_A, flag_A = LU_decomposition(A)
L_B, U_B, flag_B = LU_decomposition(B)

# =============================================================================
# # funzione scipy per la fattorizzazione LU
# P, L, U = scipy.linalg.lu(A)
# =============================================================================

'''
(d) Scrivere uno script che sfrutti l’output dell’algoritmo di fattorizzazione LU senza pivoting per calcolare
    nella maniera più efficiente possibile sia il determinante di M che il determinante di M−1 .
    Eseguire lo script scegliendo come matrice M le matrici per cui al punto b) si è mostrata l’esistenza
    della fattorizzazione LU senza pivoting.


    Per il teorema di Binèt, il determinante del prodotto di due matrici quadrate, è uguale al prodotto dei 
    determinanti di ciascuna delle due matrici: 
                    
                                                det(A * B) = det(A) * det(B) 
                            
    dove A, B sono due matrici n x n.
                            
    Nel nostro caso abbiamo che, detta M una matrice quadrata di cui sia possibile effettuare la fattorizzazione LU, 
    i.e. M = L U, il det(M) = det(L) * det(U). Entrambe sono matrici triangolari, perciò il determinante non è altro 
    che il prodotto degli elementi sulla diagonale. Nello specifico, essendo L una matrice triangolare inferiore con 
    el.ti diagonali pari a 1, il det(L) = 1 => det(M) = det(U).
    
    Ricordiamo inoltre il determinante dell'inversa di A è il reciproco del determinante di A.
'''

detA = np.prod(np.diag(U_A))
detA_inv = 1 / detA

# Verifo i risultati con np.linalg
detA_linalg = np.linalg.det(A)
print("DETERMINANTE DI A:")
print("\t Risultato np.linalg = ", detA_linalg)
print("\t Mio risultato = ", detA)

detB = np.prod(np.diag(U_B))
detB_inv = 1 / detB

# Verifico i risultati con np.linalg
detB_linalg = np.linalg.det(B)
print("DETERMINANTE DI B:")
print("\t Risultato np.linalg = ", detB_linalg)
print("\t Mio risultato = ", detB)

