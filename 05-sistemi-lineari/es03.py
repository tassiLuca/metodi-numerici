#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04 - Esercizio 3
-----------

Costruire una function per il calcolo della soluzione di una generale equazione AX = B, con X, B matrici, 
che usi la fattorizzazione LU con pivoting parziale. Utilizzarla poi per il calcolo dell'inversa delle 
matrici non singolari:
    
                                    | 3  5  7 |        
                                A = | 2  3  4 |
                                    | 5  9 11 |
        
e 

                                    | 1   2   3   4 |
                                    | 2  -4   6   8 |
                                A = | -1 -2  -3  -1 |
                                    | 5   7   0   1 |
        
confrontando i risultati ottenuti con l'output della funzione scipy.linalg.inv(A). Ripetere l'esercizio 
utilizzando nella function per il calcolo di X la fattorizzazione LU senza pivoting. Che cosa si osserva?
"""

import Sistemi_lineari as sl
import numpy as np
import scipy.linalg as spl

def solve_nsis(A, B, flagPivot):
    """  
    Risoluzione degli n sistemi lineari A x_i = b_i con i = 0, ..., n (in cui il vettore dei termini
    noti è ottenuto prendendo l'i-esima colonna della matrice B).
        
    Parametri
    ----------
    A: matrice dei coefficienti.
    B: matrice dei termini noti.
    flagPivot: booleano che permette di selezionare il tipo di fattorizzazione di A. Se è true viene 
               fattorizzata con pivoting, altrimenti no.

    Valori di ritorno.
    -------
    X :     Matrici con le soluzioni del sistema lineare.
    flag :  Booleano che è 0 se sono soddisfatti i test di applicabilità.            
    """
    m, n = A.shape
    flag = 0
    if m != n :
        flag = 1
        print("Errore: matrice non quadrata")
        return [], flag
    
    # La matrice in input A è la sempre la stessa. 
    # La fattorizzazione di A la calcoliamo quindi una sola volta.
    X = np.zeros((m, n))
    P, L, U, flag = sl.LU_pivot(A) if flagPivot else sl.LU_nopivot(A)
    if flag == 0:
        for i in range(n):
            x, flag = sl.LUsolve(L, U, P, B[:, i])
            # x è a questo punto un array di 3 righe e 1 colonna. Il metodo squeeze permette di
            # convertirlo in un array unidimensionale (si potrebbe ottenere anche con un reshape).
            X[:, i] = x.squeeze(1)
    else :
        flag = 1
        print("Errore fattorizzazione")
        return [], flag
    
    return X, flag

choice = input("Scegli Matrice: ")

matrices = {
    '1': np.array([[3, 5, 7], [2, 3, 4], [5, 9, 11]], dtype = float),
    '2': np.array([[1, 2, 3, 4], [2, -4, 6, 8], [-1, -2, -3, -1], [5, 7, 0, 1]], dtype = float)
}

# Calcolo l'inversa di A risolvendo n sistemi lineari ciascuno dei quali ha la stessa
# matrice e termine noto uguale all'i-esima colonna della matrice identità.
A = matrices.get(choice)
m, n = A.shape
B = np.eye(m)

# Per la seconda matrice il metodo senza pivoting fallisce, a causa di un elemento pivotale nullo!
X, flag = solve_nsis(A, B, False)
print("Inversa risolvendo n sistemi lineari SENZA PIVOT: \n", X)

X, flag = solve_nsis(A, B, True)
print("Inversa risolvendo n sistemi lineari CON PIVOT: \n", X)

Xpy = spl.inv(A)
print("Inversa usando scipy.linalg \n", Xpy)   