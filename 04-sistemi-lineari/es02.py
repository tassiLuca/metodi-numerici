#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04 - Esercizio 2
-----------

Calcolre la soluzione dei seguenti sistemi lineari Ax = b usando il metoo di fattorizzazione LU senza
pivoting e con pivoting parziale:
    
                                    | 1  2  3 |         | 6 |
                                A = | 0  0  1 |,    b = | 1 |
                                    | 1  3  5 |         | 9 |
        
e 

                                    | 1  1  0  3 |         | 5 |
                                    | 2  1 -1  1 |         | 3 |
                                A = | -1 2  3 -1 |,    b = | 3 |
                                    | 3 -1 -1  2 |         | 3 |
        
In entrambi i casi la soluzione esatta ha componenti x_i = 1 per ogni i = 1, ..., n. 
Come è possibile giustificare a priori l'insuccesso del metodo di fattorizzazione LU in assenza di pivot?

NOTA:   Per entrambe le matrici, la fattorizzazione LU senza pivoting si arresta poiché ci sono elementi
        pivotali nulli. Sappiamo infatti che affinché la fattorizzazione LU esista e sia unica tutte le 
        sottomatrici principali di testa di ordine 1, ..., n - 1 siano NON singolari (cioè det != 0 o, 
        equivalentemente, il suo rango è massimo). Questa condizione garantisce il fatto che gli elementi 
        pivotali siano non nulli.
        --> per il primo sistema: il minore principale di ordine 2 è nullo;
        --> per il secondo sistema, considerando la sottomatrice di testa di ordine 3,
                                            | 1  1  0 |
                                            | 2  1 -1 |
                                            | -1 2  3 |
            si nota che la terza colonna è lin. dipendente dalle prime due (e si ottiene sottraendo alla 
            seconda la prima colonna). Ergo la sottomatrice principale di testa di ordine 3 non ha rango 
            massimo, cioè non è singolare. Per questo anche per il secondo sistema la fattorizzazione LU
            si arresta.
"""

import numpy as np
import Sistemi_lineari as sl

choice = input("Scegli Matrice: ")

# NOTA: È bene specificare il tipo float, in quanto i numpy array di default dà il tipo del valore numerico
# inserito (nel nostro caso sarebbero int32), creando gravi problemi di approssimazione nella soluzione.
matrices = {
    '1': [np.array([[1, 2, 3], [0, 0, 1], [1, 3, 5]], dtype = float), 
          np.array([[6], [1], [9]], dtype = float)],
    '2': [np.array([[1, 1, 0, 3], [2, 1, -1, 1], [-1, 2, 3, -1], [3, -1, -1, 2]], dtype = float), 
          np.array([[5], [3], [3], [3]], dtype = float)]
}

A, b = matrices.get(choice)
m, n = A.shape

# Come ci dice la traccia dell'esercizio, la soluzione esatta di entrambi i sistema è
# il vettore colonn con tutte le sue componenti uguali a 1.
x_esatta = np.ones((n, 1))

# NO PIVOT 
P, L, U, flag = sl.LU_nopivot(A)
if flag == 0:
    x_nopivot, flag = sl.LUsolve(L, U, P, b)
    print("Soluzione calcolata senza pivot parziale:\n", x_nopivot)
else :
    print("Sistema non risolubile senza strategia pivotale")
    

# CON PIVOT PARZIALE
P, L, U, flag = sl.LU_pivot(A)
if flag == 0:
    x_pivot, flag = sl.LUsolve(L, U, P, b)
    print("Soluzione calcolata con pivot parziale:\n", x_pivot)
else :
    print("Sistema non risolubile con strategia pivotale")
