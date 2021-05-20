#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04 - Esercizio 8
-----------
Per valori di n = 48 : 2 : 58, si consideri il sistema lineare Ax = b con A e b così definiti:
    
        |--  1 se i = j o j = n - 1
    a = |-- -1 se i > j                     b = A [1, ..., 1]^T
        |--  0 altrimenti

in modo che risolti x = [1, 1, ..., 1]^T. Si risolva tale sistema con il metodo di fattorizzazione LU con massimo pivot
parziale e il metodo di fattorizzazione QR. Calcolare gli errori relativi da cui sono affette le soluzioni calcolate
con i due metodi e produrre, al variare di n, un grafico in scala semilogaritmica degli errori relativi calcolati.
Che cosa si osserva?
[NOTA: MOLTO SIMILE, SE NON IDENTICO ALL'ESERCIZIO 07 (al netto dell'utilizzo di una matrice dei coefficienti diversa)]
"""
import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl
import matplotlib.pyplot as plt
import Sistemi_lineari as sl

def A_matrix(n):
    '''
    Costruisce la matrice dei coefficienti A così come richiesto dalla traccia.

    Parametri
    ----------
    n: ordine della matrice.

    Valori di ritorno
    -------
    A: matrice dei coefficienti.
    '''
    A = np.zeros((n, n), dtype = float)
    for i in range(0, n) :
        for j in range(0, n) :
            if i == j or j == n-1:
                A[i, j] = 1
            elif i > j :
                A[i, j] = -1
            else :
                A[i, j] = 0
    return A

indice_condizionamento = []
err_rel_pivot = []
err_rel_qr = []

start = 48
stop = 59
step = 2

for n in range(start, stop, step):
    A = A_matrix(n)
    indice_condizionamento.append(npl.cond(A, 2))
    x_esatta = np.ones((n, 1))
    b = np.dot(A, x_esatta)
    
    # Fattorizzazione LU con pivoting parziale
    P, L, U, flag = sl.LU_pivot(A)
    if flag == 0:
        x_pivot, flag = sl.LUsolve(L, U, P, b)
    else :
        print("Sistema non risoluile con strategia pivotale")
    
    err_pivot = npl.norm(x_pivot - x_esatta, 2) / npl.norm(x_esatta, 2)
    err_rel_pivot.append(err_pivot)
    
    # Fattorizzazione QR
    '''
    A = QR <=> Ax = b <=> QRx = b. 
    Detto y = Rx si calcola:
        I)  Qy = b ----> siccome Q è ortogonale: l'inversa di Q coincide con la trasposta e quindi y = Q.T b.
        II) Rx = y ----> siccome R è triangola superiore si risolve mediante sostituzione all'indietro.
    '''
    Q, R = spl.qr(A)
    y = np.dot(Q.T, b)
    x_qr, flag = sl.Usolve(R, y)
    
    err_qr = npl.norm(x_qr - x_esatta, 2) / npl.norm(x_esatta, 2)
    err_rel_qr.append(err_qr)
    
plt.plot(range(start, stop, step), err_rel_pivot, 'ro-', 
         range(start, stop, step), err_rel_qr, 'go-')
plt.legend(['Pivot', 'QR'])
plt.xlabel('Ordine della matrice di Hankel n')
plt.ylabel('Errore relativo sulla soluzione')
plt.show() 
    