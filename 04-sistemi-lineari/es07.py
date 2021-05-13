#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04 - Esercizio 7
-----------
Per valori di n = 4 : 6 : 40, si consideri il sistema lineare Ax = b con A matrice di Hankel di ordine n e b scelto
in modo che risolti x = [1, 1, ..., 1]^T. Si risolva tale sistema con il metodo di fattorizzazione LU con massimo pivot
parziale e il metodo di fattorizzazione QR. Calcolare gli errori relativi da cui sono affette le soluzioni calcolate
con i due metodi e produrre, al variare di n, un grafico in scala semilogaritmica degli errori relativi calcolati.
Che cosa si osserva?

NOTE: Il vantaggio della fattorizzazione QR è una maggiore stabilità rispetto alla fattorizzazione LU, senza dover
      ricorrere ad una strategia di pivoting. Lo svantaggio è il "raddoppio" del costo computazionale (slide 62 teoria).
"""
import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl
import matplotlib.pyplot as plt
import Sistemi_lineari as sl

def Hankel(n):
    '''
    Costruisce la matrice di Hankel.

    Parametri
    ----------
    n: ordine della matrice di Hankel.

    Valori di ritorno
    -------
    H: matrice di Hankel di ordine n generata.
    '''
    H = np.zeros((n, n), dtype = float)
    for i in range(0, n) :
        for k in range(i + 1 - n, i + 1) :
            if (k > 0):
                H[i, n - 1 + k - i] = 2.0 ** (k + 1)
            else :
                H[i, n - 1 + k - i] = 2.0 ** (1 / (2 - k - 1))
    return H

indice_condizionamento = []
err_rel_pivot = []
err_rel_qr = []

start = 4
stop = 41
step = 6

for n in range(start, stop, step):
    A = Hankel(n)
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
    