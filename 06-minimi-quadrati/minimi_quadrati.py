#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 16:10:58 2021

06 - Approssimazione ai minimi quadrati
"""

import numpy as np
import scipy.linalg as spl 
import Sistemi_lineari as sl

def QR(x, y, n):
    """  
    Risoluzione con procedura forward di Lx = b con L triangolare inferiore.
        
    Parametri
    ----------
    x: vettore colonna con le ascisse dei punti
    y: vettore colonna con le ordinate dei punti
    n: grado del polinomio approssimante

    Valori di ritorno
    -------
    a: vettore colonna contenente i coefficienti incogniti
    """
    
    # in generale, per un polinomio di grado n vi sono n + 1 coefficienti
    H = np.vander(x, n + 1)     # Matrice di Vandermonde
    Q, R = spl.qr(H)
    y1 = np.dot(Q.T, y)
    a, flag = sl.Usolve(R[0:n+1, :], y1[0:n+1])
    return a