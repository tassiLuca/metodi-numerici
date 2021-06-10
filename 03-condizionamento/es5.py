#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esercizio 5
-----------
Condizionamento della matrice di Hilbert.
In matematica, una matrice di Hilbert è una matrice quadrata con componenti 
                    h_{i, j} = (i +j - 1)^{-1}
Ogni matrice di Hilbert è una matrice di Hankel, ovvero h_{i, j dipende solo da i + j; 
inoltre le componenti della sua matrice inversa sono numeri interi. 
La matrice di Hilbert è notoriamente una matrice fortemente malcondizionata. 
Essa interviene nei problemi di least squares fitting: data una funzione f(x), continua nell'intervallo 
x \in [0,1]. Vogliamo approssimare questa funzione con un polinomio p(x) di grado n - 1.
"""

import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl

# =============================================================================
# MATRICE DI HILBERT
# =============================================================================

n = 4
A = spl.hilbert(n)
b = np.ones(n)

x = spl.solve(A, b)
print("Soluzione esatta di Ax = b:", x)

db = 0.01 * np.array([1, -1, 1, -1], dtype = float)

dx = spl.solve(A, b + db)

# calcolo errori
err_dati = npl.norm(db, np.inf) / npl.norm(b, np.inf)
err_sol  = npl.norm(dx - x, np.inf) / npl.norm(x, np.inf)
print("Errore relativo percentuale sui dati =", err_dati * 100, "%")
print("Errore relativo percentuale sui risultati =", err_sol * 100, "%")
