#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esercizio 4
-----------
Qui addirittura, per una perturbazione dell'0.001 % corrisponde un errore relativo sul risultato del 99,999%!!
"""

import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl

A = np.array([[6,     63,     662.2], 
              [63,    662.2,  6967.8],
              [662.2, 6967.8, 73393.5664]], dtype = float)
b = np.array([1.1, 2.33, 1.7])

x = spl.solve(A, b)

dA = np.array([[1, 0, 0], 
               [0, 0, 0], 
               [0, 0, 0]], dtype = float)

dx = spl.solve(A + dA, b)

err_dati = npl.norm(dA, np.inf) / npl.norm(A, np.inf)
err_sol  = npl.norm(dx - x, np.inf) / npl.norm(x, np.inf)
print("Perturbazione sui dati:", err_dati * 100, "%")
print("Perturbazione sui risultati:", err_sol * 100, "%")
