#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Si noti come, a fronte di una piccola perturbazione dei dati dello 0,12%, l'errore sul risultato risulta essere
oltre l'81%. Questo determina che il primo sistema lineare è MAL condizionato. Al contrario il secondo sistema 
lineare è ben condizionato, perché a piccole perturbazioni (relative) nei dati corrispondono perturbazioni 
(relative) sul risultato dello stesso ordine di grandezza (0,66% sui dati e 1.3% sui risultati => stesso ordine di
grandezza).
"""

import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl

# =============================================================================
# SISTEMA 1
# =============================================================================

A = np.array([[3, 5],
              [3.01, 5.01]], dtype = float)
b = np.array([10, 1], dtype = float)

x = spl.solve(A, b)     # risolvo il sistema lineare

# perturbo il coeff di x nella seconda equazione
dA = np.array([[0, 0,], 
               [0.01, 0]], dtype = float)

x_pert = spl.solve(A + dA, b) # risolvo il sistema lineare perturbato

print("SISTEMA 1")
err_dati = npl.norm(dA, np.inf) / npl.norm(A, np.inf)
print("\t Errore relativo sui dati del sistema 1 perturbati:", err_dati * 100, "%")
err_sol = npl.norm(x_pert - x, np.inf) / npl.norm(x, np.inf)
print("\t Errore relativo sulla soluzione del sistema 1 perturbato:", err_sol * 100, "%")

# =============================================================================
# SISTEMA 2
# =============================================================================
A = np.array([[5, 10],
              [2, 1]], dtype = float)
b = np.array([15, 1], dtype = float)

x = spl.solve(A, b)

dA = np.array([[0, 0.1,], 
               [0, 0]], dtype = float)

print("A perturbata: \n", A + dA)
x_pert = spl.solve(A + dA, b)

print("SISTEMA 2 - perturbazione 1")
err_dati = npl.norm(dA, np.inf) / npl.norm(A, np.inf)
print("\t Errore relativo sui dati del sistema 1 perturbati:", err_dati * 100, "%")
err_sol = npl.norm(x_pert - x, np.inf) / npl.norm(x, np.inf)
print("\t Errore relativo sulla soluzione del sistema 1 perturbato:", err_sol * 100, "%")

dA = np.array([[0, -0.1,], 
               [0, 0]], dtype = float)

print("A perturbata: \n", A + dA)
x_pert = spl.solve(A + dA, b)

print("SISTEMA 2 - perturbazione 2")
err_dati = npl.norm(dA, np.inf) / npl.norm(A, np.inf)
print("\t Errore relativo sui dati del sistema 1 perturbati:", err_dati * 100, "%")
err_sol = npl.norm(x_pert - x, np.inf) / npl.norm(x, np.inf)
print("\t Errore relativo sulla soluzione del sistema 1 perturbato:", err_sol * 100, "%")