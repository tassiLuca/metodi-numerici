#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Appello del 13 Gennaio 2021 - Esercizio 2
-----------------------------------------
SISTEMI LINEARI
"""

import numpy as np
import numpy.linalg as npl
import sympy as sym
from sympy.utilities import lambdify
import matplotlib.pyplot as plt

a = sym.symbols('a')
A_name = np.array([[1 + a, 1],
                   [-1,   -1]])

Ainv_name = 1 / a * np.array([[1,     1], 
                         [-1, -1-a]]) 
A = lambdify(a, A_name, np)
Ainv = lambdify(a, Ainv_name, np)

'''
a)  si calcoli l’espressione della norma infinito di A al variare di a.
b)  si calcoli l’espressione della norma infinito dell'inversa di A al variare di a.
'''
norm_inf_A = 2 + a
norm_inf_Ainv = (2 + a) / a

'''
c)  si calcoli l’espressione del numero di condizionamento di A in norma infinito al variare di a, e se ne
    tracci un grafico per a ∈ [0.5, 9.5].
'''
a = np.linspace(0.5, 9.5, 100, True)
condA = (2 + a) * ((2 + a) / a)
plt.title("Condizionamento di A")
plt.xlabel("a")
plt.ylabel("Indice di condizionamento di A(a)")
plt.plot(a, condA)
plt.grid(True)
plt.show()

'''
d)  si dica per quale valore di a si ha il miglior condizionamento della matrice A;
e)  si dica per quali valori di α si ha il peggior condizionamento della matrice A;
 
    Il valore minimo di condA si ottiene per a = 2, quindi il condizionamento ottimo si ha per a = 2,
    mentre per a -> 0 e a -> +inf risulta condA -> +inf, quindi il numero di condizionamento peggiore 
    si ha per a prossimo a 0 o a grande
    
f)  per a = 10−5 e b = [2, −2].T
    - si trovi il vettore soluzione x del sistema lineare Ax = b;
    - si perturbi la matrice dei coefficienti della quantità

            δA = 0.001 ∗ [[0, 1], [0, 0]]

    si calcoli l’errore relativo sui dati e lo si confronti con l’errore relativo sulla soluzione.
    Che cosa si osserva? Come si motivano i risultati ottenuti?

    Siccome per a molto piccolo condA -> +inf, il problema risulta essere mal condizionato: a piccole variazioni
    sui dati (0.05%) corrispondono grandi variazioni sui risultati (101%).
'''

a = 1.e-5
b = np.array([2, -2])

x = npl.solve(A(a), b)
print("Soluzione esatta = ", x)

# perturbazione della matrice e soluzione del sistema perturbato
dA = 0.001 * np.array([[0, 1], 
                       [0, 0]])
dx = npl.solve(A(a) + dA, b)
print("Soluzione perturbata = ", dx)

# calcolo degli errori
err_dati = npl.norm(dA) / npl.norm(A(a))
err_sol  = npl.norm(dx) / npl.norm(x)
print("Errore sui dati = ", err_dati * 100, "%")
print("Errore sui risultati = ", err_sol * 100, "%")
