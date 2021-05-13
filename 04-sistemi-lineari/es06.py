#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04 - Esercizio 6
-----------
Dato n = 100, calcolare Q = I - 2 v v^T, dove I è la matrice identità di ordine n e v è un vettore colonna di n 
componenti formato da numeri casuali ed avente norma 2 unitaria. Quindi, per k = 1, ..., 20, porre:
                D = I
                D[n - 1, n - 1] = 10^k
                A = Q D
Tale costruzione produce una matrice A che abbia esattamente numero di condizionamento 10^k in norma 2.
Al variare di k e quindi del numero di condizionamento di A, studiare l'errore del metodo di eliminazione gaussiana
senza pivoting e con pivoting parziale nella risoluzione del sistema lineare Ax = b, dove b = A x_esatta e 
x_esatta = (1, 1, ..., 1)^T.
"""

import numpy as np
import Sistemi_lineari as sl


