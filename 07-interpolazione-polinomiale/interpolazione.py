#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interpolazione polinomiale.
"""

import numpy as np

def plagrange(nodes, j):
    '''
    Restituisce i coefficienti del j-esimo polinomio di Lagrange associato ai punti del vettore nodes.

    Parametri
    ----------
    nodes : i nodi dell'interpolazione
        j : l'indice che definisce il polinomio fondamentale di Lagrange da valutare.
        
    Valori di ritorno
    -------
    p : vettore contente i coefficienti del j-esimo polinomio di Lagrange.

    '''
    '''
    FUNZIONI BASE DI LAGRANGE.
              n
            ----                            (x - x_0)...(x - x_(j-1))(x - x_(j+1))...(x - x_n)
    L_j(x) = ||  (x-x_i) / (x_j-x_i) = ------------------------------------------------------------
             ||                         (x_j - x_0)...(x_j - x_(j-1))(x_j - x_(j+1))...(x_j - x_n)
          i=0, i!=j
    
    ---> x_0, x_1, ..., x_(j-1), x_(j+1), ..., x_n (nota che x_j non è presente) sono gli zeri del 
         polinomio fondamentale di Lagrange L_j(x) (vedi slide 9).
    '''
    
    zeroes = np.zeros_like(nodes)
    n = nodes.size
    # qui tolgo dal vettore dei nodi la j-esima componente
    if j == 0:
        zeroes = nodes[1:n]
    else:
        zeroes = np.append(nodes[0:j], nodes[j+1:n])
    
    # Preso in input un vettore, restituisce i coefficienti del polinomio che ha per zeri quel vettore.
    # Ad esempio: a = np.poly(np.array([2, 3])) restituisce in output array([ 1., -5.,  6.]). 
    # Infatti: (x - 2)(x - 3) = x^2 - 5x + 6. 
    # In questo modo ho ottenuto il numeratore.
    numerator = np.poly(zeroes)
    
    # Il denominatore è ottenuto valutando il polinomio che ha per coefficienti 
    # quelli in numerator, valutati nel nodo j-esimo escluso.
    denominator = np.polyval(numerator, nodes[j])
    
    p = numerator / denominator
    
    return p
    
def lagrange_interp(nodes, nodes_values, points_values):
    """
    Determina in un insieme di punti il valore del polinomio interpolante
    ottenuto dalla formula di Lagrange.

    Parametri
    ----------
    nodes         : vettore con i nodi dell'interpolazione
    nodes_values  : vettore con i valori dei nodi
    points_values : vettore con i punti in cui si vuole valutare il polinomio

    Valori di ritorno
    -------
    Vettore contenente i valori assunti dal polinomio interpolante.
    """
    '''
    Si deve calcolare:       
            P_n(x_i) = \sum_{j=0}^{n} y_j * L_j(x_i)    i = 0, ..., m
    Per farlo costruisco la matrice L delle funzioni base di Lagrange, cosituita da tante 
    righe quanti sono gli n + 1 coefficienti e, fissata la riga k-esima, ho il polinomio k-esimo
    valutato in tutti i punti dati. Il risultato sarà dato dal prodotto matriciale tra il vettore
    contenente gli y_j (ovvero le ordinate dei nodi di interpolazione) e la matrice L:
        
                           | L_0(x_0)  L_0(x_1)  ...  L_0(X_m) |   | y_0*L_0(x_0) + y_1*L_1(x_0) + ... + y_n*L_n(x_0) |
                           | L_1(x_0)  L_1(x_1)  ...  L_1(X_m) |   | y_0*L_0(x_1) + y_1*L_1(x_1) + ... + y_n*L_n(x_1) |
    [ y_0, y_1, ..., y_n ] |    .          .              .    | = |                                                  |
                           |    .          .      .       .    |   |                                                  |
                           |    .          .              .    |   |                                                  |
                           | L_n(x_0)  L_n(x_1)  ...  L_n(X_m) |   | y_0*L_0(x_m) + y_1*L_1(x_m) + ... + y_n*L_n(x_m) |
    '''
    n = nodes.size
    m = points_values.size
    L = np.zeros((n, m))
    for k in range(n):
        p = plagrange(nodes, k)
        L[k, :] = np.polyval(p, points_values)
        
    return np.dot(nodes_values, L)
    