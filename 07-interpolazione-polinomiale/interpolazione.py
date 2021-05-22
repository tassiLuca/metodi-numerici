#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 22 10:40:22 2021

@author: lucatassi
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
    
    
    
    
    
    
    
