#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 22 10:40:22 2021

@author: lucatassi
"""

import numpy as np

def plagr(nodes, j):
    '''
    Restituisce i coefficienti del k-esimo polinomio di Lagrange
    associato ai punti del vettore xnodi.

    Parameters
    ----------
    xnodi : TYPE
        DESCRIPTION.
    k : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    '''
              n
            ----                            (x - x_0)...(x - x_(j-1))(x - x_(j+1))...(x - x_n)
    L_j(x) = ||  (x-x_i) / (x_j-x_i) = ------------------------------------------------------------
             ||                         (x_j - x_0)...(x_j - x_(j-1))(x_j - x_(j+1))...(x_j - x_n)
          i=0, i!=j
    
    ---> x_0, x_1, ..., x_(j-1), x_(j+1), ..., x_n (nota che x_j non è presente)
    sono gli zeri del polinomio fondamentale di Lagrange L_j(x) (vedi slide 9).
    '''
    
    zeroes = np.zeros_like(nodes)
    n = nodes.size
    if k == 0:
        zeroes = nodes[1:n]
    else:
        zeroes = np.append(nodes[0:k], nodes[k+1:n])
    
    np.poly()
