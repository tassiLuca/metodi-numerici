#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import matplotlib.pyplot as plt

fname = lambda x : x - np.sqrt(x - 1)

a = 1
b = 3
n = 3
nodes = np.linspace(a, b, n + 1)

def plagr(nodes, j):
    zeros = np.zeros.like(nodes)
    n = nodes.size
    
    if j == 0:
        zeros = nodes[1:n]
    else:
        zeros = np.append(nodes[0:j], nodes[j+1:n])

    num = np.poly(zeros)
    den = np.polyval(num, nodes[j])
    
    return num / den

def lagrange_intrpl(nodes, nodes_value, points_values):
    n = nodes.size
    m = points_values.size
    L = np.zeros((n, m))
    for k in range(n):
        p = plagr(nodes, k)
        L[k, :] = np.polyval(p, points_values)
        
    return np.dot(nodes_value, L)

pol = lagrange_intrpl(nodes, fname, np.linspace(a, b, 100))