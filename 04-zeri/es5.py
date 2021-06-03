#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esercizio 5
-----------
Si converta l’equazione x^2 − 5 = 0 nel problema di punto fisso
                            x = x − c(x^2 − 5) ≡ g(x).
Si scelgano diversi valori di c che assicurino la convergenza di x_{k+1} = x_k − c(x_k^2 − 5)
a α = sqrt(5) e si confrontino i risultati applicando il metodo di iterazione del punto fisso 
quando x0 = 2.5, tolx = 1.e − 8. 

Si disegnino infine su uno stesso grafico, la retta y = x, il grafico della funzione g(x), i punti di coordinate 
(x_k, 0) per ogni k >= 0, insieme alla poligonale di vertici (x_k, x_k), (x_k, x_{k + 1}), k >= 0, così da
poter visualizzare sia la convergenza alla soluzione α = sqrt(5) che il procedimento del metodo.
"""

import numpy as np
import sympy as sym
from sympy.utilities.lambdify import lambdify
import my_zeri as zeri
import matplotlib.pyplot as plt

trigger = 2.5
tolx = 1.e-8
max_it = 1000
a = 1.5
b = 3

c = [1/20, 1/6, 3/10, 2/5]
x = sym.symbols('x')
for j in range(len(c)):
    gname = x - c[j] * (x**2 - 5)
    g  = lambdify(x, gname, np)
    dg = lambdify(x, sym.diff(gname, x, 1), np)
    approx, approx_sequence, it = zeri.iterazione(g, trigger, tolx, max_it)
    
    print("iterazioni = ", it, " soluzione = ", approx)
    
    # grafico 
    x_axis = np.linspace(a, b, 100)
    plt.plot(x_axis, x_axis, "k-", x_axis, g(x_axis))
    
    # grafico le poligonali
    Vx = []
    Vy = []
    for k in range(it):
        Vx.append(approx_sequence[k])
        Vy.append(approx_sequence[k])
        Vx.append(approx_sequence[k])
        Vy.append(approx_sequence[k + 1])
    Vy[0] = 0
    
    plt.plot(Vx, Vy, 'r', approx_sequence, [0] * (it + 1), 'or-')
    
    plt.show()
    