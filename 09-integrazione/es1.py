#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esercizio 1
-----------
Si approssimi con la formula dei trapezi e la formula di Simpson composite, per valori di N = 2^k , k = 1 : 8, 
l’integrale per le funzioni:
        - f(x) = x^10,
        - f(x) = log(x + 1),
        - f(x) = arcsin(x).
Si confrontino i valori ottenuti con l’integrale esatto e si illustri con una tabella e un grafico 
(in scala semilogaritmica) l’andamento dell’errore relativo.
"""

import sympy as sym
from sympy.utilities.lambdify import lambdify
import numpy as np
import integrazione
import matplotlib.pyplot as plt

x = sym.symbols('x')

choice = input("Scegli una funzione [1-3]: ")
functions = {
    '1': [x**10, 0.0, 1.0],
    '2': [sym.log(x + 1), 0.0, 1.0],
    '3': [sym.asin(x), 0.0, 1.0]
}
fname, a, b = functions.get(choice)
f = lambdify(x, fname, np)

res = float(sym.integrate(fname, (x, a, b)))

start = 1
stop = 8
N = [1, 2, 4, 8, 16, 32 ,64 ,128, 256]
    
res_trapezi = []
res_simpson = []
err_trapezi = np.zeros((stop - start, 1))
err_simpson = np.zeros((stop - start, 1))
for n in N:
    res_trapezi.append(integrazione.trap_comp(f, a, b, n))
    res_simpson.append(integrazione.simpson_comp(f, a, b, n))
    
err_trapezi = np.abs(np.array(res_trapezi) - res) / np.abs(res)
err_simpson = np.abs(np.array(res_simpson) - res) / np.abs(res)
plt.semilogy(N, err_trapezi, 'o-', N, err_simpson, '*-')
plt.legend(['Errore Trapezi Composita', 'Errore Simpson Composita'])
plt.show()