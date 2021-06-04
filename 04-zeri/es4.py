#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esercizio 4
-----------
Utilizzare il metodo di Newton e il metodo di Newton modificato per il calcolo dello zero di molteplicità 2 
della funzione f(x) = x^3 + x^2 − 33x + 63 con x_0 = 1, tolx = 1.e − 12 e tolf = 1.e − 12. 
Calcolare infine, a partire dai valori di {x_k} ottenuti nei due casi, la stima dell’ordine di convergenza p.

NOTE:   Per radici multiple la convergenza del metodo di Newton si riduce a lineare (slide 74). 
        Per ovviare a ciò si premoltiplica per m (= molteplicità dello zero) il rapporto f(x_i) / f'(x_i).
        Questa variante di Newton è detta Metodo di Newton Modificato e garantisce che l'ordine di convergenza sia
        quadratico (vedi slide 75).
"""

import numpy as np
import sympy as sym
from sympy.utilities.lambdify import lambdify
import my_zeri as zeri
import matplotlib.pyplot as plt

x = sym.symbols('x')
fname = x**3 + x**2 - 33*x + 63
trigger = 1
tolx = 1.e-12
tolf = 1.e-12
max_it = 500
a = 2
b = 4

f = lambdify(x, fname, np)
df = lambdify(x, sym.diff(fname, x, 1), np)

# Grafico la funzione
x_axis = np.linspace(a, b, 100)
plt.plot(x_axis, 0 * x_axis, "black", x_axis, f(x_axis))
plt.title("f(x) = " + str(fname))
plt.show()

# metodo di newton 
x_newton, approx_seq_newton, it_newton = zeri.newton(f, df, trigger, tolx, tolf, max_it)
p = zeri.stima_ordine(approx_seq_newton, it_newton)
print("METODO DI NEWTON")
print("\t alpha = ", x_newton, " iterazioni = ", it_newton, " ordine di convergenza = ", p)

# metodo di newton modificato
m = 2 # molteplicità di alpha
x_newton_m, approx_seq_newton_m, it_newton_m = zeri.newton_m(f, df, trigger, m, tolx, tolf, max_it)
p = zeri.stima_ordine(approx_seq_newton_m, it_newton_m)
print("METODO DI NEWTON MODIFICATO")
print("\t alpha = ", x_newton_m, " iterazioni = ", it_newton_m, " ordine di convergenza = ", p)