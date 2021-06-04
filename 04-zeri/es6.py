#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esercizio 6
-----------
Applicare il metodo di iterazione del punto fisso x_{k+1} = g(xk) per determinare lo zero dell’equazione 
f(x) = x**3 + 4 * x**2 − 10 quando x_0 = 1.5, tolx = 1.e − 7 e nmax = 1000, proponendo diverse scelte della 
funzione di iterazione g per garantire:
(a) convergenza lineare con costante asintotica di convergenza prossima a 0;
(b) convergenza lineare con costante asintotica di convergenza prossima a 0.5;
(c) convergenza lineare con costante asintotica di convergenza prossima a 1.
"""

import numpy as np
import my_zeri as zeri
import matplotlib.pyplot as plt
import sympy as sym
from sympy.utilities.lambdify import lambdify

f = lambda x : x**3 + 4 * x**2 - 10

tolx = 1.e-7
max_it = 1000
trigger = 1.5
a = 1
b = 1.8

x = sym.symbols('x')
functions = {
    '1': sym.sqrt(10 / (x + 4)),        # p = 1, c = 0.127229401770925
    '2': 1/2 * sym.sqrt(10 - x**3),         # p = 1, c = 0.511961226874885
    '3': (10 + x) / (x**2 + 4 * x + 1), # p = 1, c = 0.983645643784931
    '4': sym.sqrt( 10 / x - 4 * x)      # non converge
}
choice = input("Scegli g(x): ")

gname = functions.get(choice)
g  = lambdify(x, gname, np)
dg = lambdify(x, sym.diff(gname, x, 1), np)

# Calcolo la radice
root, roots_approximations, it = zeri.iterazione(g, trigger, tolx, max_it)
print("Iterazioni = ", it, ", Soluzione = ", root)

# grafico la funzione f(x)
x_axis = np.linspace(a, b, 100)
plt.plot(x_axis, 0 * x_axis, "k--", x_axis, f(x_axis), root, f(root), '*')
# grafico la funzione g(x) e la bisettrice y = x
plt.plot(x_axis, x_axis, "r--", x_axis, g(x_axis), root, g(root), 'o')
plt.title("Grafici f(x) e g(x)")
plt.legend(["y = 0", "f(x)", "Radice", "y = x", "g(x)", "Punto fisso"])
plt.show()

# Calcolo l'ordine del metodo
ordine = zeri.stima_ordine(roots_approximations, it)
# Essendo il metodo con ordine di convergenza lineare, la costante asintotica di convergenza è data
# da |g'(alfa)| dove alfa è la radice.
costante_convergenza = abs(dg(root))
print("Ordine di convergenza = ", ordine, ", Costante di convergenza = ", costante_convergenza)

# Posso giustifcare la convergenza del procedimento iterativo guardando la derivata prima di g(x)
# in un intorno della soluzione: il metodo genera una successione di iterati convergenti alla radice alfa
# ed appartenenti a questo intorno se |g'(x)|< 1 in un intorno della soluzione
plt.plot(x_axis,dg(x_axis))
plt.title('Funzione dg(x)')
plt.show()

# grafico la poligonale in modo tale sia evidente se il metodo converge o meno
plt.plot(x_axis, x_axis, 'k-', x_axis, g(x_axis))
Vx = []
Vy = []
for k in range(it):
    Vx.append(roots_approximations[k])
    Vy.append(roots_approximations[k])
    Vx.append(roots_approximations[k])
    Vy.append(roots_approximations[k + 1])
Vy[0] = 0
plt.plot(Vx, Vy, 'r', roots_approximations, np.zeros_like(roots_approximations), 'o-')
plt.title("Poligonale")
plt.show()