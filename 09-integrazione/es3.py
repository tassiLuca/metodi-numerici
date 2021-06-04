#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esercizio 3
-----------
Calcolare con la formula dei trapezi e di Simpson composite un’approssimazione dei seguenti integrali
    - cos(x)
    - x exp(x) cos(x^2) 
    - (sin(x))^α cos(x) dx, α = 13/2, 5/2, 1/2,
utilizzando valori di tolleranza tol = 1.e−k con k = 4 : 10. Rappresentare su tre grafici distinti e 
su tre tabelle, l’errore relativo (in scala logaritmica), il numero di sottointervalli N utilizzati e il 
numero di valutazioni della funzione integranda al variare di k.
"""

import numpy as np
import math
import sympy as sym
from sympy.utilities.lambdify import lambdify
import integrazione 
import matplotlib.pyplot as plt

x = sym.symbols('x')
choice = input("Scegli una funzione [1-3]: ")
functions = {
    '1': [sym.cos(x), 0.0, 2.0],
    '2': [x * sym.exp(x) * sym.cos(x**2), -2 * math.pi, 0],
    '3': [sym.sin(x)**(13.0/2.0) * sym.cos(x), 0.0,math.pi/2],
    '4': [sym.sin(x)**(5.0/2.0) * sym.cos(x), 0.0, math.pi/2],
    '5': [sym.sin(x)**(1.0/2.0) * sym.cos(x), 0.0, math.pi/2]
}
fname, a, b = functions.get(choice)
f = lambdify(x, fname, np)

res = sym.integrate(fname, (x, a, b))

trap_k = []
res_trapezi = []
steps_trapezi = []
simp_k = []
res_simpson = []
steps_simpson = []

start = 4
stop = 10
for k in range(start, stop + 1):
    tol = 10**(-k)
    trapezi, steps_trap = integrazione.trap_toll(f, a, b, tol)
    if steps_trap > 0:
        trap_k.append(k)
        res_trapezi.append(trapezi)
        steps_trapezi.append(steps_trap)
        
    simpson, steps_simp = integrazione.simp_toll(f, a, b, tol)
    if steps_simp > 0:
        simp_k.append(k)
        res_simpson.append(simpson)
        steps_simpson.append(steps_simp)
        
err_trap = np.abs(np.array(res_trapezi) - res) / np.abs(res)
err_simp = np.abs(np.array(res_simpson) - res) / np.abs(res)

plt.semilogy(trap_k, err_trap, 'o-', simp_k, err_simp, '*-')
plt.legend(["Errore trapezi al variare di tol", "Errori Simpson al variare di tol"])
plt.show()

plt.plot(trap_k, steps_trapezi, 'o-', simp_k, steps_simpson, '*-')
plt.legend(["Suddivisione trapezi al variare di tol", "Suddivisione Simpson al variare di tol"])
plt.show()