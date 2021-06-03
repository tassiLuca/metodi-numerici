#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esercizio 2
-----------
Considerare il metodo di Newton e delle secanti per approssimare la più piccola radice positiva di ù
f(x) = tan(x) − x entro una tolleranza pari a 1.e − 8 (sia per la x che per la f). 
Precisamente, si determini il punto d'innesco x_0 come la j-esima (con 1 ≤ j ≤ 4) iterata del metodo di bisezione 
su [a, b] = [3/5π, 37/25π], mentre si assuma x_-1 = a per il metodo delle secanti. 
Cosa si osserva per le diverse scelte di x0?

NOTE:   Siccome f non ha concavità fissa in [a, b], affinché il metodo di Newton converga allo zero il valore
        di innesco x_0 deve essere scelto sufficientemente vicino ad alpha (vedi slides 46, 47), altrimenti la 
        convergenza non è assicurata: si vede infatti che, scegliendo come valore di innesco un punto vicino 
        alla radice, la successione generata dal meotodo di newton converge, altrimenti no.
        
        Anche per il metodo delle secanti, seppure possa essere più veloce della regula falsi, non converge sempre. 
        La convergenza è garantita se le approssimazioni iniziali sono ’abbastanza vicine’ alla radice α:
        convergenza locale (vedi slide 34).
"""

import numpy as np
import math
import my_zeri as zeri
import sympy as sym
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt

x = sym.symbols('x')
fname = sym.tan(x) - x

f = lambdify(x, fname, np)
df = lambdify(x, sym.diff(fname, x, 1), np)

a = math.pi * 3 / 5
b = math.pi * 37 / 25
tol = 1.e-8
max_it = 500

# Grafico la funzione.
x_axis = np.linspace(a, b, 100)
plt.plot(x_axis, 0 * x_axis, "black", x_axis, f(x_axis))
plt.title("f(x) = " + str(fname))

approx, approx_sequence, it = zeri.bisezione(f, a, b, tol)
triggers = approx_sequence[0:4]
# grafico con dei punti sull'asse delle ascisse le scelte dei valori di innesco  
# determinate dalle prime 4 iterazioni del metodo di bisezione.
plt.plot(triggers, np.zeros_like(triggers), "ro")
plt.show()

for j in range(len(triggers)):
    approx_newton, approx_sequence_newton, it_newton = zeri.newton(f, df, triggers[j], tol, tol, max_it)
    approx_sec, approx_sequence_sec, it_sec = zeri.secanti(f, a, triggers[j], tol, tol, max_it)
    
    print("TRIGGER = ", triggers[j])
    print("\tNEWTON -----> alpha = ", approx_newton, " iterazioni = ", it_newton)
    print("\tSECANTI ----> alpha = ", approx_sec, " iterazioni = ", it_sec)
