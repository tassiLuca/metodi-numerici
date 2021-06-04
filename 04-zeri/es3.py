#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esercizio 3
-----------
Utilizzare il metodo di Newton per determinare le radici dell’equazione f(x) = atan(x) con x0 = 1.2, 1.4 
assumendo tolx = 1.e − 6 e tolf = 1.e − 5. Che cosa si osserva?
"""

import my_zeri as zeri
import numpy as np
import sympy as sym
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt

x = sym.symbols('x')
fname = sym.atan(x)

f = lambdify(x, fname, np)
df = lambdify(x, sym.diff(fname, x, 1), np)
tolx = 1.e-6
tolf = 1.e-5
it_max = 500
a = -5
b = 5

x_axis = np.linspace(a, b, 100)
plt.plot(x_axis, 0 * x_axis, "black", x_axis, f(x_axis))
plt.title("f(x) = " + str(fname))
plt.show()

triggers = [1.2, 1.4]

for j in range(len(triggers)):
    approx, approx_seq, it = zeri.newton(f, df, triggers[j], tolx, tolf, it_max)
    print("TRIGGER = ", triggers[j], " ----> alpha = ", approx, " iterazioni = ", it)
    