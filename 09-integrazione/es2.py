#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esercizio 2
-----------
Si utilizzino le function relative alle formule di quadratura automatica dei trapezi e di 
Simpson per fornire una approssimazione dei seguenti integrali con tol = 1.e − 6:
    - log(x)
    - sqrt(x) 
    - abs(x)
"""

import numpy as np
import sympy as sym
from sympy.utilities.lambdify import lambdify
import integrazione 

x = sym.symbols('x')
tol = 1.e-6

choice = input("Scegli funzione [1:3]: ")
functions = {
    '1': [sym.log(x),  1.0, 2.0],
    '2': [sym.sqrt(x), 0.0, 1.0],
    '3': [sym.Abs(x), -1.0, 1.0]
}
fname, a, b = functions.get(choice)
f = lambdify(x, fname, np)
res = sym.integrate(fname, (x, a, b))
print("INTEGRALE ESATTO = ", res)

res_trapezi_auto, steps_trapezi = integrazione.trap_toll(f, a, b, tol)
print("Integrale con trapezi automatica = ", res_trapezi_auto, "con ", steps_trapezi, " steps")
res_simpson_auto, steps_simpson = integrazione.simp_toll(f, a, b, tol)
print("Integrale con Simpson automatica = ", res_simpson_auto, "con ", steps_simpson, " steps")

