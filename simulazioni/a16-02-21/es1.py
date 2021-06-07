#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESAME 16 Febbraio 2021 - Esercizio 1.
ZERI DI FUNZIONI.
"""

import numpy as np
from scipy.optimize import fsolve
import sympy as sym
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt

x = sym.symbols('x')
fname = x - (1 / 3) * (sym.sqrt(30 * x - 25))
fpname = sym.diff(fname, x, 1)
f  = lambdify(x, fname, np)
df = lambdify(x, fpname, np)
a = 5 / 6
b = 25 / 6

'''
(a) si stabilisca quante radici reali ha f nell’intervallo [a, b] e si giustifichi la risposta

'''
root = fsolve(f, 2)
x_axis = np.linspace(a, b)
#np.seterr(divide = 'ignore') 
plt.plot(x_axis, 0 * x_axis, 'k', x_axis, f(x_axis), 'orange', x_axis, df(x_axis), 'y', root, f(root), 'r*')
plt.legend(["y = 0", "f(x) = " + str(fname), "f'(x) = " + str(fpname), "Radice di f(x)"])
plt.title("Grafico funzioni")
plt.show()

'''
(b) si costruisca un metodo iterativo che, partendo da x_0 = 4, converga ad α (zero di f), quadraticamente.
(c) si verifichi numericamente l’ordine di convergenza del metodo implementato al punto b);
'''

def newton_m(fname, fpname, trigger, tolx, toly, max_it, m):
    it = 0
    xk = []
    
    prv = trigger
    f_prv = fname(prv)
    df_prv = fpname(prv)
    
    while True:
        it += 1
        if abs(df_prv) < np.spacing(1):
            print("df_prv nulla")
            return prv, xk, it
        nxt = prv - m * (f_prv / df_prv)
        f_nxt = fname(nxt)
        df_nxt = fpname(nxt)
        xk.append(nxt)
        
        if it >= max_it or abs(nxt - prv) < tolx * abs(nxt) or abs(f_nxt) < toly:
            if it >= max_it:
                print("Numero massimo di iterazioni raggiunte")
            break
        else:
            prv = nxt
            f_prv = f_nxt
            df_prv = df_nxt
    
    return nxt, xk, it
        
def stima_ordine(xk, it):
    p = []
    
    for k in range(it - 3):
        p.append(np.log(abs(xk[k + 2] - xk[k + 3]) / abs(xk[k + 1] - xk[k + 2])) / 
                 np.log(abs(xk[k + 1] - xk[k + 2]) / abs(xk[k] - xk[k + 1])))
        
    return p[-1]

m = 2           # molteplicità della radice
max_it = 2048
toll = 1e-8
trigger = 4.0
sol, xk, it = newton_m(f, df, trigger, toll, toll, max_it, m)
ordine = stima_ordine(xk, it)
print("Radice di f(x): ", sol, " raggiunto con", it, "iterazioni. Ordine di convergenza: ", ordine)

'''
(d) Si rappresenti in un grafico in scala semilogaritmica sulle y il vettore dei valori assoluti di tutte le 
    approssimazioni calcolate dal procedimento iterativo (comprese tra |x0| e |α|), in funzione del numero di 
    iterazioni compiute.
'''
plt.semilogy(range(1, it + 1), xk)
plt.title("Grafico valori assoluti delle approssimazioni in funzione del numero di iterazioni.")
plt.show()

'''
(e) Si stabilisca se il metodo iterativo proposto al punto b) può convergere ad α quadraticamente anche
    partendo dall’estremo sinistro dell’intervallo, ossia da x0 = 5 / 6 , e si giustifichi la risposta. 
    
    NON converge perché in a la derivata prima di f si annulla. 
'''
sol, xk, it = newton_m(f, df, a, toll, toll, max_it, m)
print("Radice di f(x): ", sol, " raggiunto con", it, "iterazioni.")