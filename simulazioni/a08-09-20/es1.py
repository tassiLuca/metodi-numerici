#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APPELLO DELL'8 SETTEMBRE 2020
Esercizio 1
ZERI DI FUNZIONE
"""

import numpy as np
import sympy as sym
from sympy.utilities import lambdify
import matplotlib.pyplot as plt
fig = plt.figure()
from scipy.optimize import fsolve

x = sym.symbols('x')
fname   = x - 2 * sym.sqrt(x - 1)
dfname  = sym.diff(fname, x, 1)
ddfname = sym.diff(dfname, x, 1)

f   = lambdify(x, fname, np)
df  = lambdify(x, dfname, np)
ddf = lambdify(x, ddfname, np)
a = 1
b = 3

'''
a)  si stabilisca quante radici reali ha f nell’intervallo [1, 3] e si giustifichi la risposta;
    Con fsolve determino la soluzione esatta α = 2.0. Graficando la funzione e la sua derivata prima
    si osserva che α è una radice multipla di f di molteplicità 2.
'''
root = fsolve(f, 2.5)
print("Zero della funzione calcolata con fsolve: ", root)

x_axis = np.linspace(a, b)
plt.plot(x_axis, np.zeros_like(x_axis), 'k', x_axis, f(x_axis), root, df(root), 'o', root, ddf(root), '*')
plt.legend(["y = 0", "f(x) = " + str(fname), "f'(α) = 0", "f''(α) != 0"])
plt.grid(True)
plt.show()

'''
b) si costruisca un metodo iterativo che, partendo da x(0) = 3, converga ad α (zero di f ), quadraticamente;
'''
max_it = 4096
tol = 10e-12
trigger = 3
m = 2   # molteplicità di α

def newton_m(fname, dfname, m, trigger, tolx, toly, max_it):
    sequence = []    
    it = 0
    prv = trigger
    f_prv = fname(prv)
    df_prv = dfname(prv)
    
    while True:
        it += 1
        nxt = prv - m * (f_prv / df_prv)
        print("--", nxt)
        f_nxt  = fname(nxt)
        df_nxt = dfname(nxt)
        sequence.append(nxt)
        
        if it >= max_it or abs(prv - nxt) < tolx * abs(nxt) or abs(f_nxt) < toly:
            break
        else:
            prv = nxt
            f_prv = f_nxt
            df_prv = df_nxt
    
    return nxt, sequence, it
            

my_root, roots_seq, it = newton_m(f, df, m, trigger, tol, tol, max_it)
print("Zero della funzione determinato con il metodo di Newton Modificato: ", my_root, "raggiunto con", it, "iterazioni")

'''
c) si verifichi numericamente l’ordine di convergenza del metodo implementato al punto b);
'''

def stima_ordine(xk, it):
    p = []
    
    for k in range(it - 3):
        p.append(np.log(abs(xk[k + 2] - xk[k + 3]) / abs(xk[k + 1] - xk[k + 2])) / 
                 np.log(abs(xk[k + 1] - xk[k + 2]) / abs(xk[k] - xk[k + 1])))
    return p[-1]

print("Ordine di convergenza:", stima_ordine(roots_seq, it))


'''
d) si rappresenti in un grafico in scala semilogaritmica sulle y (comando semilogy eventualmente pre-
   ceduto da set(gca,’yscale’,’log’)) il vettore dei valori assoluti di tutte le approssimazioni calcolate dal
   procedimento iterativo (comprese tra |x(0)| e |α|), in funzione del numero di iterazioni compiute;
'''
plt.semilogy(range(it), np.abs(roots_seq), '-o')
plt.xlabel("Valori assoluti delle approssimazioni")
plt.ylabel("Numero di iterazioni")
plt.grid(True)
plt.show()

'''
e) si stabilisca se il metodo iterativo proposto al punto b) può convergere ad α quadraticamente anche
   partendo dall’estremo sinistro dell’intervallo, ossia da x(0) = 1, e si giustifichi la risposta.
   
   No, infatti la derivata prima di f in 1 vale -inf. Questo fa si che il primo iterato x_1 = x_0 = 1 e quindi 
   la condizione di arresto dell'errore relativo risulta essere verificata in quanto 
                                       abs(x_1 - x_0) = 0 < tolx * abs(x_1)
   Il metodo si interrompe pertanto alla prima iterazione, non producendo un'approssimazione accettabile dello zero.
'''
trigger = 1
my_root, roots_seq, it = newton_m(f, df, m, trigger, tol, tol, max_it)
print(my_root)
