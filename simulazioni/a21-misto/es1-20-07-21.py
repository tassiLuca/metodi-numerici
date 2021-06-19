#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Appello del 21 Luglio 2020 - Esercizio 1
----------------------------------------
ZERI DI FUNZIONE.
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from sympy.utilities import lambdify
from scipy.optimize import fsolve

trigger = 0
tolx = 1.e-7
max_it = 500
a = -1
b = 1

f = lambda x:  np.tan(3 / 2 * x) - 2 * np.cos(x) - x * (7 - x)

sol = fsolve(f, trigger)

print("Radice di f (fsolve) = ", sol[0])
x_axis = np.linspace(a, b)
plt.plot(x_axis, f(x_axis), sol, f(sol), '*')
plt.grid(True)
plt.show()

'''
a) La funzione di iterazione g(x) = tan(3 / 2 * x) - 2 cos(x) - x * (6 - x) genera un metodo di punto fisso
   convergente? Motivare la risposta.
    
   Per il teorema di convergenza locale di Ostrowski:
   -------------------------------------------------
   Sia α un punto fisso di g, funzione di classe C^1 in un intorno di α di raggio ρ > 0. Se |g'(x)| < 1, per ogni x 
   appartenente all'intorno di α di raggio ρ, allora comunque scelto il valore di innesco appartente all'intorno di α
   di raggio ρ, la successione delle iterate generate da g converge ad α.
   
   Nel caso in esame, dal grafico di g1'(x) si vede chiaramente che non soddisfa le hp. del teorema precedentemente
   enunciato. Pertando il metodo di iterazione non converge ad alpha.
   
(Slide 66)
'''
# Funzione simboliche
x = sym.symbols('x')
g1_name = sym.tan(3 / 2 * x) - x * (7 - x) - 2 * sym.cos(x)
dg1_name = sym.diff(g1_name, x, 1)

# Funzioni numeriche
g1  = lambdify(x, g1_name, np)
dg1 = lambdify(x, dg1_name, np)

plt.plot(x_axis, x_axis, x_axis, g1(x_axis), sol, g1(sol), 'o')
plt.legend(["Bisettrice y = x", "g1(x)", "Radice di f(x)"])
plt.grid(True)
plt.show()

rho = 0.5
x_axis = np.linspace(sol - rho, sol + rho, 100)
plt.plot(x_axis, dg1(x_axis), sol, 0, 'o')
plt.plot([-1, 1], [1, 1], '--')     # retta y = 1
plt.plot([-1, 1], [-1, -1], '--')   # retta y = -1
plt.grid(True)
plt.show()
   
'''
b) Definire una funzione d’iterazione g (diversa dalla precedente e facilmente individuabile dal problema)
   in grado di generare un metodo di punto fisso convergente.
   
   Si deduce facilmente dal grafico della derivata prima di g2 (dg2) che |dg2(x)| < 1 in un intorno della radice.
   Questo garantisce la convergenza del metodo di punto fisso.
'''

g2_name  = (sym.tan(3 / 2 * x) - 2 * sym.cos(x) + x ** 2) / 7
dg2_name = sym.diff(g2_name, x, 1)

# Funzioni numeriche
g2  = lambdify(x, g2_name, np)
dg2 = lambdify(x, dg2_name, np)

rho = 0.5
x_axis = np.linspace(sol - rho, sol + rho, 100)
plt.plot(x_axis, dg2(x_axis), sol, 0, 'o')
plt.plot([-1, 1], [1, 1], '--')     # retta y = 1
plt.plot([-1, 1], [-1, -1], '--')   # retta y = -1
plt.grid(True)
plt.show()

'''
c) Scrivere la function punto_fisso che calcola la soluzione dell'equazione non lineare
'''
def punto_fisso(gname, trigger, tolx, max_it):
    approx = []
    it = 0
    prv = trigger
    
    while True:
        it += 1
        nxt = gname(prv)
        approx.append(nxt)
        
        if it >= max_it or abs(nxt - prv) < tolx * abs(nxt):
            break
        else:
            prv = nxt
            
    return nxt, approx, it

'''
d) Dopo aver inizializzato x0 = 0, tolx = 1.e − 7, nmax = 500, calcola la soluzione sol e plotta il vettore delle 
   approssimazioni verso il vettore 1 : it.
'''

g2_sol, g2_approx, g2_it = punto_fisso(g2, trigger, tolx, max_it)
print("Radice di f (con g2) = ", g2_sol, "raggiunto con ", g2_it, " iterazioni")

x_axis = np.linspace(a, b)
plt.plot(np.arange(0, g2_it, 1), g2_approx)
plt.show()

'''
e) Determinare l’ordine di convergenza del metodo di punto fisso implementato.
    
   Per i metodi di terazione funzionale, vale il seguente:
   Sia α punto fisso di g, funzione di classe C^p con p >= 2 intero. Se per un punto x_0 la successione degli iterati
   è convergente e se:
       
                   g'(α) = g''(α) = ... = g^{p-1}(α) = 0 e g^p != 0
                   
   allora il metodo ha ordine di convergenza p e risulta che la costante di convergenza 
                                   
                                               C = |g^p(α)| / p!
        
(slide 70)

    Nel caso in esame è evidente dal grafico della derivata prima di g2 che dg2(α) != 0, per cui il metodo ha ordine 
    di convergenza 1 e, conseguentemente, la costante asintotica di convergenza è data da |g'(α)|.
'''

def stima_ordine(xk, it):
    p = []
    
    for k in range(it - 3):
        p.append(np.log(abs(xk[k + 2] - xk[k + 3]) / abs(xk[k + 1] - xk[k + 2])) / 
                 np.log(abs(xk[k + 1] - xk[k + 2]) / abs(xk[k] - xk[k + 1])))
        
    return p[-1]

g2_ordine = stima_ordine(g2_approx, g2_it)
print("\t Ordine di convergenza: ", g2_ordine)
print("\t Costante asintotica di convergenza: ", dg2(g2_sol))