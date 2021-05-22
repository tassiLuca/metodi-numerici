#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esercizio 3
-----------
La temperatura T in prossimità del suolo subisce una variazione dipendente
dalla latitudine L secondo la seguente tabella:
    
    L | -55 -45 -35 -25 -15 -5 5 15 25 35 45 55 65
    T | 3.7 3.7 3.52 3.27 3.2 3.15 3.15 3.25 3.47 3.52 3.65 3.67 3.52
    
Si vuole costruire un modello che descriva la legge T = T (L) anche per latitudini non misurate. 
A tal fine si scriva uno script che fornisca la variazione di temperatura alle latitudini L = ±42 utilizzando 
il polinomio interpolante. Visualizzare in un grafico i dati assegnati, il polinomio interpolante e
le stime di T ottenute per L = ±42.
"""

import numpy as np
import matplotlib.pyplot as plt
import interpolazione as intrpl

temperature = np.array([-55, -45, -35, -25, -15, -5, 5, 15, 25, 25, 45, 55, 65])
latitude = np.array([3.7, 3.7, 3.52, 3.27, 3.2, 3.15, 3.15, 3.25, 3.47, 3.52, 3.65, 3.67, 3.52])

points_values = np.linspace(np.min(temperature), np.max(temperature), 200)

pol = intrpl.lagrange_interp(temperature, latitude, points_values)

l1 = 42
l2 = -42
t1 = intrpl.lagrange_interp(temperature, latitude, np.array([l1])) 
t2 = intrpl.lagrange_interp(temperature, latitude, np.array([l2])) 

plt.plot(points_values, pol, '--', temperature, latitude, '*', l1, t1, 'og', l2, t2, 'og')
plt.legend(['Interpolante di Lagrange', 'Punti di interpolazione', 'Strima temperatura 42°', 'Strima temperatura -42°'])
plt.show()
