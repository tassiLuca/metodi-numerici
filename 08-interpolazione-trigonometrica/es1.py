#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esercizio 1
------------
Si tratta di una base di polinomi ortogonali e si dimostra che se i coefficienti di partenza di cui calcoliamo 
fft sono ad energia finita (cioè la loro somma non va ad infinito), all'aumentare dei nodi converge ad f.
"""

from scipy.fft import fft
import numpy as np
import matplotlib.pyplot as plt
import math

choice = input("Scegli funzione: ")
functions = {
    '1': [ lambda x: np.sin(x) - 2 * np.sin(2 * x), -math.pi, math.pi ],
    '2': [ lambda x: np.sinh(x), -2, 2 ],
    '3': [ lambda x: np.abs(x), -1, 1 ],
    '4': [ lambda x: 1 / (1 + x**2), -5, 5 ]
}
f, sx, dx = functions.get(choice)

# Costruisco n + 1 punti equispaziati in [a, b). Si noti b è escluso.
n = int(input("Introduci il valore di n: "))
steps = (dx - sx) / (n + 1)
x = np.arange(sx, dx, steps)
# Mappo i punti dell'intervallo [a, b) in [l, r) = [0, 2 * pi]
l = 0
r = 2 * math.pi
xm = (x - sx) * (r - l) / (dx - sx) + l

if n % 2 == 0:
    m = n // 2
else:
    m = (n - 1) // 2

# Passo ad fft, la funzione f e lui la interpreta come fosse campionata tra 0 e 2 * pi con passo equidistante.
y = f(x)
c = fft(y)

a = np.zeros((m + 2, ))
b = np.zeros((m + 2, ))
a0 = c[0] / (n + 1)
a[1:m+1] =  2 * c[1:m+1].real / (n + 1)
b[1:m+1] = -2 * c[1:m+1].imag / (n + 1)

if n % 2 == 0:
    a[m + 1] = 0
    b[m + 1] = 0
else:
    a[m + 1] = c[m + 1] / (n + 1) 
    b[m + 1] = 0
    
pol = a0 * np.ones((100,))
points = np.linspace(sx, dx, 100)
points_mapped = (points - sx) * (r - l) / (dx - sx) + l

# quando ricostruiamo il polinomio è necessario che i punti in cui lo andiamo a valutare siano mappati tra 0 e 2 * pi.
for k in range(1, m + 2):
   pol = pol + a[k] * np.cos(k * points_mapped) + b[k] * np.sin(k * points_mapped)

plt.plot(points, pol, 'r', x , y , 'o', points, f(points), 'b')
plt.show()