#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esercizio 3
"""

import numpy as np
from scipy.fft import fft
import math
import matplotlib.pyplot as plt

'''
Il problema fornisce n+1 (xi,yi), i=0,.n n=9 misurazioni del flusso sanguigno attraverso una sezione 
dell’arteria carotide durante un battito cardiaco ad istanti di tempo equistanti con step 1/10
Gli istanti appartengono all'intervallo [0,1).
'''
n = 9
dx = 1
sx = 0
nodes = np.linspace(0, 1, n + 1, False)
ordinates = np.array([3.7, 13.5, 5.0, 4.6, 4.1, 4.5, 4.0, 3.8, 3.7, 3.7])

if n % 2 == 0:
    m = n // 2
else:
    m = (n - 1) // 2
    
coeff = fft(ordinates)
a = np.zeros((m + 2,), dtype = complex)
b = np.zeros((m + 2,), dtype = complex)
a0 = coeff[0] / (n + 1)
a[1:m+1] =  2 * coeff[1:m+1].real / (n + 1)
b[1:m+1] = -2 * coeff[1:m+1].imag / (n + 1)

if n % 2 == 0:
    a[m+1] = 0
    b[m+1] = 0
else:
    a[m+1] = coeff[m+1] / (n + 1)
    b[m+1] = 0
    

pol = a0 * np.ones((100,))
points = np.linspace(sx, dx, 100)
l = 0
r = 2 * math.pi
points_mapped = (points - sx) * (r - l) / (dx - sx) + l

for k in range(1, m+2):
    pol = pol + a[k] * np.cos(k * points_mapped) + b[k] * np.sin(k * points_mapped)
    
plt.plot(nodes, ordinates, 'o-', points, pol.real)
plt.show()
