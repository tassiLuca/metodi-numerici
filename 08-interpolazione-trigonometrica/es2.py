#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 18:01:26 2021

@author: lucatassi
"""

import numpy as np
import math
from scipy.fft import fft
import matplotlib.pyplot as plt

def step(x):
    y = np.zeros_like(x)
    for k in range(len(x)):
        if x[k] < -1 or x[k] > k:
            y[k] = 1
    return y

n = int(input("Inserisci n: "))
dx = -3
sx = 3
nodes = np.linspace(sx, dx, n + 1, False)
ordinates = step(nodes)

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
    a[m + 1] = 0
    b[m + 1] = 0
else:
    #a[m + 1] = coeff(m + 1) / (n + 1)
    b[m + 1] = 0

pol = a0 * np.ones((100,))
points = np.linspace(sx, dx, 100)
l = 0
r = 2 * math.pi
points_mapped = (points - sx) * (r - l) / (dx - sx) + l

for k in range(1, m + 2):
    pol = pol + a[k] * np.cos(k * points_mapped) + b[k] * np.sin(k * points_mapped)
    plt.plot(nodes, ordinates, 'o', points, step(points), points, pol.real)
    plt.show()


