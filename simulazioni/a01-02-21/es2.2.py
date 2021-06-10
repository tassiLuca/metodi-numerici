#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Appello 1 Febbraio 2021 - Esercizio 2
-------------------------------------
INTEGRAZIONE ED INTERPOLAZIONE
"""

import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift
import math
import matplotlib.pyplot as plt

T  = 2
fc = 170
N = fc * T

x = lambda x : 4 * np.sin(2 * math.pi * 15 * x) + 3 * np.sin(2 * math.pi * 40 * x) + 2 * np.sin(2 * math.pi * 60 * x)
noise = lambda x : 2 * np.sin(2 * math.pi * 80 * x)

t = np.linspace(0, T, N)
xp = x(t) + noise(t)

plt.plot(t, x(t))
plt.title("Segnale \"pulito\"")
plt.grid(True)
plt.show()

plt.plot(t, xp)
plt.title("Segnale rumoroso")
plt.grid(True)
plt.show()

delta_u = 1 / T
freq = np.arange(-fc/2, fc/2, delta_u)

c = fftshift(fft(xp))
plt.plot(freq, np.abs(c))
plt.show()

ind = np.abs(freq) > 60
c[ind] = 0
plt.plot(freq, np.abs(c))
plt.show()

rec = ifft(ifftshift(c))
plt.plot(t, rec)
plt.title("Segnale ri-\"pulito\"")
plt.grid(True)
plt.show()