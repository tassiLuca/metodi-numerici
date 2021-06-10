#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esercizio 8
"""

import numpy as np
import math
import matplotlib.pyplot as plt

k = np.arange(0, -21, -1)
h = 10.0**k

des = math.cos(1)   # derivata esatta

x = 1
rai = (np.sin(x + h) - np.sin(x)) / h      # rapporto incrementale

err_rel = np.abs(rai - des) / np.abs(des)  # errore relativo

plt.plot(h, err_rel, 'b-', h, h, 'r:')
plt.xscale("log")
plt.yscale("log")
plt.legend(['Errore relativo', 'Incremento'])