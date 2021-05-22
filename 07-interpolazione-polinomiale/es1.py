#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esercizio 1
-----------
Si disegnino i grafici dei polinomi di Lagrange associati ai nodi {0, 1/4, 1/2, 3/4, 1}
e ai nodi {−1, −0.7, 0.5, 2}.
"""

import numpy as np
import matplotlib.pyplot as plt
import interpolazione as intrpl

nodes = [np.arange(0, 1.1, 1/4), np.array([-1, -0.7, 0.5, 2])]

for node in nodes:
    n = node.size
    p = interpolazi

