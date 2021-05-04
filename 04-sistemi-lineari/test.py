#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 21:09:26 2021

@author: lucatassi
"""
import numpy as np
import Sistemi_lineari as sl
import funzioni_Sistemi_lineari as slp

L = np.matrix([[1, 0, 0], [6, 2, 0],  [1, 1, 5]])
bL = np.array([7, 4, 3])

U = np.matrix([[1, 1, 5], [0, -1, 5], [0, 0, -10]])
bU = np.array([3, 4, -30])

print("funzioni prof >")
print(slp.Lsolve(L, bL))
print(slp.Usolve(U, bU))
print("mie funzioni >")
print(sl.Lsolve(L, bL))
print(sl.Usolve(U, bU))