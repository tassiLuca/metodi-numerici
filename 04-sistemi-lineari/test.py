#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 21:09:26 2021

@author: lucatassi
"""
import numpy as np
import Sistemi_lineari as sl

A = np.matrix([[2, 1], [3, 3]])
b = np.array([1, 6])
P, L, U, flag = sl.LU_pivot(A)
print(sl.LUsolve(L, U, P, b))

