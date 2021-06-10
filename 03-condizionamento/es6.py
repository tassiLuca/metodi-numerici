#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esercizio 7
"""
import numpy as np
import matplotlib.pyplot as plt
import taylor

a = -10               
b = 10                
ncampio = 1000

xc = np.linspace(a, b, ncampio)
exp_es = np.exp(xc)  # esponenziale esatta

exp_app = np.zeros((ncampio,))
nt = np.zeros((ncampio,))     #indice n della serie

for i in range(ncampio):
    exp_app[i], nt[i] = taylor.esp_taylor_1(xc[i]);

err_rel = np.abs(exp_app - exp_es) / np.abs(exp_es)

plt.plot(xc, exp_app, 'b-', xc, exp_es, 'r--')
plt.title('Approssimazione esponenziale con serie di Taylor troncata')
plt.legend(['exp_app','exp_es'])
plt.show()

plt.plot(xc, err_rel)
plt.title("Errore relativo scala cartesiana")
plt.show()

plt.plot(xc, err_rel)
plt.yscale("log")
plt.title("Errore relativo scala semi-logaritimica")
plt.show()

plt.plot(xc, nt)
plt.title('Indice n')
plt.show()

'''
--------------------------------------------------------------------------
come migliorare andamento errore relativo
--------------------------------------------------------------------------
'''

for i in range(ncampio) :
    if xc[i] >= 0:
        exp_app[i], nt[i] = taylor.esp_taylor_1(xc[i]);
    else:
        exp_app[i], nt[i] = taylor.esp_taylor_2(xc[i]);


err_rel_2 = np.abs(exp_app-exp_es) / np.abs(exp_es)

plt.plot(xc, err_rel_2)
plt.yscale("log")
plt.title('Errore relativo Algoritmo Migliorato - scala semilogaritmica')
plt.show()

