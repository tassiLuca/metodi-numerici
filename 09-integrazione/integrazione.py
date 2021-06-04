#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrazione
"""

import numpy as np

def trap_comp(fname, a, b, n):
    '''
    Calcola la formula dei trapezi composita.

    Parametri
    ----------
    fname : funzione di cui calcolare l'integrale definito
    a : estremo sinistro dell'intervallo di integrazione
    b : estremo destro dell'intervallo di integrazione
    n : numero di sottointervalli in cui suddividere l'intervallo [a, b]

    Valori di ritorno
    -------
    result : il valore dell'integrale approssimato con formula dei trapezi composita
    '''
    
    h = (b - a) / n     # differenza tra due sotto-intervalli successivi
    # Creo i nodi in base al passo dei sotto-intervalli. 
    # Nota: come valore di stop metto b + h in quanto i valori sono generati 
    # entro l'intervallo [start, stop) con stop escluso.
    nodes = np.arange(a, b + h, h)
    f = fname(nodes)
    result = h / 2 * (f[0] + 2 * np.sum(f[1:n]) + f[n])
    return result
    
def simpson_comp(fname, a, b, n):
    '''
    Calcola la formula di Simpson composita.

    Parametri
    ----------
    fname : funzione di cui calcolare l'integrale definito
    a : estremo sinistro dell'intervallo di integrazione
    b : estremo destro dell'intervallo di integrazione
    n : numero di sottointervalli in cui suddividere l'intervallo [a, b]

    Valori di ritorno
    -------
    result : il valore dell'integrale approssimato con formula di Simpson composita
    '''
    
    h = (b - a) / (2 * n)     # differenza tra due sotto-intervalli successivi
    # Creo i nodi in base al passo dei sotto-intervalli. 
    # Nota: come valore di stop metto b + h in quanto i valori sono generati 
    # entro l'intervallo [start, stop) con stop escluso.
    nodes = np.arange(a, b + h, h)
    f = fname(nodes)
    result = h / 3 * (f[0] + 2 * np.sum(f[2:2*n+2:2]) + 4 * np.sum(f[1:2*n+2:2]) + f[2*n])
    return result


def trap_toll(fname, a, b, tol):
    '''
    Algoritmo quadratura automatico con Trapezi.

    Parametri
    ----------
    fname : funzione di cui calcolare l'integrale definito
    a : estremo sinistro dell'intervallo di integrazione
    b : estremo destro dell'intervallo di integrazione
    tol : tolleranza

    Valori di ritorno
    -------
    integral : valore dell'integrale approssimato
    steps : numero di sottointervalli usati per ridurre l'errore ad essere inferiore di tol
    '''
    max_steps = 2048
    err = 1
    
    steps = 1
    integral = trap_comp(fname, a, b, steps)    
    while steps <= max_steps and err > tol:
        steps = 2 * steps
        integral_double_steps = trap_comp(fname, a, b, steps)
        err = abs(integral - integral_double_steps) / 3
        integral = integral_double_steps
    
    if steps > max_steps:
        print("Reached max number of sub-intervals")
        steps = 0
        integral = []
        
    return integral, steps
    
def simp_toll(fname, a, b, tol):
    '''
    Algoritmo quadratura automatico con Simpson.

    Parametri
    ----------
    fname : funzione di cui calcolare l'integrale definito
    a : estremo sinistro dell'intervallo di integrazione
    b : estremo destro dell'intervallo di integrazione
    tol : tolleranza

    Valori di ritorno
    -------
    integral : valore dell'integrale approssimato
    steps : numero di sottointervalli usati per ridurre l'errore ad essere inferiore di tol
    '''
    max_steps = 2048
    err = 1
    
    steps = 1
    integral = simpson_comp(fname, a, b, steps)    
    while steps <= max_steps and err > tol:
        steps = 2 * steps
        integral_double_steps = simpson_comp(fname, a, b, steps)
        err = abs(integral - integral_double_steps) / 15
        integral = integral_double_steps
    
    if steps > max_steps:
        print("Reached max number of sub-intervals")
        steps = 0
        integral = []
        
    return integral, steps