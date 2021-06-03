#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esercizio 1: sperimentazione numerica.

Confrontare i metodi sopra implementati nei casi seguenti:
• f (x) = exp(−x) − (x + 1) in [−1, 2] con x0 = −0.5, x−1 = −0.3, tolx = 1.e − 12, tolf = 1.e − 12;
• f (x) = log2 (x + 3) − 2 in [−1, 2] con x0 = −0.5, x−1 = 0.5, tolx = 1.e − 12, tolf = 1.e − 12;
• f (x) = sqrt(x) − x^2 / 4 in [1, 3] con x0 = 1.8, x−1 = 1.5, tolx = 1.e − 12, tolf = 1.e − 12.

Mostrare in un grafico in scala semilogaritmica sulle ordinate (comando semilogy) l’andamento di 
ek = |xk − α|, k = 1, ..., n_iterazioni, sapendo che α = 0, 1, 24/3 nei tre casi.
Calcolare infine, a partire dai valori di {x_k} con k sufficientemente grande, la stima dell’ordine 
di convergenza p come e si confronti il valore ottenuto con quello atteso.
"""

import my_zeri as roots
import matplotlib.pyplot as plt
import sympy as sym
import numpy as np
from sympy.utilities.lambdify import lambdify

# Variabile indipendente x
x = sym.symbols('x')

# Espressioni simboliche delle tre funzioni che si vogliono confrontare
f1 = sym.exp(-x) - (x + 1)
f2 = sym.log(x + 3, 2) - 2
f3 = sym.sqrt(x) - (x**2) / 4

# Raccolta informazioni di ciascuna funzione 
# - espressione simbolica funzione
# - zero esatto della funzione
# - estremo sinistro
# - estremo destro
# - x_0
# - x_-1
functions = ([f1, 0, -1, 2, -0.5, -0.3], 
             [f2, 1, -1, 2, -0.5, 0.5], 
             [f3, 2**(4 / 3), 1, 3, 1.8, 1.5])
tol_x = 1e-12
tol_f = 1e-12
n_max = 500

for function in functions:
    # spacchetto la lista
    f_name, alpha, a, b, x_0, x_m1 = function 
    # trasformo in espressione numerica la funzione e la sua derivata
    f  = lambdify(x, f_name, np)
    df = lambdify(x, sym.diff(f_name, x, 1), np)
    
    # Grafico la funzione f sull'intervallo [a, b]
    x_axis = np.linspace(a, b, 100)
    plt.subplot(1, 2, 1)
    plt.title("f(x) = " + str(f_name))
    plt.plot(x_axis, 0 * x_axis, 'black', x_axis, f(x_axis), 'r-')

    # Applico i metodi per trovare gli zeri di funzione approssimati
    alpha_bisec, approximations_bisec, iterations_bisec = roots.bisezione(f, a, b, tol_x)
    alpha_chord, approximations_chord, iterations_chord = roots.corde(f, df, x_0, tol_x, tol_f, n_max)
    alpha_falsi, approximations_falsi, iterations_falsi = roots.regula_falsi(f, a, b, tol_x, n_max)
    alpha_newton, approximations_newton, iterations_newton = roots.newton(f, df, x_0, tol_x, tol_f, n_max)
    alpha_sec, approximations_sec, iterations_sec = roots.secanti(f, x_m1, x_0, tol_x, tol_f, n_max)

    # Calcolo gli errori
    err_bisec = np.abs(np.array(approximations_bisec)-alpha)
    err_chord = np.abs(np.array(approximations_chord)-alpha)
    err_falsi = np.abs(np.array(approximations_falsi)-alpha)
    err_newton = np.abs(np.array(approximations_newton)-alpha)
    err_sec = np.abs(np.array(approximations_sec)-alpha)
    
    # Grafico in scala logaritmica l'andamento dell'errore ad ogni passo per i tre metodi
    plt.subplot(1, 2, 2)
    plt.semilogy(range(iterations_bisec), err_bisec, 'go-', 
                 range(iterations_chord), err_chord, 'mo-', 
                 range(iterations_falsi), err_falsi, 'bo-', 
                 range(iterations_newton), err_newton, 'co-', 
                 range(iterations_sec), err_sec, 'ro-')
    plt.legend(['Bisezione', 'Corde', 'Regula Falsi', 'Newton', 'Secanti'])
    plt.show()

    # Calcolo ordine di convergenza di ogni metodo
    convergence_bisec = roots.stima_ordine(approximations_bisec, iterations_bisec)
    convergence_chord = roots.stima_ordine(approximations_chord, iterations_chord)
    convergence_falsi = roots.stima_ordine(approximations_falsi, iterations_falsi)
    convergence_newton = roots.stima_ordine(approximations_newton, iterations_newton)
    convergence_sec = roots.stima_ordine(approximations_sec, iterations_sec)

    print("Funzione:", f_name)
    print("Bisezione > alpha={:f}, iterazioni={:d}, ordine di convergenza={:e}".format(alpha_bisec, iterations_bisec, convergence_bisec))
    print("Corde > alpha={:f}, iterazioni={:d}, ordine di convergenza {:e}".format(alpha_chord, iterations_chord, convergence_chord))
    print("Falsi > alpha={:f}, iterazioni={:d}, ordine di convergenza {:e}".format(alpha_falsi, iterations_falsi, convergence_falsi))
    print("Newton > alpha={:f}, iterazioni={:d}, ordine di convergenza {:e}".format(alpha_newton, iterations_newton, convergence_newton))
    print("Secanti > alpha={:f}, iterazioni={:d}, ordine di convergenza {:e}".format(alpha_sec, iterations_sec, convergence_sec))
    print("--------------------------------------------------------------")
    