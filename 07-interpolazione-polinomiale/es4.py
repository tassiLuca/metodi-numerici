#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esercizio 4
-----------
Scrivere uno script che calcoli il polinomio interpolante un insieme di punti P_i = (x_i , y_i ), i = 0, ..., n, 
nella forma di Lagrange con xi scelti dall’utente come:
    • punti equidistanti in un intervallo [a, b],
    • punti definiti dai nodi di Chebyshev nell’intervallo [a, b] e y_i = f (x_i) ottenuti dalla valutazione nei 
      punti xi di una funzione test f : [a, b] → R. 
      
Testare lo script sulle funzioni:
    • f(x) = sin(x) − 2*sin(2x), x ∈ [−π, π],
    • f(x) = sinh(x), x ∈ [−2, 2],
    • f(x) = |x|, x ∈ [−1, 1],
    • f(x) = 1/(1 + x2), x ∈ [−5, 5] (funzione di Runge).
    
Calcolare l’errore di interpolazione r(x) = f(x) − p(x), tra la funzione test f(x) e il polinomio di interpolazione 
p(x). Visualizzare il grafico di f(x) e p(x), ed il grafico di |r(x)|. Cosa si osserva? Cosa accade all’aumentare del
grado n di p(x)? (Si costruisca una tabella che riporti i valori di ||r(x)||∞ al variare di n).

NOTE: Le prime due funzioni soddisfano le hp. del teorema pg 26: sono di classe C infinito con derivate equilimitate
      in [a, b]. Ciò garantisce che la successione dei polinomi interpolanti converga ad f uniformemente in [a, b]
      su una qualunque distribuzione di nodi. Le funzioni successive, invece, non soddisfando le hp. del teorema,
      mostrano come, se si considerano nodi di interpolazione equispaziati in [a, b], all'aumentare del numero di 
      punti di interpolazione (e quindi al crescere del grado n del polinomio) si presentano fitte oscillazioni agli
      estremi dell'intervallo, tipiche dei polinomi di grado elevato. Per ovviare a questo problema si considerano dei
      nodi distribuiti in modo più fitto vicino agli estremi dell’intervallo: i nodi di Chebyshev (vedi slides 
      da pg 30).
      
      E l'ultima funzione? Slide 36.
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import interpolazione as intrpl

def _chebyshev_nodes(a, b, n):
    '''
    Calcola i nodi di Chebyshev.

    Parametri
    ----------
    a: estremo sinistro dell'intervallo
    b: estremo destro dell'intervallo
    n: numero di nodi
        
    Valori di ritorno
    -------
    x: vettore con i nodi di Chebyshev.

    '''
    t1 = (a + b) / 2
    t2 = (a - b) / 2
    x = np.zeros((n+1, ))
    for i in range (n + 1):
        x[i] = t1 + t2 * np.cos(((2 * i + 1) * math.pi) / (2 * (n + 1)))
    return x

# Seleziono la funzione
function_choice = input("Scegli funzione: ")
functions = {
    '1': [lambda x: np.sin(x) - 2 * np.sin(2*x), - math.pi, math.pi],
    '2': [lambda x: np.sinh(x), -2, 2],
    '3': [lambda x: np.abs(x), -1, 1],
    '4': [lambda x: 1 / (1 + x**2), -5, 5],
    '5': [lambda x: np.sqrt(np.abs(x)), -1, 1]
}
function, a, b = functions.get(function_choice)

# Seleziono il tipo di nodi: equispaziati o di Chebyshev
nodes_choice = input("Scegli tipo punti di interpolazione --> (1) equidistanti; (2) Chebyshev: ")
n = int(input("Grado del polinomio: "))
nodes_types = {
    '1': np.linspace(a, b, n + 1),
    '2': _chebyshev_nodes(a, b, n)
}
nodes = nodes_types.get(nodes_choice)
nodes_values = function(nodes)

# Punti di valutazione per l'interpolante
points_values = np.linspace(a, b, 200)

# Calcolo il polinomio interpolante
pol = intrpl.lagrange_interp(nodes, nodes_values, points_values)
# Calcolo l'errore di interpolazione e la sua norma ∞
error = np.abs(function(points_values) - pol)
error_norm = np.linalg.norm(error, np.inf)

# Grafico i risultati
plt.plot(nodes, nodes_values, '*', points_values, pol, '--', points_values, function(points_values))
plt.legend(['Funzione', 'Interpolante di Lagrange', 'Punti di interpolazioni'])
plt.show()
plt.plot(points_values, error)
plt.legend(['Errore di interpolazione'])
plt.show()

