#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esercizio 5
-----------
Per n = 5, 10, 15, 20 fornire un’approssimazione della costante di Lebesgue scegliendo x_1, x_2 , ..., x_n+1 
equispaziati in [−1, 1] oppure coincidenti con i nodi di Chebyshev x_i = cos(((2i + 1)π) / (2(n + 1))) i = 0, ..., n.

NOTA: Dalla definizione di Λ_n (Lebesgue) si vede facilmente che la scelta dei nodi dell’interpolazione x_i, con 
      i = 0, ..., n è fondamentale per il valore che può assumere la costante di Lebesgue.
      Anche se in entrambi i casi (nodi di Chebyshev ed equispaziati) per n → +∞ la costante di Lebesgue tende a 
      infinito, per i nodi di Chebyshev la crescita è logaritmica invece che esponenziale (vd. slide 40).
        
      In ogni caso, però, se vengono scelti gradi n troppo elevati, il problema dell’interpolazione polinomiale 
      risulta sensibile alle perturbazioni sui dati. Per interpolare un numero elevato di nodi, al fine di
      evitare l’utilizzo di un polinomio pn di grado n elevato, si consiglia l’utilizzo delle *spline interpolatorie*:
      si suddivide l'intervallo in più sotto-intervalli, scegliendo per ciascuno di questi un polinomio interpolatorio
      di grado n generalmente basso (in modo tale che non comporti i problemi visti), e raccordando due polinomi 
      successivi in modo tale vi sia continuità delle prime n-1 derivate.
"""

import numpy as np
import matplotlib.pyplot as plt
import interpolazione as intrpl

start = 5
stop = 25
step = 5
a = -1      # estremo sinistro dell'intervallo di interpolazione
b = 1       # estremo destro dell'intervallo di interpolazione
points_values = np.linspace(a, b, 200)

# Vettore contenente le costanti di Lebesgue per ogni n nel caso di nodi equispaziati e di Chebyshev
# la dimnesione è calcolata con (stop - start) / step in quanto bisogna allocare un array costituito 
# da tanti quante sono i nodi.
dim = int((stop - start) / step) 
equi_lebegue = np.zeros((4, 1))
cheby_lebegue = np.zeros((4, 1))

j = 0
for n in range(start, stop, step):
    equi_nodes = np.linspace(a, b, n + 1)
    cheby_nodes = intrpl.chebyshev_nodes(a, b, n)
    
    equi_lebesgue_acc = np.zeros((200, 1))
    cheby_lebesgue_acc = np.zeros((200, 1))
    for i in range(n + 1):
        equi_pol, flag = intrpl.plagrange(equi_nodes, i)
        # Accumulo i valori assoluti di tutti gli n + 1 polinomi di Lagrange sui nodi equispaziati
        equi_lebesgue_acc = equi_lebesgue_acc + np.abs(np.polyval(equi_pol, points_values))
        
        cheby_pol, flag = intrpl.plagrange(cheby_nodes, i)
        # Accumulo i valori assoluti di tutti gli n + 1 polinomi di Lagrange sui nodi di Chebyshev
        cheby_lebesgue_acc = cheby_lebesgue_acc + np.abs(np.polyval(cheby_pol, points_values))

    equi_lebegue[j] = np.max(equi_lebesgue_acc)
    cheby_lebegue[j] = np.max(cheby_lebesgue_acc)
    j = j + 1
  
print("Costante di Lebesgue con nodi equispaziati al variare di n: \n", equi_lebegue)
plt.plot(range(start, stop, step), equi_lebegue, '*-')
print("Costante di Lebesgue con nodi di Chebyshev al variare di n: \n", cheby_lebegue)
plt.plot(range(start, stop, step), cheby_lebegue, '*-')
plt.legend(['Nodi equispaziati', 'Nodi di Chebyshev'])
plt.title("Costante di Lebesgue")
plt.show()