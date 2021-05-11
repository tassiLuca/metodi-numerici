#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
04 - Esercizio 1
-----------

Al variare di n = 100, ..., 200 costruire la matrice:
            A[i, j] = sqrt(2 / (n + 1)) * sin((i + 1) * (j - 1) * pi / (n + 1))
i, j = 0, ..., n - 1 di dimensione n * n, definire la soluzione x_esatta = (1:n)^T e calcolare il termine
noto come b = A * x_esatta. Utilizzando le diverse function implementate calcolare x_nopivot, x_pivot, e 
confrontarli con x_esatta usando i grafici in scala semilogaritmica dell'errore relativo al variare di n. 
Confrontare inoltre la soluzione ottenuta con scipy.linalg.solve(A, b) con x_esatta.

Inoltre, per verificare che il numero di operazioni sia proporzionale a n^3, con n dimensione del sistema, 
riportare in un grafico in scala semilogaritmica il tempo impiegato a risolvere il sistema al variare di n. 
Il grafico dovrebbe essere *asintoticamente* una retta con pendenza 3. Per verificarne la pendenza, 
disegnare contemporaneamente anche la curva n^3 e controllare che siano parallele. La strategia di pivot
non dovrebbe influenzare il risultato.
'''

import numpy as np
import numpy.linalg as npl       # modulo per il calcolo delle norme
import scipy.linalg as spl       # modulo per la risoluzione di sistemi lineari tramite scipy.linalg.solve(A, b)
import math                      # modulo per funzioni matematiche (sqrt, sin, ...)
import Sistemi_lineari as sl     # file con i metodi numerici diretti per la risoluzione dei sistemi lineari
import matplotlib.pyplot as plt  # per i grafici --> wrapper attorno a plot che rende gli assi logaritmici
import time                      # per calcolare il tempo di esecuzione di una porzione di codice

start = 100
stop = 200

errore_rel_nopivot = []
tempo_nopivot = []
errore_rel_pivot = []
tempo_pivot = []
errore_rel_solve = []
tempo_solve = []

for n in range(start, stop):
    print("In esecuzione > n =", n)
    A = np.empty((n, n), dtype = float)
    for i in range(n):
        for j in range(n):
            A[i, j] = math.sqrt(2 / (n + 1)) * math.sin((i + 1) * (j + 1) * math.pi / (n + 1))
            
    # Siccome se imponessi un termine noto non avrei modo di calcolare la soluzione esatta, 
    # ma solo una soluzione approssimata ottenuta mediante i metodi diretti studiati, impongo 
    # che la soluzione del sistema sia x_esatta e calcolo conseguentemente il vettore dei 
    # termini noti b come il prodotto A * x_esatta.
    x_esatta = np.arange(1, n + 1).reshape((n, 1))
    b = np.dot(A, x_esatta)
    
    # Calcolo la soluzione con il metodo di Gauss senza pivot
    t1 = time.perf_counter()
    P, L, U, flag = sl.LU_nopivot(A)
    x, flag = sl.LUsolve(L, U, P, b)
    t2 = time.perf_counter()
    errore_rel_nopivot.append(npl.norm(x - x_esatta, 1) / npl.norm(x_esatta, 1))
    tempo_nopivot.append(t2 - t1)
    
    # Calcolo la soluzione con il metodo di Gauss con pivot parziale
    t1 = time.perf_counter()
    P, L, U, flag = sl.LU_pivot(A)
    x, flag = sl.LUsolve(L, U, P, b)
    t2 = time.perf_counter()
    errore_rel_pivot.append(npl.norm(x - x_esatta, 1) / npl.norm(x_esatta, 1))
    tempo_pivot.append(t2 - t1)
    
    # Calcolo la soluzione con il metodo di scipy
    t1 = time.perf_counter()
    x = spl.solve(A, b)
    t2 = time.perf_counter()
    errore_rel_solve.append(npl.norm(x - x_esatta, 1) / npl.norm(x_esatta, 1))
    tempo_solve.append(t2 - t1)

print("Finito!")

plt.semilogy(range(start, stop), errore_rel_nopivot, 
             range(start, stop), errore_rel_pivot, 
             range(start, stop), errore_rel_solve)
plt.legend(['No pivot', 'Pivot', 'Solve'])
plt.show()

curva = np.arange(start, stop) ** 3
plt.semilogy(range(start, stop), tempo_nopivot, 
             range(start, stop), tempo_pivot, 
             range(start, stop), tempo_solve, 
             range(start, stop), curva)
plt.legend(['No pivot', 'Pivot', 'Solve', 'n^3'])
plt.show()