#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Appello del 15 Gennaio 2021 - Esercizio 2
-----------------------------------------
SISTEMI LINEARI
"""

import numpy as np
import scipy.linalg as spl
import matplotlib.pyplot as plt

'''
a) Scrivere la function LU_nopivot che, presa in input una matrice A, restituisce in output le matrici
   L e U associate al metodo di eliminazione gaussiana senza pivoting.
'''

def LU_nopivot(A):
    n, m = A.shape 
    if n != m:
        print("ERRORE: matrice non quadrate")
        return [], [], 1
    
    U = A.copy()
    for k in range(n):
        if U[k, k] == 0:
            print("ERROR: elemento pivotale nullo")
            return [], [], 1
        
        U[k+1:n, k] = U[k+1:n, k] / U[k, k]
        U[k+1:n, k+1:n] = U[k+1:n, k+1:n] - np.dot(U[k+1:n, k], U[k, k+1:n])
        
    L = np.tril(U, -1) + np.eye(n)
    U = np.triu(U)
    return L, U, 0

'''
a) Scrivere la function che implementa il metodo delle sostituzioni all’indietro per risolvere un sistema
   lineare con matrice dei coefficienti triangolare superiore.
'''

def backward(A, b):
    n, m = A.shape
    if n != m:
        print("ERRORE: matrice non quadrata")
        return [], 1
    elif np.all(np.diag(A)) != True:
        print("ERRORE: det(A) = 0")
        return [], 1
    
    x = np.zeros(n)
    for i in range(n):
        s = np.dot(A[i, i+1:n], x[i+1:n])
        x[i] = (b[i] - s) / A[i, i]
        
    return x, 0

'''
c) Scrivere la function che implementa il metodo delle sostituzioni in avanti per risolvere un sistema
   lineare con matrice dei coefficienti triangolare inferiore.
'''

def forward(A, b):
    n, m = A.shape
    if n != m:
        print("ERRORE: matrice non quadrata")
        return [], 1
    elif np.all(np.diag(A)) != True:
        print("ERRORE: det(A) = 0")
        return [], 1
    
    x = np.zeros(n)
    for i in range(n):
        s = np.dot(A[i, :i], x[:i])
        x[i] = (b[i] - s) / A[i, i]
        
    return x, 0

'''
d) Scrivere lo script in cui si sfruttano la fattorizzazione LU di A e le function implementate in b) e c) per 
   calcolare le soluzioni dei sistemi lineari
                           A.T x = b     e     A**2 x = c
   con A = pascal(n), b = A.T ∗ ones(n, 1), c = A**2 ∗ ones(n, 1) per tutti i valori di n tali che 5 <= n <= 10.
   
   NOTE: (1) La matrice di Pascal è una particolare matrice che contiene al suo interno un triangolo di Tartaglia,
             ossia i coefficienti binomiali. Ad esempio la matrice di pascal di ordine 6:
                
                                       | 1	1	1	1	1	1	1   |
                                       | 1	2	3	4	5	6	7   |
                                       | 1	3	6	10	15	21	28  |
                                A(6) = | 1	4	10	20	35	56	84  |
                                       | 1	5	15	35	70	126	210 |
                                       | 1	6	21	56	126	252	462 |
                                       | 1	7	28	84	210	462	924 |
                                       
             Si osservi che (a) A è simmetrica (i.e. A.T = A) e (b) è mal condizionata per n grandi (vedi grafici). 
             In generale vale:
                                          K_2(A.T A) = K_2(A)**2                                
             (slide norme - pg 14).
             Per (a) A**2 = A.T A => K_2(A**2) = K_2(A)**2. Ergo la matrice A**2 ha indice di condizionamento 
             peggiore di quello di A.
       
         (2) Prestare attenzione alla traccia: per il calcolo del sistema A**2 x = c NON si deve determinare la 
             fattorizzazione di A^2, bensì si deve risolvere sfruttando solo la fattorizzazioni LU di A.
            
         (3) Risolvo A.T x = b <=> (L U).T x = b <=> U.T L.T x = b 
             Si ricorda infatti che il trasposto di un prodotto è il prodotto dei trasposti, con ordine inverso 
             (il prodotto matriciale non gode della proprietà commutativa!)
            
         (4) Risolvo A**2 x = b <=> A A x = b <=> L U L U x = b. Per risolvere L U L U x = b qui utilizzo la funzione
             LULU_solve che estende il processo di risoluzione di LU_solve:
                IV)  U x = y3
                III) L y3 = y2
                II)  U y2 = y1
                I)   L y1 = c
         
e) Relativamente alla risoluzione del sistema lineare con matrice A2, il procedimento indicato al punto
   d) ha qualche vantaggio? Motivare la risposta.
    
   Conviene utilizzare la strategia proposta per la soluzione del secondo sistema e NON calcolare la fattorizzazione
   LU di A**2, perchè A è mal condizionata e il sistema lineare con matrice A**2 ha un indice di condizionamento 
   pari al quadrato di A (vd. note p.to (a)), pertanto è opportuno risolvere i 4 sistemi lineari con matrici 
   triangolare che hanno sicuramente un indice di condizionamento molto inferiore di A**2.
'''

def LU_solve(L, U, b):
    y, flag = forward(L, b)
    if flag == 0:
        x, flag = backward(U, y)
    
    return x, flag    

def LULU_solve(L, U, c):
    '''
    Risoluzione in "cascata" di L U L U x = c
    '''
    y1, flag = forward(L, c)
    y2, flag = backward(U, y1)
    y3, flag = forward(L, y2)
    x,  flag = backward(U, y3)
    return x

start = 5
stop = 10
step = 1
condA = []
condAA = []

for n in range(start, stop + step, step):
    A = spl.pascal(n)
    # calcolo indice di condizionamento di A e di A^2
    condA.append(spl.norm(A) * spl.norm(spl.inv(A)))
    condAA.append(spl.norm(np.dot(A, A)) * spl.norm(spl.inv(np.dot(A, A))))

    b = np.dot(A.T, np.ones((n, 1)))
    c = np.dot(np.dot(A, A), np.ones((n, 1)))
    
    # fattorizzazione di A
    L, U, flag = LU_nopivot(A)

    # soluzione A.T x = b
    x1 = LU_solve(U.T, L.T, b)
    # soluzione A^2 x = c
    x2 = LULU_solve(L, U, c)
    
    # stampa soluzioni
    print("========================= n =", n, "=========================")
    print("Soluzione del sistema lineare A.T x = b \n", x1)
    print("\n Soluzione del sistema lineare A^2 x = b \n", x2)
    
# grafico indice di condizionamento di A
x_axis = np.arange(start, stop + step, step)
plt.semilogy(x_axis, condA, x_axis, condAA)
plt.xlabel("n")
plt.ylabel("Indice di condizionamento K")
plt.legend(["K(A)", "K(A^2)"])
plt.title("Indice di condizionamento della matrice di pascal(n) \n al variare di n (in scala logaritmica)")
plt.grid(True)
plt.show()
