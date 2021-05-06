# -*- coding: utf-8 -*-
"""
04 - File per la soluzione di sistemi Lineari
"""
import numpy as np

# -------------------------------------------------------------------------------------------------------
# Metodi di sostituzione per la risoluzione di sistemi lineari con matrice dei coefficienti triangolare.
# -------------------------------------------------------------------------------------------------------

def _solve_applicability(L, m, n):
    '''
    Funzione per check di applicabilità dei metodi di sostituzione in avanti e 
    all'indietro per  la risoluzione di sistemi lineari con matrice triangolare.
    
    Parametri
    ----------
    L :     Matrice triangolare inferiore.
    m, n:   Numero di righe e colonne della matrice L.

    Valori di ritorno.
    -------
    flag :  Booleano che è 0 se sono soddisfatti entrambi i test di 
            applicabilità, 1 altrimenti.
    '''
    
    # Test dimensione
    if n != m:
        print('Errore: matrice non quadrata.')
        return 1
    
    # Test singolarità: è necessario che tutti gli elementi diagonali di L
    # siano NON nulli (visto che il determinante di una matrice triangolare 
    # si calcola come il prodotto degli elementi sulla diagonale).
    # NON SI DOVREBBE FARE UN CONTROLLO CON EPS?
    if np.all(np.diag(L)) != True:
        print('Errore: Elemento diagonale nullo.') 
        return 1
     
        
def Lsolve(L,b): 
    """  
    Risoluzione con procedura forward di Lx = b con L triangolare inferiore.
        
    Parametri
    ----------
    L: Matrice triangolare inferiore.
    b: Termine noto.

    Valori di ritorno.
    -------
    flag :  Booleano che è 0 se sono soddisfatti i test di applicabilità.            
    x :     Soluzione del sistema lineare.
    """
    
    m, n = L.shape
    if _solve_applicability(L, m, n) :
        return [], 1
     
    # Preallocazione vettore soluzione
    x = np.zeros((n, 1))
    # N.B.: range(n) va da zero (se non specificato lo start) a n escluso.
    for i in range(n):
        # scalare = vettore riga * vettore colonna
        s = np.dot(L[i,:i], x[:i]) # sommatoria
        x[i] = (b[i] - s) / L[i,i]
    return x, 0


def Usolve(U, b):
    """  
    Risoluzione con procedura backward di Ux = b con U triangolare superiore.
        
    Parametri
    ----------
    U: Matrice triangolare superiore.
    b: Termine noto.

    Valori di ritorno.
    -------
    flag :  Booleano che è 0 se sono soddisfatti i test di applicabilità.            
    x :     Soluzione del sistema lineare.
    """
    m, n = U.shape
    if _solve_applicability(U, m, n) :
        return [], 1
     
    # Preallocazione vettore soluzione
    x = np.zeros((n, 1))
    # N.B.: range(n) va da n-1 a 0.
    for i in range(n - 1, -1, -1):
        # scalare = vettore riga * vettore colonna
        s = np.dot(U[i, i+1:n], x[i+1:n]) # sommatoria
        x[i] = (b[i] - s) / U[i,i]
    return x, 0

def LUsolve(L,U,P,b):
     """
     Risoluzione a partire da PA = LU assegnata
     
     Dato un sistema lineare Ax = b, con A matrice dei coefficienti e b vettore colonna dei termini noti.
     Il sistema è equivalente al sistema PAx = Pb con P matrice di permutazione. 
     Scopo della fattorizzazione: trovare due matrici L triangolare inferiore e U triangolare superiore 
     t.c. PA = LU. 
     
     PAx = Pb <=> P(LU)x = Pb. Chiamo Ux = y. 
     I)  Risolvo il sistema lineare PLy = Pb, determinando y con il metodo di sostituzione in avanti (dato 
         che L è triangolare inferiore). Nota: la matrice di permutazione P opere delle opportune permutazioni 
         invarianti di riga al sistema lineare.
     Una volta determinato y:
     II) Risolvo il sistema lineare Ux = y, determinando x con il metodo di sostituzione all'indietro (dato 
         che U è triangolare superiore).                                                 
     """
     Pb = np.dot(P,b)
     y, flag = Lsolve(L, Pb)
     if flag == 0:
         x, flag = Usolve(U, y)
     else:
        return [],flag

     return x,flag

# -------------------------------------------------------------------------------------------------------
# Metodo di Gauss (o eliminazione gaussiana).
# -------------------------------------------------------------------------------------------------------

def LU_nopivotb(A):
    """  
    Fattorizzazione PA = LU senza pivoting **versione base**
        
    Parametri
    ----------
    A: Matrice dei coefficienti.

    Valori di ritorno.
    -------
    L: Matrice triangolare inferiore
    U: Matrice triangolare superiore
    P: Matrice identità (senza pivoting non vi è alcuna matrice di permutazione)
    """

    # Test dimensione
    m, n = A.shape
    flag = 0;
    if n != m:
        print("Matrice non quadrata")
        L, U, P, flag = [], [], [], 1 
        return P, L, U, flag
  
    # Inizializzo la matrice di permutazione con l'identità
    P = np.eye(n)
    U = A.copy()
    # Fattorizzazione - ciclo esterno: sono necessari n - 1 passi per la fattorizzazione di U.
    for k in range(n-1):
        # Test pivot: i minori principali della matrice A devono essere diversi da zero.
        if U[k, k] == 0:
            print('Elemento diagonale nullo')
            L, U, P, flag = [], [], [], 1 
            return P, L, U, flag

        '''
        Al primo passo si annullano tutti i coefficienti della prima colonna dalla seconda riga fino all'n-esima. 
        Al secondo tutti i coefficienti della seconda colonna dalla terza riga fino all'n-esima.
        ...
        Al passo k tutti i coefficienti della k-esima colonna dalla (k+1)-esima riga fino all'n-esima.
        
        => ciclo per i = k + 1 fino a n
        
        Per far ciò devo calcolare i corrispondenti moltiplicatori, che memorizzo direttamente nella colonna k-esima
        sotto l'elemento diagonale k+1. Questo lo posso far in virtù del fatto che tutti i coefficienti del triangolo
        inferiore diventeranno nulli e che l'elemento u(i,j) del passo (k+1)-esimo è ottenuto a partire da
        u(i,j) - m(i,k) * u(k,j) del passo k-esimo.
        
                                                       Passo k = 1:    
                                                 calcolo moltiplicatori
                                                 
        | u(1,1)  u(1,2)  ...  u(1,n) |     | u(1,1)  u(1,2)  ...  u(1,n) |       
        | u(2,1)  u(2,2)  ...  u(2,n) |     | m(2,1)  u(2,2)  ...  u(2,n) |      
        |   .       .      .     .    |     |   .        .     .     .    |      
        |   .       .      .     .    | --> |   .        .     .     .    | -->  
        |   .       .      .     .    |     |   .        .     .     .    |      
        | u(n,1)  u(n,2)  ...  u(n,n) |     | m(n,1)  u(n,2)  ...  u(n,n) |     
        
                                                               Passo k = 2 ottenuto con
                                                       u(i,j) - m(i,k) * u(k,j) del passo k = 1, 
                                                              sapendo che m(i,k) = u(i,k)       
                                                        
                                            | u(1,1)          u(1,2)         ...          u(1,n)        |
                                            | m(2,1)   u(2,2)-u(2,1)*u(1,2)  ...   u(1,n)-u(2,1)*u(1,n) |  
                                            |   .               .             .             .           |   
                                        --> |   .               .             .             .           |
                                            |   .               .             .             .           | 
                                            | m(n,1)  u(n,2)-u(n,1)*u(n-1,2) ...  u(n,n)-u(n,1)*u(n-1,n)|

        Alla fine dell'algoritmo: 
        - Nel triangolo inferiore di U vi sono i coefficienti moltiplicativi.
        - L è quindi ottenuta aggiungendo alla matrice identità la parte triangolare inferiore di U.
        - U è invece ottenuta estraendo la sua parte triangolare inferiore più la diagonale.
        '''
        for i in range(k+1, n):
            U[i,k] = U[i,k] / U[k,k]               # Calcolo il moltiplicatore.
            for j in range(k+1,n):                                       
                U[i,j] = U[i,j] - U[i,k] * U[k,j]  # Eliminazione gaussiana sulla matrice
     
    L = np.tril(U,-1) + np.eye(n)   # Estrae i moltiplicatori
    U = np.triu(U)                  # Estrae la parte triangolare superiore + diagonale
    return P, L, U, flag


def LU_nopivotv(A):
    """  
    Fattorizzazione PA = LU senza pivoting **versione intermedia**
        
    Parametri
    ----------
    A: Matrice dei coefficienti.

    Valori di ritorno.
    -------
    L: Matrice triangolare inferiore
    U: Matrice triangolare superiore
    P: Matrice identità (senza pivoting non vi è alcuna matrice di permutazione)
    """
    m, n = A.shape
   
    flag = 0;
    if n != m:
        print("Matrice non quadrata")
        L, U, P, flag = [], [], [], 1 
        return P, L, U, flag
  
    P = np.eye(n)
    U = A.copy()
    for k in range(n-1):
        if U[k, k] == 0:
            print('Elemento diagonale nullo')
            L, U, P, flag = [], [], [], 1 
            return P, L, U, flag

        for i in range(k+1, n):
            U[i, k] = U[i, k] / U[k, k]                              
            U[i, k+1:n] = U[i, k+1:n] - U[i, k] * U[k, k+1:n]  
     
    L = np.tril(U,-1) + np.eye(n)
    U = np.triu(U)                
    return P, L, U, flag


def LU_nopivot(A):
    """  
    Fattorizzazione PA = LU senza pivoting **versione vettorizzata**
        
    Parametri
    ----------
    A: Matrice dei coefficienti.

    Valori di ritorno.
    -------
    L: Matrice triangolare inferiore
    U: Matrice triangolare superiore
    P: Matrice identità (senza pivoting non vi è alcuna matrice di permutazione)
    """
    m, n = A.shape
   
    flag = 0;
    if n != m:
        print("Matrice non quadrata")
        L, U, P, flag = [], [], [], 1 
        return P, L, U, flag
  
    P = np.eye(n);
    U = A.copy();
    for k in range(n-1):
        if U[k,k] == 0:
            print('Elemento diagonale nullo')
            L, U, P, flag = [], [], [], 1 
            return P, L, U, flag

        U[k+1:n, k] = U[k+1:n, k] / U[k, k]                                   
        U[k+1:n, k+1:n] = U[k+1:n, k+1:n] - np.outer(U[k+1:n, k], U[k, k+1:n]) 
  
    L = np.tril(U,-1) + np.eye(n)  
    U = np.triu(U)           
    return P, L, U, flag


def swapRows(A,k,r):
    '''
    Scambia le righe k e r della matrice A data in input.

    Parametri
    ----------
    A: Matrice dei coefficienti.
    k, r: Righe della matrice dei coefficienti che si vogliono scambiare.
    '''
    
    A[[k,r],:] = A[[r,k],:]

def LU_pivot(A):
    """  
    Fattorizzazione PA = LU con pivoting parziale
        
    Parametri
    ----------
    A: Matrice dei coefficienti.

    Valori di ritorno.
    -------
    L: Matrice triangolare inferiore
    U: Matrice triangolare superiore
    P: Matrice identità (senza pivoting non vi è alcuna matrice di permutazione)
    """
    m, n = A.shape
   
    flag = 0;
    if n != m:
        print("Matrice non quadrata")
        L, U, P, flag = [], [], [], 1 
        return P, L, U, flag
  
    P = np.eye(n);
    U = A.copy();
    for k in range(n-1):     
        '''
        Nella colonna k-esima della matrice ricerco l'indice di riga r >= k in cui si trova
        l'elemento di modulo massimo e si scambia la riga k-esima con la riga r-esima, applicando
        il medesimo scambio anche alla matrice di permutazione P.
        
        NOTA: argmax ritorna l'indice del massimo valore presente. Siccome gli abbiamo dato in ingresso solo 
        il sottovettore che va dalla riga k-esima alla riga n-esima bisogna aggiungergli l'offset di k.
        '''
        r = np.argmax(abs(U[k:n,k])) + k
        if r != k:
            swapRows(P,k,r)
            swapRows(U,k,r)

        U[k+1:n, k] = U[k+1:n, k] / U[k, k]                                   
        U[k+1:n, k+1:n] = U[k+1:n, k+1:n] - np.outer(U[k+1:n, k], U[k, k+1:n]) 
  
    L = np.tril(U,-1) + np.eye(n)  
    U = np.triu(U)           
    return P, L, U, flag