# -*- coding: utf-8 -*-
"""
04 - File per la soluzione di sistemi Lineari
"""
import numpy as np

def _solve_applicability(L, m, n):
    '''
    Funzione per check di applicabilità dei metodi di sostituzione in 
    avanti e indietro per la risoluzione di sistemi lineari con matrice 
    triangolare.
    
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
    Risoluzione con procedura forward di Lx=b con L triangolare inferiore.
        
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

def Usolve(L, b):
    """  
    Risoluzione con procedura backward di Ux=b con U triangolare superiore.
        
    Parametri
    ----------
    U: Matrice triangolare superiore.
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
    # N.B.: range(n) va da n-1 a 0.
    for i in range(n - 1, -1, -1):
        # scalare = vettore riga * vettore colonna
        s = np.dot(L[i, i+1:n], x[i+1:n]) # sommatoria
        x[i] = (b[i] - s) / L[i,i]
    return x, 0

def LU_nopivot(A):
    """
    % Fattorizzazione PA=LU senza pivot   versione vettorizzata
    In output:
    L matrice triangolare inferiore
    U matrice triangolare superiore
    P matrice identità
    tali che  LU=PA=A
    """
    # Test dimensione
    m,n=A.shape
   
    flag=0;
    if n!=m:
      print("Matrice non quadrata")
      L,U,P,flag=[],[],[],1 
      return P,L,U,flag
  
    P=np.eye(n);
    U=A.copy();
 # Fattorizzazione
    for k in range(n-1):
       #Test pivot 
          if U[k,k]==0:
            print('elemento diagonale nullo')
            L,U,P,flag=[],[],[],1 
            return P,L,U,flag

  #     Eliminazione gaussiana
          U[k+1:n,k]=U[k+1:n,k]/U[k,k]                                   # Memorizza i moltiplicatori	  
          U[k+1:n,k+1:n]=U[k+1:n,k+1:n]-np.outer(U[k+1:n,k],U[k,k+1:n])  # Eliminazione gaussiana sulla matrice
     
  
    L=np.tril(U,-1)+np.eye(n)  # Estrae i moltiplicatori 
    U=np.triu(U)           # Estrae la parte triangolare superiore + diagonale
    return P,L,U,flag

def LU_nopivotv(A):
    """
    % Fattorizzazione PA=LU senza pivot   versione vettorizzata intermedia
    In output:
    L matrice triangolare inferiore
    U matrice triangolare superiore
    P matrice identità
    tali che  LU=PA=A
    """
    # Test dimensione
    m,n=A.shape
   
    flag=0;
    if n!=m:
      print("Matrice non quadrata")
      L,U,P,flag=[],[],[],1 
      return P,L,U,flag
  
    P=np.eye(n);
    U=A.copy();
 # Fattorizzazione
    for k in range(n-1):
       #Test pivot 
          if U[k,k]==0:
            print('elemento diagonale nullo')
            L,U,P,flag=[],[],[],1 
            return P,L,U,flag

  #     Eliminazione gaussiana
          for i in range(k+1,n):
             U[i,k]=U[i,k]/U[k,k]                                   # Memorizza i moltiplicatori	  
             U[i,k+1:n]=U[i,k+1:n]-U[i,k]*U[k,k+1:n]  # Eliminazione gaussiana sulla matrice
     
  
    L=np.tril(U,-1)+np.eye(n)  # Estrae i moltiplicatori 
    U=np.triu(U)           # Estrae la parte triangolare superiore + diagonale
    return P,L,U,flag

def LU_nopivotb(A):
    """
    % Fattorizzazione PA=LU senza pivot  versione base
    In output:
    L matrice triangolare inferiore
    U matrice triangolare superiore
    P matrice identità
    tali che  LU=PA=A
    """
    # Test dimensione
    m,n=A.shape
    flag=0;
    if n!=m:
      print("Matrice non quadrata")
      L,U,P,flag=[],[],[],1 
      return P,L,U,flag
  
    P=np.eye(n);
    U=A.copy();
 # Fattorizzazione
    for k in range(n-1):
         #Test pivot 
         
         
          if U[k,k]==0:
            print('elemento diagonale nullo')
            L,U,P,flag=[],[],[],1 
            return P,L,U,flag

  #     Eliminazione gaussiana
          for i in range(k+1,n):
                U[i,k]=U[i,k]/U[k,k]
                for j in range(k+1,n):                                 # Memorizza i moltiplicatori	  
                  U[i,j]=U[i,j]-U[i,k]*U[k,j]  # Eliminazione gaussiana sulla matrice
     
  
    L=np.tril(U,-1)+np.eye(n)  # Estrae i moltiplicatori 
    U=np.triu(U)           # Estrae la parte triangolare superiore + diagonale
    return P,L,U,flag