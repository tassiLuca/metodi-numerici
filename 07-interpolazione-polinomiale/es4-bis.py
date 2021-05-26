# -*- coding: utf-8 -*-
"""
es4-bis
I nodi di chebyshev bastano?
"""


import numpy as np
import interpolazione as intrpl
import matplotlib.pyplot as plt
import math


def zeri_Cheb(a,b,n):
    t1=(a+b)/2
    t2=(b-a)/2
    x=np.zeros((n+1,))
    for k in range(n+1):
        x[k]=t1+t2*np.cos(((2*k+1)/(2*(n+1))*math.pi))

    return x
                                                
f,a,b = lambda x: np.sqrt(np.abs(x)), -1, 1

# punti di valutazione per l'interpolante
xx=np.linspace(a,b,200);

for n in range(5, 25):
    x=zeri_Cheb(a,b,n)
    y = f(x)
    
    pol = intrpl.lagrange_interp(x,y,xx);

    plt.plot(xx,pol,'b--',x,y,'r*',xx,f(xx),'m-');
    plt.legend(['interpolante di Lagrange','punti di interpolazione','Funzione']);
    plt.show()
    
    r=np.abs(f(xx)-pol)
    norm_inf_r=np.linalg.norm(r,np.inf)
    print("n = ", n, " --> Norma infinito di r ",norm_inf_r)




