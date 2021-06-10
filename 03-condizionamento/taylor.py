#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Funzioni per l'esericizio 7
'''

def esp_taylor_1(x):
    """
    Input: x  
    Output: appr_new=exp(x), n=ultimo indice di sommatoria considerato
    Calcolo exp(x) con serie di Taylor troncata in accordo a test dell'incremento 
    """
    appr_old = 0.e0
    appr_new = 1.e0
    temp = 1.e0 
    n = 0
    while appr_old != appr_new:
          appr_old = appr_new
          n = n + 1;
          temp = temp * x / n      # calcolo n! come (n-1)!*n e x^n come x^(n-1)*x
          appr_new = appr_old + temp
    return appr_new,n
 
def esp_taylor_2(x):
    """
    Output: appr_new=exp(x), n=ultimo indice di sommatoria considerato
    Calcolo exp(x) con serie di Taylor troncata in accordo a test dell'incremento 
    usando l'espressione equivalente exp(-x)=1/exp(x)
    """
    appr_new, n = esp_taylor_1(-x);
    appr_new = 1.e0 / appr_new
    return appr_new, n