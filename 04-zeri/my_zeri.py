#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 18:28:49 2021

@author: lucatassi
"""

import numpy as np
import math

def sign(x):
    return math.copysign(1, x)


def bisezione(fname, a, b, tol):
    f_a = fname(a)
    f_b = fname(b)
    if sign(f_a) == sign(f_b):
        print("ERRORE")
        return [], [], 0
    
    eps = np.spacing(1)
    approx_sequence = []
    it = 0
    max_it = int(math.ceil(math.log((b - a) / tol) / math.log(2)))
    
    while it < max_it and abs(b - a) >= tol + eps * max(abs(a), abs(b)):
        it += 1
        middle = a + (b - a) / 2
        f_middle = fname(middle)
        approx_sequence.append(middle)
        
        if f_middle == 0:
            break
        elif sign(f_middle) == sign(f_a):
            a = middle
            f_a = f_middle
        elif sign(f_middle) == sign(f_b):
            b = middle
            f_b = f_middle
            
    return middle, approx_sequence, it

def regula_falsi(fname, a, b, tol, max_it):
    f_a = fname(a)
    f_b = fname(b)
    if sign(f_a) == sign(f_b):
        print("ERRORE")
        return [], [], 0
    
    eps = np.spacing(1)
    approx_sequence = []
    it = 0
    f_xi = a
    
    while it < max_it and abs(b - a) >= tol + eps * max(abs(a), abs(b)) and abs(f_xi) >= tol:
        it += 1 
        xi = a - f_a * ((b - a) / (f_b - f_a))
        f_xi = fname(xi)
        approx_sequence.append(xi)
        
        if f_xi == 0:
            break
        elif sign(f_xi) == sign(f_a):
            a = xi
            f_a = f_xi
        elif sign(f_xi) == sign(f_b):
            b = xi
            f_b = f_xi
            
    return xi, approx_sequence, it

def corde(fname, fpname, trigger, tolx, toly, max_it):
    approx_sequence = []
    it = 0
    m = fpname(trigger)
    prev_x = trigger
  
    while True:
        it += 1
        d = fname(prev_x) / m
        next_x = prev_x - d
        f_next = fname(next_x)
        approx_sequence.append(next_x)
        
        if it >= max_it or abs(d) < tolx * abs(next_x) or abs(f_next) < toly:
            if it >= max_it:
                print("ERROR: max_it reached")
            break
        else :
            prev_x = next_x
    return next_x, approx_sequence, it

def secanti(fname, xm1, x0, tolx, toly, max_it):
    approx_sequence = []
    it = 0
    f_xim1 = fname(xm1)
    f_xi = fname(x0)
    xim1 = xm1
    xi = x0
    
    while True:
        it += 1
        xip1 = xi - f_xi * ((xi - xim1) / (f_xi - f_xim1))
        f_xip1 = fname(xip1)
        approx_sequence.append(xip1)
        
        if it >= max_it or abs(xip1 - xi) < tolx * abs(xi) or abs(f_xip1) < toly:
            break
        else:
            xim1 = xi
            xi = xip1
            f_xim1 = f_xi
            f_xi = f_xip1
            
    return xip1, approx_sequence, it
        
def newton(fname, fpname, trigger, tolx, toly, it_max):
    approx_sequence = []
    it = 0
    prev = trigger
    f_prev = fname(trigger)
    fp_prev = fpname(trigger)
    
    while True:
        it += 1
        if abs(fp_prev) < np.spacing(1):
            print("ERRORE newton: derivata nulla = ", fp_prev)
            return prev, approx_sequence, it
        nxt = prev - (f_prev / fp_prev)
        f_nxt = fname(nxt)
        fp_nxt = fpname(nxt)
        approx_sequence.append(nxt)
        
        if it >= it_max or abs(nxt - prev) < tolx * abs(nxt) or abs(f_nxt) < toly :
            break
        else :
            prev = nxt
            f_prev = f_nxt
            fp_prev = fp_nxt
            
    return nxt, approx_sequence, it

def newton_m(fname, fpname, trigger, m, tolx, toly, it_max):
    approx_sequence = []
    it = 0
    prv = trigger
    f_prv = fname(prv)
    fp_prv = fpname(prv)
    
    while True:
        it += 1
        if abs(fp_prv) < np.spacing(1):
            print("ERRORE: derivata nulla")
            return prv, approx_sequence, it
        nxt = prv - m * (f_prv / fp_prv)
        f_nxt = fname(nxt)
        fp_nxt = fpname(nxt)
        approx_sequence.append(nxt)
        
        if it >= it_max or abs(nxt - prv) < tolx * abs(prv) or abs(f_nxt) < toly :
            break
        else:
            prv = nxt
            f_prv = f_nxt
            fp_prv = fp_nxt
    
    return nxt, approx_sequence, it
            
def iterazione(gname, trigger, tolx, it_max):
    approx_sequence = [trigger]
    it = 0
    prv = trigger
    
    while True:
        it += 1
        nxt = gname(prv)
        approx_sequence.append(nxt)
        
        if it >= it_max or abs(nxt - prv) < tolx * abs(nxt):
            print("max number of iterations reached") if it == it_max else print()
            break
        else:
            prv = nxt
    
    return nxt, approx_sequence, it

def stima_ordine(xk, iterazioni):
    p = []
    
    for k in range(iterazioni - 3):
        p.append(np.log(abs(xk[k + 2] - xk[k + 3]) / abs(xk[k + 1] - xk[k+2])) /
                 np.log(abs(xk[k + 1] - xk[k + 2]) / abs(xk[k] - xk[k + 1])))
        
    ordine = p[-1]
    return ordine
    
    
    
    