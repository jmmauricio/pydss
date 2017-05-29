#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 10:01:08 2017

@author: jmmauricio
"""

import numpy as np

def nr(xi_0,u_0,lam,Lam, max_iter=1000,tol = 1.0e-8):
    '''
    Newton-Raphson method
    
    Parameters
    ----------
    xi_0 : array_like
        Initial guess for xi.
    u_0 : array_like
        Systen inputs.
    lam : function vector
        DAE system equations
    lam : function vector
        DAE system jacobian.
        
    Returns
    -------
    xi_1 : array_like
        Obtained result.  
    '''
    
    for it in range(max_iter):
        Dxi = np.linalg.solve(-Lam(xi_0,u_0),lam(xi_0,u_0))
        xi_1 = xi_0 + Dxi
        if np.linalg.norm(Dxi,np.infty)<tol: 
            break
        if it==max_iter: 
            print('No convergence after', it)
            break
        xi_0 = xi_1
    return xi_1


def ssa(xi_0,u_0,lam,Lam,h_eval,N_x):
    
    N_t = xi_0.shape[0]
    N_y = N_t - N_x
       
    xi = nr(xi_0,u_0,lam,Lam)

    x = xi[0:N_x]
    y = xi[N_x:(N_x+N_y)]
    
    Lam_0 = Lam(xi,u_0)
    
    F_x = Lam_0[0:N_x  ,  0:N_x]
    F_y = Lam_0[0:N_x  ,N_x:N_t]
    G_x = Lam_0[N_x:N_t,  0:N_x]
    G_y = Lam_0[N_x:N_t,N_x:N_t]
    
    z = h_eval(xi,u_0)
    A =  F_x - F_y @ np.linalg.inv(G_y) @ G_x
    lambdas, Phi = np.linalg.eig(A)
    
    sigmas = lambdas.real
    omegas = lambdas.imag

    freqs = omegas/(2*np.pi);
    zetas = sigmas/np.sqrt(sigmas**2+omegas**2)    
    
    return lambdas, z,freqs,zetas

