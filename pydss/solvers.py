#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 10:01:08 2017

@author: jmmauricio
"""

import numpy as np
import numba

def nrsolve(xi_0,u_0,lam,Lam, max_iter=1000,tol = 1.0e-8):
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


def ssasolve(xi_0,u_0,lam,Lam,h_eval,N_x):
    
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


@numba.jit(nopython=True,cache=True)
def ssa(struct):
      
    N_x = struct[0]['N_x']
    N_y = struct[0]['N_y']
    x = struct[0]['x']
    y = struct[0]['y']
    
    F_x = struct[0]['F_x']
    F_y = struct[0]['F_y']
    G_x = struct[0]['G_x']
    G_y = struct[0]['G_y']
    
 
    A =  F_x - F_y @ np.linalg.inv(G_y) @ G_x
    lambdas, Phi = np.linalg.eig(A)
    
    sigmas = lambdas.real
    omegas = lambdas.imag

    freqs = omegas/(2*np.pi);
    zetas = sigmas/np.sqrt(sigmas**2+omegas**2)    
    
    return lambdas,freqs,zetas


@numba.jit(nopython=True,cache=True)
def nr(sys_update,struct):
    N_x = struct[0]['N_x']
    N_y = struct[0]['N_y']
    N_l = N_x+N_y
    Lam = np.zeros((N_l,N_l))
    lam = np.zeros((N_x+N_y,1))
    xi_0 = np.ones((N_x+N_y,1))
    xi_0[0:N_x,:]   = struct[0]['x']
    xi_0[N_x:N_l,:] = struct[0]['y']    
    max_iter = 100
    tol = 1.0e-8
    sys_update(struct,0,0)
    for it in range(max_iter):
        lam[0:N_x,:]   = struct[0]['f']
        lam[N_x:N_l,:] = struct[0]['g']
        
        Lam[0:N_x,0:N_x]     = struct[0]['F_x']
        Lam[0:N_x,N_x:N_l]   = struct[0]['F_y']
        Lam[N_x:N_l,0:N_x]   = struct[0]['G_x']
        Lam[N_x:N_l,N_x:N_l] = struct[0]['G_y']
        
        Dxi = np.linalg.solve(-Lam,lam)
        xi_1 = xi_0 + Dxi

        xi_0 = xi_1
        
        struct[0]['x'][:] = xi_0[0  :N_x,:]
        struct[0]['y'][:] = xi_0[N_x:N_l,:]
        
        sys_update(struct,0,0)
        if np.linalg.norm(Dxi,np.infty)<tol: 
            #print(it)
            break
    return xi_0