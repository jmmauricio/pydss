#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 11:41:44 2017

@author: jmmauricio
"""

import numpy as np
import numba

@numba.jit(nopython=True,cache=True)
def pf_eval(params,max_iter=20):
    Y_vv =  params[0].Y_vv
    inv_Y_ii = params[0].inv_Y_ii
    Y_iv =  params[0].Y_iv
    N_v = params[0].N_nodes_v
    N_i = params[0].N_nodes_i
    pq_3pn_int = params[0].pq_3pn_int
    pq_3pn     = params[0].pq_3pn
    V_node = params[0].V_node
    I_node = params[0].I_node
    
    V_unknown = np.copy(V_node[N_v:])
    I_known   = np.copy(I_node[N_v:])
    V_known   = np.copy(V_node[0:N_v])
    
    Y_vi =  Y_iv.T
    V_unknown_0 = V_unknown
    
    for iteration in range(max_iter):

        for it in range(pq_3pn_int.shape[0]):
            
            V_abc = V_unknown[pq_3pn_int[it][0:3],0]
            S_abc = pq_3pn[it,:]
           
            I_known[pq_3pn_int[it][0:3],0] = np.conj(S_abc/V_abc)
            I_known[pq_3pn_int[it][3],0] =  -np.sum(I_known[pq_3pn_int[it][0:3],0])
        
        V_unknown = inv_Y_ii @ ( I_known - Y_iv @ V_known)
        
        if np.linalg.norm(V_unknown - V_unknown_0,np.inf) <1.0e-8: break
        V_unknown_0 = V_unknown

    I_unknown =Y_vv @ V_known + Y_vi @ V_unknown
    
    V_node[0:N_v,:] = V_known 
    V_node[N_v:,:]  = V_unknown 

    I_node[0:N_v,:] = I_unknown 
    I_node[N_v:,:]  = I_known 
        
    return V_node,I_node



                


 

