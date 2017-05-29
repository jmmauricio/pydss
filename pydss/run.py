#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 11:41:44 2017

@author: jmmauricio
"""

import numpy as np
import numba
from pf import pf_eval
from electric import vsc_former,ctrl_vsc_former_phasor
from ctrl import secondary_ctrl

@numba.jit(nopython=True,cache=True)
def run_eval(params, params_pf,params_vsc,params_secondary):
    
    # simulation parameters
    N_steps = params[0].N_steps
    Dt = params[0].Dt
    Dt_out = params[0].Dt_out
    
    Dt_secondary = params_secondary[0].Dt_secondary    

    # power flow 
    pf_eval(params_pf)   
     
    # initializations
    for it in range(len(params_vsc)):
               
        params_vsc[it].v_abcn_0[:] = np.copy(params_pf[0].V_node[params_vsc[it].nodes])             
        params_vsc[it].i_abcn_0[:] = np.copy(params_pf[0].I_node[params_vsc[it].nodes])
        params_vsc[it].x[:] = np.zeros((4,1))
        ctrl_vsc_former_phasor(0.0,0,params_vsc,params_secondary)  
     
    
    t = 0.0
    t_out = 0.0
    t_secondary = 0.0
    it_out = 0
    it_pert = 0
    
    params[0]['T'][it_out,0] = t
    params[0].out_cplx_i[it_out,:] = params_pf[0].I_node.T
    params[0].out_cplx_v[it_out,:] = params_pf[0].V_node.T

    # initial output saving
    for it in range(len(params_vsc)):
        params[0].T_sink[0,it] = params_vsc[it].T_sink
        params[0].T_j_igbt_abcn[0,4*it:4*(it+1)] = params_vsc[it].T_j_igbt_abcn.T  
                
    # main loop
    for it in range(N_steps):

        t += Dt

        ## from power flow to elements
        for it in range(len(params_vsc)):                
            params_vsc[it].i_abcn[:] = params_pf[0].I_node[params_vsc[it].nodes] 

        ## derivatives update                     
        ctrl_vsc_former_phasor(t,1,params_vsc,params_secondary)  

        ## ode solver
        for it in range(len(params_vsc)):                
            params_vsc[it].x[:] += Dt*params_vsc[it].f[:]        
        
        ## elements outputs update
        ctrl_vsc_former_phasor(t,3,params_vsc,params_secondary)  

        ## from elements to power flow
        for it in range(len(params_vsc)):             
            params_pf[0].V_node[params_vsc[it].nodes,:] = params_vsc[it].v_abcn[:] 
            
        ## power flow update    
        pf_eval(params_pf)     
        
        
        # perturbations        
        if t >= params[0].perturbations_times[it_pert][0]:
            if params[0].perturbations_types[it_pert][0] == 1:
                params_pf[0].pq_3pn[params[0].perturbations_int[it_pert],:] = params[0].perturbations_cplx[it_pert,0:3]
            if it_pert < len(params[0].perturbations_times)-1:
                it_pert += 1
                
            
        # secondary update           
        if t > t_secondary+Dt_secondary:
            V_mean = (np.sum(np.abs(params_pf[0].V_node[0:3,:]))+ np.sum(np.abs(params_pf[0].V_node[4:7,:])))/6.0 
            params_secondary[0].V = V_mean
            secondary_ctrl(t,1,params_secondary,params_vsc)
            
            params_secondary[0].x[:] += Dt_secondary*params_secondary[0].f[:]
            secondary_ctrl(t,2,params_secondary,params_vsc)
            t_secondary = t 

        
        # output update             
        if t > t_out+Dt_out:
            it_out += 1  
            params[0]['T'][it_out,0] = t
            
            for it in range(len(params_vsc)):
                params[0].T_sink[it_out,it] = params_vsc[it].T_sink
                params[0].T_j_igbt_abcn[it_out,4*it:4*(it+1)] = params_vsc[it].T_j_igbt_abcn.T 

                
            params[0].out_cplx_i[it_out,:] = params_pf[0].I_node.T 
            params[0].out_cplx_v[it_out,:] = params_pf[0].V_node.T  
            params[0].N_outs = it_out
            t_out = t 
            