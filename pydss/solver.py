#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 23:44:26 2017

@author: jmmauricio
"""

import numba 
import numpy as np 
import diode
from numba import float64,int64





#@numba.jit((float64,float64,int64,int64,float64[:],float64[:]),nopython=True)
@numba.jit(nopython=True)
def run(dt,t_end,decimation,N_states,param_diodes,x_0_diodes):
    
    rect_1 = diode.rectifier6pulse()
    src_1 = diode.source3ph()
#    feval = o.f_eval
    N_steps = int(np.ceil(t_end/dt))
    N_out = int(np.ceil(t_end/dt/decimation))
    
    T = np.zeros((N_out,1))
    X = np.zeros((N_out,N_states))
    
    t = 0.0

    X[0,:] = x_0_diodes
    x = x_0_diodes
    
    it = 0
    it_decimation_1 = 0
    it_decimation_out = 0
    it_save = 0

    u = src_1.h_eval(t,0.0,np.array([0.0]),np.array([0.0]))
    
    solver = 1
    for it in range(N_steps):  
        

        
        if t > 1.0:
            rect_1.R_dc = 10.0
            
        if it_decimation_1 >= 10:
            
            u = src_1.h_eval(t,0.0,np.array([0.0]),np.array([0.0]))
            it_decimation_1 = 0        
        
            
        t += dt        

        if solver == 1:
            # solver forward euler
            
            x = x + dt*(rect_1.f_eval(t,u,x,0.0))
        
        if solver == 2:
            # solver trapezoidal 1 step
            f_1 = diode.f_diode(t,0.0,x,0.0,param_diodes)
            x_1 = x + dt*f_1
            x = x + 0.5*dt*(diode.f_diode(t,u,x_1,0.0,param_diodes) + f_1)
       
        if it_decimation_out >= decimation:
            it_save += 1  
            X[it_save,:] = x
            T[it_save,:] = t
            it_decimation_out = 0
        
        it_decimation_1 += 1
        it_decimation_out += 1
              
        
        
    return T,X

   
def solve(param):
    dt = param['dt']
    t_end = param['t_end']   
    decimation = param['decimation']
    N_states = param['N_states'] 

    param_diodes = param['diodes']['param']
    x_0_diodes = param['diodes']['x_0']
    
    T,X = run(dt,t_end,decimation,N_states,param_diodes,x_0_diodes)
    
    return T,X
    
    
    
@numba.jit
def run1(param):

    dt = param['dt']
    t_end = param['t_end']   
    decimation = param['decimation']
    N_states = param['N_states'] 
    
    N_out = np.ceil(t_end/dt/decimation)+1
    
    T = np.zeros((N_out,1))
    X = np.zeros((N_out,N_states))

    param_diodes = param['diodes']['param']
    x_0_diodes = param['diodes']['x_0']
    
    t = 0.0
    x_0 = x_0_diodes
    x = x_0

    X[0,:] = x
    
    
    it = 0
    it_decimation = 0
    it_save = 0

    while t < t_end:   
        if t > 1.0:
            param[0] = 10.0
            
        t += dt        

        if param['solver'] ==  'fweuler':
            # solver trapezoidal 1 step
            x = x + dt*(diode.f_diode(t,0.0,x,0.0,param_diodes))
        
#        if param['solver'] == 'trapezoidal':
#            # solver trapezoidal 1 step
#            f_1 = diode.f_diode(t,0.0,x,0.0,param_diodes)
#            x_1 = x + dt*f_1
#            x = x + 0.5*dt*(diode.f_diode(t,0.0,x_1,0.0,param_diodes) + f_1)

        it += 1
        
        
        if it_decimation >= decimation:
            it_save += 1  
            X[it_save,:] = x
            T[it_save,:] = t
            it_decimation = 0
            
        it_decimation += 1
              
        
        
    return T,X

if __name__ == "__main__":
    
    import time
    
    param = {'dt': 1e-6, 't_end': 5.0, 'decimation': 200, 'N_states':4, 'solver':'trapezoidal'}
    
    R_dc = 1000.0
    C = 2000e-6
    R = 1.0
    L = 1e-3
    
    param_diodes = {'diodes':{'param': np.array( [10,  325.0, 2.0*np.pi*50.0, L, R, C ]),
                              'x_0': np.array([0.0,0.0,0.0,800.0])}}
        
    param.update(param_diodes)
    
    t_0 = time.time()
    T,X = solve(param)
    
    print(time.time()-t_0)


   