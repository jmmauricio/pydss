#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 23:44:26 2017

@author: jmmauricio
"""

import numba 
import numpy as np 
from numba import jitclass          # import the decorator
from numba import int32, float64 , int64   # import the types



@numba.jit
def f_diode(t,u,x,y,param):
    
    i_a = x[0]
    i_b = x[1]
    i_c = x[2]
    v_dc = x[3]
    
    R_dc = param[0]
    V_ac = param[1]
    omega = param[2]
    L = param[3]
    R = param[4]
    C = param[5]
    
    v_s_a = V_ac*np.sin(omega*t)
    v_s_b = V_ac*np.sin(omega*t - 2.0/3.0*np.pi)
    v_s_c = V_ac*np.sin(omega*t - 4.0/3.0*np.pi)
    
    u_a = 0.0
    if i_a >= 0.0: u_a = 1.0       
    
    u_b = 0.0
    if i_b >= 0.0: u_b = 1.0 
     
    u_c = 0.0
    if i_c >= 0.0: u_c = 1.0 
                   
    e_a = (2*u_a - u_b - u_c)/3.0*v_dc
    e_b = (2*u_b - u_c - u_a)/3.0*v_dc
    e_c = (2*u_c - u_a - u_b)/3.0*v_dc

    di_a = 1.0/L*(v_s_a - R*i_a - e_a)
    di_b = 1.0/L*(v_s_b - R*i_b - e_b)
    di_c = 1.0/L*(v_s_c - R*i_c - e_c)

    i_d = u_a*i_a + u_b*i_b + u_c*i_c

    dv_dc = 1.0/C*(i_d - (v_dc-400)/R_dc)

    return np.array([di_a, di_b,di_c,dv_dc])



spec = [
    ('N_states', int32),               # a simple scalar field
    ('x_0', float64[:]), 
    ('L', float64), 
    ('R', float64), 
    ('C', float64),  
    ('R_dc', float64),              
]
@jitclass(spec)
class rectifier6pulse(object):
    
    def __init__(self):
        
        self.N_states = 4
        self.x_0 = np.array([0.0,0.0,0.0,0.0])
        self.L = 1.0e-3
        self.R = 0.1
        self.C = 2000.0e-6
        self.R_dc = 100.0
        
    def f_eval(self,t,u,x,y):
        
        i_a = x[0]
        i_b = x[1]
        i_c = x[2]
        v_dc = x[3]

        v_s_a = u[0]
        v_s_b = u[1]
        v_s_c = u[2]
        
        R_dc = self.R_dc
        L = self.L
        R = self.R
        C = self.C

        
        u_a = 0.0
        if i_a >= 0.0: u_a = 1.0       
        
        u_b = 0.0
        if i_b >= 0.0: u_b = 1.0 
         
        u_c = 0.0
        if i_c >= 0.0: u_c = 1.0 
                       
        e_a = (2*u_a - u_b - u_c)/3.0*v_dc
        e_b = (2*u_b - u_c - u_a)/3.0*v_dc
        e_c = (2*u_c - u_a - u_b)/3.0*v_dc
    
        di_a = 1.0/L*(v_s_a - R*i_a - e_a)
        di_b = 1.0/L*(v_s_b - R*i_b - e_b)
        di_c = 1.0/L*(v_s_c - R*i_c - e_c)
    
        i_d = u_a*i_a + u_b*i_b + u_c*i_c
    
        dv_dc = 1.0/C*(i_d - (v_dc-400)/R_dc)
    
        return np.array([di_a, di_b,di_c,dv_dc])




        
spec = [
    ('N_states', int32),               # a simple scalar field
    ('x_0', float64[:]), 
    ('L', float64), 
    ('R', float64), 
    ('C', float64),  
    ('R_dc', float64),   
    ('Phi', float64),  
    ('N_pp', float64),             
]
@jitclass(spec)
class pmsm_rectifier(object):
    
    def __init__(self):
        
        S_n = 30.0e3
        n_r_n = 200.0 # rotor nominal mechanical speed
        omega_r_n = n_r_n*2.0*np.pi/60.0
        U_rms = 400.0
        V_peak = U_rms*np.sqrt(2.0/3.0)
        N_pp = 8
        omega_e_n = N_pp * omega_r_n
        
        Phi = V_peak/omega_e_n
        
        X_pu = 0.1
        Z_b = U_rms**2/S_n
        X = X_pu*Z_b
        L = X/omega_e_n
        
        rend = 0.95
        
        P_loss = (1.0-rend)*S_n
        I_n = S_n/(np.sqrt(3.0)*U_rms)
        R = P_loss/(3.0*I_n**2)
        
        self.N_states = 5
        self.x_0 = np.array([0.0,0.0,0.0,800.0,0.0])
        self.L = L
        self.R = R
        self.C = 2000.0e-6
        self.R_dc = 100.0
        self.Phi = Phi
        self.N_pp = N_pp
        
    def f_eval(self,t,u,x,y):

        R_dc = self.R_dc
        L = self.L
        R = self.R
        C = self.C
        Phi = self.Phi
        N_pp = self.N_pp
        
        i_a = x[0]
        i_b = x[1]
        i_c = x[2]
        v_dc = x[3]
        theta = x[4]
   
        omega = u[0]

        v_s_a = Phi*N_pp*omega*np.sin(theta)
        v_s_b = Phi*N_pp*omega*np.sin(theta - 2.0/3.0*np.pi)
        v_s_c = Phi*N_pp*omega*np.sin(theta - 4.0/3.0*np.pi)
            
        u_a = 0.0
        if i_a >= 0.0: u_a = 1.0       
        
        u_b = 0.0
        if i_b >= 0.0: u_b = 1.0 
         
        u_c = 0.0
        if i_c >= 0.0: u_c = 1.0 
                       
        e_a = (2*u_a - u_b - u_c)/3.0*v_dc
        e_b = (2*u_b - u_c - u_a)/3.0*v_dc
        e_c = (2*u_c - u_a - u_b)/3.0*v_dc
    
        di_a = 1.0/L*(v_s_a - R*i_a - e_a)
        di_b = 1.0/L*(v_s_b - R*i_b - e_b)
        di_c = 1.0/L*(v_s_c - R*i_c - e_c)
    
        i_d = u_a*i_a + u_b*i_b + u_c*i_c
    
        dv_dc = 1.0/C*(i_d - (v_dc)/R_dc)
        
        dtheta = N_pp*omega
    
        return np.array([di_a, di_b,di_c,dv_dc,dtheta])
    
    def h_eval(self,t,u,x,y):
        
        i_a = x[0]
        i_b = x[1]
        i_c = x[2]
        theta = x[4]
        
        Phi = self.Phi
        N_pp = self.N_pp
        
        p_s_a = N_pp*Phi*np.sin(theta)
        p_s_b = N_pp*Phi*np.sin(theta - 2.0/3.0*np.pi)
        p_s_c = N_pp*Phi*np.sin(theta - 4.0/3.0*np.pi)
        
        tau_e = i_a*p_s_a + i_b*p_s_b + i_c*p_s_c
        
        return np.array([tau_e])


if __name__ == "__main__":
    
    pmsm_1 = pmsm_rectifier()
    pmsm_1.f_eval(0.0,np.array([0.0]),np.array([0.0,0.0,0.0,0.0,0.0]),np.array([0.0]))