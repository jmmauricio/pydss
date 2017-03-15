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
    ('J', float64), 
    ('K_b', float64),           
]
@jitclass(spec)
class rotatory1(object):
    
    def __init__(self):
        
        P_n = 30.0e3
        n_n = 200.0
        omega_n = n_n*2.0*np.pi/60
        H = 3.5
        
        # H = 0.5*J*omega_n**2/P_n
        J = 2.0*H*P_n/(omega_n**2)
        
        # P = K_b*omega**2
        K_b = 0.1*P_n/omega_n**2.0 
        
        
        self.N_states = 1
        self.x_0 = np.array([0.0])
        self.J = J
        self.K_b = K_b

        
    def f_eval(self,t,u,x,y):
        
        omega = x[0]

        tau_1 = u[0]
        tau_2 = u[1]
        
        J = self.J
        K_b = self.K_b
    
        domega= 1.0/J*(tau_1 - K_b*np.abs(omega)*omega - tau_2)
    
        return np.array([domega])

    def h_eval(self,t,u,x,y):
        
        omega = x[0]

        return np.array([omega])


spec = [
    ('N_states', int32),               # a simple scalar field
    ('x_0', float64[:,:]), 
    ('x', float64[:,:]), 
    ('f', float64[:,:]), 
    ('J', float64), 
    ('K_b', float64),   
    ('c_coefs', float64[:]),
    ('K_gear', float64),
    ('tau_r', float64), 
    ('tau_t', float64), 
    ('beta', float64),   
    ('nu', float64), 
    ('p_m', float64), 
    ('Radio', float64),       
]
@jitclass(spec)
class wind_turbine(object):
    
    def __init__(self):
        
        self.N_states = 1
        self.x_0 = np.zeros((1,1))
        self.x = np.zeros((1,1))
        self.f = np.zeros((1,1))
        
        c_1 = 0.5176
        c_2 = 116.
        c_3 = 0.4
        c_4 = 5.
        c_5 = 21.
        c_6 = 0.0068
         
        self.c_coefs = np.array([c_1,c_2,c_3,c_4,c_5,c_6])
        
         
        
        P_n = 30.0e3
        n_n = 200.0
        omega_n = n_n*2.0*np.pi/60
        H = 3.5
        
        # H = 0.5*J*omega_n**2/P_n
        J = 2.0*H*P_n/(omega_n**2)
        
        # P = K_b*omega**2
        K_b = 0.1*P_n/omega_n**2.0 
        
        
        self.N_states = 1

        self.J = J
        self.K_b = K_b
        self.K_gear = 1.0

        self.tau_r = 0.0
        self.tau_t = 0.0
        self.p_m = 0.0
        self.beta = 0.0
        self.Radio = 10.0
        self.nu = 10.0
        
        self.x[0] = omega_n
        
    def aero(self,omega_t,nu,beta):
        
        Radio = self.Radio
        c_1,c_2,c_3,c_4,c_5,c_6 = self.c_coefs
        
        lambda_val = (omega_t*Radio)/nu      
        
        inv_lambda_i = 1.0/(lambda_val+0.008*beta) - 0.035/(beta**3+1.0)
        c_p  = c_1*(c_2*inv_lambda_i - c_3*beta - c_4)*np.exp(-c_5*inv_lambda_i)+c_6*lambda_val
       
        rho = 1.2922 # kg/m**3
        A = np.pi*Radio**2
        p_m = c_p*rho*A/2.0*nu**3.0
        
        return p_m
        
    def f_eval(self):

        nu = self.nu
        tau_r = self.tau_r
        beta = self.beta
        
        J = self.J
        K_b = self.K_b
        
        omega_r = self.x[0]
    
        omega_t = omega_r/self.K_gear

        p_m = self.aero(omega_t,nu,beta)
        
        tau_t = np.zeros((1,))
        if omega_t[0]>0.001:               
            tau_t = p_m/omega_t
            
        domega= 1.0/J*(tau_t - K_b*np.abs(omega_t)*omega_t - tau_r)

        self.f[0] = domega
        self.tau_t = tau_t[0]
        self.p_m = p_m[0]
        return self.f
    
    def h_eval(self,t,u,x,y):
        
        omega = x[0]
    
        return np.array([omega])



       

if __name__ == "__main__":
    
    w1 = wind_turbine()
    w1.f_eval()
