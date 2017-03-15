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


pi_spec = [
    ('N_states', int32),               # a simple scalar field
    ('x_0', float64[:,:]), 
    ('f', float64[:,:]), 
    ('h', float64[:,:]), 
    ('K_p', float64), 
    ('T_pi', float64),           
]

@jitclass(pi_spec)
class pi(object):
    
    def __init__(self):
        
        self.N_states = 1
        self.x_0 = np.zeros((1,1))
        self.f = np.zeros((1,1))        
        self.h = np.zeros((1,1))
        
        self.K_p = 0.001/0.0001
        self.T_pi = 0.001

        
    def f_eval(self,t,u,x,y):
        
        ref = u[0]
        meas = u[1]
    
        error = ref - meas
        
        self.f[0] = error
    
        return self.f


    def h_eval(self,t,u,x,y):
        
        xi = x[0]
        
        K_p = self.K_p
        T_pi = self.T_pi
    
        ref = u[0]
        meas   = u[1]
    
        error = ref - meas
        
        self.h[0] = K_p*(error + 1.0/T_pi*xi)
        
        return self.h


spec = [
    ('N_states', int32),               # a simple scalar field
    ('R', float64), 
    ('L', float64),   
    ('C', float64),  
    ('K_p_i', float64),   
    ('T_pi_i', float64), 
    ('Tau_i_d', float64),  
    ('x_0', float64[:,:]),
    ('x', float64[:,:]), 
    ('f', float64[:,:]), 
    ('h', float64[:,:]), 
    ('T_clark', float64[:,:]),    
    ('T_inv_clark', float64[:,:]),
    ('eta_abc', float64[:,:])
]

@jitclass(spec)
class ctrl_vsc(object):
    
    def __init__(self):
               
        self.N_states = 3
        self.PWM_mode = 3
        self.x_0 =np.zeros((3,1))
        
        self.x_0[0] = 0.0
        self.x_0[1] = 0.0
        self.x_0[2] = 0.0               
        
        self.x = self.x_0        

        self.L = 1.0e-3
        self.R = 1.0
        self.C = 2200.0e-6
        
        self.Tau_i_d = 1.0e-3

        self.K_p = self.L/self.Tau_i_d
        self.Tau_pi = self.L/self.R
        
        pi = np.pi
        self.T_clark =np.zeros((3,3))
        self.T_clark[0,0:3] = 2.0/3.0*np.array([np.cos(0.0), np.cos(2.0/3.0*pi), np.cos(4.0/3.0*pi)])
        self.T_clark[1,0:3] = 2.0/3.0*np.array([np.sin(0.0), np.sin(2.0/3.0*pi), np.sin(4.0/3.0*pi)])
        self.T_clark[2,0:3] = 2.0/3.0*np.array([        0.5,                0.5,                0.5])


        self.T_inv_clark = np.linalg.inv(self.T_clark)
 
        self.f =np.zeros((3,1))
        self.h =np.zeros((7,1))
        
        self.eta_abc =np.zeros((3,1))

        
    def f_eval(self,t,u,x,y):
        
        xi_d   = x[0]
        xi_q   = x[1]
        xi_vdc = x[2]
        
        R = self.R
        L = self.L
        C = self.C
        
        i_d_ref = u[0]
        i_q_ref = u[1]
        v_dc_ref = u[2]
            
        i_d = u[3]
        i_q = u[4]
        v_dc= u[5]
        
        p = u[6]
        q = u[7]
   
                    
        error_i_d = i_d_ref - i_d
        error_i_q = i_q_ref - i_q
        error_v_dc = v_dc_ref - v_dc
        
        eta_dq = self.T_clark @ self.eta_abc
        eta_d = eta_dq[0]
        eta_q = eta_dq[1]

        
        v_s_dq = self.T_clark @ u[3:6,:]
        v_s_d = v_s_dq[0]
        v_s_q = v_s_dq[1]       

    
        dxi_d = error_i_d
        dxi_q = error_i_q
        dxv_dc = error_v_dc
        
        self.f[0] = dxi_d
        self.f[1] = dxi_q
        self.f[2] = dxv_dc       
        
        return self.f


    def h_eval(self,t,u,x,y):
        
        eta_dqz = np.zeros((3,1))
        
        xi_d   = x[0]
        xi_q   = x[1]
        xi_vdc = x[2]
    
    

        eta_dqz[0] = 1.0/L*(0.5*eta_d*v_dc - R*i_d - v_s_d)
        eta_dqz[1] = 1.0/L*(0.5*eta_q*v_dc - R*i_q - v_s_q)
        eta_dqz[2] = 0.0
        
        eta_abc = self.T_inv_clark @ eta_dqz
        
        self.h[0:3] = eta_abc
        
        return self.h
    
    def pwm(self,t,u):
        
        Dt = 50.0e-6
        
        if self.PWM_mode == 1:

            self.eta_abc[0] = -1.0
            self.eta_abc[1] = -1.0
            self.eta_abc[2] = -1.0
            
            c =   np.arcsin(np.sin(2.0*np.pi*2500*t))/(np.pi/2.0)
            if u[0,0]>c: self.eta_abc[0] = 1.0
            if u[1,0]>c: self.eta_abc[1] = 1.0
            if u[2,0]>c: self.eta_abc[2] = 1.0

        if self.PWM_mode == 2:

            pwm_a = np.zeros((200,))-1
            pwm_b = np.zeros((200,))-1
            pwm_c = np.zeros((200,))-1
            
            t_pwm = np.linspace(0.0,50.0e-6,200)+t
            carrier =   np.arcsin(np.sin(2.0*np.pi*5000*t_pwm))
            pwm_a[carrier>u[0,:]] = 1.0
            pwm_b[carrier>u[1,:]] = 1.0        
            pwm_c[carrier>u[2,:]] = 1.0   
    
            self.eta_abc[0] = np.sum(pwm_a)/200.0
            self.eta_abc[1] = np.sum(pwm_b)/200.0
            self.eta_abc[2] = np.sum(pwm_c)/200.0

        if self.PWM_mode == 3:
            
            self.eta_abc[0] = 0.0
            self.eta_abc[1] = 0.0
            self.eta_abc[2] = 0.0
            
            t_1 = t
            t_2 = t + Dt
            
            c_1 =   np.arcsin(np.sin(2.0*np.pi*5000*t_1))/(np.pi/2.0)
            c_2 =   np.arcsin(np.sin(2.0*np.pi*5000*t_2))/(np.pi/2.0)
            
            
            
            for it in range(3):
                
                t_s = (t_2-t_1)/(c_2 - c_1)*(u[it,0] - c_1) + t_1
                    
                if u[it,0]>c_1 and u[it,0]<c_2:
                    self.eta_abc[it,0] = -((t_s - t_1)*(-1.0) + (t_2-t_s)*( 1.0))/Dt
                
                if u[it,0]<c_1 and u[it,0]>c_2:
                    self.eta_abc[it,0]  = -((t_s - t_1)*( 1.0) + (t_2-t_s)*(-1.0))/Dt
                
                if u[it,0]>c_1 and u[it,0]>c_2:
                    self.eta_abc[it,0]  = 1.0
                    
                if u[it,0]<c_1 and u[it,0]<c_2:
                    self.eta_abc[it,0]  = -1.0
                    
            return self.eta_abc


    def ctrl_design(self):
        
        self.K_p = self.L/self.Tau_i_d
        self.Tau_pi = self.L/self.R
  
      
pi_spec = [
    ('N_states', int32),               # a simple scalar field
    ('x_0', float64[:,:]), 
    ('f', float64[:,:]), 
    ('h', float64[:,:]), 
    ('K_p', float64), 
    ('T_pi', float64),           
]

@jitclass(pi_spec)
class pi(object):
    
    def __init__(self):
        
        self.N_states = 1
        self.x_0 = np.zeros((1,1))
        self.f = np.zeros((1,1))        
        self.h = np.zeros((1,1))
        
        self.K_p = 0.001/0.0001
        self.T_pi = 0.001

        
    def f_eval(self,t,u,x,y):
        
        ref = u[0]
        meas = u[1]
    
        error = ref - meas
        
        self.f[0] = error
    
        return self.f


    def h_eval(self,t,u,x,y):
        
        xi = x[0]
        
        K_p = self.K_p
        T_pi = self.T_pi
    
        ref = u[0]
        meas   = u[1]
    
        error = ref - meas
        
        self.h[0] = K_p*(error + 1.0/T_pi*xi)
        
        return self.h
    
    
@numba.jit(nopython=True)
def run():
    dt = 5e-6
    t_end = 0.1  
    decimation = 10
    N_steps = int(t_end/dt)
    N_saves = int(t_end/dt/decimation)
    T = np.zeros((N_saves,1))
    X = np.zeros((N_saves,2))
    
    t = 0.0
    x_0 = np.zeros((2,1))
    x = x_0

    X[0,:] = x[:,0]
    
    param = np.array([0.001, # L
                      1.0,   # R
                      0.001/0.0001, # K_p
                      0.001   # T_pi
                     ])    

    u = np.zeros((1,1))

    it = 0
    it_decimation = 0
    it_save = 0
    solver = 1
    for it in range(N_steps):   
        if t > 0.01:
            u[0] = 10.0
            
        t += dt        

        if solver == 1:
            # solver trapezoidal 1 step
            x = x + dt*(f_eval(t,u,x,0.0,param))
        
        if solver == 2:
            # solver trapezoidal 1 step
            f_1 = f_eval(t,u,x,0.0,param)
            x_1 = x + dt*f_1
            x = x + 0.5*dt*(f_eval(t,u,x,0.0,param)+ f_1)

        

        if it_decimation >= decimation:
            it_save += 1  
            X[it_save,:] = x[:,0]
            T[it_save,:] = t
            it_decimation = 0
            
        it_decimation += 1              
        
        
    return T,X
    

if __name__ == "__main__":
    
    test_model = 'pi'

    if test_model == 'pi':
        pi_1 = pi()
        
        t = 0.0
        u = np.zeros((2,1))
        x = np.zeros((1,1))
        y = 0.0    
        
        print(pi_1.f_eval(t,u,x,y))
        