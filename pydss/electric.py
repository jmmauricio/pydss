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


spec = [
    ('N_states', int32),               # a simple scalar field
    ('x_0', float64[:,:]), 
    ('f', float64[:,:]), 
    ('h', float64[:,:]), 
    ('R', float64), 
    ('L', float64),           
]
@jitclass(spec)
class rl(object):
    
    def __init__(self):
        
        self.N_states = 1
        self.x_0 = np.zeros((1,1))
        self.f = np.zeros((1,1))        
        self.h = np.zeros((1,1))
        
        self.R = 1.0
        self.L = 0.001

        
    def f_eval(self,t,u,x,y):
        
        i_l = x[0]
        
        R = self.R
        L = self.L
    
        v = u[0]
    
        di_l = 1.0/L*(v - R*i_l)
        
        self.f[0] = di_l
    
        return self.f


    def h_eval(self,t,u,x,y):
        
        i_l = x[0]
    
        self.h[0] = i_l
    
        return self.h


spec = [
    ('N_states', int32),               # a simple scalar field
    ('PWM_mode', int32), 
    ('R', float64), 
    ('L', float64),   
    ('C', float64),  
    ('x_0', float64[:,:]),
    ('x', float64[:,:]), 
    ('f', float64[:,:]), 
    ('h', float64[:,:]), 
    ('T_clark', float64[:,:]),    
    ('T_inv_clark', float64[:,:]),
    ('eta_abc', float64[:,:])
]

@jitclass(spec)
class vsc(object):
    
    def __init__(self):
        
        S_n = 30.0e3
        freq = 50.0
        Omega = 2.0*np.pi*freq
        U_rms = 400.0      
        X_pu = 0.1
        rend = 0.95
        H_c = 5.0e-3 # capacitor inertia constant
        
        V_dc_n = 2.0*U_rms   # nominal DC voltage
        
        Z_b = U_rms**2/S_n
        X = X_pu*Z_b
        L = X/Omega
        
        P_loss = (1.0-rend)*S_n
        I_n = S_n/(np.sqrt(3.0)*U_rms)
        R = P_loss/(3.0*I_n**2)

        C = 2.0*S_n*H_c/(V_dc_n**2)
        
        self.N_states = 3
        self.PWM_mode = 3
        self.x_0 =np.zeros((3,1))
        self.x_0[0] = 0.0
        self.x_0[1] = 0.0
        self.x_0[2] = V_dc_n               
    

    
        self.x = self.x_0        
        self.L = L
        self.R = R
        self.C = C
        
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
        
        i_d  = x[0]
        i_q  = x[1]
        v_dc = x[2]
        
        R = self.R
        L = self.L
        C = self.C
        
        Dt = 50.0e-6

        if self.PWM_mode == 0:

            self.eta_abc[0] = u[0,:]
            self.eta_abc[1] = u[1,:]
            self.eta_abc[2] = u[2,:]
            
        if self.PWM_mode > 0:
            self.pwm(t,u)
                
                
            
        eta_dq = self.T_clark @ self.eta_abc
        eta_d = eta_dq[0]
        eta_q = eta_dq[1]

        
        v_s_dq = self.T_clark @ u[3:6,:]
        v_s_d = v_s_dq[0]
        v_s_q = v_s_dq[1]       
        
        i_dc = u[6,0]
    
        di_d = 1.0/L*(0.5*eta_d*v_dc - R*i_d - v_s_d)
        di_q = 1.0/L*(0.5*eta_q*v_dc - R*i_q - v_s_q)
        dv_dc = 1.0/C*( -3.0/4.0*(eta_d*i_d + eta_q*i_q) + i_dc )
        
        self.f[0] = di_d
        self.f[1] = di_q
        self.f[2] = dv_dc       
        
        return self.f


    def h_eval(self,t,u,x,y):
        
        i_dq0 = np.zeros((3,1))

        i_dq0[0]  = x[0]
        i_dq0[1]  = x[1]    
        i_dq0[2]  = 0.0
    
        i_abc = self.T_inv_clark @ i_dq0
        v_dc = x[2]

        self.h[0:3] = i_abc
        self.h[3] = v_dc        
        self.h[4:7] = self.eta_abc
        
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


spec = [
    ('V_peak', float64), 
    ('omega', float64),  
    ('h', float64[:,:])           
]
@jitclass(spec)
class source3ph(object):
    
    def __init__(self):
        
        U_rms = 400.0
        
        self.V_peak = np.sqrt(2.0/3.0)*U_rms
        self.omega = 2.0*np.pi*50.0
        
        self.h = np.zeros((3,1))
        
    def h_eval(self,t):
        
        V_peak = self.V_peak
        omega = self.omega 
        
        v_s_a = V_peak*np.sin(omega*t)
        v_s_b = V_peak*np.sin(omega*t - 2.0/3.0*np.pi)
        v_s_c = V_peak*np.sin(omega*t - 4.0/3.0*np.pi)
        
        self.h[0] = v_s_a
        self.h[1] = v_s_b        
        self.h[2] = v_s_c        

        return self.h

     
spec = [
    ('V_peak', float64), 
    ('omega', float64),  
    ('h', float64[:,:])           
]
@jitclass(spec)
class sourceNph(object):
    
    def __init__(self):
        
        U_rms = 400.0
        
        self.V_peak = np.sqrt(2.0/3.0)*U_rms
        self.omega = 2.0*np.pi*50.0
        
        self.h = np.zeros((1,1))
        
    def h_eval(self,t,vec):
        
        V_peak = self.V_peak
        omega = self.omega 
        
        N = vec.shape[0]
        self.h = np.zeros((N,1))
        for it in range(N):

            V_peak = np.abs(vec[it,0])
            angle = np.angle(vec[it,0])
            v_s_a = V_peak*np.sin(omega*t + angle)
            
            self.h[it] = v_s_a     

        return self.h



spec = [
    ('N_states', int64),               # a simple scalar field
    ('x_0', float64[:,:]), 
    ('x', float64[:,:]), 
    ('f', float64[:,:]), 
    ('L', float64), 
    ('R', float64),  
    ('C', float64), 
    ('R_dc', float64), 
    ('v_dc_src',float64),
    ('h', float64[:,:]),  
    ('v_s_abc', float64[:,:])           
]
@jitclass(spec)
class rectifier_3ph(object):
    
    def __init__(self):
        
        self.N_states = 4
        self.x_0 = np.zeros((4,1))
               
        self.L = 5.0e-3
        self.R = 0.1
        self.R_dc = 10.0
        self.C = 2200.0e-6
        
        self.v_dc_src = 0.0
        
        self.v_s_abc = np.zeros((3,1))
        self.x = np.zeros((4,1))
        self.f = np.zeros((4,1))

        
    def f_eval(self,t):
    
        i_a = self.x[0,0]
        i_b = self.x[1,0]
        i_c = self.x[2,0]
        v_dc = self.x[3,0]
        
        R_dc = self.R_dc
        L = self.L 
        R = self.R 
        C = self.C 
        
        v_s_a = self.v_s_abc[0]
        v_s_b = self.v_s_abc[1]
        v_s_c = self.v_s_abc[2]
        
        v_dc_src = self.v_dc_src
        
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
    
        dv_dc = 1.0/C*(i_d - (v_dc-v_dc_src)/R_dc)
#        
        self.f[0] = di_a
        self.f[1] = di_b
        self.f[2] = di_c
        self.f[3] = dv_dc
    
        return self.f



spec = [
    ('N_states', int64),               # a simple scalar field
    ('x_0', float64[:,:]), 
    ('x', float64[:,:]), 
    ('f', float64[:,:]), 
    ('L', float64), 
    ('R', float64),  
    ('C', float64), 
    ('R_dc', float64), 
    ('v_dc_src',float64),
    ('Phi',float64),
    ('N_pp',float64),
    ('omega_r',float64),
    ('tau_r',float64),
    ('h', float64[:,:]),  
    ('v_s_abc', float64[:,:])           
]
@jitclass(spec)
class pmsm_rectifier(object):
    
    def __init__(self):
        
        self.N_states = 5
        self.x_0 = np.zeros((5,1))
               
        self.L = 5.0e-3
        self.R = 0.1
        self.R_dc = 10.0
        self.C = 2200.0e-6
        self.Phi =  1.0
        self.N_pp =  4.0
        
        self.omega_r = 100*np.pi/self.N_pp 
        self.tau_r = 0.0
        
        self.v_s_abc = np.zeros((3,1))
        self.x = np.zeros((5,1))
        self.f = np.zeros((5,1))

        
    def f_eval(self,t):
    
        i_a = self.x[0,0]
        i_b = self.x[1,0]
        i_c = self.x[2,0]
        v_dc = self.x[3,0]
        theta_r = self.x[4,0]
        
        R_dc = self.R_dc
        L = self.L 
        R = self.R 
        C = self.C 
        Phi = self.Phi
        N_pp = self.N_pp 
        omega_r = self.omega_r 
        
        
        v_a = Phi*N_pp*omega_r*np.sin(N_pp*theta_r)
        v_b = Phi*N_pp*omega_r*np.sin(N_pp*theta_r - 2.0/3.0*np.pi)
        v_c = Phi*N_pp*omega_r*np.sin(N_pp*theta_r - 4.0/3.0*np.pi)
        
        v_dc_src = self.v_dc_src
        
        u_a = 0.0
        if i_a >= 0.0: u_a = 1.0       
        
        u_b = 0.0
        if i_b >= 0.0: u_b = 1.0 
         
        u_c = 0.0
        if i_c >= 0.0: u_c = 1.0 
                       
        e_a = (2*u_a - u_b - u_c)/3.0*v_dc
        e_b = (2*u_b - u_c - u_a)/3.0*v_dc
        e_c = (2*u_c - u_a - u_b)/3.0*v_dc
    
        di_a = 1.0/L*(v_a - R*i_a - e_a)
        di_b = 1.0/L*(v_b - R*i_b - e_b)
        di_c = 1.0/L*(v_c - R*i_c - e_c)
    
        i_d = u_a*i_a + u_b*i_b + u_c*i_c
    
        dv_dc = 1.0/C*(i_d - (v_dc-v_dc_src)/R_dc)
        dtheta_r = omega_r
        
        p_e = i_a*v_a + i_b*v_b + i_c*v_c
        
        tau_r = p_e/omega_r
#
        self.tau_r = tau_r
        
        self.f[0] = di_a
        self.f[1] = di_b
        self.f[2] = di_c
        self.f[3] = dv_dc
        self.f[4] = dtheta_r
    
        return self.f
    
    
    

if __name__ == "__main__":
    
    test_model = 'pmsm_rectifier'
    
    if test_model == 'vsc':
        vsc = vsc()
        
        t = 250.0e-6
        u = np.zeros((7,1))
        u[0] = 0
        u[1] = 0
        u[2] = 0
        x = np.zeros((3,1))
        y = 0.0    
#        
        print(vsc.f_eval(t,u,x,y))
        print(vsc.h_eval(t,u,x,y))
        
    if test_model == 'source3ph':
        source3ph = source3ph()
        
        t = 0.0
        u = np.zeros((7,1))
        x = np.zeros((3,1))
#        y = 0.0    
#        
        print(source3ph.h_eval(t))


    if test_model == 'sourceNph':
        sourceNph = sourceNph()
        
        t = 0.1
        u = np.zeros((7,1))
        x = np.zeros((3,1))
#        y = 0.0    

        

        V_v = np.array([[  2.31000000e+02+1j],
               [ -1.15500000e+02],
               [ -1.15500000e+02],
               [  4.26325641e-14],
               [  2.28708695e+02],
               [ -1.14133231e+02],
               [ -1.15984269e+02],
               [ -3.08695462e-01],
               [  2.19043477e+02],
               [ -1.04086027e+02],
               [ -1.22001473e+02],
               [ -1.54347731e+00],
               [  2.10213259e+02],
               [ -9.42198491e+01],
               [ -1.19152651e+02],
               [ -2.04325916e+00]])
        print(sourceNph.h_eval(t,V_v))
        
    if test_model == 'rectifier_3ph':
        obj_s = source3ph()
        obj_r = rectifier_3ph()
        obj_r.f_eval(0.0)
        
    if test_model == 'pmsm_rectifier':
        obj_pr = pmsm_rectifier()
        obj_pr.f_eval(0.0)