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
import json

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
def park(abc,theta):
    pi = np.pi
    T_park =np.zeros((3,3))
    T_park[0,0:3] = 2.0/3.0*np.array([np.cos(theta), np.cos(theta-2.0/3.0*pi), np.cos(theta-4.0/3.0*pi)])
    T_park[1,0:3] = 2.0/3.0*np.array([np.sin(theta), np.sin(theta-2.0/3.0*pi), np.sin(theta-4.0/3.0*pi)])
    T_park[2,0:3] = 2.0/3.0*np.array([        0.5,                0.5,                0.5])

    dq0 = T_park @ abc
    
    return dq0

@numba.jit(nopython=True)
def ipark(dq0,theta):
    pi = np.pi
    T_inv_park =np.zeros((3,3))
    T_inv_park[0,0:3] = np.array([           np.cos(theta),  np.sin(theta),            1.0])
    T_inv_park[1,0:3] = np.array([np.cos(theta-2.0/3.0*pi),  np.sin(theta-2.0/3.0*pi), 1.0])
    T_inv_park[2,0:3] = np.array([np.cos(theta-4.0/3.0*pi),  np.sin(theta-4.0/3.0*pi), 1.0])
   
    abc = T_inv_park @ dq0
    
    return abc
    
 
pi = np.pi
T_clark =np.zeros((3,3))
T_clark[0,0:3] = 2.0/3.0*np.array([np.cos(0.0), np.cos(0.0-2.0/3.0*pi), np.cos(0.0-4.0/3.0*pi)])
T_clark[1,0:3] = 2.0/3.0*np.array([np.sin(0.0), np.sin(0.0-2.0/3.0*pi), np.sin(0.0-4.0/3.0*pi)])
T_clark[2,0:3] = 2.0/3.0*np.array([        0.5,                0.5,                0.5])


T_inv_clark =np.zeros((3,3))
T_inv_clark[0,0:3] = np.array([        np.cos(0.0),  np.sin(0.0),         1.0])
T_inv_clark[1,0:3] = np.array([np.cos(-2.0/3.0*pi),  np.sin(-2.0/3.0*pi), 1.0])
T_inv_clark[2,0:3] = np.array([np.cos(-4.0/3.0*pi),  np.sin(-4.0/3.0*pi), 1.0])
      
@numba.jit(nopython=True)
def clark(abc):
    if abc.shape[1] == 1:
        dq0 = T_clark @ abc  
    if abc.shape[1] > 1:
        N_samples = abc.shape[1]
        dq0 = np.zeros((3,N_samples))
        for it in range(N_samples):
            dq0[:,it]  = T_clark @ abc[:,it]
    return dq0

@numba.jit(nopython=True)
def iclark(dq0):      
        abc = T_inv_clark @ dq0        
        return abc
        

#@numba.jit(nopython=True)
def park2(abc,t,omega):
    if abc.shape[1] == 1:
        dq = np.zeros((2,1),dtype=np.complex128)
        ab0 = T_clark @ abc 
        aux = (ab0[0] +1j * ab0[1]) * np.exp(1j*omega*t)
        dq[0] = aux.real
        dq[1] = aux.imag
        
        
    if abc.shape[1] > 1:
        N_samples = abc.shape[1]
        dq = np.zeros((2,N_samples),dtype=np.complex128)
        for it in range(N_samples):
            ab0_v  = T_clark @ abc[:,it]
            #print(ab0_v)
            aux_v = (ab0_v[0] +1j * ab0_v[1]) * np.exp(1j*omega*t[it])
            dq[0,it] = aux_v.real
            dq[1,it] = aux_v.imag
    return dq
    
    
@numba.jit(nopython=True)
def interp(x,X,Y):
    
    idx = np.argmax(X>x)-1
    return (Y[idx+1] - Y[idx])/ (X[idx+1] - X[idx]) * (x - X[idx]) + Y[idx]    
    
    
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
    

class secondary(object):
    
    def __init__(self,json_file):
        
        json_data = open(json_file).read().replace("'",'"')
        data = json.loads(json_data)
        
        secondaries = data['secondary']
        
        dt_secondary = np.dtype([('ctrl_mode', 'int32'),('Dt_secondary', np.float64),('S_base', np.float64),
                       ('K_p_v',np.float64),('K_i_v',np.float64),('K_p_p',np.float64),('K_i_p',np.float64),
                       ('V',np.float64),('V_ref',np.float64),
                       ('DV',np.float64),
                       ('x',np.float64,(2,1)),('f',np.float64,(2,1))
                        ])
            
        
        secondary_list = []
        
        for item in secondaries:
            secondary_list += [(item['ctrl_mode'],item['Dt_secondary'],item['S_base'],
                          item['K_p_v'], item['K_i_v'],item['K_p_p'], item['K_i_p'],
                          231.0,231.0,
                          0.0,
                          np.zeros((2,1)),np.zeros((2,1))
                          )]
            
        self.secondary_list = secondary_list
        self.params_secondary = np.rec.array(secondary_list,dtype=dt_secondary)    
        
        
@numba.jit(nopython=True,cache=True)
def secondary_ctrl(t,mode,params, params_vsc):
    '''
    0: 'fix'
    1: 'v'
    2:  
    3:  
    4:  
    5:  
        
    DV: single reference     
    DV_remote: multiple references    
    '''
    
    ctrl_mode = params[0].ctrl_mode  
    S_base = params[0].S_base  
    K_p_v = params[0].K_p_v  
    K_i_v = params[0].K_i_v 
    K_p_p = params[0].K_p_p  
    K_i_p = params[0].K_i_p     
    V = params[0].V
    V_ref = params[0].V_ref
#    
    
    if mode == 1:
        
        if params[0].ctrl_mode == 0: # 'fix'
            params[0].DV = 0.0 
            

        if params[0].ctrl_mode == 1:  # 'v'

            V_mean = V
            error =  V_ref - V_mean
            params[0].f[0] = error
            
        if params[0].ctrl_mode == 2:  # 'p'
            S_total = np.sum(params_vsc.S_base)
            
            
            P_demand_est = 0.0
            for it in range(2):
                P_demand_est += np.sum(params_vsc[it].S[:]).real
            
            for it in range(2):
                P_ref = P_demand_est * params_vsc[it].S_base/S_total
                DP = P_ref - np.sum(params_vsc[it].S.real)          # from VSCs controllers
                   
                error = DP
                if error> 1000e3:
                    error = 1000e3
                if error< -1000e3:
                    error = -1000e3 

                params[0].f[it] = error                          

    if mode == 2:
        
        if params[0].ctrl_mode == 0: # 'fix'
            DV = 0.0
            
        if params[0].ctrl_mode == 1:  # 'v'
            
            V_mean = np.abs(params[0].V)
            error =  231.0 - V_mean
#            print(V_mean)
            DV = K_p_v*error + K_i_v * params[it].x[0,0]
                        
            params[0].DV = DV
            
            for it in range(2):           
                params_vsc[it].DV_remote = DV

        if params[0].ctrl_mode == 2:  # 'p'
            S_total = np.sum(params_vsc.S_base)
            P_demand_est = 0.0
            for it in range(2):
                P_demand_est += np.abs(np.sum(params_vsc[it].S[:]).real )

            for it in range(2):
                P_ref = P_demand_est * params_vsc[it].S_base/S_total
                DP = P_ref - np.sum(params_vsc[it].S.real)

                DV = K_p_p*DP/S_base + K_i_p/S_base*params[0].x[it,0]
                if DV >10.0:
                    DV =10.0
                    params[0].f[it] = 0.0
                if DV < -10.0:
                    DV = -10.0
                    params[0].f[it] = 0.0
                params_vsc[it].DV_remote = DV


                    
#                self.DV_array[it] = DV
        

         
         
if __name__ == "__main__":
    
    test_model = 'park2'

    if test_model == 'park2':
        abc = np.zeros((3,1000))
        park2(abc,np.zeros(1000),0.0)
        
    if test_model == 'park':
        abc = np.zeros((3,1))
        ipark(abc,0.0)
        
    if test_model == 'clark':
        import electric
        T = np.linspace(0.0,0.04,400)
        V_peak = 400.0*np.sqrt(2.0/3.0)
        V_a = V_peak*np.exp(1j*np.deg2rad(0.0))   
        V_b = V_peak*np.exp(1j*np.deg2rad(-120.0))   
        V_c = V_peak*np.exp(1j*np.deg2rad(-240.0)) 
        V_p = np.array([[V_a],[V_b],[V_c]]) # peak values
        V = V_p/np.sqrt(2.0)
        v_abc = electric.ph2inst(V_p,T)

        v_dq0 = park2(v_abc,T,2.0*np.pi*50)
        clark(v_abc)

    if test_model == 'pi':
        pi_1 = pi()
        
        t = 0.0
        u = np.zeros((2,1))
        x = np.zeros((1,1))
        y = 0.0    
        
        print(pi_1.f_eval(t,u,x,y))
        