#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 23:44:26 2017

@author: jmmauricio
"""

import matplotlib.pyplot as plt
import numba 
import numpy as np 
from numba import jitclass          # import the decorator
from numba import int32, float64 , int64, complex128   # import the types
import ctrl
import numpy.fft 
import json

class vsc(object):
    
    def __init__(self,json_file):
        
        json_data = open(json_file).read().replace("'",'"')
        data = json.loads(json_data)
        
        vscs = data['vsc']
        
        dt_vsc = np.dtype([('ctrl_mode', 'int32'),
                  ('S_base',np.float64), ('V_dc',np.float64),
                  ('K_v',np.float64),('K_ang',np.float64),('K_f',np.float64),('K_p',np.float64),('K_i',np.float64), ('nodes',np.int32,(4,)),
                  ('T_v',np.float64),('T_ang',np.float64),
                  ('v_abcn_0',np.complex128,(4,1)),('i_abcn_0',np.complex128,(4,1)),
                  ('v_abcn',np.complex128,(4,1)),('i_abcn',np.complex128,(4,1)),
                  ('S',np.complex128,(3,1)),
                  ('S_0',np.complex128,(3,1)),
                  ('x',np.float64,(4,1)),('f',np.float64,(4,1)),('h',np.float64,(3,1)),
                  ('DV_remote',np.float64),
                  ('a_i','float64'), ('b_i','float64'), ('c_i','float64'), ('d_i','float64'), ('e_i','float64'), 
                  ('a_d','float64'), ('b_d','float64'), ('c_d','float64'), ('d_d','float64'), ('e_d','float64'),
                  ('Rth_sink','float64'), ('Rth_c_igbt','float64'), ('Rth_c_diode','float64'), ('Rth_j_igbt','float64'), ('Rth_j_diode','float64'),
                  ('T_a','float64'), ('Cth_sink','float64'),  ('N_switch_sink','float64'),
                  ('p_igbt_abcn','float64',(4,1)), 
                  ('p_diode_abcn','float64',(4,1)),
                  ('T_j_igbt_abcn','float64',(4,1)), 
                  ('T_j_diode_abcn','float64',(4,1)),
                  ('T_sink','float64'),
                  ])
            
        
        vsc_list = []
        
        for item in vscs:
            vsc_list += [(item['ctrl_mode'],
                          item['s_n_kVA']*1000.0, item['V_dc'],
                          item['K_v'], item['K_ang'], item['K_f'], item['K_p'], item['K_i'], item['nodes_vknown'],
                          item['T_v'], item['T_ang'],
                          np.zeros((4,1)),np.zeros((4,1)),
                          np.zeros((4,1)),np.zeros((4,1)),
                          np.zeros((3,1)),
                          np.zeros((3,1)),
                          np.zeros((4,1)),np.zeros((4,1)),np.zeros((3,1)),  
                          0.0,
                          item['a_i'], item['b_i'], item['c_i'], item['d_i'], item['e_i'], 
                          item['a_d'], item['b_d'], item['c_d'], item['d_d'], item['e_d'],
                          item['Rth_sink'], item['Rth_c_igbt'], item['Rth_c_diode'], item['Rth_j_igbt'], item['Rth_j_diode'],
                          item['T_a'], item['Cth_sink'], item['N_switch_sink'],
                          np.zeros((4,1)),
                          np.zeros((4,1)),
                          np.zeros((4,1)),
                          np.zeros((4,1)),
                          0.0
                          )]
            
        self.vsc_list = vsc_list
        self.params_vsc = np.rec.array(vsc_list,dtype=dt_vsc)    



@numba.jit(nopython=True,cache=True)
def ctrl_vsc_phasor(t,mode,params,params_remote):
    '''
    
    Parameters
    ----------


    mode: int
        0: ini, 1:der, 2:out
    
    '''

    
    
# %% initialization    
    if mode == 0:  # ini
        for it in range(len(params)):
            V_abc_0 = params[it].v_abcn_0[0:3,:] # phase to neutral abc voltages (without neutral)
            I_abc_0 = params[it].i_abcn_0[0:3,:] # phase currents (without neutral)
            params[it].S_0[:] = V_abc_0*np.conj(I_abc_0) # phase complex power
        
    
 

# %% derivatives    
    if mode == 1:  # der
        
        for it in range(len(params)):
            
            S_base = params[it].S_base
            T_v = params[it].T_v
            T_ang = params[it].T_ang
            
            V_abc = params[it].v_abcn[0:3]   # phase to neutral abc voltages (without neutral)
            I_abc = params[it].i_abcn[0:3,:] # phase currents (without neutral)                    
            S = V_abc*np.conj(I_abc) # phase complex power
            params[it].S[:] = S[:]   
            
            I_abc_m = np.abs(I_abc)
            fp_abc = np.cos(np.angle(I_abc) - np.angle(V_abc))
            
            a_i = params[it].a_i
            b_i = params[it].b_i
            c_i = params[it].c_i
            d_i = params[it].d_i
            e_i = params[it].e_i
            
            a_d = params[it].a_d
            b_d = params[it].b_d
            c_d = params[it].c_d
            d_d = params[it].d_d
            e_d = params[it].e_d

            Rth_sink  = params[it].Rth_sink
            Rth_c_igbt = params[it].Rth_c_igbt
            Rth_c_diode  = params[it].Rth_c_diode
            Rth_j_igbt = params[it].Rth_j_igbt
            Rth_j_diode = params[it].Rth_j_diode
            T_a = params[it].T_a
            Cth_sink  = params[it].Cth_sink
                          
            I_abcn = params[it].i_abcn
            V_abcn = params[it].v_abcn
            m = (V_abcn*np.sqrt(2)/params[it].V_dc*2)[:]
            fp = np.cos(np.angle(I_abcn) - np.angle(V_abcn))[:]
            I_abcn_m = np.abs(params[it].i_abcn)[:]
            
            params[it].p_igbt_abcn[:]  = (a_i + (b_i + c_i*m*fp)*I_abcn_m + (d_i + e_i*m*fp)*I_abcn_m**2).real
            params[it].p_diode_abcn[:] = (a_d + (b_d + c_d*m*fp)*I_abcn_m + (d_d + e_d*m*fp)*I_abcn_m**2).real

            
            N_switch_sink = params[it].N_switch_sink
            p_igbt_total  = 2*np.sum(params[it].p_igbt_abcn)
            p_diode_total = 2*np.sum(params[it].p_diode_abcn)            
#            
                     
            params[it].T_j_igbt_abcn[:]   = params[it].T_sink + (Rth_c_igbt + Rth_j_igbt )*(params[it].p_igbt_abcn[:])
#            params[it].T_j_diode_abcn  = params[it].T_sink + (Rth_c_diode+ Rth_j_diode)*(params[it].p_diode_abcn)
            
            T_sink =  params[it].x[3:4,0]
            #print(T_sink[0])
            params[it].f[3:4,0] = 1.0/Cth_sink*(T_a + Rth_sink/N_switch_sink*(p_igbt_total + p_diode_total)-T_sink) # angle from frequency           
            
            
            params[it].T_sink =  params[it].x[3,0]  
            
            DV_remote = params[it].DV_remote

            if params.ctrl_mode[it] == 3:  # p-v, q-ang

                K_v = params[it].K_v 
                K_ang = params[it].K_ang
            

                          
                DS_abc = S-params[it].S_0*0 # complex power increment
                
                DP_abc = DS_abc.real
                DQ_abc = DS_abc.imag
  
                DV_m_ref =  -K_v*np.sum(DP_abc)/S_base
                Dang_ref =   K_ang*np.sum(DQ_abc)/S_base
            
                params[it].f[0:1,:] = 1.0/T_v*(DV_m_ref - params[it].x[0:1,:])  # voltage control dynamics
                params[it].f[1:2,:] = 1.0/T_ang*(Dang_ref - params[it].x[1:2,:])  # angle control dynamics
                params[it].f[2:3,:] = 0.0  # angle from frequency

            if params.ctrl_mode[it] == 4:  # i-v ruben
            
                K_p = params[it].K_p 
                K_i = params[it].K_i
                I_max = S_base/690.0
                
                for it_ph in range(3):
                    error = I_abc_m[it_ph,0] - I_max
                    DV = K_p * error + K_i*params[it].x[it_ph:(it_ph+1),0]
                    params[it].f[it_ph:(it_ph+1),:] = 0.0
                    if DV[0]>0.0:
                        params[it].f[it_ph:(it_ph+1),:] = error
                    if DV[0]>20.0:
                        params[it].f[it_ph:(it_ph+1),:] = 0.0                        
            

# %% out
    if mode == 3: # out

        
        
        for it in range(len(params)):
            
            params[it].v_abcn[:] = params[it].v_abcn_0[:]
            
            V_abc_0 = params[it].v_abcn_0[0:3,:] # phase to neutral abc voltages (without neutral)
            S_base = params[it].S_base

            DV_remote = params[it].DV_remote
#            print(DV_remote)
                        
            if params.ctrl_mode[it] == 1:

                params[it].v_abcn[:] = params[it].v_abcn_0[:] * (1+DV_remote)



            if params.ctrl_mode[it] == 3: # p-v, q-ang
                DV_m = params[it].x[0:1,:]

                params[it].v_abcn[:] = params[it].v_abcn_0[:] * (1+DV_m+DV_remote)
                
            if params.ctrl_mode[it] == 4:  # i-v ruben
            
                K_p = params[it].K_p 
                K_i = params[it].K_i
                I_abc = params[it].i_abcn[0:3,:] # phase currents (without neutral)                    
               
                I_abc_m = np.abs(I_abc)
                I_max = S_base/690.0
                
                for it_ph in range(3):
                    error = I_abc_m[it_ph,0] - I_max
                    DV = K_p * error + K_i*params[it].x[it_ph:(it_ph+1),0]
                    
                    if DV[0]>0.0:
                        if DV[0]>20.0:
                            DV[0]=20

#                        print(DV)
                    params[it].v_abcn[:] = params[it].v_abcn_0[:] * (1-DV/231+DV_remote)
                        
                                        

            V_abc = params[it].v_abcn[0:3]   # phase to neutral abc voltages (without neutral)
            I_abc = params[it].i_abcn[0:3,:] # phase currents (without neutral)
#            
#            
            S = V_abc*np.conj(I_abc) # phase complex power
            params[it].S[:] = S[:] 
  
                
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
class vsc_emag(object):
    
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
    ('vec', complex128), 
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
    ('vec', complex128[:,:]), 
    ('omega', float64),  
    ('h', float64[:,:])           
]
@jitclass(spec)
class sourceNph(object):
    
    def __init__(self):
              
        self.omega = 2.0*np.pi*50.0   
        self.h = np.zeros((1,1))
        self.vec = np.zeros((4,1),dtype=np.complex128) 
        
    def h_eval(self,t):
        
        omega = self.omega 
        
        N = self.vec.shape[0]
        self.h = np.zeros((N,1))
       
        for it in range(N):
            self.h[it]  = (self.vec[it,0]*np.exp(1j*omega*t)).real
            
        return self.h

@numba.jit()
def ph2inst(phasors,t,omega=2.0*np.pi*50.0):
    
    if type(t) == float:       
        N = phasors.shape[0]
        h = np.zeros((N,1))
       
        for it in range(N):
            h[it]  = phasors[it,0]*np.exp(1j*omega*t).real

    if type(t) ==  np.ndarray: 
        N_t   = t.shape[0]
        N_phs = phasors.shape[0]
        h = np.zeros((N_phs,N_t))
       
        for it_t in range(N_t):
            for it_ph in range(N_phs):
                h[it_ph,it_t]  = (phasors[it_ph,0]*np.exp(1j*omega*t[it_t])).real
            
    return h


@numba.jit(nopython=True)
def dft(x):
    ''' 
    10.1109/TPWRD.2010.2048764
    '''
    N = len(x)
    f = np.zeros((5,1))+0j
    

    n = 1000
    
    for k in range(5):
        for h in range(N-1):
            f[k] += x[n-N+1+h] * np.exp(- 1j * 2.0*np.pi * k * h / N)/N*2
        
    return f


@numba.jit(nopython=True)
def inst2ph(instant,t,Dt):
    
    t_new = np.arange(t[0],t[-1],Dt)
        
    return numpy.fft(instant)


spec = [
    ('N_states', int64),               # a simple scalar field
    ('x_0', float64[:,:]), 
    ('x', float64[:,:]), 
    ('e_abc', float64[:,:]), 
    ('e_dq0', float64[:,:]), 
    ('i_abc', float64[:,:]), 
    ('i_dq0', float64[:,:]), 
    ('f', float64[:,:]), 
    ('L', float64), 
    ('R', float64),  
    ('C', float64), 
    ('R_dc', float64), 
    ('v_dc_src',float64),
    ('h', float64[:,:]),  
    ('v_s_abc', float64[:,:]) ,   
    ('S_n', float64), 
    ('U_n', float64), 
    ('freq_n', float64)       
]
@jitclass(spec)
class rectifier_3ph(object):
    
    def __init__(self):
        
        self.N_states = 3
        self.x_0 = np.zeros((3,1))
               
        self.L = 1.0e-3
        self.R = 0.1
        self.R_dc = 0.1
        self.C = 1000.0e-6
        
        self.v_dc_src = 0.0
        
        self.v_s_abc = np.zeros((3,1))
        self.x = np.zeros((3,1))
        self.f = np.zeros((3,1))

        self.e_abc = np.zeros((3,1))
        self.e_dq0 = np.zeros((3,1))
        self.i_abc = np.zeros((3,1))
        self.i_dq0 = np.zeros((3,1))

        self.S_n = 1000.0
        self.U_n = 12.0
        self.freq_n = 100.0
        
    def f_eval(self,t):
        
        # parameters
        R_dc = self.R_dc
        L = self.L 
        R = self.R 
        C = self.C        
        v_dc_src = self.v_dc_src

        
        # states to variables
        self.i_dq0[0] = self.x[0,0]
        self.i_dq0[1] = self.x[1,0]
        self.i_dq0[2] = 0.0
        v_dc = self.x[2,0]
        
        # dq currents from states
        i_d = self.i_dq0[0]
        i_q = self.i_dq0[1]
        
        # abc currents from states
        self.i_abc = ctrl.iclark(self.i_dq0)
        i_a = self.i_abc[0,0]
        i_b = self.i_abc[1,0]  
        i_c = self.i_abc[2,0]    
 
        # grid system voltages
        v_s_dq0 = ctrl.clark(self.v_s_abc[0:3,:])
        v_s_d = v_s_dq0[0,0] 
        v_s_q = v_s_dq0[1,0]        
        
        # diode logic in abc
        u_a = 0.0
        if i_a >= 0.0: u_a = 1.0       
        
        u_b = 0.0
        if i_b >= 0.0: u_b = 1.0 
         
        u_c = 0.0
        if i_c >= 0.0: u_c = 1.0 
                       
        self.e_abc[0]  = (2*u_a - u_b - u_c)/3.0*v_dc
        self.e_abc[1]  = (2*u_b - u_c - u_a)/3.0*v_dc
        self.e_abc[2]  = (2*u_c - u_a - u_b)/3.0*v_dc
        
        # diode logic in dq0
        self.e_dq0 = ctrl.clark(self.e_abc)
        e_d = self.e_dq0[0]
        e_q = self.e_dq0[1]    
        
        
        di_d = 1.0/L*(v_s_d - R*i_d - e_d)
        di_q = 1.0/L*(v_s_q - R*i_q - e_q)
               
        i_dc_rect = u_a*i_a + u_b*i_b + u_c*i_c
    
        dv_dc = 1.0/C*(i_dc_rect - (v_dc-v_dc_src)/R_dc)
#        
        self.f[0] = di_d
        self.f[1] = di_q
        self.f[2] = dv_dc
    
        return self.f

    def design(self):
        
        S_n = self.S_n
        freq_n = self.freq_n
        Omega_n = 2.0*np.pi*freq_n
        U_rms = self.U_n
        
        X_pu = 0.1
        rend = 0.95
        H_c = 5.0e-3 # capacitor inertia constant
        
        V_dc_n = 2.0*U_rms   # nominal DC voltage
        
        Z_b = U_rms**2/S_n
        X = X_pu*Z_b
        self.L = X/Omega_n
        
        P_loss = (1.0-rend)*S_n
        I_n = S_n/(np.sqrt(3.0)*U_rms)
        self.R = P_loss/(3.0*I_n**2)

        self.C = 2.0*S_n*H_c/(V_dc_n**2)        

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
    ('v_s_abc', float64[:,:]),
    ('S_n', float64), 
    ('U_n', float64), 
    ('omega_r_n', float64), 
    ('freq_n', float64),  
    ('p_e', float64),
    ('i_dc', float64),
    ('p_dc', float64),
    ('v_dc', float64)
                   
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

        self.S_n = 1000.0
        self.U_n = 12.0
        self.freq_n = 100.0

        self.p_e = 0.0
        self.i_dc = 0.0
        self.p_dc = 0.0
        self.v_dc = 0.0
        
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
        i_dc = (v_dc-v_dc_src)/R_dc
    
        dv_dc = 1.0/C*(i_d - i_dc)
        dtheta_r = omega_r
        
        p_e = i_a*v_a + i_b*v_b + i_c*v_c
        
        tau_r = p_e/omega_r
#
        self.tau_r = tau_r
        self.p_e = p_e
        self.i_dc = i_dc
        self.p_dc = v_dc*i_dc
        self.v_dc = v_dc
        
        self.f[0] = di_a
        self.f[1] = di_b
        self.f[2] = di_c
        self.f[3] = dv_dc
        self.f[4] = dtheta_r
    
        return self.f
    

    def design(self):
        
        S_n = self.S_n
        freq_n = self.freq_n
        Omega_n = 2.0*np.pi*freq_n
        U_rms = self.U_n
        
        X_pu = 0.1
        rend = 0.95
        H_c = 5.0e-3 # capacitor inertia constant
        
        V_dc_n = 2.0*U_rms   # nominal DC voltage
        
        Z_b = U_rms**2/S_n
        X = X_pu*Z_b
        self.L = X/Omega_n
        
        P_loss = (1.0-rend)*S_n
        I_n = S_n/(np.sqrt(3.0)*U_rms)
        self.R = P_loss/(3.0*I_n**2)

        self.C = 2.0*S_n*H_c/(V_dc_n**2)      
        
        self.Phi =  U_rms*np.sqrt(2.0/3.0)/Omega_n
    
    

if __name__ == "__main__":
    
    test_model = 'vsc'
    
    if test_model == 'vsc':
        import ctrl
        secondary_obj = ctrl.secondary('cigre_lv_isolated.json')
        vsc_objs = vsc('cigre_lv_isolated.json')
        params_secondary = secondary_obj.params_secondary     
        params_vsc = vsc_objs.params_vsc
        
        ctrl_vsc_phasor(0.0,0,params_vsc,params_secondary)
        ctrl_vsc_phasor(0.0,1,params_vsc,params_secondary)
        ctrl_vsc_phasor(0.0,2,params_vsc,params_secondary)


            
        vsc1 = vsc('cigre_lv_isolated.json')
        vsc1 = vsc('cigre_lv_isolated.json')

    if test_model == 'inst2ph':
        t = np.linspace(0.0,0.02,1000)
        x = np.sin(2.0*np.pi*50*t)+0.2*np.sin(2.0*np.pi*150*t)
        print(dft(x))
        
        plt.plot(np.abs(dft(x)))
        plt.show()
        #inst2ph(x,t,0.02)
        
        
    if test_model == 'vsc_emag':
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

        

        V_v = np.array([
              [  2.31000000e+02+1j],
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
		
        sourceNph.vec = V_v
        print(sourceNph.h_eval(t))

    if test_model == 'ph2inst':

        
        t = np.linspace(0.0,0.02,100)

        

        V_v = np.array([
              [  2.31000000e+02+1j],
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
        print(ph2inst(V_v,t))
        
    if test_model == 'rectifier_3ph':
        V = np.zeros((3,1),dtype=np.complex128)
        V[0] = 325*np.exp(1j*np.deg2rad(0.0))   
        V[1] = 325*np.exp(1j*np.deg2rad(-120.0))   
        V[2] = 300*np.exp(1j*np.deg2rad(-240.0))   
        obj_s = sourceNph()
        obj_s.vec = V
        
        obj_r = rectifier_3ph()
        obj_r.v_s_abc = obj_s.h_eval(0.0)
        obj_r.f_eval(0.0)
        
    if test_model == 'pmsm_rectifier':
        obj_pr = pmsm_rectifier()
        obj_pr.f_eval(0.0)
        obj_pr.design()