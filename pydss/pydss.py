#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 13:04:45 2017

@author: jmmauricio
"""

import numpy as np 
import numba
from numba import float64,int32,int64,complex128
import json
import re
import time
from pf import pf_eval
from electric import vsc,ctrl_vsc_phasor
from ctrl import secondary,secondary_ctrl

#lines = [
#        {'bus_j': 'R1',  'bus_k': 'R2',  'code': 'UG1', 'm': 35.0, },
#        {'bus_j': 'R2',  'bus_k': 'R3',  'code': 'UG1', 'm': 35.0, },
#        {'bus_j': 'R3',  'bus_k': 'R4',  'code': 'UG1', 'm': 35.0, },
#        {'bus_j': 'R4',  'bus_k': 'R5',  'code': 'UG1', 'm': 35.0, },
#        {'bus_j': 'R5',  'bus_k': 'R6',  'code': 'UG1', 'm': 35.0, },
#        {'bus_j': 'R6',  'bus_k': 'R7',  'code': 'UG1', 'm': 35.0, },
#        {'bus_j': 'R7',  'bus_k': 'R8',  'code': 'UG1', 'm': 35.0, },
#        {'bus_j': 'R8',  'bus_k': 'R9',  'code': 'UG1', 'm': 35.0, },
#        {'bus_j': 'R9',  'bus_k': 'R10', 'code': 'UG1', 'm': 35.0, },
#        {'bus_j': 'R3',  'bus_k': 'R11', 'code': 'UG3', 'm': 30.0, },
#        {'bus_j': 'R4',  'bus_k': 'R12', 'code': 'UG3', 'm': 35.0, },
#        {'bus_j': 'R12', 'bus_k': 'R13', 'code': 'UG3', 'm': 35.0, },
#        {'bus_j': 'R13', 'bus_k': 'R14', 'code': 'UG3', 'm': 35.0, },
#        {'bus_j': 'R14', 'bus_k': 'R15', 'code': 'UG3', 'm': 30.0, },
#        {'bus_j': 'R6',  'bus_k': 'R16', 'code': 'UG3', 'm': 30.0, },
#        {'bus_j': 'R9',  'bus_k': 'R17', 'code': 'UG3', 'm': 30.0, },
#        {'bus_j': 'R10', 'bus_k': 'R18', 'code': 'UG3', 'm': 30.0, },
#        ]
#loads = [
#        {'bus': 'R11', 'kVA': 15.0, 'fp': 0.95, 'type':'3P+N'},
#        {'bus': 'R15', 'kVA': 52.0, 'fp': 0.95, 'type':'3P+N'},
#        {'bus': 'R16', 'kVA': 55.0, 'fp': 0.95, 'type':'3P+N'},
#        {'bus': 'R17', 'kVA': 35.0, 'fp': 0.95, 'type':'3P+N'},
#        {'bus': 'R18', 'kVA': 47.0, 'fp': 0.95, 'type':'3P+N'},  
#        ]
#buses = [
#        {'bus': 'R1',  'pos_x':   0, 'pos_y': -0*35, 'units':'m'},
#        {'bus': 'R2',  'pos_x':   0, 'pos_y': -1*35, 'units':'m'},
#        {'bus': 'R3',  'pos_x':   0, 'pos_y': -2*35, 'units':'m'},
#        {'bus': 'R4',  'pos_x':   0, 'pos_y': -3*35, 'units':'m'},
#        {'bus': 'R5',  'pos_x':   0, 'pos_y': -4*35, 'units':'m'},
#        {'bus': 'R6',  'pos_x':   0, 'pos_y': -5*35, 'units':'m'},
#        {'bus': 'R7',  'pos_x':   0, 'pos_y': -6*35, 'units':'m'},
#        {'bus': 'R8',  'pos_x':   0, 'pos_y': -7*35, 'units':'m'},
#        {'bus': 'R9',  'pos_x':   0, 'pos_y': -8*35, 'units':'m'},
#        {'bus': 'R10', 'pos_x':   0, 'pos_y': -9*35, 'units':'m'},
#        {'bus': 'R11', 'pos_x': -30, 'pos_y': -2*35, 'units':'m'},
#        {'bus': 'R12', 'pos_x':  35, 'pos_y': -3*35, 'units':'m'},
#        {'bus': 'R13', 'pos_x':  70, 'pos_y': -3*35, 'units':'m'},
#        {'bus': 'R14', 'pos_x': 105, 'pos_y': -3*35, 'units':'m'},
#        {'bus': 'R15', 'pos_x': 105, 'pos_y': -4*35, 'units':'m'},
#        {'bus': 'R16', 'pos_x': -35, 'pos_y': -5*35, 'units':'m'},
#        {'bus': 'R17', 'pos_x':  30, 'pos_y': -8*35, 'units':'m'},
#        {'bus': 'R18', 'pos_x': -30, 'pos_y': -9*35, 'units':'m'}
#        ]
#v_sources = [
#        {'bus': 'R1', 'kV':[00.22692174980229055]*3+[0], 'deg':[0,120,240,0], 'pos_x':0.0, 'pos_y':0}
#        ]
#adeg = 120.0
#vscs = [
#         {'bus':1, 'type':'v','kV':[0.231]*3+[0], 'deg':[0,adeg,adeg*2,0], 'pos_x':0.0, 'pos_y':0},
#         {'bus':2, 'type':'i','A':[0.0]*3,'fp':[0.8]*3, 'pos_x':0.2, 'pos_y':0.0},
#         {'bus':3, 'type':'i','A':[0.0]*3,'fp':[1.0]*3, 'pos_x':0.4, 'pos_y':0},
#         {'bus':4, 'type':'i','A':[0.0]*3,'fp':[1.0]*3, 'pos_x':0.4, 'pos_y':-0.2},  
#         {'bus':5, 'type':'i','A':[-10.0]*3,'fp':[1.0]*3, 'pos_x':0.6, 'pos_y':0},   
#        ]

line_codes = {'OH1':[[0.540 + 0.777j, 0.049 + 0.505j, 0.049 + 0.462j, 0.049 + 0.436j],
                      [0.049 + 0.505j, 0.540 + 0.777j, 0.049 + 0.505j, 0.049 + 0.462j],
                      [0.049 + 0.462j, 0.049 + 0.505j, 0.540 + 0.777j, 0.049 + 0.505j],
                      [0.049 + 0.436j, 0.049 + 0.462j, 0.049 + 0.505j, 0.540 + 0.777j]],
              'OH2':[[1.369 + 0.812j, 0.049 + 0.505j, 0.049 + 0.462j, 0.049 + 0.436j], 
                     [0.049 + 0.505j, 1.369 + 0.812j, 0.049 + 0.505j, 0.049 + 0.462j], 
                     [0.049 + 0.462j, 0.049 + 0.505j, 1.369 + 0.812j, 0.049 + 0.505j], 
                     [0.049 + 0.436j, 0.049 + 0.462j, 0.049 + 0.505j, 1.369 + 0.812j]],
              'OH3':[[2.065 + 0.825j, 0.049 + 0.505j, 0.049 + 0.462j, 0.049 + 0.436j], 
                     [0.049 + 0.505j, 2.065 + 0.825j, 0.049 + 0.505j, 0.049 + 0.462j], 
                     [0.049 + 0.462j, 0.049 + 0.505j, 2.065 + 0.825j, 0.049 + 0.505j], 
                     [0.049 + 0.436j, 0.049 + 0.462j, 0.049 + 0.505j, 2.065 + 0.825j]], 
              'UG1':[[0.211 + 0.747j, 0.049 + 0.673j, 0.049 + 0.651j, 0.049 + 0.673j], 
                     [0.049 + 0.673j, 0.211 + 0.747j, 0.049 + 0.673j, 0.049 + 0.651j], 
                     [0.049 + 0.651j, 0.049 + 0.673j, 0.211 + 0.747j, 0.049 + 0.673j], 
                     [0.049 + 0.673j, 0.049 + 0.651j, 0.049 + 0.673j, 0.211 + 0.747j]],
              'UG2':[[0.314 + 0.762j, 0.049 + 0.687j, 0.049 + 0.665j, 0.049 + 0.687j], 
                     [0.049 + 0.687j, 0.314 + 0.762j, 0.049 + 0.687j, 0.049 + 0.665j], 
                     [0.049 + 0.665j, 0.049 + 0.687j, 0.314 + 0.762j, 0.049 + 0.687j], 
                     [0.049 + 0.687j, 0.049 + 0.665j, 0.049 + 0.687j, 0.314 + 0.762j]], 
              'UG3':[[0.871 + 0.797j, 0.049 + 0.719j, 0.049 + 0.697j, 0.049 + 0.719j], 
                     [0.049 + 0.719j, 0.871 + 0.797j, 0.049 + 0.719j, 0.049 + 0.697j], 
                     [0.049 + 0.697j, 0.049 + 0.719j, 0.871 + 0.797j, 0.049 + 0.719j], 
                     [0.049 + 0.719j, 0.049 + 0.697j, 0.049 + 0.719j, 0.871 + 0.797j]],
              'EQU':[[0.871 + 0.797j, 0.049 + 0.719j, 0.049 + 0.719j, 0.049 + 0.719j], 
                     [0.049 + 0.719j, 0.871 + 0.797j, 0.049 + 0.719j, 0.049 + 0.719j], 
                     [0.049 + 0.719j, 0.049 + 0.719j, 0.871 + 0.797j, 0.049 + 0.719j], 
                     [0.049 + 0.719j, 0.049 + 0.719j, 0.049 + 0.719j, 0.871 + 0.797j]],
              'TR1':[[0.0032+0.0128j, 0.000j, 0.000j, 0.000j], 
                     [0.000j, 0.0032+0.0128j, 0.000j, 0.000j], 
                     [0.000j, 0.000j, 0.0032+0.0128j, 0.000j],  
                     [0.000j, 0.000j, 0.000j, 0.0032+0.0128j]],
              'PN1':[[0.314 + 0.762j, 0.049 + 0.687j], 
                     [0.049 + 0.687j, 0.314 + 0.762j]],
              'NN1':[[0.871 + 0.797j, 0.049 + 0.719j, 0.049 + 0.719j], 
                     [0.049 + 0.719j, 0.871 + 0.797j, 0.049 + 0.719j], 
                     [0.049 + 0.719j, 0.049 + 0.719j, 0.871 + 0.797j]]}

class pydss(object):
    '''
    
    
    
    P+N : 1
    3P  : 2
    3P+N: 3
    
    '''
    
    
    def __init__(self,json_file):
        
        self.json_file = json_file
        self.json_data = open(json_file).read().replace("'",'"')
        data = json.loads(self.json_data)
        self.data = data
        
        # power flow options
        self.max_iter = 20
        
        # run options
        self.N_steps = 1000
        self.Dt = 10.0e-3
        self.Dt_out = 0.01
        
        
        
        lines = data['lines']
        loads = data['loads']
        v_sources = data['v_sources']
        buses = data['buses']
        
        self.lines = lines
        self.loads = loads
        self.buses = buses
        
        N_nodes_default = 4
        nodes = []
        A_n_cols = 0
        it_col = 0
        v_sources_nodes = []

        
        N_v_known = 0
        ## Known voltages
        V_known_list = []
        for v_source in v_sources:
            v_source_nodes = []
            if not 'bus_nodes' in v_source:   # if nodes are not declared, default nodes are created
                v_source.update({'bus_nodes': list(range(1,N_nodes_default+1))})
            for item in  v_source['bus_nodes']: # the list of nodes '[<bus>.<node>.<node>...]' is created 
                node = '{:s}.{:s}'.format(v_source['bus'], str(item))
                if not node in nodes: 
                    nodes +=[node]
                v_source_nodes += [N_v_known]
                N_v_known += 1
            for volt,ang in  zip(v_source['kV'],v_source['deg']): # known voltages list is created 
                V_known_list += [1000.0*volt*np.exp(1j*np.deg2rad(ang))]
            v_sources_nodes += [v_source_nodes]    # global nodes for each vsources update  
        V_known = np.array(V_known_list).reshape(len(V_known_list),1) # known voltages list numpy array
        self.v_sources_nodes = v_sources_nodes

        ## Known currents
        S_known_list = []
        pq_3pn_int_list = []
        pq_3pn_list = []
        pq_3p_int_list = []
        pq_3p_list = []
        pq_pn_int_list = []
        pq_pn_list = []
        it_node_i = 0
        for load in loads:
            if not 'bus_nodes' in load:   # if nodes are not declared, default nodes are created
                load.update({'bus_nodes': list(range(1,N_nodes_default+1))})
            for item in  load['bus_nodes']: # the list of nodes '[<bus>.<node>.<node>...]' is created 
                node = '{:s}.{:s}'.format(load['bus'], str(item))
                if not node in nodes: nodes +=[node] 
                
            if load['type'] == '3P+N':
                pq_3pn_int_list += [list(it_node_i + np.array([0,1,2,3]))]
                it_node_i += 4
                if 'kVA' in load:
                    if type(load['kVA']) == float:
                        S = -1000.0*load['kVA']*np.exp(1j*np.arccos(load['fp'])*np.sign(load['fp']))
                        pq_3pn_list += [[S/3,S/3,S/3]]
                    if type(load['kVA']) == list:
                        pq = []
                        for s,fp in zip(load['kVA'],load['fp']):                            
                            pq += [-1000.0*s*np.exp(1j*np.arccos(fp)*np.sign(fp))]
                        pq_3pn_list += [pq]

                        
                        
            if load['type'] == '3P':
                pq_3p_int_list += [list(it_node_i + np.array([0,1,2,3]))]
                it_node_i += 4
                if 'kVA' in load:
                    if type(load['kVA']) == float:
                        pq_3p_list += [[-1000.0*load['kVA']*np.exp(1j*np.arccos(load['fp'])*np.sign(load['fp']))]]

#            if load['type'] == 'P+N':
#                p_node = load['bus_nodes']
#                pq_pn_int_list += [list(it_node_i + np.array([0,1,2,3]))]
#                it_node_i += 4
#                if 'kVA' in load:
#                    if type(load['kVA']) == float:
#                        pq_3p_list += [[1000.0*load['kVA']*np.exp(1j*np.arccos(load['fp'])*np.sign(load['fp']))]]
#                        
                        
#            for kVA,fp in  zip(load['kVA'],load['fp']): # known complex power list 
#                S_known_list += [1000.0*kVA*np.exp(1j*np.arccos(fp)*np.sign(fp))]
        pq_3pn_int = np.array(pq_3pn_int_list) # known complex power list to numpy array
        pq_3pn = np.array(pq_3pn_list) # known complex power list to numpy array
        pq_3p_int = np.array(pq_3p_int_list) # known complex power list to numpy array
        pq_3p = np.array(pq_3p_list) # known complex power list to numpy array
        
        for line in lines:
            N_conductors = len(line_codes[line['code']])
            A_n_cols += N_conductors
            if not 'bus_j_nodes' in line:   # if nodes are not declared, default nodes are created
                line.update({'bus_j_nodes': list(range(1,N_nodes_default+1))})
            if not 'bus_k_nodes' in line:   # if nodes are not declared, default nodes are created
                line.update({'bus_k_nodes': list(range(1,N_nodes_default+1))})

            for item in  line['bus_j_nodes']: # the list of nodes '[<bus>.<node>.<node>...]' is created 
                node_j = '{:s}.{:s}'.format(line['bus_j'], str(item))
                if not node_j in nodes: nodes +=[node_j]
            for item in  line['bus_k_nodes']: # the list of nodes '[<bus>.<node>.<node>...]' is created 
                node_k = '{:s}.{:s}'.format(line['bus_k'], str(item))
                if not node_k in nodes: nodes +=[node_k]

        N_nodes = len(nodes)

        A = np.zeros((N_nodes,A_n_cols))

        it_col = 0
        Z_line_list =  []
        for line in lines:
            for item in  line['bus_j_nodes']: # the list of nodes '[<bus>.<node>.<node>...]' is created 
                node_j = '{:s}.{:s}'.format(line['bus_j'], str(item))
                row = nodes.index(node_j)
                col = it_col
                A[row,col] = 1
    
    #            for item in  line['bus_k_nodes']: # the list of nodes '[<bus>.<node>.<node>...]' is created 
                node_k = '{:s}.{:s}'.format(line['bus_k'], str(item))
                row = nodes.index(node_k)
                col = it_col
                A[row,col] = -1
                it_col +=1   

            Z_line_list += [line['m']*0.001*np.array(line_codes[line['code']])]   # Line code to list of Z lines

        Y_lines = self.diag_2d_inv(Z_line_list)
        A_v = A[0:N_v_known,:]   
        N_nodes_i = N_nodes-N_v_known
        A_i = A[N_v_known:(N_v_known+N_nodes_i),:] 

        self.A = A
        self.nodes = nodes
        self.N_nodes = len(nodes)
        self.N_nodes_i = N_nodes_i
        self.N_nodes_v = self.N_nodes  - N_nodes_i
        self.A_v = A_v
        self.A_i = A_i
        
        self.Y = A @ Y_lines @ A.T
        self.Y_lines = Y_lines
        self.Y_ii = A_i @ Y_lines @ A_i.T
        self.Y_iv = A_i @ Y_lines @ A_v.T
        self.Y_vv = A_v @ Y_lines @ A_v.T
        self.Y_vi = A_v @ Y_lines @ A_i.T

        self.inv_Y_ii = np.linalg.inv(self.Y_ii)
        self.pq_3pn_int = pq_3pn_int
        self.pq_3pn = pq_3pn
        self.pq_3p_int = pq_3p_int
        self.pq_3p = pq_3p
        self.nodes = nodes
        self.V_known = V_known
        self.Y_lines = Y_lines
        

        
    def pf_eval(self):
        
        V_unknown_0 = np.zeros((self.N_nodes_i,1),dtype=np.complex128)+231 
        self.I_node = np.vstack((np.zeros((self.N_nodes_v,1)),
                             np.zeros((self.N_nodes_i,1))))+0j
        self.V_node = np.vstack((self.V_known,V_unknown_0 ))
        
        for it in range(int(self.N_nodes_i/4)): # change if not 4 wires
            
            V_unknown_0[4*it+0] = self.V_known[0]
            V_unknown_0[4*it+1] = self.V_known[1]
            V_unknown_0[4*it+2] = self.V_known[2]
            V_unknown_0[4*it+3] = 0.0
            
        N_i = self.N_nodes_i
        N_v = self.N_nodes_v 

        
        dt_pf = np.dtype([
                  ('Y_vv',np.complex128,(N_v,N_v)),('Y_iv',np.complex128,(N_i,N_v)),('inv_Y_ii',np.complex128,(N_i,N_i)),
                  ('I_node',np.complex128,(N_v+N_i,1)),('V_node',np.complex128,(N_v+N_i,1)),
                  ('pq_3pn_int',np.int32,self.pq_3pn_int.shape),('pq_3pn',np.complex128,self.pq_3pn.shape),
                  ('N_nodes_v',np.int32),('N_nodes_i',np.int32)] )
    
        params_pf = np.rec.array([(
                                self.Y_vv,self.Y_iv,self.inv_Y_ii,
                                self.I_node,self.V_node,
                                self.pq_3pn_int,self.pq_3pn,
                                self.N_nodes_v,self.N_nodes_i)],dtype=dt_pf)  
                  
        V_node,I_node = pf_eval(params_pf) 

        self.V_node = V_node
        self.I_node = I_node 
        self.params_pf = params_pf 


    def run_eval(self):

            
            secondary_obj = secondary(self.json_file)
            vsc_objs = vsc(self.json_file)

            params_secondary = secondary_obj.params_secondary
            self.params_secondary = params_secondary           
            
            params_vsc = vsc_objs.params_vsc
            self.params_vsc = params_vsc
            
            self.params_secondary = params_secondary
            
            Dt = self.Dt
            Dt_out = self.Dt_out
            
            N_nodes = self.N_nodes
            N_steps =  self.N_steps
            N_outs = int(N_steps*Dt/Dt_out)
            
            dt_run = np.dtype([('N_steps', 'int32'),
                               ('Dt',np.float64),
                               ('Dt_out',np.float64),
                               ('T', np.float64,(N_outs,1)),
                               ('T_j_igbt_abcn', np.complex128,(N_outs,8)),
                               ('T_sink', np.complex128,(N_outs,2)),
                               ('out_cplx_i', np.complex128,(N_outs,N_nodes)),
                               ('out_cplx_v', np.complex128,(N_outs,N_nodes)),
                               ('N_outs', 'int32')])  
            
            
            params_run = np.rec.array([(N_steps,
                                        Dt,
                                        Dt_out,
                                        np.zeros((N_outs,1)), # T
                                        np.zeros((N_outs,8)), # T_j_igbt_abcn
                                        np.zeros((N_outs,2)), # T_sink
                                        np.zeros((N_outs,N_nodes)),
                                        np.zeros((N_outs,N_nodes)),                                       
                                        N_outs)],dtype=dt_run)    
                  
            self.params_run = params_run
            
            run_eval(params_run,self.params_pf,params_vsc,params_secondary)
            
#            params_run[0].out_cplx_i = params_run[0].out_cplx_i[0:params_run[0].N_outs,:]
            
                


            
    def diag_2d_inv(self, Z_line_list):

        N_cols = 0

        for Z_line in Z_line_list:
            N_cols += Z_line.shape[1]

        Y_lines = np.zeros((N_cols,N_cols))+0j

        it = 0
        for Z_line in Z_line_list:
            Y_line = np.linalg.inv(Z_line)
            N = Y_line.shape[0] 
            Y_lines[it:(it+N),it:(it+N)] = Y_line
            it += N

        return Y_lines


    def get_v(self):
        
        V_sorted = []
        I_sorted = []
        S_sorted = []
        self.V_results = self.V_node
        self.I_results = self.I_node
        
        nodes2string = ['v_an','v_bn','v_cn','v_gn']
        for bus in self.buses:
            nodes_in_bus = []
            for node in range(10):
                bus_node = '{:s}.{:s}'.format(str(bus['bus']),str(node))
                if bus_node in self.nodes:
                    V = self.V_results[self.nodes.index(bus_node)][0]
                    V_sorted += [V]
                    nodes_in_bus += [node]
            for node in range(10):
                bus_node = '{:s}.{:s}'.format(str(bus['bus']),str(node))
                if bus_node in self.nodes:
                    I = self.I_results[self.nodes.index(bus_node)][0]
                    I_sorted += [I]
            if len(nodes_in_bus)==4:   # if 3 phases + neutral
                v_ag = V_sorted[-4]
                v_bg = V_sorted[-3]
                v_cg = V_sorted[-2]
                v_ng = V_sorted[-1]
                i_a = I_sorted[-4]
                i_b = I_sorted[-3]
                i_c = I_sorted[-2]
                i_n = I_sorted[-1]
                s_a = (v_ag-v_ng)*np.conj(i_a)
                s_b = (v_bg-v_ng)*np.conj(i_b)
                s_c = (v_cg-v_ng)*np.conj(i_c)
                bus.update({'v_an':np.abs(v_ag-v_ng),
                            'v_bn':np.abs(v_bg-v_ng),
                            'v_cn':np.abs(v_cg-v_ng),
                            'v_ng':np.abs(v_ng)})
                bus.update({'deg_an':np.angle(v_ag-v_ng, deg=True),
                            'deg_bn':np.angle(v_bg-v_ng, deg=True),
                            'deg_cn':np.angle(v_cg-v_ng, deg=True),
                            'deg_ng':np.angle(v_ng, deg=True)})
                bus.update({'v_ab':np.abs(v_ag-v_bg),
                            'v_bc':np.abs(v_bg-v_cg),
                            'v_ca':np.abs(v_cg-v_ag)})
                bus.update({'p_a':s_a.real,
                            'p_b':s_b.real,
                            'p_c':s_c.real})
                bus.update({'q_a':s_a.imag,
                            'q_b':s_b.imag,
                            'q_c':s_c.imag})
        self.V = np.array(V_sorted).reshape(len(V_sorted),1)
        return self.V              
        
    def get_i(self):
       
        I_lines = self.Y_lines @ self.A.T @ self.V_results
        
        it_single_line = 0
        for line in self.lines:
            I_a = (I_lines[it_single_line,0])
            I_b = (I_lines[it_single_line+1,0])
            I_c = (I_lines[it_single_line+2,0])
            I_n = (I_lines[it_single_line+3,0])
            it_single_line += len(line['bus_j_nodes'])
            line.update({'i_a_m':np.abs(I_a)})
            line.update({'i_b_m':np.abs(I_b)})
            line.update({'i_c_m':np.abs(I_c)})
            line.update({'i_n_m':np.abs(I_n)})
            line.update({'deg_a':np.angle(I_a, deg=True)})
            line.update({'deg_b':np.angle(I_b, deg=True)})
            line.update({'deg_c':np.angle(I_c, deg=True)})
            line.update({'deg_n':np.angle(I_n, deg=True)})
                        
            

    def bokeh_tools(self):

        
        self.bus_tooltip = '''
            <div>
            bus_id = @bus_id 
            <table border="1">
                <tr>
                <td>v<sub>an</sub> =  @v_an  &ang; @deg_an V </td> <td> S<sub>a</sub> = @p_a + j@q_a </td>
                </tr>
                      <tr>
                      <td> </td> <td>v<sub>ab</sub>= @v_ab V</td>
                      </tr>
                <tr>
                <td>v<sub>bn</sub> = @v_bn &ang; @deg_bn V </td><td> S<sub>b</sub> = @p_b + j@q_b </td>
                </tr>
                      <tr>
                      <td> </td><td>v<sub>bc</sub>= @v_bc V</td>
                      </tr>
                <tr>
                <td>v<sub>cn</sub>  = @v_cn &ang; @deg_cn V </td>  <td>S<sub>c</sub> = @p_c + j@q_c </td>
                </tr> 
                    <tr>
                     <td> </td> <td>v<sub>ca</sub>= @v_ca V</td>
                    </tr>
               <tr>
                <td>v<sub>ng</sub>    = @v_ng &ang; @deg_ng V</td>  <td>S<sub>abc</sub> = @p_abc + j@q_abc </td>
              </tr>
            </table>
            </div>
            '''
            
        x = [item['pos_x'] for item in self.buses]
        y = [item['pos_y'] for item in self.buses]
        bus_id = [item['bus'] for item in self.buses]
        v_an = [item['v_an'] for item in self.buses]
        v_bn = [item['v_bn'] for item in self.buses]
        v_cn = [item['v_cn'] for item in self.buses]
        v_ng = [item['v_ng'] for item in self.buses]
        v_an = [item['v_an'] for item in self.buses]
        deg_an = [item['deg_an'] for item in self.buses]
        deg_bn = [item['deg_bn'] for item in self.buses]
        deg_cn = [item['deg_cn'] for item in self.buses]
        deg_ng = [item['deg_ng'] for item in self.buses]
        v_ab = [item['v_ab'] for item in self.buses]
        v_bc = [item['v_bc'] for item in self.buses]
        v_ca = [item['v_ca'] for item in self.buses]
        p_a = ['{:2.2f}'.format(item['p_a']/1000) for item in self.buses]
        p_b = ['{:2.2f}'.format(item['p_b']/1000) for item in self.buses]
        p_c = ['{:2.2f}'.format(item['p_c']/1000) for item in self.buses]
        q_a = ['{:2.2f}'.format(item['q_a']/1000) for item in self.buses]
        q_b = ['{:2.2f}'.format(item['q_b']/1000) for item in self.buses]
        q_c = ['{:2.2f}'.format(item['q_c']/1000) for item in self.buses]   
        p_abc = ['{:2.2f}'.format((item['p_a'] +item['p_b']+item['p_c'])/1000) for item in self.buses] 
        q_abc = ['{:2.2f}'.format((item['q_a'] +item['q_b']+item['q_c'])/1000) for item in self.buses] 
        self.bus_data = dict(x=x, y=y, bus_id=bus_id,
                             v_an=v_an, v_bn=v_bn, v_cn=v_cn, v_ng=v_ng, 
                             deg_an=deg_an, deg_bn=deg_bn, deg_cn=deg_cn, 
                             deg_ng=deg_ng,v_ab=v_ab,v_bc=v_bc,v_ca=v_ca,
                             p_a=p_a,p_b=p_b,p_c=p_c,
                             q_a=q_a,q_b=q_b,q_c=q_c,
                             p_abc=p_abc,q_abc=q_abc)
        
        self.line_tooltip = '''
            <div>
            line id = @line_id 
            <table border="1">
                <tr>
                <td>I<sub>a</sub> =  @i_a_m &ang; @deg_a </td>
                </tr>
                <tr>
                <td>I<sub>b</sub> =  @i_b_m &ang; @deg_b </td>
                </tr>
                <tr>
                <td>I<sub>c</sub> =  @i_c_m &ang; @deg_c </td>
                </tr>
                <tr>
                <td>I<sub>n</sub> =  @i_n_m &ang; @deg_n </td>
                </tr>
            </table>            
            </div>
            '''
            
        bus_id_to_x = dict(zip(bus_id,x))
        bus_id_to_y = dict(zip(bus_id,y))
        
        x_j = [bus_id_to_x[item['bus_j']] for item in self.lines]
        y_j = [bus_id_to_y[item['bus_j']] for item in self.lines]
        x_k = [bus_id_to_x[item['bus_k']] for item in self.lines]
        y_k = [bus_id_to_y[item['bus_k']] for item in self.lines]
        
        x_s = []
        y_s = []
        for line in self.lines:
            x_s += [[ bus_id_to_x[line['bus_j']] , bus_id_to_x[line['bus_k']]]]
            y_s += [[ bus_id_to_y[line['bus_j']] , bus_id_to_y[line['bus_k']]]]
            
        i_a_m = [item['i_a_m'] for item in self.lines]
        i_b_m = [item['i_b_m'] for item in self.lines]
        i_c_m = [item['i_c_m'] for item in self.lines]
        i_n_m = [item['i_n_m'] for item in self.lines]
        
        deg_a = [item['deg_a'] for item in self.lines]
        deg_b = [item['deg_b'] for item in self.lines]
        deg_c = [item['deg_c'] for item in self.lines]
        deg_n = [item['deg_n'] for item in self.lines]        
        line_id = ['{:s}-{:s}'.format(item['bus_j'],item['bus_k']) for item in self.lines]
#        self.line_data = dict(x_j=x_j, x_k=x_k, y_j=y_j, y_k=y_k, line_id=line_id,
#                             i_a_m=i_a_m)
        self.line_data = dict(x_s=x_s, y_s=y_s, line_id=line_id,
                             i_a_m=i_a_m, i_b_m=i_b_m, i_c_m=i_c_m, i_n_m=i_n_m,
                             deg_a=deg_a, deg_b=deg_b, deg_c=deg_c, deg_n=deg_n)
        return self.bus_data


@numba.jit(nopython=True,cache=True)
def run_eval(params, params_pf,params_vsc,params_secondary):
    N_steps = params[0].N_steps
    Dt = params[0].Dt
    Dt_out = params[0].Dt_out
    
    Dt_secondary = params_secondary[0].Dt_secondary    

    pf_eval(params_pf)   
     
    for it in range(len(params_vsc)):
               
        params_vsc[it].v_abcn_0[:] = np.copy(params_pf[0].V_node[params_vsc[it].nodes])             
        params_vsc[it].i_abcn_0[:] = np.copy(params_pf[0].I_node[params_vsc[it].nodes])
        params_vsc[it].x[:] = np.zeros((4,1))
     
    
    t = 0.0
    t_out = 0.0
    t_secondary = 0.0
    it_out = 0
    
    params[0]['T'][it_out,0] = t
    params[0].out_cplx_i[it_out,:] = params_pf[0].I_node.T
    params[0].out_cplx_v[it_out,:] = params_pf[0].V_node.T
    
    for it in range(N_steps):

        t += Dt

        for it in range(len(params_vsc)):                
            params_vsc[it].i_abcn[:] = params_pf[0].I_node[params_vsc[it].nodes]
                       
        ctrl_vsc_phasor(t,1,params_vsc,params_secondary)  

        for it in range(len(params_vsc)):                
            params_vsc[it].x[:] += Dt*params_vsc[0].f[:]       
        
        ctrl_vsc_phasor(t,3,params_vsc,params_secondary)  

        for it in range(len(params_vsc)):             
            params_pf[0].V_node[params_vsc[it].nodes,:] = params_vsc[it].v_abcn[:]
        
        pf_eval(params_pf)   
        
        if t>0.5: 
            
            params_pf[0].pq_3pn[0,:] = 0.0

        if t>30.0:
            
            params_pf[0].pq_3pn[5,0] =  -20e3
##            params_pf[0].pq_3pn[params_pf[0].pq_3pn_int[1,1]] = 50
##            params_pf[0].pq_3pn[params_pf[0].pq_3pn_int[1,2]] = 50
#            
        if t > t_secondary+Dt_secondary:
            V_mean = (np.sum(np.abs(params_pf[0].V_node[0:3,:]))+ np.sum(np.abs(params_pf[0].V_node[4:7,:])))/6.0 
            params_secondary[0].V = V_mean
            secondary_ctrl(t,1,params_secondary,params_vsc)
            
            params_secondary[0].x[:] += Dt_secondary*params_secondary[0].f[:]
            secondary_ctrl(t,2,params_secondary,params_vsc)
#            print(params_vsc.DV_remote)
            t_secondary = t 
             
        if t > t_out+Dt_out:
            it_out += 1 
            #print(np.abs(params_vsc[0].x))
            params[0]['T'][it_out,0] = t
            params[0].T_sink[it_out,0] = params_vsc[0].T_sink
            params[0].T_sink[it_out,1] = params_vsc[1].T_sink
            params[0].T_j_igbt_abcn[it_out,0:4] = params_vsc[0].T_j_igbt_abcn.T 
            params[0].T_j_igbt_abcn[it_out,4:8] = params_vsc[1].T_j_igbt_abcn.T 
            params[0].out_cplx_i[it_out,:] = params_pf[0].I_node.T 
            params[0].out_cplx_v[it_out,:] = params_pf[0].V_node.T  
            params[0].N_outs = it_out
            t_out = t 
            
    
    
    
    
#@numba.jit(nopython=True,cache=True)
#def pf_eval(params,max_iter=20):
#    Y_vv =  params[0].Y_vv
#    inv_Y_ii = params[0].inv_Y_ii
#    Y_iv =  params[0].Y_iv
#    N_v = params[0].N_nodes_v
#    N_i = params[0].N_nodes_i
#    pq_3pn_int = params[0].pq_3pn_int
#    pq_3pn     = params[0].pq_3pn
#    V_node = params[0].V_node
#    I_node = params[0].I_node
#    
#    V_unknown = np.copy(V_node[N_v:])
#    I_known   = np.copy(I_node[N_v:])
#    V_known   = np.copy(V_node[0:N_v])
#    
#    Y_vi =  Y_iv.T
#    V_unknown_0 = V_unknown
#    
#    for iteration in range(max_iter):
#
#    
#        for it in range(pq_3pn_int.shape[0]):
#            
#
#            V_abc = V_unknown[pq_3pn_int[it][0:3],0]
#            S_abc = pq_3pn[it,:]
#           
#            I_known[pq_3pn_int[it][0:3],0] = np.conj(S_abc/V_abc)
#            I_known[pq_3pn_int[it][3],0] =  -np.sum(I_known[pq_3pn_int[it][0:3],0])
#
#        
#        V_unknown = inv_Y_ii @ ( I_known - Y_iv @ V_known)
#        
#        if np.linalg.norm(V_unknown - V_unknown_0,np.inf) <1.0e-8: break
#        V_unknown_0 = V_unknown
#
#    I_unknown =Y_vv @ V_known + Y_vi @ V_unknown
#    
#    V_node[0:N_v,:] = V_known 
#    V_node[N_v:,:]  = V_unknown 
#
#    I_node[0:N_v,:] = I_unknown 
#    I_node[N_v:,:]  = I_known 
#        
#    return V_node,I_node


        
spec = [
    ('max_iter', int32), 
    ('iterations', int32), 
    ('N_nodes_i', int32), 
    ('N_nodes_v', int32), 
    ('N_nodes', int32), 
    ('ctrl_primary_mode',int32),
    ('ctrl_secondary_mode',int32),
    ('V_known', complex128[:,:]), 
    ('I_known', complex128[:,:]), 
    ('I_known_0', complex128[:,:]), 
    ('V_unknown', complex128[:,:]), 
    ('I_unknown', complex128[:,:]), 
    ('V_unknown_0', complex128[:,:]), 
    ('V_node', complex128[:,:]), 
    ('I_node', complex128[:,:]), 
    ('pq_3pn_int', int64[:,:]),               # a simple scalar field    
    ('pq_3pn', complex128[:,:]),
    ('Y_ii', complex128[:,:]),
    ('Y_iv', complex128[:,:]),
    ('Y_vv', complex128[:,:]),
    ('Y_vi', complex128[:,:]),
    ('inv_Y_ii', complex128[:,:]),
    ('x_0', float64[:,:]),
    ('x', float64[:,:]),
    ('X', float64[:,:]),
    ('T', float64[:,:]),
    ('out', float64[:,:]),
    ('out_cplx', complex128[:,:])
    ]

   
#@numba.jitclass(spec)
class system(object):
    
    def __init__(self):
        
        # general data
        self.N_nodes_i = 0
        self.N_nodes_v = 0
        self.N_nodes = 0
        
        # Voltages
        self.V_known = np.zeros((0,0),dtype=np.complex128)
        self.V_unknown = np.zeros((0,0),dtype=np.complex128)
        self.V_unknown_0 = np.zeros((0,0),dtype=np.complex128) # initial guess
        self.V_node = np.zeros((0,0),dtype=np.complex128) # full node Volatges
        
        # Currents
        self.I_known = np.zeros((0,0),dtype=np.complex128)
        self.I_unknown = np.zeros((0,0),dtype=np.complex128)
        self.I_known_0 = np.zeros((0,0),dtype=np.complex128)   # initial guess
        self.I_node = np.zeros((0,0),dtype=np.complex128) # full node Currents
        
        # Admitance matrices  
        # Y = [Y_vv  Y_vi]
        #     [Y_iv  Y_ii]
        self.Y_vv = np.zeros((1,1),dtype=np.complex128)
        self.Y_vi = np.zeros((1,1),dtype=np.complex128)
        self.Y_iv = np.zeros((1,1),dtype=np.complex128)
        self.Y_ii = np.zeros((1,1),dtype=np.complex128)
        self.inv_Y_ii = np.zeros((1,1),dtype=np.complex128) # Y_ii inverse
        
        # Known powers
        self.pq_3pn_int = np.zeros((1,1),dtype=np.int64)
        self.pq_3pn = np.zeros((1,1),dtype=np.complex128)

        # Solver paramters
        self.max_iter = 50
        
        self.iterations = 0
        
        self.ctrl_primary_mode = 0
        self.ctrl_secondary_mode = 2
        
        
        
        
#        #    def pf(V_known,known_1,known_2,i_fp_modes,pq_modes,Y_vv,Y_vi,Y_iv,Y_ii,V_unknown_0):
#    def pf_eval(self):
#        
#        V_known = self.V_known
#        Y_vv = self.Y_vv
#        Y_iv = self.Y_iv
#        Y_vi = self.Y_vi
#        #Y_ii = self.Y_ii
#        inv_Y_ii = self.inv_Y_ii
#        pq_3pn_int = self.pq_3pn_int
#        pq_3pn = self.pq_3pn
#
#        V_unknown = np.copy(self.V_unknown_0)
#        I_known = np.copy(self.I_known_0)
#        
#        max_iter  = self.max_iter
#        
#        V_unknown_0 = V_unknown
#        for iteration in range(max_iter):
#
#        
#            for it in range(pq_3pn_int.shape[0]):
#
#                V_abc = V_unknown[pq_3pn_int[it][0:3],0]
#                S_abc = pq_3pn[it,:]
#               
#                I_known[pq_3pn_int[it][0:3],0] = np.conj(S_abc/V_abc)
#                I_known[pq_3pn_int[it][3],0] =  -np.sum(I_known[pq_3pn_int[it][0:3],0])
#        
#
#            V_unknown = inv_Y_ii @ ( I_known- Y_iv @ V_known)
#            
#            if np.linalg.norm(V_unknown - V_unknown_0,np.inf) <1.0e-8: break
#            V_unknown_0 = V_unknown
##        V_unknown = inv_Y_ii @ (I_known - Y_iv @ V_known)
#        I_unknown =Y_vv @ V_known + Y_vi @ V_unknown
#        
#        self.iterations = iteration
#        self.I_known = I_known
#        self.V_unknown = V_unknown
#        self.I_unknown = I_unknown
#        
#        self.V_node[0:self.N_nodes_v,:] = self.V_known 
#        self.V_node[self.N_nodes_v:self.N_nodes,:] = self.V_unknown 
#
#        self.I_node[0:self.N_nodes_v,:] = self.I_unknown 
#        self.I_node[self.N_nodes_v:self.N_nodes,:] = self.I_known 
#        
#       
#        return self.V_node,self.I_node

    def run_eval(self):
        N_steps = 10000
        N_states = 6
        x = np.zeros((N_states,1))
        f = np.zeros((N_states,1))
        self.X = np.zeros((N_states,N_steps))
        self.out = np.zeros((25,N_steps))
        self.out_cplx = np.zeros((78,N_steps),np.complex128)
        
        self.T = np.zeros((N_steps,1))
        ctrl1 = ctrl_vsc_phasor()
        ctrl2 = ctrl_vsc_phasor()
        secondary1 = secondary()
        secondary1.V = np.zeros((6,1),np.complex128)
        
        V_node,I_node = pf_eval(self.Y_vv,
        self.Y_iv,
        self.inv_Y_ii,

        self.pq_3pn_int,
        self.pq_3pn,
        self.N_nodes_v,
        self.N_nodes_i
        )
        
        ctrl1.v_abcn_0 = np.copy(self.V_known[0:4])
        ctrl2.v_abcn_0 = np.copy(self.V_known[4:8])

        ctrl1.i_abcn_0 = np.copy(self.I_unknown[0:4])
        ctrl2.i_abcn_0 = np.copy(self.I_unknown[4:8])
        
        ctrl1.v_abcn = np.copy(ctrl1.v_abcn_0)
        ctrl2.v_abcn = np.copy(ctrl2.v_abcn_0)
    
        ctrl1.mode = self.ctrl_primary_mode
        ctrl2.mode = self.ctrl_primary_mode
        secondary1.mode = self.ctrl_secondary_mode
    
        #ctrl2.Dang = 0.0
       
        dt = 2.0e-3
        t = 0.0
        t_sec = 1.0
        self.X[:,0] = x[:,0]
        
        secondary1.V_ref = 240.0 
        
        secondary1.S_base[0]= 100.0e3
        secondary1.S_base[1]= 100.0e3
        
         
        
        for it in range(N_steps):  
            
            
            t += dt
            
            ctrl1.x = x[0:3]
            ctrl2.x = x[3:6]
            ctrl1.h_eval()
            ctrl2.h_eval()
 
            self.V_known[0:4]= ctrl1.v_abcn           
            self.V_known[4:8]= ctrl2.v_abcn
            
            if t>1.0:
                self.pq_3pn[0,0] = 0.0
                self.pq_3pn[0,1] = 0.0
                self.pq_3pn[0,2] = 0.0

            if t>10.0:
                self.pq_3pn[3,0] = 0j
                self.pq_3pn[3,1] = 0j
                self.pq_3pn[3,2] = 0j
                
            if t>t_sec:

                secondary1.S_array[0] = np.sum(ctrl1.S[:,0])
                secondary1.S_array[1] = np.sum(ctrl2.S[:,0])
                
                secondary1.V[0:3,0] = self.V_known[0:3,0]
                secondary1.V[3:6,0] = self.V_known[4:7,0]
                secondary1.x += 1.0*secondary1.f_eval()
                
                secondary1.h_eval()
                
                ctrl1.DV_secondary = secondary1.DV_array[0]
                ctrl2.DV_secondary = secondary1.DV_array[1]
   
                t_sec = t+1.0

            self.V_unknown_0 = self.V_unknown
            self.I_known_0 = self.I_known
        
           
            self.pf_eval()
                    
            
            
            ctrl1.i_abcn = self.I_node[0:4]
            ctrl2.i_abcn = self.I_node[4:8]
            
            f[0:3] = ctrl1.f_eval() 
            f[3:6] = ctrl2.f_eval() 

            x = x + dt*f
            
            self.X[:,it] = x[:,0]
            self.T[it,0] = t
            self.out[0:4,it] = np.abs(self.I_node[0:4,0])
            self.out[4:8,it] = np.abs(self.I_node[4:8,0])
            self.out[8:12,it] = np.abs(self.V_node[0:4,0])
            self.out[12:16,it] = np.abs(self.V_node[4:8,0])            
            self.out[16:20,it] = np.angle(self.V_node[0:4,0])
            self.out[20:24,it] = np.angle(self.V_node[4:8,0])  
            self.out[24,it]  = self.iterations
            
            self.out_cplx[0:72,it]  = self.V_node[0:72,0]
            self.out_cplx[72:75,it]  = ctrl1.S[:,0]
            self.out_cplx[75:78,it]  = ctrl2.S[:,0]
            
#def run_eval():

                
       
class opendss(object):
    
    def __init__(self):
        
        pass
    
    def pyss2opendss(self):
        
        string = ''
        for item in sys.loads:
            string += 'New Load.L_{:s} '.format(item['bus'])
            string += 'Phases=3 Bus1={:s} kV=0.231 kVA={:2.3f} PF={:2.2f}'.format(item['bus'],item['kVA'],item['fp'])    
            string += '\n' 
        for item in sys.lines:
            # New Line.LINE1 Bus1=1 Bus2=2 
            string += 'New Line.LINE_{:s}_{:s} Bus1={:s} Bus2={:s} '.format(item['bus_j'],item['bus_k'],item['bus_j'],item['bus_k'])
            string += 'phases=3 Linecode={:s} Length={:f} Units=m'.format(item['code'],item['m'])    
            string += '\n'         
        for item in line_codes:
            #New LineCode.UG3  nphases=3  BaseFreq=50 
            #~ rmatrix = (1.152 | 0.321   1.134 | 0.33 0.321 1.152)
            #~ xmatrix = (0.458  | 0.39 0.477   | 0.359 0.390 0.458)
            #~ units=km 
            string += 'New LineCode.{:s} '.format(item)
            Z_list = line_codes[item]
            N_conductors = len(Z_list)
            string += 'nphases={:d}  BaseFreq=50 \n'.format(N_conductors) 
            Z = np.array(Z_list)
            R = Z.real
            X = Z.imag
            string += '~ rmatrix = ('
            for it in range(N_conductors):
                row = R[it,0:it+1]
                for item_col in row:
                    string += '{:f} '.format(item_col)
                if it == N_conductors-1:
                    string += ')\n'
                else:
                    string += '| '
            string += '~ xmatrix = ('
            for it in range(N_conductors):
                row = X[it,0:it+1]
                for item_col in row:
                    string += '{:f} '.format(item_col)
                if it == N_conductors-1:
                    string += ')\n'
                else:
                    string += '| '                
            string += '~ units=km \n'
        return string
            
    def read_v_results(self, file):
        
        fobj = open(file)
        
        lines = fobj.readlines()
               
        for line in lines:
            print(line[5:6])
            
        return string        
    

   



#    def __init__(self):
#               
#
#        self.N_states = 3
#        self.x_0 =np.zeros((self.N_states,1))           
#        
#        self.x = np.copy(self.x_0)   
#
#        self.mode = 0
#        self.K_v = 0.1
#        self.K_ang = 0.0
#        self.K_f = 0.0        
#        
#        self.T_v = 0.2
#        self.T_ang = 0.2
#        
#        self.freq = 50.0  
#        
#        self.Dang = 0.0 
#        
#        self.v_abcn_0 =np.zeros((4,1),dtype=np.complex128)   
#        self.i_abcn_0 =np.zeros((4,1),dtype=np.complex128)
#        self.i_abcn =np.zeros((4,1),dtype=np.complex128)
#        self.v_abcn =np.zeros((4,1),dtype=np.complex128)
#        self.v_abcn_m_ref =np.zeros((4,1))
#
#        self.S =np.zeros((3,1),dtype=np.complex128)
#        self.f =np.zeros((3,1),dtype=np.float64)
#        self.h =np.zeros((4,1),dtype=np.float64)
#        
#        
#        self.ctrl_design()
#        
#        self.DV_secondary = 0.0
#
#
#        
#    def f_eval(self):
#        '''
#        0: 'fix'
#        1: 'pf_qv'
#        2: 'pv_qf'
#        3: 'droop_generalized'
#        4: 'iv'
#        5: 'nl_droop'
#        '''
#        if self.mode == 0: # 'fix'
#            self.v_abcn = self.v_abcn_0
#            V_abc = self.v_abcn[0:3]   # phase to neutral abc voltages (without neutral)
#            I_abc = self.i_abcn[0:3,:] # phase currents (without neutral)
#            
#            S = V_abc*np.conj(I_abc) # phase complex power
#            self.S = S
#            
#
##        if self.mode == 2:
##            K_v = self.K_v
##            K_f = self.K_f
##            
##            V_abc_0 = self.v_abcn_0[0:3]   # phase to neutral abc voltages (without neutral)
##            I_abc_0 = self.i_abcn_0[0:3,:] # phase currents (without neutral)
##            
##            S_0 = V_abc_0*np.conj(I_abc_0) # phase complex power
##            
##            V_abc = self.v_abcn[0:3]   # phase to neutral abc voltages (without neutral)
##            I_abc = self.i_abcn[0:3,:] # phase currents (without neutral)
##            
##            S = V_abc*np.conj(I_abc) # phase complex power
##            self.S = S
##            
##            DS_abc = S-S_0*0 # complex power increment
##            
##            DP_abc = DS_abc.real
##            DQ_abc = DS_abc.imag
##            
##            K_v = 1e-10
##            K_f = 2e-9
##
##            DV_m_ref =   -K_v*np.sum(DP_abc)
##            Dfreq_ref =  -K_f*np.sum(DQ_abc)
##            
###            print(np.sum(DQ_abc))
##            
##            Dang = self.x[2:3]
##            
##            self.f[0:1,:] = 1.0/self.T_v*(DV_m_ref - self.x[0:1])  # voltage control dynamics
##            self.f[1:2,:] = 1.0/self.T_ang*(Dang - self.x[1:2])  # angle control dynamics
##            self.f[2:3,:] = 2.0*np.pi*Dfreq_ref  # angle from frequency
##            
##    
#        if self.mode == 3:
#            K_v = self.K_v
#            K_ang = self.K_ang
#            
#            V_abc_0 = self.v_abcn_0[0:3]   # phase to neutral abc voltages (without neutral)
#            I_abc_0 = self.i_abcn_0[0:3,:] # phase currents (without neutral)
#            
#            S_0 = V_abc_0*np.conj(I_abc_0) # phase complex power
#            
#            V_abc = self.v_abcn[0:3]   # phase to neutral abc voltages (without neutral)
#            I_abc = self.i_abcn[0:3,:] # phase currents (without neutral)
#            
#            S = V_abc*np.conj(I_abc) # phase complex power
#            self.S = S
#            
#            DS_abc = S-S_0*0 # complex power increment
#            
#            DP_abc = DS_abc.real
#            DQ_abc = DS_abc.imag
#            
#            K_v   = 1e-6    def pf_eval(self):
        
#        V_unknown_0 = np.zeros((self.N_nodes_i,1),dtype=np.complex128)+231
#        
#        for it in range(int(self.N_nodes_i/4)): # change if not 4 wires
#            
#            V_unknown_0[4*it+0] = self.V_known[0]
#            V_unknown_0[4*it+1] = self.V_known[1]
#            V_unknown_0[4*it+2] = self.V_known[2]
#            V_unknown_0[4*it+3] = 0.0
#            
#                
#        I = np.vstack((np.zeros((self.N_nodes_v,1)),
#                       np.zeros((self.N_nodes_i,1))))+0j
#        V = np.vstack((self.V_known,V_unknown_0 ))
#        V_node,I_node = pf_eval(self.Y_vv,
#                self.Y_iv,
#                self.inv_Y_ii,
#                I,V,
#                self.pq_3pn_int,
#                self.pq_3pn,
#                self.N_nodes_v,
#                self.N_nodes_i
#                )
#
#        self.V_node = V_node
#        self.I_node = I_node
##            K_ang = 1e-5
##
##            DV_m_ref =  -K_v*np.sum(DP_abc)
##            Dang_ref =  K_ang*np.sum(DQ_abc)
##            
###            print(np.sum(DQ_abc))
##            
##            Dang = self.x[2:3]
##            
##            self.f[0:1,:] = 1.0/self.T_v*(DV_m_ref - self.x[0:1])  # voltage control dynamics
##            self.f[1:2,:] = 1.0/self.T_ang*(Dang_ref - self.x[1:2])  # angle control dynamics
##            self.f[2:3,:] = 0.0  # angle from frequency
##    
###            
###        if self.mode == 6:
###            K_v = self.K_v
###            s_0 = self.v_abcn_0[0:3]*np.conj(self.i_abcn_0[0:3,:])
###            s = self.v_abcn[0:3]*np.conj(self.i_abcn[0:3,:])
###            Dp_abc = s.real - s_0.real
###            Dv_abc_m_ref =  -K_v*Dp_abc
###            self.f[0:3,:] = 1.0/self.T_v*(Dv_abc_m_ref - self.x[0:3])   
###            
###        if self.mode == 4:
###            K_v = self.K_v
###            Di_abc = np.abs(self.i_abcn[0:3,:]) - np.abs(self.i_abcn_0[0:3,:])
###            Dv_abc_m_ref =  -K_v*Di_abc
###            self.f[0:3,:] = 1.0/self.T_v*(Dv_abc_m_ref - self.x[0:3])            
###            
##        
##        return self.f
##
##
##    def h_eval(self):        
##        '''
##        0: 'fix'
##        1: 'pf_qv'
##        2: 'pv_qf'
##        3: 'pv_qa'  # working
##        4: 'iv'
##        5: 'nl_droop'
##        '''
##        if self.mode == 0:
##            #print(self.DV_secondary)
##            self.v_abcn = self.v_abcn_0 *  (1+self.DV_secondary)  * np.exp(1j*self.Dang)
##            
###        if self.mode == 2:       
###
###            
###            DV_f = self.x[0:1]  # voltage control dynamics
###            Dang = self.x[1:2]  # angle control dynamics
###            
###            angles_abc = np.angle(self.v_abcn_0[0:3])
###            module_abc = np.abs(self.v_abcn_0[0:3]) + self.x
####            self.v_abcn[0:3] =self.v_abcn_0[0:3] * (1+self.x)
###            
####            print(Dang)
###            self.v_abcn[0] =module_abc[0] * np.exp(1j*(angles_abc[0]+Dang))
###            self.v_abcn[1] =module_abc[1] * np.exp(1j*(angles_abc[1]+Dang))
###            self.v_abcn[2] =module_abc[2] * np.exp(1j*(angles_abc[2]+Dang))
###            
###            self.v_abcn[3] = 0.0
###            
###        if self.mode == 3: # 'pv_qa'
###        
###            DV_m = self.x[0:1]  # voltage control dynamics
###            Dang = self.x[1:2]  # angle control dynamics
###
###            self.v_abcn[0:3] = self.v_abcn_0[0:3] * (1+DV_m+self.DV_secondary) * np.exp(1j*Dang)
###            
###        if self.mode == 4:
###            angles_abc = np.angle(self.v_abcn_0[0:3])
###            module_abc = np.abs(self.v_abcn_0[0:3]) + self.x
###            self.v_abcn[0:3] = module_abc * np.exp(1j*angles_abc)
###            
###            
###        return self.h
##
##    def ctrl_design(self):
##        
##         self.K_v = 2e-6
#
#
#
#
#
##spec = [
##    ('N_states', int64),               # a simple scalar field
##    ('mode', int32), # 0: fixed V, 1: gen droop (f), 2: nl droop
##    ('K_v', float64),  # gain for voltage module
##    ('K_ang', float64),  # gain for voltage angle
##    ('K_f', float64),  # gain affecting  frequency
##    ('T_v', float64), 
##    ('T_ang', float64),  
##    ('freq', float64), 
##    ('Dang', float64),    
##    ('DV_secondary', float64),  
##    ('x_0', float64[:,:]),
##    ('x', float64[:,:]), 
##    ('f', float64[:,:]), 
##    ('h', float64[:,:]), 
##    ('i_abcn', complex128[:,:]),    
##    ('v_abcn', complex128[:,:]),
##    ('v_abcn_0', complex128[:,:]),
##    ('i_abcn_0', complex128[:,:]),
##    ('v_abcn_m_ref', float64[:,:]),
##    ('S', complex128[:,:])
##]
##
###@numba.jitclass(spec)
##class ctrl_vsc_phasor(object):
##    
##    def __init__(self):
##               
##
##        self.N_states = 3
##        self.x_0 =np.zeros((self.N_states,1))           
##        
##        self.x = np.copy(self.x_0)   
##
##        self.mode = 0
##        self.K_v = 0.1
##        self.K_ang = 0.0
##        self.K_f = 0.0        
##        
##        self.T_v = 0.2    def pf_eval(self):
#        
#        V_unknown_0 = np.zeros((self.N_nodes_i,1),dtype=np.complex128)+231
#        
#        for it in range(int(self.N_nodes_i/4)): # change if not 4 wires
#            
#            V_unknown_0[4*it+0] = self.V_known[0]
#            V_unknown_0[4*it+1] = self.V_known[1]
#            V_unknown_0[4*it+2] = self.V_known[2]
#            V_unknown_0[4*it+3] = 0.0
#            
#                
#        I = np.vstack((np.zeros((self.N_nodes_v,1)),
#                       np.zeros((self.N_nodes_i,1))))+0j
#        V = np.vstack((self.V_known,V_unknown_0 ))
#        V_node,I_node = pf_eval(self.Y_vv,
#                self.Y_iv,
#                self.inv_Y_ii,
#                I,V,
#                self.pq_3pn_int,
#                self.pq_3pn,
#                self.N_nodes_v,
#                self.N_nodes_i
#                )
#
#        self.V_node = V_node
#        self.I_node = I_node
##        self.T_ang = 0.2
##        
##        self.freq = 50.0  
##        
##        self.Dang = 0.0 
##        
##        self.v_abcn_0 =np.zeros((4,1),dtype=np.complex128)   
##        self.i_abcn_0 =np.zeros((4,1),dtype=np.complex128)
##        self.i_abcn =np.zeros((4,1),dtype=np.complex128)
##        self.v_abcn =np.zeros((4,1),dtype=np.complex128)
##        self.v_abcn_m_ref =np.zeros((4,1))
##
##        self.S =np.zeros((3,1),dtype=np.complex128)
##        self.f =np.zeros((3,1),dtype=np.float64)
##        self.h =np.zeros((4,1),dtype=np.float64)
##        
##        
##        self.ctrl_design()
##        
##        self.DV_secondary = 0.0
##
##
##        
##    def f_eval(self):
##        '''
##        0: 'fix'
##        1: 'pf_qv'
##        2: 'pv_qf'
##        3: 'droop_generalized'
##        4: 'iv'
##        5: 'nl_droop'
##        '''
##        if self.mode == 0: # 'fix'
##            self.v_abcn = self.v_abcn_0
##            V_abc = self.v_abcn[0:3]   # phase to neutral abc voltages (without neutral)
##            I_abc = self.i_abcn[0:3,:] # phase currents (without neutral)
##            
##            S = V_abc*np.conj(I_abc) # phase complex power
##            self.S = S
##            
##
###        if self.mode == 2:
###            K_v = self.K_v
###            K_f = self.K_f
###            
###            V_abc_0 = self.v_abcn_0[0:3]   # phase to neutral abc voltages (without neutral)
###            I_abc_0 = self.i_abcn_0[0:3,:] # phase currents (without neutral)
###            
###            S_0 = V_abc_0*np.conj(I_abc_0) # phase complex power
###            
###            V_abc = self.v_abcn[0:3]   # phase to neutral abc voltages (without neutral)
###            I_abc = self.i_abcn[0:3,:] # phase currents (without neutral)
###            
###            S = V_abc*np.conj(I_abc) # phase complex power
###            self.S = S
###            
###            DS_abc = S-S_0*0 # complex power increment
###            
###            DP_abc = DS_abc.real
###            DQ_abc = DS_abc.imag
###            
###            K_v = 1e-10
###            K_f = 2e-9
###
###            DV_m_ref =   -K_v*np.sum(DP_abc)
###            Dfreq_ref =  -K_f*np.sum(DQ_abc)
###            
####            print(np.sum(DQ_abc))
###            
###            Dang = self.x[2:3]
###            
###            self.f[0:1,:] = 1.0/self.T_v*(DV_m_ref - self.x[0:1])  # voltage control dynamics
###            self.f[1:2,:] = 1.0/self.T_ang*(Dang - self.x[1:2])  # angle control dynamics
###            self.f[2:3,:] = 2.0*np.pi*Dfreq_ref  # angle from frequency
###            
###    
##        if self.mode == 3:
##            K_v = self.K_v
##            K_ang = self.K_ang
##            
##            V_abc_0 = self.v_abcn_0[0:3]   # phase to neutral abc voltages (without neutral)
##            I_abc_0 = self.i_abcn_0[0:3,:] # phase currents (without neutral)
##            
##            S_0 = V_abc_0*np.conj(I_abc_0) # phase complex power
##            
##            V_abc = self.v_abcn[0:3]   # phase to neutral abc voltages (without neutral)
##            I_abc = self.i_abcn[0:3,:] # phase currents (without neutral)
##            
##            S = V_abc*np.conj(I_abc) # phase complex power
##            self.S = S
##            
##            DS_abc = S-S_0*0 # complex power increment
##            
##            DP_abc = DS_abc.real
##            DQ_abc = DS_abc.imag
##            
##            K_v   = 1e-6
##            K_ang = 1e-5
##
##            DV_m_ref =  -K_v*np.sum(DP_abc)
##            Dang_ref =  K_ang*np.sum(DQ_abc)
##            
###            print(np.sum(DQ_abc))
##            
##            Dang = self.x[2:3]
##            
##            self.f[0:1,:] = 1.0/self.T_v*(DV_m_ref - self.x[0:1])  # voltage control dynamics
##            self.f[1:2,:] = 1.0/self.T_ang*(Dang_ref - self.x[1:2])  # angle control dynamics
##            self.f[2:3,:] = 0.0  # angle from frequency
##    
###            
###        if self.mode == 6:
###            K_v = self.K_v
###            s_0 = self.v_abcn_0[0:3]*np.conj(self.i_abcn_0[0:3,:])
###            s = self.v_abcn[0:3]*np.conj(self.i_abcn[0:3,:])
###            Dp_abc = s.real - s_0.real
###            Dv_abc_m_ref =  -K_v*Dp_abc
###            self.f[0:3,:] = 1.0/self.T_v*(Dv_abc_m_ref - self.x[0:3])   
###            
###        if self.mode == 4:
###            K_v = self.K_v
###            Di_abc = np.abs(self.i_abcn[0:3,:]) - np.abs(self.i_abcn_0[0:3,:])
###            Dv_abc_m_ref =  -K_v*Di_abc
###            self.f[0:3,:] = 1.0/self.T_v*(Dv_abc_m_ref - self.x[0:3])            
###            
##        
##        return self.f
##
##
##    def h_eval(self):        
##        '''
##        0: 'fix'
##        1: 'pf_qv'
##        2: 'pv_qf'
##        3: 'pv_qa'  # working
##        4: 'iv'
##        5: 'nl_droop'
##        '''
##        if self.mode == 0:
##            #print(self.DV_secondary)
##            self.v_abcn = self.v_abcn_0 *  (1+self.DV_secondary)  * np.exp(1j*self.Dang)
##            
###        if self.mode == 2:       
###
###            
###            DV_f = self.x[0:1]  # voltage control dynamics
###            Dang = self.x[1:2]  # angle control dynamics
###            
###            angles_abc = np.angle(self.v_abcn_0[0:3])
###            module_abc = np.abs(self.v_abcn_0[0:3]) + self.x
####            self.v_abcn[0:3] =self.v_abcn_0[0:3] * (1+self.x)
###            
####            print(Dang)
###            self.v_abcn[0] =module_abc[0] * np.exp(1j*(angles_abc[0]+Dang))
###            self.v_abcn[1] =module_abc[1] * np.exp(1j*(angles_abc[1]+Dang))
###            self.v_abcn[2] =module_abc[2] * np.exp(1j*(angles_abc[2]+Dang))
###            
###            self.v_abcn[3] = 0.0
###            
###        if self.mode == 3: # 'pv_qa'
###        
###            DV_m = self.x[0:1]  # voltage control dynamics
###            Dang = self.x[1:2]  # angle control dynamics
###
###            self.v_abcn[0:3] = self.v_abcn_0[0:3] * (1+DV_m+self.DV_secondary) * np.exp(1j*Dang)
###            
###        if self.mode == 4:
###            angles_abc = np.angle(self.v_abcn_0[0:3])
###            module_abc = np.abs(self.v_abcn_0[0:3]) + self.x
###            self.v_abcn[0:3] = module_abc * np.exp(1j*angles_abc)
###            
###            
###        return self.h
##
##    def ctrl_design(self):
##        
##         self.K_v = 2e-6


#spec = [
#    ('N_states', int64),               # a simple scalar field
#    ('x_0', float64[:,:]),
#    ('x', float64[:,:]), 
#    ('f', float64[:,:]), 
#    ('h', float64[:,:]), 
#    ('mode', int32), # 0: fixed V, 1: gen droop (f), 2: nl droop
#    ('K_p_v', float64),  # gain for voltage module
#    ('K_i_v', float64),  # gain for voltage angle
#    ('V_ref', float64),  # gain affecting  frequency
#    ('V_0', complex128[:,:]),
#    ('V', complex128[:,:]),
#    ('DV', float64),
#    ('DV_array', float64[:,:]),
#    ('S_array', complex128[:,:]),
#    ('S_base', float64[:,:])
#]
#
##@numba.jitclass(spec)
#class secondary(object):
#    
#    def __init__(self):
#               
#
#        self.N_states = 2
#        self.x_0 =np.zeros((self.N_states,1))   
#        self.f = np.zeros((self.N_states,1))          
#        self.h = np.zeros((self.N_states,1)) 
#        
#        self.x = np.copy(self.x_0)
#
#        self.mode = 0
#        self.K_p_v = 0.0001
#        self.K_i_v = 0.001
#        
#        self.V_0 =np.zeros((4,1),dtype=np.complex128)   
#        self.V =np.zeros((4,1),dtype=np.complex128)   
#        
#        self.V_ref = 231.0
#        self.DV = 0.0
#        
#        self.S_base = np.zeros((2,1))
#        self.S_array = np.zeros((2,1),dtype=np.complex128) 
#        self.DV_array = np.zeros((2,1))
#        
#    def f_eval(self):
#        '''
#        0: 'fix'
#        1: 'v'
#        2:  
#        3:  
#        4:  
#        5:  
#        '''
#        if self.mode == 0: # 'fix'
#            self.DV = 0.0 
#            
#
#        if self.mode == 1:  # 'v'
#
#            V_mean = np.mean(np.abs(self.V))
#            error =  self.V_ref - V_mean
#            self.f[0] = error
#            
#        if self.mode == 2:  # 'p'
#            S_total = np.sum(self.S_base)
#            P_demand_est = np.sum(self.S_array.real)    
#            for it in range(2):
#                P_ref = P_demand_est * self.S_base[it]/S_total
#                DP = P_ref - self.S_array[it].real
#                   
#                error =  DP
#                if error> 10e3:
#                    error = 10e3
#                if error< -10e3:
#                    error = -10e3                    
#                self.f[it] = error                          
#
#
#        return self.f
#
#
#    def h_eval(self):        
#        '''
#        0: 'fix'
#        1: 'pf_qv'
#        2: 'pv_qf'
#        3: 'pv_qa'
#        4: 'iv'
#        5: 'nl_droop'
#        '''
#        if self.mode == 0:
#            self.DV = 0.0
#            
#        if self.mode == 1:  # 'v'
#            K_p_v = self.K_p_v
#            K_i_v = self.K_i_v
#            
#            V_mean = np.mean(np.abs(self.V))
#            error =  231.0 - V_mean
#            
#            self.DV = K_p_v*error + K_i_v * self.x[0,0]
#                        
#            self.h[0] = self.DV
#
#        if self.mode == 2:  # 'p'
#            S_total = np.sum(self.S_base)
#            P_demand_est = np.sum(self.S_array.real)    
#            for it in range(2):
#                P_ref = P_demand_est * self.S_base[it]/S_total
#                DP = P_ref - self.S_array[it].real
#                DV = DP*0.000000002 + 0.00000005* self.x[it,0]
#                if DV >10.0:
#                    DV =10.0
#                    self.f[it] = 0.0
#                if DV < -10.0:
#                    DV = -10.0
#                    self.f[it] = 0.0
#                    
#                self.DV_array[it] = DV
#            
#                
#        return self.h
#
#    def ctrl_design(self):
#        
#         self.K_v = 2e-6



spec = [('value', float64[:,:]),
                 ('cplx_value_1', complex128[:,:]),
                 ('cplx_value_2', complex128[:,:]),
                 ('cplx_value_3', complex128[:,:]),
                 ('cplx_value_4', complex128[:,:]),
                 ('cplx_value_5', complex128[:,:]),
                 ('cplx_value_6', complex128[:,:]),
                 ('cplx_value_7', complex128[:,:]),
                 ('cplx_value_8', complex128[:,:]),
                 ('cplx_value_9', complex128[:,:]),
                 ('cplx_value_10', complex128[:,:]),
                 ('cplx_value_11', complex128[:,:]),
                 ('cplx_value_12', complex128[:,:])                 
                 ]

#@numba.jitclass(spec)
class Car(object):

    def __init__(self):
        self.value = np.zeros((2000,1))
        h1 = House()
        #self.cplx_value_1 = np.zeros((2000,1),dtype=np.complex128)
        
    #@property
    def twice(self):
        return  np.abs(self.cplx_value_1 )
    
    def twice1(self):
        return  np.abs(self.cplx_value_1 )

    def twice2(self):
        return  np.abs(self.cplx_value_1 )
    
spec = [('value', float64[:,:]),
                 ('cplx_value_1', complex128[:,:]),
                 ('cplx_value_2', complex128[:,:]),
                 ('cplx_value_3', complex128[:,:]),
                 ('cplx_value_4', complex128[:,:]),
                 ('cplx_value_5', complex128[:,:]),
                 ('cplx_value_6', complex128[:,:]),
                 ('cplx_value_7', complex128[:,:]),
                 ('cplx_value_8', complex128[:,:]),
                 ('cplx_value_9', complex128[:,:]),
                 ('cplx_value_10', complex128[:,:]),
                 ('cplx_value_11', complex128[:,:]),
                 ('cplx_value_12', complex128[:,:])                 
                 ]

@numba.jitclass(spec)
class House(object):

    def __init__(self):
        self.value = np.zeros((2000,1))
        #self.cplx_value_1 = np.zeros((2000,1),dtype=np.complex128)
        
    #@property
    def twice(self):
        return  np.abs(self.cplx_value_1 )
    
    def twice1(self):
        return  np.abs(self.cplx_value_1 )

    def twice2(self):
        return  np.abs(self.cplx_value_1 )
    
    
    
if __name__ == "__main__":
    import time
    
#    t0 = time.time()
#    sys1 = system()
#    print('time: {:f}'.format(time.time()-t0))
# 
#    t0 = time.time()
#    sys1 = system()
#    print('time: {:f}'.format(time.time()-t0))
    
    t0 = time.time()
    sys1 = pydss('cigre_lv_isolated.json')
    sys1.pf_eval()
    sys1.run_eval()
    print('time: {:f}'.format(time.time()-t0))

#    t0 = time.time()
#    sys1 = pydss('cigre_lv_isolated.json')
#    sys1.system.run_eval()
#    print('time: {:f}'.format(time.time()-t0))
       
#    odss = opendss()
#    
#    odss.read_v_results('/home/jmmauricio/Documents/public/workspace/pydss/pydss/opendss/cigre_lv_VLN_Node.Txt')
#    
#    sys1 = pydss('cigre_lv_isolated.json')
#    sys1 = pydss('bench_3bus.json')
##    print(opendss(sys1))
#    sys1.pf_eval()
#    sys1.get_v()
#    sys1.system.run_eval()
#    pq_3pn_int = sys1.pq_3pn_int
#    pq_3pn = sys1.pq_3pn
#    Y_ii = sys1.Y_ii
#    Y_iv = sys1.Y_iv
#    Y_vv = sys1.Y_vv
#    Y_vi = sys1.Y_vi  
#    inv_Y_ii = sys1.inv_Y_ii
#    
#    V_known = sys1.V_known
#    
#    N_nodes_i = sys1.N_nodes_i 
#    N_nodes_v = sys1.N_nodes_v
#    N_nodes = sys1.N_nodes
#    V_node = np.zeros((sys1.N_nodes,1),dtype=np.complex128)
#    I_node = np.zeros((sys1.N_nodes,1),dtype=np.complex128)
#    
#    
#    I_known_0 = np.zeros((sys1.N_nodes_i,1),dtype=np.complex128)
#    V_unknown_0 =  np.zeros((sys1.N_nodes_i,1),dtype=np.complex128)+231   
##        
#    o= run()
#    sys1.get_v()
#    sys1.get_i()
#    sys1.bokeh_tools()
#    

    #        
#        
#        
#        V_unknown = V_unknown_0
#        for it in range(max_iter):
#            if len(i_fp_modes)>0:
#                I_known[i_fp_modes] = known_1[i_fp_modes]*np.exp(1j*(np.angle(V_unknown[i_fp_modes]) + known_2[i_fp_modes]))
#            # if len(pq_modes)>0:
#            #    I_known[pq_modes] = 1000.0*np.conj((known_1[pq_modes] +1j*known_2[pq_modes])/V_unknown[pq_modes])
#    
#            I_known[3::4] = I_known[0::4]+I_known[1::4]+I_known[2::4]
#            V_unknown =inv(Y_ii)@(I_known - Y_iv @ V_known)
#            
#            error = np.abs((V_unknown - V_unknown_0))
#            if np.max(error) < 1.0e-6: break
#            
#            V_unknown_0 = V_unknown
#    
#        I_unknown =Y_vv @ V_known + Y_vi @ V_unknown
#        
#        return V_unknown,I_unknown,I_known,it
        #return it        
#sys1 = syst()
##print(sys1.pq_3pn_int)
##print(sys1.pq_3pn)
##
##print(sys1.pq_3p_int)
##print(sys1.pq_3p)
##
##print(sys1.nodes)
#
#sys1.pf_eval()
##print(sys1.pf_eval())
#t_0 = time.time(); sys1.pf.pf_eval(); print(time.time()-t_0)
#
#fobj = open('matrix.txt','w')
#
#I_known = sys1.pf.I_known
#V_unknown = sys1.pf.V_unknown
#V_known = sys1.pf.V_known
#
#V = np.vstack((V_known,V_unknown))
#S = V_unknown*np.conj(I_known)
#mat = np.hstack((np.abs(sys1.pf.I_known),np.angle(sys1.pf.I_known,deg=True)))
#mat = np.hstack((np.abs(sys1.pf.V_unknown),np.angle(sys1.pf.V_unknown,deg=True)))
#mat = np.abs(V[0::4]-V[1::4])
##mat = np.hstack((np.abs(S/1000),S.real/1000,S.imag/1000))
#for it_row in range(mat.shape[0]):
#    fobj.write(sys1.nodes[4*it_row]+ ' ' + sys1.nodes[4*it_row+1] +' ')
#    for it_col in range(mat.shape[1]):
#        
#        fobj.write('{:2.2f} '.format(mat[it_row,it_col]))
#    fobj.write('\n')
#fobj.close()
#        