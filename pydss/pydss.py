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
from pf import pf_eval,set_load_factor
from run import run_eval
from electric import vsc,vsc_former
from ctrl import secondary
import time
from scipy import sparse
from scipy.sparse import linalg as sla

line_codes_lib = {'OH1':[[0.540 + 0.777j, 0.049 + 0.505j, 0.049 + 0.462j, 0.049 + 0.436j],
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
         'UG1_luna':[[0.211 + 0.747j, 0.049 +0.6657j, 0.049 +0.6657j, 0.049 +0.6657j], 
                     [0.049 +0.6657j, 0.211 + 0.747j, 0.049 +0.6657j, 0.049 +0.6657j], 
                     [0.049 +0.6657j, 0.049 +0.6657j, 0.211 + 0.747j, 0.049 +0.6657j], 
                     [0.049 +0.6657j, 0.049 +0.6657j, 0.049 +0.6657j, 0.211 + 0.747j]],
              'UG2':[[0.314 + 0.762j, 0.049 + 0.687j,0.049 + 0.665j, 0.049 + 0.687j], 
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
                     [0.049 + 0.719j, 0.049 + 0.719j, 0.871 + 0.797j]],
              'UG1w3':[[ 0.28700247+0.16535143j,  0.12115403+0.11008501j,0.12500247+0.06935143j],
                       [ 0.12115403+0.11008501j,  0.27947509+0.20221853j,0.12115403+0.11008501j],
                       [ 0.12500247+0.06935143j,  0.12115403+0.11008501j,0.28700247+0.16535143j]],
              'UG3w3':[[ 1.15225232+0.45874501j, 0.32098424+0.39046333j, 0.33025232+0.35874501j],
                       [ 0.32098424+0.39046333j, 1.13401861+0.4779049j, 0.32098424+0.39046333j],
                       [ 0.33025232+0.35874501j,  0.32098424+0.39046333j,1.15225232+0.45874501j]]                       
              }


class pydss(object):
    '''
    
    
    
    P+N : 1
    3P  : 2
    3P+N: 3
    
    '''
    
    
    def __init__(self):
        self.a = 0


    def read(self,json_file):        
        self.json_file = json_file
        self.json_data = open(json_file).read().replace("'",'"')
        data = json.loads(self.json_data)
        self.data = data
        
        flog = open('log.txt','w')
        
        # power flow options
        self.max_iter = 20
        
        # run options
        self.N_steps = 1000
        self.Dt = 10.0e-3
        self.Dt_out = 0.01
        
        if 'transformers' in data:
            transformers = data['transformers']
        else:
            transformers = []
        lines = data['lines']
        line_codes_data = data['line_codes']
        loads = data['loads']
        v_sources = data['v_sources']
        buses = data['buses']

        if 'groundings' in data:
            groundings = data['groundings']
        else:
            groundings = []
            
        self.transformers = transformers
        self.lines = lines
        self.loads = loads
        self.buses = buses
        
        N_nodes_default = 4
        nodes = []
        A_n_cols = 0
        it_col = 0
        v_sources_nodes = []

        node_sorter = []
        N_v_known = 0
        N_nz_nodes = 0  # number on non zero current nodes
        
        
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

        N_nz_nodes += N_v_known
        
        ## Known currents
        S_known_list = []
        pq_3pn_int_list = []
        pq_3pn_list = []
        pq_3p_int_list = []
        pq_3p_list = []
        pq_1pn_int_list = []
        pq_1pn_list = []
        pq_1p_int_list = []
        pq_1p_list = []
        it_node_i = 0
        
        ### Loads
        t_0 = time.time()
        for load in loads:
            if not 'bus_nodes' in load:   # if nodes are not declared, default nodes are created
                if load['type']=='3P':
                    load.update({'bus_nodes': list(range(1,3+1))})
                if load['type']=='3P+N':
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
                pq_3p_int_list += [list(it_node_i + np.array([0,1,2]))]
                it_node_i += 3
                if 'kVA' in load:
                    S_va = -1000.0*load['kVA']
                    phi = np.arccos(load['fp'])
                    if type(load['kVA']) == float:
                        pq_3p_list += [[S_va*np.exp(1j*phi*np.sign(load['fp']))]]

            if load['type'] == '1P+N':
                node_j = '{:s}.{:s}'.format(load['bus'],str(load['bus_nodes'][0]))
                if not node_j in nodes: nodes +=[node_j]
                node_k = '{:s}.{:s}'.format(load['bus'],str(load['bus_nodes'][1]))
                if not node_k in nodes: nodes +=[node_k]
                pq_1pn_int_list += [np.array([nodes.index(node_j),nodes.index(node_k)])-N_v_known]
                it_node_i += 2
                if 'kVA' in load:
                    S_va = -1000.0*load['kVA']
                    phi = np.arccos(load['fp'])
                    pq_1pn_list += [[S_va*np.exp(1j*phi*np.sign(load['fp']))]]
                if 'kW' in load:
                    S_va = -1000.0*load['kW']/load['fp']
                    phi = np.arccos(load['fp'])
                    pq_1pn_list += [[S_va*np.exp(1j*phi*np.sign(load['fp']))]]
        
        
            if load['type'] == '1P':
                pq_1p_int_list += [[it_node_i ]]
                it_node_i += 1
                if 'kVA' in load:
                    S_va = -1000.0*load['kVA']
                    phi = np.arccos(load['fp'])
                    pq_1p_list += [[S_va*np.exp(1j*phi*np.sign(load['fp']))]]
                if 'kW' in load:
                    S_va = -1000.0*load['kW']/load['fp']
                    phi = np.arccos(load['fp'])
                    pq_1p_list += [[S_va*np.exp(1j*phi*np.sign(load['fp']))]]

        
        N_pq_3p = len(pq_3p_list)
        N_pq_3pn = len(pq_3pn_list)
        N_pq_1p = len(pq_1p_list)
        N_pq_1pn = len(pq_1pn_list)
        N_nz_nodes += it_node_i                               
#            for kVA,fp in  zip(load['kVA'],load['fp']): # known complex power list 
#                S_known_list += [1000.0*kVA*np.exp(1j*np.arccos(fp)*np.sign(fp))]
        pq_3pn_int = np.array(pq_3pn_int_list) # known complex power list to numpy array
        pq_3pn = np.array(pq_3pn_list) # known complex power list to numpy array
        pq_3p_int = np.array(pq_3p_int_list) # known complex power list to numpy array
        pq_3p = np.array(pq_3p_list) # known complex power list to numpy array
        pq_1pn_int = np.array(pq_1pn_int_list) # known complex power list to numpy array
        pq_1pn = np.array(pq_1pn_list) # known complex power list to numpy array
        pq_1p_int = np.array(pq_1p_int_list) # known complex power list to numpy array
        pq_1p = np.array(pq_1p_list) # known complex power list to numpy array


        ### Transformers to nodes
        for trafo in transformers:
            if trafo['connection'] == 'Dyn11':
                N_nodes_primary_default = trafo['conductors_1']
                N_nodes_secondary_default = trafo['conductors_2']
                N_trafo_nodes = N_nodes_primary_default+N_nodes_secondary_default
                
            if trafo['connection'] == 'Dyg11_3w':
                N_nodes_primary_default = trafo['conductors_1']
                N_nodes_secondary_default = trafo['conductors_2']
                N_trafo_nodes = N_nodes_primary_default+N_nodes_secondary_default
                
            A_n_cols += N_trafo_nodes
            if not 'bus_j_nodes' in trafo:   # if nodes are not declared, default nodes are created
                trafo.update({'bus_j_nodes': list(range(1,N_nodes_primary_default+1))})
            if not 'bus_k_nodes' in trafo:   # if nodes are not declared, default nodes are created
                trafo.update({'bus_k_nodes': list(range(1,N_nodes_secondary_default+1))})

            for item in  trafo['bus_j_nodes']: # the list of nodes '[<bus>.<node>.<node>...]' is created 
                node_j = '{:s}.{:s}'.format(trafo['bus_j'], str(item))
                if not node_j in nodes: nodes +=[node_j]
            for item in  trafo['bus_k_nodes']: # the list of nodes '[<bus>.<node>.<node>...]' is created 
                node_k = '{:s}.{:s}'.format(trafo['bus_k'], str(item))
                if not node_k in nodes: nodes +=[node_k]

        flog.write('Transformers read in: {:2.3f} s'.format(time.time()-t_0)) 
        t_0 = time.time()
                
        ### Lines to nodes
        for line in lines:
            line_code = line['code']
            if not line_code in line_codes_lib:
                R = np.array(data['line_codes'][line_code]['R'])
                X = np.array(data['line_codes'][line_code]['X'])
                Z = R + 1j*X
                line_codes_lib.update({line_code:Z.tolist()})
            N_conductors = len(line_codes_lib[line['code']])
            A_n_cols += N_conductors
            if not 'bus_j_nodes' in line:   # if nodes are not declared, default nodes are created
                line.update({'bus_j_nodes': list(range(1,N_conductors+1))})
            if not 'bus_k_nodes' in line:   # if nodes are not declared, default nodes are created
                line.update({'bus_k_nodes': list(range(1,N_conductors+1))})

            for item in  line['bus_j_nodes']: # the list of nodes '[<bus>.<node>.<node>...]' is created 
                node_j = '{:s}.{:s}'.format(line['bus_j'], str(item))
                if not node_j in nodes: nodes +=[node_j]
            for item in  line['bus_k_nodes']: # the list of nodes '[<bus>.<node>.<node>...]' is created 
                node_k = '{:s}.{:s}'.format(line['bus_k'], str(item))
                if not node_k in nodes: nodes +=[node_k]

        N_nodes = len(nodes)

        ### Groundings (no new nodes, only aditionals columns in A matriz)
        for ground in groundings:
            A_n_cols += 1        
        
        
        ## Y primitive
        A = np.zeros((N_nodes,A_n_cols)) # incidence matrix (dense)
        A_sp = sparse.lil_matrix((N_nodes, A_n_cols), dtype=np.float32) # incidence matrix (sparse)
        it_col = 0  # column in incidence matrix
        
        
        ### Transformers to Y primitive
        Y_trafos_prims =  []
        for trafo in transformers:
            S_n = trafo['S_n_kVA']*1000.0
            U_1n = trafo['U_1_kV']*1000.0
            U_2n = trafo['U_2_kV']*1000.0
            Z_cc_pu = trafo['R_cc_pu'] +1j*trafo['X_cc_pu']
            connection = trafo['connection']
            
            Y_trafo_prim = trafo_yprim(S_n,U_1n,U_2n,Z_cc_pu,connection=connection)
            Y_trafos_prims +=  [Y_trafo_prim]

            for item in  trafo['bus_j_nodes']: # the list of nodes '[<bus>.<node>.<node>...]' is created 
                node_j = '{:s}.{:s}'.format(trafo['bus_j'], str(item))
                row = nodes.index(node_j)
                col = it_col
                A_sp[row,col] = 1
                A[row,col] = 1
                it_col +=1  
    
            for item in  trafo['bus_k_nodes']: # the list of nodes '[<bus>.<node>.<node>...]' is created 
                node_k = '{:s}.{:s}'.format(trafo['bus_k'], str(item))
                row = nodes.index(node_k)
                col = it_col
                A_sp[row,col] = 1
                A[row,col] = 1
                it_col +=1 
 
        Y_trafos_primitive = diag_2d(Y_trafos_prims)        # dense
        Y_trafos_primitive = diag_2dsparse(Y_trafos_prims)  # sparse
        N_trafos_len = Y_trafos_primitive.shape[0]
        
        ### Lines to Y primitive       
        Z_line_list =  []
        for line in lines:
            for item in  line['bus_j_nodes']: # the list of nodes '[<bus>.<node>.<node>...]' is created 
                node_j = '{:s}.{:s}'.format(line['bus_j'], str(item))
                row = nodes.index(node_j)
                col = it_col
                A_sp[row,col] = 1
                A[row,col] = 1
                #it_col +=1   
    
            #for item in  line['bus_k_nodes']: # the list of nodes '[<bus>.<node>.<node>...]' is created 
                node_k = '{:s}.{:s}'.format(line['bus_k'], str(item))
                row = nodes.index(node_k)
                col = it_col
                A_sp[row,col] = -1
                A[row,col] = -1
                it_col +=1   

            Z_line_list += [line['m']*0.001*np.array(line_codes_lib[line['code']])]   # Line code to list of Z lines

        ### Groundings       
        for grounding in groundings:
            for item in  grounding['bus_nodes']: # the list of nodes '[<bus>.<node>.<node>...]' is created 
                node_j = '{:s}.{:s}'.format(grounding['bus'], str(item))
                row = nodes.index(node_j)
                col = it_col
                A_sp[row,col] = 1
                A[row,col] = 1
                it_col +=1  
                Z_line_list += [np.array(grounding['R'] + 1j*grounding['X']).reshape((1,1))]   # Line code to list of Z lines
                
                
        Y_lines_primitive = diag_2d_inv(Z_line_list)        # dense
        Y_lines_primitive = diag_2dsparse_inv(Z_line_list)  # sparse
        
        N_lines_len  = Y_lines_primitive.shape[0]
        
        ### Full  Y primitive         
        N_prim = N_trafos_len + N_lines_len
       
#        #### dense
#        if N_trafos_len>0:
#            Y_primitive = np.vstack((np.hstack((Y_trafos_primitive,np.zeros((N_trafos_len,N_lines_len)))),
#                                     np.hstack((np.zeros((N_lines_len,N_trafos_len)),Y_lines_primitive))))
#        else:
#            Y_primitive = Y_lines_primitive
            
       
        #### sparse
        Y_primitive_sp = sparse.csc_matrix((N_prim,N_prim),dtype=np.complex128)
       
        Y_primitive_sp[N_trafos_len:N_prim,N_trafos_len:N_prim] = Y_lines_primitive     

        
        if N_trafos_len>0:
            Y_primitive_sp[0:N_trafos_len,0:N_trafos_len] = Y_trafos_primitive

        A_v = A[0:N_v_known,:]   
        N_nodes_i = N_nodes-N_v_known
        A_i = A_sp[N_v_known:(N_v_known+N_nodes_i),:] 

        self.A = A
        self.nodes = nodes
        self.N_nodes = len(nodes)
        self.N_nodes_i = N_nodes_i
        self.N_nodes_v = self.N_nodes  - N_nodes_i
        self.A_v = A_v
        self.A_i = A_i

        
#        self.Y = A.T @ Y_primitive @ A (dense)
        self.Y = A_sp @ Y_primitive_sp @ A_sp.T        
        self.Y_primitive = Y_primitive_sp.toarray()
        self.Y_ii = A_i @ Y_primitive_sp @ A_i.T
        self.Y_iv = A_i @ Y_primitive_sp @ A_v.T
        self.Y_vv = A_v @ Y_primitive_sp @ A_v.T
        self.Y_vi = A_v @ Y_primitive_sp @ A_i.T

        flog.write('Ys calc {:2.3f}\n'.format(time.time()-t_0))
        t_0 = time.time()
        
        #self.inv_Y_ii = sparse.linalg.inv()
        self.inv_Y_ii = inv_splu(sparse.csc_matrix(self.Y_ii))        
        flog.write('inv_Y_ii {:2.3f}\n'.format(time.time()-t_0))
        
        self.N_pq_3pn = N_pq_3pn
        self.pq_3pn_int = pq_3pn_int
        self.pq_3pn = pq_3pn
        self.N_pq_3p = N_pq_3p
        self.pq_3p_int = pq_3p_int
        self.pq_3p = pq_3p
        self.N_pq_1p = N_pq_1p
        self.pq_1p_int = pq_1p_int
        self.pq_1p = pq_1p
        self.N_pq_1pn = N_pq_1pn
        self.pq_1pn_int = pq_1pn_int
        self.pq_1pn = pq_1pn
        self.V_known = V_known
        
        self.I_node = np.vstack((np.zeros((self.N_nodes_v,1)),
                                 np.zeros((self.N_nodes_i,1))))+0j

        self.A_n_cols = A_n_cols
        self.Y_primitive_sp = Y_primitive_sp
        self.A_sp = A_sp
        
        self.N_nz_nodes = N_nz_nodes
 
        node_sorter = []
        for bus in self.buses:
            N_nodes = 0
            for node in range(10):
                bus_node = '{:s}.{:s}'.format(str(bus['bus']),str(node))
                if bus_node in self.nodes:
                    node_idx = self.nodes.index(bus_node) 
                    node_sorter += [node_idx]
                    N_nodes += 1
                bus.update({'N_nodes':N_nodes})
        self.node_sorter = node_sorter

        flog.close()        



    def pf(self):
        
        N_i = self.N_nodes_i
        N_v = self.N_nodes_v 
        
        V_unknown_0 = np.zeros((self.N_nodes_i,1),dtype=np.complex128) 

        
        
#        for it in range(int(self.N_nodes_i/4)): # change if not 4 wires
#            
#            V_unknown_0[4*it+0] = self.V_known[0]
#            V_unknown_0[4*it+1] = self.V_known[1]
#            V_unknown_0[4*it+2] = self.V_known[2]
#            V_unknown_0[4*it+3] = 0.0
#        
        buses_in_data_file = [item['bus'] for item in self.buses]
        it = 0
        for node in self.nodes[N_v:]:
            bus,node = node.split('.')
            self.theta_0 = np.deg2rad(0.0)
            
            V_m_nom = self.buses[buses_in_data_file.index(bus)]['U_kV']*1000.0/np.sqrt(3)
            if node == '1': 
                V_unknown_0[it] = V_m_nom*np.exp(1j*(self.theta_0))
            if node == '2': 
                V_unknown_0[it] = V_m_nom*np.exp(1j*(self.theta_0-2.0/3.0*np.pi))
            if node == '3': 
                V_unknown_0[it] = V_m_nom*np.exp(1j*(self.theta_0-4.0/3.0*np.pi))
            if node == '4': 
                V_unknown_0[it] = 0.0
                
            it+=1
            
            
            
        self.V_node = np.vstack((self.V_known,V_unknown_0 ))
        
        if self.pq_3pn_int.shape[0] == 0:
            self.pq_3pn_int = np.array([[0,0,0,0]])
            self.pq_3pn = np.array([[0,0,0]])
            
        if self.pq_3p_int.shape[0] == 0:
            self.pq_3p_int = np.array([[0,0,0]])
            self.pq_3p = np.array([[0,0,0]])
            
        if self.pq_1p_int.shape[0] == 0:
            self.pq_1p_int = np.array([[0]])
            self.pq_1p = np.array([[0]])
            
        dt_pf = np.dtype([
                  ('Y_vv',np.complex128,(N_v,N_v)),('Y_iv',np.complex128,(N_i,N_v)),
                  ('inv_Y_ii',np.complex128,(N_i,N_i)),('Y_ii',np.complex128,(N_i,N_i)),
                  ('I_node',np.complex128,(N_v+N_i,1)),('V_node',np.complex128,(N_v+N_i,1)),
                  ('N_pq_1p',np.int32),('pq_1p_int',np.int32,self.pq_1p_int.shape),('pq_1p',np.complex128,self.pq_1p.shape),('pq_1p_0',np.complex128,self.pq_1p.shape),                  
                  ('N_pq_1pn',np.int32),('pq_1pn_int',np.int32,self.pq_1pn_int.shape),('pq_1pn',np.complex128,self.pq_1pn.shape),('pq_1pn_0',np.complex128,self.pq_1pn.shape),
                  ('N_pq_3p',np.int32),('pq_3p_int',np.int32,self.pq_3p_int.shape),('pq_3p',np.complex128,self.pq_3p.shape),('pq_3p_0',np.complex128,self.pq_3p.shape),
                  ('N_pq_3pn',np.int32),('pq_3pn_int',np.int32,self.pq_3pn_int.shape),('pq_3pn',np.complex128,self.pq_3pn.shape),('pq_3pn_0',np.complex128,self.pq_3pn.shape),
                  ('N_nodes_v',np.int32),('N_nodes_i',np.int32),('iters',np.int32),('N_nz_nodes',np.int32)] )
    

        params_pf = np.rec.array([(
                                self.Y_vv,self.Y_iv,
                                self.inv_Y_ii,self.Y_ii.toarray(), 
                                self.I_node,self.V_node,
                                self.N_pq_1p, self.pq_1p_int,self.pq_1p,np.copy(self.pq_1p),
                                self.N_pq_1pn, self.pq_1pn_int,self.pq_1pn,np.copy(self.pq_1pn),
                                self.N_pq_3p, self.pq_3p_int,self.pq_3p,np.copy(self.pq_3p),
                                self.N_pq_3pn, self.pq_3pn_int,self.pq_3pn,np.copy(self.pq_3pn),
                                self.N_nodes_v,self.N_nodes_i,0,self.N_nz_nodes)],dtype=dt_pf)  
                  
        V_node,I_node = pf_eval(params_pf) 

        self.V_node = V_node
        self.I_node = I_node 
        self.params_pf = params_pf 
        
    def read_loads_shapes(self,json_file):        
        self.json_file = json_file
        self.json_data = open(json_file).read().replace("'",'"')
        data = json.loads(self.json_data)
        self.load_shapes = data

        ts_list = []
        shapes_list = []
        N_loads = 0
        for load in self.loads:
            shape_id = load['shape']
            ts_list += [self.load_shapes[shape_id]['t_s']]
            shapes_list += [self.load_shapes[shape_id]['shape']]
            N_times = len(self.load_shapes[shape_id]['shape'])
            N_loads += 1                
        dtype = np.dtype([('time',np.float64,(N_loads,N_times)),
                          ('shapes',np.float64,(N_loads,N_times)),
                          ('N_loads',np.int32), ('N_times',np.int32)])
        self.params_lshapes = np.rec.array([(np.array(ts_list),np.array(shapes_list),
                                       N_loads, N_times)],dtype=dtype) 
       
    def read_perturbations(self):
        
        buses_names = [item['bus'] for item in self.loads]
        p = self.data['perturbations']
        N_perturbations = len(p)
        
        load_new_values_list = []
        perturbations_int = []
        perturbations_times_list = []
        perturbations_types_list = []
        
        for it in range(N_perturbations):
            if self.data['perturbations'][it]['type'] == 'load_new_value':
                load_new_values_list += [np.hstack((np.array(p[it]['kw_abc'])*1000.0+np.array(p[it]['kvar_abc'])*1000.0j,np.array([0.0])))] 
                perturbations_times_list += [p[it]['time']]
                perturbations_types_list += [[1]]
                perturbations_int += [buses_names.index(p[it]['bus'])]
                
        self.N_perturbations = N_perturbations  
        
        if self.N_perturbations>0:
            self.load_new_values = np.array(load_new_values_list)
            self.perturbations_int = np.array(perturbations_int).reshape(N_perturbations,1)
            self.perturbations_times = np.array(perturbations_times_list).reshape(N_perturbations,1)     
            self.perturbations_types = np.array(perturbations_types_list).reshape(N_perturbations,1)
        
        
    def run(self):

            self.read_perturbations()
            
            if 'secondary' in self.data:
                secondary_obj = secondary(self.json_file)
            
            if 'vsc' in self.data: vsc_objs = vsc(self.json_file)
            if 'vsc_former' in self.data: vsc_former_objs = vsc_former(self.json_file)
            
            params_secondary = secondary_obj.params_secondary
            self.params_secondary = params_secondary           
            
            params_vsc = vsc_former_objs.params_vsc
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
                               ('T_j_igbt_abcn', np.complex128,(N_outs,4*len(self.params_vsc))),
                               ('T_sink', np.complex128,(N_outs,len(self.params_vsc))),
                               ('out_cplx_i', np.complex128,(N_outs,N_nodes)),
                               ('out_cplx_v', np.complex128,(N_outs,N_nodes)),
                               ('N_outs', 'int32'),
                               ('perturbations_int', 'int32', (self.N_perturbations,1)),
                               ('perturbations_types', 'int32', (self.N_perturbations,1)),
                               ('perturbations_times', np.float64, (self.N_perturbations,1)),
                               ('perturbations_cplx', np.complex128,(self.N_perturbations,4)),
                               ])  
            
            
            params_run = np.rec.array([(N_steps,
                                        Dt,
                                        Dt_out,
                                        np.zeros((N_outs,1)), # T
                                        np.zeros((N_outs,4*len(self.params_vsc))), # T_j_igbt_abcn
                                        np.zeros((N_outs,len(self.params_vsc))), # T_sink
                                        np.zeros((N_outs,N_nodes)),
                                        np.zeros((N_outs,N_nodes)),                                       
                                        N_outs,
                                        self.perturbations_int,
                                        self.perturbations_types,
                                        self.perturbations_times,
                                        self.load_new_values,                                   
                                        )],dtype=dt_run)    
                  
            self.params_run = params_run
            
            run_eval(params_run,self.params_pf,params_vsc,params_secondary)
            
#            params_run[0].out_cplx_i = params_run[0].out_cplx_i[0:params_run[0].N_outs,:]
            
                


            


    def get_v(self):
        
        V_sorted = []
        I_sorted = []
        S_sorted = []
        start_node = 0
        self.V_results = self.V_node
        self.I_results = self.I_node
        
        V_sorted = self.V_node[self.node_sorter]
        I_sorted = self.I_node[self.node_sorter]   
        
        nodes2string = ['v_an','v_bn','v_cn','v_gn']
        for bus in self.buses:
            N_nodes = bus['N_nodes'] 
#            for node in range(5):
#                bus_node = '{:s}.{:s}'.format(str(bus['bus']),str(node))
#                if bus_node in self.nodes:
#                    V = self.V_results[self.nodes.index(bus_node)][0]
#                    V_sorted += [V]
#                    nodes_in_bus += [node]
#            for node in range(5):
#                bus_node = '{:s}.{:s}'.format(str(bus['bus']),str(node))
#                if bus_node in self.nodes:
#                    I = self.I_results[self.nodes.index(bus_node)][0]
#                    I_sorted += [I]
            if N_nodes==3:   # if 3 phases
                v_ag = V_sorted[start_node+0,0]
                v_bg = V_sorted[start_node+1,0]
                v_cg = V_sorted[start_node+2,0]

                i_a = I_sorted[start_node+0,0]
                i_b = I_sorted[start_node+1,0]
                i_c = I_sorted[start_node+2,0]
                
                s_a = (v_ag)*np.conj(i_a)
                s_b = (v_bg)*np.conj(i_b)
                s_c = (v_cg)*np.conj(i_c)
                
                start_node += 3
                bus.update({'v_an':np.abs(v_ag),
                            'v_bn':np.abs(v_bg),
                            'v_cn':np.abs(v_cg),
                            'v_ng':0.0})
                bus.update({'deg_an':np.angle(v_ag, deg=True),
                            'deg_bn':np.angle(v_bg, deg=True),
                            'deg_cn':np.angle(v_cg, deg=True),
                            'deg_ng':np.angle(0, deg=True)})
                bus.update({'v_ab':np.abs(v_ag-v_bg),
                            'v_bc':np.abs(v_bg-v_cg),
                            'v_ca':np.abs(v_cg-v_ag)})
                bus.update({'p_a':s_a.real,
                            'p_b':s_b.real,
                            'p_c':s_c.real})
                bus.update({'q_a':s_a.imag,
                            'q_b':s_b.imag,
                            'q_c':s_c.imag})
            if N_nodes==4:   # if 3 phases + neutral
                v_ag = V_sorted[start_node+0,0]
                v_bg = V_sorted[start_node+1,0]
                v_cg = V_sorted[start_node+2,0]
                v_ng = V_sorted[start_node+3,0]
                i_a = I_sorted[start_node+0,0]
                i_b = I_sorted[start_node+1,0]
                i_c = I_sorted[start_node+2,0]
                i_n = I_sorted[start_node+3,0]  
                
                v_an = v_ag-v_ng
                v_bn = v_bg-v_ng                
                v_cn = v_cg-v_ng
                
                s_a = (v_an)*np.conj(i_a)
                s_b = (v_bn)*np.conj(i_b)
                s_c = (v_cn)*np.conj(i_c)
                bus.update({'v_an':np.abs(v_an),
                            'v_bn':np.abs(v_bn),
                            'v_cn':np.abs(v_cn),
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
    
                start_node += 4
        self.V = np.array(V_sorted).reshape(len(V_sorted),1) 
        return 0 #self.V              
        
    def get_i(self):

                
        t_0 = time.time()
        
        I_lines = self.Y_primitive_sp @ self.A_sp.T @ self.V_results
        
        t_0 = time.time()
        
        it_single_line = 0
        for trafo in self.transformers:

            cond_1 = trafo['conductors_1'] 
            cond_2 = trafo['conductors_2']   
            
            I_1a = (I_lines[it_single_line,0])
            I_1b = (I_lines[it_single_line+1,0])
            I_1c = (I_lines[it_single_line+2,0])
            I_1n = (I_lines[it_single_line+3,0])
            
            I_2a = (I_lines[it_single_line+cond_1+0,0])
            I_2b = (I_lines[it_single_line+cond_1+1,0])
            I_2c = (I_lines[it_single_line+cond_1+2,0])
            I_2n = (I_lines[it_single_line+cond_1+3,0])
            
            #I_n = (I_lines[it_single_line+3,0])
            if cond_1 <=3:
                I_1n = I_1a+I_1b+I_1c
            if cond_2 <=3:
                I_2n = I_2a+I_2b+I_2c
                
            it_single_line += cond_1 + cond_2
            trafo.update({'i_1a_m':np.abs(I_1a)})
            trafo.update({'i_1b_m':np.abs(I_1b)})
            trafo.update({'i_1c_m':np.abs(I_1c)})
            trafo.update({'i_1n_m':np.abs(I_1n)})
            trafo.update({'i_2a_m':np.abs(I_2a)})
            trafo.update({'i_2b_m':np.abs(I_2b)})
            trafo.update({'i_2c_m':np.abs(I_2c)})
            trafo.update({'i_2n_m':np.abs(I_2n)})
            trafo.update({'deg_1a':np.angle(I_1a, deg=True)})
            trafo.update({'deg_1b':np.angle(I_1b, deg=True)})
            trafo.update({'deg_1c':np.angle(I_1c, deg=True)})
            trafo.update({'deg_1n':np.angle(I_1n, deg=True)})
            trafo.update({'deg_2a':np.angle(I_2a, deg=True)})
            trafo.update({'deg_2b':np.angle(I_2b, deg=True)})
            trafo.update({'deg_2c':np.angle(I_2c, deg=True)})
            trafo.update({'deg_2n':np.angle(I_2n, deg=True)})
                        
        self.I_lines = I_lines
        for line in self.lines:
            N_conductors = len(line['bus_j_nodes'])
            if N_conductors == 3:
                I_a = (I_lines[it_single_line,0])
                I_b = (I_lines[it_single_line+1,0])
                I_c = (I_lines[it_single_line+2,0])
                #I_n = (I_lines[it_single_line+3,0])
                I_n = I_a+I_b+I_c
                it_single_line += N_conductors
                line.update({'i_a_m':np.abs(I_a)})
                line.update({'i_b_m':np.abs(I_b)})
                line.update({'i_c_m':np.abs(I_c)})
                line.update({'i_n_m':np.abs(I_n)})
                line.update({'deg_a':np.angle(I_a, deg=True)})
                line.update({'deg_b':np.angle(I_b, deg=True)})
                line.update({'deg_c':np.angle(I_c, deg=True)})
                line.update({'deg_n':np.angle(I_n, deg=True)})
            if N_conductors == 4:
                I_a = (I_lines[it_single_line,0])
                I_b = (I_lines[it_single_line+1,0])
                I_c = (I_lines[it_single_line+2,0])
                I_n = (I_lines[it_single_line+3,0])
                it_single_line += N_conductors
                line.update({'i_a_m':np.abs(I_a)})
                line.update({'i_b_m':np.abs(I_b)})
                line.update({'i_c_m':np.abs(I_c)})
                line.update({'i_n_m':np.abs(I_n)})
                line.update({'deg_a':np.angle(I_a, deg=True)})
                line.update({'deg_b':np.angle(I_b, deg=True)})
                line.update({'deg_c':np.angle(I_c, deg=True)})
                line.update({'deg_n':np.angle(I_n, deg=True)})            
        self.I_lines = I_lines
                        

        t_0 = time.time()            

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
        v_an = ['{:2.2f}'.format(float(item['v_an'])) for item in self.buses]
        v_bn = ['{:2.2f}'.format(float(item['v_bn'])) for item in self.buses]
        v_cn = ['{:2.2f}'.format(float(item['v_cn'])) for item in self.buses]
        v_ng = ['{:2.2f}'.format(float(item['v_ng'])) for item in self.buses]

        deg_an = ['{:2.2f}'.format(float(item['deg_an'])) for item in self.buses]
        deg_bn = ['{:2.2f}'.format(float(item['deg_bn'])) for item in self.buses]
        deg_cn = ['{:2.2f}'.format(float(item['deg_cn'])) for item in self.buses]
        deg_ng = ['{:2.2f}'.format(float(item['deg_ng'])) for item in self.buses]
        v_ab = [item['v_ab'] for item in self.buses]
        v_bc = [item['v_bc'] for item in self.buses]
        v_ca = [item['v_ca'] for item in self.buses]
        p_a = ['{:2.2f}'.format(float(item['p_a']/1000)) for item in self.buses]
        p_b = ['{:2.2f}'.format(float(item['p_b']/1000)) for item in self.buses]
        p_c = ['{:2.2f}'.format(float(item['p_c']/1000)) for item in self.buses]
        q_a = ['{:2.2f}'.format(float(item['q_a']/1000)) for item in self.buses]
        q_b = ['{:2.2f}'.format(float(item['q_b']/1000)) for item in self.buses]
        q_c = ['{:2.2f}'.format(float(item['q_c']/1000)) for item in self.buses]   
        p_abc = ['{:2.2f}'.format(float((item['p_a'] +item['p_b']+item['p_c'])/1000)) for item in self.buses] 
        q_abc = ['{:2.2f}'.format(float((item['q_a'] +item['q_b']+item['q_c'])/1000)) for item in self.buses]
        s_radio = []
        for item in self.buses:            
            s_total = np.abs(item['p_a'] + 1j*item['q_a']) + np.abs(item['p_b'] + 1j*item['q_b']) +  np.abs(item['p_c'] + 1j*item['q_c'])
            scale = 0.01
            s_scaled = s_total*scale
            if s_scaled>20.0:
                s_scaled = 20.0
            s_radio += [s_scaled]
        self.bus_data = dict(x=x, y=y, bus_id=bus_id,
                             v_an=v_an, v_bn=v_bn, v_cn=v_cn, v_ng=v_ng, 
                             deg_an=deg_an, deg_bn=deg_bn, deg_cn=deg_cn, 
                             deg_ng=deg_ng,v_ab=v_ab,v_bc=v_bc,v_ca=v_ca,
                             p_a=p_a,p_b=p_b,p_c=p_c,
                             q_a=q_a,q_b=q_b,q_c=q_c,
                             p_abc=p_abc,q_abc=q_abc,
                             s_radio=s_radio)
        
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


def diag_2d_inv(Z_line_list):

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

def diag_2dsparse_inv(Z_line_list):

    N_cols = 0

    for Z_line in Z_line_list:
        N_cols += Z_line.shape[1]

    Y_lines = sparse.lil_matrix((N_cols,N_cols),dtype=np.complex128)

    it = 0
    for Z_line in Z_line_list:
        Y_line = np.linalg.inv(Z_line)
        N = Y_line.shape[0] 
        Y_lines[it:(it+N),it:(it+N)] = Y_line
        it += N

    return Y_lines

def diag_2dsparse(Y_prim_list):

    N_cols = 0

    for Y_prim in Y_prim_list:
        N_cols += Y_prim.shape[1]

    Y_prims = sparse.lil_matrix((N_cols,N_cols),dtype=np.complex128)

    it = 0
    for Y_prim in Y_prim_list:
        N = Y_prim.shape[0] 
        Y_prims[it:(it+N),it:(it+N)] = Y_prim
        it += N

    return Y_prims

def diag_2d(Y_prim_list):

    N_cols = 0

    for Y_prim in Y_prim_list:
        N_cols += Y_prim.shape[1]

    Y_prims = np.zeros((N_cols,N_cols),dtype=np.complex128)

    it = 0
    for Y_prim in Y_prim_list:
        N = Y_prim.shape[0] 
        Y_prims[it:(it+N),it:(it+N)] = Y_prim
        it += N

    return Y_prims

def inv_splu(A_sparse):
    N = A_sparse.shape[0]
    lu = sla.splu(A_sparse)
    return lu.solve(np.eye(N,dtype=np.complex128))
    #return np.linalg.inv(A_sparse.toarray())

def trafo_yprim(S_n,U_1n,U_2n,Z_cc,connection='Dyg11'):
    '''
    Trafo primitive as developed in: (in the paper Ynd11)
    R. C. Dugan and S. Santoso, “An example of 3-phase transformer modeling for distribution system analysis,” 
    2003 IEEE PES Transm. Distrib. Conf. Expo. (IEEE Cat. No.03CH37495), vol. 3, pp. 1028–1032, 2003. 
    
    '''

    if connection=='Dyn1':
        z_a = Z_cc*1.0**2/S_n*3
        z_b = Z_cc*1.0**2/S_n*3
        z_c = Z_cc*1.0**2/S_n*3
        U_1 = U_1n
        U_2 = U_2n/np.sqrt(3)
        Z_B = np.array([[z_a, 0.0, 0.0],
                        [0.0, z_b, 0.0],
                        [0.0, 0.0, z_c],])                             
        N_a = np.array([[ 1/U_1,     0],
                         [-1/U_1,     0],
                         [     0, 1/U_2],
                         [     0,-1/U_2]])           
        N_row_a = np.hstack((N_a,np.zeros((4,4))))
        N_row_b = np.hstack((np.zeros((4,2)),N_a,np.zeros((4,2))))
        N_row_c = np.hstack((np.zeros((4,4)),N_a))
        
        N = np.vstack((N_row_a,N_row_b,N_row_c))

        B = np.array([[ 1, 0, 0],
                      [-1, 0, 0],
                      [ 0, 1, 0],
                      [ 0,-1, 0],
                      [ 0, 0, 1],
                      [ 0, 0,-1]])
    
        Y_1 = B @ np.linalg.inv(Z_B) @ B.T
        Y_w = N @ Y_1 @ N.T
        A_trafo = np.zeros((7,12))

        A_trafo[0,0] = 1.0
        A_trafo[0,9] = 1.0
        A_trafo[1,1] = 1.0
        A_trafo[1,4] = 1.0
        A_trafo[2,5] = 1.0
        A_trafo[2,8] = 1.0

        A_trafo[3,2] = 1.0
        A_trafo[4,6] = 1.0
        A_trafo[5,10] = 1.0
        
        A_trafo[6,3] = 1.0
        A_trafo[6,7] = 1.0
        A_trafo[6,11] = 1.0


    if connection=='Dyn5':
        z_a = Z_cc*1.0**2/S_n*3
        z_b = Z_cc*1.0**2/S_n*3
        z_c = Z_cc*1.0**2/S_n*3
        U_1 = U_1n
        U_2 = U_2n/np.sqrt(3)
        Z_B = np.array([[z_a, 0.0, 0.0],
                        [0.0, z_b, 0.0],
                        [0.0, 0.0, z_c],])                             
        N_a = np.array([[ 1/U_1,     0],
                         [-1/U_1,     0],
                         [     0, 1/U_2],
                         [     0,-1/U_2]])           
        N_row_a = np.hstack((N_a,np.zeros((4,4))))
        N_row_b = np.hstack((np.zeros((4,2)),N_a,np.zeros((4,2))))
        N_row_c = np.hstack((np.zeros((4,4)),N_a))
        
        N = np.vstack((N_row_a,N_row_b,N_row_c))

        B = np.array([[ 1, 0, 0],
                      [-1, 0, 0],
                      [ 0, 1, 0],
                      [ 0,-1, 0],
                      [ 0, 0, 1],
                      [ 0, 0,-1]])
    
        Y_1 = B @ np.linalg.inv(Z_B) @ B.T
        Y_w = N @ Y_1 @ N.T
        A_trafo = np.zeros((7,12))

        A_trafo[0,1] = 1.0
        A_trafo[0,4] = 1.0
        A_trafo[1,5] = 1.0
        A_trafo[1,8] = 1.0
        A_trafo[2,0] = 1.0
        A_trafo[2,9] = 1.0

        A_trafo[3,2] = 1.0
        A_trafo[4,6] = 1.0
        A_trafo[5,10] = 1.0
        
        A_trafo[6,3] = 1.0
        A_trafo[6,7] = 1.0
        A_trafo[6,11] = 1.0


    if connection=='Dyn11':
        z_a = Z_cc*1.0**2/S_n*3
        z_b = Z_cc*1.0**2/S_n*3
        z_c = Z_cc*1.0**2/S_n*3
        U_1 = U_1n
        U_2 = U_2n/np.sqrt(3)
        Z_B = np.array([[z_a, 0.0, 0.0],
                        [0.0, z_b, 0.0],
                        [0.0, 0.0, z_c],])                             
        N_a = np.array([[ 1/U_1,     0],
                         [-1/U_1,     0],
                         [     0, 1/U_2],
                         [     0,-1/U_2]])           
        N_row_a = np.hstack((N_a,np.zeros((4,4))))
        N_row_b = np.hstack((np.zeros((4,2)),N_a,np.zeros((4,2))))
        N_row_c = np.hstack((np.zeros((4,4)),N_a))
        
        N = np.vstack((N_row_a,N_row_b,N_row_c))

        B = np.array([[ 1, 0, 0],
                      [-1, 0, 0],
                      [ 0, 1, 0],
                      [ 0,-1, 0],
                      [ 0, 0, 1],
                      [ 0, 0,-1]])
    
        Y_1 = B @ np.linalg.inv(Z_B) @ B.T
        Y_w = N @ Y_1 @ N.T
        A_trafo = np.zeros((7,12))

        A_trafo[0,1] = 1.0
        A_trafo[0,4] = 1.0
        A_trafo[1,5] = 1.0
        A_trafo[1,8] = 1.0
        A_trafo[2,0] = 1.0
        A_trafo[2,9] = 1.0

        A_trafo[3,3] = 1.0
        A_trafo[4,7] = 1.0
        A_trafo[5,11] = 1.0
        
        A_trafo[6,2] = 1.0
        A_trafo[6,6] = 1.0
        A_trafo[6,10] = 1.0

        
    if connection=='Dyg11_3w':
        z_a = Z_cc*1.0**2/S_n*3
        z_b = Z_cc*1.0**2/S_n*3
        z_c = Z_cc*1.0**2/S_n*3
        U_1 = U_1n
        U_2 = U_2n/np.sqrt(3)
        Z_B = np.array([[z_a, 0.0, 0.0],
                        [0.0, z_b, 0.0],
                        [0.0, 0.0, z_c],])                             
        N_a = np.array([[ 1/U_1,     0],
                         [-1/U_1,     0],
                         [     0, 1/U_2],
                         [     0,-1/U_2]])           
        N_row_a = np.hstack((N_a,np.zeros((4,4))))
        N_row_b = np.hstack((np.zeros((4,2)),N_a,np.zeros((4,2))))
        N_row_c = np.hstack((np.zeros((4,4)),N_a))
        
        N = np.vstack((N_row_a,N_row_b,N_row_c))

        B = np.array([[ 1, 0, 0],
                      [-1, 0, 0],
                      [ 0, 1, 0],
                      [ 0,-1, 0],
                      [ 0, 0, 1],
                      [ 0, 0,-1]])
    
        Y_1 = B @ np.linalg.inv(Z_B) @ B.T
        Y_w = N @ Y_1 @ N.T
        A_trafo = np.zeros((6,12))

        A_trafo[0,0] = 1.0
        A_trafo[0,9] = 1.0
        A_trafo[1,1] = 1.0
        A_trafo[1,4] = 1.0
        A_trafo[2,5] = 1.0
        A_trafo[2,8] = 1.0

        A_trafo[3,2]  = 1.0
        A_trafo[4,6]  = 1.0
        A_trafo[5,10] = 1.0
                
    if connection=='Ynd11':
        z_a = Z_cc*1.0**2/S_n*3
        z_b = Z_cc*1.0**2/S_n*3
        z_c = Z_cc*1.0**2/S_n*3
        U_1 = U_1n/np.sqrt(3)
        U_2 = U_2n
        Z_B = np.array([[z_a, 0.0, 0.0],
                        [0.0, z_b, 0.0],
                        [0.0, 0.0, z_c],])   

        B = np.array([[ 1, 0, 0],
                      [-1, 0, 0],
                      [ 0, 1, 0],
                      [ 0,-1, 0],
                      [ 0, 0, 1],
                      [ 0, 0,-1]])
                          
        N_a = np.array([[ 1/U_1,     0],
                        [-1/U_1,     0],
                        [     0, 1/U_2],
                        [     0,-1/U_2]])           
        N_row_a = np.hstack((N_a,np.zeros((4,4))))
        N_row_b = np.hstack((np.zeros((4,2)),N_a,np.zeros((4,2))))
        N_row_c = np.hstack((np.zeros((4,4)),N_a))
        
        N = np.vstack((N_row_a,N_row_b,N_row_c))

        Y_1 = B @ np.linalg.inv(Z_B) @ B.T
        Y_w = N @ Y_1 @ N.T
        A_trafo = np.zeros((7,12))
        A_trafo[0,0] = 1.0
        A_trafo[1,4] = 1.0
        A_trafo[2,8] = 1.0
        
        A_trafo[3,1] = 1.0
        A_trafo[3,5] = 1.0
        A_trafo[3,9] = 1.0
        
        A_trafo[4,2] = 1.0
        A_trafo[4,11] = 1.0
        A_trafo[5,3] = 1.0
        A_trafo[5,6] = 1.0
        A_trafo[6,7] = 1.0
        A_trafo[6,10] = 1.0
        
        
    Y_prim = A_trafo @ Y_w @ A_trafo.T
    
    return Y_prim


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
    

   


def abcn2abc(Z_abcn):
    '''
    From the primitive impedance matrix, the phase impedance matrix 
    can be obtained from Kron reduction:
    
    '''
    Z_pp = Z_abcn[0:3,0:3] 
    Z_pn = Z_abcn[0:3,3:]
    Z_np = Z_abcn[3:, 0:3]
    Z_nn = Z_abcn[3:,3:]
    Z_abc = Z_pp - Z_pn @ np.linalg.inv(Z_nn) @ Z_np
    return Z_abc
    




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

   
def opendss2pydss(self,files_dict):
    
    
    return files_dict  

if __name__ == "__main__":
    import time 
    test ='luna_1'

    if test=='luna_1': 
        sys1 = pydss()
        t_0 = time.time()
        sys1.read('./luna/luna_1_4w.json')  # Load data
        sys1.pf()
#        sys1.get_v()
#        sys1.get_i()
#        print('iters: ', sys1.params_pf['iters'])
        
    if test=='cigre_lv': 
        sys1 = pydss()
        t_0 = time.time()
        sys1.read('cigre_lv.json')  # Load data
        print('sys1.read()',time.time()-t_0) 
        t_0 = time.time()
        sys1.pf()
        print('sys1.pf()',time.time()-t_0) 
        t_0 = time.time()
        
    if test=='lv_europe_connected_load1': 
        sys1 = pydss()
        t_0 = time.time()
        t_0 = time.time()
        sys1.read('lv_europe_connected_load1.json')  # Load data
        print('sys1.read()',time.time()-t_0) 
        t_0 = time.time()
        sys1.pf()
        print('sys1.pf()',time.time()-t_0) 
        t_0 = time.time()

    if test=='bus4_1p_load':
        sys1 = pydss()
        sys1.read('bus4_1p_load.json')
        sys1.pf()
        sys1.get_v()
        sys1.get_i()
        #sys1.bokeh_tools()
        
    if test=='lveurope':
        sys1 = pydss()
        sys1.read('lv_europe_connected.json')  # Load data
        sys1.pf()
        sys1.get_v()
        sys1.get_i()
        sys1.bokeh_tools()
        
    if test=='opendss2pydss':
        trafos_file = './Feeder_1/Transformers.txt'
        buses_file = './Feeder_1/buscoord.dss'
        linecodes_file = './Feeder_1/LineCode.txt'
        lines_file = './Feeder_1/Lines.txt'        
        loads_file = './Feeder_1/Loads.txt'
        load_shapes_file = './Feeder_1/LoadShapes.txt' 
        line_dict = {}
        
        buses = []
        
        ## Transformers
        
        fobj_trafos = open(trafos_file, 'r')
        trafos_list = [] 
        for trafo in fobj_trafos.readlines():
            item_odss = 'Buses=['
            start_idx = trafo.find(item_odss)
            end_idx = trafo.find(']',start_idx)
            value = trafo[((start_idx+len(item_odss))):end_idx]
            bus_j,bus_k = value.split(' ') 

            item_odss = 'Conns=['
            start_idx = trafo.find(item_odss)
            end_idx = trafo.find(']',start_idx)
            value = trafo[((start_idx+len(item_odss))):end_idx]
            conn_j,conn_k = value.split(' ') 
            
            item_odss = 'kVs=['
            start_idx = trafo.find(item_odss)
            end_idx = trafo.find(']',start_idx)
            value = trafo[((start_idx+len(item_odss))):end_idx]
            U_1_kV,U_2_kV = value.split(' ') 
                       
            item_odss = 'kVAs=['
            start_idx = trafo.find(item_odss)
            end_idx = trafo.find(']',start_idx)
            value = trafo[((start_idx+len(item_odss))):end_idx]
            S_n_kVA,S_n_kVA_2 = value.split(' ')

            item_odss = 'XHL='
            start_idx = trafo.find(item_odss)
            end_idx = trafo.find(']',start_idx)
            value = trafo[((start_idx+len(item_odss))):end_idx]
            X_pu= value
            
            if conn_j == 'Delta' and conn_k == 'Wye':
                connection = 'Dyg11_3w'
                trafo_dict = {"bus_j": bus_j,  "bus_k": bus_k,  "S_n_kVA": float(S_n_kVA), 
                              "U_1_kV":float(U_1_kV), "U_2_kV":float(U_2_kV), "R_cc_pu": float(0.0001), 
                              "X_cc_pu":float(X_pu)/100.0, "connection": "Dyg11_3w", "conductors_1":3, "conductors_2":3 }

   
            if not bus_j in buses:
                buses += [bus_j]
            if not bus_k in buses:
                buses += [bus_k]                
            trafos_list += [trafo_dict]                       
        
        ## Lines
        
        fobj_lines = open(lines_file, 'r')
        
        read_list = [('Bus1','bus_j','str',1),
                     ('Bus2','bus_k','str',1),
                     ('Linecode','code','str',1),
                     ('Length','m','float',1.0),
                     ('phases','N_conductors','int',1)]
        
        lines_list = []
        
        for line in fobj_lines.readlines():
            line_dict = {}
            for item_odss,item_py,tipo,scale in read_list:
                start_idx = line.find(item_odss + '=')
                end_idx = line.find(' ',start_idx)
                
                value = line[((start_idx+len(item_odss))+1):end_idx]
                if tipo=='float':
                    value = float(value)*scale
                if tipo=='int':
                    value = int(value)*scale               
                
                line_dict.update({item_py:value})
            
            bus_j = line_dict['bus_j']
            if not bus_j in buses:
                buses += [bus_j]
            bus_k = line_dict['bus_k']
            if not bus_k in buses:
                buses += [bus_k]                
            
            lines_list += [line_dict]    
            
            
        ## Loads
        loads_list = []
        load_dict = {}
        fobj_loads = open(loads_file, 'r')
        
        read_list = [('Phases','phases','int',1),
                     ('Bus1','bus','str',1),
                     ('kW','kW','float',1),
                     ('PF','fp','float',1.0),
                     ('Daily','shape','str',1)]
        
        load_list = []
        for load in fobj_loads.readlines():
            load_dict = {}
            for item_odss,item_py,tipo,scale in read_list:
                start_idx = load.find(item_odss + '=')
                end_idx = load.find(' ',start_idx)
                
                value = load[((start_idx+len(item_odss))+1):end_idx]
                if tipo=='float':
                    value = float(value)*scale
                if tipo=='int':
                    value = int(value)*scale               
                        
                if item_odss == 'Bus1':
                    if load_dict['phases'] == 1:
                        value,node = value.split('.') 
                        
                        load_dict.update({'bus_nodes':[int(node)],'type':'1P'})
                        
                load_dict.update({item_py:value})
            loads_list += [load_dict]

            bus = load_dict['bus']
            if not bus in buses:
                buses += [bus]

           

        ## Buses
        
        buses_list = []
        buses_array = np.genfromtxt(buses_file, delimiter=',', skip_header=1, dtype=[('nodes','S10'),('pos_x','f8'),('pos_y','f8')])   
          
        for bus in buses:
            idx = np.where(buses_array['nodes']==np.array(bus,dtype='S10'))[0].astype(np.int32)
            print(idx)
            pos_x = buses_array['pos_x'][idx]
            pos_y = buses_array['pos_y'][idx]
            buses_dict = {"bus": bus,  
                          "pos_x": float(pos_x), 
                          "pos_y": float(pos_y), 
                          "units": "m", 'U_kV':0.416}
            buses_list += [buses_dict]                        
            
        ## Line codes
        
        fobj_linecodes = open(linecodes_file, 'r')

        alpha = np.exp(2.0/3*np.pi*1j)
        Asym =  np.array([[1, 1, 1],
                          [1, alpha**2, alpha],
                          [1, alpha, alpha**2]])
        inv_Asym = np.linalg.inv(Asym)

        line_codes = {}
        for linecode in fobj_linecodes.readlines():
            item_odss = 'LineCode.'
            start_idx = linecode.find(item_odss)
            end_idx = linecode.find(' ',start_idx)
            linecode_id = linecode[((start_idx+len(item_odss))):end_idx]
            l = {}
            for item in ['R1','R0','X1','X0','C1','C0']:
                item_odss = item + '='
                start_idx = linecode.find(item_odss)
                end_idx = linecode.find(' ',start_idx)
                value = linecode[((start_idx+len(item_odss))):end_idx]
                l.update({item:float(value)})
            Z_012 = np.array([[l['R0']+1j*l['X0'],0,0],
                              [0,l['R1']+1j*l['X1'],0],
                              [0,0,l['R1']+1j*l['X1']]])
            Z_abc = inv_Asym @ Z_012 @ Asym
            
            line_codes.update({linecode_id:{'R':Z_abc.real.tolist(),
                                            'X':Z_abc.imag.tolist()}})
            json.dumps(line_codes)    
        
        v_sources_list = [
		{"bus": "SourceBus","bus_nodes": [1, 2, 3], "deg": [0, -120, -240], "kV": [11.547, 11.547, 11.547]}]
        
        from collections import OrderedDict
        

 
    
        # Load Shapes
        fobj_loadshapes = open(load_shapes_file, 'r')    
        
        load_shapes_list = [] 
        for load_shape in fobj_loadshapes.readlines():
            item_odss = 'New Loadshape.'
            start_idx = load_shape.find(item_odss)
            end_idx = load_shape.find(' ',start_idx+4)
            shape_id = load_shape[((start_idx+len(item_odss))):end_idx]

            item_odss = 'npts='
            start_idx = load_shape.find(item_odss)
            end_idx = load_shape.find(' ',start_idx)
            npts = int(load_shape[((start_idx+len(item_odss))):end_idx])

            item_odss = 'interval='
            start_idx = load_shape.find(item_odss)
            end_idx = load_shape.find(' ',start_idx)
            interval = float(load_shape[((start_idx+len(item_odss))):end_idx])
            
            item_odss = 'mult=(file='
            start_idx = load_shape.find(item_odss)
            end_idx = load_shape.find(')',start_idx+4)
            shape_file = load_shape[((start_idx+len(item_odss))):end_idx]
            shape_file = shape_file.replace('\\','/')
            #fobj_shape = open(, 'r') 
            values = np.loadtxt(shape_file)
            t_s = np.linspace(0,npts*interval*3600,npts)
            load_shapes_list += [(shape_id,{'t_s':t_s.tolist(), 'shape':values.tolist()})] #, 't_s':t_s.tolist(), 'shape':values.tolist()}]
        dict_shapes = OrderedDict(load_shapes_list)

        out = json.dumps(dict_shapes) 
        out = out.replace('},','},\n')
        out = out.replace('],','],\n')
        fobj_out = open('load_shapes_list.json','w')
        fobj_out.write(out)
        fobj_out.close()
        
        
 
        data_dict = OrderedDict([('transformers',trafos_list),
                     ('lines',lines_list),
                     ('buses',buses_list),
                     ('v_sources',v_sources_list),
                     ('loads',loads_list),
                     ('line_codes',line_codes)])
        out = json.dumps(data_dict) 
        out = out.replace('},','},\n')
        out = out.replace('],','],\n')
        fobj_out = open('out.json','w')
        fobj_out.write(out)
        fobj_out.close()            
            
        json_data = open('out.json').read().replace("'",'"')
        data = json.loads(json_data)
        
#        sys1 = pydss('out.json')
#        sys1.pf()
#        sys1.get_v()
#        sys1.get_i()

#            item_odss = 'kVAs=['
#            start_idx = trafo.find(item_odss)
#            end_idx = trafo.find(']',start_idx)
#            value = trafo[((start_idx+len(item_odss))):end_idx]
#            S_n_kVA,S_n_kVA_2 = value.split(' ')
#
#            item_odss = 'XHL='
#            start_idx = trafo.find(item_odss)
#            end_idx = trafo.find(']',start_idx)
#            value = trafo[((start_idx+len(item_odss))):end_idx]
#            X_pu= value
#            
#            if conn_j == 'Delta' and conn_k == 'Wye':
#                connection = 'Dyg11_3w'
#                trafo_dict = {"bus_j": "R0",  "bus_k": "R1",  "S_n_kVA": float(S_n_kVA), 
#                              "U_1_kV":float(U_1_kV), "U_2_kV":float(U_2_kV), "R_cc_pu": float(0.0001), 
#                              "X_cc_pu":float(X_pu)/100.0, "connection": "Dyg11_3w", "conductors_1":3, "conductors_2":3 }





              
#            print(load_dict)


            
#            Bus1=762 Bus2=770 phases=3 Linecode=4c_.06 Length=5.2347 Units=m
#            lines_list = []
#    t0 = time.time()
#    sys1 = system()
#    print('time: {:f}'.format(time.time()-t0))
# 
#    t0 = time.time()
#    sys1 = system()
#    print('time: {:f}'.format(time.time()-t0))
    
#    t0 = time.time()
##    sys1 = pydss('cigre_lv_isolated_3gformers.json')
##    sys1 = pydss('thermal_test.json')
#    sys1 = pydss('cigre_lv_isolated.json')
#    sys1.pf()
#    sys1.run()
##    print('time: {:f}'.format(time.time()-t0))

    if test=='pf':
        sys1 = pydss('cigre_lv_connected.json')
        sys1.pf()
        sys1.get_v()      # post process voltages
        sys1.get_i()      # post process currents
#        print(sys1.buses[0])
#        print(sys1.buses[1])
#        print(sys1.buses[10])
#        sys1.bokeh_tools()

    if test=='pf_3w':
        sys1 = pydss('cigre_lv_connected_3w.json')
        sys1.pf()
        sys1.get_v()      # post process voltages
#       sys1.get_i()      # post process currents
#        print(sys1.buses[0])
#        print(sys1.buses[1])
        print(sys1.buses[10]['v_an'])
        print(sys1.buses[10]['v_bn'])
        print(sys1.buses[10]['v_cn'])
#        sys1.bokeh_tools()
        

    if test=='trafo':
        S_n = 630.0e3
        U_1n = 400.0
        U_2n = 20.0e3
        Z_cc_pu = 0.01+0.04j
        Y_trafo_prim = trafo_yprim(S_n,U_1n,U_2n,Z_cc_pu,type='Ynd11')
        
        Z_UG3_3w = abcn2abc(np.array(line_codes['UG3'])) 

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