#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 12:24:53 2017

@author: jmmauricio
"""


def sym2num(symsystem,out_file):
    
    f = symsystem['f']
    g = symsystem['g']
    x = symsystem['x']
    y = symsystem['y']
    u = symsystem['u']
    F_x = symsystem['F_x']
    F_y = symsystem['F_y']
    G_x = symsystem['G_x']
    G_y = symsystem['G_y']
    params = symsystem['params']
        
    N_x = f.shape[0]
    N_y = g.shape[0]
    N_u = u.shape[0]

    ## Class
    string =  'import numpy as np\n'
    string += 'import numba\n'

    tab = '\n    '
    tab2 = '\n        ' 
    string += 'class system(object):\n'
    string += tab + 'def __init__(self,json_file=None):\n'
    string += tab2 + "self.N_x = {:d}".format(N_x)
    string += tab2 + "self.N_y = {:d}".format(N_y)
    string += tab2 + "dt_list = []"
    string += tab2 + "struct_list = []"
    string += tab2 + "dt_list += [('N_x',np.int32)]".format(N_x)
    string += tab2 + "dt_list += [('N_y',np.int32)]".format(N_y)
    string += tab2 + "struct_list += [{:d}]".format(N_x)
    string += tab2 + "struct_list += [{:d}]".format(N_y)

    string += tab2 + "dt_list += [('x',np.float64,({:d},1))]".format(N_x)
    string += tab2 + "dt_list += [('y',np.float64,({:d},1))]".format(N_y)
    string += tab2 + "dt_list += [('u',np.float64,({:d},1))]".format(N_u)
    string += tab2 + "struct_list += [np.zeros(({:d},1))]".format(N_x)
    string += tab2 + "struct_list += [np.zeros(({:d},1))]".format(N_y)
    string += tab2 + "struct_list += [np.zeros(({:d},1))]".format(N_u)

    string += tab2 + "dt_list += [('f',np.float64,({:d},1))]".format(N_x)
    string += tab2 + "dt_list += [('g',np.float64,({:d},1))]".format(N_y)
    string += tab2 + "struct_list += [np.zeros(({:d},1))]".format(N_x)
    string += tab2 + "struct_list += [np.zeros(({:d},1))]".format(N_y)

    string += tab2 + "dt_list += [('F_x',np.float64,({:d},{:d}))]".format(N_x,N_x)
    string += tab2 + "dt_list += [('F_y',np.float64,({:d},{:d}))]".format(N_x,N_y)
    string += tab2 + "dt_list += [('G_x',np.float64,({:d},{:d}))]".format(N_y,N_x)
    string += tab2 + "dt_list += [('G_y',np.float64,({:d},{:d}))]".format(N_y,N_y)

    string += tab2 + "struct_list += [np.zeros(({:d},{:d}))]".format(N_x,N_x)
    string += tab2 + "struct_list += [np.zeros(({:d},{:d}))]".format(N_x,N_y)
    string += tab2 + "struct_list += [np.zeros(({:d},{:d}))]".format(N_y,N_x)
    string += tab2 + "struct_list += [np.zeros(({:d},{:d}))]".format(N_y,N_y)


    string += "\n"
    for item in params: # constants
        string += tab2 + "dt_list += [('{:s}',np.float64)]".format(item)

    string += "\n"
    for item in params: # constants
        string += tab2 + "struct_list += [{:f}] # {:s}".format(params[item], item)

    string += tab2 + "self.struct = np.rec.array([struct_list],dtype=dt_list) \n"  


    string += tab + 'def ss(self,xi):\n'
    string += tab2 + "x = xi[0:self.N_x]"
    string += tab2 + "y = xi[self.N_x:(self.N_x+self.N_y)]"
    string += tab2 + "self.struct['x'] = x"
    string += tab2 + "self.struct['y'] = y"
    string += tab2 + 'update(self.struct,0,0)\n'
    string += tab2 + "lam = np.vstack((self.struct['f'],self.struct['g']))"
    string += tab2 + "return lam"


    ## update funtion
    string += "\n"*3   
    string += "@numba.jit(nopython=True, cache=True)\n"    
    string += 'def update(struct,call,item):\n'

    string += "\n"
    for it in range(N_x): # dynamic states
        string += "    {:s} = struct[item]['x'][{:s},0]\n".format(str(x[it]),str(it))

    string += "\n"
    for it in range(N_y): # algebraic states
        string += "    {:s} = struct[item]['y'][{:s},0]\n".format(str(y[it]),str(it))

    string += "\n"
    for it in range(N_u): # inputs
        string += "    {:s} = struct[item]['u'][{:s},0]\n".format(str(u[it]),str(it))

    string += "\n"
    for item in params: # constants
        string += "    {:s} = struct[item]['{:s}']\n".format(item, item)


    string += "\n"
    for it in range(N_x): # dynamic equations
        string += '    d{:s} = {:s}\n'.format(str(x[it]),str(f[it]))

    string += "\n"
    for it in range(N_x): # dynamic equations
        string += "    struct[item]['f'][{:s},0] = d{:s} \n".format(str(it),str(x[it]))

    string += "\n" * 3    

    string += "\n"
    for it in range(N_y): # algebraic equations
        string += '    g_{:s} = {:s}\n'.format(str(it),str(g[it]))

    string += "\n"
    for it in range(N_y): # algebraic equations
        string += "    struct[item]['g'][{:s},0] = g_{:s} \n".format(str(it),str(it))


    string += "\n"
    for ix in range(N_x): # F_x
        for iy in range(N_x): 
            element = F_x[ix,iy]
            if element!=0:
                string += tab + "struct[item]['F_x'][{:s},{:s}] = {:s} ".format(str(ix),str(iy),str(F_x[ix,iy]))

    string += "\n"
    for ix in range(N_x): # F_y
        for iy in range(N_y): 
            element = F_y[ix,iy]
            if element!=0:
                string += tab + "struct[item]['F_y'][{:s},{:s}] = {:s} ".format(str(ix),str(iy),str(F_y[ix,iy]))

    string += "\n"
    for ix in range(N_y): # G_x
        for iy in range(N_x): 
            element = G_x[ix,iy]
            if element!=0:
                string += tab + "struct[item]['G_x'][{:s},{:s}] = {:s} ".format(str(ix),str(iy),str(G_x[ix,iy]))

    string += "\n"
    for ix in range(N_y): # G_y
        for iy in range(N_y): 
            element = G_y[ix,iy]
            if element!=0:
                string += tab + "struct[item]['G_y'][{:s},{:s}] = {:s} ".format(str(ix),str(iy),str(G_y[ix,iy]))
    string += "\n"


    replaces = [('cos','np.cos'),('sin','np.sin')]
    for item in replaces:
        string = string.replace(item[0],item[1])
    fobj = open(out_file,'w')    
    fobj.write(string)
    fobj.close()
