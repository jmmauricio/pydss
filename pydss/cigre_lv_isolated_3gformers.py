{
"lines":[
        {"bus_j": "R1",  "bus_k": "R2",  "code": "UG1", "m": 35.0 },
        {"bus_j": "R2",  "bus_k": "R3",  "code": "UG1", "m": 35.0 },
        {"bus_j": "R3",  "bus_k": "R4",  "code": "UG1", "m": 35.0 },
        {"bus_j": "R4",  "bus_k": "R5",  "code": "UG1", "m": 35.0 },
        {"bus_j": "R5",  "bus_k": "R6",  "code": "UG1", "m": 35.0 },
        {"bus_j": "R6",  "bus_k": "R7",  "code": "UG1", "m": 35.0 },
        {"bus_j": "R7",  "bus_k": "R8",  "code": "UG1", "m": 35.0 },
        {"bus_j": "R8",  "bus_k": "R9",  "code": "UG1", "m": 35.0 },
        {"bus_j": "R9",  "bus_k": "R10", "code": "UG1", "m": 35.0 },
        {"bus_j": "R3",  "bus_k": "R11", "code": "UG3", "m": 30.0 },
        {"bus_j": "R4",  "bus_k": "R12", "code": "UG3", "m": 35.0 },
        {"bus_j": "R12", "bus_k": "R13", "code": "UG3", "m": 35.0 },
        {"bus_j": "R13", "bus_k": "R14", "code": "UG3", "m": 35.0 },
        {"bus_j": "R14", "bus_k": "R15", "code": "UG3", "m": 30.0 },
        {"bus_j": "R6",  "bus_k": "R16", "code": "UG3", "m": 30.0 },
        {"bus_j": "R9",  "bus_k": "R17", "code": "UG3", "m": 30.0 },
        {"bus_j": "R10", "bus_k": "R18", "code": "UG3", "m": 30.0 }
        ],
"buses":[
		{"bus": "R1",  "Vbase":0.231, "pos_x": 0, "pos_y":   0, "units": "m"},
		{"bus": "R2",  "Vbase":0.231, "pos_x": 0, "pos_y": -35, "units": "m"},
		{"bus": "R3",  "Vbase":0.231, "pos_x": 0, "pos_y": -70, "units": "m"},
		{"bus": "R4",  "Vbase":0.231, "pos_x": 0, "pos_y": -105, "units": "m"},
		{"bus": "R5",  "Vbase":0.231, "pos_x": 0, "pos_y": -140, "units": "m"},
		{"bus": "R6",  "Vbase":0.231, "pos_x": 0, "pos_y": -175, "units": "m"},
		{"bus": "R7",  "Vbase":0.231, "pos_x": 0, "pos_y": -210, "units": "m"},
		{"bus": "R8",  "Vbase":0.231, "pos_x": 0, "pos_y": -245, "units": "m"},
		{"bus": "R9",  "Vbase":0.231, "pos_x": 0, "pos_y": -280, "units": "m"},
		{"bus": "R10", "Vbase":0.231, "pos_x": 0, "pos_y": -315, "units": "m"},
		{"bus": "R11", "Vbase":0.231, "pos_x": -30, "pos_y": -70, "units": "m"},
		{"bus": "R12", "Vbase":0.231, "pos_x": 35, "pos_y": -105, "units": "m"},
		{"bus": "R13", "Vbase":0.231, "pos_x": 70, "pos_y": -105, "units": "m"},
		{"bus": "R14", "Vbase":0.231, "pos_x": 105, "pos_y": -105, "units": "m"},
		{"bus": "R15", "Vbase":0.231, "pos_x": 105, "pos_y": -140, "units": "m"},
		{"bus": "R16", "Vbase":0.231, "pos_x": -35, "pos_y": -175, "units": "m"},
		{"bus": "R17", "Vbase":0.231, "pos_x": 30, "pos_y": -280, "units": "m"},
		{"bus": "R18", "Vbase":0.231, "pos_x": -30, "pos_y": -315, "units": "m"}
		],
"v_sources":[
         {"bus": "R2", "bus_nodes": [1, 2, 3, 4], "deg": [-30, -150, -270, 0], "kV": [0.23094, 0.23094, 0.23094, 0]},
		{"bus": "R14","bus_nodes": [1, 2, 3, 4], "deg": [-30, -150, -270, 0], "kV": [0.23094, 0.23094, 0.23094, 0]},
		{"bus": "R10","bus_nodes": [1, 2, 3, 4], "deg": [-30, -150, -270, 0], "kV": [0.23094, 0.23094, 0.23094, 0]}
		],
"loads":[
	    {"bus": "R1" , "kVA": 0.0, "fp": 0.95, "type":"3P+N"},
        {"bus": "R11", "kVA": 15.0, "fp": 0.95, "type":"3P+N"},
        {"bus": "R15", "kVA": 52.0, "fp": 0.95, "type":"3P+N"},
        {"bus": "R16", "kVA": 55.0, "fp": 0.95, "type":"3P+N"},
        {"bus": "R17", "kVA": 35.0, "fp": 0.95, "type":"3P+N"},
        {"bus": "R18", "kVA": 47.0, "fp": 0.95, "type":"3P+N"}  
        ],
"vsc":[  {"s_n_kVA":120.0, "V_dc":700.0, "ctrl_mode":3, "K_v":0.3, "K_ang":0.05, "K_f":0.05, "T_v":0.05, "T_ang":0.5, "K_p":0.02, "K_i":0.5, "nodes_vknown":[0,1,2,3],
           "a_i":1.5257e+01, "b_i":-2.7435e-02, "c_i":9.5767e-03, "d_i":7.8929e-03, "e_i":4.0265e-03, 
           "a_d":6.2857e-01, "b_d":1.0374e-01, "c_d":-1.2941e-01, "d_d":3.0008e-03, "e_d":-1.3320e-03,
           "Rth_sink":0.0129, "Rth_c_igbt":0.02, "Rth_c_diode":0.05, "Rth_j_igbt":0.15, "Rth_j_diode":0.15,"T_a":25.0, 
           "Cth_sink":6.9767, "N_switch_sink":6}, 
		{"s_n_kVA":120.0, "V_dc":700.0, "ctrl_mode":3, "K_v":0.3, "K_ang":0.05, "K_f":0.05, "T_v":0.05, "T_ang":0.5, "K_p":0.02, "K_i":0.5, "nodes_vknown":[4,5,6,7],
           "a_i":1.5257e+01, "b_i":-2.7435e-02, "c_i":9.5767e-03, "d_i":7.8929e-03, "e_i":4.0265e-03, 
           "a_d":6.2857e-01, "b_d":1.0374e-01, "c_d":-1.2941e-01, "d_d":3.0008e-03, "e_d":-1.3320e-03,
           "Rth_sink":0.0129, "Rth_c_igbt":0.02, "Rth_c_diode":0.05, "Rth_j_igbt":0.15, "Rth_j_diode":0.15,"T_a":25.0, 
           "Cth_sink":6.9767, "N_switch_sink":6},
		{"s_n_kVA":120.0, "V_dc":700.0, "ctrl_mode":3, "K_v":0.3, "K_ang":0.05, "K_f":0.05, "T_v":0.05, "T_ang":0.5, "K_p":0.02, "K_i":0.5, "nodes_vknown":[8,9,10,11],
           "a_i":1.5257e+01, "b_i":-2.7435e-02, "c_i":9.5767e-03, "d_i":7.8929e-03, "e_i":4.0265e-03, 
           "a_d":6.2857e-01, "b_d":1.0374e-01, "c_d":-1.2941e-01, "d_d":3.0008e-03, "e_d":-1.3320e-03,
           "Rth_sink":0.0129, "Rth_c_igbt":0.02, "Rth_c_diode":0.05, "Rth_j_igbt":0.15, "Rth_j_diode":0.15,"T_a":25.0, 
           "Cth_sink":6.9767, "N_switch_sink":6}
	  ],
"secondary":[
	    {"ctrl_mode":1, "K_p_v": 0.0001, "K_i_v": 0.002, "K_p_p": 0.001, "K_i_p": 0.05, "Dt_secondary":1.0, "S_base":200.0e3}
        ],
"line_codes":
		{"OH1":
		{"R": [[ 0.54 ,  0.049,  0.049,  0.049],
			   [ 0.049,  0.54 ,  0.049,  0.049],
			   [ 0.049,  0.049,  0.54 ,  0.049],
			   [ 0.049,  0.049,  0.049,  0.54 ]]}
       },
"perturbations":[
        {"type":"load_new_value", "time":20.5, "bus":"R18", "kw_abc":[10,10,10], "kvar_abc":[6,6,6] }
        ]
}