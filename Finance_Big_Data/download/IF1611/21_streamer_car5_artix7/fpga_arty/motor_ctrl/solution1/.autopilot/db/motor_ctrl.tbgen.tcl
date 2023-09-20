set C_TypeInfoList {{ 
"motor_ctrl" : [[], { "return": [[], "void"]} , [{"ExternC" : 0}], [],["0","1","2","3","4","5","6","7","8","9","10","11"],""],
 "0": [ "r_pwm", [["volatile"],"12"],""],
 "1": [ "r_dir", [["volatile"],"12"],""],
 "2": [ "mem_wreq", [["volatile"],"12"],""],
 "3": [ "mem_wack", [["volatile"],"12"],""],
 "4": [ "mem_rreq", [["volatile"],"12"],""],
 "5": [ "mem_rack", [["volatile"],"12"],""],
 "6": [ "mem_dout", [["volatile"],"13"],""],
 "7": [ "mem_din", [["volatile"],"13"],""],
 "8": [ "mem_addr", [["volatile"],"13"],""],
 "9": [ "l_pwm", [["volatile"],"12"],""],
 "10": [ "l_dir", [["volatile"],"12"],""],
 "11": [ "dummy_tmr_out", [["volatile"],"12"],""], 
"12": [ "uint1", {"typedef": [[[], {"scalar": "uint1"}],""]}], 
"13": [ "uint8", {"typedef": [[[], {"scalar": "uint8"}],""]}]
}}
set moduleName motor_ctrl
set isCombinational 0
set isDatapathOnly 0
set isPipelined 0
set pipeline_type none
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set C_modelName {motor_ctrl}
set C_modelType { void 0 }
set C_modelArgList {
	{ dummy_tmr_out int 1 regular {pointer 2 volatile } {global 2}  }
	{ l_dir int 1 regular {pointer 2 volatile } {global 2}  }
	{ l_pwm int 1 regular {pointer 2 volatile } {global 2}  }
	{ r_dir int 1 regular {pointer 2 volatile } {global 2}  }
	{ r_pwm int 1 regular {pointer 2 volatile } {global 2}  }
	{ mem_addr int 8 regular {pointer 2 volatile } {global 2}  }
	{ mem_din int 8 regular {pointer 0 volatile } {global 0}  }
	{ mem_dout int 8 regular {pointer 2 volatile } {global 2}  }
	{ mem_wreq int 1 regular {pointer 2 volatile } {global 2}  }
	{ mem_wack int 1 regular {pointer 0 volatile } {global 0}  }
	{ mem_rreq int 1 regular {pointer 2 volatile } {global 2}  }
	{ mem_rack int 1 regular {pointer 0 volatile } {global 0}  }
}
set C_modelArgMapList {[ 
	{ "Name" : "dummy_tmr_out", "interface" : "wire", "bitwidth" : 1, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "dummy_tmr_out","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "l_dir", "interface" : "wire", "bitwidth" : 1, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "l_dir","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "l_pwm", "interface" : "wire", "bitwidth" : 1, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "l_pwm","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "r_dir", "interface" : "wire", "bitwidth" : 1, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "r_dir","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "r_pwm", "interface" : "wire", "bitwidth" : 1, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "r_pwm","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "mem_addr", "interface" : "wire", "bitwidth" : 8, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":7,"cElement": [{"cName": "mem_addr","cData": "uint8","bit_use": { "low": 0,"up": 7},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "mem_din", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY", "bitSlice":[{"low":0,"up":7,"cElement": [{"cName": "mem_din","cData": "uint8","bit_use": { "low": 0,"up": 7},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "mem_dout", "interface" : "wire", "bitwidth" : 8, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":7,"cElement": [{"cName": "mem_dout","cData": "uint8","bit_use": { "low": 0,"up": 7},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "mem_wreq", "interface" : "wire", "bitwidth" : 1, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "mem_wreq","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "mem_wack", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "mem_wack","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "mem_rreq", "interface" : "wire", "bitwidth" : 1, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "mem_rreq","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "mem_rack", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "mem_rack","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} ]}
# RTL Port declarations: 
set portNum 27
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ dummy_tmr_out_i sc_in sc_lv 1 signal 0 } 
	{ dummy_tmr_out_o sc_out sc_lv 1 signal 0 } 
	{ l_dir_i sc_in sc_lv 1 signal 1 } 
	{ l_dir_o sc_out sc_lv 1 signal 1 } 
	{ l_pwm_i sc_in sc_lv 1 signal 2 } 
	{ l_pwm_o sc_out sc_lv 1 signal 2 } 
	{ r_dir_i sc_in sc_lv 1 signal 3 } 
	{ r_dir_o sc_out sc_lv 1 signal 3 } 
	{ r_pwm_i sc_in sc_lv 1 signal 4 } 
	{ r_pwm_o sc_out sc_lv 1 signal 4 } 
	{ mem_addr_i sc_in sc_lv 8 signal 5 } 
	{ mem_addr_o sc_out sc_lv 8 signal 5 } 
	{ mem_din sc_in sc_lv 8 signal 6 } 
	{ mem_dout_i sc_in sc_lv 8 signal 7 } 
	{ mem_dout_o sc_out sc_lv 8 signal 7 } 
	{ mem_wreq_i sc_in sc_lv 1 signal 8 } 
	{ mem_wreq_o sc_out sc_lv 1 signal 8 } 
	{ mem_wack sc_in sc_lv 1 signal 9 } 
	{ mem_rreq_i sc_in sc_lv 1 signal 10 } 
	{ mem_rreq_o sc_out sc_lv 1 signal 10 } 
	{ mem_rack sc_in sc_lv 1 signal 11 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "dummy_tmr_out_i", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "dummy_tmr_out", "role": "i" }} , 
 	{ "name": "dummy_tmr_out_o", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "dummy_tmr_out", "role": "o" }} , 
 	{ "name": "l_dir_i", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "l_dir", "role": "i" }} , 
 	{ "name": "l_dir_o", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "l_dir", "role": "o" }} , 
 	{ "name": "l_pwm_i", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "l_pwm", "role": "i" }} , 
 	{ "name": "l_pwm_o", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "l_pwm", "role": "o" }} , 
 	{ "name": "r_dir_i", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "r_dir", "role": "i" }} , 
 	{ "name": "r_dir_o", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "r_dir", "role": "o" }} , 
 	{ "name": "r_pwm_i", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "r_pwm", "role": "i" }} , 
 	{ "name": "r_pwm_o", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "r_pwm", "role": "o" }} , 
 	{ "name": "mem_addr_i", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "mem_addr", "role": "i" }} , 
 	{ "name": "mem_addr_o", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "mem_addr", "role": "o" }} , 
 	{ "name": "mem_din", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "mem_din", "role": "default" }} , 
 	{ "name": "mem_dout_i", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "mem_dout", "role": "i" }} , 
 	{ "name": "mem_dout_o", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "mem_dout", "role": "o" }} , 
 	{ "name": "mem_wreq_i", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem_wreq", "role": "i" }} , 
 	{ "name": "mem_wreq_o", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem_wreq", "role": "o" }} , 
 	{ "name": "mem_wack", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem_wack", "role": "default" }} , 
 	{ "name": "mem_rreq_i", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem_rreq", "role": "i" }} , 
 	{ "name": "mem_rreq_o", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem_rreq", "role": "o" }} , 
 	{ "name": "mem_rack", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem_rack", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "4", "5", "6", "7"], "CDFG" : "motor_ctrl", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "dummy_tmr_out", "Type" : "None", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_motor_ctrl_wait_tmr_fu_359", "Port" : "dummy_tmr_out"}]}, 
		{"Name" : "l_dir", "Type" : "None", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "l_pwm", "Type" : "None", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "r_dir", "Type" : "None", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "r_pwm", "Type" : "None", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "mem_addr", "Type" : "None", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_motor_ctrl_write_mem_fu_365", "Port" : "mem_addr"}, 
			{"SubInst" : "grp_motor_ctrl_read_mem_fu_406", "Port" : "mem_addr"}]}, 
		{"Name" : "mem_din", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_motor_ctrl_read_mem_fu_406", "Port" : "mem_din"}]}, 
		{"Name" : "mem_dout", "Type" : "None", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_motor_ctrl_write_mem_fu_365", "Port" : "mem_dout"}]}, 
		{"Name" : "mem_wreq", "Type" : "None", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_motor_ctrl_write_mem_fu_365", "Port" : "mem_wreq"}]}, 
		{"Name" : "mem_wack", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_motor_ctrl_write_mem_fu_365", "Port" : "mem_wack"}]}, 
		{"Name" : "mem_rreq", "Type" : "None", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_motor_ctrl_read_mem_fu_406", "Port" : "mem_rreq"}]}, 
		{"Name" : "mem_rack", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_motor_ctrl_read_mem_fu_406", "Port" : "mem_rack"}]}],
		"WaitState" : [
		{"State" : "ap_ST_st22_fsm_21", "FSM" : "ap_CS_fsm", "SubInst" : "grp_motor_ctrl_diff_angle_fu_353"},
		{"State" : "ap_ST_st58_fsm_57", "FSM" : "ap_CS_fsm", "SubInst" : "grp_motor_ctrl_wait_tmr_fu_359"},
		{"State" : "ap_ST_st4_fsm_3", "FSM" : "ap_CS_fsm", "SubInst" : "grp_motor_ctrl_write_mem_fu_365"},
		{"State" : "ap_ST_st6_fsm_5", "FSM" : "ap_CS_fsm", "SubInst" : "grp_motor_ctrl_write_mem_fu_365"},
		{"State" : "ap_ST_st8_fsm_7", "FSM" : "ap_CS_fsm", "SubInst" : "grp_motor_ctrl_write_mem_fu_365"},
		{"State" : "ap_ST_st10_fsm_9", "FSM" : "ap_CS_fsm", "SubInst" : "grp_motor_ctrl_write_mem_fu_365"},
		{"State" : "ap_ST_st22_fsm_21", "FSM" : "ap_CS_fsm", "SubInst" : "grp_motor_ctrl_write_mem_fu_365"},
		{"State" : "ap_ST_st24_fsm_23", "FSM" : "ap_CS_fsm", "SubInst" : "grp_motor_ctrl_write_mem_fu_365"},
		{"State" : "ap_ST_st26_fsm_25", "FSM" : "ap_CS_fsm", "SubInst" : "grp_motor_ctrl_write_mem_fu_365"},
		{"State" : "ap_ST_st28_fsm_27", "FSM" : "ap_CS_fsm", "SubInst" : "grp_motor_ctrl_write_mem_fu_365"},
		{"State" : "ap_ST_st30_fsm_29", "FSM" : "ap_CS_fsm", "SubInst" : "grp_motor_ctrl_write_mem_fu_365"},
		{"State" : "ap_ST_st32_fsm_31", "FSM" : "ap_CS_fsm", "SubInst" : "grp_motor_ctrl_write_mem_fu_365"},
		{"State" : "ap_ST_st34_fsm_33", "FSM" : "ap_CS_fsm", "SubInst" : "grp_motor_ctrl_write_mem_fu_365"},
		{"State" : "ap_ST_st36_fsm_35", "FSM" : "ap_CS_fsm", "SubInst" : "grp_motor_ctrl_write_mem_fu_365"},
		{"State" : "ap_ST_st38_fsm_37", "FSM" : "ap_CS_fsm", "SubInst" : "grp_motor_ctrl_write_mem_fu_365"},
		{"State" : "ap_ST_st40_fsm_39", "FSM" : "ap_CS_fsm", "SubInst" : "grp_motor_ctrl_write_mem_fu_365"},
		{"State" : "ap_ST_st42_fsm_41", "FSM" : "ap_CS_fsm", "SubInst" : "grp_motor_ctrl_write_mem_fu_365"},
		{"State" : "ap_ST_st44_fsm_43", "FSM" : "ap_CS_fsm", "SubInst" : "grp_motor_ctrl_write_mem_fu_365"},
		{"State" : "ap_ST_st46_fsm_45", "FSM" : "ap_CS_fsm", "SubInst" : "grp_motor_ctrl_write_mem_fu_365"},
		{"State" : "ap_ST_st46_fsm_45", "FSM" : "ap_CS_fsm", "SubInst" : "grp_motor_ctrl_write_mem_fu_365"},
		{"State" : "ap_ST_st49_fsm_48", "FSM" : "ap_CS_fsm", "SubInst" : "grp_motor_ctrl_write_mem_fu_365"},
		{"State" : "ap_ST_st50_fsm_49", "FSM" : "ap_CS_fsm", "SubInst" : "grp_motor_ctrl_write_mem_fu_365"},
		{"State" : "ap_ST_st58_fsm_57", "FSM" : "ap_CS_fsm", "SubInst" : "grp_motor_ctrl_write_mem_fu_365"},
		{"State" : "ap_ST_st60_fsm_59", "FSM" : "ap_CS_fsm", "SubInst" : "grp_motor_ctrl_write_mem_fu_365"},
		{"State" : "ap_ST_st62_fsm_61", "FSM" : "ap_CS_fsm", "SubInst" : "grp_motor_ctrl_write_mem_fu_365"},
		{"State" : "ap_ST_st64_fsm_63", "FSM" : "ap_CS_fsm", "SubInst" : "grp_motor_ctrl_write_mem_fu_365"},
		{"State" : "ap_ST_st12_fsm_11", "FSM" : "ap_CS_fsm", "SubInst" : "grp_motor_ctrl_read_mem_fu_406"},
		{"State" : "ap_ST_st14_fsm_13", "FSM" : "ap_CS_fsm", "SubInst" : "grp_motor_ctrl_read_mem_fu_406"},
		{"State" : "ap_ST_st16_fsm_15", "FSM" : "ap_CS_fsm", "SubInst" : "grp_motor_ctrl_read_mem_fu_406"},
		{"State" : "ap_ST_st18_fsm_17", "FSM" : "ap_CS_fsm", "SubInst" : "grp_motor_ctrl_read_mem_fu_406"},
		{"State" : "ap_ST_st20_fsm_19", "FSM" : "ap_CS_fsm", "SubInst" : "grp_motor_ctrl_read_mem_fu_406"},
		{"State" : "ap_ST_st52_fsm_51", "FSM" : "ap_CS_fsm", "SubInst" : "grp_motor_ctrl_read_mem_fu_406"}],
		"SubBlockPort" : []},
	{"Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_motor_ctrl_diff_angle_fu_353", "Parent" : "0", "Child" : ["2", "3"], "CDFG" : "motor_ctrl_diff_angle", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "target", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "value_r", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}],
		"WaitState" : [],
		"SubBlockPort" : []},
	{"Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_motor_ctrl_diff_angle_fu_353.motor_ctrl_urem_19ns_17ns_19_23_seq_U12", "Parent" : "1", "Child" : []},
	{"Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_motor_ctrl_diff_angle_fu_353.motor_ctrl_urem_21ns_17ns_20_25_seq_U13", "Parent" : "1", "Child" : []},
	{"Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_motor_ctrl_wait_tmr_fu_359", "Parent" : "0", "Child" : [], "CDFG" : "motor_ctrl_wait_tmr", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "dummy_tmr_out", "Type" : "OVld", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : []}],
		"WaitState" : [],
		"SubBlockPort" : []},
	{"Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_motor_ctrl_write_mem_fu_365", "Parent" : "0", "Child" : [], "CDFG" : "motor_ctrl_write_mem", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "addr", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "data", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "mem_addr", "Type" : "Vld", "Direction" : "O", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "mem_dout", "Type" : "Vld", "Direction" : "O", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "mem_wreq", "Type" : "Vld", "Direction" : "O", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "mem_wack", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}],
		"WaitState" : [],
		"SubBlockPort" : []},
	{"Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_motor_ctrl_read_mem_fu_406", "Parent" : "0", "Child" : [], "CDFG" : "motor_ctrl_read_mem", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "addr", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "mem_addr", "Type" : "Vld", "Direction" : "O", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "mem_rreq", "Type" : "Vld", "Direction" : "O", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "mem_din", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "mem_rack", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}],
		"WaitState" : [],
		"SubBlockPort" : []},
	{"Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_motor_ctrl_bin2char_fu_425", "Parent" : "0", "Child" : [], "CDFG" : "motor_ctrl_bin2char", "VariableLatency" : "0", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "1", "ControlExist" : "0",
		"Port" : [
		{"Name" : "val_r", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}],
		"WaitState" : [],
		"SubBlockPort" : []}]}

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "-1", "Max" : "-1"}
	, {"Name" : "Interval", "Min" : "0", "Max" : "0"}
]}

set Spec2ImplPortList { 
	dummy_tmr_out { ap_none {  { dummy_tmr_out_i in_data 0 1 }  { dummy_tmr_out_o out_data 1 1 } } }
	l_dir { ap_none {  { l_dir_i in_data 0 1 }  { l_dir_o out_data 1 1 } } }
	l_pwm { ap_none {  { l_pwm_i in_data 0 1 }  { l_pwm_o out_data 1 1 } } }
	r_dir { ap_none {  { r_dir_i in_data 0 1 }  { r_dir_o out_data 1 1 } } }
	r_pwm { ap_none {  { r_pwm_i in_data 0 1 }  { r_pwm_o out_data 1 1 } } }
	mem_addr { ap_none {  { mem_addr_i in_data 0 8 }  { mem_addr_o out_data 1 8 } } }
	mem_din { ap_none {  { mem_din in_data 0 8 } } }
	mem_dout { ap_none {  { mem_dout_i in_data 0 8 }  { mem_dout_o out_data 1 8 } } }
	mem_wreq { ap_none {  { mem_wreq_i in_data 0 1 }  { mem_wreq_o out_data 1 1 } } }
	mem_wack { ap_none {  { mem_wack in_data 0 1 } } }
	mem_rreq { ap_none {  { mem_rreq_i in_data 0 1 }  { mem_rreq_o out_data 1 1 } } }
	mem_rack { ap_none {  { mem_rack in_data 0 1 } } }
}

set busDeadlockParameterList { 
}

# RTL port scheduling information:
set fifoSchedulingInfoList { 
}

# RTL bus port read request latency information:
set busReadReqLatencyList { 
}

# RTL bus port write response latency information:
set busWriteResLatencyList { 
}

# RTL array port load latency information:
set memoryLoadLatencyList { 
}
