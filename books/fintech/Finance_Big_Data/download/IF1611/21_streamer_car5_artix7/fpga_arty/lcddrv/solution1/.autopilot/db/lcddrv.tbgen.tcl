set C_TypeInfoList {{ 
"lcddrv" : [[], { "return": [[], "void"]} , [{"ExternC" : 0}], [],["0","1","2","3","4","5","6","7","8"],""],
 "0": [ "rs", [["volatile"],"9"],""],
 "1": [ "mem_req", [["volatile"],"9"],""],
 "2": [ "mem_din", [["volatile"],"10"],""],
 "3": [ "mem_addr", [["volatile"],"11"],""],
 "4": [ "mem_ack", [["volatile"],"9"],""],
 "5": [ "ind", [["volatile"],"9"],""],
 "6": [ "en", [["volatile"],"9"],""],
 "7": [ "dummy_tmr_out", [["volatile"],"9"],""],
 "8": [ "data", [["volatile"],"12"],""], 
"9": [ "uint1", {"typedef": [[[], {"scalar": "uint1"}],""]}], 
"12": [ "uint4", {"typedef": [[[], {"scalar": "uint4"}],""]}], 
"11": [ "uint5", {"typedef": [[[], {"scalar": "uint5"}],""]}], 
"10": [ "uint8", {"typedef": [[[], {"scalar": "uint8"}],""]}]
}}
set moduleName lcddrv
set isCombinational 0
set isDatapathOnly 0
set isPipelined 0
set pipeline_type none
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set C_modelName {lcddrv}
set C_modelType { void 0 }
set C_modelArgList {
	{ dummy_tmr_out int 1 regular {pointer 2 volatile } {global 2}  }
	{ rs int 1 regular {pointer 2 volatile } {global 2}  }
	{ en int 1 regular {pointer 2 volatile } {global 2}  }
	{ data int 4 regular {pointer 2 volatile } {global 2}  }
	{ ind int 1 regular {pointer 0 volatile } {global 0}  }
	{ mem_addr int 5 regular {pointer 2 volatile } {global 2}  }
	{ mem_din int 8 regular {pointer 0 volatile } {global 0}  }
	{ mem_req int 1 regular {pointer 2 volatile } {global 2}  }
	{ mem_ack int 1 regular {pointer 0 volatile } {global 0}  }
}
set C_modelArgMapList {[ 
	{ "Name" : "dummy_tmr_out", "interface" : "wire", "bitwidth" : 1, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "dummy_tmr_out","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "rs", "interface" : "wire", "bitwidth" : 1, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "rs","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "en", "interface" : "wire", "bitwidth" : 1, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "en","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "data", "interface" : "wire", "bitwidth" : 4, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":3,"cElement": [{"cName": "data","cData": "uint4","bit_use": { "low": 0,"up": 3},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "ind", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "ind","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "mem_addr", "interface" : "wire", "bitwidth" : 5, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":4,"cElement": [{"cName": "mem_addr","cData": "uint5","bit_use": { "low": 0,"up": 4},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "mem_din", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY", "bitSlice":[{"low":0,"up":7,"cElement": [{"cName": "mem_din","cData": "uint8","bit_use": { "low": 0,"up": 7},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "mem_req", "interface" : "wire", "bitwidth" : 1, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "mem_req","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "mem_ack", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "mem_ack","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} ]}
# RTL Port declarations: 
set portNum 21
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ dummy_tmr_out_i sc_in sc_lv 1 signal 0 } 
	{ dummy_tmr_out_o sc_out sc_lv 1 signal 0 } 
	{ rs_i sc_in sc_lv 1 signal 1 } 
	{ rs_o sc_out sc_lv 1 signal 1 } 
	{ en_i sc_in sc_lv 1 signal 2 } 
	{ en_o sc_out sc_lv 1 signal 2 } 
	{ data_i sc_in sc_lv 4 signal 3 } 
	{ data_o sc_out sc_lv 4 signal 3 } 
	{ ind sc_in sc_lv 1 signal 4 } 
	{ mem_addr_i sc_in sc_lv 5 signal 5 } 
	{ mem_addr_o sc_out sc_lv 5 signal 5 } 
	{ mem_din sc_in sc_lv 8 signal 6 } 
	{ mem_req_i sc_in sc_lv 1 signal 7 } 
	{ mem_req_o sc_out sc_lv 1 signal 7 } 
	{ mem_ack sc_in sc_lv 1 signal 8 } 
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
 	{ "name": "rs_i", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "rs", "role": "i" }} , 
 	{ "name": "rs_o", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "rs", "role": "o" }} , 
 	{ "name": "en_i", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "en", "role": "i" }} , 
 	{ "name": "en_o", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "en", "role": "o" }} , 
 	{ "name": "data_i", "direction": "in", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "data", "role": "i" }} , 
 	{ "name": "data_o", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "data", "role": "o" }} , 
 	{ "name": "ind", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "ind", "role": "default" }} , 
 	{ "name": "mem_addr_i", "direction": "in", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "mem_addr", "role": "i" }} , 
 	{ "name": "mem_addr_o", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "mem_addr", "role": "o" }} , 
 	{ "name": "mem_din", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "mem_din", "role": "default" }} , 
 	{ "name": "mem_req_i", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem_req", "role": "i" }} , 
 	{ "name": "mem_req_o", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem_req", "role": "o" }} , 
 	{ "name": "mem_ack", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem_ack", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "5", "7", "8"], "CDFG" : "lcddrv", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "dummy_tmr_out", "Type" : "None", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_lcddrv_lcd_send_cmd_fu_179", "Port" : "dummy_tmr_out"}, 
			{"SubInst" : "grp_lcddrv_init_lcd_fu_167", "Port" : "dummy_tmr_out"}, 
			{"SubInst" : "grp_lcddrv_wait_tmr_fu_198", "Port" : "dummy_tmr_out"}]}, 
		{"Name" : "rs", "Type" : "None", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_lcddrv_lcd_send_cmd_fu_179", "Port" : "rs"}, 
			{"SubInst" : "grp_lcddrv_init_lcd_fu_167", "Port" : "rs"}]}, 
		{"Name" : "en", "Type" : "None", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_lcddrv_lcd_send_cmd_fu_179", "Port" : "en"}, 
			{"SubInst" : "grp_lcddrv_init_lcd_fu_167", "Port" : "en"}]}, 
		{"Name" : "data", "Type" : "None", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_lcddrv_lcd_send_cmd_fu_179", "Port" : "data"}, 
			{"SubInst" : "grp_lcddrv_init_lcd_fu_167", "Port" : "data"}]}, 
		{"Name" : "ind", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "mem_addr", "Type" : "None", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_lcddrv_read_mem_fu_206", "Port" : "mem_addr"}]}, 
		{"Name" : "mem_din", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_lcddrv_read_mem_fu_206", "Port" : "mem_din"}]}, 
		{"Name" : "mem_req", "Type" : "None", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_lcddrv_read_mem_fu_206", "Port" : "mem_req"}]}, 
		{"Name" : "mem_ack", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_lcddrv_read_mem_fu_206", "Port" : "mem_ack"}]}],
		"WaitState" : [
		{"State" : "ap_ST_st4_fsm_3", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_init_lcd_fu_167"},
		{"State" : "ap_ST_st6_fsm_5", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_lcd_send_cmd_fu_179"},
		{"State" : "ap_ST_st8_fsm_7", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_lcd_send_cmd_fu_179"},
		{"State" : "ap_ST_st14_fsm_13", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_lcd_send_cmd_fu_179"},
		{"State" : "ap_ST_st11_fsm_10", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_lcd_send_cmd_fu_179"},
		{"State" : "ap_ST_st13_fsm_12", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_lcd_send_cmd_fu_179"},
		{"State" : "ap_ST_st16_fsm_15", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_lcd_send_cmd_fu_179"},
		{"State" : "ap_ST_st19_fsm_18", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_lcd_send_cmd_fu_179"},
		{"State" : "ap_ST_st21_fsm_20", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_lcd_send_cmd_fu_179"},
		{"State" : "ap_ST_st22_fsm_21", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_wait_tmr_fu_198"},
		{"State" : "ap_ST_st10_fsm_9", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_read_mem_fu_206"},
		{"State" : "ap_ST_st18_fsm_17", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_read_mem_fu_206"}],
		"SubBlockPort" : []},
	{"Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_lcddrv_init_lcd_fu_167", "Parent" : "0", "Child" : ["2", "4"], "CDFG" : "lcddrv_init_lcd", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "dummy_tmr_out", "Type" : "OVld", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_lcddrv_lcd_send_cmd_fu_36", "Port" : "dummy_tmr_out"}, 
			{"SubInst" : "grp_lcddrv_wait_tmr_fu_57", "Port" : "dummy_tmr_out"}]}, 
		{"Name" : "en", "Type" : "Vld", "Direction" : "O", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_lcddrv_lcd_send_cmd_fu_36", "Port" : "en"}]}, 
		{"Name" : "rs", "Type" : "Vld", "Direction" : "O", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_lcddrv_lcd_send_cmd_fu_36", "Port" : "rs"}]}, 
		{"Name" : "data", "Type" : "Vld", "Direction" : "O", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_lcddrv_lcd_send_cmd_fu_36", "Port" : "data"}]}],
		"WaitState" : [
		{"State" : "ap_ST_st4_fsm_3", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_lcd_send_cmd_fu_36"},
		{"State" : "ap_ST_st8_fsm_7", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_lcd_send_cmd_fu_36"},
		{"State" : "ap_ST_st12_fsm_11", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_lcd_send_cmd_fu_36"},
		{"State" : "ap_ST_st16_fsm_15", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_lcd_send_cmd_fu_36"},
		{"State" : "ap_ST_st18_fsm_17", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_lcd_send_cmd_fu_36"},
		{"State" : "ap_ST_st20_fsm_19", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_lcd_send_cmd_fu_36"},
		{"State" : "ap_ST_st24_fsm_23", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_lcd_send_cmd_fu_36"},
		{"State" : "ap_ST_st26_fsm_25", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_lcd_send_cmd_fu_36"},
		{"State" : "ap_ST_st30_fsm_29", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_lcd_send_cmd_fu_36"},
		{"State" : "ap_ST_st32_fsm_31", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_lcd_send_cmd_fu_36"},
		{"State" : "ap_ST_st36_fsm_35", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_lcd_send_cmd_fu_36"},
		{"State" : "ap_ST_st38_fsm_37", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_lcd_send_cmd_fu_36"},
		{"State" : "ap_ST_st2_fsm_1", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_wait_tmr_fu_57"},
		{"State" : "ap_ST_st6_fsm_5", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_wait_tmr_fu_57"},
		{"State" : "ap_ST_st10_fsm_9", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_wait_tmr_fu_57"},
		{"State" : "ap_ST_st14_fsm_13", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_wait_tmr_fu_57"},
		{"State" : "ap_ST_st22_fsm_21", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_wait_tmr_fu_57"},
		{"State" : "ap_ST_st28_fsm_27", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_wait_tmr_fu_57"},
		{"State" : "ap_ST_st34_fsm_33", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_wait_tmr_fu_57"},
		{"State" : "ap_ST_st40_fsm_39", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_wait_tmr_fu_57"}],
		"SubBlockPort" : []},
	{"Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lcddrv_init_lcd_fu_167.grp_lcddrv_lcd_send_cmd_fu_36", "Parent" : "1", "Child" : ["3"], "CDFG" : "lcddrv_lcd_send_cmd", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "mode", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "wd", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "en", "Type" : "Vld", "Direction" : "O", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "dummy_tmr_out", "Type" : "OVld", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_lcddrv_wait_tmr_fu_67", "Port" : "dummy_tmr_out"}]}, 
		{"Name" : "rs", "Type" : "Vld", "Direction" : "O", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "data", "Type" : "Vld", "Direction" : "O", "BlockSignal" : [], "SubConnect" : []}],
		"WaitState" : [
		{"State" : "ap_ST_st3_fsm_2", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_wait_tmr_fu_67"},
		{"State" : "ap_ST_st5_fsm_4", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_wait_tmr_fu_67"},
		{"State" : "ap_ST_st7_fsm_6", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_wait_tmr_fu_67"},
		{"State" : "ap_ST_st9_fsm_8", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_wait_tmr_fu_67"}],
		"SubBlockPort" : []},
	{"Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lcddrv_init_lcd_fu_167.grp_lcddrv_lcd_send_cmd_fu_36.grp_lcddrv_wait_tmr_fu_67", "Parent" : "2", "Child" : [], "CDFG" : "lcddrv_wait_tmr", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "tmr", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "dummy_tmr_out", "Type" : "OVld", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : []}],
		"WaitState" : [],
		"SubBlockPort" : []},
	{"Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lcddrv_init_lcd_fu_167.grp_lcddrv_wait_tmr_fu_57", "Parent" : "1", "Child" : [], "CDFG" : "lcddrv_wait_tmr", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "tmr", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "dummy_tmr_out", "Type" : "OVld", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : []}],
		"WaitState" : [],
		"SubBlockPort" : []},
	{"Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_lcddrv_lcd_send_cmd_fu_179", "Parent" : "0", "Child" : ["6"], "CDFG" : "lcddrv_lcd_send_cmd", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "mode", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "wd", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "en", "Type" : "Vld", "Direction" : "O", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "dummy_tmr_out", "Type" : "OVld", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_lcddrv_wait_tmr_fu_67", "Port" : "dummy_tmr_out"}]}, 
		{"Name" : "rs", "Type" : "Vld", "Direction" : "O", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "data", "Type" : "Vld", "Direction" : "O", "BlockSignal" : [], "SubConnect" : []}],
		"WaitState" : [
		{"State" : "ap_ST_st3_fsm_2", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_wait_tmr_fu_67"},
		{"State" : "ap_ST_st5_fsm_4", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_wait_tmr_fu_67"},
		{"State" : "ap_ST_st7_fsm_6", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_wait_tmr_fu_67"},
		{"State" : "ap_ST_st9_fsm_8", "FSM" : "ap_CS_fsm", "SubInst" : "grp_lcddrv_wait_tmr_fu_67"}],
		"SubBlockPort" : []},
	{"Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lcddrv_lcd_send_cmd_fu_179.grp_lcddrv_wait_tmr_fu_67", "Parent" : "5", "Child" : [], "CDFG" : "lcddrv_wait_tmr", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "tmr", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "dummy_tmr_out", "Type" : "OVld", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : []}],
		"WaitState" : [],
		"SubBlockPort" : []},
	{"Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_lcddrv_wait_tmr_fu_198", "Parent" : "0", "Child" : [], "CDFG" : "lcddrv_wait_tmr", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "tmr", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "dummy_tmr_out", "Type" : "OVld", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : []}],
		"WaitState" : [],
		"SubBlockPort" : []},
	{"Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_lcddrv_read_mem_fu_206", "Parent" : "0", "Child" : [], "CDFG" : "lcddrv_read_mem", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "addr", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "mem_addr", "Type" : "Vld", "Direction" : "O", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "mem_req", "Type" : "Vld", "Direction" : "O", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "mem_din", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "mem_ack", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}],
		"WaitState" : [],
		"SubBlockPort" : []}]}

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "-1", "Max" : "-1"}
	, {"Name" : "Interval", "Min" : "0", "Max" : "0"}
]}

set Spec2ImplPortList { 
	dummy_tmr_out { ap_none {  { dummy_tmr_out_i in_data 0 1 }  { dummy_tmr_out_o out_data 1 1 } } }
	rs { ap_none {  { rs_i in_data 0 1 }  { rs_o out_data 1 1 } } }
	en { ap_none {  { en_i in_data 0 1 }  { en_o out_data 1 1 } } }
	data { ap_none {  { data_i in_data 0 4 }  { data_o out_data 1 4 } } }
	ind { ap_none {  { ind in_data 0 1 } } }
	mem_addr { ap_none {  { mem_addr_i in_data 0 5 }  { mem_addr_o out_data 1 5 } } }
	mem_din { ap_none {  { mem_din in_data 0 8 } } }
	mem_req { ap_none {  { mem_req_i in_data 0 1 }  { mem_req_o out_data 1 1 } } }
	mem_ack { ap_none {  { mem_ack in_data 0 1 } } }
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
