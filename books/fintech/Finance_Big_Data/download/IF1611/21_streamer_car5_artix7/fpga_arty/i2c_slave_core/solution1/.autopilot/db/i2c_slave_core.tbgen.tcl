set C_TypeInfoList {{ 
"i2c_slave_core" : [[], { "return": [[], "void"]} , [{"ExternC" : 0}], [],["0","1","2","3","4","5","6","7","8","9","10","11","12"],""],
 "0": [ "mem_wreq", [["volatile"],"13"],""],
 "1": [ "mem_wack", [["volatile"],"13"],""],
 "2": [ "mem_rreq", [["volatile"],"13"],""],
 "3": [ "mem_rack", [["volatile"],"13"],""],
 "4": [ "mem_dout", [["volatile"],"14"],""],
 "5": [ "mem_din", [["volatile"],"14"],""],
 "6": [ "mem_addr", [["volatile"],"14"],""],
 "7": [ "i2c_val", [[],"15"],""],
 "8": [ "i2c_sda_out", [["volatile"],"13"],""],
 "9": [ "i2c_sda_oe", [["volatile"],"13"],""],
 "10": [ "i2c_in", [["volatile"],"16"],""],
 "11": [ "dev_addr_in", [["volatile"],"17"],""],
 "12": [ "auto_inc_regad_in", [["volatile"],"13"],""], 
"13": [ "uint1", {"typedef": [[[], {"scalar": "uint1"}],""]}], 
"14": [ "uint8", {"typedef": [[[], {"scalar": "uint8"}],""]}], 
"16": [ "uint2", {"typedef": [[[], {"scalar": "uint2"}],""]}], 
"17": [ "uint7", {"typedef": [[[], {"scalar": "uint7"}],""]}], 
"15": [ "uint2", {"typedef": [[[], {"scalar": "uint2"}],""]}]
}}
set moduleName i2c_slave_core
set isCombinational 0
set isDatapathOnly 0
set isPipelined 0
set pipeline_type none
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set C_modelName {i2c_slave_core}
set C_modelType { void 0 }
set C_modelArgList {
	{ i2c_in int 2 regular {pointer 0 volatile } {global 0}  }
	{ i2c_sda_out int 1 regular {pointer 2 volatile } {global 2}  }
	{ i2c_sda_oe int 1 regular {pointer 2 volatile } {global 2}  }
	{ dev_addr_in int 7 regular {pointer 0 volatile } {global 0}  }
	{ auto_inc_regad_in int 1 regular {pointer 0 volatile } {global 0}  }
	{ mem_addr int 8 regular {pointer 2 volatile } {global 2}  }
	{ mem_din int 8 regular {pointer 0 volatile } {global 0}  }
	{ mem_dout int 8 regular {pointer 2 volatile } {global 2}  }
	{ mem_wreq int 1 regular {pointer 2 volatile } {global 2}  }
	{ mem_wack int 1 regular {pointer 0 volatile } {global 0}  }
	{ mem_rreq int 1 regular {pointer 2 volatile } {global 2}  }
	{ mem_rack int 1 regular {pointer 0 volatile } {global 0}  }
}
set C_modelArgMapList {[ 
	{ "Name" : "i2c_in", "interface" : "wire", "bitwidth" : 2, "direction" : "READONLY", "bitSlice":[{"low":0,"up":1,"cElement": [{"cName": "i2c_in","cData": "uint2","bit_use": { "low": 0,"up": 1},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "i2c_sda_out", "interface" : "wire", "bitwidth" : 1, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "i2c_sda_out","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "i2c_sda_oe", "interface" : "wire", "bitwidth" : 1, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "i2c_sda_oe","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "dev_addr_in", "interface" : "wire", "bitwidth" : 7, "direction" : "READONLY", "bitSlice":[{"low":0,"up":6,"cElement": [{"cName": "dev_addr_in","cData": "uint7","bit_use": { "low": 0,"up": 6},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "auto_inc_regad_in", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "auto_inc_regad_in","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "mem_addr", "interface" : "wire", "bitwidth" : 8, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":7,"cElement": [{"cName": "mem_addr","cData": "uint8","bit_use": { "low": 0,"up": 7},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "mem_din", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY", "bitSlice":[{"low":0,"up":7,"cElement": [{"cName": "mem_din","cData": "uint8","bit_use": { "low": 0,"up": 7},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "mem_dout", "interface" : "wire", "bitwidth" : 8, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":7,"cElement": [{"cName": "mem_dout","cData": "uint8","bit_use": { "low": 0,"up": 7},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "mem_wreq", "interface" : "wire", "bitwidth" : 1, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "mem_wreq","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "mem_wack", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "mem_wack","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "mem_rreq", "interface" : "wire", "bitwidth" : 1, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "mem_rreq","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "mem_rack", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "mem_rack","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} ]}
# RTL Port declarations: 
set portNum 24
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ i2c_in sc_in sc_lv 2 signal 0 } 
	{ i2c_sda_out_i sc_in sc_lv 1 signal 1 } 
	{ i2c_sda_out_o sc_out sc_lv 1 signal 1 } 
	{ i2c_sda_oe_i sc_in sc_lv 1 signal 2 } 
	{ i2c_sda_oe_o sc_out sc_lv 1 signal 2 } 
	{ dev_addr_in sc_in sc_lv 7 signal 3 } 
	{ auto_inc_regad_in sc_in sc_lv 1 signal 4 } 
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
 	{ "name": "i2c_in", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "i2c_in", "role": "default" }} , 
 	{ "name": "i2c_sda_out_i", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "i2c_sda_out", "role": "i" }} , 
 	{ "name": "i2c_sda_out_o", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "i2c_sda_out", "role": "o" }} , 
 	{ "name": "i2c_sda_oe_i", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "i2c_sda_oe", "role": "i" }} , 
 	{ "name": "i2c_sda_oe_o", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "i2c_sda_oe", "role": "o" }} , 
 	{ "name": "dev_addr_in", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "dev_addr_in", "role": "default" }} , 
 	{ "name": "auto_inc_regad_in", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "auto_inc_regad_in", "role": "default" }} , 
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
	{"Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2"], "CDFG" : "i2c_slave_core", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "i2c_in", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "i2c_sda_out", "Type" : "None", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "i2c_sda_oe", "Type" : "None", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "dev_addr_in", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "auto_inc_regad_in", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "mem_addr", "Type" : "None", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_i2c_slave_core_read_mem_fu_460", "Port" : "mem_addr"}, 
			{"SubInst" : "grp_i2c_slave_core_write_mem_fu_444", "Port" : "mem_addr"}]}, 
		{"Name" : "mem_din", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_i2c_slave_core_read_mem_fu_460", "Port" : "mem_din"}]}, 
		{"Name" : "mem_dout", "Type" : "None", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_i2c_slave_core_write_mem_fu_444", "Port" : "mem_dout"}]}, 
		{"Name" : "mem_wreq", "Type" : "None", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_i2c_slave_core_write_mem_fu_444", "Port" : "mem_wreq"}]}, 
		{"Name" : "mem_wack", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_i2c_slave_core_write_mem_fu_444", "Port" : "mem_wack"}]}, 
		{"Name" : "mem_rreq", "Type" : "None", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_i2c_slave_core_read_mem_fu_460", "Port" : "mem_rreq"}]}, 
		{"Name" : "mem_rack", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_i2c_slave_core_read_mem_fu_460", "Port" : "mem_rack"}]}, 
		{"Name" : "i2c_val", "Type" : "Vld", "Direction" : "O", "BlockSignal" : [], "SubConnect" : []}],
		"WaitState" : [
		{"State" : "ap_ST_st55_fsm_54", "FSM" : "ap_CS_fsm", "SubInst" : "grp_i2c_slave_core_write_mem_fu_444"},
		{"State" : "ap_ST_st34_fsm_33", "FSM" : "ap_CS_fsm", "SubInst" : "grp_i2c_slave_core_read_mem_fu_460"},
		{"State" : "ap_ST_st44_fsm_43", "FSM" : "ap_CS_fsm", "SubInst" : "grp_i2c_slave_core_read_mem_fu_460"}],
		"SubBlockPort" : []},
	{"Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_i2c_slave_core_write_mem_fu_444", "Parent" : "0", "Child" : [], "CDFG" : "i2c_slave_core_write_mem", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "addr", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "data", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "mem_addr", "Type" : "Vld", "Direction" : "O", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "mem_dout", "Type" : "Vld", "Direction" : "O", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "mem_wreq", "Type" : "Vld", "Direction" : "O", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "mem_wack", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}],
		"WaitState" : [],
		"SubBlockPort" : []},
	{"Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_i2c_slave_core_read_mem_fu_460", "Parent" : "0", "Child" : [], "CDFG" : "i2c_slave_core_read_mem", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "addr", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "mem_addr", "Type" : "Vld", "Direction" : "O", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "mem_rreq", "Type" : "Vld", "Direction" : "O", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "mem_din", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "mem_rack", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}],
		"WaitState" : [],
		"SubBlockPort" : []}]}

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "-1", "Max" : "-1"}
	, {"Name" : "Interval", "Min" : "0", "Max" : "0"}
]}

set Spec2ImplPortList { 
	i2c_in { ap_none {  { i2c_in in_data 0 2 } } }
	i2c_sda_out { ap_none {  { i2c_sda_out_i in_data 0 1 }  { i2c_sda_out_o out_data 1 1 } } }
	i2c_sda_oe { ap_none {  { i2c_sda_oe_i in_data 0 1 }  { i2c_sda_oe_o out_data 1 1 } } }
	dev_addr_in { ap_none {  { dev_addr_in in_data 0 7 } } }
	auto_inc_regad_in { ap_none {  { auto_inc_regad_in in_data 0 1 } } }
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
