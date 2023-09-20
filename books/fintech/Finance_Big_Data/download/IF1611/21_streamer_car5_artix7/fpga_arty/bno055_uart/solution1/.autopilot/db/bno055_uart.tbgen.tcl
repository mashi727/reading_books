set C_TypeInfoList {{ 
"bno055_uart" : [[], { "return": [[], "void"]} , [{"ExternC" : 0}], [],["0","1","2","3","4","5","6","7","8","9"],""],
 "0": [ "uart_tx", [["volatile"],"10"],""],
 "1": [ "uart_rx", [["volatile"],"10"],""],
 "2": [ "mem_wreq", [["volatile"],"10"],""],
 "3": [ "mem_wack", [["volatile"],"10"],""],
 "4": [ "mem_rreq", [["volatile"],"10"],""],
 "5": [ "mem_rack", [["volatile"],"10"],""],
 "6": [ "mem_dout", [["volatile"],"11"],""],
 "7": [ "mem_din", [["volatile"],"11"],""],
 "8": [ "mem_addr", [["volatile"],"11"],""],
 "9": [ "dummy_tmr_out", [["volatile"],"10"],""], 
"11": [ "uint8", {"typedef": [[[], {"scalar": "uint8"}],""]}], 
"10": [ "uint1", {"typedef": [[[], {"scalar": "uint1"}],""]}]
}}
set moduleName bno055_uart
set isCombinational 0
set isDatapathOnly 0
set isPipelined 0
set pipeline_type none
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set C_modelName {bno055_uart}
set C_modelType { void 0 }
set C_modelArgList {
	{ dummy_tmr_out int 1 regular {pointer 2 volatile } {global 2}  }
	{ uart_rx int 1 regular {pointer 0 volatile } {global 0}  }
	{ uart_tx int 1 regular {pointer 2 volatile } {global 2}  }
	{ mem_addr int 8 regular {pointer 2 volatile } {global 2}  }
	{ mem_din int 8 regular {pointer 0 volatile } {global 0}  }
	{ mem_dout int 8 regular {pointer 2 volatile } {global 2}  }
	{ mem_wreq int 1 regular {pointer 2 volatile } {global 2}  }
	{ mem_wack int 1 regular {pointer 0 volatile } {global 0}  }
	{ mem_rreq int 1 regular {pointer 0 volatile } {global 0}  }
	{ mem_rack int 1 regular {pointer 0 volatile } {global 0}  }
}
set C_modelArgMapList {[ 
	{ "Name" : "dummy_tmr_out", "interface" : "wire", "bitwidth" : 1, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "dummy_tmr_out","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "uart_rx", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "uart_rx","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "uart_tx", "interface" : "wire", "bitwidth" : 1, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "uart_tx","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "mem_addr", "interface" : "wire", "bitwidth" : 8, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":7,"cElement": [{"cName": "mem_addr","cData": "uint8","bit_use": { "low": 0,"up": 7},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "mem_din", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY", "bitSlice":[{"low":0,"up":7,"cElement": [{"cName": "mem_din","cData": "uint8","bit_use": { "low": 0,"up": 7},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "mem_dout", "interface" : "wire", "bitwidth" : 8, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":7,"cElement": [{"cName": "mem_dout","cData": "uint8","bit_use": { "low": 0,"up": 7},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "mem_wreq", "interface" : "wire", "bitwidth" : 1, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "mem_wreq","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "mem_wack", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "mem_wack","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "mem_rreq", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "mem_rreq","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "mem_rack", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "mem_rack","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} ]}
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
	{ uart_rx sc_in sc_lv 1 signal 1 } 
	{ uart_tx_i sc_in sc_lv 1 signal 2 } 
	{ uart_tx_o sc_out sc_lv 1 signal 2 } 
	{ mem_addr_i sc_in sc_lv 8 signal 3 } 
	{ mem_addr_o sc_out sc_lv 8 signal 3 } 
	{ mem_din sc_in sc_lv 8 signal 4 } 
	{ mem_dout_i sc_in sc_lv 8 signal 5 } 
	{ mem_dout_o sc_out sc_lv 8 signal 5 } 
	{ mem_wreq_i sc_in sc_lv 1 signal 6 } 
	{ mem_wreq_o sc_out sc_lv 1 signal 6 } 
	{ mem_wack sc_in sc_lv 1 signal 7 } 
	{ mem_rreq sc_in sc_lv 1 signal 8 } 
	{ mem_rack sc_in sc_lv 1 signal 9 } 
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
 	{ "name": "uart_rx", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "uart_rx", "role": "default" }} , 
 	{ "name": "uart_tx_i", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "uart_tx", "role": "i" }} , 
 	{ "name": "uart_tx_o", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "uart_tx", "role": "o" }} , 
 	{ "name": "mem_addr_i", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "mem_addr", "role": "i" }} , 
 	{ "name": "mem_addr_o", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "mem_addr", "role": "o" }} , 
 	{ "name": "mem_din", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "mem_din", "role": "default" }} , 
 	{ "name": "mem_dout_i", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "mem_dout", "role": "i" }} , 
 	{ "name": "mem_dout_o", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "mem_dout", "role": "o" }} , 
 	{ "name": "mem_wreq_i", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem_wreq", "role": "i" }} , 
 	{ "name": "mem_wreq_o", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem_wreq", "role": "o" }} , 
 	{ "name": "mem_wack", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem_wack", "role": "default" }} , 
 	{ "name": "mem_rreq", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem_rreq", "role": "default" }} , 
 	{ "name": "mem_rack", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem_rack", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "8", "15", "22", "23", "24", "25"], "CDFG" : "bno055_uart", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "dummy_tmr_out", "Type" : "None", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_bno055_uart_wait_tmr_fu_244", "Port" : "dummy_tmr_out"}, 
			{"SubInst" : "grp_bno055_uart_uart_write_reg_fu_225", "Port" : "dummy_tmr_out"}, 
			{"SubInst" : "grp_bno055_uart_uart_read_reg_fu_210", "Port" : "dummy_tmr_out"}, 
			{"SubInst" : "grp_bno055_uart_uart_read_reg16_fu_200", "Port" : "dummy_tmr_out"}]}, 
		{"Name" : "uart_rx", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_bno055_uart_uart_write_reg_fu_225", "Port" : "uart_rx"}, 
			{"SubInst" : "grp_bno055_uart_uart_read_reg_fu_210", "Port" : "uart_rx"}, 
			{"SubInst" : "grp_bno055_uart_uart_read_reg16_fu_200", "Port" : "uart_rx"}]}, 
		{"Name" : "uart_tx", "Type" : "None", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_bno055_uart_uart_write_reg_fu_225", "Port" : "uart_tx"}, 
			{"SubInst" : "grp_bno055_uart_uart_read_reg_fu_210", "Port" : "uart_tx"}, 
			{"SubInst" : "grp_bno055_uart_uart_read_reg16_fu_200", "Port" : "uart_tx"}]}, 
		{"Name" : "mem_addr", "Type" : "None", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_bno055_uart_write_mem_fu_254", "Port" : "mem_addr"}]}, 
		{"Name" : "mem_din", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "mem_dout", "Type" : "None", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_bno055_uart_write_mem_fu_254", "Port" : "mem_dout"}]}, 
		{"Name" : "mem_wreq", "Type" : "None", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_bno055_uart_write_mem_fu_254", "Port" : "mem_wreq"}]}, 
		{"Name" : "mem_wack", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_bno055_uart_write_mem_fu_254", "Port" : "mem_wack"}]}, 
		{"Name" : "mem_rreq", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "mem_rack", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}],
		"WaitState" : [
		{"State" : "ap_ST_st33_fsm_32", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_uart_read_reg16_fu_200"},
		{"State" : "ap_ST_st13_fsm_12", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_uart_read_reg_fu_210"},
		{"State" : "ap_ST_st19_fsm_18", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_uart_read_reg_fu_210"},
		{"State" : "ap_ST_st25_fsm_24", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_uart_read_reg_fu_210"},
		{"State" : "ap_ST_st31_fsm_30", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_uart_read_reg_fu_210"},
		{"State" : "ap_ST_st5_fsm_4", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_uart_write_reg_fu_225"},
		{"State" : "ap_ST_st7_fsm_6", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_uart_write_reg_fu_225"},
		{"State" : "ap_ST_st11_fsm_10", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_uart_write_reg_fu_225"},
		{"State" : "ap_ST_st15_fsm_14", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_uart_write_reg_fu_225"},
		{"State" : "ap_ST_st21_fsm_20", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_uart_write_reg_fu_225"},
		{"State" : "ap_ST_st27_fsm_26", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_uart_write_reg_fu_225"},
		{"State" : "ap_ST_st9_fsm_8", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_wait_tmr_fu_244"},
		{"State" : "ap_ST_st17_fsm_16", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_wait_tmr_fu_244"},
		{"State" : "ap_ST_st23_fsm_22", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_wait_tmr_fu_244"},
		{"State" : "ap_ST_st29_fsm_28", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_wait_tmr_fu_244"},
		{"State" : "ap_ST_st35_fsm_34", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_wait_tmr_fu_244"},
		{"State" : "ap_ST_st5_fsm_4", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_write_mem_fu_254"},
		{"State" : "ap_ST_st7_fsm_6", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_write_mem_fu_254"},
		{"State" : "ap_ST_st33_fsm_32", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_write_mem_fu_254"},
		{"State" : "ap_ST_st35_fsm_34", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_write_mem_fu_254"},
		{"State" : "ap_ST_st37_fsm_36", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_write_mem_fu_254"},
		{"State" : "ap_ST_st39_fsm_38", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_write_mem_fu_254"},
		{"State" : "ap_ST_st41_fsm_40", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_write_mem_fu_254"},
		{"State" : "ap_ST_st43_fsm_42", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_write_mem_fu_254"}],
		"SubBlockPort" : []},
	{"Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_bno055_uart_uart_read_reg16_fu_200", "Parent" : "0", "Child" : ["2", "5", "7"], "CDFG" : "bno055_uart_uart_read_reg16", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "uart_tx", "Type" : "Vld", "Direction" : "O", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_bno055_uart_uart_send_byte_fu_51", "Port" : "uart_tx"}]}, 
		{"Name" : "dummy_tmr_out", "Type" : "OVld", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_bno055_uart_wait_tmr_fu_64", "Port" : "dummy_tmr_out"}, 
			{"SubInst" : "grp_bno055_uart_uart_send_byte_fu_51", "Port" : "dummy_tmr_out"}, 
			{"SubInst" : "grp_bno055_uart_uart_receive_byte_fu_43", "Port" : "dummy_tmr_out"}]}, 
		{"Name" : "uart_rx", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_bno055_uart_uart_receive_byte_fu_43", "Port" : "uart_rx"}]}],
		"WaitState" : [
		{"State" : "ap_ST_st12_fsm_11", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_uart_receive_byte_fu_43"},
		{"State" : "ap_ST_st14_fsm_13", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_uart_receive_byte_fu_43"},
		{"State" : "ap_ST_st16_fsm_15", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_uart_receive_byte_fu_43"},
		{"State" : "ap_ST_st18_fsm_17", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_uart_receive_byte_fu_43"},
		{"State" : "ap_ST_st20_fsm_19", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_uart_receive_byte_fu_43"},
		{"State" : "ap_ST_st2_fsm_1", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_uart_send_byte_fu_51"},
		{"State" : "ap_ST_st4_fsm_3", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_uart_send_byte_fu_51"},
		{"State" : "ap_ST_st6_fsm_5", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_uart_send_byte_fu_51"},
		{"State" : "ap_ST_st8_fsm_7", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_uart_send_byte_fu_51"},
		{"State" : "ap_ST_st10_fsm_9", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_wait_tmr_fu_64"}],
		"SubBlockPort" : []},
	{"Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bno055_uart_uart_read_reg16_fu_200.grp_bno055_uart_uart_receive_byte_fu_43", "Parent" : "1", "Child" : ["3", "4"], "CDFG" : "bno055_uart_uart_receive_byte", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "uart_rx", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_bno055_uart_read_uart_rx_fu_95", "Port" : "uart_rx"}]}, 
		{"Name" : "dummy_tmr_out", "Type" : "OVld", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_bno055_uart_wait_tmr_fu_85", "Port" : "dummy_tmr_out"}]}],
		"WaitState" : [
		{"State" : "ap_ST_st5_fsm_4", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_wait_tmr_fu_85"},
		{"State" : "ap_ST_st7_fsm_6", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_wait_tmr_fu_85"},
		{"State" : "ap_ST_st10_fsm_9", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_wait_tmr_fu_85"},
		{"State" : "ap_ST_st12_fsm_11", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_wait_tmr_fu_85"}],
		"SubBlockPort" : []},
	{"Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_bno055_uart_uart_read_reg16_fu_200.grp_bno055_uart_uart_receive_byte_fu_43.grp_bno055_uart_wait_tmr_fu_85", "Parent" : "2", "Child" : [], "CDFG" : "bno055_uart_wait_tmr", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "tmr", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "dummy_tmr_out", "Type" : "OVld", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : []}],
		"WaitState" : [],
		"SubBlockPort" : []},
	{"Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_bno055_uart_uart_read_reg16_fu_200.grp_bno055_uart_uart_receive_byte_fu_43.grp_bno055_uart_read_uart_rx_fu_95", "Parent" : "2", "Child" : [], "CDFG" : "bno055_uart_read_uart_rx", "VariableLatency" : "0", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "1", "ControlExist" : "0",
		"Port" : [
		{"Name" : "uart_rx", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}],
		"WaitState" : [],
		"SubBlockPort" : []},
	{"Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bno055_uart_uart_read_reg16_fu_200.grp_bno055_uart_uart_send_byte_fu_51", "Parent" : "1", "Child" : ["6"], "CDFG" : "bno055_uart_uart_send_byte", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "data", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "uart_tx", "Type" : "Vld", "Direction" : "O", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "dummy_tmr_out", "Type" : "OVld", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_bno055_uart_wait_tmr_fu_71", "Port" : "dummy_tmr_out"}]}],
		"WaitState" : [
		{"State" : "ap_ST_st4_fsm_3", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_wait_tmr_fu_71"},
		{"State" : "ap_ST_st7_fsm_6", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_wait_tmr_fu_71"},
		{"State" : "ap_ST_st9_fsm_8", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_wait_tmr_fu_71"}],
		"SubBlockPort" : []},
	{"Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_bno055_uart_uart_read_reg16_fu_200.grp_bno055_uart_uart_send_byte_fu_51.grp_bno055_uart_wait_tmr_fu_71", "Parent" : "5", "Child" : [], "CDFG" : "bno055_uart_wait_tmr", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "tmr", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "dummy_tmr_out", "Type" : "OVld", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : []}],
		"WaitState" : [],
		"SubBlockPort" : []},
	{"Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bno055_uart_uart_read_reg16_fu_200.grp_bno055_uart_wait_tmr_fu_64", "Parent" : "1", "Child" : [], "CDFG" : "bno055_uart_wait_tmr", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "tmr", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "dummy_tmr_out", "Type" : "OVld", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : []}],
		"WaitState" : [],
		"SubBlockPort" : []},
	{"Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_bno055_uart_uart_read_reg_fu_210", "Parent" : "0", "Child" : ["9", "12", "14"], "CDFG" : "bno055_uart_uart_read_reg", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "reg_addr", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "uart_tx", "Type" : "Vld", "Direction" : "O", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_bno055_uart_uart_send_byte_fu_56", "Port" : "uart_tx"}]}, 
		{"Name" : "dummy_tmr_out", "Type" : "OVld", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_bno055_uart_uart_receive_byte_fu_47", "Port" : "dummy_tmr_out"}, 
			{"SubInst" : "grp_bno055_uart_wait_tmr_fu_68", "Port" : "dummy_tmr_out"}, 
			{"SubInst" : "grp_bno055_uart_uart_send_byte_fu_56", "Port" : "dummy_tmr_out"}]}, 
		{"Name" : "uart_rx", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_bno055_uart_uart_receive_byte_fu_47", "Port" : "uart_rx"}]}],
		"WaitState" : [
		{"State" : "ap_ST_st12_fsm_11", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_uart_receive_byte_fu_47"},
		{"State" : "ap_ST_st14_fsm_13", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_uart_receive_byte_fu_47"},
		{"State" : "ap_ST_st16_fsm_15", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_uart_receive_byte_fu_47"},
		{"State" : "ap_ST_st18_fsm_17", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_uart_receive_byte_fu_47"},
		{"State" : "ap_ST_st2_fsm_1", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_uart_send_byte_fu_56"},
		{"State" : "ap_ST_st4_fsm_3", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_uart_send_byte_fu_56"},
		{"State" : "ap_ST_st6_fsm_5", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_uart_send_byte_fu_56"},
		{"State" : "ap_ST_st8_fsm_7", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_uart_send_byte_fu_56"},
		{"State" : "ap_ST_st10_fsm_9", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_wait_tmr_fu_68"}],
		"SubBlockPort" : []},
	{"Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bno055_uart_uart_read_reg_fu_210.grp_bno055_uart_uart_receive_byte_fu_47", "Parent" : "8", "Child" : ["10", "11"], "CDFG" : "bno055_uart_uart_receive_byte", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "uart_rx", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_bno055_uart_read_uart_rx_fu_95", "Port" : "uart_rx"}]}, 
		{"Name" : "dummy_tmr_out", "Type" : "OVld", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_bno055_uart_wait_tmr_fu_85", "Port" : "dummy_tmr_out"}]}],
		"WaitState" : [
		{"State" : "ap_ST_st5_fsm_4", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_wait_tmr_fu_85"},
		{"State" : "ap_ST_st7_fsm_6", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_wait_tmr_fu_85"},
		{"State" : "ap_ST_st10_fsm_9", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_wait_tmr_fu_85"},
		{"State" : "ap_ST_st12_fsm_11", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_wait_tmr_fu_85"}],
		"SubBlockPort" : []},
	{"Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_bno055_uart_uart_read_reg_fu_210.grp_bno055_uart_uart_receive_byte_fu_47.grp_bno055_uart_wait_tmr_fu_85", "Parent" : "9", "Child" : [], "CDFG" : "bno055_uart_wait_tmr", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "tmr", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "dummy_tmr_out", "Type" : "OVld", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : []}],
		"WaitState" : [],
		"SubBlockPort" : []},
	{"Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_bno055_uart_uart_read_reg_fu_210.grp_bno055_uart_uart_receive_byte_fu_47.grp_bno055_uart_read_uart_rx_fu_95", "Parent" : "9", "Child" : [], "CDFG" : "bno055_uart_read_uart_rx", "VariableLatency" : "0", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "1", "ControlExist" : "0",
		"Port" : [
		{"Name" : "uart_rx", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}],
		"WaitState" : [],
		"SubBlockPort" : []},
	{"Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bno055_uart_uart_read_reg_fu_210.grp_bno055_uart_uart_send_byte_fu_56", "Parent" : "8", "Child" : ["13"], "CDFG" : "bno055_uart_uart_send_byte", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "data", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "uart_tx", "Type" : "Vld", "Direction" : "O", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "dummy_tmr_out", "Type" : "OVld", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_bno055_uart_wait_tmr_fu_71", "Port" : "dummy_tmr_out"}]}],
		"WaitState" : [
		{"State" : "ap_ST_st4_fsm_3", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_wait_tmr_fu_71"},
		{"State" : "ap_ST_st7_fsm_6", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_wait_tmr_fu_71"},
		{"State" : "ap_ST_st9_fsm_8", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_wait_tmr_fu_71"}],
		"SubBlockPort" : []},
	{"Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_bno055_uart_uart_read_reg_fu_210.grp_bno055_uart_uart_send_byte_fu_56.grp_bno055_uart_wait_tmr_fu_71", "Parent" : "12", "Child" : [], "CDFG" : "bno055_uart_wait_tmr", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "tmr", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "dummy_tmr_out", "Type" : "OVld", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : []}],
		"WaitState" : [],
		"SubBlockPort" : []},
	{"Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bno055_uart_uart_read_reg_fu_210.grp_bno055_uart_wait_tmr_fu_68", "Parent" : "8", "Child" : [], "CDFG" : "bno055_uart_wait_tmr", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "tmr", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "dummy_tmr_out", "Type" : "OVld", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : []}],
		"WaitState" : [],
		"SubBlockPort" : []},
	{"Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_bno055_uart_uart_write_reg_fu_225", "Parent" : "0", "Child" : ["16", "19", "21"], "CDFG" : "bno055_uart_uart_write_reg", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "reg_addr", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "data", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "uart_tx", "Type" : "Vld", "Direction" : "O", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_bno055_uart_uart_send_byte_fu_52", "Port" : "uart_tx"}]}, 
		{"Name" : "dummy_tmr_out", "Type" : "OVld", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_bno055_uart_wait_tmr_fu_64", "Port" : "dummy_tmr_out"}, 
			{"SubInst" : "grp_bno055_uart_uart_receive_byte_fu_44", "Port" : "dummy_tmr_out"}, 
			{"SubInst" : "grp_bno055_uart_uart_send_byte_fu_52", "Port" : "dummy_tmr_out"}]}, 
		{"Name" : "uart_rx", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_bno055_uart_uart_receive_byte_fu_44", "Port" : "uart_rx"}]}],
		"WaitState" : [
		{"State" : "ap_ST_st14_fsm_13", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_uart_receive_byte_fu_44"},
		{"State" : "ap_ST_st16_fsm_15", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_uart_receive_byte_fu_44"},
		{"State" : "ap_ST_st2_fsm_1", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_uart_send_byte_fu_52"},
		{"State" : "ap_ST_st4_fsm_3", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_uart_send_byte_fu_52"},
		{"State" : "ap_ST_st6_fsm_5", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_uart_send_byte_fu_52"},
		{"State" : "ap_ST_st8_fsm_7", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_uart_send_byte_fu_52"},
		{"State" : "ap_ST_st10_fsm_9", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_uart_send_byte_fu_52"},
		{"State" : "ap_ST_st12_fsm_11", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_wait_tmr_fu_64"}],
		"SubBlockPort" : []},
	{"Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bno055_uart_uart_write_reg_fu_225.grp_bno055_uart_uart_receive_byte_fu_44", "Parent" : "15", "Child" : ["17", "18"], "CDFG" : "bno055_uart_uart_receive_byte", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "uart_rx", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_bno055_uart_read_uart_rx_fu_95", "Port" : "uart_rx"}]}, 
		{"Name" : "dummy_tmr_out", "Type" : "OVld", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_bno055_uart_wait_tmr_fu_85", "Port" : "dummy_tmr_out"}]}],
		"WaitState" : [
		{"State" : "ap_ST_st5_fsm_4", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_wait_tmr_fu_85"},
		{"State" : "ap_ST_st7_fsm_6", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_wait_tmr_fu_85"},
		{"State" : "ap_ST_st10_fsm_9", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_wait_tmr_fu_85"},
		{"State" : "ap_ST_st12_fsm_11", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_wait_tmr_fu_85"}],
		"SubBlockPort" : []},
	{"Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_bno055_uart_uart_write_reg_fu_225.grp_bno055_uart_uart_receive_byte_fu_44.grp_bno055_uart_wait_tmr_fu_85", "Parent" : "16", "Child" : [], "CDFG" : "bno055_uart_wait_tmr", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "tmr", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "dummy_tmr_out", "Type" : "OVld", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : []}],
		"WaitState" : [],
		"SubBlockPort" : []},
	{"Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_bno055_uart_uart_write_reg_fu_225.grp_bno055_uart_uart_receive_byte_fu_44.grp_bno055_uart_read_uart_rx_fu_95", "Parent" : "16", "Child" : [], "CDFG" : "bno055_uart_read_uart_rx", "VariableLatency" : "0", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "1", "ControlExist" : "0",
		"Port" : [
		{"Name" : "uart_rx", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}],
		"WaitState" : [],
		"SubBlockPort" : []},
	{"Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bno055_uart_uart_write_reg_fu_225.grp_bno055_uart_uart_send_byte_fu_52", "Parent" : "15", "Child" : ["20"], "CDFG" : "bno055_uart_uart_send_byte", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "data", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "uart_tx", "Type" : "Vld", "Direction" : "O", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "dummy_tmr_out", "Type" : "OVld", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : [
			{"SubInst" : "grp_bno055_uart_wait_tmr_fu_71", "Port" : "dummy_tmr_out"}]}],
		"WaitState" : [
		{"State" : "ap_ST_st4_fsm_3", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_wait_tmr_fu_71"},
		{"State" : "ap_ST_st7_fsm_6", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_wait_tmr_fu_71"},
		{"State" : "ap_ST_st9_fsm_8", "FSM" : "ap_CS_fsm", "SubInst" : "grp_bno055_uart_wait_tmr_fu_71"}],
		"SubBlockPort" : []},
	{"Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_bno055_uart_uart_write_reg_fu_225.grp_bno055_uart_uart_send_byte_fu_52.grp_bno055_uart_wait_tmr_fu_71", "Parent" : "19", "Child" : [], "CDFG" : "bno055_uart_wait_tmr", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "tmr", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "dummy_tmr_out", "Type" : "OVld", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : []}],
		"WaitState" : [],
		"SubBlockPort" : []},
	{"Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bno055_uart_uart_write_reg_fu_225.grp_bno055_uart_wait_tmr_fu_64", "Parent" : "15", "Child" : [], "CDFG" : "bno055_uart_wait_tmr", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "tmr", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "dummy_tmr_out", "Type" : "OVld", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : []}],
		"WaitState" : [],
		"SubBlockPort" : []},
	{"Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_bno055_uart_wait_tmr_fu_244", "Parent" : "0", "Child" : [], "CDFG" : "bno055_uart_wait_tmr", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "tmr", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "dummy_tmr_out", "Type" : "OVld", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : []}],
		"WaitState" : [],
		"SubBlockPort" : []},
	{"Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_bno055_uart_write_mem_fu_254", "Parent" : "0", "Child" : [], "CDFG" : "bno055_uart_write_mem", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "addr", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "data", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "mem_addr", "Type" : "Vld", "Direction" : "O", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "mem_dout", "Type" : "Vld", "Direction" : "O", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "mem_wreq", "Type" : "Vld", "Direction" : "O", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "mem_wack", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}],
		"WaitState" : [],
		"SubBlockPort" : []},
	{"Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_bno055_uart_bin2char_fu_277", "Parent" : "0", "Child" : [], "CDFG" : "bno055_uart_bin2char", "VariableLatency" : "0", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "1", "ControlExist" : "0",
		"Port" : [
		{"Name" : "val_r", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}],
		"WaitState" : [],
		"SubBlockPort" : []},
	{"Level" : "1", "Path" : "`AUTOTB_DUT_INST.bno055_uart_mul_mul_8ns_16ns_20_1_U27", "Parent" : "0", "Child" : []}]}

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "-1", "Max" : "-1"}
	, {"Name" : "Interval", "Min" : "0", "Max" : "0"}
]}

set Spec2ImplPortList { 
	dummy_tmr_out { ap_none {  { dummy_tmr_out_i in_data 0 1 }  { dummy_tmr_out_o out_data 1 1 } } }
	uart_rx { ap_none {  { uart_rx in_data 0 1 } } }
	uart_tx { ap_none {  { uart_tx_i in_data 0 1 }  { uart_tx_o out_data 1 1 } } }
	mem_addr { ap_none {  { mem_addr_i in_data 0 8 }  { mem_addr_o out_data 1 8 } } }
	mem_din { ap_none {  { mem_din in_data 0 8 } } }
	mem_dout { ap_none {  { mem_dout_i in_data 0 8 }  { mem_dout_o out_data 1 8 } } }
	mem_wreq { ap_none {  { mem_wreq_i in_data 0 1 }  { mem_wreq_o out_data 1 1 } } }
	mem_wack { ap_none {  { mem_wack in_data 0 1 } } }
	mem_rreq { ap_none {  { mem_rreq in_data 0 1 } } }
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
