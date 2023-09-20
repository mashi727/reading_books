set moduleName lcddrv_init_lcd
set isCombinational 0
set isDatapathOnly 0
set isPipelined 0
set pipeline_type none
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set C_modelName {lcddrv_init_lcd}
set C_modelType { void 0 }
set C_modelArgList {
	{ dummy_tmr_out int 1 regular {pointer 2 volatile } {global 2}  }
	{ en int 1 regular {pointer 1 volatile } {global 1}  }
	{ rs int 1 regular {pointer 1 volatile } {global 1}  }
	{ data int 4 regular {pointer 1 volatile } {global 1}  }
}
set C_modelArgMapList {[ 
	{ "Name" : "dummy_tmr_out", "interface" : "wire", "bitwidth" : 1, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "dummy_tmr_out","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "en", "interface" : "wire", "bitwidth" : 1, "direction" : "WRITEONLY", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "en","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "rs", "interface" : "wire", "bitwidth" : 1, "direction" : "WRITEONLY", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "rs","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "data", "interface" : "wire", "bitwidth" : 4, "direction" : "WRITEONLY", "bitSlice":[{"low":0,"up":3,"cElement": [{"cName": "data","cData": "uint4","bit_use": { "low": 0,"up": 3},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} ]}
# RTL Port declarations: 
set portNum 15
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ dummy_tmr_out_i sc_in sc_lv 1 signal 0 } 
	{ dummy_tmr_out_o sc_out sc_lv 1 signal 0 } 
	{ dummy_tmr_out_o_ap_vld sc_out sc_logic 1 outvld 0 } 
	{ en sc_out sc_lv 1 signal 1 } 
	{ en_ap_vld sc_out sc_logic 1 outvld 1 } 
	{ rs sc_out sc_lv 1 signal 2 } 
	{ rs_ap_vld sc_out sc_logic 1 outvld 2 } 
	{ data sc_out sc_lv 4 signal 3 } 
	{ data_ap_vld sc_out sc_logic 1 outvld 3 } 
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
 	{ "name": "dummy_tmr_out_o_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "dummy_tmr_out", "role": "o_ap_vld" }} , 
 	{ "name": "en", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "en", "role": "default" }} , 
 	{ "name": "en_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "en", "role": "ap_vld" }} , 
 	{ "name": "rs", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "rs", "role": "default" }} , 
 	{ "name": "rs_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "rs", "role": "ap_vld" }} , 
 	{ "name": "data", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "data", "role": "default" }} , 
 	{ "name": "data_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "data", "role": "ap_vld" }}  ]}

set RtlHierarchyInfo {[
	{"Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "3"], "CDFG" : "lcddrv_init_lcd", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
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
	{"Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_lcddrv_lcd_send_cmd_fu_36", "Parent" : "0", "Child" : ["2"], "CDFG" : "lcddrv_lcd_send_cmd", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
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
	{"Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lcddrv_lcd_send_cmd_fu_36.grp_lcddrv_wait_tmr_fu_67", "Parent" : "1", "Child" : [], "CDFG" : "lcddrv_wait_tmr", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "tmr", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "dummy_tmr_out", "Type" : "OVld", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : []}],
		"WaitState" : [],
		"SubBlockPort" : []},
	{"Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_lcddrv_wait_tmr_fu_57", "Parent" : "0", "Child" : [], "CDFG" : "lcddrv_wait_tmr", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "tmr", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "dummy_tmr_out", "Type" : "OVld", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : []}],
		"WaitState" : [],
		"SubBlockPort" : []}]}

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "56247", "Max" : "560000247"}
	, {"Name" : "Interval", "Min" : "56247", "Max" : "560000247"}
]}

set Spec2ImplPortList { 
	dummy_tmr_out { ap_ovld {  { dummy_tmr_out_i in_data 0 1 }  { dummy_tmr_out_o out_data 1 1 }  { dummy_tmr_out_o_ap_vld out_vld 1 1 } } }
	en { ap_vld {  { en out_data 1 1 }  { en_ap_vld out_vld 1 1 } } }
	rs { ap_vld {  { rs out_data 1 1 }  { rs_ap_vld out_vld 1 1 } } }
	data { ap_vld {  { data out_data 1 4 }  { data_ap_vld out_vld 1 1 } } }
}
