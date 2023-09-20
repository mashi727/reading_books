set C_TypeInfoList {{ 
"sharedmem" : [[], { "return": [[], "void"]} , [{"ExternC" : 0}], [],["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27"],""],
 "0": [ "w_req3", [["volatile"],"28"],""],
 "1": [ "w_req2", [["volatile"],"28"],""],
 "2": [ "w_req1", [["volatile"],"28"],""],
 "3": [ "w_req0", [["volatile"],"28"],""],
 "4": [ "w_ack3", [["volatile"],"28"],""],
 "5": [ "w_ack2", [["volatile"],"28"],""],
 "6": [ "w_ack1", [["volatile"],"28"],""],
 "7": [ "w_ack0", [["volatile"],"28"],""],
 "8": [ "r_req3", [["volatile"],"28"],""],
 "9": [ "r_req2", [["volatile"],"28"],""],
 "10": [ "r_req1", [["volatile"],"28"],""],
 "11": [ "r_req0", [["volatile"],"28"],""],
 "12": [ "r_ack3", [["volatile"],"28"],""],
 "13": [ "r_ack2", [["volatile"],"28"],""],
 "14": [ "r_ack1", [["volatile"],"28"],""],
 "15": [ "r_ack0", [["volatile"],"28"],""],
 "16": [ "dout3", [["volatile"],"29"],""],
 "17": [ "dout2", [["volatile"],"29"],""],
 "18": [ "dout1", [["volatile"],"29"],""],
 "19": [ "dout0", [["volatile"],"29"],""],
 "20": [ "din3", [["volatile"],"29"],""],
 "21": [ "din2", [["volatile"],"29"],""],
 "22": [ "din1", [["volatile"],"29"],""],
 "23": [ "din0", [["volatile"],"29"],""],
 "24": [ "addr3", [["volatile"],"29"],""],
 "25": [ "addr2", [["volatile"],"29"],""],
 "26": [ "addr1", [["volatile"],"29"],""],
 "27": [ "addr0", [["volatile"],"29"],""], 
"29": [ "uint8", {"typedef": [[[], {"scalar": "uint8"}],""]}], 
"28": [ "uint1", {"typedef": [[[], {"scalar": "uint1"}],""]}]
}}
set moduleName sharedmem
set isCombinational 0
set isDatapathOnly 0
set isPipelined 0
set pipeline_type none
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set C_modelName {sharedmem}
set C_modelType { void 0 }
set C_modelArgList {
	{ addr0 int 8 regular {pointer 0 volatile } {global 0}  }
	{ din0 int 8 regular {pointer 0 volatile } {global 0}  }
	{ dout0 int 8 regular {pointer 2 volatile } {global 2}  }
	{ r_req0 int 1 regular {pointer 0 volatile } {global 0}  }
	{ r_ack0 int 1 regular {pointer 2 volatile } {global 2}  }
	{ w_req0 int 1 regular {pointer 0 volatile } {global 0}  }
	{ w_ack0 int 1 regular {pointer 2 volatile } {global 2}  }
	{ addr1 int 8 regular {pointer 0 volatile } {global 0}  }
	{ din1 int 8 regular {pointer 0 volatile } {global 0}  }
	{ dout1 int 8 regular {pointer 2 volatile } {global 2}  }
	{ r_req1 int 1 regular {pointer 0 volatile } {global 0}  }
	{ r_ack1 int 1 regular {pointer 2 volatile } {global 2}  }
	{ w_req1 int 1 regular {pointer 0 volatile } {global 0}  }
	{ w_ack1 int 1 regular {pointer 2 volatile } {global 2}  }
	{ addr2 int 8 regular {pointer 0 volatile } {global 0}  }
	{ din2 int 8 regular {pointer 0 volatile } {global 0}  }
	{ dout2 int 8 regular {pointer 2 volatile } {global 2}  }
	{ r_req2 int 1 regular {pointer 0 volatile } {global 0}  }
	{ r_ack2 int 1 regular {pointer 2 volatile } {global 2}  }
	{ w_req2 int 1 regular {pointer 0 volatile } {global 0}  }
	{ w_ack2 int 1 regular {pointer 2 volatile } {global 2}  }
	{ addr3 int 8 regular {pointer 0 volatile } {global 0}  }
	{ din3 int 8 regular {pointer 0 volatile } {global 0}  }
	{ dout3 int 8 regular {pointer 2 volatile } {global 2}  }
	{ r_req3 int 1 regular {pointer 0 volatile } {global 0}  }
	{ r_ack3 int 1 regular {pointer 2 volatile } {global 2}  }
	{ w_req3 int 1 regular {pointer 0 volatile } {global 0}  }
	{ w_ack3 int 1 regular {pointer 2 volatile } {global 2}  }
}
set C_modelArgMapList {[ 
	{ "Name" : "addr0", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY", "bitSlice":[{"low":0,"up":7,"cElement": [{"cName": "addr0","cData": "uint8","bit_use": { "low": 0,"up": 7},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "din0", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY", "bitSlice":[{"low":0,"up":7,"cElement": [{"cName": "din0","cData": "uint8","bit_use": { "low": 0,"up": 7},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "dout0", "interface" : "wire", "bitwidth" : 8, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":7,"cElement": [{"cName": "dout0","cData": "uint8","bit_use": { "low": 0,"up": 7},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "r_req0", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "r_req0","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "r_ack0", "interface" : "wire", "bitwidth" : 1, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "r_ack0","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "w_req0", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "w_req0","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "w_ack0", "interface" : "wire", "bitwidth" : 1, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "w_ack0","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "addr1", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY", "bitSlice":[{"low":0,"up":7,"cElement": [{"cName": "addr1","cData": "uint8","bit_use": { "low": 0,"up": 7},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "din1", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY", "bitSlice":[{"low":0,"up":7,"cElement": [{"cName": "din1","cData": "uint8","bit_use": { "low": 0,"up": 7},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "dout1", "interface" : "wire", "bitwidth" : 8, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":7,"cElement": [{"cName": "dout1","cData": "uint8","bit_use": { "low": 0,"up": 7},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "r_req1", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "r_req1","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "r_ack1", "interface" : "wire", "bitwidth" : 1, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "r_ack1","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "w_req1", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "w_req1","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "w_ack1", "interface" : "wire", "bitwidth" : 1, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "w_ack1","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "addr2", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY", "bitSlice":[{"low":0,"up":7,"cElement": [{"cName": "addr2","cData": "uint8","bit_use": { "low": 0,"up": 7},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "din2", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY", "bitSlice":[{"low":0,"up":7,"cElement": [{"cName": "din2","cData": "uint8","bit_use": { "low": 0,"up": 7},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "dout2", "interface" : "wire", "bitwidth" : 8, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":7,"cElement": [{"cName": "dout2","cData": "uint8","bit_use": { "low": 0,"up": 7},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "r_req2", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "r_req2","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "r_ack2", "interface" : "wire", "bitwidth" : 1, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "r_ack2","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "w_req2", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "w_req2","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "w_ack2", "interface" : "wire", "bitwidth" : 1, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "w_ack2","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "addr3", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY", "bitSlice":[{"low":0,"up":7,"cElement": [{"cName": "addr3","cData": "uint8","bit_use": { "low": 0,"up": 7},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "din3", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY", "bitSlice":[{"low":0,"up":7,"cElement": [{"cName": "din3","cData": "uint8","bit_use": { "low": 0,"up": 7},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "dout3", "interface" : "wire", "bitwidth" : 8, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":7,"cElement": [{"cName": "dout3","cData": "uint8","bit_use": { "low": 0,"up": 7},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "r_req3", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "r_req3","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "r_ack3", "interface" : "wire", "bitwidth" : 1, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "r_ack3","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "w_req3", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "w_req3","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "w_ack3", "interface" : "wire", "bitwidth" : 1, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "w_ack3","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} ]}
# RTL Port declarations: 
set portNum 46
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ addr0 sc_in sc_lv 8 signal 0 } 
	{ din0 sc_in sc_lv 8 signal 1 } 
	{ dout0_i sc_in sc_lv 8 signal 2 } 
	{ dout0_o sc_out sc_lv 8 signal 2 } 
	{ r_req0 sc_in sc_lv 1 signal 3 } 
	{ r_ack0_i sc_in sc_lv 1 signal 4 } 
	{ r_ack0_o sc_out sc_lv 1 signal 4 } 
	{ w_req0 sc_in sc_lv 1 signal 5 } 
	{ w_ack0_i sc_in sc_lv 1 signal 6 } 
	{ w_ack0_o sc_out sc_lv 1 signal 6 } 
	{ addr1 sc_in sc_lv 8 signal 7 } 
	{ din1 sc_in sc_lv 8 signal 8 } 
	{ dout1_i sc_in sc_lv 8 signal 9 } 
	{ dout1_o sc_out sc_lv 8 signal 9 } 
	{ r_req1 sc_in sc_lv 1 signal 10 } 
	{ r_ack1_i sc_in sc_lv 1 signal 11 } 
	{ r_ack1_o sc_out sc_lv 1 signal 11 } 
	{ w_req1 sc_in sc_lv 1 signal 12 } 
	{ w_ack1_i sc_in sc_lv 1 signal 13 } 
	{ w_ack1_o sc_out sc_lv 1 signal 13 } 
	{ addr2 sc_in sc_lv 8 signal 14 } 
	{ din2 sc_in sc_lv 8 signal 15 } 
	{ dout2_i sc_in sc_lv 8 signal 16 } 
	{ dout2_o sc_out sc_lv 8 signal 16 } 
	{ r_req2 sc_in sc_lv 1 signal 17 } 
	{ r_ack2_i sc_in sc_lv 1 signal 18 } 
	{ r_ack2_o sc_out sc_lv 1 signal 18 } 
	{ w_req2 sc_in sc_lv 1 signal 19 } 
	{ w_ack2_i sc_in sc_lv 1 signal 20 } 
	{ w_ack2_o sc_out sc_lv 1 signal 20 } 
	{ addr3 sc_in sc_lv 8 signal 21 } 
	{ din3 sc_in sc_lv 8 signal 22 } 
	{ dout3_i sc_in sc_lv 8 signal 23 } 
	{ dout3_o sc_out sc_lv 8 signal 23 } 
	{ r_req3 sc_in sc_lv 1 signal 24 } 
	{ r_ack3_i sc_in sc_lv 1 signal 25 } 
	{ r_ack3_o sc_out sc_lv 1 signal 25 } 
	{ w_req3 sc_in sc_lv 1 signal 26 } 
	{ w_ack3_i sc_in sc_lv 1 signal 27 } 
	{ w_ack3_o sc_out sc_lv 1 signal 27 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "addr0", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "addr0", "role": "default" }} , 
 	{ "name": "din0", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "din0", "role": "default" }} , 
 	{ "name": "dout0_i", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "dout0", "role": "i" }} , 
 	{ "name": "dout0_o", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "dout0", "role": "o" }} , 
 	{ "name": "r_req0", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "r_req0", "role": "default" }} , 
 	{ "name": "r_ack0_i", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "r_ack0", "role": "i" }} , 
 	{ "name": "r_ack0_o", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "r_ack0", "role": "o" }} , 
 	{ "name": "w_req0", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "w_req0", "role": "default" }} , 
 	{ "name": "w_ack0_i", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "w_ack0", "role": "i" }} , 
 	{ "name": "w_ack0_o", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "w_ack0", "role": "o" }} , 
 	{ "name": "addr1", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "addr1", "role": "default" }} , 
 	{ "name": "din1", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "din1", "role": "default" }} , 
 	{ "name": "dout1_i", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "dout1", "role": "i" }} , 
 	{ "name": "dout1_o", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "dout1", "role": "o" }} , 
 	{ "name": "r_req1", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "r_req1", "role": "default" }} , 
 	{ "name": "r_ack1_i", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "r_ack1", "role": "i" }} , 
 	{ "name": "r_ack1_o", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "r_ack1", "role": "o" }} , 
 	{ "name": "w_req1", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "w_req1", "role": "default" }} , 
 	{ "name": "w_ack1_i", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "w_ack1", "role": "i" }} , 
 	{ "name": "w_ack1_o", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "w_ack1", "role": "o" }} , 
 	{ "name": "addr2", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "addr2", "role": "default" }} , 
 	{ "name": "din2", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "din2", "role": "default" }} , 
 	{ "name": "dout2_i", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "dout2", "role": "i" }} , 
 	{ "name": "dout2_o", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "dout2", "role": "o" }} , 
 	{ "name": "r_req2", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "r_req2", "role": "default" }} , 
 	{ "name": "r_ack2_i", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "r_ack2", "role": "i" }} , 
 	{ "name": "r_ack2_o", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "r_ack2", "role": "o" }} , 
 	{ "name": "w_req2", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "w_req2", "role": "default" }} , 
 	{ "name": "w_ack2_i", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "w_ack2", "role": "i" }} , 
 	{ "name": "w_ack2_o", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "w_ack2", "role": "o" }} , 
 	{ "name": "addr3", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "addr3", "role": "default" }} , 
 	{ "name": "din3", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "din3", "role": "default" }} , 
 	{ "name": "dout3_i", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "dout3", "role": "i" }} , 
 	{ "name": "dout3_o", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "dout3", "role": "o" }} , 
 	{ "name": "r_req3", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "r_req3", "role": "default" }} , 
 	{ "name": "r_ack3_i", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "r_ack3", "role": "i" }} , 
 	{ "name": "r_ack3_o", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "r_ack3", "role": "o" }} , 
 	{ "name": "w_req3", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "w_req3", "role": "default" }} , 
 	{ "name": "w_ack3_i", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "w_ack3", "role": "i" }} , 
 	{ "name": "w_ack3_o", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "w_ack3", "role": "o" }}  ]}

set RtlHierarchyInfo {[
	{"Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1"], "CDFG" : "sharedmem", "VariableLatency" : "1", "AlignedPipeline" : "0", "UnalignedPipeline" : "0", "ProcessNetwork" : "0", "Combinational" : "0", "ControlExist" : "1",
		"Port" : [
		{"Name" : "addr0", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "din0", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "dout0", "Type" : "None", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "r_req0", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "r_ack0", "Type" : "None", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "w_req0", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "w_ack0", "Type" : "None", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "addr1", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "din1", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "dout1", "Type" : "None", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "r_req1", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "r_ack1", "Type" : "None", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "w_req1", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "w_ack1", "Type" : "None", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "addr2", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "din2", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "dout2", "Type" : "None", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "r_req2", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "r_ack2", "Type" : "None", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "w_req2", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "w_ack2", "Type" : "None", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "addr3", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "din3", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "dout3", "Type" : "None", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "r_req3", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "r_ack3", "Type" : "None", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "w_req3", "Type" : "None", "Direction" : "I", "BlockSignal" : [], "SubConnect" : []}, 
		{"Name" : "w_ack3", "Type" : "None", "Direction" : "IO", "BlockSignal" : [], "SubConnect" : []}],
		"WaitState" : [],
		"SubBlockPort" : []},
	{"Level" : "1", "Path" : "`AUTOTB_DUT_INST.mem_U", "Parent" : "0", "Child" : []}]}

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "-1", "Max" : "-1"}
	, {"Name" : "Interval", "Min" : "0", "Max" : "0"}
]}

set Spec2ImplPortList { 
	addr0 { ap_none {  { addr0 in_data 0 8 } } }
	din0 { ap_none {  { din0 in_data 0 8 } } }
	dout0 { ap_none {  { dout0_i in_data 0 8 }  { dout0_o out_data 1 8 } } }
	r_req0 { ap_none {  { r_req0 in_data 0 1 } } }
	r_ack0 { ap_none {  { r_ack0_i in_data 0 1 }  { r_ack0_o out_data 1 1 } } }
	w_req0 { ap_none {  { w_req0 in_data 0 1 } } }
	w_ack0 { ap_none {  { w_ack0_i in_data 0 1 }  { w_ack0_o out_data 1 1 } } }
	addr1 { ap_none {  { addr1 in_data 0 8 } } }
	din1 { ap_none {  { din1 in_data 0 8 } } }
	dout1 { ap_none {  { dout1_i in_data 0 8 }  { dout1_o out_data 1 8 } } }
	r_req1 { ap_none {  { r_req1 in_data 0 1 } } }
	r_ack1 { ap_none {  { r_ack1_i in_data 0 1 }  { r_ack1_o out_data 1 1 } } }
	w_req1 { ap_none {  { w_req1 in_data 0 1 } } }
	w_ack1 { ap_none {  { w_ack1_i in_data 0 1 }  { w_ack1_o out_data 1 1 } } }
	addr2 { ap_none {  { addr2 in_data 0 8 } } }
	din2 { ap_none {  { din2 in_data 0 8 } } }
	dout2 { ap_none {  { dout2_i in_data 0 8 }  { dout2_o out_data 1 8 } } }
	r_req2 { ap_none {  { r_req2 in_data 0 1 } } }
	r_ack2 { ap_none {  { r_ack2_i in_data 0 1 }  { r_ack2_o out_data 1 1 } } }
	w_req2 { ap_none {  { w_req2 in_data 0 1 } } }
	w_ack2 { ap_none {  { w_ack2_i in_data 0 1 }  { w_ack2_o out_data 1 1 } } }
	addr3 { ap_none {  { addr3 in_data 0 8 } } }
	din3 { ap_none {  { din3 in_data 0 8 } } }
	dout3 { ap_none {  { dout3_i in_data 0 8 }  { dout3_o out_data 1 8 } } }
	r_req3 { ap_none {  { r_req3 in_data 0 1 } } }
	r_ack3 { ap_none {  { r_ack3_i in_data 0 1 }  { r_ack3_o out_data 1 1 } } }
	w_req3 { ap_none {  { w_req3 in_data 0 1 } } }
	w_ack3 { ap_none {  { w_ack3_i in_data 0 1 }  { w_ack3_o out_data 1 1 } } }
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
