############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2016 Xilinx, Inc. All Rights Reserved.
############################################################
open_project sharedmem
set_top sharedmem
add_files sharedmem.c
add_files -tb sharedmem.c
open_solution "solution1"
set_part {xc7a35ticsg324-1l} -tool vivado
create_clock -period 10 -name default
#source "./sharedmem/solution1/directives.tcl"
csim_design
csynth_design
cosim_design
export_design -format ip_catalog
