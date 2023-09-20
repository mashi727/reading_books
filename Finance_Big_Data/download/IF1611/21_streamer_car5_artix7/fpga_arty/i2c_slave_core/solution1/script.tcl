############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2016 Xilinx, Inc. All Rights Reserved.
############################################################
open_project i2c_slave_core
set_top i2c_slave_core
add_files i2c_slave_core.c
open_solution "solution1"
set_part {xc7a35ticsg324-1l} -tool vivado
create_clock -period 10 -name default
#source "./i2c_slave_core/solution1/directives.tcl"
#csim_design
csynth_design
#cosim_design
export_design -format ip_catalog
