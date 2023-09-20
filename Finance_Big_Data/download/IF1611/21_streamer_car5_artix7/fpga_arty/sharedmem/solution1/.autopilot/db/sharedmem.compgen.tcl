# This script segment is generated automatically by AutoPilot

# Memory (RAM/ROM)  definition:
set ID 0
set MemName sharedmem_mem
set CoreName ap_simcore_mem
set PortList { 2 3 }
set DataWd 8
set AddrRange 256
set AddrWd 8
set impl_style block
set TrueReset 0
set HasInitializer 0
set IsROM 0
set ROMData {}
set NumOfStage 2
set MaxLatency -1
set DelayBudget 2.71
set ClkPeriod 10
set RegisteredInput 0
if {${::AESL::PGuard_simmodel_gen}} {
if {[info proc ap_gen_simcore_mem] == "ap_gen_simcore_mem"} {
    eval "ap_gen_simcore_mem { \
    id ${ID} \
    name ${MemName} \
    corename ${CoreName}  \
    op mem \
    reset_level 1 \
    sync_rst true \
    stage_num ${NumOfStage}  \
    registered_input ${RegisteredInput} \
    port_num 2 \
    port_list \{${PortList}\} \
    data_wd ${DataWd} \
    addr_wd ${AddrWd} \
    addr_range ${AddrRange} \
    style ${impl_style} \
    true_reset ${TrueReset} \
    delay_budget ${DelayBudget} \
    clk_period ${ClkPeriod} \
    HasInitializer ${HasInitializer} \
    rom_data \{${ROMData}\} \
 } "
} else {
    puts "@W \[IMPL-102\] Cannot find ap_gen_simcore_mem, check your platform lib"
}
}


if {${::AESL::PGuard_rtl_comp_handler}} {
  ::AP::rtl_comp_handler $MemName
}


set CoreName RAM
if {${::AESL::PGuard_autocg_gen} && ${::AESL::PGuard_autocg_ipmgen}} {
if {[info proc ::AESL_LIB_VIRTEX::xil_gen_RAM] == "::AESL_LIB_VIRTEX::xil_gen_RAM"} {
    eval "::AESL_LIB_VIRTEX::xil_gen_RAM { \
    id ${ID} \
    name ${MemName} \
    corename ${CoreName}  \
    op mem \
    reset_level 1 \
    sync_rst true \
    stage_num ${NumOfStage}  \
    registered_input ${RegisteredInput} \
    port_num 2 \
    port_list \{${PortList}\} \
    data_wd ${DataWd} \
    addr_wd ${AddrWd} \
    addr_range ${AddrRange} \
    style ${impl_style} \
    true_reset ${TrueReset} \
    delay_budget ${DelayBudget} \
    clk_period ${ClkPeriod} \
    HasInitializer ${HasInitializer} \
    rom_data \{${ROMData}\} \
 } "
  } else {
    puts "@W \[IMPL-104\] Cannot find ::AESL_LIB_VIRTEX::xil_gen_RAM, check your platform lib"
  }
}


# clear list
if {${::AESL::PGuard_autoexp_gen}} {
    cg_default_interface_gen_dc_begin
    cg_default_interface_gen_bundle_begin
    AESL_LIB_XILADAPTER::native_axis_begin
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 1 \
    name addr0 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_addr0 \
    op interface \
    ports { addr0 { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 2 \
    name din0 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_din0 \
    op interface \
    ports { din0 { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 3 \
    name dout0 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_dout0 \
    op interface \
    ports { dout0_i { I 8 vector } dout0_o { O 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 4 \
    name r_req0 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_r_req0 \
    op interface \
    ports { r_req0 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 5 \
    name r_ack0 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_r_ack0 \
    op interface \
    ports { r_ack0_i { I 1 vector } r_ack0_o { O 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 6 \
    name w_req0 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_w_req0 \
    op interface \
    ports { w_req0 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 7 \
    name w_ack0 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_w_ack0 \
    op interface \
    ports { w_ack0_i { I 1 vector } w_ack0_o { O 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 8 \
    name addr1 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_addr1 \
    op interface \
    ports { addr1 { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 9 \
    name din1 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_din1 \
    op interface \
    ports { din1 { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 10 \
    name dout1 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_dout1 \
    op interface \
    ports { dout1_i { I 8 vector } dout1_o { O 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11 \
    name r_req1 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_r_req1 \
    op interface \
    ports { r_req1 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 12 \
    name r_ack1 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_r_ack1 \
    op interface \
    ports { r_ack1_i { I 1 vector } r_ack1_o { O 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 13 \
    name w_req1 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_w_req1 \
    op interface \
    ports { w_req1 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 14 \
    name w_ack1 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_w_ack1 \
    op interface \
    ports { w_ack1_i { I 1 vector } w_ack1_o { O 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 15 \
    name addr2 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_addr2 \
    op interface \
    ports { addr2 { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 16 \
    name din2 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_din2 \
    op interface \
    ports { din2 { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 17 \
    name dout2 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_dout2 \
    op interface \
    ports { dout2_i { I 8 vector } dout2_o { O 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 18 \
    name r_req2 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_r_req2 \
    op interface \
    ports { r_req2 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 19 \
    name r_ack2 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_r_ack2 \
    op interface \
    ports { r_ack2_i { I 1 vector } r_ack2_o { O 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 20 \
    name w_req2 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_w_req2 \
    op interface \
    ports { w_req2 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 21 \
    name w_ack2 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_w_ack2 \
    op interface \
    ports { w_ack2_i { I 1 vector } w_ack2_o { O 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 22 \
    name addr3 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_addr3 \
    op interface \
    ports { addr3 { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 23 \
    name din3 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_din3 \
    op interface \
    ports { din3 { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 24 \
    name dout3 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_dout3 \
    op interface \
    ports { dout3_i { I 8 vector } dout3_o { O 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 25 \
    name r_req3 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_r_req3 \
    op interface \
    ports { r_req3 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 26 \
    name r_ack3 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_r_ack3 \
    op interface \
    ports { r_ack3_i { I 1 vector } r_ack3_o { O 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 27 \
    name w_req3 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_w_req3 \
    op interface \
    ports { w_req3 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 28 \
    name w_ack3 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_w_ack3 \
    op interface \
    ports { w_ack3_i { I 1 vector } w_ack3_o { O 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id -1 \
    name ap_ctrl \
    type ap_ctrl \
    reset_level 1 \
    sync_rst true \
    corename ap_ctrl \
    op interface \
    ports { ap_start { I 1 bit } ap_ready { O 1 bit } ap_done { O 1 bit } ap_idle { O 1 bit } } \
} "
}


# Adapter definition:
set PortName ap_clk
set DataWd 1 
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc cg_default_interface_gen_clock] == "cg_default_interface_gen_clock"} {
eval "cg_default_interface_gen_clock { \
    id -2 \
    name ${PortName} \
    reset_level 1 \
    sync_rst true \
    corename apif_ap_clk \
    data_wd ${DataWd} \
    op interface \
}"
} else {
puts "@W \[IMPL-113\] Cannot find bus interface model in the library. Ignored generation of bus interface for '${PortName}'"
}
}


# Adapter definition:
set PortName ap_rst
set DataWd 1 
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc cg_default_interface_gen_reset] == "cg_default_interface_gen_reset"} {
eval "cg_default_interface_gen_reset { \
    id -3 \
    name ${PortName} \
    reset_level 1 \
    sync_rst true \
    corename apif_ap_rst \
    data_wd ${DataWd} \
    op interface \
}"
} else {
puts "@W \[IMPL-114\] Cannot find bus interface model in the library. Ignored generation of bus interface for '${PortName}'"
}
}



# merge
if {${::AESL::PGuard_autoexp_gen}} {
    cg_default_interface_gen_dc_end
    cg_default_interface_gen_bundle_end
    AESL_LIB_XILADAPTER::native_axis_end
}


