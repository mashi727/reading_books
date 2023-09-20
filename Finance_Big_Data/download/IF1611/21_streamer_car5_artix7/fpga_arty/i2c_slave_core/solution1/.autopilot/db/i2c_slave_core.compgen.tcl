# This script segment is generated automatically by AutoPilot

# clear list
if {${::AESL::PGuard_autoexp_gen}} {
    cg_default_interface_gen_dc_begin
    cg_default_interface_gen_bundle_begin
    AESL_LIB_XILADAPTER::native_axis_begin
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11 \
    name i2c_in \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_i2c_in \
    op interface \
    ports { i2c_in { I 2 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 12 \
    name i2c_sda_out \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_i2c_sda_out \
    op interface \
    ports { i2c_sda_out_i { I 1 vector } i2c_sda_out_o { O 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 13 \
    name i2c_sda_oe \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_i2c_sda_oe \
    op interface \
    ports { i2c_sda_oe_i { I 1 vector } i2c_sda_oe_o { O 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 14 \
    name dev_addr_in \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_dev_addr_in \
    op interface \
    ports { dev_addr_in { I 7 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 15 \
    name auto_inc_regad_in \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_auto_inc_regad_in \
    op interface \
    ports { auto_inc_regad_in { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 16 \
    name mem_addr \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_mem_addr \
    op interface \
    ports { mem_addr_i { I 8 vector } mem_addr_o { O 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 17 \
    name mem_din \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_mem_din \
    op interface \
    ports { mem_din { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 18 \
    name mem_dout \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_mem_dout \
    op interface \
    ports { mem_dout_i { I 8 vector } mem_dout_o { O 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 19 \
    name mem_wreq \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_mem_wreq \
    op interface \
    ports { mem_wreq_i { I 1 vector } mem_wreq_o { O 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 20 \
    name mem_wack \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_mem_wack \
    op interface \
    ports { mem_wack { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 21 \
    name mem_rreq \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_mem_rreq \
    op interface \
    ports { mem_rreq_i { I 1 vector } mem_rreq_o { O 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 22 \
    name mem_rack \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_mem_rack \
    op interface \
    ports { mem_rack { I 1 vector } } \
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


