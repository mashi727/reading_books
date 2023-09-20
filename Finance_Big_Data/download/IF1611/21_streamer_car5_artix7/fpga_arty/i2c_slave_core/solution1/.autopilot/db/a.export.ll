; ModuleID = 'D:/21_streamer_car5_artix7/fpga_arty/i2c_slave_core/solution1/.autopilot/db/a.o.2.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-w64-mingw32"

@mem_wreq = global i1 false, align 1
@mem_wack = common global i1 false, align 1
@mem_rreq = global i1 false, align 1
@mem_rack = common global i1 false, align 1
@mem_dout = global i8 0, align 1
@mem_din = common global i8 0, align 1
@mem_addr = global i8 0, align 1
@i2c_val = common global i2 0, align 1
@i2c_sda_out = global i1 true, align 1
@i2c_sda_oe = global i1 false, align 1
@i2c_in = common global i2 0, align 1
@dev_addr_in = common global i7 0, align 1
@auto_inc_regad_in = common global i1 false, align 1
@p_str6 = private unnamed_addr constant [12 x i8] c"hls_label_4\00", align 1
@p_str5 = private unnamed_addr constant [17 x i8] c"label_read_start\00", align 1
@p_str1 = private unnamed_addr constant [8 x i8] c"ap_none\00", align 1
@p_str = private unnamed_addr constant [1 x i8] zeroinitializer, align 1

define internal fastcc void @i2c_slave_core_write_mem(i8 zeroext %addr, i8 zeroext %data) nounwind uwtable {
  %data_read = call i8 @_ssdm_op_Read.ap_auto.i8(i8 %data) nounwind
  %addr_read = call i8 @_ssdm_op_Read.ap_auto.i8(i8 %addr) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @mem_addr, i8 %addr_read) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @mem_dout, i8 %data_read) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @mem_wreq, i1 false) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  br label %._crit_edge

._crit_edge:                                      ; preds = %._crit_edge, %0
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @mem_addr, i8 %addr_read) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @mem_dout, i8 %data_read) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @mem_wreq, i1 true) nounwind
  %mem_wack_read = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @mem_wack) nounwind
  br i1 %mem_wack_read, label %1, label %._crit_edge

; <label>:1                                       ; preds = %._crit_edge
  call void (...)* @_ssdm_op_Wait(i32 1)
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @mem_addr, i8 %addr_read) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @mem_dout, i8 %data_read) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @mem_wreq, i1 false) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  ret void
}

define internal fastcc zeroext i8 @i2c_slave_core_read_mem(i8 zeroext %addr) nounwind uwtable {
  %addr_read = call i8 @_ssdm_op_Read.ap_auto.i8(i8 %addr) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @mem_addr, i8 %addr_read) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @mem_rreq, i1 true) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  br label %._crit_edge

._crit_edge:                                      ; preds = %._crit_edge, %0
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @mem_addr, i8 %addr_read) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @mem_rreq, i1 true) nounwind
  %dt = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @mem_din) nounwind
  %mem_rack_read = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @mem_rack) nounwind
  br i1 %mem_rack_read, label %1, label %._crit_edge

; <label>:1                                       ; preds = %._crit_edge
  call void (...)* @_ssdm_op_Wait(i32 1)
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @mem_addr, i8 %addr_read) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @mem_rreq, i1 false) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  ret i8 %dt
}

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

define void @i2c_slave_core() noreturn nounwind uwtable {
  %dev_addr_2 = alloca i7
  %reg_addr_4 = alloca i8
  %p_Val2_33 = alloca i8
  call void (...)* @_ssdm_op_SpecTopModule()
  %i2c_in_load = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i2* @i2c_in, [8 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind
  %i2c_sda_out_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @i2c_sda_out) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @i2c_sda_out, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind
  %i2c_sda_oe_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @i2c_sda_oe) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @i2c_sda_oe, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind
  %dev_addr_in_load = call i7 @_ssdm_op_Read.ap_none.volatile.i7P(i7* @dev_addr_in) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i7* @dev_addr_in, [8 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind
  %auto_inc_regad_in_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @auto_inc_regad_in) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @auto_inc_regad_in, [8 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind
  %mem_addr_load = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @mem_addr) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i8* @mem_addr, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind
  %mem_din_load = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @mem_din) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i8* @mem_din, [8 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind
  %mem_dout_load = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @mem_dout) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i8* @mem_dout, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind
  %mem_wreq_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @mem_wreq) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @mem_wreq, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind
  %mem_wack_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @mem_wack) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @mem_wack, [8 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind
  %mem_rreq_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @mem_rreq) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @mem_rreq, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind
  %mem_rack_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @mem_rack) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @mem_rack, [8 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @i2c_sda_out, i1 true) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @i2c_sda_oe, i1 false) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @mem_addr, i8 0) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @mem_dout, i8 0) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @mem_wreq, i1 false) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @mem_rreq, i1 false) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  br label %.backedge

.backedge.loopexit:                               ; preds = %._crit_edge101
  store i2 %p_Val2_65, i2* @i2c_val, align 1
  store i8 %reg_data_8, i8* %p_Val2_33
  store i8 %re_7, i8* %reg_addr_4
  store i7 %de_2, i7* %dev_addr_2
  br label %.backedge.backedge

.backedge.loopexit83:                             ; preds = %.preheader
  store i2 %p_Val2_63, i2* @i2c_val, align 1
  store i8 %reg_data_4, i8* %p_Val2_33
  store i8 %re_7, i8* %reg_addr_4
  store i7 %de_2, i7* %dev_addr_2
  br label %.backedge.backedge

.backedge.loopexit84:                             ; preds = %._crit_edge92
  store i2 %p_Val2_53, i2* @i2c_val, align 1
  store i8 %reg_data_3, i8* %p_Val2_33
  store i8 %re_2, i8* %reg_addr_4
  br label %.backedge.backedge

.backedge.loopexit88:                             ; preds = %15
  store i2 %p_Val2_48, i2* @i2c_val, align 1
  store i8 %reg_data, i8* %p_Val2_33
  br label %.backedge.backedge

.backedge.loopexit102:                            ; preds = %.preheader50
  store i2 %p_Val2_35, i2* @i2c_val, align 1
  br label %.backedge.backedge

.backedge.loopexit104:                            ; preds = %.preheader52
  store i2 %p_Val2_34, i2* @i2c_val, align 1
  br label %.backedge.backedge

.backedge.backedge:                               ; preds = %.backedge.loopexit104, %.backedge.loopexit102, %.backedge.loopexit88, %.backedge.loopexit84, %.backedge.loopexit83, %.backedge.loopexit
  br label %.backedge

.backedge:                                        ; preds = %.backedge.backedge, %0
  %loop_begin = call i32 (...)* @_ssdm_op_SpecLoopBegin() nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @i2c_sda_out, i1 true) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @i2c_sda_oe, i1 false) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  br label %.critedge

.critedge:                                        ; preds = %.critedge.backedge, %.backedge
  %p_Val2_s = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind
  %tmp = trunc i2 %p_Val2_s to i1
  br i1 %tmp, label %1, label %.critedge.backedge

; <label>:1                                       ; preds = %.critedge
  %tmp_1 = call i1 @_ssdm_op_BitSelect.i1.i2.i32(i2 %p_Val2_s, i32 1)
  br i1 %tmp_1, label %.preheader52.preheader, label %.critedge.backedge

.preheader52.preheader:                           ; preds = %1
  store i2 %p_Val2_s, i2* @i2c_val, align 1
  br label %.preheader52

.critedge.backedge:                               ; preds = %1, %.critedge
  br label %.critedge

.preheader52:                                     ; preds = %2, %.preheader52.preheader
  %p_Val2_34 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind
  %tmp_2 = trunc i2 %p_Val2_34 to i1
  br i1 %tmp_2, label %2, label %.backedge.loopexit104

; <label>:2                                       ; preds = %.preheader52
  %tmp_4 = call i1 @_ssdm_op_BitSelect.i1.i2.i32(i2 %p_Val2_34, i32 1)
  br i1 %tmp_4, label %.preheader52, label %.preheader50.preheader

.preheader50.preheader:                           ; preds = %2
  store i2 %p_Val2_34, i2* @i2c_val, align 1
  br label %.preheader50

.preheader50:                                     ; preds = %3, %.preheader50.preheader
  %p_Val2_35 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind
  %tmp_5 = call i1 @_ssdm_op_BitSelect.i1.i2.i32(i2 %p_Val2_35, i32 1)
  br i1 %tmp_5, label %.backedge.loopexit102, label %3

; <label>:3                                       ; preds = %.preheader50
  %tmp_8 = trunc i2 %p_Val2_35 to i1
  br i1 %tmp_8, label %.preheader50, label %.preheader49

.preheader49.loopexit:                            ; preds = %._crit_edge
  store i7 %dev_addr, i7* %dev_addr_2
  br label %.preheader49

.preheader49:                                     ; preds = %3, %.preheader49.loopexit
  %storemerge = phi i2 [ %p_Val2_38, %.preheader49.loopexit ], [ %p_Val2_35, %3 ]
  %bit_cnt = phi i3 [ %bit_cnt_6, %.preheader49.loopexit ], [ 0, %3 ]
  %dev_addr_2_load = load i7* %dev_addr_2
  store i2 %storemerge, i2* @i2c_val, align 1
  %exitcond1 = icmp eq i3 %bit_cnt, -1
  %empty = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 7, i64 7, i64 7) nounwind
  %bit_cnt_6 = add i3 %bit_cnt, 1
  br i1 %exitcond1, label %5, label %.preheader48

.preheader48:                                     ; preds = %.preheader49, %.preheader48
  %p_Val2_36 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind
  %tmp_9 = trunc i2 %p_Val2_36 to i1
  br i1 %tmp_9, label %4, label %.preheader48

; <label>:4                                       ; preds = %.preheader48
  store i2 %p_Val2_36, i2* @i2c_val, align 1
  %tmp_13 = call i1 @_ssdm_op_BitSelect.i1.i2.i32(i2 %p_Val2_36, i32 1)
  %tmp_14 = trunc i7 %dev_addr_2_load to i6
  %dev_addr = call i7 @_ssdm_op_BitConcatenate.i7.i6.i1(i6 %tmp_14, i1 %tmp_13)
  br label %._crit_edge

._crit_edge:                                      ; preds = %._crit_edge, %4
  %p_Val2_38 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind
  %tmp_16 = trunc i2 %p_Val2_38 to i1
  br i1 %tmp_16, label %._crit_edge, label %.preheader49.loopexit

; <label>:5                                       ; preds = %.preheader49
  %dev_addr_in_read = call i7 @_ssdm_op_Read.ap_none.volatile.i7P(i7* @dev_addr_in) nounwind
  %not_s = icmp ne i7 %dev_addr_2_load, %dev_addr_in_read
  br label %._crit_edge84

._crit_edge84:                                    ; preds = %._crit_edge84, %5
  %p_Val2_37 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind
  %tmp_12 = trunc i2 %p_Val2_37 to i1
  br i1 %tmp_12, label %6, label %._crit_edge84

; <label>:6                                       ; preds = %._crit_edge84
  store i2 %p_Val2_37, i2* @i2c_val, align 1
  %tmp_15 = call i1 @_ssdm_op_BitSelect.i1.i2.i32(i2 %p_Val2_37, i32 1)
  %ignore_0_s = or i1 %not_s, %tmp_15
  br label %._crit_edge85

._crit_edge85:                                    ; preds = %._crit_edge85, %6
  %p_Val2_39 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind
  %tmp_17 = trunc i2 %p_Val2_39 to i1
  br i1 %tmp_17, label %._crit_edge85, label %7

; <label>:7                                       ; preds = %._crit_edge85
  store i2 %p_Val2_39, i2* @i2c_val, align 1
  call void (...)* @_ssdm_op_Wait(i32 1)
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @i2c_sda_out, i1 %ignore_0_s) nounwind
  %not_ignore_1 = xor i1 %ignore_0_s, true
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @i2c_sda_oe, i1 %not_ignore_1) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  br label %._crit_edge86

._crit_edge86:                                    ; preds = %._crit_edge86, %7
  %p_Val2_40 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind
  %tmp_18 = trunc i2 %p_Val2_40 to i1
  br i1 %tmp_18, label %.preheader47.preheader, label %._crit_edge86

.preheader47.preheader:                           ; preds = %._crit_edge86
  store i2 %p_Val2_40, i2* @i2c_val, align 1
  br label %.preheader47

.preheader47:                                     ; preds = %.preheader47, %.preheader47.preheader
  %p_Val2_41 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind
  %tmp_19 = trunc i2 %p_Val2_41 to i1
  br i1 %tmp_19, label %.preheader47, label %8

; <label>:8                                       ; preds = %.preheader47
  store i2 %p_Val2_41, i2* @i2c_val, align 1
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @i2c_sda_oe, i1 false) nounwind
  br label %9

; <label>:9                                       ; preds = %11, %8
  %bit_cnt_1 = phi i4 [ 0, %8 ], [ %bit_cnt_7, %11 ]
  %reg_addr_4_load = load i8* %reg_addr_4
  %exitcond2 = icmp eq i4 %bit_cnt_1, -8
  %empty_3 = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 8, i64 8, i64 8) nounwind
  %bit_cnt_7 = add i4 %bit_cnt_1, 1
  br i1 %exitcond2, label %12, label %.preheader46

.preheader46:                                     ; preds = %9, %.preheader46
  %p_Val2_42 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind
  %tmp_20 = trunc i2 %p_Val2_42 to i1
  br i1 %tmp_20, label %10, label %.preheader46

; <label>:10                                      ; preds = %.preheader46
  store i2 %p_Val2_42, i2* @i2c_val, align 1
  %tmp_22 = call i1 @_ssdm_op_BitSelect.i1.i2.i32(i2 %p_Val2_42, i32 1)
  %tmp_23 = trunc i8 %reg_addr_4_load to i7
  %reg_addr = call i8 @_ssdm_op_BitConcatenate.i8.i7.i1(i7 %tmp_23, i1 %tmp_22)
  br label %._crit_edge87

._crit_edge87:                                    ; preds = %._crit_edge87, %10
  %p_Val2_44 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind
  %tmp_24 = trunc i2 %p_Val2_44 to i1
  br i1 %tmp_24, label %._crit_edge87, label %11

; <label>:11                                      ; preds = %._crit_edge87
  store i2 %p_Val2_44, i2* @i2c_val, align 1
  store i8 %reg_addr, i8* %reg_addr_4
  br label %9

; <label>:12                                      ; preds = %9
  call void (...)* @_ssdm_op_Wait(i32 1)
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @i2c_sda_out, i1 %ignore_0_s) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @i2c_sda_oe, i1 %not_ignore_1) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  br label %._crit_edge88

._crit_edge88:                                    ; preds = %._crit_edge88, %12
  %p_Val2_43 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind
  %tmp_21 = trunc i2 %p_Val2_43 to i1
  br i1 %tmp_21, label %.preheader45.preheader, label %._crit_edge88

.preheader45.preheader:                           ; preds = %._crit_edge88
  store i2 %p_Val2_43, i2* @i2c_val, align 1
  br label %.preheader45

.preheader45:                                     ; preds = %.preheader45, %.preheader45.preheader
  %p_Val2_45 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind
  %tmp_25 = trunc i2 %p_Val2_45 to i1
  br i1 %tmp_25, label %.preheader45, label %13

; <label>:13                                      ; preds = %.preheader45
  store i2 %p_Val2_45, i2* @i2c_val, align 1
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @i2c_sda_oe, i1 false) nounwind
  br label %._crit_edge89

._crit_edge89:                                    ; preds = %._crit_edge89, %13
  %p_Val2_46 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind
  %tmp_26 = trunc i2 %p_Val2_46 to i1
  br i1 %tmp_26, label %14, label %._crit_edge89

; <label>:14                                      ; preds = %._crit_edge89
  %p_Val2_33_load = load i8* %p_Val2_33
  store i2 %p_Val2_46, i2* @i2c_val, align 1
  %tmp_27 = call i1 @_ssdm_op_BitSelect.i1.i2.i32(i2 %p_Val2_46, i32 1)
  %tmp_28 = trunc i8 %p_Val2_33_load to i7
  %reg_data = call i8 @_ssdm_op_BitConcatenate.i8.i7.i1(i7 %tmp_28, i1 %tmp_27)
  br label %._crit_edge91

._crit_edge91:                                    ; preds = %16, %14
  %p_Val2_47 = phi i2 [ %p_Val2_48, %16 ], [ %p_Val2_46, %14 ]
  %pre_i2c_sda_val = call i1 @_ssdm_op_BitSelect.i1.i2.i32(i2 %p_Val2_47, i32 1)
  %p_Val2_48 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind
  %tmp_30 = trunc i2 %p_Val2_48 to i1
  br i1 %tmp_30, label %15, label %.preheader40.preheader

.preheader40.preheader:                           ; preds = %._crit_edge91
  store i2 %p_Val2_48, i2* @i2c_val, align 1
  br label %.preheader40

; <label>:15                                      ; preds = %._crit_edge91
  br i1 %ignore_0_s, label %.backedge.loopexit88, label %16

; <label>:16                                      ; preds = %15
  %tmp_31 = call i1 @_ssdm_op_BitSelect.i1.i2.i32(i2 %p_Val2_48, i32 1)
  %p_not1 = xor i1 %pre_i2c_sda_val, true
  %brmerge = or i1 %tmp_31, %p_not1
  br i1 %brmerge, label %._crit_edge91, label %.preheader41.preheader

.preheader41.preheader:                           ; preds = %16
  store i2 %p_Val2_48, i2* @i2c_val, align 1
  br label %.preheader41

.preheader41:                                     ; preds = %.preheader41, %.preheader41.preheader
  %p_Val2_49 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind
  %tmp_33 = trunc i2 %p_Val2_49 to i1
  br i1 %tmp_33, label %.preheader41, label %.preheader34

.preheader40:                                     ; preds = %22, %23, %.preheader40.preheader
  %bit_cnt_2 = phi i1 [ true, %.preheader40.preheader ], [ false, %23 ], [ false, %22 ]
  %reg_data_1 = phi i8 [ %reg_data, %.preheader40.preheader ], [ %reg_data_2, %23 ], [ %reg_data_2, %22 ]
  %re_2 = phi i8 [ %reg_addr_4_load, %.preheader40.preheader ], [ %p_re_2, %23 ], [ %re_2, %22 ]
  %bit_cnt_2_cast = zext i1 %bit_cnt_2 to i4
  br label %17

; <label>:17                                      ; preds = %20, %.preheader40
  %bit_cnt_3 = phi i4 [ %bit_cnt_2_cast, %.preheader40 ], [ %bit_cnt_8, %20 ]
  %reg_data_2 = phi i8 [ %reg_data_1, %.preheader40 ], [ %reg_data_3, %20 ]
  %tmp_32 = call i1 @_ssdm_op_BitSelect.i1.i4.i32(i4 %bit_cnt_3, i32 3)
  br i1 %tmp_32, label %21, label %.preheader37

.preheader37:                                     ; preds = %17, %.preheader37
  %p_Val2_50 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind
  %tmp_34 = trunc i2 %p_Val2_50 to i1
  br i1 %tmp_34, label %18, label %.preheader37

; <label>:18                                      ; preds = %.preheader37
  store i2 %p_Val2_50, i2* @i2c_val, align 1
  %tmp_36 = call i1 @_ssdm_op_BitSelect.i1.i2.i32(i2 %p_Val2_50, i32 1)
  %tmp_37 = trunc i8 %reg_data_2 to i7
  %reg_data_3 = call i8 @_ssdm_op_BitConcatenate.i8.i7.i1(i7 %tmp_37, i1 %tmp_36)
  br label %._crit_edge92

._crit_edge92:                                    ; preds = %19, %18
  %p_Val2_52 = phi i2 [ %p_Val2_53, %19 ], [ %p_Val2_50, %18 ]
  %tmp_38 = call i1 @_ssdm_op_BitSelect.i1.i2.i32(i2 %p_Val2_52, i32 1)
  %p_Val2_53 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind
  %tmp_39 = call i1 @_ssdm_op_BitSelect.i1.i2.i32(i2 %p_Val2_53, i32 1)
  %tmp_6 = xor i1 %tmp_39, true
  %brmerge1 = or i1 %tmp_38, %tmp_6
  br i1 %brmerge1, label %19, label %.backedge.loopexit84

; <label>:19                                      ; preds = %._crit_edge92
  %tmp_43 = trunc i2 %p_Val2_53 to i1
  br i1 %tmp_43, label %._crit_edge92, label %20

; <label>:20                                      ; preds = %19
  store i2 %p_Val2_53, i2* @i2c_val, align 1
  %bit_cnt_8 = add i4 %bit_cnt_3, 1
  br label %17

; <label>:21                                      ; preds = %17
  call void (...)* @_ssdm_op_Wait(i32 1)
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @i2c_sda_out, i1 %ignore_0_s) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @i2c_sda_oe, i1 %not_ignore_1) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  br label %._crit_edge93

._crit_edge93:                                    ; preds = %._crit_edge93, %21
  %p_Val2_51 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind
  %tmp_35 = trunc i2 %p_Val2_51 to i1
  br i1 %tmp_35, label %.preheader35.preheader, label %._crit_edge93

.preheader35.preheader:                           ; preds = %._crit_edge93
  store i2 %p_Val2_51, i2* @i2c_val, align 1
  br label %.preheader35

.preheader35:                                     ; preds = %.preheader35, %.preheader35.preheader
  %p_Val2_56 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind
  %tmp_42 = trunc i2 %p_Val2_56 to i1
  br i1 %tmp_42, label %.preheader35, label %22

; <label>:22                                      ; preds = %.preheader35
  store i2 %p_Val2_56, i2* @i2c_val, align 1
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @i2c_sda_oe, i1 false) nounwind
  br i1 %ignore_0_s, label %.preheader40, label %23

; <label>:23                                      ; preds = %22
  call void (...)* @_ssdm_op_Wait(i32 1)
  call fastcc void @i2c_slave_core_write_mem(i8 zeroext %re_2, i8 zeroext %reg_data_2)
  call void (...)* @_ssdm_op_Wait(i32 1)
  %auto_inc_regad_in_read = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @auto_inc_regad_in) nounwind
  %reg_addr_1 = add i8 %re_2, 1
  %p_re_2 = select i1 %auto_inc_regad_in_read, i8 %reg_addr_1, i8 %re_2
  br label %.preheader40

.preheader34:                                     ; preds = %._crit_edge96, %.preheader41
  %storemerge1 = phi i2 [ %p_Val2_49, %.preheader41 ], [ %p_Val2_58, %._crit_edge96 ]
  %bit_cnt_4 = phi i3 [ 0, %.preheader41 ], [ %bit_cnt_9, %._crit_edge96 ]
  %de_2 = phi i7 [ %dev_addr_2_load, %.preheader41 ], [ %dev_addr_1, %._crit_edge96 ]
  store i2 %storemerge1, i2* @i2c_val, align 1
  %exitcond = icmp eq i3 %bit_cnt_4, -1
  %empty_4 = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 7, i64 7, i64 7) nounwind
  %bit_cnt_9 = add i3 %bit_cnt_4, 1
  br i1 %exitcond, label %26, label %24

; <label>:24                                      ; preds = %.preheader34
  call void (...)* @_ssdm_op_SpecLoopName([17 x i8]* @p_str5) nounwind
  br label %._crit_edge95

._crit_edge95:                                    ; preds = %._crit_edge95, %24
  %p_Val2_55 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind
  %tmp_41 = trunc i2 %p_Val2_55 to i1
  br i1 %tmp_41, label %25, label %._crit_edge95

; <label>:25                                      ; preds = %._crit_edge95
  store i2 %p_Val2_55, i2* @i2c_val, align 1
  %tmp_45 = call i1 @_ssdm_op_BitSelect.i1.i2.i32(i2 %p_Val2_55, i32 1)
  %tmp_46 = trunc i7 %de_2 to i6
  %dev_addr_1 = call i7 @_ssdm_op_BitConcatenate.i7.i6.i1(i6 %tmp_46, i1 %tmp_45)
  br label %._crit_edge96

._crit_edge96:                                    ; preds = %._crit_edge96, %25
  %p_Val2_58 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind
  %tmp_48 = trunc i2 %p_Val2_58 to i1
  br i1 %tmp_48, label %._crit_edge96, label %.preheader34

; <label>:26                                      ; preds = %.preheader34
  %dev_addr_in_read_1 = call i7 @_ssdm_op_Read.ap_none.volatile.i7P(i7* @dev_addr_in) nounwind
  %not_2 = icmp ne i7 %de_2, %dev_addr_in_read_1
  br label %._crit_edge97

._crit_edge97:                                    ; preds = %._crit_edge97, %26
  %p_Val2_54 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind
  %tmp_40 = trunc i2 %p_Val2_54 to i1
  br i1 %tmp_40, label %27, label %._crit_edge97

; <label>:27                                      ; preds = %._crit_edge97
  store i2 %p_Val2_54, i2* @i2c_val, align 1
  %tmp_44 = call i1 @_ssdm_op_BitSelect.i1.i2.i32(i2 %p_Val2_54, i32 1)
  %tmp_7 = xor i1 %tmp_44, true
  %p_ignore_2 = or i1 %not_2, %tmp_7
  br label %._crit_edge98

._crit_edge98:                                    ; preds = %._crit_edge98, %27
  %p_Val2_57 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind
  %tmp_47 = trunc i2 %p_Val2_57 to i1
  br i1 %tmp_47, label %._crit_edge98, label %_ifconv1

_ifconv1:                                         ; preds = %._crit_edge98
  store i2 %p_Val2_57, i2* @i2c_val, align 1
  call void (...)* @_ssdm_op_Wait(i32 1)
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @i2c_sda_out, i1 %p_ignore_2) nounwind
  %not_2_not = xor i1 %not_2, true
  %not_ignore_3 = and i1 %tmp_44, %not_2_not
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @i2c_sda_oe, i1 %not_ignore_3) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  %reg_data_5 = call fastcc zeroext i8 @i2c_slave_core_read_mem(i8 zeroext %reg_addr_4_load)
  call void (...)* @_ssdm_op_Wait(i32 1)
  %auto_inc_regad_in_read_1 = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @auto_inc_regad_in) nounwind
  %reg_addr_2 = add i8 %reg_addr_4_load, 1
  %re_1_s = select i1 %p_ignore_2, i8 %reg_addr_4_load, i8 %reg_addr_2
  %re_6 = select i1 %auto_inc_regad_in_read_1, i8 %re_1_s, i8 %reg_addr_4_load
  br label %._crit_edge100

._crit_edge100:                                   ; preds = %._crit_edge100, %_ifconv1
  %p_Val2_59 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind
  %tmp_49 = trunc i2 %p_Val2_59 to i1
  br i1 %tmp_49, label %.preheader33.preheader, label %._crit_edge100

.preheader33.preheader:                           ; preds = %._crit_edge100
  store i2 %p_Val2_59, i2* @i2c_val, align 1
  br label %.preheader33

.preheader33:                                     ; preds = %.preheader33, %.preheader33.preheader
  %p_Val2_60 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind
  %tmp_50 = trunc i2 %p_Val2_60 to i1
  br i1 %tmp_50, label %.preheader33, label %.preheader31.preheader

.preheader31.preheader:                           ; preds = %.preheader33
  store i2 %p_Val2_60, i2* @i2c_val, align 1
  br label %.preheader31

.preheader31:                                     ; preds = %35, %.preheader31.preheader
  %terminate_read = phi i1 [ %terminate_read_1, %35 ], [ false, %.preheader31.preheader ]
  %p_Val2_61 = phi i8 [ %reg_data_6, %35 ], [ %reg_data_5, %.preheader31.preheader ]
  %re_7 = phi i8 [ %re_8, %35 ], [ %re_6, %.preheader31.preheader ]
  %tmp_3 = call i32 (...)* @_ssdm_op_SpecRegionBegin([12 x i8]* @p_str6) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  %tmp_51 = call i1 @_ssdm_op_BitSelect.i1.i8.i32(i8 %p_Val2_61, i32 7)
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @i2c_sda_out, i1 %tmp_51) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @i2c_sda_oe, i1 %not_ignore_3) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  br label %28

; <label>:28                                      ; preds = %._crit_edge102, %.preheader31
  %bit_cnt_5 = phi i4 [ 0, %.preheader31 ], [ %bit_cnt_10, %._crit_edge102 ]
  %reg_data_4 = phi i8 [ %p_Val2_61, %.preheader31 ], [ %reg_data_8, %._crit_edge102 ]
  %tmp_52 = call i1 @_ssdm_op_BitSelect.i1.i4.i32(i4 %bit_cnt_5, i32 3)
  %empty_5 = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 1, i64 8, i64 4) nounwind
  %bit_cnt_10 = add i4 %bit_cnt_5, 1
  br i1 %tmp_52, label %_ifconv, label %.preheader

.preheader:                                       ; preds = %28, %29
  %p_Val2_63 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind
  %brmerge2 = or i1 %p_ignore_2, %terminate_read
  br i1 %brmerge2, label %.backedge.loopexit83, label %29

; <label>:29                                      ; preds = %.preheader
  %tmp_54 = trunc i2 %p_Val2_63 to i1
  br i1 %tmp_54, label %30, label %.preheader

; <label>:30                                      ; preds = %29
  store i2 %p_Val2_63, i2* @i2c_val, align 1
  %tmp_56 = shl i8 %reg_data_4, 1
  %reg_data_8 = or i8 %tmp_56, 1
  br label %._crit_edge101

._crit_edge101:                                   ; preds = %31, %30
  %p_Val2_65 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind
  %tmp_58 = call i1 @_ssdm_op_BitSelect.i1.i2.i32(i2 %p_Val2_65, i32 1)
  %tmp_s = xor i1 %tmp_58, true
  %brmerge3 = or i1 %pre_i2c_sda_val, %tmp_s
  br i1 %brmerge3, label %31, label %.backedge.loopexit

; <label>:31                                      ; preds = %._crit_edge101
  %tmp_59 = trunc i2 %p_Val2_65 to i1
  br i1 %tmp_59, label %._crit_edge101, label %32

; <label>:32                                      ; preds = %31
  store i2 %p_Val2_65, i2* @i2c_val, align 1
  %tmp_10 = icmp ult i4 %bit_cnt_5, 7
  br i1 %tmp_10, label %33, label %._crit_edge102

; <label>:33                                      ; preds = %32
  call void (...)* @_ssdm_op_Wait(i32 1)
  %tmp_60 = call i1 @_ssdm_op_BitSelect.i1.i8.i32(i8 %reg_data_8, i32 7)
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @i2c_sda_out, i1 %tmp_60) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  br label %._crit_edge102

._crit_edge102:                                   ; preds = %33, %32
  br label %28

_ifconv:                                          ; preds = %28
  call void (...)* @_ssdm_op_Wait(i32 1)
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @i2c_sda_oe, i1 false) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  %reg_data_6 = call fastcc zeroext i8 @i2c_slave_core_read_mem(i8 zeroext %re_7)
  call void (...)* @_ssdm_op_Wait(i32 1)
  %auto_inc_regad_in_read_2 = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @auto_inc_regad_in) nounwind
  %reg_addr_3 = add i8 %re_7, 1
  %re_7_s = select i1 %p_ignore_2, i8 %re_7, i8 %reg_addr_3
  %re_8 = select i1 %auto_inc_regad_in_read_2, i8 %re_7_s, i8 %re_7
  br label %._crit_edge104

._crit_edge104:                                   ; preds = %._crit_edge104, %_ifconv
  %p_Val2_62 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind
  %tmp_53 = trunc i2 %p_Val2_62 to i1
  br i1 %tmp_53, label %34, label %._crit_edge104

; <label>:34                                      ; preds = %._crit_edge104
  store i2 %p_Val2_62, i2* @i2c_val, align 1
  %terminate_read_1 = call i1 @_ssdm_op_BitSelect.i1.i2.i32(i2 %p_Val2_62, i32 1)
  br label %._crit_edge105

._crit_edge105:                                   ; preds = %._crit_edge105, %34
  %p_Val2_64 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind
  %tmp_57 = trunc i2 %p_Val2_64 to i1
  br i1 %tmp_57, label %._crit_edge105, label %35

; <label>:35                                      ; preds = %._crit_edge105
  store i2 %p_Val2_64, i2* @i2c_val, align 1
  %empty_6 = call i32 (...)* @_ssdm_op_SpecRegionEnd([12 x i8]* @p_str6, i32 %tmp_3) nounwind
  br label %.preheader31
}

define weak void @_ssdm_op_Write.ap_none.volatile.i8P(i8*, i8) {
entry:
  store i8 %1, i8* %0
  ret void
}

define weak void @_ssdm_op_Write.ap_none.volatile.i1P(i1*, i1) {
entry:
  store i1 %1, i1* %0
  ret void
}

define weak void @_ssdm_op_Wait(...) nounwind {
entry:
  ret void
}

define weak void @_ssdm_op_SpecTopModule(...) nounwind {
entry:
  ret void
}

define weak i32 @_ssdm_op_SpecRegionEnd(...) {
entry:
  ret i32 0
}

define weak i32 @_ssdm_op_SpecRegionBegin(...) {
entry:
  ret i32 0
}

define weak i32 @_ssdm_op_SpecLoopTripCount(...) {
entry:
  ret i32 0
}

define weak void @_ssdm_op_SpecLoopName(...) nounwind {
entry:
  ret void
}

define weak i32 @_ssdm_op_SpecLoopBegin(...) {
entry:
  ret i32 0
}

define weak void @_ssdm_op_SpecInterface(...) nounwind {
entry:
  ret void
}

define weak i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8*) {
entry:
  %empty = load i8* %0
  ret i8 %empty
}

define weak i7 @_ssdm_op_Read.ap_none.volatile.i7P(i7*) {
entry:
  %empty = load i7* %0
  ret i7 %empty
}

define weak i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2*) {
entry:
  %empty = load i2* %0
  ret i2 %empty
}

define weak i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1*) {
entry:
  %empty = load i1* %0
  ret i1 %empty
}

define weak i8 @_ssdm_op_Read.ap_auto.i8(i8) {
entry:
  ret i8 %0
}

declare i7 @_ssdm_op_PartSelect.i7.i8.i32.i32(i8, i32, i32) nounwind readnone

declare i6 @_ssdm_op_PartSelect.i6.i7.i32.i32(i7, i32, i32) nounwind readnone

declare i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2, i32, i32) nounwind readnone

define weak i1 @_ssdm_op_BitSelect.i1.i8.i32(i8, i32) nounwind readnone {
entry:
  %empty = trunc i32 %1 to i8
  %empty_7 = shl i8 1, %empty
  %empty_8 = and i8 %0, %empty_7
  %empty_9 = icmp ne i8 %empty_8, 0
  ret i1 %empty_9
}

define weak i1 @_ssdm_op_BitSelect.i1.i4.i32(i4, i32) nounwind readnone {
entry:
  %empty = trunc i32 %1 to i4
  %empty_10 = shl i4 1, %empty
  %empty_11 = and i4 %0, %empty_10
  %empty_12 = icmp ne i4 %empty_11, 0
  ret i1 %empty_12
}

define weak i1 @_ssdm_op_BitSelect.i1.i2.i32(i2, i32) nounwind readnone {
entry:
  %empty = trunc i32 %1 to i2
  %empty_13 = shl i2 1, %empty
  %empty_14 = and i2 %0, %empty_13
  %empty_15 = icmp ne i2 %empty_14, 0
  ret i1 %empty_15
}

define weak i8 @_ssdm_op_BitConcatenate.i8.i7.i1(i7, i1) nounwind readnone {
entry:
  %empty = zext i7 %0 to i8
  %empty_16 = zext i1 %1 to i8
  %empty_17 = shl i8 %empty, 1
  %empty_18 = or i8 %empty_17, %empty_16
  ret i8 %empty_18
}

define weak i7 @_ssdm_op_BitConcatenate.i7.i6.i1(i6, i1) nounwind readnone {
entry:
  %empty = zext i6 %0 to i7
  %empty_19 = zext i1 %1 to i7
  %empty_20 = shl i7 %empty, 1
  %empty_21 = or i7 %empty_20, %empty_19
  ret i7 %empty_21
}

!hls.encrypted.func = !{}
!llvm.map.gv = !{!0, !7, !12, !17, !22, !27, !32, !37, !42, !47, !52, !57, !62}

!0 = metadata !{metadata !1, i1* @mem_wreq}
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0, i32 0, metadata !3}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"mem_wreq", metadata !5, metadata !"uint1", i32 0, i32 0}
!5 = metadata !{metadata !6}
!6 = metadata !{i32 0, i32 0, i32 1}
!7 = metadata !{metadata !8, i1* @mem_wack}
!8 = metadata !{metadata !9}
!9 = metadata !{i32 0, i32 0, metadata !10}
!10 = metadata !{metadata !11}
!11 = metadata !{metadata !"mem_wack", metadata !5, metadata !"uint1", i32 0, i32 0}
!12 = metadata !{metadata !13, i1* @mem_rreq}
!13 = metadata !{metadata !14}
!14 = metadata !{i32 0, i32 0, metadata !15}
!15 = metadata !{metadata !16}
!16 = metadata !{metadata !"mem_rreq", metadata !5, metadata !"uint1", i32 0, i32 0}
!17 = metadata !{metadata !18, i1* @mem_rack}
!18 = metadata !{metadata !19}
!19 = metadata !{i32 0, i32 0, metadata !20}
!20 = metadata !{metadata !21}
!21 = metadata !{metadata !"mem_rack", metadata !5, metadata !"uint1", i32 0, i32 0}
!22 = metadata !{metadata !23, i8* @mem_dout}
!23 = metadata !{metadata !24}
!24 = metadata !{i32 0, i32 7, metadata !25}
!25 = metadata !{metadata !26}
!26 = metadata !{metadata !"mem_dout", metadata !5, metadata !"uint8", i32 0, i32 7}
!27 = metadata !{metadata !28, i8* @mem_din}
!28 = metadata !{metadata !29}
!29 = metadata !{i32 0, i32 7, metadata !30}
!30 = metadata !{metadata !31}
!31 = metadata !{metadata !"mem_din", metadata !5, metadata !"uint8", i32 0, i32 7}
!32 = metadata !{metadata !33, i8* @mem_addr}
!33 = metadata !{metadata !34}
!34 = metadata !{i32 0, i32 7, metadata !35}
!35 = metadata !{metadata !36}
!36 = metadata !{metadata !"mem_addr", metadata !5, metadata !"uint8", i32 0, i32 7}
!37 = metadata !{metadata !38, i2* @i2c_val}
!38 = metadata !{metadata !39}
!39 = metadata !{i32 0, i32 1, metadata !40}
!40 = metadata !{metadata !41}
!41 = metadata !{metadata !"i2c_val", metadata !5, metadata !"uint2", i32 0, i32 1}
!42 = metadata !{metadata !43, i1* @i2c_sda_out}
!43 = metadata !{metadata !44}
!44 = metadata !{i32 0, i32 0, metadata !45}
!45 = metadata !{metadata !46}
!46 = metadata !{metadata !"i2c_sda_out", metadata !5, metadata !"uint1", i32 0, i32 0}
!47 = metadata !{metadata !48, i1* @i2c_sda_oe}
!48 = metadata !{metadata !49}
!49 = metadata !{i32 0, i32 0, metadata !50}
!50 = metadata !{metadata !51}
!51 = metadata !{metadata !"i2c_sda_oe", metadata !5, metadata !"uint1", i32 0, i32 0}
!52 = metadata !{metadata !53, i2* @i2c_in}
!53 = metadata !{metadata !54}
!54 = metadata !{i32 0, i32 1, metadata !55}
!55 = metadata !{metadata !56}
!56 = metadata !{metadata !"i2c_in", metadata !5, metadata !"uint2", i32 0, i32 1}
!57 = metadata !{metadata !58, i7* @dev_addr_in}
!58 = metadata !{metadata !59}
!59 = metadata !{i32 0, i32 6, metadata !60}
!60 = metadata !{metadata !61}
!61 = metadata !{metadata !"dev_addr_in", metadata !5, metadata !"uint7", i32 0, i32 6}
!62 = metadata !{metadata !63, i1* @auto_inc_regad_in}
!63 = metadata !{metadata !64}
!64 = metadata !{i32 0, i32 0, metadata !65}
!65 = metadata !{metadata !66}
!66 = metadata !{metadata !"auto_inc_regad_in", metadata !5, metadata !"uint1", i32 0, i32 0}
