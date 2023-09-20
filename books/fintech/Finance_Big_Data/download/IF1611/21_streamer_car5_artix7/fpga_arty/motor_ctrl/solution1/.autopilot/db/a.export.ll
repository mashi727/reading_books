; ModuleID = 'D:/21_streamer_car5_artix7/fpga_arty/motor_ctrl/solution1/.autopilot/db/a.o.2.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-w64-mingw32"

@r_pwm = global i1 false, align 1
@r_dir = global i1 false, align 1
@mem_wreq = global i1 false, align 1
@mem_wack = common global i1 false, align 1
@mem_rreq = global i1 false, align 1
@mem_rack = common global i1 false, align 1
@mem_dout = global i8 0, align 1
@mem_din = common global i8 0, align 1
@mem_addr = global i8 0, align 1
@l_pwm = global i1 false, align 1
@l_dir = global i1 false, align 1
@dummy_tmr_out = global i1 false, align 1
@p_str1 = private unnamed_addr constant [8 x i8] c"ap_none\00", align 1
@p_str = private unnamed_addr constant [1 x i8] zeroinitializer, align 1

define internal fastcc void @motor_ctrl_write_mem(i8 zeroext %addr, i8 zeroext %data) nounwind uwtable {
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

define internal fastcc void @motor_ctrl_wait_tmr() nounwind uwtable {
  call void (...)* @_ssdm_op_Wait(i32 1)
  br label %1

; <label>:1                                       ; preds = %2, %0
  %t = phi i17 [ 0, %0 ], [ %t_1, %2 ]
  %exitcond = icmp eq i17 %t, -31072
  %empty = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 100000, i64 100000, i64 100000) nounwind
  %t_1 = add i17 %t, 1
  br i1 %exitcond, label %3, label %2

; <label>:2                                       ; preds = %1
  %dummy_tmr_out_read = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @dummy_tmr_out) nounwind
  %not_s = xor i1 %dummy_tmr_out_read, true
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @dummy_tmr_out, i1 %not_s) nounwind
  br label %1

; <label>:3                                       ; preds = %1
  call void (...)* @_ssdm_op_Wait(i32 1)
  ret void
}

define internal fastcc zeroext i8 @motor_ctrl_read_mem(i8 zeroext %addr) nounwind uwtable {
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

define void @motor_ctrl() noreturn nounwind uwtable {
  %mtr_pwm_cnt = alloca i32, align 4
  call void (...)* @_ssdm_op_SpecTopModule()
  %dummy_tmr_out_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @dummy_tmr_out) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @dummy_tmr_out, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind
  %l_dir_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @l_dir) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @l_dir, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind
  %l_pwm_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @l_pwm) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @l_pwm, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind
  %r_dir_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @r_dir) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @r_dir, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind
  %r_pwm_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @r_pwm) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @r_pwm, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind
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
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @l_dir, i1 false) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @l_pwm, i1 false) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @r_dir, i1 false) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @r_pwm, i1 false) nounwind
  store volatile i32 0, i32* %mtr_pwm_cnt, align 4
  call void (...)* @_ssdm_op_Wait(i32 1)
  call fastcc void @motor_ctrl_write_mem(i8 zeroext -128, i8 zeroext 0)
  call fastcc void @motor_ctrl_write_mem(i8 zeroext -123, i8 zeroext 0)
  call fastcc void @motor_ctrl_write_mem(i8 zeroext -122, i8 zeroext 0)
  call void (...)* @_ssdm_op_Wait(i32 1)
  br label %1

; <label>:1                                       ; preds = %2, %0
  %i = phi i6 [ 0, %0 ], [ %i_1, %2 ]
  %i_cast = zext i6 %i to i8
  %exitcond = icmp eq i6 %i, -32
  %empty = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 32, i64 32, i64 32) nounwind
  %i_1 = add i6 %i, 1
  br i1 %exitcond, label %.preheader, label %2

; <label>:2                                       ; preds = %1
  call fastcc void @motor_ctrl_write_mem(i8 zeroext %i_cast, i8 zeroext 32)
  br label %1

.preheader:                                       ; preds = %1, %._crit_edge11
  %chR_dir = phi i1 [ %chR_dir_5, %._crit_edge11 ], [ false, %1 ]
  %chL_dir = phi i1 [ %chL_dir_5, %._crit_edge11 ], [ false, %1 ]
  %loop_begin = call i32 (...)* @_ssdm_op_SpecLoopBegin() nounwind
  %eh = call fastcc zeroext i8 @motor_ctrl_read_mem(i8 zeroext -127)
  %el = call fastcc zeroext i8 @motor_ctrl_read_mem(i8 zeroext -126)
  %et = call i16 @_ssdm_op_BitConcatenate.i16.i8.i8(i8 %eh, i8 %el)
  call void (...)* @_ssdm_op_Wait(i32 1)
  br label %._crit_edge

._crit_edge:                                      ; preds = %._crit_edge, %.preheader
  %tmp_s = call fastcc zeroext i8 @motor_ctrl_read_mem(i8 zeroext -123)
  %tmp_1 = icmp eq i8 %tmp_s, 0
  br i1 %tmp_1, label %._crit_edge, label %3

; <label>:3                                       ; preds = %._crit_edge
  call void (...)* @_ssdm_op_Wait(i32 1)
  %eh_1 = call fastcc zeroext i8 @motor_ctrl_read_mem(i8 zeroext -125)
  %el_1 = call fastcc zeroext i8 @motor_ctrl_read_mem(i8 zeroext -124)
  %e = call i16 @_ssdm_op_BitConcatenate.i16.i8.i8(i8 %eh_1, i8 %el_1)
  call void (...)* @_ssdm_op_Wait(i32 1)
  %tmp_3 = call i4 @_ssdm_op_PartSelect.i4.i8.i32.i32(i8 %eh_1, i32 4, i32 7)
  %tmp_4 = call fastcc zeroext i7 @motor_ctrl_bin2char(i4 zeroext %tmp_3) nounwind
  %p_trunc_ext = zext i7 %tmp_4 to i8
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 3, i8 zeroext %p_trunc_ext)
  %tmp_2 = trunc i8 %eh_1 to i4
  %tmp_6 = call fastcc zeroext i7 @motor_ctrl_bin2char(i4 zeroext %tmp_2) nounwind
  %p_trunc115_ext = zext i7 %tmp_6 to i8
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 4, i8 zeroext %p_trunc115_ext)
  %tmp_8 = call i4 @_ssdm_op_PartSelect.i4.i8.i32.i32(i8 %el_1, i32 4, i32 7)
  %tmp_9 = call fastcc zeroext i7 @motor_ctrl_bin2char(i4 zeroext %tmp_8) nounwind
  %p_trunc116_ext = zext i7 %tmp_9 to i8
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 5, i8 zeroext %p_trunc116_ext)
  %tmp_5 = trunc i8 %el_1 to i4
  %tmp_7 = call fastcc zeroext i7 @motor_ctrl_bin2char(i4 zeroext %tmp_5) nounwind
  %p_trunc117_ext = zext i7 %tmp_7 to i8
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 6, i8 zeroext %p_trunc117_ext)
  %tmp_10 = call i4 @_ssdm_op_PartSelect.i4.i8.i32.i32(i8 %eh, i32 4, i32 7)
  %tmp_11 = call fastcc zeroext i7 @motor_ctrl_bin2char(i4 zeroext %tmp_10) nounwind
  %p_trunc118_ext = zext i7 %tmp_11 to i8
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 8, i8 zeroext %p_trunc118_ext)
  %tmp_12 = trunc i8 %eh to i4
  %tmp_13 = call fastcc zeroext i7 @motor_ctrl_bin2char(i4 zeroext %tmp_12) nounwind
  %p_trunc119_ext = zext i7 %tmp_13 to i8
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 9, i8 zeroext %p_trunc119_ext)
  %tmp_14 = call i4 @_ssdm_op_PartSelect.i4.i8.i32.i32(i8 %el, i32 4, i32 7)
  %tmp_15 = call fastcc zeroext i7 @motor_ctrl_bin2char(i4 zeroext %tmp_14) nounwind
  %p_trunc120_ext = zext i7 %tmp_15 to i8
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 10, i8 zeroext %p_trunc120_ext)
  %tmp_16 = trunc i8 %el to i4
  %tmp_17 = call fastcc zeroext i7 @motor_ctrl_bin2char(i4 zeroext %tmp_16) nounwind
  %p_trunc121_ext = zext i7 %tmp_17 to i8
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 11, i8 zeroext %p_trunc121_ext)
  %diff_agl = call fastcc i21 @motor_ctrl_diff_angle(i16 zeroext %et, i16 zeroext %e) nounwind
  %tmp_18 = call i4 @_ssdm_op_PartSelect.i4.i21.i32.i32(i21 %diff_agl, i32 12, i32 15)
  %tmp_19 = call fastcc zeroext i7 @motor_ctrl_bin2char(i4 zeroext %tmp_18) nounwind
  %p_trunc122_ext = zext i7 %tmp_19 to i8
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 16, i8 zeroext %p_trunc122_ext)
  %tmp_20 = call i4 @_ssdm_op_PartSelect.i4.i21.i32.i32(i21 %diff_agl, i32 8, i32 11)
  %tmp_21 = call fastcc zeroext i7 @motor_ctrl_bin2char(i4 zeroext %tmp_20) nounwind
  %p_trunc123_ext = zext i7 %tmp_21 to i8
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 17, i8 zeroext %p_trunc123_ext)
  %tmp_22 = call i4 @_ssdm_op_PartSelect.i4.i21.i32.i32(i21 %diff_agl, i32 4, i32 7)
  %tmp_23 = call fastcc zeroext i7 @motor_ctrl_bin2char(i4 zeroext %tmp_22) nounwind
  %p_trunc124_ext = zext i7 %tmp_23 to i8
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 18, i8 zeroext %p_trunc124_ext)
  %tmp_24 = trunc i21 %diff_agl to i4
  %tmp_25 = call fastcc zeroext i7 @motor_ctrl_bin2char(i4 zeroext %tmp_24) nounwind
  %p_trunc125_ext = zext i7 %tmp_25 to i8
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 19, i8 zeroext %p_trunc125_ext)
  %tmp_26 = icmp slt i21 %diff_agl, -249
  br i1 %tmp_26, label %4, label %5

; <label>:4                                       ; preds = %3
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 21, i8 zeroext 76)
  br label %6

; <label>:5                                       ; preds = %3
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 21, i8 zeroext 45)
  br label %6

; <label>:6                                       ; preds = %5, %4
  %too_left = phi i1 [ true, %4 ], [ false, %5 ]
  %tmp_27 = icmp sgt i21 %diff_agl, 249
  br i1 %tmp_27, label %7, label %8

; <label>:7                                       ; preds = %6
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 22, i8 zeroext 82)
  br label %_ifconv

; <label>:8                                       ; preds = %6
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 22, i8 zeroext 45)
  br label %_ifconv

_ifconv:                                          ; preds = %8, %7
  %too_right = phi i1 [ true, %7 ], [ false, %8 ]
  %mode = call fastcc zeroext i8 @motor_ctrl_read_mem(i8 zeroext -128)
  %tmp_36 = call i5 @_ssdm_op_PartSelect.i5.i8.i32.i32(i8 %mode, i32 3, i32 7)
  %zext_cast = zext i5 %tmp_36 to i12
  %mul = mul i12 43, %zext_cast
  %chR_pwm_cast = call i4 @_ssdm_op_PartSelect.i4.i12.i32.i32(i12 %mul, i32 7, i32 10)
  %tmp_37 = trunc i8 %mode to i3
  %brmerge = or i1 %too_right, %too_left
  %p_s = select i1 %too_left, i4 0, i4 %chR_pwm_cast
  %p_1 = select i1 %too_left, i4 -6, i4 %chR_pwm_cast
  %chL_pwm_4 = select i1 %too_right, i4 0, i4 -6
  %not_too_right_s = xor i1 %too_right, true
  %chR_pwm_6 = select i1 %too_right, i4 -6, i4 0
  %sel_tmp = icmp eq i3 %tmp_37, 1
  %sel_tmp2 = and i1 %sel_tmp, %not_too_right_s
  %sel_tmp3 = select i1 %sel_tmp2, i4 %p_s, i4 0
  %sel_tmp4 = icmp eq i3 %tmp_37, 3
  %sel_tmp6 = and i1 %sel_tmp4, %not_too_right_s
  %sel_tmp7 = select i1 %sel_tmp6, i4 %p_1, i4 %sel_tmp3
  %sel_tmp8 = icmp eq i3 %tmp_37, -3
  %sel_tmp9 = and i1 %sel_tmp8, %brmerge
  %sel_tmp1 = select i1 %sel_tmp9, i4 -6, i4 %sel_tmp7
  %sel_tmp5 = icmp eq i3 %tmp_37, -1
  %sel_tmp10 = and i1 %sel_tmp5, %brmerge
  %sel_tmp11 = select i1 %sel_tmp10, i4 %chR_pwm_6, i4 %sel_tmp1
  %sel_tmp12 = and i1 %sel_tmp, %too_right
  %sel_tmp13 = and i1 %sel_tmp4, %too_right
  %sel_tmp14 = select i1 %sel_tmp13, i4 0, i4 -6
  %tmp = or i1 %sel_tmp13, %sel_tmp12
  %sel_tmp15 = select i1 %tmp, i4 %sel_tmp14, i4 %sel_tmp11
  %sel_tmp16 = xor i1 %brmerge, true
  %sel_tmp17 = and i1 %sel_tmp8, %sel_tmp16
  %sel_tmp18 = and i1 %sel_tmp5, %sel_tmp16
  %tmp_28 = or i1 %sel_tmp18, %sel_tmp17
  %chR_pwm_8 = select i1 %tmp_28, i4 0, i4 %sel_tmp15
  %sel_tmp19 = select i1 %sel_tmp2, i4 %p_1, i4 0
  %sel_tmp20 = select i1 %sel_tmp6, i4 %p_s, i4 %sel_tmp19
  %sel_tmp21 = select i1 %sel_tmp9, i4 %chL_pwm_4, i4 %sel_tmp20
  %sel_tmp22 = select i1 %sel_tmp10, i4 -6, i4 %sel_tmp21
  %sel_tmp23 = select i1 %sel_tmp13, i4 -6, i4 0
  %sel_tmp24 = select i1 %tmp, i4 %sel_tmp23, i4 %sel_tmp22
  %chL_pwm_8 = select i1 %tmp_28, i4 0, i4 %sel_tmp24
  %sel_tmp56_not = icmp ne i3 %tmp_37, 1
  %not_sel_tmp = or i1 %too_right, %sel_tmp56_not
  %sel_tmp25 = and i1 %chR_dir, %not_sel_tmp
  %sel_tmp26 = or i1 %sel_tmp6, %sel_tmp25
  %sel_tmp27 = select i1 %sel_tmp9, i1 %not_too_right_s, i1 %sel_tmp26
  %sel_tmp28 = select i1 %sel_tmp10, i1 %not_too_right_s, i1 %sel_tmp27
  %not_sel_tmp1 = xor i1 %sel_tmp12, true
  %sel_tmp29 = and i1 %sel_tmp28, %not_sel_tmp1
  %sel_tmp30 = or i1 %sel_tmp13, %sel_tmp29
  %sel_tmp31 = select i1 %sel_tmp17, i1 %chR_dir, i1 %sel_tmp30
  %chR_dir_5 = select i1 %sel_tmp18, i1 %chR_dir, i1 %sel_tmp31
  %sel_tmp32 = and i1 %chL_dir, %not_sel_tmp
  %sel_tmp33 = or i1 %sel_tmp6, %sel_tmp32
  %sel_tmp34 = select i1 %sel_tmp9, i1 %too_right, i1 %sel_tmp33
  %sel_tmp35 = select i1 %sel_tmp10, i1 %too_right, i1 %sel_tmp34
  %sel_tmp36 = and i1 %sel_tmp35, %not_sel_tmp1
  %sel_tmp37 = or i1 %sel_tmp13, %sel_tmp36
  %sel_tmp38 = select i1 %sel_tmp17, i1 %chL_dir, i1 %sel_tmp37
  %chL_dir_5 = select i1 %sel_tmp18, i1 %chL_dir, i1 %sel_tmp38
  call void (...)* @_ssdm_op_Wait(i32 1)
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @r_dir, i1 %chR_dir_5) nounwind
  %mtr_pwm_cnt_load = load volatile i32* %mtr_pwm_cnt, align 4
  %tmp_29 = zext i4 %chR_pwm_8 to i32
  %tmp_30 = icmp slt i32 %mtr_pwm_cnt_load, %tmp_29
  br i1 %tmp_30, label %9, label %10

; <label>:9                                       ; preds = %_ifconv
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @r_pwm, i1 true) nounwind
  br label %11

; <label>:10                                      ; preds = %_ifconv
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @r_pwm, i1 false) nounwind
  br label %11

; <label>:11                                      ; preds = %10, %9
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @l_dir, i1 %chL_dir_5) nounwind
  %mtr_pwm_cnt_load_1 = load volatile i32* %mtr_pwm_cnt, align 4
  %tmp_31 = zext i4 %chL_pwm_8 to i32
  %tmp_32 = icmp slt i32 %mtr_pwm_cnt_load_1, %tmp_31
  br i1 %tmp_32, label %12, label %13

; <label>:12                                      ; preds = %11
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @l_pwm, i1 true) nounwind
  br label %14

; <label>:13                                      ; preds = %11
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @l_pwm, i1 false) nounwind
  br label %14

; <label>:14                                      ; preds = %13, %12
  call void (...)* @_ssdm_op_Wait(i32 1)
  %mtr_pwm_cnt_load_2 = load volatile i32* %mtr_pwm_cnt, align 4
  %mtr_pwm_cnt_1 = add nsw i32 %mtr_pwm_cnt_load_2, 1
  store volatile i32 %mtr_pwm_cnt_1, i32* %mtr_pwm_cnt, align 4
  %mtr_pwm_cnt_load_3 = load volatile i32* %mtr_pwm_cnt, align 4
  %tmp_33 = icmp sgt i32 %mtr_pwm_cnt_load_3, 9
  br i1 %tmp_33, label %15, label %._crit_edge11

; <label>:15                                      ; preds = %14
  store volatile i32 0, i32* %mtr_pwm_cnt, align 4
  br label %._crit_edge11

._crit_edge11:                                    ; preds = %15, %14
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 24, i8 zeroext 48)
  %tmp_34 = call fastcc zeroext i7 @motor_ctrl_bin2char(i4 zeroext %chL_pwm_8) nounwind
  %p_trunc127_ext = zext i7 %tmp_34 to i8
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 25, i8 zeroext %p_trunc127_ext)
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 27, i8 zeroext 48)
  %tmp_35 = call fastcc zeroext i7 @motor_ctrl_bin2char(i4 zeroext %chR_pwm_8) nounwind
  %p_trunc129_ext = zext i7 %tmp_35 to i8
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 28, i8 zeroext %p_trunc129_ext)
  call fastcc void @motor_ctrl_wait_tmr()
  br label %.preheader
}

declare i8 @llvm.part.select.i8(i8, i32, i32) nounwind readnone

declare i21 @llvm.part.select.i21(i21, i32, i32) nounwind readnone

declare i12 @llvm.part.select.i12(i12, i32, i32) nounwind readnone

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

define internal fastcc i21 @motor_ctrl_diff_angle(i16 zeroext %target, i16 %value) readnone {
  %value_read = call i16 @_ssdm_op_Read.ap_auto.i16(i16 %value)
  %target_read = call i16 @_ssdm_op_Read.ap_auto.i16(i16 %target)
  %tmp_cast3 = zext i16 %value_read to i18
  %tmp_cast = zext i16 %value_read to i17
  %tmp_cast_8 = zext i16 %target_read to i17
  %tmp_s = sub i17 %tmp_cast, %tmp_cast_8
  %tmp_53_cast = sext i17 %tmp_s to i19
  %tmp_36 = add i17 -1, %tmp_cast_8
  %tmp_37 = sub i17 %tmp_36, %tmp_cast
  %tmp_38 = icmp sgt i17 %tmp_37, -18001
  %smax2 = select i1 %tmp_38, i17 %tmp_37, i17 -18001
  %smax2_cast = sext i17 %smax2 to i18
  %tmp_39 = sub i17 36000, %tmp_cast_8
  %tmp_57_cast_cast = sext i17 %tmp_39 to i19
  %tmp1 = add i18 %smax2_cast, %tmp_cast3
  %tmp1_cast_cast = sext i18 %tmp1 to i19
  %tmp_40 = add i19 %tmp1_cast_cast, %tmp_57_cast_cast
  %tmp_41 = urem i19 %tmp_40, 36000
  %tmp_42 = sub i19 %tmp_40, %tmp_41
  %tmp_61_cast = sext i19 %tmp_42 to i20
  %tmp_43 = sub i19 %tmp_53_cast, %tmp_42
  %tmp_62_cast = sext i19 %tmp_43 to i20
  %tmp_44 = icmp sgt i19 %tmp_43, -18000
  %smax1 = select i1 %tmp_44, i19 %tmp_43, i19 -18000
  %smax1_cast = sext i19 %smax1 to i20
  %tmp_45 = add i17 35999, %tmp_cast_8
  %tmp_64_cast = zext i17 %tmp_45 to i18
  %tmp_46 = sub i18 %tmp_64_cast, %tmp_cast3
  %tmp_65_cast_cast = sext i18 %tmp_46 to i21
  %tmp2 = add i20 %tmp_61_cast, %smax1_cast
  %tmp2_cast_cast = sext i20 %tmp2 to i21
  %tmp_47 = add i21 %tmp2_cast_cast, %tmp_65_cast_cast
  %tmp_48 = urem i21 %tmp_47, 36000
  %tmp = trunc i21 %tmp_48 to i20
  %tmp_49 = sub i20 %tmp_62_cast, %tmp
  %tmp_69_cast = sext i20 %tmp_49 to i21
  %retval = add i21 %tmp_69_cast, %tmp_47
  ret i21 %retval
}

define internal fastcc zeroext i7 @motor_ctrl_bin2char(i4 zeroext %val) readnone {
  %val_read = call i4 @_ssdm_op_Read.ap_auto.i4(i4 %val)
  switch i4 %val_read, label %15 [
    i4 0, label %._crit_edge
    i4 1, label %1
    i4 2, label %2
    i4 3, label %3
    i4 4, label %4
    i4 5, label %5
    i4 6, label %6
    i4 7, label %7
    i4 -8, label %8
    i4 -7, label %9
    i4 -6, label %10
    i4 -5, label %11
    i4 -4, label %12
    i4 -3, label %13
    i4 -2, label %14
  ]

; <label>:1                                       ; preds = %0
  br label %._crit_edge

; <label>:2                                       ; preds = %0
  br label %._crit_edge

; <label>:3                                       ; preds = %0
  br label %._crit_edge

; <label>:4                                       ; preds = %0
  br label %._crit_edge

; <label>:5                                       ; preds = %0
  br label %._crit_edge

; <label>:6                                       ; preds = %0
  br label %._crit_edge

; <label>:7                                       ; preds = %0
  br label %._crit_edge

; <label>:8                                       ; preds = %0
  br label %._crit_edge

; <label>:9                                       ; preds = %0
  br label %._crit_edge

; <label>:10                                      ; preds = %0
  br label %._crit_edge

; <label>:11                                      ; preds = %0
  br label %._crit_edge

; <label>:12                                      ; preds = %0
  br label %._crit_edge

; <label>:13                                      ; preds = %0
  br label %._crit_edge

; <label>:14                                      ; preds = %0
  br label %._crit_edge

; <label>:15                                      ; preds = %0
  br label %._crit_edge

._crit_edge:                                      ; preds = %15, %14, %13, %12, %11, %10, %9, %8, %7, %6, %5, %4, %3, %2, %1, %0
  %retval = phi i7 [ -58, %15 ], [ -59, %14 ], [ -60, %13 ], [ -61, %12 ], [ -62, %11 ], [ -63, %10 ], [ 57, %9 ], [ 56, %8 ], [ 55, %7 ], [ 54, %6 ], [ 53, %5 ], [ 52, %4 ], [ 51, %3 ], [ 50, %2 ], [ 49, %1 ], [ 48, %0 ]
  ret i7 %retval
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

define weak i32 @_ssdm_op_SpecLoopTripCount(...) {
entry:
  ret i32 0
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

define weak i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1*) {
entry:
  %empty = load i1* %0
  ret i1 %empty
}

define weak i8 @_ssdm_op_Read.ap_auto.i8(i8) {
entry:
  ret i8 %0
}

define weak i4 @_ssdm_op_Read.ap_auto.i4(i4) {
entry:
  ret i4 %0
}

define weak i16 @_ssdm_op_Read.ap_auto.i16(i16) {
entry:
  ret i16 %0
}

define weak i5 @_ssdm_op_PartSelect.i5.i8.i32.i32(i8, i32, i32) nounwind readnone {
entry:
  %empty = call i8 @llvm.part.select.i8(i8 %0, i32 %1, i32 %2)
  %empty_9 = trunc i8 %empty to i5
  ret i5 %empty_9
}

define weak i4 @_ssdm_op_PartSelect.i4.i8.i32.i32(i8, i32, i32) nounwind readnone {
entry:
  %empty = call i8 @llvm.part.select.i8(i8 %0, i32 %1, i32 %2)
  %empty_10 = trunc i8 %empty to i4
  ret i4 %empty_10
}

define weak i4 @_ssdm_op_PartSelect.i4.i21.i32.i32(i21, i32, i32) nounwind readnone {
entry:
  %empty = call i21 @llvm.part.select.i21(i21 %0, i32 %1, i32 %2)
  %empty_11 = trunc i21 %empty to i4
  ret i4 %empty_11
}

define weak i4 @_ssdm_op_PartSelect.i4.i12.i32.i32(i12, i32, i32) nounwind readnone {
entry:
  %empty = call i12 @llvm.part.select.i12(i12 %0, i32 %1, i32 %2)
  %empty_12 = trunc i12 %empty to i4
  ret i4 %empty_12
}

declare i3 @_ssdm_op_PartSelect.i3.i8.i32.i32(i8, i32, i32) nounwind readnone

declare i20 @_ssdm_op_PartSelect.i20.i21.i32.i32(i21, i32, i32) nounwind readnone

define weak i16 @_ssdm_op_BitConcatenate.i16.i8.i8(i8, i8) nounwind readnone {
entry:
  %empty = zext i8 %0 to i16
  %empty_13 = zext i8 %1 to i16
  %empty_14 = shl i16 %empty, 8
  %empty_15 = or i16 %empty_14, %empty_13
  ret i16 %empty_15
}

!hls.encrypted.func = !{}
!llvm.map.gv = !{!0, !7, !12, !17, !22, !27, !32, !37, !42, !47, !52, !57}

!0 = metadata !{metadata !1, i1* @r_pwm}
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0, i32 0, metadata !3}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"r_pwm", metadata !5, metadata !"uint1", i32 0, i32 0}
!5 = metadata !{metadata !6}
!6 = metadata !{i32 0, i32 0, i32 1}
!7 = metadata !{metadata !8, i1* @r_dir}
!8 = metadata !{metadata !9}
!9 = metadata !{i32 0, i32 0, metadata !10}
!10 = metadata !{metadata !11}
!11 = metadata !{metadata !"r_dir", metadata !5, metadata !"uint1", i32 0, i32 0}
!12 = metadata !{metadata !13, i1* @mem_wreq}
!13 = metadata !{metadata !14}
!14 = metadata !{i32 0, i32 0, metadata !15}
!15 = metadata !{metadata !16}
!16 = metadata !{metadata !"mem_wreq", metadata !5, metadata !"uint1", i32 0, i32 0}
!17 = metadata !{metadata !18, i1* @mem_wack}
!18 = metadata !{metadata !19}
!19 = metadata !{i32 0, i32 0, metadata !20}
!20 = metadata !{metadata !21}
!21 = metadata !{metadata !"mem_wack", metadata !5, metadata !"uint1", i32 0, i32 0}
!22 = metadata !{metadata !23, i1* @mem_rreq}
!23 = metadata !{metadata !24}
!24 = metadata !{i32 0, i32 0, metadata !25}
!25 = metadata !{metadata !26}
!26 = metadata !{metadata !"mem_rreq", metadata !5, metadata !"uint1", i32 0, i32 0}
!27 = metadata !{metadata !28, i1* @mem_rack}
!28 = metadata !{metadata !29}
!29 = metadata !{i32 0, i32 0, metadata !30}
!30 = metadata !{metadata !31}
!31 = metadata !{metadata !"mem_rack", metadata !5, metadata !"uint1", i32 0, i32 0}
!32 = metadata !{metadata !33, i8* @mem_dout}
!33 = metadata !{metadata !34}
!34 = metadata !{i32 0, i32 7, metadata !35}
!35 = metadata !{metadata !36}
!36 = metadata !{metadata !"mem_dout", metadata !5, metadata !"uint8", i32 0, i32 7}
!37 = metadata !{metadata !38, i8* @mem_din}
!38 = metadata !{metadata !39}
!39 = metadata !{i32 0, i32 7, metadata !40}
!40 = metadata !{metadata !41}
!41 = metadata !{metadata !"mem_din", metadata !5, metadata !"uint8", i32 0, i32 7}
!42 = metadata !{metadata !43, i8* @mem_addr}
!43 = metadata !{metadata !44}
!44 = metadata !{i32 0, i32 7, metadata !45}
!45 = metadata !{metadata !46}
!46 = metadata !{metadata !"mem_addr", metadata !5, metadata !"uint8", i32 0, i32 7}
!47 = metadata !{metadata !48, i1* @l_pwm}
!48 = metadata !{metadata !49}
!49 = metadata !{i32 0, i32 0, metadata !50}
!50 = metadata !{metadata !51}
!51 = metadata !{metadata !"l_pwm", metadata !5, metadata !"uint1", i32 0, i32 0}
!52 = metadata !{metadata !53, i1* @l_dir}
!53 = metadata !{metadata !54}
!54 = metadata !{i32 0, i32 0, metadata !55}
!55 = metadata !{metadata !56}
!56 = metadata !{metadata !"l_dir", metadata !5, metadata !"uint1", i32 0, i32 0}
!57 = metadata !{metadata !58, i1* @dummy_tmr_out}
!58 = metadata !{metadata !59}
!59 = metadata !{i32 0, i32 0, metadata !60}
!60 = metadata !{metadata !61}
!61 = metadata !{metadata !"dummy_tmr_out", metadata !5, metadata !"uint1", i32 0, i32 0}
