; ModuleID = 'D:/21_streamer_car5_artix7/fpga_arty/bno055_uart/solution1/.autopilot/db/a.o.2.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-w64-mingw32"

@uart_tx = common global i1 false, align 1
@uart_rx = common global i1 false, align 1
@mem_wreq = global i1 false, align 1
@mem_wack = common global i1 false, align 1
@mem_rreq = global i1 false, align 1
@mem_rack = common global i1 false, align 1
@mem_dout = global i8 0, align 1
@mem_din = common global i8 0, align 1
@mem_addr = global i8 0, align 1
@dummy_tmr_out = global i1 false, align 1
@p_str6 = private unnamed_addr constant [8 x i8] c"ap_none\00", align 1
@p_str = private unnamed_addr constant [1 x i8] zeroinitializer, align 1

define internal fastcc void @bno055_uart_write_mem(i8 zeroext %addr, i8 zeroext %data) nounwind uwtable {
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

define internal fastcc void @bno055_uart_wait_tmr(i28 %tmr) {
  %tmr_read = call i28 @_ssdm_op_Read.ap_auto.i28(i28 %tmr)
  call void (...)* @_ssdm_op_Wait(i32 1)
  br label %1

; <label>:1                                       ; preds = %2, %0
  %t = phi i27 [ 0, %0 ], [ %t_1, %2 ]
  %t_cast = zext i27 %t to i28
  %exitcond = icmp eq i28 %t_cast, %tmr_read
  %empty = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 1, i64 100000000, i64 0) nounwind
  %t_1 = add i27 %t, 1
  br i1 %exitcond, label %3, label %2

; <label>:2                                       ; preds = %1
  %dummy_tmr_out_read = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @dummy_tmr_out)
  %not_s = xor i1 %dummy_tmr_out_read, true
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @dummy_tmr_out, i1 %not_s)
  br label %1

; <label>:3                                       ; preds = %1
  call void (...)* @_ssdm_op_Wait(i32 1)
  ret void
}

define internal fastcc void @bno055_uart_uart_write_reg(i7 zeroext %reg_addr, i8 %data) {
  %data_read = call i8 @_ssdm_op_Read.ap_auto.i8(i8 %data)
  %reg_addr_read = call i7 @_ssdm_op_Read.ap_auto.i7(i7 %reg_addr)
  %reg_addr_cast = zext i7 %reg_addr_read to i8
  call fastcc void @bno055_uart_uart_send_byte(i8 zeroext -86)
  call void (...)* @_ssdm_op_Wait(i32 1)
  call fastcc void @bno055_uart_uart_send_byte(i8 zeroext 0)
  call void (...)* @_ssdm_op_Wait(i32 1)
  call fastcc void @bno055_uart_uart_send_byte(i8 zeroext %reg_addr_cast)
  call void (...)* @_ssdm_op_Wait(i32 1)
  call fastcc void @bno055_uart_uart_send_byte(i8 zeroext 1)
  call void (...)* @_ssdm_op_Wait(i32 1)
  call fastcc void @bno055_uart_uart_send_byte(i8 zeroext %data_read)
  call void (...)* @_ssdm_op_Wait(i32 1)
  call fastcc void @bno055_uart_wait_tmr(i28 1)
  call void (...)* @_ssdm_op_Wait(i32 1)
  %empty = call fastcc zeroext i8 @bno055_uart_uart_receive_byte()
  call void (...)* @_ssdm_op_Wait(i32 1)
  %empty_8 = call fastcc zeroext i8 @bno055_uart_uart_receive_byte()
  call void (...)* @_ssdm_op_Wait(i32 1)
  ret void
}

define internal fastcc void @bno055_uart_uart_send_byte(i8 zeroext %data) nounwind uwtable {
  %data_read = call i8 @_ssdm_op_Read.ap_auto.i8(i8 %data) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @uart_tx, i1 false) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  call fastcc void @bno055_uart_wait_tmr(i28 868) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  br label %1

; <label>:1                                       ; preds = %2, %0
  %i = phi i4 [ 0, %0 ], [ %i_1, %2 ]
  %p_Val2_s = phi i8 [ %data_read, %0 ], [ %tmp, %2 ]
  %exitcond = icmp eq i4 %i, -8
  %empty = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 8, i64 8, i64 8) nounwind
  %i_1 = add i4 %i, 1
  br i1 %exitcond, label %3, label %2

; <label>:2                                       ; preds = %1
  %dt = trunc i8 %p_Val2_s to i1
  %data_assign = call i7 @_ssdm_op_PartSelect.i7.i8.i32.i32(i8 %p_Val2_s, i32 1, i32 7)
  %tmp = zext i7 %data_assign to i8
  call void (...)* @_ssdm_op_Wait(i32 1)
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @uart_tx, i1 %dt) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  call fastcc void @bno055_uart_wait_tmr(i28 868) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  br label %1

; <label>:3                                       ; preds = %1
  call void (...)* @_ssdm_op_Wait(i32 1)
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @uart_tx, i1 true) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  call fastcc void @bno055_uart_wait_tmr(i28 868) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  ret void
}

define internal fastcc zeroext i8 @bno055_uart_uart_receive_byte() nounwind uwtable {
  br label %._crit_edge

._crit_edge:                                      ; preds = %._crit_edge, %0
  call void (...)* @_ssdm_op_Wait(i32 1)
  %tmp = call fastcc zeroext i1 @bno055_uart_read_uart_rx()
  br i1 %tmp, label %.preheader, label %._crit_edge

.preheader:                                       ; preds = %._crit_edge, %._crit_edge1
  %timer = phi i24 [ %timer_1, %._crit_edge1 ], [ 0, %._crit_edge ]
  %tmp_1 = icmp ult i24 %timer, -6777216
  %empty = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 1, i64 10000000, i64 5000000) nounwind
  %timer_1 = add i24 %timer, 1
  br i1 %tmp_1, label %1, label %.loopexit

; <label>:1                                       ; preds = %.preheader
  call void (...)* @_ssdm_op_Wait(i32 1)
  %tmp_2 = call fastcc zeroext i1 @bno055_uart_read_uart_rx()
  br i1 %tmp_2, label %._crit_edge1, label %2

; <label>:2                                       ; preds = %1
  call void (...)* @_ssdm_op_Wait(i32 1)
  call fastcc void @bno055_uart_wait_tmr(i28 217) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  %tmp_3 = call fastcc zeroext i1 @bno055_uart_read_uart_rx()
  br i1 %tmp_3, label %._crit_edge1, label %.loopexit

._crit_edge1:                                     ; preds = %2, %1
  br label %.preheader

.loopexit:                                        ; preds = %2, %.preheader
  call void (...)* @_ssdm_op_Wait(i32 1)
  call fastcc void @bno055_uart_wait_tmr(i28 651) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  br label %3

; <label>:3                                       ; preds = %4, %.loopexit
  %data = phi i8 [ 0, %.loopexit ], [ %data_1, %4 ]
  %i = phi i4 [ 0, %.loopexit ], [ %i_2, %4 ]
  %exitcond = icmp eq i4 %i, -8
  %empty_9 = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 8, i64 8, i64 8) nounwind
  %i_2 = add i4 %i, 1
  br i1 %exitcond, label %5, label %4

; <label>:4                                       ; preds = %3
  call void (...)* @_ssdm_op_Wait(i32 1)
  call fastcc void @bno055_uart_wait_tmr(i28 434) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  %tmp_5 = call fastcc zeroext i1 @bno055_uart_read_uart_rx()
  call void (...)* @_ssdm_op_Wait(i32 1)
  %tmp_6 = call i7 @_ssdm_op_PartSelect.i7.i8.i32.i32(i8 %data, i32 1, i32 7)
  %tmp_4 = select i1 %tmp_5, i1 true, i1 false
  %data_1 = call i8 @_ssdm_op_BitConcatenate.i8.i1.i7(i1 %tmp_4, i7 %tmp_6)
  call void (...)* @_ssdm_op_Wait(i32 1)
  call fastcc void @bno055_uart_wait_tmr(i28 434) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  br label %3

; <label>:5                                       ; preds = %3
  ret i8 %data
}

define internal fastcc zeroext i16 @bno055_uart_uart_read_reg16() nounwind uwtable {
  call fastcc void @bno055_uart_uart_send_byte(i8 zeroext -86)
  call void (...)* @_ssdm_op_Wait(i32 1)
  call fastcc void @bno055_uart_uart_send_byte(i8 zeroext 1)
  call void (...)* @_ssdm_op_Wait(i32 1)
  call fastcc void @bno055_uart_uart_send_byte(i8 zeroext 26)
  call void (...)* @_ssdm_op_Wait(i32 1)
  call fastcc void @bno055_uart_uart_send_byte(i8 zeroext 2)
  call void (...)* @_ssdm_op_Wait(i32 1)
  call fastcc void @bno055_uart_wait_tmr(i28 1) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  %buf = call fastcc zeroext i8 @bno055_uart_uart_receive_byte()
  call void (...)* @_ssdm_op_Wait(i32 1)
  %tmp = icmp eq i8 %buf, -69
  br i1 %tmp, label %1, label %2

; <label>:1                                       ; preds = %0
  call void (...)* @_ssdm_op_Wait(i32 1)
  %empty = call fastcc zeroext i8 @bno055_uart_uart_receive_byte()
  call void (...)* @_ssdm_op_Wait(i32 1)
  %bh = call fastcc zeroext i8 @bno055_uart_uart_receive_byte()
  call void (...)* @_ssdm_op_Wait(i32 1)
  %bl = call fastcc zeroext i8 @bno055_uart_uart_receive_byte()
  %tmp_9 = call i16 @_ssdm_op_BitConcatenate.i16.i8.i8(i8 %bh, i8 %bl)
  br label %5

; <label>:2                                       ; preds = %0
  %tmp_s = icmp eq i8 %buf, -18
  br i1 %tmp_s, label %3, label %4

; <label>:3                                       ; preds = %2
  call void (...)* @_ssdm_op_Wait(i32 1)
  %buf_1 = call fastcc zeroext i8 @bno055_uart_uart_receive_byte()
  %tmp_1 = zext i8 %buf_1 to i16
  br label %5

; <label>:4                                       ; preds = %2
  %tmp_2 = zext i8 %buf to i16
  br label %5

; <label>:5                                       ; preds = %4, %3, %1
  %p_0 = phi i16 [ %tmp_9, %1 ], [ %tmp_1, %3 ], [ %tmp_2, %4 ]
  ret i16 %p_0
}

define internal fastcc zeroext i8 @bno055_uart_uart_read_reg(i8 zeroext %reg_addr) nounwind uwtable {
  %reg_addr_read = call i8 @_ssdm_op_Read.ap_auto.i8(i8 %reg_addr) nounwind
  call fastcc void @bno055_uart_uart_send_byte(i8 zeroext -86)
  call void (...)* @_ssdm_op_Wait(i32 1)
  call fastcc void @bno055_uart_uart_send_byte(i8 zeroext 1)
  call void (...)* @_ssdm_op_Wait(i32 1)
  call fastcc void @bno055_uart_uart_send_byte(i8 zeroext %reg_addr_read)
  call void (...)* @_ssdm_op_Wait(i32 1)
  call fastcc void @bno055_uart_uart_send_byte(i8 zeroext 1)
  call void (...)* @_ssdm_op_Wait(i32 1)
  call fastcc void @bno055_uart_wait_tmr(i28 1) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  %buf = call fastcc zeroext i8 @bno055_uart_uart_receive_byte()
  call void (...)* @_ssdm_op_Wait(i32 1)
  %tmp = icmp eq i8 %buf, -69
  br i1 %tmp, label %1, label %2

; <label>:1                                       ; preds = %0
  call void (...)* @_ssdm_op_Wait(i32 1)
  %empty = call fastcc zeroext i8 @bno055_uart_uart_receive_byte()
  call void (...)* @_ssdm_op_Wait(i32 1)
  %buf_2 = call fastcc zeroext i8 @bno055_uart_uart_receive_byte()
  br label %._crit_edge

; <label>:2                                       ; preds = %0
  %tmp_s = icmp eq i8 %buf, -18
  br i1 %tmp_s, label %3, label %._crit_edge

; <label>:3                                       ; preds = %2
  call void (...)* @_ssdm_op_Wait(i32 1)
  %buf_3 = call fastcc zeroext i8 @bno055_uart_uart_receive_byte()
  br label %._crit_edge

._crit_edge:                                      ; preds = %3, %2, %1
  %p_0 = phi i8 [ %buf_2, %1 ], [ %buf_3, %3 ], [ %buf, %2 ]
  ret i8 %p_0
}

define internal fastcc zeroext i1 @bno055_uart_read_uart_rx() nounwind uwtable {
  %uart_rx_read = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @uart_rx) nounwind
  ret i1 %uart_rx_read
}

declare i8 @llvm.part.select.i8(i8, i32, i32) nounwind readnone

declare i20 @llvm.part.select.i20(i20, i32, i32) nounwind readnone

declare i16 @llvm.part.select.i16(i16, i32, i32) nounwind readnone

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

define void @bno055_uart() noreturn nounwind uwtable {
  call void (...)* @_ssdm_op_SpecTopModule()
  %dummy_tmr_out_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @dummy_tmr_out) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @dummy_tmr_out, [8 x i8]* @p_str6, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind
  %uart_rx_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @uart_rx) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @uart_rx, [8 x i8]* @p_str6, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind
  %uart_tx_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @uart_tx) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @uart_tx, [8 x i8]* @p_str6, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind
  %mem_addr_load = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @mem_addr) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i8* @mem_addr, [8 x i8]* @p_str6, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind
  %mem_din_load = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @mem_din) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i8* @mem_din, [8 x i8]* @p_str6, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind
  %mem_dout_load = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @mem_dout) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i8* @mem_dout, [8 x i8]* @p_str6, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind
  %mem_wreq_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @mem_wreq) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @mem_wreq, [8 x i8]* @p_str6, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind
  %mem_wack_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @mem_wack) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @mem_wack, [8 x i8]* @p_str6, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind
  %mem_rreq_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @mem_rreq) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @mem_rreq, [8 x i8]* @p_str6, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind
  %mem_rack_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @mem_rack) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @mem_rack, [8 x i8]* @p_str6, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind
  call fastcc void @bno055_uart_write_mem(i8 zeroext 21, i8 zeroext 0)
  call fastcc void @bno055_uart_write_mem(i8 zeroext 22, i8 zeroext 0)
  call void (...)* @_ssdm_op_Wait(i32 1)
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @uart_tx, i1 true) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  call fastcc void @bno055_uart_uart_write_reg(i7 zeroext 7, i8 zeroext 0) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  call fastcc void @bno055_uart_uart_write_reg(i7 zeroext 63, i8 zeroext -64) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  call fastcc void @bno055_uart_wait_tmr(i28 100000000) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  call fastcc void @bno055_uart_uart_write_reg(i7 zeroext 7, i8 zeroext 0) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  %dt = call fastcc zeroext i8 @bno055_uart_uart_read_reg(i8 zeroext 62)
  call void (...)* @_ssdm_op_Wait(i32 1)
  %tmp = call i6 @_ssdm_op_PartSelect.i6.i8.i32.i32(i8 %dt, i32 2, i32 7)
  %tmp_s = call i8 @_ssdm_op_BitConcatenate.i8.i6.i2(i6 %tmp, i2 0)
  call fastcc void @bno055_uart_uart_write_reg(i7 zeroext 62, i8 zeroext %tmp_s) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  call fastcc void @bno055_uart_wait_tmr(i28 10000000) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  %dt_1 = call fastcc zeroext i8 @bno055_uart_uart_read_reg(i8 zeroext 61)
  call void (...)* @_ssdm_op_Wait(i32 1)
  %tmp_9 = call i4 @_ssdm_op_PartSelect.i4.i8.i32.i32(i8 %dt_1, i32 4, i32 7)
  %tmp_3 = call i8 @_ssdm_op_BitConcatenate.i8.i4.i4(i4 %tmp_9, i4 0)
  %tmp_4 = or i8 %tmp_3, 12
  call fastcc void @bno055_uart_uart_write_reg(i7 zeroext 61, i8 zeroext %tmp_4) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  call fastcc void @bno055_uart_wait_tmr(i28 100000000) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  %dt_2 = call fastcc zeroext i8 @bno055_uart_uart_read_reg(i8 zeroext 59)
  call void (...)* @_ssdm_op_Wait(i32 1)
  %tmp_5 = or i8 %dt_2, -128
  call fastcc void @bno055_uart_uart_write_reg(i7 zeroext 59, i8 zeroext %tmp_5) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  call fastcc void @bno055_uart_wait_tmr(i28 10000000) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  br label %1

; <label>:1                                       ; preds = %1, %0
  %dt_3 = call fastcc zeroext i8 @bno055_uart_uart_read_reg(i8 zeroext 0)
  call void (...)* @_ssdm_op_Wait(i32 1)
  %tmp_7 = call i4 @_ssdm_op_PartSelect.i4.i8.i32.i32(i8 %dt_3, i32 4, i32 7)
  %tmp_8 = call fastcc zeroext i7 @bno055_uart_bin2char(i4 zeroext %tmp_7) nounwind
  %p_trunc_ext = zext i7 %tmp_8 to i8
  call fastcc void @bno055_uart_write_mem(i8 zeroext 0, i8 zeroext %p_trunc_ext)
  %tmp_10 = trunc i8 %dt_3 to i4
  %tmp_1 = call fastcc zeroext i7 @bno055_uart_bin2char(i4 zeroext %tmp_10) nounwind
  %p_trunc2_ext = zext i7 %tmp_1 to i8
  call fastcc void @bno055_uart_write_mem(i8 zeroext 1, i8 zeroext %p_trunc2_ext)
  call void (...)* @_ssdm_op_Wait(i32 1)
  %e_tmp = call fastcc zeroext i16 @bno055_uart_uart_read_reg16()
  call void (...)* @_ssdm_op_Wait(i32 1)
  %tmp_11 = trunc i16 %e_tmp to i8
  %e_tmp_1 = call i8 @_ssdm_op_PartSelect.i8.i16.i32.i32(i16 %e_tmp, i32 8, i32 15)
  %e = call i16 @_ssdm_op_BitConcatenate.i16.i8.i8(i8 %tmp_11, i8 %e_tmp_1)
  %e_cast = zext i16 %e to i20
  %e_1 = mul i20 100, %e_cast
  call void (...)* @_ssdm_op_Wait(i32 1)
  call fastcc void @bno055_uart_write_mem(i8 zeroext -123, i8 zeroext 0)
  %tmp_2 = call i8 @_ssdm_op_PartSelect.i8.i20.i32.i32(i20 %e_1, i32 12, i32 19)
  call fastcc void @bno055_uart_write_mem(i8 zeroext -125, i8 zeroext %tmp_2)
  %tmp_6 = call i8 @_ssdm_op_PartSelect.i8.i20.i32.i32(i20 %e_1, i32 4, i32 11)
  call fastcc void @bno055_uart_write_mem(i8 zeroext -124, i8 zeroext %tmp_6)
  call fastcc void @bno055_uart_write_mem(i8 zeroext -123, i8 zeroext 1)
  call void (...)* @_ssdm_op_Wait(i32 1)
  call void (...)* @_ssdm_op_Wait(i32 1)
  call fastcc void @bno055_uart_wait_tmr(i28 1000000) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  br label %1
}

define internal fastcc zeroext i7 @bno055_uart_bin2char(i4 zeroext %val) readnone {
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

define weak i7 @_ssdm_op_Read.ap_auto.i7(i7) {
entry:
  ret i7 %0
}

define weak i4 @_ssdm_op_Read.ap_auto.i4(i4) {
entry:
  ret i4 %0
}

define weak i28 @_ssdm_op_Read.ap_auto.i28(i28) {
entry:
  ret i28 %0
}

define weak i8 @_ssdm_op_PartSelect.i8.i20.i32.i32(i20, i32, i32) nounwind readnone {
entry:
  %empty = call i20 @llvm.part.select.i20(i20 %0, i32 %1, i32 %2)
  %empty_10 = trunc i20 %empty to i8
  ret i8 %empty_10
}

define weak i8 @_ssdm_op_PartSelect.i8.i16.i32.i32(i16, i32, i32) nounwind readnone {
entry:
  %empty = call i16 @llvm.part.select.i16(i16 %0, i32 %1, i32 %2)
  %empty_11 = trunc i16 %empty to i8
  ret i8 %empty_11
}

define weak i7 @_ssdm_op_PartSelect.i7.i8.i32.i32(i8, i32, i32) nounwind readnone {
entry:
  %empty = call i8 @llvm.part.select.i8(i8 %0, i32 %1, i32 %2)
  %empty_12 = trunc i8 %empty to i7
  ret i7 %empty_12
}

define weak i6 @_ssdm_op_PartSelect.i6.i8.i32.i32(i8, i32, i32) nounwind readnone {
entry:
  %empty = call i8 @llvm.part.select.i8(i8 %0, i32 %1, i32 %2)
  %empty_13 = trunc i8 %empty to i6
  ret i6 %empty_13
}

define weak i4 @_ssdm_op_PartSelect.i4.i8.i32.i32(i8, i32, i32) nounwind readnone {
entry:
  %empty = call i8 @llvm.part.select.i8(i8 %0, i32 %1, i32 %2)
  %empty_14 = trunc i8 %empty to i4
  ret i4 %empty_14
}

declare i1 @_ssdm_op_PartSelect.i1.i8.i32.i32(i8, i32, i32) nounwind readnone

define weak i1 @_ssdm_op_BitSelect.i1.i8.i32(i8, i32) nounwind readnone {
entry:
  %empty = trunc i32 %1 to i8
  %empty_15 = shl i8 1, %empty
  %empty_16 = and i8 %0, %empty_15
  %empty_17 = icmp ne i8 %empty_16, 0
  ret i1 %empty_17
}

define weak i8 @_ssdm_op_BitConcatenate.i8.i6.i2(i6, i2) nounwind readnone {
entry:
  %empty = zext i6 %0 to i8
  %empty_18 = zext i2 %1 to i8
  %empty_19 = shl i8 %empty, 2
  %empty_20 = or i8 %empty_19, %empty_18
  ret i8 %empty_20
}

define weak i8 @_ssdm_op_BitConcatenate.i8.i4.i4(i4, i4) nounwind readnone {
entry:
  %empty = zext i4 %0 to i8
  %empty_21 = zext i4 %1 to i8
  %empty_22 = shl i8 %empty, 4
  %empty_23 = or i8 %empty_22, %empty_21
  ret i8 %empty_23
}

define weak i8 @_ssdm_op_BitConcatenate.i8.i1.i7(i1, i7) nounwind readnone {
entry:
  %empty = zext i1 %0 to i8
  %empty_24 = zext i7 %1 to i8
  %empty_25 = shl i8 %empty, 7
  %empty_26 = or i8 %empty_25, %empty_24
  ret i8 %empty_26
}

define weak i16 @_ssdm_op_BitConcatenate.i16.i8.i8(i8, i8) nounwind readnone {
entry:
  %empty = zext i8 %0 to i16
  %empty_27 = zext i8 %1 to i16
  %empty_28 = shl i16 %empty, 8
  %empty_29 = or i16 %empty_28, %empty_27
  ret i16 %empty_29
}

!hls.encrypted.func = !{}
!llvm.map.gv = !{!0, !7, !12, !17, !22, !27, !32, !37, !42, !47}

!0 = metadata !{metadata !1, i1* @uart_tx}
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0, i32 0, metadata !3}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"uart_tx", metadata !5, metadata !"uint1", i32 0, i32 0}
!5 = metadata !{metadata !6}
!6 = metadata !{i32 0, i32 0, i32 1}
!7 = metadata !{metadata !8, i1* @uart_rx}
!8 = metadata !{metadata !9}
!9 = metadata !{i32 0, i32 0, metadata !10}
!10 = metadata !{metadata !11}
!11 = metadata !{metadata !"uart_rx", metadata !5, metadata !"uint1", i32 0, i32 0}
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
!47 = metadata !{metadata !48, i1* @dummy_tmr_out}
!48 = metadata !{metadata !49}
!49 = metadata !{i32 0, i32 0, metadata !50}
!50 = metadata !{metadata !51}
!51 = metadata !{metadata !"dummy_tmr_out", metadata !5, metadata !"uint1", i32 0, i32 0}
