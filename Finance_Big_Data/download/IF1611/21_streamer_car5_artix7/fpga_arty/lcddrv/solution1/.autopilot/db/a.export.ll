; ModuleID = 'D:/21_streamer_car5_artix7/fpga_arty/lcddrv/solution1/.autopilot/db/a.o.2.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-w64-mingw32"

@rs = global i1 false, align 1
@mem_req = global i1 false, align 1
@mem_din = common global i8 0, align 1
@mem_addr = global i5 0, align 1
@mem_ack = common global i1 false, align 1
@ind = global i1 false, align 1
@en = global i1 false, align 1
@dummy_tmr_out = global i1 false, align 1
@data = global i4 0, align 1
@p_str2 = private unnamed_addr constant [12 x i8] c"hls_label_0\00", align 1
@p_str1 = private unnamed_addr constant [8 x i8] c"ap_none\00", align 1
@p_str = private unnamed_addr constant [1 x i8] zeroinitializer, align 1

define internal fastcc void @lcddrv_wait_tmr(i25 %tmr) {
  %tmr_read = call i25 @_ssdm_op_Read.ap_auto.i25(i25 %tmr)
  call void (...)* @_ssdm_op_Wait(i32 1)
  br label %1

; <label>:1                                       ; preds = %2, %0
  %t = phi i24 [ 0, %0 ], [ %t_1, %2 ]
  %t_cast = zext i24 %t to i25
  %exitcond = icmp eq i25 %t_cast, %tmr_read
  %empty = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 1000, i64 10000000, i64 0) nounwind
  %t_1 = add i24 %t, 1
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

define internal fastcc zeroext i8 @lcddrv_read_mem(i5 zeroext %addr) nounwind uwtable {
  %addr_read = call i5 @_ssdm_op_Read.ap_auto.i5(i5 %addr) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  call void @_ssdm_op_Write.ap_none.volatile.i5P(i5* @mem_addr, i5 %addr_read) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @mem_req, i1 true) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  br label %._crit_edge

._crit_edge:                                      ; preds = %._crit_edge, %0
  call void @_ssdm_op_Write.ap_none.volatile.i5P(i5* @mem_addr, i5 %addr_read) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @mem_req, i1 true) nounwind
  %dt = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @mem_din) nounwind
  %mem_ack_read = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @mem_ack) nounwind
  br i1 %mem_ack_read, label %1, label %._crit_edge

; <label>:1                                       ; preds = %._crit_edge
  call void (...)* @_ssdm_op_Wait(i32 1)
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @mem_req, i1 false) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  ret i8 %dt
}

declare i8 @llvm.part.select.i8(i8, i32, i32) nounwind readnone

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

define void @lcddrv() noreturn nounwind uwtable {
  call void (...)* @_ssdm_op_SpecTopModule()
  %dummy_tmr_out_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @dummy_tmr_out) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @dummy_tmr_out, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind
  %rs_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @rs) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @rs, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind
  %en_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @en) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @en, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind
  %data_load = call i4 @_ssdm_op_Read.ap_none.volatile.i4P(i4* @data) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i4* @data, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind
  %ind_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @ind) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @ind, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind
  %mem_addr_load = call i5 @_ssdm_op_Read.ap_none.volatile.i5P(i5* @mem_addr) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i5* @mem_addr, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind
  %mem_din_load = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @mem_din) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i8* @mem_din, [8 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind
  %mem_req_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @mem_req) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @mem_req, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind
  %mem_ack_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @mem_ack) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @mem_ack, [8 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind
  call fastcc void @lcddrv_init_lcd()
  br label %1

; <label>:1                                       ; preds = %7, %0
  %loop_begin = call i32 (...)* @_ssdm_op_SpecLoopBegin() nounwind
  %tmp = call i32 (...)* @_ssdm_op_SpecRegionBegin([12 x i8]* @p_str2) nounwind
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext false, i4 zeroext -8)
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext false, i4 zeroext 0)
  br label %2

; <label>:2                                       ; preds = %3, %1
  %pos = phi i5 [ 0, %1 ], [ %pos_2, %3 ]
  %exitcond1 = icmp eq i5 %pos, -16
  %empty = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 16, i64 16, i64 16) nounwind
  %pos_2 = add i5 %pos, 1
  br i1 %exitcond1, label %4, label %3

; <label>:3                                       ; preds = %2
  %dt = call fastcc zeroext i8 @lcddrv_read_mem(i5 zeroext %pos)
  %tmp_1 = call i4 @_ssdm_op_PartSelect.i4.i8.i32.i32(i8 %dt, i32 4, i32 7)
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext true, i4 zeroext %tmp_1)
  %tmp_2 = trunc i8 %dt to i4
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext true, i4 zeroext %tmp_2)
  br label %2

; <label>:4                                       ; preds = %2
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext false, i4 zeroext -4)
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext false, i4 zeroext 0)
  br label %5

; <label>:5                                       ; preds = %6, %4
  %pos_1 = phi i6 [ 16, %4 ], [ %pos_3, %6 ]
  %exitcond = icmp eq i6 %pos_1, -32
  %empty_4 = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 16, i64 16, i64 16) nounwind
  br i1 %exitcond, label %7, label %6

; <label>:6                                       ; preds = %5
  %tmp_3 = trunc i6 %pos_1 to i5
  %dt_1 = call fastcc zeroext i8 @lcddrv_read_mem(i5 zeroext %tmp_3)
  %tmp_5 = call i4 @_ssdm_op_PartSelect.i4.i8.i32.i32(i8 %dt_1, i32 4, i32 7)
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext true, i4 zeroext %tmp_5)
  %tmp_4 = trunc i8 %dt_1 to i4
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext true, i4 zeroext %tmp_4)
  %pos_3 = add i6 1, %pos_1
  br label %5

; <label>:7                                       ; preds = %5
  call void (...)* @_ssdm_op_Wait(i32 1)
  call fastcc void @lcddrv_wait_tmr(i25 10000000) nounwind
  %empty_5 = call i32 (...)* @_ssdm_op_SpecRegionEnd([12 x i8]* @p_str2, i32 %tmp) nounwind
  br label %1
}

define internal fastcc void @lcddrv_lcd_send_cmd(i1 zeroext %mode, i4 zeroext %wd) nounwind uwtable {
  %wd_read = call i4 @_ssdm_op_Read.ap_auto.i4(i4 %wd) nounwind
  %mode_read = call i1 @_ssdm_op_Read.ap_auto.i1(i1 %mode) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @en, i1 false) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  call fastcc void @lcddrv_wait_tmr(i25 1000) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @en, i1 false) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @rs, i1 %mode_read) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i4P(i4* @data, i4 %wd_read) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  call fastcc void @lcddrv_wait_tmr(i25 1000) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @en, i1 true) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @rs, i1 %mode_read) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i4P(i4* @data, i4 %wd_read) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  call fastcc void @lcddrv_wait_tmr(i25 1000) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @en, i1 false) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @rs, i1 %mode_read) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i4P(i4* @data, i4 %wd_read) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  call fastcc void @lcddrv_wait_tmr(i25 1000) nounwind
  ret void
}

define internal fastcc void @lcddrv_init_lcd() nounwind uwtable {
  call fastcc void @lcddrv_wait_tmr(i25 2000000) nounwind
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext false, i4 zeroext 3)
  call fastcc void @lcddrv_wait_tmr(i25 500000) nounwind
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext false, i4 zeroext 3)
  call fastcc void @lcddrv_wait_tmr(i25 50000) nounwind
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext false, i4 zeroext 3)
  call fastcc void @lcddrv_wait_tmr(i25 50000) nounwind
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext false, i4 zeroext 2)
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext false, i4 zeroext 2)
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext false, i4 zeroext -8)
  call fastcc void @lcddrv_wait_tmr(i25 10000) nounwind
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext false, i4 zeroext 0)
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext false, i4 zeroext -4)
  call fastcc void @lcddrv_wait_tmr(i25 10000) nounwind
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext false, i4 zeroext 0)
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext false, i4 zeroext 1)
  call fastcc void @lcddrv_wait_tmr(i25 200000) nounwind
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext false, i4 zeroext 0)
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext false, i4 zeroext 2)
  call fastcc void @lcddrv_wait_tmr(i25 10000) nounwind
  ret void
}

define weak void @_ssdm_op_Write.ap_none.volatile.i5P(i5*, i5) {
entry:
  store i5 %1, i5* %0
  ret void
}

define weak void @_ssdm_op_Write.ap_none.volatile.i4P(i4*, i4) {
entry:
  store i4 %1, i4* %0
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

define weak i5 @_ssdm_op_Read.ap_none.volatile.i5P(i5*) {
entry:
  %empty = load i5* %0
  ret i5 %empty
}

define weak i4 @_ssdm_op_Read.ap_none.volatile.i4P(i4*) {
entry:
  %empty = load i4* %0
  ret i4 %empty
}

define weak i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1*) {
entry:
  %empty = load i1* %0
  ret i1 %empty
}

define weak i5 @_ssdm_op_Read.ap_auto.i5(i5) {
entry:
  ret i5 %0
}

define weak i4 @_ssdm_op_Read.ap_auto.i4(i4) {
entry:
  ret i4 %0
}

define weak i25 @_ssdm_op_Read.ap_auto.i25(i25) {
entry:
  ret i25 %0
}

define weak i1 @_ssdm_op_Read.ap_auto.i1(i1) {
entry:
  ret i1 %0
}

declare i5 @_ssdm_op_PartSelect.i5.i6.i32.i32(i6, i32, i32) nounwind readnone

define weak i4 @_ssdm_op_PartSelect.i4.i8.i32.i32(i8, i32, i32) nounwind readnone {
entry:
  %empty = call i8 @llvm.part.select.i8(i8 %0, i32 %1, i32 %2)
  %empty_6 = trunc i8 %empty to i4
  ret i4 %empty_6
}

!hls.encrypted.func = !{}
!llvm.map.gv = !{!0, !7, !12, !17, !22, !27, !32, !37, !42}

!0 = metadata !{metadata !1, i1* @rs}
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0, i32 0, metadata !3}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"rs", metadata !5, metadata !"uint1", i32 0, i32 0}
!5 = metadata !{metadata !6}
!6 = metadata !{i32 0, i32 0, i32 1}
!7 = metadata !{metadata !8, i1* @mem_req}
!8 = metadata !{metadata !9}
!9 = metadata !{i32 0, i32 0, metadata !10}
!10 = metadata !{metadata !11}
!11 = metadata !{metadata !"mem_req", metadata !5, metadata !"uint1", i32 0, i32 0}
!12 = metadata !{metadata !13, i8* @mem_din}
!13 = metadata !{metadata !14}
!14 = metadata !{i32 0, i32 7, metadata !15}
!15 = metadata !{metadata !16}
!16 = metadata !{metadata !"mem_din", metadata !5, metadata !"uint8", i32 0, i32 7}
!17 = metadata !{metadata !18, i5* @mem_addr}
!18 = metadata !{metadata !19}
!19 = metadata !{i32 0, i32 4, metadata !20}
!20 = metadata !{metadata !21}
!21 = metadata !{metadata !"mem_addr", metadata !5, metadata !"uint5", i32 0, i32 4}
!22 = metadata !{metadata !23, i1* @mem_ack}
!23 = metadata !{metadata !24}
!24 = metadata !{i32 0, i32 0, metadata !25}
!25 = metadata !{metadata !26}
!26 = metadata !{metadata !"mem_ack", metadata !5, metadata !"uint1", i32 0, i32 0}
!27 = metadata !{metadata !28, i1* @ind}
!28 = metadata !{metadata !29}
!29 = metadata !{i32 0, i32 0, metadata !30}
!30 = metadata !{metadata !31}
!31 = metadata !{metadata !"ind", metadata !5, metadata !"uint1", i32 0, i32 0}
!32 = metadata !{metadata !33, i1* @en}
!33 = metadata !{metadata !34}
!34 = metadata !{i32 0, i32 0, metadata !35}
!35 = metadata !{metadata !36}
!36 = metadata !{metadata !"en", metadata !5, metadata !"uint1", i32 0, i32 0}
!37 = metadata !{metadata !38, i1* @dummy_tmr_out}
!38 = metadata !{metadata !39}
!39 = metadata !{i32 0, i32 0, metadata !40}
!40 = metadata !{metadata !41}
!41 = metadata !{metadata !"dummy_tmr_out", metadata !5, metadata !"uint1", i32 0, i32 0}
!42 = metadata !{metadata !43, i4* @data}
!43 = metadata !{metadata !44}
!44 = metadata !{i32 0, i32 3, metadata !45}
!45 = metadata !{metadata !46}
!46 = metadata !{metadata !"data", metadata !5, metadata !"uint4", i32 0, i32 3}
