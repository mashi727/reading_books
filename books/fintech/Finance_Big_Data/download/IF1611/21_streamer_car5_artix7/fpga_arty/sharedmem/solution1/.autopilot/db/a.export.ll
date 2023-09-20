; ModuleID = 'D:/21_streamer_car5_artix7/fpga_arty/sharedmem/solution1/.autopilot/db/a.o.2.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-w64-mingw32"

@w_req3 = common global i1 false, align 1
@w_req2 = common global i1 false, align 1
@w_req1 = common global i1 false, align 1
@w_req0 = common global i1 false, align 1
@w_ack3 = global i1 false, align 1
@w_ack2 = global i1 false, align 1
@w_ack1 = global i1 false, align 1
@w_ack0 = global i1 false, align 1
@r_req3 = common global i1 false, align 1
@r_req2 = common global i1 false, align 1
@r_req1 = common global i1 false, align 1
@r_req0 = common global i1 false, align 1
@r_ack3 = global i1 false, align 1
@r_ack2 = global i1 false, align 1
@r_ack1 = global i1 false, align 1
@r_ack0 = global i1 false, align 1
@dout3 = global i8 0, align 1
@dout2 = global i8 0, align 1
@dout1 = global i8 0, align 1
@dout0 = global i8 0, align 1
@din3 = common global i8 0, align 1
@din2 = common global i8 0, align 1
@din1 = common global i8 0, align 1
@din0 = common global i8 0, align 1
@addr3 = global i8 0, align 1
@addr2 = global i8 0, align 1
@addr1 = global i8 0, align 1
@addr0 = global i8 0, align 1
@p_str1 = private unnamed_addr constant [1 x i8] zeroinitializer, align 1
@p_str = private unnamed_addr constant [8 x i8] c"ap_none\00", align 1

define void @sharedmem() noreturn nounwind uwtable {
  %mem = alloca [256 x i8], align 16
  call void (...)* @_ssdm_op_SpecTopModule()
  %addr0_load = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @addr0) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i8* @addr0, [8 x i8]* @p_str, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str1, [1 x i8]* @p_str1, [1 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1) nounwind
  %din0_load = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @din0) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i8* @din0, [8 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1, [1 x i8]* @p_str1, [1 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1) nounwind
  %dout0_load = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @dout0) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i8* @dout0, [8 x i8]* @p_str, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str1, [1 x i8]* @p_str1, [1 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1) nounwind
  %r_req0_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @r_req0) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @r_req0, [8 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1, [1 x i8]* @p_str1, [1 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1) nounwind
  %r_ack0_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @r_ack0) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @r_ack0, [8 x i8]* @p_str, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str1, [1 x i8]* @p_str1, [1 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1) nounwind
  %w_req0_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @w_req0) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @w_req0, [8 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1, [1 x i8]* @p_str1, [1 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1) nounwind
  %w_ack0_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @w_ack0) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @w_ack0, [8 x i8]* @p_str, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str1, [1 x i8]* @p_str1, [1 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1) nounwind
  %addr1_load = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @addr1) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i8* @addr1, [8 x i8]* @p_str, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str1, [1 x i8]* @p_str1, [1 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1) nounwind
  %din1_load = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @din1) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i8* @din1, [8 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1, [1 x i8]* @p_str1, [1 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1) nounwind
  %dout1_load = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @dout1) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i8* @dout1, [8 x i8]* @p_str, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str1, [1 x i8]* @p_str1, [1 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1) nounwind
  %r_req1_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @r_req1) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @r_req1, [8 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1, [1 x i8]* @p_str1, [1 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1) nounwind
  %r_ack1_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @r_ack1) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @r_ack1, [8 x i8]* @p_str, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str1, [1 x i8]* @p_str1, [1 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1) nounwind
  %w_req1_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @w_req1) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @w_req1, [8 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1, [1 x i8]* @p_str1, [1 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1) nounwind
  %w_ack1_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @w_ack1) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @w_ack1, [8 x i8]* @p_str, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str1, [1 x i8]* @p_str1, [1 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1) nounwind
  %addr2_load = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @addr2) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i8* @addr2, [8 x i8]* @p_str, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str1, [1 x i8]* @p_str1, [1 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1) nounwind
  %din2_load = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @din2) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i8* @din2, [8 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1, [1 x i8]* @p_str1, [1 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1) nounwind
  %dout2_load = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @dout2) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i8* @dout2, [8 x i8]* @p_str, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str1, [1 x i8]* @p_str1, [1 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1) nounwind
  %r_req2_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @r_req2) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @r_req2, [8 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1, [1 x i8]* @p_str1, [1 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1) nounwind
  %r_ack2_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @r_ack2) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @r_ack2, [8 x i8]* @p_str, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str1, [1 x i8]* @p_str1, [1 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1) nounwind
  %w_req2_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @w_req2) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @w_req2, [8 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1, [1 x i8]* @p_str1, [1 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1) nounwind
  %w_ack2_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @w_ack2) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @w_ack2, [8 x i8]* @p_str, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str1, [1 x i8]* @p_str1, [1 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1) nounwind
  %addr3_load = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @addr3) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i8* @addr3, [8 x i8]* @p_str, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str1, [1 x i8]* @p_str1, [1 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1) nounwind
  %din3_load = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @din3) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i8* @din3, [8 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1, [1 x i8]* @p_str1, [1 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1) nounwind
  %dout3_load = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @dout3) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i8* @dout3, [8 x i8]* @p_str, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str1, [1 x i8]* @p_str1, [1 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1) nounwind
  %r_req3_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @r_req3) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @r_req3, [8 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1, [1 x i8]* @p_str1, [1 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1) nounwind
  %r_ack3_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @r_ack3) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @r_ack3, [8 x i8]* @p_str, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str1, [1 x i8]* @p_str1, [1 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1) nounwind
  %w_req3_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @w_req3) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @w_req3, [8 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1, [1 x i8]* @p_str1, [1 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1) nounwind
  %w_ack3_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @w_ack3) nounwind
  call void (...)* @_ssdm_op_SpecInterface(i1* @w_ack3, [8 x i8]* @p_str, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str1, [1 x i8]* @p_str1, [1 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str1) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @r_ack0, i1 false) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @w_ack0, i1 false) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @r_ack1, i1 false) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @w_ack1, i1 false) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @r_ack2, i1 false) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @w_ack2, i1 false) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @r_ack3, i1 false) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @w_ack3, i1 false) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  br label %1

; <label>:1                                       ; preds = %2, %0
  %addr = phi i9 [ 0, %0 ], [ %addr_1, %2 ]
  %exitcond = icmp eq i9 %addr, -256
  %empty = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 256, i64 256, i64 256) nounwind
  %addr_1 = add i9 %addr, 1
  br i1 %exitcond, label %.preheader, label %2

; <label>:2                                       ; preds = %1
  %tmp_s = zext i9 %addr to i64
  %mem_addr = getelementptr inbounds [256 x i8]* %mem, i64 0, i64 %tmp_s
  store i8 0, i8* %mem_addr, align 1
  br label %1

.preheader:                                       ; preds = %._crit_edge, %._crit_edge2, %._crit_edge4, %10, %._crit_edge6, %21, %32, %43, %54, %55, %1
  %ch = phi i3 [ 0, %1 ], [ 0, %55 ], [ 0, %54 ], [ 0, %43 ], [ 0, %32 ], [ 0, %21 ], [ 1, %._crit_edge ], [ 2, %._crit_edge2 ], [ 3, %._crit_edge4 ], [ -4, %._crit_edge6 ], [ 0, %10 ]
  %loop_begin = call i32 (...)* @_ssdm_op_SpecLoopBegin() nounwind
  switch i3 %ch, label %55 [
    i3 0, label %3
    i3 1, label %11
    i3 2, label %22
    i3 3, label %33
    i3 -4, label %44
  ]

; <label>:3                                       ; preds = %.preheader
  %r_req0_read = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @r_req0) nounwind
  br i1 %r_req0_read, label %._crit_edge, label %4

; <label>:4                                       ; preds = %3
  %w_req0_read = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @w_req0) nounwind
  br i1 %w_req0_read, label %._crit_edge, label %5

._crit_edge:                                      ; preds = %4, %3
  br label %.preheader

; <label>:5                                       ; preds = %4
  %r_req1_read_1 = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @r_req1) nounwind
  br i1 %r_req1_read_1, label %._crit_edge2, label %6

; <label>:6                                       ; preds = %5
  %w_req1_read_1 = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @w_req1) nounwind
  br i1 %w_req1_read_1, label %._crit_edge2, label %7

._crit_edge2:                                     ; preds = %6, %5
  br label %.preheader

; <label>:7                                       ; preds = %6
  %r_req2_read_2 = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @r_req2) nounwind
  br i1 %r_req2_read_2, label %._crit_edge4, label %8

; <label>:8                                       ; preds = %7
  %w_req2_read_2 = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @w_req2) nounwind
  br i1 %w_req2_read_2, label %._crit_edge4, label %9

._crit_edge4:                                     ; preds = %8, %7
  br label %.preheader

; <label>:9                                       ; preds = %8
  %r_req3_read_2 = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @r_req3) nounwind
  br i1 %r_req3_read_2, label %._crit_edge6, label %10

; <label>:10                                      ; preds = %9
  %w_req3_read_2 = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @w_req3) nounwind
  br i1 %w_req3_read_2, label %._crit_edge6, label %.preheader

._crit_edge6:                                     ; preds = %10, %9
  br label %.preheader

; <label>:11                                      ; preds = %.preheader
  %r_req0_read_1 = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @r_req0) nounwind
  br i1 %r_req0_read_1, label %12, label %16

; <label>:12                                      ; preds = %11
  call void (...)* @_ssdm_op_Wait(i32 1)
  %addr0_read = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @addr0) nounwind
  %tmp_2 = zext i8 %addr0_read to i64
  %mem_addr_1 = getelementptr inbounds [256 x i8]* %mem, i64 0, i64 %tmp_2
  %mem_load = load i8* %mem_addr_1, align 1
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @dout0, i8 %mem_load) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  br label %13

; <label>:13                                      ; preds = %14, %12
  %r_req0_read_2 = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @r_req0) nounwind
  br i1 %r_req0_read_2, label %14, label %15

; <label>:14                                      ; preds = %13
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @r_ack0, i1 true) nounwind
  br label %13

; <label>:15                                      ; preds = %13
  call void (...)* @_ssdm_op_Wait(i32 1)
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @r_ack0, i1 false) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  br label %21

; <label>:16                                      ; preds = %11
  %w_req0_read_1 = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @w_req0) nounwind
  br i1 %w_req0_read_1, label %17, label %._crit_edge8

; <label>:17                                      ; preds = %16
  call void (...)* @_ssdm_op_Wait(i32 1)
  %din0_read = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @din0) nounwind
  %addr0_read_1 = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @addr0) nounwind
  %tmp_6 = zext i8 %addr0_read_1 to i64
  %mem_addr_5 = getelementptr inbounds [256 x i8]* %mem, i64 0, i64 %tmp_6
  store i8 %din0_read, i8* %mem_addr_5, align 1
  call void (...)* @_ssdm_op_Wait(i32 1)
  br label %18

; <label>:18                                      ; preds = %19, %17
  %w_req0_read_2 = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @w_req0) nounwind
  br i1 %w_req0_read_2, label %19, label %20

; <label>:19                                      ; preds = %18
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @w_ack0, i1 true) nounwind
  br label %18

; <label>:20                                      ; preds = %18
  call void (...)* @_ssdm_op_Wait(i32 1)
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @w_ack0, i1 false) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  br label %._crit_edge8

._crit_edge8:                                     ; preds = %20, %16
  br label %21

; <label>:21                                      ; preds = %._crit_edge8, %15
  br label %.preheader

; <label>:22                                      ; preds = %.preheader
  %r_req1_read = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @r_req1) nounwind
  br i1 %r_req1_read, label %23, label %27

; <label>:23                                      ; preds = %22
  call void (...)* @_ssdm_op_Wait(i32 1)
  %addr1_read = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @addr1) nounwind
  %tmp_3 = zext i8 %addr1_read to i64
  %mem_addr_2 = getelementptr inbounds [256 x i8]* %mem, i64 0, i64 %tmp_3
  %mem_load_1 = load i8* %mem_addr_2, align 1
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @dout1, i8 %mem_load_1) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  br label %24

; <label>:24                                      ; preds = %25, %23
  %r_req1_read_2 = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @r_req1) nounwind
  br i1 %r_req1_read_2, label %25, label %26

; <label>:25                                      ; preds = %24
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @r_ack1, i1 true) nounwind
  br label %24

; <label>:26                                      ; preds = %24
  call void (...)* @_ssdm_op_Wait(i32 1)
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @r_ack1, i1 false) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  br label %32

; <label>:27                                      ; preds = %22
  %w_req1_read = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @w_req1) nounwind
  br i1 %w_req1_read, label %28, label %._crit_edge9

; <label>:28                                      ; preds = %27
  call void (...)* @_ssdm_op_Wait(i32 1)
  %din1_read = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @din1) nounwind
  %addr1_read_1 = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @addr1) nounwind
  %tmp_7 = zext i8 %addr1_read_1 to i64
  %mem_addr_6 = getelementptr inbounds [256 x i8]* %mem, i64 0, i64 %tmp_7
  store i8 %din1_read, i8* %mem_addr_6, align 1
  call void (...)* @_ssdm_op_Wait(i32 1)
  br label %29

; <label>:29                                      ; preds = %30, %28
  %w_req1_read_2 = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @w_req1) nounwind
  br i1 %w_req1_read_2, label %30, label %31

; <label>:30                                      ; preds = %29
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @w_ack1, i1 true) nounwind
  br label %29

; <label>:31                                      ; preds = %29
  call void (...)* @_ssdm_op_Wait(i32 1)
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @w_ack1, i1 false) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  br label %._crit_edge9

._crit_edge9:                                     ; preds = %31, %27
  br label %32

; <label>:32                                      ; preds = %._crit_edge9, %26
  br label %.preheader

; <label>:33                                      ; preds = %.preheader
  %r_req2_read = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @r_req2) nounwind
  br i1 %r_req2_read, label %34, label %38

; <label>:34                                      ; preds = %33
  call void (...)* @_ssdm_op_Wait(i32 1)
  %addr2_read = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @addr2) nounwind
  %tmp_4 = zext i8 %addr2_read to i64
  %mem_addr_3 = getelementptr inbounds [256 x i8]* %mem, i64 0, i64 %tmp_4
  %mem_load_2 = load i8* %mem_addr_3, align 1
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @dout2, i8 %mem_load_2) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  br label %35

; <label>:35                                      ; preds = %36, %34
  %r_req2_read_1 = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @r_req2) nounwind
  br i1 %r_req2_read_1, label %36, label %37

; <label>:36                                      ; preds = %35
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @r_ack2, i1 true) nounwind
  br label %35

; <label>:37                                      ; preds = %35
  call void (...)* @_ssdm_op_Wait(i32 1)
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @r_ack2, i1 false) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  br label %43

; <label>:38                                      ; preds = %33
  %w_req2_read = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @w_req2) nounwind
  br i1 %w_req2_read, label %39, label %._crit_edge10

; <label>:39                                      ; preds = %38
  call void (...)* @_ssdm_op_Wait(i32 1)
  %din2_read = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @din2) nounwind
  %addr2_read_1 = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @addr2) nounwind
  %tmp_8 = zext i8 %addr2_read_1 to i64
  %mem_addr_7 = getelementptr inbounds [256 x i8]* %mem, i64 0, i64 %tmp_8
  store i8 %din2_read, i8* %mem_addr_7, align 1
  call void (...)* @_ssdm_op_Wait(i32 1)
  br label %40

; <label>:40                                      ; preds = %41, %39
  %w_req2_read_1 = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @w_req2) nounwind
  br i1 %w_req2_read_1, label %41, label %42

; <label>:41                                      ; preds = %40
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @w_ack2, i1 true) nounwind
  br label %40

; <label>:42                                      ; preds = %40
  call void (...)* @_ssdm_op_Wait(i32 1)
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @w_ack2, i1 false) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  br label %._crit_edge10

._crit_edge10:                                    ; preds = %42, %38
  br label %43

; <label>:43                                      ; preds = %._crit_edge10, %37
  br label %.preheader

; <label>:44                                      ; preds = %.preheader
  %r_req3_read = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @r_req3) nounwind
  br i1 %r_req3_read, label %45, label %49

; <label>:45                                      ; preds = %44
  call void (...)* @_ssdm_op_Wait(i32 1)
  %addr3_read = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @addr3) nounwind
  %tmp_5 = zext i8 %addr3_read to i64
  %mem_addr_4 = getelementptr inbounds [256 x i8]* %mem, i64 0, i64 %tmp_5
  %mem_load_3 = load i8* %mem_addr_4, align 1
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @dout3, i8 %mem_load_3) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  br label %46

; <label>:46                                      ; preds = %47, %45
  %r_req3_read_1 = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @r_req3) nounwind
  br i1 %r_req3_read_1, label %47, label %48

; <label>:47                                      ; preds = %46
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @r_ack3, i1 true) nounwind
  br label %46

; <label>:48                                      ; preds = %46
  call void (...)* @_ssdm_op_Wait(i32 1)
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @r_ack3, i1 false) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  br label %54

; <label>:49                                      ; preds = %44
  %w_req3_read = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @w_req3) nounwind
  br i1 %w_req3_read, label %50, label %._crit_edge11

; <label>:50                                      ; preds = %49
  call void (...)* @_ssdm_op_Wait(i32 1)
  %din3_read = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @din3) nounwind
  %addr3_read_1 = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @addr3) nounwind
  %tmp_9 = zext i8 %addr3_read_1 to i64
  %mem_addr_8 = getelementptr inbounds [256 x i8]* %mem, i64 0, i64 %tmp_9
  store i8 %din3_read, i8* %mem_addr_8, align 1
  call void (...)* @_ssdm_op_Wait(i32 1)
  br label %51

; <label>:51                                      ; preds = %52, %50
  %w_req3_read_1 = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @w_req3) nounwind
  br i1 %w_req3_read_1, label %52, label %53

; <label>:52                                      ; preds = %51
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @w_ack3, i1 true) nounwind
  br label %51

; <label>:53                                      ; preds = %51
  call void (...)* @_ssdm_op_Wait(i32 1)
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @w_ack3, i1 false) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  br label %._crit_edge11

._crit_edge11:                                    ; preds = %53, %49
  br label %54

; <label>:54                                      ; preds = %._crit_edge11, %48
  br label %.preheader

; <label>:55                                      ; preds = %.preheader
  call void (...)* @_ssdm_op_Wait(i32 1)
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @r_ack0, i1 false) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @w_ack0, i1 false) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @r_ack1, i1 false) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @w_ack1, i1 false) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @r_ack2, i1 false) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @w_ack2, i1 false) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @r_ack3, i1 false) nounwind
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @w_ack3, i1 false) nounwind
  call void (...)* @_ssdm_op_Wait(i32 1)
  br label %.preheader
}

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

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

!hls.encrypted.func = !{}
!llvm.map.gv = !{!0, !7, !12, !17, !22, !27, !32, !37, !42, !47, !52, !57, !62, !67, !72, !77, !82, !87, !92, !97, !102, !107, !112, !117, !122, !127, !132, !137}

!0 = metadata !{metadata !1, i1* @w_req3}
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0, i32 0, metadata !3}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"w_req3", metadata !5, metadata !"uint1", i32 0, i32 0}
!5 = metadata !{metadata !6}
!6 = metadata !{i32 0, i32 0, i32 1}
!7 = metadata !{metadata !8, i1* @w_req2}
!8 = metadata !{metadata !9}
!9 = metadata !{i32 0, i32 0, metadata !10}
!10 = metadata !{metadata !11}
!11 = metadata !{metadata !"w_req2", metadata !5, metadata !"uint1", i32 0, i32 0}
!12 = metadata !{metadata !13, i1* @w_req1}
!13 = metadata !{metadata !14}
!14 = metadata !{i32 0, i32 0, metadata !15}
!15 = metadata !{metadata !16}
!16 = metadata !{metadata !"w_req1", metadata !5, metadata !"uint1", i32 0, i32 0}
!17 = metadata !{metadata !18, i1* @w_req0}
!18 = metadata !{metadata !19}
!19 = metadata !{i32 0, i32 0, metadata !20}
!20 = metadata !{metadata !21}
!21 = metadata !{metadata !"w_req0", metadata !5, metadata !"uint1", i32 0, i32 0}
!22 = metadata !{metadata !23, i1* @w_ack3}
!23 = metadata !{metadata !24}
!24 = metadata !{i32 0, i32 0, metadata !25}
!25 = metadata !{metadata !26}
!26 = metadata !{metadata !"w_ack3", metadata !5, metadata !"uint1", i32 0, i32 0}
!27 = metadata !{metadata !28, i1* @w_ack2}
!28 = metadata !{metadata !29}
!29 = metadata !{i32 0, i32 0, metadata !30}
!30 = metadata !{metadata !31}
!31 = metadata !{metadata !"w_ack2", metadata !5, metadata !"uint1", i32 0, i32 0}
!32 = metadata !{metadata !33, i1* @w_ack1}
!33 = metadata !{metadata !34}
!34 = metadata !{i32 0, i32 0, metadata !35}
!35 = metadata !{metadata !36}
!36 = metadata !{metadata !"w_ack1", metadata !5, metadata !"uint1", i32 0, i32 0}
!37 = metadata !{metadata !38, i1* @w_ack0}
!38 = metadata !{metadata !39}
!39 = metadata !{i32 0, i32 0, metadata !40}
!40 = metadata !{metadata !41}
!41 = metadata !{metadata !"w_ack0", metadata !5, metadata !"uint1", i32 0, i32 0}
!42 = metadata !{metadata !43, i1* @r_req3}
!43 = metadata !{metadata !44}
!44 = metadata !{i32 0, i32 0, metadata !45}
!45 = metadata !{metadata !46}
!46 = metadata !{metadata !"r_req3", metadata !5, metadata !"uint1", i32 0, i32 0}
!47 = metadata !{metadata !48, i1* @r_req2}
!48 = metadata !{metadata !49}
!49 = metadata !{i32 0, i32 0, metadata !50}
!50 = metadata !{metadata !51}
!51 = metadata !{metadata !"r_req2", metadata !5, metadata !"uint1", i32 0, i32 0}
!52 = metadata !{metadata !53, i1* @r_req1}
!53 = metadata !{metadata !54}
!54 = metadata !{i32 0, i32 0, metadata !55}
!55 = metadata !{metadata !56}
!56 = metadata !{metadata !"r_req1", metadata !5, metadata !"uint1", i32 0, i32 0}
!57 = metadata !{metadata !58, i1* @r_req0}
!58 = metadata !{metadata !59}
!59 = metadata !{i32 0, i32 0, metadata !60}
!60 = metadata !{metadata !61}
!61 = metadata !{metadata !"r_req0", metadata !5, metadata !"uint1", i32 0, i32 0}
!62 = metadata !{metadata !63, i1* @r_ack3}
!63 = metadata !{metadata !64}
!64 = metadata !{i32 0, i32 0, metadata !65}
!65 = metadata !{metadata !66}
!66 = metadata !{metadata !"r_ack3", metadata !5, metadata !"uint1", i32 0, i32 0}
!67 = metadata !{metadata !68, i1* @r_ack2}
!68 = metadata !{metadata !69}
!69 = metadata !{i32 0, i32 0, metadata !70}
!70 = metadata !{metadata !71}
!71 = metadata !{metadata !"r_ack2", metadata !5, metadata !"uint1", i32 0, i32 0}
!72 = metadata !{metadata !73, i1* @r_ack1}
!73 = metadata !{metadata !74}
!74 = metadata !{i32 0, i32 0, metadata !75}
!75 = metadata !{metadata !76}
!76 = metadata !{metadata !"r_ack1", metadata !5, metadata !"uint1", i32 0, i32 0}
!77 = metadata !{metadata !78, i1* @r_ack0}
!78 = metadata !{metadata !79}
!79 = metadata !{i32 0, i32 0, metadata !80}
!80 = metadata !{metadata !81}
!81 = metadata !{metadata !"r_ack0", metadata !5, metadata !"uint1", i32 0, i32 0}
!82 = metadata !{metadata !83, i8* @dout3}
!83 = metadata !{metadata !84}
!84 = metadata !{i32 0, i32 7, metadata !85}
!85 = metadata !{metadata !86}
!86 = metadata !{metadata !"dout3", metadata !5, metadata !"uint8", i32 0, i32 7}
!87 = metadata !{metadata !88, i8* @dout2}
!88 = metadata !{metadata !89}
!89 = metadata !{i32 0, i32 7, metadata !90}
!90 = metadata !{metadata !91}
!91 = metadata !{metadata !"dout2", metadata !5, metadata !"uint8", i32 0, i32 7}
!92 = metadata !{metadata !93, i8* @dout1}
!93 = metadata !{metadata !94}
!94 = metadata !{i32 0, i32 7, metadata !95}
!95 = metadata !{metadata !96}
!96 = metadata !{metadata !"dout1", metadata !5, metadata !"uint8", i32 0, i32 7}
!97 = metadata !{metadata !98, i8* @dout0}
!98 = metadata !{metadata !99}
!99 = metadata !{i32 0, i32 7, metadata !100}
!100 = metadata !{metadata !101}
!101 = metadata !{metadata !"dout0", metadata !5, metadata !"uint8", i32 0, i32 7}
!102 = metadata !{metadata !103, i8* @din3}
!103 = metadata !{metadata !104}
!104 = metadata !{i32 0, i32 7, metadata !105}
!105 = metadata !{metadata !106}
!106 = metadata !{metadata !"din3", metadata !5, metadata !"uint8", i32 0, i32 7}
!107 = metadata !{metadata !108, i8* @din2}
!108 = metadata !{metadata !109}
!109 = metadata !{i32 0, i32 7, metadata !110}
!110 = metadata !{metadata !111}
!111 = metadata !{metadata !"din2", metadata !5, metadata !"uint8", i32 0, i32 7}
!112 = metadata !{metadata !113, i8* @din1}
!113 = metadata !{metadata !114}
!114 = metadata !{i32 0, i32 7, metadata !115}
!115 = metadata !{metadata !116}
!116 = metadata !{metadata !"din1", metadata !5, metadata !"uint8", i32 0, i32 7}
!117 = metadata !{metadata !118, i8* @din0}
!118 = metadata !{metadata !119}
!119 = metadata !{i32 0, i32 7, metadata !120}
!120 = metadata !{metadata !121}
!121 = metadata !{metadata !"din0", metadata !5, metadata !"uint8", i32 0, i32 7}
!122 = metadata !{metadata !123, i8* @addr3}
!123 = metadata !{metadata !124}
!124 = metadata !{i32 0, i32 7, metadata !125}
!125 = metadata !{metadata !126}
!126 = metadata !{metadata !"addr3", metadata !5, metadata !"uint8", i32 0, i32 7}
!127 = metadata !{metadata !128, i8* @addr2}
!128 = metadata !{metadata !129}
!129 = metadata !{i32 0, i32 7, metadata !130}
!130 = metadata !{metadata !131}
!131 = metadata !{metadata !"addr2", metadata !5, metadata !"uint8", i32 0, i32 7}
!132 = metadata !{metadata !133, i8* @addr1}
!133 = metadata !{metadata !134}
!134 = metadata !{i32 0, i32 7, metadata !135}
!135 = metadata !{metadata !136}
!136 = metadata !{metadata !"addr1", metadata !5, metadata !"uint8", i32 0, i32 7}
!137 = metadata !{metadata !138, i8* @addr0}
!138 = metadata !{metadata !139}
!139 = metadata !{i32 0, i32 7, metadata !140}
!140 = metadata !{metadata !141}
!141 = metadata !{metadata !"addr0", metadata !5, metadata !"uint8", i32 0, i32 7}
