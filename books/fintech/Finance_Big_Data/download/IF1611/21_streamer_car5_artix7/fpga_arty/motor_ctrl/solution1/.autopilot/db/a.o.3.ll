; ModuleID = 'D:/21_streamer_car5_artix7/fpga_arty/motor_ctrl/solution1/.autopilot/db/a.o.3.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-w64-mingw32"

@r_pwm = global i1 false, align 1                 ; [#uses=5 type=i1*]
@r_dir = global i1 false, align 1                 ; [#uses=4 type=i1*]
@mem_wreq = global i1 false, align 1              ; [#uses=5 type=i1*]
@mem_wack = common global i1 false, align 1       ; [#uses=3 type=i1*]
@mem_rreq = global i1 false, align 1              ; [#uses=5 type=i1*]
@mem_rack = common global i1 false, align 1       ; [#uses=3 type=i1*]
@mem_dout = global i8 0, align 1                  ; [#uses=5 type=i8*]
@mem_din = common global i8 0, align 1            ; [#uses=3 type=i8*]
@mem_addr = global i8 0, align 1                  ; [#uses=8 type=i8*]
@l_pwm = global i1 false, align 1                 ; [#uses=5 type=i1*]
@l_dir = global i1 false, align 1                 ; [#uses=4 type=i1*]
@dummy_tmr_out = global i1 false, align 1         ; [#uses=4 type=i1*]
@p_str1 = private unnamed_addr constant [8 x i8] c"ap_none\00", align 1 ; [#uses=12 type=[8 x i8]*]
@p_str = private unnamed_addr constant [1 x i8] zeroinitializer, align 1 ; [#uses=48 type=[1 x i8]*]

; [#uses=24]
define internal fastcc void @motor_ctrl_write_mem(i8 zeroext %addr, i8 zeroext %data) nounwind uwtable {
  %data_read = call i8 @_ssdm_op_Read.ap_auto.i8(i8 %data) nounwind ; [#uses=3 type=i8]
  call void @llvm.dbg.value(metadata !{i8 %data_read}, i64 0, metadata !62), !dbg !71 ; [debug line = 85:34] [debug variable = data]
  %addr_read = call i8 @_ssdm_op_Read.ap_auto.i8(i8 %addr) nounwind ; [#uses=3 type=i8]
  call void @llvm.dbg.value(metadata !{i8 %addr_read}, i64 0, metadata !72), !dbg !73 ; [debug line = 85:22] [debug variable = addr]
  call void @llvm.dbg.value(metadata !{i8 %addr}, i64 0, metadata !72), !dbg !73 ; [debug line = 85:22] [debug variable = addr]
  call void @llvm.dbg.value(metadata !{i8 %data}, i64 0, metadata !62), !dbg !71 ; [debug line = 85:34] [debug variable = data]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !74 ; [debug line = 88:2]
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @mem_addr, i8 %addr_read) nounwind, !dbg !76 ; [debug line = 89:2]
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @mem_dout, i8 %data_read) nounwind, !dbg !77 ; [debug line = 90:2]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @mem_wreq, i1 false) nounwind, !dbg !78 ; [debug line = 91:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !79 ; [debug line = 92:2]
  br label %._crit_edge, !dbg !80                 ; [debug line = 94:2]

._crit_edge:                                      ; preds = %._crit_edge, %0
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @mem_addr, i8 %addr_read) nounwind, !dbg !81 ; [debug line = 95:3]
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @mem_dout, i8 %data_read) nounwind, !dbg !83 ; [debug line = 96:3]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @mem_wreq, i1 true) nounwind, !dbg !84 ; [debug line = 97:3]
  %mem_wack_read = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @mem_wack) nounwind, !dbg !85 ; [#uses=1 type=i1] [debug line = 98:2]
  br i1 %mem_wack_read, label %1, label %._crit_edge, !dbg !85 ; [debug line = 98:2]

; <label>:1                                       ; preds = %._crit_edge
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !86 ; [debug line = 99:2]
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @mem_addr, i8 %addr_read) nounwind, !dbg !87 ; [debug line = 101:2]
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @mem_dout, i8 %data_read) nounwind, !dbg !88 ; [debug line = 102:2]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @mem_wreq, i1 false) nounwind, !dbg !89 ; [debug line = 103:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !90 ; [debug line = 104:2]
  ret void, !dbg !91                              ; [debug line = 105:1]
}

; [#uses=1]
define internal fastcc void @motor_ctrl_wait_tmr() nounwind uwtable {
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !92 ; [debug line = 77:2]
  br label %1, !dbg !99                           ; [debug line = 78:7]

; <label>:1                                       ; preds = %2, %0
  %t = phi i17 [ 0, %0 ], [ %t_1, %2 ]            ; [#uses=2 type=i17]
  %exitcond = icmp eq i17 %t, -31072, !dbg !99    ; [#uses=1 type=i1] [debug line = 78:7]
  %empty = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 100000, i64 100000, i64 100000) nounwind ; [#uses=0 type=i32]
  %t_1 = add i17 %t, 1, !dbg !101                 ; [#uses=1 type=i17] [debug line = 78:23]
  br i1 %exitcond, label %3, label %2, !dbg !99   ; [debug line = 78:7]

; <label>:2                                       ; preds = %1
  %dummy_tmr_out_read = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @dummy_tmr_out) nounwind, !dbg !102 ; [#uses=1 type=i1] [debug line = 79:3]
  %not_s = xor i1 %dummy_tmr_out_read, true, !dbg !102 ; [#uses=1 type=i1] [debug line = 79:3]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @dummy_tmr_out, i1 %not_s) nounwind, !dbg !102 ; [debug line = 79:3]
  call void @llvm.dbg.value(metadata !{i17 %t_1}, i64 0, metadata !104), !dbg !101 ; [debug line = 78:23] [debug variable = t]
  br label %1, !dbg !101                          ; [debug line = 78:23]

; <label>:3                                       ; preds = %1
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !105 ; [debug line = 81:2]
  ret void, !dbg !106                             ; [debug line = 82:1]
}

; [#uses=6]
define internal fastcc zeroext i8 @motor_ctrl_read_mem(i8 zeroext %addr) nounwind uwtable {
  %addr_read = call i8 @_ssdm_op_Read.ap_auto.i8(i8 %addr) nounwind ; [#uses=3 type=i8]
  call void @llvm.dbg.value(metadata !{i8 %addr_read}, i64 0, metadata !107), !dbg !111 ; [debug line = 108:22] [debug variable = addr]
  call void @llvm.dbg.value(metadata !{i8 %addr}, i64 0, metadata !107), !dbg !111 ; [debug line = 108:22] [debug variable = addr]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !112 ; [debug line = 113:2]
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @mem_addr, i8 %addr_read) nounwind, !dbg !114 ; [debug line = 114:2]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @mem_rreq, i1 true) nounwind, !dbg !115 ; [debug line = 115:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !116 ; [debug line = 116:2]
  br label %._crit_edge, !dbg !117                ; [debug line = 118:2]

._crit_edge:                                      ; preds = %._crit_edge, %0
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @mem_addr, i8 %addr_read) nounwind, !dbg !118 ; [debug line = 119:3]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @mem_rreq, i1 true) nounwind, !dbg !120 ; [debug line = 120:3]
  %dt = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @mem_din) nounwind, !dbg !121 ; [#uses=1 type=i8] [debug line = 121:3]
  call void @llvm.dbg.value(metadata !{i8 %dt}, i64 0, metadata !122), !dbg !121 ; [debug line = 121:3] [debug variable = dt]
  %mem_rack_read = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @mem_rack) nounwind, !dbg !123 ; [#uses=1 type=i1] [debug line = 122:2]
  br i1 %mem_rack_read, label %1, label %._crit_edge, !dbg !123 ; [debug line = 122:2]

; <label>:1                                       ; preds = %._crit_edge
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !124 ; [debug line = 123:2]
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @mem_addr, i8 %addr_read) nounwind, !dbg !125 ; [debug line = 125:2]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @mem_rreq, i1 false) nounwind, !dbg !126 ; [debug line = 126:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !127 ; [debug line = 127:2]
  ret i8 %dt, !dbg !128                           ; [debug line = 129:2]
}

; [#uses=0]
define void @motor_ctrl() noreturn nounwind uwtable {
  %mtr_pwm_cnt = alloca i32, align 4              ; [#uses=7 type=i32*]
  call void (...)* @_ssdm_op_SpecTopModule(), !dbg !129 ; [debug line = 178:1]
  %dummy_tmr_out_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @dummy_tmr_out) nounwind, !dbg !134 ; [#uses=0 type=i1] [debug line = 179:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @dummy_tmr_out, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !134 ; [debug line = 179:1]
  %l_dir_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @l_dir) nounwind, !dbg !135 ; [#uses=0 type=i1] [debug line = 181:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @l_dir, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !135 ; [debug line = 181:1]
  %l_pwm_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @l_pwm) nounwind, !dbg !136 ; [#uses=0 type=i1] [debug line = 182:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @l_pwm, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !136 ; [debug line = 182:1]
  %r_dir_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @r_dir) nounwind, !dbg !137 ; [#uses=0 type=i1] [debug line = 183:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @r_dir, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !137 ; [debug line = 183:1]
  %r_pwm_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @r_pwm) nounwind, !dbg !138 ; [#uses=0 type=i1] [debug line = 184:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @r_pwm, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !138 ; [debug line = 184:1]
  %mem_addr_load = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @mem_addr) nounwind, !dbg !139 ; [#uses=0 type=i8] [debug line = 186:1]
  call void (...)* @_ssdm_op_SpecInterface(i8* @mem_addr, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !139 ; [debug line = 186:1]
  %mem_din_load = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @mem_din) nounwind, !dbg !140 ; [#uses=0 type=i8] [debug line = 187:1]
  call void (...)* @_ssdm_op_SpecInterface(i8* @mem_din, [8 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !140 ; [debug line = 187:1]
  %mem_dout_load = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @mem_dout) nounwind, !dbg !141 ; [#uses=0 type=i8] [debug line = 188:1]
  call void (...)* @_ssdm_op_SpecInterface(i8* @mem_dout, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !141 ; [debug line = 188:1]
  %mem_wreq_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @mem_wreq) nounwind, !dbg !142 ; [#uses=0 type=i1] [debug line = 189:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @mem_wreq, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !142 ; [debug line = 189:1]
  %mem_wack_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @mem_wack) nounwind, !dbg !143 ; [#uses=0 type=i1] [debug line = 190:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @mem_wack, [8 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !143 ; [debug line = 190:1]
  %mem_rreq_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @mem_rreq) nounwind, !dbg !144 ; [#uses=0 type=i1] [debug line = 191:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @mem_rreq, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !144 ; [debug line = 191:1]
  %mem_rack_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @mem_rack) nounwind, !dbg !145 ; [#uses=0 type=i1] [debug line = 192:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @mem_rack, [8 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !145 ; [debug line = 192:1]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !146 ; [debug line = 207:2]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @l_dir, i1 false) nounwind, !dbg !147 ; [debug line = 208:2]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @l_pwm, i1 false) nounwind, !dbg !148 ; [debug line = 209:2]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @r_dir, i1 false) nounwind, !dbg !149 ; [debug line = 210:2]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @r_pwm, i1 false) nounwind, !dbg !150 ; [debug line = 211:2]
  store volatile i32 0, i32* %mtr_pwm_cnt, align 4, !dbg !151 ; [debug line = 218:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !152 ; [debug line = 220:2]
  call fastcc void @motor_ctrl_write_mem(i8 zeroext -128, i8 zeroext 0), !dbg !153 ; [debug line = 221:2]
  call fastcc void @motor_ctrl_write_mem(i8 zeroext -123, i8 zeroext 0), !dbg !154 ; [debug line = 222:2]
  call fastcc void @motor_ctrl_write_mem(i8 zeroext -122, i8 zeroext 0), !dbg !155 ; [debug line = 223:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !156 ; [debug line = 224:2]
  br label %1, !dbg !157                          ; [debug line = 226:7]

; <label>:1                                       ; preds = %2, %0
  %i = phi i6 [ 0, %0 ], [ %i_1, %2 ]             ; [#uses=3 type=i6]
  %i_cast = zext i6 %i to i8, !dbg !157           ; [#uses=1 type=i8] [debug line = 226:7]
  %exitcond = icmp eq i6 %i, -32, !dbg !157       ; [#uses=1 type=i1] [debug line = 226:7]
  %empty = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 32, i64 32, i64 32) nounwind ; [#uses=0 type=i32]
  %i_1 = add i6 %i, 1, !dbg !159                  ; [#uses=1 type=i6] [debug line = 226:22]
  br i1 %exitcond, label %.preheader, label %2, !dbg !157 ; [debug line = 226:7]

; <label>:2                                       ; preds = %1
  call fastcc void @motor_ctrl_write_mem(i8 zeroext %i_cast, i8 zeroext 32), !dbg !160 ; [debug line = 227:3]
  call void @llvm.dbg.value(metadata !{i6 %i_1}, i64 0, metadata !162), !dbg !159 ; [debug line = 226:22] [debug variable = i]
  br label %1, !dbg !159                          ; [debug line = 226:22]

.preheader:                                       ; preds = %._crit_edge11, %1
  %chR_dir = phi i1 [ %chR_dir_5, %._crit_edge11 ], [ false, %1 ] ; [#uses=3 type=i1]
  %chL_dir = phi i1 [ %chL_dir_5, %._crit_edge11 ], [ false, %1 ] ; [#uses=3 type=i1]
  %loop_begin = call i32 (...)* @_ssdm_op_SpecLoopBegin() nounwind ; [#uses=0 type=i32]
  %eh = call fastcc zeroext i8 @motor_ctrl_read_mem(i8 zeroext -127), !dbg !163 ; [#uses=3 type=i8] [debug line = 251:8]
  call void @llvm.dbg.value(metadata !{i8 %eh}, i64 0, metadata !165), !dbg !163 ; [debug line = 251:8] [debug variable = eh]
  %el = call fastcc zeroext i8 @motor_ctrl_read_mem(i8 zeroext -126), !dbg !166 ; [#uses=3 type=i8] [debug line = 252:8]
  call void @llvm.dbg.value(metadata !{i8 %el}, i64 0, metadata !167), !dbg !166 ; [debug line = 252:8] [debug variable = el]
  %et = call i16 @_ssdm_op_BitConcatenate.i16.i8.i8(i8 %eh, i8 %el) ; [#uses=1 type=i16]
  call void @llvm.dbg.value(metadata !{i16 %et}, i64 0, metadata !168), !dbg !171 ; [debug line = 253:3] [debug variable = et]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !172 ; [debug line = 255:3]
  br label %._crit_edge, !dbg !173                ; [debug line = 256:3]

._crit_edge:                                      ; preds = %._crit_edge, %.preheader
  %tmp_s = call fastcc zeroext i8 @motor_ctrl_read_mem(i8 zeroext -123), !dbg !174 ; [#uses=1 type=i8] [debug line = 256:10]
  %tmp_1 = icmp eq i8 %tmp_s, 0, !dbg !174        ; [#uses=1 type=i1] [debug line = 256:10]
  br i1 %tmp_1, label %._crit_edge, label %3, !dbg !174 ; [debug line = 256:10]

; <label>:3                                       ; preds = %._crit_edge
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !175 ; [debug line = 258:3]
  %eh_1 = call fastcc zeroext i8 @motor_ctrl_read_mem(i8 zeroext -125), !dbg !176 ; [#uses=3 type=i8] [debug line = 259:8]
  call void @llvm.dbg.value(metadata !{i8 %eh_1}, i64 0, metadata !165), !dbg !176 ; [debug line = 259:8] [debug variable = eh]
  %el_1 = call fastcc zeroext i8 @motor_ctrl_read_mem(i8 zeroext -124), !dbg !177 ; [#uses=3 type=i8] [debug line = 260:8]
  call void @llvm.dbg.value(metadata !{i8 %el_1}, i64 0, metadata !167), !dbg !177 ; [debug line = 260:8] [debug variable = el]
  %e = call i16 @_ssdm_op_BitConcatenate.i16.i8.i8(i8 %eh_1, i8 %el_1) ; [#uses=1 type=i16]
  call void @llvm.dbg.value(metadata !{i16 %e}, i64 0, metadata !178), !dbg !179 ; [debug line = 261:3] [debug variable = e]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !180 ; [debug line = 262:3]
  %tmp_3 = call i4 @_ssdm_op_PartSelect.i4.i8.i32.i32(i8 %eh_1, i32 4, i32 7), !dbg !181 ; [#uses=1 type=i4] [debug line = 265:16]
  %tmp_4 = call fastcc zeroext i7 @motor_ctrl_bin2char(i4 zeroext %tmp_3) nounwind, !dbg !181 ; [#uses=1 type=i7] [debug line = 265:16]
  %p_trunc_ext = zext i7 %tmp_4 to i8, !dbg !181  ; [#uses=1 type=i8] [debug line = 265:16]
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 3, i8 zeroext %p_trunc_ext), !dbg !181 ; [debug line = 265:16]
  %tmp_2 = trunc i8 %eh_1 to i4, !dbg !182        ; [#uses=1 type=i4] [debug line = 266:16]
  %tmp_6 = call fastcc zeroext i7 @motor_ctrl_bin2char(i4 zeroext %tmp_2) nounwind, !dbg !182 ; [#uses=1 type=i7] [debug line = 266:16]
  %p_trunc115_ext = zext i7 %tmp_6 to i8, !dbg !182 ; [#uses=1 type=i8] [debug line = 266:16]
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 4, i8 zeroext %p_trunc115_ext), !dbg !182 ; [debug line = 266:16]
  %tmp_8 = call i4 @_ssdm_op_PartSelect.i4.i8.i32.i32(i8 %el_1, i32 4, i32 7), !dbg !183 ; [#uses=1 type=i4] [debug line = 267:16]
  %tmp_9 = call fastcc zeroext i7 @motor_ctrl_bin2char(i4 zeroext %tmp_8) nounwind, !dbg !183 ; [#uses=1 type=i7] [debug line = 267:16]
  %p_trunc116_ext = zext i7 %tmp_9 to i8, !dbg !183 ; [#uses=1 type=i8] [debug line = 267:16]
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 5, i8 zeroext %p_trunc116_ext), !dbg !183 ; [debug line = 267:16]
  %tmp_5 = trunc i8 %el_1 to i4, !dbg !184        ; [#uses=1 type=i4] [debug line = 268:16]
  %tmp_7 = call fastcc zeroext i7 @motor_ctrl_bin2char(i4 zeroext %tmp_5) nounwind, !dbg !184 ; [#uses=1 type=i7] [debug line = 268:16]
  %p_trunc117_ext = zext i7 %tmp_7 to i8, !dbg !184 ; [#uses=1 type=i8] [debug line = 268:16]
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 6, i8 zeroext %p_trunc117_ext), !dbg !184 ; [debug line = 268:16]
  %tmp_10 = call i4 @_ssdm_op_PartSelect.i4.i8.i32.i32(i8 %eh, i32 4, i32 7), !dbg !185 ; [#uses=1 type=i4] [debug line = 270:16]
  %tmp_11 = call fastcc zeroext i7 @motor_ctrl_bin2char(i4 zeroext %tmp_10) nounwind, !dbg !185 ; [#uses=1 type=i7] [debug line = 270:16]
  %p_trunc118_ext = zext i7 %tmp_11 to i8, !dbg !185 ; [#uses=1 type=i8] [debug line = 270:16]
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 8, i8 zeroext %p_trunc118_ext), !dbg !185 ; [debug line = 270:16]
  %tmp_12 = trunc i8 %eh to i4, !dbg !186         ; [#uses=1 type=i4] [debug line = 271:16]
  %tmp_13 = call fastcc zeroext i7 @motor_ctrl_bin2char(i4 zeroext %tmp_12) nounwind, !dbg !186 ; [#uses=1 type=i7] [debug line = 271:16]
  %p_trunc119_ext = zext i7 %tmp_13 to i8, !dbg !186 ; [#uses=1 type=i8] [debug line = 271:16]
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 9, i8 zeroext %p_trunc119_ext), !dbg !186 ; [debug line = 271:16]
  %tmp_14 = call i4 @_ssdm_op_PartSelect.i4.i8.i32.i32(i8 %el, i32 4, i32 7), !dbg !187 ; [#uses=1 type=i4] [debug line = 272:17]
  %tmp_15 = call fastcc zeroext i7 @motor_ctrl_bin2char(i4 zeroext %tmp_14) nounwind, !dbg !187 ; [#uses=1 type=i7] [debug line = 272:17]
  %p_trunc120_ext = zext i7 %tmp_15 to i8, !dbg !187 ; [#uses=1 type=i8] [debug line = 272:17]
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 10, i8 zeroext %p_trunc120_ext), !dbg !187 ; [debug line = 272:17]
  %tmp_16 = trunc i8 %el to i4, !dbg !188         ; [#uses=1 type=i4] [debug line = 273:17]
  %tmp_17 = call fastcc zeroext i7 @motor_ctrl_bin2char(i4 zeroext %tmp_16) nounwind, !dbg !188 ; [#uses=1 type=i7] [debug line = 273:17]
  %p_trunc121_ext = zext i7 %tmp_17 to i8, !dbg !188 ; [#uses=1 type=i8] [debug line = 273:17]
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 11, i8 zeroext %p_trunc121_ext), !dbg !188 ; [debug line = 273:17]
  %diff_agl = call fastcc i21 @motor_ctrl_diff_angle(i16 zeroext %et, i16 zeroext %e) nounwind, !dbg !189 ; [#uses=6 type=i21] [debug line = 281:14]
  %tmp_18 = call i4 @_ssdm_op_PartSelect.i4.i21.i32.i32(i21 %diff_agl, i32 12, i32 15), !dbg !190 ; [#uses=1 type=i4] [debug line = 284:17]
  %tmp_19 = call fastcc zeroext i7 @motor_ctrl_bin2char(i4 zeroext %tmp_18) nounwind, !dbg !190 ; [#uses=1 type=i7] [debug line = 284:17]
  %p_trunc122_ext = zext i7 %tmp_19 to i8, !dbg !190 ; [#uses=1 type=i8] [debug line = 284:17]
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 16, i8 zeroext %p_trunc122_ext), !dbg !190 ; [debug line = 284:17]
  %tmp_20 = call i4 @_ssdm_op_PartSelect.i4.i21.i32.i32(i21 %diff_agl, i32 8, i32 11), !dbg !191 ; [#uses=1 type=i4] [debug line = 285:17]
  %tmp_21 = call fastcc zeroext i7 @motor_ctrl_bin2char(i4 zeroext %tmp_20) nounwind, !dbg !191 ; [#uses=1 type=i7] [debug line = 285:17]
  %p_trunc123_ext = zext i7 %tmp_21 to i8, !dbg !191 ; [#uses=1 type=i8] [debug line = 285:17]
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 17, i8 zeroext %p_trunc123_ext), !dbg !191 ; [debug line = 285:17]
  %tmp_22 = call i4 @_ssdm_op_PartSelect.i4.i21.i32.i32(i21 %diff_agl, i32 4, i32 7), !dbg !192 ; [#uses=1 type=i4] [debug line = 286:17]
  %tmp_23 = call fastcc zeroext i7 @motor_ctrl_bin2char(i4 zeroext %tmp_22) nounwind, !dbg !192 ; [#uses=1 type=i7] [debug line = 286:17]
  %p_trunc124_ext = zext i7 %tmp_23 to i8, !dbg !192 ; [#uses=1 type=i8] [debug line = 286:17]
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 18, i8 zeroext %p_trunc124_ext), !dbg !192 ; [debug line = 286:17]
  %tmp_24 = trunc i21 %diff_agl to i4, !dbg !193  ; [#uses=1 type=i4] [debug line = 287:17]
  %tmp_25 = call fastcc zeroext i7 @motor_ctrl_bin2char(i4 zeroext %tmp_24) nounwind, !dbg !193 ; [#uses=1 type=i7] [debug line = 287:17]
  %p_trunc125_ext = zext i7 %tmp_25 to i8, !dbg !193 ; [#uses=1 type=i8] [debug line = 287:17]
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 19, i8 zeroext %p_trunc125_ext), !dbg !193 ; [debug line = 287:17]
  %tmp_26 = icmp slt i21 %diff_agl, -249, !dbg !194 ; [#uses=1 type=i1] [debug line = 289:3]
  br i1 %tmp_26, label %4, label %5, !dbg !194    ; [debug line = 289:3]

; <label>:4                                       ; preds = %3
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 21, i8 zeroext 76), !dbg !195 ; [debug line = 291:4]
  br label %6, !dbg !197                          ; [debug line = 292:3]

; <label>:5                                       ; preds = %3
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 21, i8 zeroext 45), !dbg !198 ; [debug line = 295:4]
  br label %6

; <label>:6                                       ; preds = %5, %4
  %too_left = phi i1 [ true, %4 ], [ false, %5 ]  ; [#uses=3 type=i1]
  %tmp_27 = icmp sgt i21 %diff_agl, 249, !dbg !200 ; [#uses=1 type=i1] [debug line = 298:3]
  br i1 %tmp_27, label %7, label %8, !dbg !200    ; [debug line = 298:3]

; <label>:7                                       ; preds = %6
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 22, i8 zeroext 82), !dbg !201 ; [debug line = 300:4]
  br label %_ifconv, !dbg !203                    ; [debug line = 301:3]

; <label>:8                                       ; preds = %6
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 22, i8 zeroext 45), !dbg !204 ; [debug line = 304:4]
  br label %_ifconv

_ifconv:                                          ; preds = %8, %7
  %too_right = phi i1 [ true, %7 ], [ false, %8 ] ; [#uses=9 type=i1]
  %mode = call fastcc zeroext i8 @motor_ctrl_read_mem(i8 zeroext -128), !dbg !206 ; [#uses=2 type=i8] [debug line = 308:10]
  call void @llvm.dbg.value(metadata !{i8 %mode}, i64 0, metadata !207), !dbg !206 ; [debug line = 308:10] [debug variable = mode]
  %tmp_36 = call i5 @_ssdm_op_PartSelect.i5.i8.i32.i32(i8 %mode, i32 3, i32 7), !dbg !208 ; [#uses=1 type=i5] [debug line = 310:3]
  %zext_cast = zext i5 %tmp_36 to i12, !dbg !208  ; [#uses=1 type=i12] [debug line = 310:3]
  %mul = mul i12 43, %zext_cast, !dbg !208        ; [#uses=1 type=i12] [debug line = 310:3]
  %chR_pwm_cast = call i4 @_ssdm_op_PartSelect.i4.i12.i32.i32(i12 %mul, i32 7, i32 10), !dbg !208 ; [#uses=2 type=i4] [debug line = 310:3]
  %tmp_37 = trunc i8 %mode to i3, !dbg !209       ; [#uses=5 type=i3] [debug line = 312:3]
  %brmerge = or i1 %too_right, %too_left, !dbg !210 ; [#uses=3 type=i1] [debug line = 342:4]
  %p_s = select i1 %too_left, i4 0, i4 %chR_pwm_cast, !dbg !212 ; [#uses=2 type=i4] [debug line = 321:9]
  %p_1 = select i1 %too_left, i4 -6, i4 %chR_pwm_cast, !dbg !212 ; [#uses=2 type=i4] [debug line = 321:9]
  %chL_pwm_4 = select i1 %too_right, i4 0, i4 -6, !dbg !213 ; [#uses=1 type=i4] [debug line = 347:5]
  %not_too_right_s = xor i1 %too_right, true, !dbg !213 ; [#uses=4 type=i1] [debug line = 347:5]
  %chR_pwm_6 = select i1 %too_right, i4 -6, i4 0, !dbg !215 ; [#uses=1 type=i4] [debug line = 368:5]
  %sel_tmp = icmp eq i3 %tmp_37, 1                ; [#uses=2 type=i1]
  %sel_tmp2 = and i1 %sel_tmp, %not_too_right_s   ; [#uses=2 type=i1]
  %sel_tmp3 = select i1 %sel_tmp2, i4 %p_s, i4 0  ; [#uses=1 type=i4]
  %sel_tmp4 = icmp eq i3 %tmp_37, 3               ; [#uses=2 type=i1]
  %sel_tmp6 = and i1 %sel_tmp4, %not_too_right_s  ; [#uses=4 type=i1]
  %sel_tmp7 = select i1 %sel_tmp6, i4 %p_1, i4 %sel_tmp3 ; [#uses=1 type=i4]
  %sel_tmp8 = icmp eq i3 %tmp_37, -3              ; [#uses=2 type=i1]
  %sel_tmp9 = and i1 %sel_tmp8, %brmerge          ; [#uses=4 type=i1]
  %sel_tmp1 = select i1 %sel_tmp9, i4 -6, i4 %sel_tmp7 ; [#uses=1 type=i4]
  %sel_tmp5 = icmp eq i3 %tmp_37, -1              ; [#uses=2 type=i1]
  %sel_tmp10 = and i1 %sel_tmp5, %brmerge         ; [#uses=4 type=i1]
  %sel_tmp11 = select i1 %sel_tmp10, i4 %chR_pwm_6, i4 %sel_tmp1 ; [#uses=1 type=i4]
  %sel_tmp12 = and i1 %sel_tmp, %too_right        ; [#uses=2 type=i1]
  %sel_tmp13 = and i1 %sel_tmp4, %too_right       ; [#uses=5 type=i1]
  %sel_tmp14 = select i1 %sel_tmp13, i4 0, i4 -6  ; [#uses=1 type=i4]
  %tmp = or i1 %sel_tmp13, %sel_tmp12             ; [#uses=2 type=i1]
  %sel_tmp15 = select i1 %tmp, i4 %sel_tmp14, i4 %sel_tmp11 ; [#uses=1 type=i4]
  %sel_tmp16 = xor i1 %brmerge, true, !dbg !210   ; [#uses=2 type=i1] [debug line = 342:4]
  %sel_tmp17 = and i1 %sel_tmp8, %sel_tmp16       ; [#uses=3 type=i1]
  %sel_tmp18 = and i1 %sel_tmp5, %sel_tmp16       ; [#uses=3 type=i1]
  %tmp_28 = or i1 %sel_tmp18, %sel_tmp17          ; [#uses=2 type=i1]
  %chR_pwm_8 = select i1 %tmp_28, i4 0, i4 %sel_tmp15 ; [#uses=2 type=i4]
  %sel_tmp19 = select i1 %sel_tmp2, i4 %p_1, i4 0 ; [#uses=1 type=i4]
  %sel_tmp20 = select i1 %sel_tmp6, i4 %p_s, i4 %sel_tmp19 ; [#uses=1 type=i4]
  %sel_tmp21 = select i1 %sel_tmp9, i4 %chL_pwm_4, i4 %sel_tmp20 ; [#uses=1 type=i4]
  %sel_tmp22 = select i1 %sel_tmp10, i4 -6, i4 %sel_tmp21 ; [#uses=1 type=i4]
  %sel_tmp23 = select i1 %sel_tmp13, i4 -6, i4 0  ; [#uses=1 type=i4]
  %sel_tmp24 = select i1 %tmp, i4 %sel_tmp23, i4 %sel_tmp22 ; [#uses=1 type=i4]
  %chL_pwm_8 = select i1 %tmp_28, i4 0, i4 %sel_tmp24 ; [#uses=2 type=i4]
  %sel_tmp56_not = icmp ne i3 %tmp_37, 1          ; [#uses=1 type=i1]
  %not_sel_tmp = or i1 %too_right, %sel_tmp56_not ; [#uses=2 type=i1]
  %sel_tmp25 = and i1 %chR_dir, %not_sel_tmp      ; [#uses=1 type=i1]
  %sel_tmp26 = or i1 %sel_tmp6, %sel_tmp25        ; [#uses=1 type=i1]
  %sel_tmp27 = select i1 %sel_tmp9, i1 %not_too_right_s, i1 %sel_tmp26 ; [#uses=1 type=i1]
  %sel_tmp28 = select i1 %sel_tmp10, i1 %not_too_right_s, i1 %sel_tmp27 ; [#uses=1 type=i1]
  %not_sel_tmp1 = xor i1 %sel_tmp12, true         ; [#uses=2 type=i1]
  %sel_tmp29 = and i1 %sel_tmp28, %not_sel_tmp1   ; [#uses=1 type=i1]
  %sel_tmp30 = or i1 %sel_tmp13, %sel_tmp29       ; [#uses=1 type=i1]
  %sel_tmp31 = select i1 %sel_tmp17, i1 %chR_dir, i1 %sel_tmp30 ; [#uses=1 type=i1]
  %chR_dir_5 = select i1 %sel_tmp18, i1 %chR_dir, i1 %sel_tmp31 ; [#uses=2 type=i1]
  %sel_tmp32 = and i1 %chL_dir, %not_sel_tmp      ; [#uses=1 type=i1]
  %sel_tmp33 = or i1 %sel_tmp6, %sel_tmp32        ; [#uses=1 type=i1]
  %sel_tmp34 = select i1 %sel_tmp9, i1 %too_right, i1 %sel_tmp33 ; [#uses=1 type=i1]
  %sel_tmp35 = select i1 %sel_tmp10, i1 %too_right, i1 %sel_tmp34 ; [#uses=1 type=i1]
  %sel_tmp36 = and i1 %sel_tmp35, %not_sel_tmp1   ; [#uses=1 type=i1]
  %sel_tmp37 = or i1 %sel_tmp13, %sel_tmp36       ; [#uses=1 type=i1]
  %sel_tmp38 = select i1 %sel_tmp17, i1 %chL_dir, i1 %sel_tmp37 ; [#uses=1 type=i1]
  %chL_dir_5 = select i1 %sel_tmp18, i1 %chL_dir, i1 %sel_tmp38 ; [#uses=2 type=i1]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !217 ; [debug line = 390:3]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @r_dir, i1 %chR_dir_5) nounwind, !dbg !218 ; [debug line = 391:3]
  call void @llvm.dbg.value(metadata !{i32* %mtr_pwm_cnt}, i64 0, metadata !219), !dbg !222 ; [debug line = 392:3] [debug variable = mtr_pwm_cnt]
  %mtr_pwm_cnt_load = load volatile i32* %mtr_pwm_cnt, align 4, !dbg !222 ; [#uses=1 type=i32] [debug line = 392:3]
  %tmp_29 = zext i4 %chR_pwm_8 to i32, !dbg !222  ; [#uses=1 type=i32] [debug line = 392:3]
  %tmp_30 = icmp slt i32 %mtr_pwm_cnt_load, %tmp_29, !dbg !222 ; [#uses=1 type=i1] [debug line = 392:3]
  br i1 %tmp_30, label %9, label %10, !dbg !222   ; [debug line = 392:3]

; <label>:9                                       ; preds = %_ifconv
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @r_pwm, i1 true) nounwind, !dbg !223 ; [debug line = 393:4]
  br label %11, !dbg !223                         ; [debug line = 393:4]

; <label>:10                                      ; preds = %_ifconv
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @r_pwm, i1 false) nounwind, !dbg !224 ; [debug line = 395:4]
  br label %11

; <label>:11                                      ; preds = %10, %9
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @l_dir, i1 %chL_dir_5) nounwind, !dbg !225 ; [debug line = 397:3]
  call void @llvm.dbg.value(metadata !{i32* %mtr_pwm_cnt}, i64 0, metadata !219), !dbg !226 ; [debug line = 398:3] [debug variable = mtr_pwm_cnt]
  %mtr_pwm_cnt_load_1 = load volatile i32* %mtr_pwm_cnt, align 4, !dbg !226 ; [#uses=1 type=i32] [debug line = 398:3]
  %tmp_31 = zext i4 %chL_pwm_8 to i32, !dbg !226  ; [#uses=1 type=i32] [debug line = 398:3]
  %tmp_32 = icmp slt i32 %mtr_pwm_cnt_load_1, %tmp_31, !dbg !226 ; [#uses=1 type=i1] [debug line = 398:3]
  br i1 %tmp_32, label %12, label %13, !dbg !226  ; [debug line = 398:3]

; <label>:12                                      ; preds = %11
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @l_pwm, i1 true) nounwind, !dbg !227 ; [debug line = 399:4]
  br label %14, !dbg !227                         ; [debug line = 399:4]

; <label>:13                                      ; preds = %11
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @l_pwm, i1 false) nounwind, !dbg !228 ; [debug line = 401:4]
  br label %14

; <label>:14                                      ; preds = %13, %12
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !229 ; [debug line = 402:3]
  call void @llvm.dbg.value(metadata !{i32* %mtr_pwm_cnt}, i64 0, metadata !219), !dbg !230 ; [debug line = 405:3] [debug variable = mtr_pwm_cnt]
  %mtr_pwm_cnt_load_2 = load volatile i32* %mtr_pwm_cnt, align 4, !dbg !230 ; [#uses=1 type=i32] [debug line = 405:3]
  %mtr_pwm_cnt_1 = add nsw i32 %mtr_pwm_cnt_load_2, 1, !dbg !230 ; [#uses=1 type=i32] [debug line = 405:3]
  call void @llvm.dbg.value(metadata !{i32 %mtr_pwm_cnt_1}, i64 0, metadata !219), !dbg !230 ; [debug line = 405:3] [debug variable = mtr_pwm_cnt]
  store volatile i32 %mtr_pwm_cnt_1, i32* %mtr_pwm_cnt, align 4, !dbg !230 ; [debug line = 405:3]
  %mtr_pwm_cnt_load_3 = load volatile i32* %mtr_pwm_cnt, align 4, !dbg !231 ; [#uses=1 type=i32] [debug line = 406:3]
  %tmp_33 = icmp sgt i32 %mtr_pwm_cnt_load_3, 9, !dbg !231 ; [#uses=1 type=i1] [debug line = 406:3]
  br i1 %tmp_33, label %15, label %._crit_edge11, !dbg !231 ; [debug line = 406:3]

; <label>:15                                      ; preds = %14
  store volatile i32 0, i32* %mtr_pwm_cnt, align 4, !dbg !232 ; [debug line = 407:4]
  br label %._crit_edge11, !dbg !234              ; [debug line = 408:3]

._crit_edge11:                                    ; preds = %15, %14
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 24, i8 zeroext 48), !dbg !235 ; [debug line = 411:17]
  %tmp_34 = call fastcc zeroext i7 @motor_ctrl_bin2char(i4 zeroext %chL_pwm_8) nounwind, !dbg !236 ; [#uses=1 type=i7] [debug line = 412:17]
  %p_trunc127_ext = zext i7 %tmp_34 to i8, !dbg !236 ; [#uses=1 type=i8] [debug line = 412:17]
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 25, i8 zeroext %p_trunc127_ext), !dbg !236 ; [debug line = 412:17]
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 27, i8 zeroext 48), !dbg !237 ; [debug line = 414:17]
  %tmp_35 = call fastcc zeroext i7 @motor_ctrl_bin2char(i4 zeroext %chR_pwm_8) nounwind, !dbg !238 ; [#uses=1 type=i7] [debug line = 415:17]
  %p_trunc129_ext = zext i7 %tmp_35 to i8, !dbg !238 ; [#uses=1 type=i8] [debug line = 415:17]
  call fastcc void @motor_ctrl_write_mem(i8 zeroext 28, i8 zeroext %p_trunc129_ext), !dbg !238 ; [debug line = 415:17]
  call fastcc void @motor_ctrl_wait_tmr(), !dbg !239 ; [debug line = 420:3]
  br label %.preheader, !dbg !240                 ; [debug line = 421:2]
}

; [#uses=2]
declare i8 @llvm.part.select.i8(i8, i32, i32) nounwind readnone

; [#uses=1]
declare i21 @llvm.part.select.i21(i21, i32, i32) nounwind readnone

; [#uses=1]
declare i12 @llvm.part.select.i12(i12, i32, i32) nounwind readnone

; [#uses=27]
declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

; [#uses=1]
define internal fastcc i21 @motor_ctrl_diff_angle(i16 zeroext %target, i16 %value) readnone {
  %value_read = call i16 @_ssdm_op_Read.ap_auto.i16(i16 %value) ; [#uses=2 type=i16]
  call void @llvm.dbg.value(metadata !{i16 %value_read}, i64 0, metadata !241), !dbg !247 ; [debug line = 133:40] [debug variable = value]
  %target_read = call i16 @_ssdm_op_Read.ap_auto.i16(i16 %target) ; [#uses=1 type=i16]
  call void @llvm.dbg.value(metadata !{i16 %target_read}, i64 0, metadata !248), !dbg !249 ; [debug line = 133:25] [debug variable = target]
  call void @llvm.dbg.value(metadata !{i16 %target}, i64 0, metadata !248), !dbg !249 ; [debug line = 133:25] [debug variable = target]
  call void @llvm.dbg.value(metadata !{i16 %value}, i64 0, metadata !241), !dbg !247 ; [debug line = 133:40] [debug variable = value]
  %tmp_cast3 = zext i16 %value_read to i18, !dbg !250 ; [#uses=2 type=i18] [debug line = 138:2]
  %tmp_cast = zext i16 %value_read to i17, !dbg !250 ; [#uses=2 type=i17] [debug line = 138:2]
  %tmp_cast_8 = zext i16 %target_read to i17, !dbg !250 ; [#uses=4 type=i17] [debug line = 138:2]
  %tmp_s = sub i17 %tmp_cast, %tmp_cast_8, !dbg !250 ; [#uses=1 type=i17] [debug line = 138:2]
  %tmp_53_cast = sext i17 %tmp_s to i19, !dbg !250 ; [#uses=1 type=i19] [debug line = 138:2]
  %tmp_36 = add i17 -1, %tmp_cast_8, !dbg !252    ; [#uses=1 type=i17] [debug line = 139:2]
  %tmp_37 = sub i17 %tmp_36, %tmp_cast, !dbg !252 ; [#uses=2 type=i17] [debug line = 139:2]
  %tmp_38 = icmp sgt i17 %tmp_37, -18001          ; [#uses=1 type=i1]
  %smax2 = select i1 %tmp_38, i17 %tmp_37, i17 -18001 ; [#uses=1 type=i17]
  %smax2_cast = sext i17 %smax2 to i18            ; [#uses=1 type=i18]
  %tmp_39 = sub i17 36000, %tmp_cast_8, !dbg !252 ; [#uses=1 type=i17] [debug line = 139:2]
  %tmp_57_cast_cast = sext i17 %tmp_39 to i19, !dbg !252 ; [#uses=1 type=i19] [debug line = 139:2]
  %tmp1 = add i18 %smax2_cast, %tmp_cast3, !dbg !252 ; [#uses=1 type=i18] [debug line = 139:2]
  %tmp1_cast_cast = sext i18 %tmp1 to i19, !dbg !252 ; [#uses=1 type=i19] [debug line = 139:2]
  %tmp_40 = add i19 %tmp1_cast_cast, %tmp_57_cast_cast, !dbg !252 ; [#uses=2 type=i19] [debug line = 139:2]
  %tmp_41 = urem i19 %tmp_40, 36000, !dbg !252    ; [#uses=1 type=i19] [debug line = 139:2]
  %tmp_42 = sub i19 %tmp_40, %tmp_41, !dbg !253   ; [#uses=2 type=i19] [debug line = 141:2]
  %tmp_61_cast = sext i19 %tmp_42 to i20, !dbg !253 ; [#uses=1 type=i20] [debug line = 141:2]
  %tmp_43 = sub i19 %tmp_53_cast, %tmp_42, !dbg !253 ; [#uses=3 type=i19] [debug line = 141:2]
  %tmp_62_cast = sext i19 %tmp_43 to i20, !dbg !253 ; [#uses=1 type=i20] [debug line = 141:2]
  %tmp_44 = icmp sgt i19 %tmp_43, -18000          ; [#uses=1 type=i1]
  %smax1 = select i1 %tmp_44, i19 %tmp_43, i19 -18000 ; [#uses=1 type=i19]
  %smax1_cast = sext i19 %smax1 to i20            ; [#uses=1 type=i20]
  %tmp_45 = add i17 35999, %tmp_cast_8, !dbg !253 ; [#uses=1 type=i17] [debug line = 141:2]
  %tmp_64_cast = zext i17 %tmp_45 to i18, !dbg !253 ; [#uses=1 type=i18] [debug line = 141:2]
  %tmp_46 = sub i18 %tmp_64_cast, %tmp_cast3, !dbg !253 ; [#uses=1 type=i18] [debug line = 141:2]
  %tmp_65_cast_cast = sext i18 %tmp_46 to i21, !dbg !253 ; [#uses=1 type=i21] [debug line = 141:2]
  %tmp2 = add i20 %tmp_61_cast, %smax1_cast, !dbg !253 ; [#uses=1 type=i20] [debug line = 141:2]
  %tmp2_cast_cast = sext i20 %tmp2 to i21, !dbg !253 ; [#uses=1 type=i21] [debug line = 141:2]
  %tmp_47 = add i21 %tmp2_cast_cast, %tmp_65_cast_cast, !dbg !253 ; [#uses=2 type=i21] [debug line = 141:2]
  %tmp_48 = urem i21 %tmp_47, 36000, !dbg !253    ; [#uses=1 type=i21] [debug line = 141:2]
  %tmp = trunc i21 %tmp_48 to i20, !dbg !253      ; [#uses=1 type=i20] [debug line = 141:2]
  %tmp_49 = sub i20 %tmp_62_cast, %tmp, !dbg !253 ; [#uses=1 type=i20] [debug line = 141:2]
  %tmp_69_cast = sext i20 %tmp_49 to i21, !dbg !253 ; [#uses=1 type=i21] [debug line = 141:2]
  %retval = add i21 %tmp_69_cast, %tmp_47, !dbg !250 ; [#uses=1 type=i21] [debug line = 138:2]
  call void @llvm.dbg.value(metadata !{i21 %retval}, i64 0, metadata !254), !dbg !250 ; [debug line = 138:2] [debug variable = retval]
  ret i21 %retval, !dbg !255                      ; [debug line = 144:2]
}

; [#uses=14]
define internal fastcc zeroext i7 @motor_ctrl_bin2char(i4 zeroext %val) readnone {
  %val_read = call i4 @_ssdm_op_Read.ap_auto.i4(i4 %val) ; [#uses=1 type=i4]
  call void @llvm.dbg.value(metadata !{i4 %val_read}, i64 0, metadata !256), !dbg !262 ; [debug line = 148:22] [debug variable = val]
  call void @llvm.dbg.value(metadata !{i4 %val}, i64 0, metadata !256), !dbg !262 ; [debug line = 148:22] [debug variable = val]
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
  ], !dbg !263                                    ; [debug line = 153:2]

; <label>:1                                       ; preds = %0
  br label %._crit_edge, !dbg !265                ; [debug line = 155:24]

; <label>:2                                       ; preds = %0
  br label %._crit_edge, !dbg !267                ; [debug line = 156:24]

; <label>:3                                       ; preds = %0
  br label %._crit_edge, !dbg !268                ; [debug line = 157:24]

; <label>:4                                       ; preds = %0
  br label %._crit_edge, !dbg !269                ; [debug line = 158:24]

; <label>:5                                       ; preds = %0
  br label %._crit_edge, !dbg !270                ; [debug line = 159:24]

; <label>:6                                       ; preds = %0
  br label %._crit_edge, !dbg !271                ; [debug line = 160:24]

; <label>:7                                       ; preds = %0
  br label %._crit_edge, !dbg !272                ; [debug line = 161:24]

; <label>:8                                       ; preds = %0
  br label %._crit_edge, !dbg !273                ; [debug line = 162:24]

; <label>:9                                       ; preds = %0
  br label %._crit_edge, !dbg !274                ; [debug line = 163:24]

; <label>:10                                      ; preds = %0
  br label %._crit_edge, !dbg !275                ; [debug line = 164:25]

; <label>:11                                      ; preds = %0
  br label %._crit_edge, !dbg !276                ; [debug line = 165:25]

; <label>:12                                      ; preds = %0
  br label %._crit_edge, !dbg !277                ; [debug line = 166:25]

; <label>:13                                      ; preds = %0
  br label %._crit_edge, !dbg !278                ; [debug line = 167:25]

; <label>:14                                      ; preds = %0
  br label %._crit_edge, !dbg !279                ; [debug line = 168:25]

; <label>:15                                      ; preds = %0
  br label %._crit_edge, !dbg !280                ; [debug line = 170:2]

._crit_edge:                                      ; preds = %15, %14, %13, %12, %11, %10, %9, %8, %7, %6, %5, %4, %3, %2, %1, %0
  %retval = phi i7 [ -58, %15 ], [ -59, %14 ], [ -60, %13 ], [ -61, %12 ], [ -62, %11 ], [ -63, %10 ], [ 57, %9 ], [ 56, %8 ], [ 55, %7 ], [ 54, %6 ], [ 53, %5 ], [ 52, %4 ], [ 51, %3 ], [ 50, %2 ], [ 49, %1 ], [ 48, %0 ] ; [#uses=1 type=i7]
  ret i7 %retval, !dbg !281                       ; [debug line = 172:2]
}

; [#uses=9]
define weak void @_ssdm_op_Write.ap_none.volatile.i8P(i8*, i8) {
entry:
  store i8 %1, i8* %0
  ret void
}

; [#uses=17]
define weak void @_ssdm_op_Write.ap_none.volatile.i1P(i1*, i1) {
entry:
  store i1 %1, i1* %0
  ret void
}

; [#uses=18]
define weak void @_ssdm_op_Wait(...) nounwind {
entry:
  ret void
}

; [#uses=1]
define weak void @_ssdm_op_SpecTopModule(...) nounwind {
entry:
  ret void
}

; [#uses=2]
define weak i32 @_ssdm_op_SpecLoopTripCount(...) {
entry:
  ret i32 0
}

; [#uses=1]
define weak i32 @_ssdm_op_SpecLoopBegin(...) {
entry:
  ret i32 0
}

; [#uses=12]
define weak void @_ssdm_op_SpecInterface(...) nounwind {
entry:
  ret void
}

; [#uses=4]
define weak i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8*) {
entry:
  %empty = load i8* %0                            ; [#uses=1 type=i8]
  ret i8 %empty
}

; [#uses=12]
define weak i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1*) {
entry:
  %empty = load i1* %0                            ; [#uses=1 type=i1]
  ret i1 %empty
}

; [#uses=3]
define weak i8 @_ssdm_op_Read.ap_auto.i8(i8) {
entry:
  ret i8 %0
}

; [#uses=1]
define weak i4 @_ssdm_op_Read.ap_auto.i4(i4) {
entry:
  ret i4 %0
}

; [#uses=2]
define weak i16 @_ssdm_op_Read.ap_auto.i16(i16) {
entry:
  ret i16 %0
}

; [#uses=1]
define weak i5 @_ssdm_op_PartSelect.i5.i8.i32.i32(i8, i32, i32) nounwind readnone {
entry:
  %empty = call i8 @llvm.part.select.i8(i8 %0, i32 %1, i32 %2) ; [#uses=1 type=i8]
  %empty_9 = trunc i8 %empty to i5                ; [#uses=1 type=i5]
  ret i5 %empty_9
}

; [#uses=4]
define weak i4 @_ssdm_op_PartSelect.i4.i8.i32.i32(i8, i32, i32) nounwind readnone {
entry:
  %empty = call i8 @llvm.part.select.i8(i8 %0, i32 %1, i32 %2) ; [#uses=1 type=i8]
  %empty_10 = trunc i8 %empty to i4               ; [#uses=1 type=i4]
  ret i4 %empty_10
}

; [#uses=3]
define weak i4 @_ssdm_op_PartSelect.i4.i21.i32.i32(i21, i32, i32) nounwind readnone {
entry:
  %empty = call i21 @llvm.part.select.i21(i21 %0, i32 %1, i32 %2) ; [#uses=1 type=i21]
  %empty_11 = trunc i21 %empty to i4              ; [#uses=1 type=i4]
  ret i4 %empty_11
}

; [#uses=1]
define weak i4 @_ssdm_op_PartSelect.i4.i12.i32.i32(i12, i32, i32) nounwind readnone {
entry:
  %empty = call i12 @llvm.part.select.i12(i12 %0, i32 %1, i32 %2) ; [#uses=1 type=i12]
  %empty_12 = trunc i12 %empty to i4              ; [#uses=1 type=i4]
  ret i4 %empty_12
}

; [#uses=0]
declare i3 @_ssdm_op_PartSelect.i3.i8.i32.i32(i8, i32, i32) nounwind readnone

; [#uses=0]
declare i20 @_ssdm_op_PartSelect.i20.i21.i32.i32(i21, i32, i32) nounwind readnone

; [#uses=2]
define weak i16 @_ssdm_op_BitConcatenate.i16.i8.i8(i8, i8) nounwind readnone {
entry:
  %empty = zext i8 %0 to i16                      ; [#uses=1 type=i16]
  %empty_13 = zext i8 %1 to i16                   ; [#uses=1 type=i16]
  %empty_14 = shl i16 %empty, 8                   ; [#uses=1 type=i16]
  %empty_15 = or i16 %empty_14, %empty_13         ; [#uses=1 type=i16]
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
!62 = metadata !{i32 786689, metadata !63, metadata !"data", metadata !64, i32 33554517, metadata !67, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!63 = metadata !{i32 786478, i32 0, metadata !64, metadata !"write_mem", metadata !"write_mem", metadata !"", metadata !64, i32 85, metadata !65, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (i8, i8)* @motor_ctrl_write_mem, null, null, metadata !69, i32 86} ; [ DW_TAG_subprogram ]
!64 = metadata !{i32 786473, metadata !"motor_ctrl.c", metadata !"D:\5C21_streamer_car5_artix7\5Cfpga_arty", null} ; [ DW_TAG_file_type ]
!65 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !66, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!66 = metadata !{null, metadata !67, metadata !67}
!67 = metadata !{i32 786454, null, metadata !"uint8", metadata !64, i32 10, i64 0, i64 0, i64 0, i32 0, metadata !68} ; [ DW_TAG_typedef ]
!68 = metadata !{i32 786468, null, metadata !"uint8", null, i32 0, i64 8, i64 8, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!69 = metadata !{metadata !70}
!70 = metadata !{i32 786468}                      ; [ DW_TAG_base_type ]
!71 = metadata !{i32 85, i32 34, metadata !63, null}
!72 = metadata !{i32 786689, metadata !63, metadata !"addr", metadata !64, i32 16777301, metadata !67, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!73 = metadata !{i32 85, i32 22, metadata !63, null}
!74 = metadata !{i32 88, i32 2, metadata !75, null}
!75 = metadata !{i32 786443, metadata !63, i32 86, i32 1, metadata !64, i32 3} ; [ DW_TAG_lexical_block ]
!76 = metadata !{i32 89, i32 2, metadata !75, null}
!77 = metadata !{i32 90, i32 2, metadata !75, null}
!78 = metadata !{i32 91, i32 2, metadata !75, null}
!79 = metadata !{i32 92, i32 2, metadata !75, null}
!80 = metadata !{i32 94, i32 2, metadata !75, null}
!81 = metadata !{i32 95, i32 3, metadata !82, null}
!82 = metadata !{i32 786443, metadata !75, i32 94, i32 5, metadata !64, i32 4} ; [ DW_TAG_lexical_block ]
!83 = metadata !{i32 96, i32 3, metadata !82, null}
!84 = metadata !{i32 97, i32 3, metadata !82, null}
!85 = metadata !{i32 98, i32 2, metadata !82, null}
!86 = metadata !{i32 99, i32 2, metadata !75, null}
!87 = metadata !{i32 101, i32 2, metadata !75, null}
!88 = metadata !{i32 102, i32 2, metadata !75, null}
!89 = metadata !{i32 103, i32 2, metadata !75, null}
!90 = metadata !{i32 104, i32 2, metadata !75, null}
!91 = metadata !{i32 105, i32 1, metadata !75, null}
!92 = metadata !{i32 77, i32 2, metadata !93, null}
!93 = metadata !{i32 786443, metadata !94, i32 74, i32 1, metadata !64, i32 0} ; [ DW_TAG_lexical_block ]
!94 = metadata !{i32 786478, i32 0, metadata !64, metadata !"wait_tmr", metadata !"wait_tmr", metadata !"", metadata !64, i32 73, metadata !95, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, metadata !69, i32 74} ; [ DW_TAG_subprogram ]
!95 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !96, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!96 = metadata !{null, metadata !97}
!97 = metadata !{i32 786454, null, metadata !"uint32", metadata !64, i32 34, i64 0, i64 0, i64 0, i32 0, metadata !98} ; [ DW_TAG_typedef ]
!98 = metadata !{i32 786468, null, metadata !"uint32", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!99 = metadata !{i32 78, i32 7, metadata !100, null}
!100 = metadata !{i32 786443, metadata !93, i32 78, i32 2, metadata !64, i32 1} ; [ DW_TAG_lexical_block ]
!101 = metadata !{i32 78, i32 23, metadata !100, null}
!102 = metadata !{i32 79, i32 3, metadata !103, null}
!103 = metadata !{i32 786443, metadata !100, i32 78, i32 28, metadata !64, i32 2} ; [ DW_TAG_lexical_block ]
!104 = metadata !{i32 786688, metadata !93, metadata !"t", metadata !64, i32 76, metadata !97, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!105 = metadata !{i32 81, i32 2, metadata !93, null}
!106 = metadata !{i32 82, i32 1, metadata !93, null}
!107 = metadata !{i32 786689, metadata !108, metadata !"addr", metadata !64, i32 16777324, metadata !67, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!108 = metadata !{i32 786478, i32 0, metadata !64, metadata !"read_mem", metadata !"read_mem", metadata !"", metadata !64, i32 108, metadata !109, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i8 (i8)* @motor_ctrl_read_mem, null, null, metadata !69, i32 109} ; [ DW_TAG_subprogram ]
!109 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !110, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!110 = metadata !{metadata !67, metadata !67}
!111 = metadata !{i32 108, i32 22, metadata !108, null}
!112 = metadata !{i32 113, i32 2, metadata !113, null}
!113 = metadata !{i32 786443, metadata !108, i32 109, i32 1, metadata !64, i32 5} ; [ DW_TAG_lexical_block ]
!114 = metadata !{i32 114, i32 2, metadata !113, null}
!115 = metadata !{i32 115, i32 2, metadata !113, null}
!116 = metadata !{i32 116, i32 2, metadata !113, null}
!117 = metadata !{i32 118, i32 2, metadata !113, null}
!118 = metadata !{i32 119, i32 3, metadata !119, null}
!119 = metadata !{i32 786443, metadata !113, i32 118, i32 5, metadata !64, i32 6} ; [ DW_TAG_lexical_block ]
!120 = metadata !{i32 120, i32 3, metadata !119, null}
!121 = metadata !{i32 121, i32 3, metadata !119, null}
!122 = metadata !{i32 786688, metadata !113, metadata !"dt", metadata !64, i32 111, metadata !67, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!123 = metadata !{i32 122, i32 2, metadata !119, null}
!124 = metadata !{i32 123, i32 2, metadata !113, null}
!125 = metadata !{i32 125, i32 2, metadata !113, null}
!126 = metadata !{i32 126, i32 2, metadata !113, null}
!127 = metadata !{i32 127, i32 2, metadata !113, null}
!128 = metadata !{i32 129, i32 2, metadata !113, null}
!129 = metadata !{i32 178, i32 1, metadata !130, null}
!130 = metadata !{i32 786443, metadata !131, i32 177, i32 1, metadata !64, i32 10} ; [ DW_TAG_lexical_block ]
!131 = metadata !{i32 786478, i32 0, metadata !64, metadata !"motor_ctrl", metadata !"motor_ctrl", metadata !"", metadata !64, i32 176, metadata !132, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 false, void ()* @motor_ctrl, null, null, metadata !69, i32 177} ; [ DW_TAG_subprogram ]
!132 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !133, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!133 = metadata !{null}
!134 = metadata !{i32 179, i32 1, metadata !130, null}
!135 = metadata !{i32 181, i32 1, metadata !130, null}
!136 = metadata !{i32 182, i32 1, metadata !130, null}
!137 = metadata !{i32 183, i32 1, metadata !130, null}
!138 = metadata !{i32 184, i32 1, metadata !130, null}
!139 = metadata !{i32 186, i32 1, metadata !130, null}
!140 = metadata !{i32 187, i32 1, metadata !130, null}
!141 = metadata !{i32 188, i32 1, metadata !130, null}
!142 = metadata !{i32 189, i32 1, metadata !130, null}
!143 = metadata !{i32 190, i32 1, metadata !130, null}
!144 = metadata !{i32 191, i32 1, metadata !130, null}
!145 = metadata !{i32 192, i32 1, metadata !130, null}
!146 = metadata !{i32 207, i32 2, metadata !130, null}
!147 = metadata !{i32 208, i32 2, metadata !130, null}
!148 = metadata !{i32 209, i32 2, metadata !130, null}
!149 = metadata !{i32 210, i32 2, metadata !130, null}
!150 = metadata !{i32 211, i32 2, metadata !130, null}
!151 = metadata !{i32 218, i32 2, metadata !130, null}
!152 = metadata !{i32 220, i32 2, metadata !130, null}
!153 = metadata !{i32 221, i32 2, metadata !130, null}
!154 = metadata !{i32 222, i32 2, metadata !130, null}
!155 = metadata !{i32 223, i32 2, metadata !130, null}
!156 = metadata !{i32 224, i32 2, metadata !130, null}
!157 = metadata !{i32 226, i32 7, metadata !158, null}
!158 = metadata !{i32 786443, metadata !130, i32 226, i32 2, metadata !64, i32 11} ; [ DW_TAG_lexical_block ]
!159 = metadata !{i32 226, i32 22, metadata !158, null}
!160 = metadata !{i32 227, i32 3, metadata !161, null}
!161 = metadata !{i32 786443, metadata !158, i32 226, i32 27, metadata !64, i32 12} ; [ DW_TAG_lexical_block ]
!162 = metadata !{i32 786688, metadata !130, metadata !"i", metadata !64, i32 205, metadata !67, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!163 = metadata !{i32 251, i32 8, metadata !164, null}
!164 = metadata !{i32 786443, metadata !130, i32 246, i32 12, metadata !64, i32 13} ; [ DW_TAG_lexical_block ]
!165 = metadata !{i32 786688, metadata !130, metadata !"eh", metadata !64, i32 203, metadata !67, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!166 = metadata !{i32 252, i32 8, metadata !164, null}
!167 = metadata !{i32 786688, metadata !130, metadata !"el", metadata !64, i32 203, metadata !67, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!168 = metadata !{i32 786688, metadata !130, metadata !"et", metadata !64, i32 204, metadata !169, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!169 = metadata !{i32 786454, null, metadata !"uint16", metadata !64, i32 18, i64 0, i64 0, i64 0, i32 0, metadata !170} ; [ DW_TAG_typedef ]
!170 = metadata !{i32 786468, null, metadata !"uint16", null, i32 0, i64 16, i64 16, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!171 = metadata !{i32 253, i32 3, metadata !164, null}
!172 = metadata !{i32 255, i32 3, metadata !164, null}
!173 = metadata !{i32 256, i32 3, metadata !164, null}
!174 = metadata !{i32 256, i32 10, metadata !164, null}
!175 = metadata !{i32 258, i32 3, metadata !164, null}
!176 = metadata !{i32 259, i32 8, metadata !164, null}
!177 = metadata !{i32 260, i32 8, metadata !164, null}
!178 = metadata !{i32 786688, metadata !130, metadata !"e", metadata !64, i32 204, metadata !169, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!179 = metadata !{i32 261, i32 3, metadata !164, null}
!180 = metadata !{i32 262, i32 3, metadata !164, null}
!181 = metadata !{i32 265, i32 16, metadata !164, null}
!182 = metadata !{i32 266, i32 16, metadata !164, null}
!183 = metadata !{i32 267, i32 16, metadata !164, null}
!184 = metadata !{i32 268, i32 16, metadata !164, null}
!185 = metadata !{i32 270, i32 16, metadata !164, null}
!186 = metadata !{i32 271, i32 16, metadata !164, null}
!187 = metadata !{i32 272, i32 17, metadata !164, null}
!188 = metadata !{i32 273, i32 17, metadata !164, null}
!189 = metadata !{i32 281, i32 14, metadata !164, null}
!190 = metadata !{i32 284, i32 17, metadata !164, null}
!191 = metadata !{i32 285, i32 17, metadata !164, null}
!192 = metadata !{i32 286, i32 17, metadata !164, null}
!193 = metadata !{i32 287, i32 17, metadata !164, null}
!194 = metadata !{i32 289, i32 3, metadata !164, null}
!195 = metadata !{i32 291, i32 4, metadata !196, null}
!196 = metadata !{i32 786443, metadata !164, i32 289, i32 72, metadata !64, i32 14} ; [ DW_TAG_lexical_block ]
!197 = metadata !{i32 292, i32 3, metadata !196, null}
!198 = metadata !{i32 295, i32 4, metadata !199, null}
!199 = metadata !{i32 786443, metadata !164, i32 293, i32 8, metadata !64, i32 15} ; [ DW_TAG_lexical_block ]
!200 = metadata !{i32 298, i32 3, metadata !164, null}
!201 = metadata !{i32 300, i32 4, metadata !202, null}
!202 = metadata !{i32 786443, metadata !164, i32 298, i32 58, metadata !64, i32 16} ; [ DW_TAG_lexical_block ]
!203 = metadata !{i32 301, i32 3, metadata !202, null}
!204 = metadata !{i32 304, i32 4, metadata !205, null}
!205 = metadata !{i32 786443, metadata !164, i32 302, i32 8, metadata !64, i32 17} ; [ DW_TAG_lexical_block ]
!206 = metadata !{i32 308, i32 10, metadata !164, null}
!207 = metadata !{i32 786688, metadata !130, metadata !"mode", metadata !64, i32 202, metadata !67, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!208 = metadata !{i32 310, i32 3, metadata !164, null}
!209 = metadata !{i32 312, i32 3, metadata !164, null}
!210 = metadata !{i32 342, i32 4, metadata !211, null}
!211 = metadata !{i32 786443, metadata !164, i32 312, i32 24, metadata !64, i32 18} ; [ DW_TAG_lexical_block ]
!212 = metadata !{i32 321, i32 9, metadata !211, null}
!213 = metadata !{i32 347, i32 5, metadata !214, null}
!214 = metadata !{i32 786443, metadata !211, i32 346, i32 9, metadata !64, i32 24} ; [ DW_TAG_lexical_block ]
!215 = metadata !{i32 368, i32 5, metadata !216, null}
!216 = metadata !{i32 786443, metadata !211, i32 367, i32 9, metadata !64, i32 28} ; [ DW_TAG_lexical_block ]
!217 = metadata !{i32 390, i32 3, metadata !164, null}
!218 = metadata !{i32 391, i32 3, metadata !164, null}
!219 = metadata !{i32 786688, metadata !130, metadata !"mtr_pwm_cnt", metadata !64, i32 195, metadata !220, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!220 = metadata !{i32 786485, null, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !221} ; [ DW_TAG_volatile_type ]
!221 = metadata !{i32 786468, null, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!222 = metadata !{i32 392, i32 3, metadata !164, null}
!223 = metadata !{i32 393, i32 4, metadata !164, null}
!224 = metadata !{i32 395, i32 4, metadata !164, null}
!225 = metadata !{i32 397, i32 3, metadata !164, null}
!226 = metadata !{i32 398, i32 3, metadata !164, null}
!227 = metadata !{i32 399, i32 4, metadata !164, null}
!228 = metadata !{i32 401, i32 4, metadata !164, null}
!229 = metadata !{i32 402, i32 3, metadata !164, null}
!230 = metadata !{i32 405, i32 3, metadata !164, null}
!231 = metadata !{i32 406, i32 3, metadata !164, null}
!232 = metadata !{i32 407, i32 4, metadata !233, null}
!233 = metadata !{i32 786443, metadata !164, i32 406, i32 26, metadata !64, i32 31} ; [ DW_TAG_lexical_block ]
!234 = metadata !{i32 408, i32 3, metadata !233, null}
!235 = metadata !{i32 411, i32 17, metadata !164, null}
!236 = metadata !{i32 412, i32 17, metadata !164, null}
!237 = metadata !{i32 414, i32 17, metadata !164, null}
!238 = metadata !{i32 415, i32 17, metadata !164, null}
!239 = metadata !{i32 420, i32 3, metadata !164, null}
!240 = metadata !{i32 421, i32 2, metadata !164, null}
!241 = metadata !{i32 786689, metadata !242, metadata !"value", metadata !64, i32 33554565, metadata !169, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!242 = metadata !{i32 786478, i32 0, metadata !64, metadata !"diff_angle", metadata !"diff_angle", metadata !"", metadata !64, i32 133, metadata !243, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, metadata !69, i32 134} ; [ DW_TAG_subprogram ]
!243 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !244, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!244 = metadata !{metadata !245, metadata !169, metadata !169}
!245 = metadata !{i32 786454, null, metadata !"int32", metadata !64, i32 34, i64 0, i64 0, i64 0, i32 0, metadata !246} ; [ DW_TAG_typedef ]
!246 = metadata !{i32 786468, null, metadata !"int32", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!247 = metadata !{i32 133, i32 40, metadata !242, null}
!248 = metadata !{i32 786689, metadata !242, metadata !"target", metadata !64, i32 16777349, metadata !169, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!249 = metadata !{i32 133, i32 25, metadata !242, null}
!250 = metadata !{i32 138, i32 2, metadata !251, null}
!251 = metadata !{i32 786443, metadata !242, i32 134, i32 1, metadata !64, i32 7} ; [ DW_TAG_lexical_block ]
!252 = metadata !{i32 139, i32 2, metadata !251, null}
!253 = metadata !{i32 141, i32 2, metadata !251, null}
!254 = metadata !{i32 786688, metadata !251, metadata !"retval", metadata !64, i32 136, metadata !245, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!255 = metadata !{i32 144, i32 2, metadata !251, null}
!256 = metadata !{i32 786689, metadata !257, metadata !"val", metadata !64, i32 16777364, metadata !260, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!257 = metadata !{i32 786478, i32 0, metadata !64, metadata !"bin2char", metadata !"bin2char", metadata !"", metadata !64, i32 148, metadata !258, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, metadata !69, i32 149} ; [ DW_TAG_subprogram ]
!258 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !259, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!259 = metadata !{metadata !67, metadata !260}
!260 = metadata !{i32 786454, null, metadata !"uint4", metadata !64, i32 6, i64 0, i64 0, i64 0, i32 0, metadata !261} ; [ DW_TAG_typedef ]
!261 = metadata !{i32 786468, null, metadata !"uint4", null, i32 0, i64 4, i64 4, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!262 = metadata !{i32 148, i32 22, metadata !257, null}
!263 = metadata !{i32 153, i32 2, metadata !264, null}
!264 = metadata !{i32 786443, metadata !257, i32 149, i32 1, metadata !64, i32 8} ; [ DW_TAG_lexical_block ]
!265 = metadata !{i32 155, i32 24, metadata !266, null}
!266 = metadata !{i32 786443, metadata !264, i32 153, i32 15, metadata !64, i32 9} ; [ DW_TAG_lexical_block ]
!267 = metadata !{i32 156, i32 24, metadata !266, null}
!268 = metadata !{i32 157, i32 24, metadata !266, null}
!269 = metadata !{i32 158, i32 24, metadata !266, null}
!270 = metadata !{i32 159, i32 24, metadata !266, null}
!271 = metadata !{i32 160, i32 24, metadata !266, null}
!272 = metadata !{i32 161, i32 24, metadata !266, null}
!273 = metadata !{i32 162, i32 24, metadata !266, null}
!274 = metadata !{i32 163, i32 24, metadata !266, null}
!275 = metadata !{i32 164, i32 25, metadata !266, null}
!276 = metadata !{i32 165, i32 25, metadata !266, null}
!277 = metadata !{i32 166, i32 25, metadata !266, null}
!278 = metadata !{i32 167, i32 25, metadata !266, null}
!279 = metadata !{i32 168, i32 25, metadata !266, null}
!280 = metadata !{i32 170, i32 2, metadata !266, null}
!281 = metadata !{i32 172, i32 2, metadata !264, null}
