; ModuleID = 'D:/21_streamer_car5_artix7/fpga_arty/motor_ctrl/solution1/.autopilot/db/a.o.2.bc'
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
@.str1 = private unnamed_addr constant [8 x i8] c"ap_none\00", align 1 ; [#uses=12 type=[8 x i8]*]
@.str = private unnamed_addr constant [1 x i8] zeroinitializer, align 1 ; [#uses=48 type=[1 x i8]*]

; [#uses=24]
define internal fastcc void @write_mem(i8 zeroext %addr, i8 zeroext %data) nounwind uwtable {
  call void (...)* @_ssdm_SpecKeepAssert(i8 %addr) nounwind, !hlsrange !84
  call void @llvm.dbg.value(metadata !{i8 %addr}, i64 0, metadata !85), !dbg !91 ; [debug line = 85:22] [debug variable = addr]
  call void @llvm.dbg.value(metadata !{i8 %data}, i64 0, metadata !92), !dbg !93 ; [debug line = 85:34] [debug variable = data]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !94 ; [debug line = 88:2]
  store volatile i8 %addr, i8* @mem_addr, align 1, !dbg !96 ; [debug line = 89:2]
  store volatile i8 %data, i8* @mem_dout, align 1, !dbg !97 ; [debug line = 90:2]
  store volatile i1 false, i1* @mem_wreq, align 1, !dbg !98 ; [debug line = 91:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !99 ; [debug line = 92:2]
  br label %._crit_edge, !dbg !100                ; [debug line = 94:2]

._crit_edge:                                      ; preds = %._crit_edge, %0
  store volatile i8 %addr, i8* @mem_addr, align 1, !dbg !101 ; [debug line = 95:3]
  store volatile i8 %data, i8* @mem_dout, align 1, !dbg !103 ; [debug line = 96:3]
  store volatile i1 true, i1* @mem_wreq, align 1, !dbg !104 ; [debug line = 97:3]
  %mem_wack.load = load volatile i1* @mem_wack, align 1, !dbg !105 ; [#uses=1 type=i1] [debug line = 98:2]
  br i1 %mem_wack.load, label %1, label %._crit_edge, !dbg !105 ; [debug line = 98:2]

; <label>:1                                       ; preds = %._crit_edge
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !106 ; [debug line = 99:2]
  store volatile i8 %addr, i8* @mem_addr, align 1, !dbg !107 ; [debug line = 101:2]
  store volatile i8 %data, i8* @mem_dout, align 1, !dbg !108 ; [debug line = 102:2]
  store volatile i1 false, i1* @mem_wreq, align 1, !dbg !109 ; [debug line = 103:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !110 ; [debug line = 104:2]
  ret void, !dbg !111                             ; [debug line = 105:1]
}

; [#uses=1]
define internal fastcc void @wait_tmr() nounwind uwtable {
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !112 ; [debug line = 77:2]
  br label %1, !dbg !119                          ; [debug line = 78:7]

; <label>:1                                       ; preds = %3, %0
  %t = phi i17 [ 0, %0 ], [ %t.1, %3 ]            ; [#uses=2 type=i17]
  %exitcond = icmp eq i17 %t, -31072, !dbg !119   ; [#uses=1 type=i1] [debug line = 78:7]
  %2 = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 100000, i64 100000, i64 100000) nounwind ; [#uses=0 type=i32]
  br i1 %exitcond, label %4, label %3, !dbg !119  ; [debug line = 78:7]

; <label>:3                                       ; preds = %1
  %dummy_tmr_out.load = load volatile i1* @dummy_tmr_out, align 1, !dbg !121 ; [#uses=1 type=i1] [debug line = 79:3]
  %not. = xor i1 %dummy_tmr_out.load, true, !dbg !121 ; [#uses=1 type=i1] [debug line = 79:3]
  store volatile i1 %not., i1* @dummy_tmr_out, align 1, !dbg !121 ; [debug line = 79:3]
  %t.1 = add i17 %t, 1, !dbg !123                 ; [#uses=1 type=i17] [debug line = 78:23]
  call void @llvm.dbg.value(metadata !{i17 %t.1}, i64 0, metadata !124), !dbg !123 ; [debug line = 78:23] [debug variable = t]
  br label %1, !dbg !123                          ; [debug line = 78:23]

; <label>:4                                       ; preds = %1
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !125 ; [debug line = 81:2]
  ret void, !dbg !126                             ; [debug line = 82:1]
}

; [#uses=6]
define internal fastcc zeroext i8 @read_mem(i8 zeroext %addr) nounwind uwtable {
  call void (...)* @_ssdm_SpecKeepAssert(i8 %addr) nounwind, !hlsrange !127
  call void @llvm.dbg.value(metadata !{i8 %addr}, i64 0, metadata !128), !dbg !132 ; [debug line = 108:22] [debug variable = addr]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !133 ; [debug line = 113:2]
  store volatile i8 %addr, i8* @mem_addr, align 1, !dbg !135 ; [debug line = 114:2]
  store volatile i1 true, i1* @mem_rreq, align 1, !dbg !136 ; [debug line = 115:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !137 ; [debug line = 116:2]
  br label %._crit_edge, !dbg !138                ; [debug line = 118:2]

._crit_edge:                                      ; preds = %._crit_edge, %0
  store volatile i8 %addr, i8* @mem_addr, align 1, !dbg !139 ; [debug line = 119:3]
  store volatile i1 true, i1* @mem_rreq, align 1, !dbg !141 ; [debug line = 120:3]
  %dt = load volatile i8* @mem_din, align 1, !dbg !142 ; [#uses=1 type=i8] [debug line = 121:3]
  call void @llvm.dbg.value(metadata !{i8 %dt}, i64 0, metadata !143), !dbg !142 ; [debug line = 121:3] [debug variable = dt]
  %mem_rack.load = load volatile i1* @mem_rack, align 1, !dbg !144 ; [#uses=1 type=i1] [debug line = 122:2]
  br i1 %mem_rack.load, label %1, label %._crit_edge, !dbg !144 ; [debug line = 122:2]

; <label>:1                                       ; preds = %._crit_edge
  %dt.lcssa = phi i8 [ %dt, %._crit_edge ]        ; [#uses=1 type=i8]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !145 ; [debug line = 123:2]
  store volatile i8 %addr, i8* @mem_addr, align 1, !dbg !146 ; [debug line = 125:2]
  store volatile i1 false, i1* @mem_rreq, align 1, !dbg !147 ; [debug line = 126:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !148 ; [debug line = 127:2]
  ret i8 %dt.lcssa, !dbg !149                     ; [debug line = 129:2]
}

; [#uses=0]
define void @motor_ctrl() noreturn nounwind uwtable {
  %mtr_pwm_cnt = alloca i32, align 4              ; [#uses=7 type=i32*]
  call void (...)* @_ssdm_op_SpecTopModule(), !dbg !150 ; [debug line = 178:1]
  %dummy_tmr_out.load = load volatile i1* @dummy_tmr_out, align 1, !dbg !155 ; [#uses=0 type=i1] [debug line = 179:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @dummy_tmr_out, [8 x i8]* @.str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !155 ; [debug line = 179:1]
  %l_dir.load = load volatile i1* @l_dir, align 1, !dbg !156 ; [#uses=0 type=i1] [debug line = 181:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @l_dir, [8 x i8]* @.str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !156 ; [debug line = 181:1]
  %l_pwm.load = load volatile i1* @l_pwm, align 1, !dbg !157 ; [#uses=0 type=i1] [debug line = 182:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @l_pwm, [8 x i8]* @.str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !157 ; [debug line = 182:1]
  %r_dir.load = load volatile i1* @r_dir, align 1, !dbg !158 ; [#uses=0 type=i1] [debug line = 183:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @r_dir, [8 x i8]* @.str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !158 ; [debug line = 183:1]
  %r_pwm.load = load volatile i1* @r_pwm, align 1, !dbg !159 ; [#uses=0 type=i1] [debug line = 184:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @r_pwm, [8 x i8]* @.str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !159 ; [debug line = 184:1]
  %mem_addr.load = load volatile i8* @mem_addr, align 1, !dbg !160 ; [#uses=0 type=i8] [debug line = 186:1]
  call void (...)* @_ssdm_op_SpecInterface(i8* @mem_addr, [8 x i8]* @.str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !160 ; [debug line = 186:1]
  %mem_din.load = load volatile i8* @mem_din, align 1, !dbg !161 ; [#uses=0 type=i8] [debug line = 187:1]
  call void (...)* @_ssdm_op_SpecInterface(i8* @mem_din, [8 x i8]* @.str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !161 ; [debug line = 187:1]
  %mem_dout.load = load volatile i8* @mem_dout, align 1, !dbg !162 ; [#uses=0 type=i8] [debug line = 188:1]
  call void (...)* @_ssdm_op_SpecInterface(i8* @mem_dout, [8 x i8]* @.str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !162 ; [debug line = 188:1]
  %mem_wreq.load = load volatile i1* @mem_wreq, align 1, !dbg !163 ; [#uses=0 type=i1] [debug line = 189:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @mem_wreq, [8 x i8]* @.str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !163 ; [debug line = 189:1]
  %mem_wack.load = load volatile i1* @mem_wack, align 1, !dbg !164 ; [#uses=0 type=i1] [debug line = 190:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @mem_wack, [8 x i8]* @.str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !164 ; [debug line = 190:1]
  %mem_rreq.load = load volatile i1* @mem_rreq, align 1, !dbg !165 ; [#uses=0 type=i1] [debug line = 191:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @mem_rreq, [8 x i8]* @.str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !165 ; [debug line = 191:1]
  %mem_rack.load = load volatile i1* @mem_rack, align 1, !dbg !166 ; [#uses=0 type=i1] [debug line = 192:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @mem_rack, [8 x i8]* @.str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !166 ; [debug line = 192:1]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !167 ; [debug line = 207:2]
  store volatile i1 false, i1* @l_dir, align 1, !dbg !168 ; [debug line = 208:2]
  store volatile i1 false, i1* @l_pwm, align 1, !dbg !169 ; [debug line = 209:2]
  store volatile i1 false, i1* @r_dir, align 1, !dbg !170 ; [debug line = 210:2]
  store volatile i1 false, i1* @r_pwm, align 1, !dbg !171 ; [debug line = 211:2]
  store volatile i32 0, i32* %mtr_pwm_cnt, align 4, !dbg !172 ; [debug line = 218:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !173 ; [debug line = 220:2]
  call fastcc void @write_mem(i8 zeroext -128, i8 zeroext 0), !dbg !174 ; [debug line = 221:2]
  call fastcc void @write_mem(i8 zeroext -123, i8 zeroext 0), !dbg !175 ; [debug line = 222:2]
  call fastcc void @write_mem(i8 zeroext -122, i8 zeroext 0), !dbg !176 ; [debug line = 223:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !177 ; [debug line = 224:2]
  br label %1, !dbg !178                          ; [debug line = 226:7]

; <label>:1                                       ; preds = %3, %0
  %i = phi i6 [ 0, %0 ], [ %i.1, %3 ]             ; [#uses=3 type=i6]
  %i.cast = zext i6 %i to i8, !dbg !178           ; [#uses=1 type=i8] [debug line = 226:7]
  %exitcond = icmp eq i6 %i, -32, !dbg !178       ; [#uses=1 type=i1] [debug line = 226:7]
  %2 = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 32, i64 32, i64 32) nounwind ; [#uses=0 type=i32]
  br i1 %exitcond, label %.preheader.preheader, label %3, !dbg !178 ; [debug line = 226:7]

.preheader.preheader:                             ; preds = %1
  br label %.preheader

; <label>:3                                       ; preds = %1
  call fastcc void @write_mem(i8 zeroext %i.cast, i8 zeroext 32), !dbg !180 ; [debug line = 227:3]
  %i.1 = add i6 %i, 1, !dbg !182                  ; [#uses=1 type=i6] [debug line = 226:22]
  call void @llvm.dbg.value(metadata !{i6 %i.1}, i64 0, metadata !183), !dbg !182 ; [debug line = 226:22] [debug variable = i]
  br label %1, !dbg !182                          ; [debug line = 226:22]

.preheader:                                       ; preds = %._crit_edge11, %.preheader.preheader
  %chR_dir = phi i1 [ %chR_dir.5, %._crit_edge11 ], [ false, %.preheader.preheader ] ; [#uses=3 type=i1]
  %chL_dir = phi i1 [ %chL_dir.5, %._crit_edge11 ], [ false, %.preheader.preheader ] ; [#uses=3 type=i1]
  %loop_begin = call i32 (...)* @_ssdm_op_SpecLoopBegin() nounwind ; [#uses=0 type=i32]
  %eh = call fastcc zeroext i8 @read_mem(i8 zeroext -127), !dbg !184 ; [#uses=3 type=i8] [debug line = 251:8]
  call void @llvm.dbg.value(metadata !{i8 %eh}, i64 0, metadata !186), !dbg !184 ; [debug line = 251:8] [debug variable = eh]
  %el = call fastcc zeroext i8 @read_mem(i8 zeroext -126), !dbg !187 ; [#uses=3 type=i8] [debug line = 252:8]
  call void @llvm.dbg.value(metadata !{i8 %el}, i64 0, metadata !188), !dbg !187 ; [debug line = 252:8] [debug variable = el]
  %et = call i16 @_ssdm_op_BitConcatenate.i16.i8.i8(i8 %eh, i8 %el) ; [#uses=1 type=i16]
  call void @llvm.dbg.value(metadata !{i16 %et}, i64 0, metadata !189), !dbg !192 ; [debug line = 253:3] [debug variable = et]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !193 ; [debug line = 255:3]
  br label %._crit_edge, !dbg !194                ; [debug line = 256:3]

._crit_edge:                                      ; preds = %._crit_edge, %.preheader
  %tmp. = call fastcc zeroext i8 @read_mem(i8 zeroext -123), !dbg !195 ; [#uses=1 type=i8] [debug line = 256:10]
  %tmp.1 = icmp eq i8 %tmp., 0, !dbg !195         ; [#uses=1 type=i1] [debug line = 256:10]
  br i1 %tmp.1, label %._crit_edge, label %4, !dbg !195 ; [debug line = 256:10]

; <label>:4                                       ; preds = %._crit_edge
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !196 ; [debug line = 258:3]
  %eh.1 = call fastcc zeroext i8 @read_mem(i8 zeroext -125), !dbg !197 ; [#uses=3 type=i8] [debug line = 259:8]
  call void @llvm.dbg.value(metadata !{i8 %eh.1}, i64 0, metadata !186), !dbg !197 ; [debug line = 259:8] [debug variable = eh]
  %el.1 = call fastcc zeroext i8 @read_mem(i8 zeroext -124), !dbg !198 ; [#uses=3 type=i8] [debug line = 260:8]
  call void @llvm.dbg.value(metadata !{i8 %el.1}, i64 0, metadata !188), !dbg !198 ; [debug line = 260:8] [debug variable = el]
  %e = call i16 @_ssdm_op_BitConcatenate.i16.i8.i8(i8 %eh.1, i8 %el.1) ; [#uses=1 type=i16]
  call void @llvm.dbg.value(metadata !{i16 %e}, i64 0, metadata !199), !dbg !200 ; [debug line = 261:3] [debug variable = e]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !201 ; [debug line = 262:3]
  %tmp.2 = lshr i8 %eh.1, 4, !dbg !202            ; [#uses=1 type=i8] [debug line = 265:16]
  %tmp.3 = trunc i8 %tmp.2 to i4, !dbg !202       ; [#uses=1 type=i4] [debug line = 265:16]
  %tmp.4 = call fastcc zeroext i7 @bin2char(i4 zeroext %tmp.3) nounwind, !dbg !202 ; [#uses=1 type=i7] [debug line = 265:16]
  %.trunc.ext = zext i7 %tmp.4 to i8, !dbg !202   ; [#uses=1 type=i8] [debug line = 265:16]
  call fastcc void @write_mem(i8 zeroext 3, i8 zeroext %.trunc.ext), !dbg !202 ; [debug line = 265:16]
  %tmp.5 = trunc i8 %eh.1 to i4, !dbg !203        ; [#uses=1 type=i4] [debug line = 266:16]
  %tmp.6 = call fastcc zeroext i7 @bin2char(i4 zeroext %tmp.5) nounwind, !dbg !203 ; [#uses=1 type=i7] [debug line = 266:16]
  %.trunc115.ext = zext i7 %tmp.6 to i8, !dbg !203 ; [#uses=1 type=i8] [debug line = 266:16]
  call fastcc void @write_mem(i8 zeroext 4, i8 zeroext %.trunc115.ext), !dbg !203 ; [debug line = 266:16]
  %tmp.7 = lshr i8 %el.1, 4, !dbg !204            ; [#uses=1 type=i8] [debug line = 267:16]
  %tmp.8 = trunc i8 %tmp.7 to i4, !dbg !204       ; [#uses=1 type=i4] [debug line = 267:16]
  %tmp.9 = call fastcc zeroext i7 @bin2char(i4 zeroext %tmp.8) nounwind, !dbg !204 ; [#uses=1 type=i7] [debug line = 267:16]
  %.trunc116.ext = zext i7 %tmp.9 to i8, !dbg !204 ; [#uses=1 type=i8] [debug line = 267:16]
  call fastcc void @write_mem(i8 zeroext 5, i8 zeroext %.trunc116.ext), !dbg !204 ; [debug line = 267:16]
  %tmp.10 = trunc i8 %el.1 to i4, !dbg !205       ; [#uses=1 type=i4] [debug line = 268:16]
  %tmp.11 = call fastcc zeroext i7 @bin2char(i4 zeroext %tmp.10) nounwind, !dbg !205 ; [#uses=1 type=i7] [debug line = 268:16]
  %.trunc117.ext = zext i7 %tmp.11 to i8, !dbg !205 ; [#uses=1 type=i8] [debug line = 268:16]
  call fastcc void @write_mem(i8 zeroext 6, i8 zeroext %.trunc117.ext), !dbg !205 ; [debug line = 268:16]
  %tmp.12 = lshr i8 %eh, 4, !dbg !206             ; [#uses=1 type=i8] [debug line = 270:16]
  %tmp.13 = trunc i8 %tmp.12 to i4, !dbg !206     ; [#uses=1 type=i4] [debug line = 270:16]
  %tmp.14 = call fastcc zeroext i7 @bin2char(i4 zeroext %tmp.13) nounwind, !dbg !206 ; [#uses=1 type=i7] [debug line = 270:16]
  %.trunc118.ext = zext i7 %tmp.14 to i8, !dbg !206 ; [#uses=1 type=i8] [debug line = 270:16]
  call fastcc void @write_mem(i8 zeroext 8, i8 zeroext %.trunc118.ext), !dbg !206 ; [debug line = 270:16]
  %tmp.15 = trunc i8 %eh to i4, !dbg !207         ; [#uses=1 type=i4] [debug line = 271:16]
  %tmp.16 = call fastcc zeroext i7 @bin2char(i4 zeroext %tmp.15) nounwind, !dbg !207 ; [#uses=1 type=i7] [debug line = 271:16]
  %.trunc119.ext = zext i7 %tmp.16 to i8, !dbg !207 ; [#uses=1 type=i8] [debug line = 271:16]
  call fastcc void @write_mem(i8 zeroext 9, i8 zeroext %.trunc119.ext), !dbg !207 ; [debug line = 271:16]
  %tmp.17 = lshr i8 %el, 4, !dbg !208             ; [#uses=1 type=i8] [debug line = 272:17]
  %tmp.18 = trunc i8 %tmp.17 to i4, !dbg !208     ; [#uses=1 type=i4] [debug line = 272:17]
  %tmp.19 = call fastcc zeroext i7 @bin2char(i4 zeroext %tmp.18) nounwind, !dbg !208 ; [#uses=1 type=i7] [debug line = 272:17]
  %.trunc120.ext = zext i7 %tmp.19 to i8, !dbg !208 ; [#uses=1 type=i8] [debug line = 272:17]
  call fastcc void @write_mem(i8 zeroext 10, i8 zeroext %.trunc120.ext), !dbg !208 ; [debug line = 272:17]
  %tmp.20 = trunc i8 %el to i4, !dbg !209         ; [#uses=1 type=i4] [debug line = 273:17]
  %tmp.21 = call fastcc zeroext i7 @bin2char(i4 zeroext %tmp.20) nounwind, !dbg !209 ; [#uses=1 type=i7] [debug line = 273:17]
  %.trunc121.ext = zext i7 %tmp.21 to i8, !dbg !209 ; [#uses=1 type=i8] [debug line = 273:17]
  call fastcc void @write_mem(i8 zeroext 11, i8 zeroext %.trunc121.ext), !dbg !209 ; [debug line = 273:17]
  %diff_agl = call fastcc i21 @diff_angle(i16 zeroext %et, i16 zeroext %e) nounwind, !dbg !210 ; [#uses=6 type=i21] [debug line = 281:14]
  %diff_agl.cast1 = trunc i21 %diff_agl to i8, !dbg !210 ; [#uses=1 type=i8] [debug line = 281:14]
  %diff_agl.cast2 = trunc i21 %diff_agl to i12, !dbg !210 ; [#uses=1 type=i12] [debug line = 281:14]
  %diff_agl.cast = trunc i21 %diff_agl to i16, !dbg !210 ; [#uses=1 type=i16] [debug line = 281:14]
  %tmp.22 = lshr i16 %diff_agl.cast, 12, !dbg !211 ; [#uses=1 type=i16] [debug line = 284:17]
  %tmp.23 = trunc i16 %tmp.22 to i4, !dbg !211    ; [#uses=1 type=i4] [debug line = 284:17]
  %tmp.24 = call fastcc zeroext i7 @bin2char(i4 zeroext %tmp.23) nounwind, !dbg !211 ; [#uses=1 type=i7] [debug line = 284:17]
  %.trunc122.ext = zext i7 %tmp.24 to i8, !dbg !211 ; [#uses=1 type=i8] [debug line = 284:17]
  call fastcc void @write_mem(i8 zeroext 16, i8 zeroext %.trunc122.ext), !dbg !211 ; [debug line = 284:17]
  %tmp.25 = lshr i12 %diff_agl.cast2, 8, !dbg !212 ; [#uses=1 type=i12] [debug line = 285:17]
  %tmp.26 = trunc i12 %tmp.25 to i4, !dbg !212    ; [#uses=1 type=i4] [debug line = 285:17]
  %tmp.27 = call fastcc zeroext i7 @bin2char(i4 zeroext %tmp.26) nounwind, !dbg !212 ; [#uses=1 type=i7] [debug line = 285:17]
  %.trunc123.ext = zext i7 %tmp.27 to i8, !dbg !212 ; [#uses=1 type=i8] [debug line = 285:17]
  call fastcc void @write_mem(i8 zeroext 17, i8 zeroext %.trunc123.ext), !dbg !212 ; [debug line = 285:17]
  %tmp.28 = lshr i8 %diff_agl.cast1, 4, !dbg !213 ; [#uses=1 type=i8] [debug line = 286:17]
  %tmp.29 = trunc i8 %tmp.28 to i4, !dbg !213     ; [#uses=1 type=i4] [debug line = 286:17]
  %tmp.30 = call fastcc zeroext i7 @bin2char(i4 zeroext %tmp.29) nounwind, !dbg !213 ; [#uses=1 type=i7] [debug line = 286:17]
  %.trunc124.ext = zext i7 %tmp.30 to i8, !dbg !213 ; [#uses=1 type=i8] [debug line = 286:17]
  call fastcc void @write_mem(i8 zeroext 18, i8 zeroext %.trunc124.ext), !dbg !213 ; [debug line = 286:17]
  %tmp.31 = trunc i21 %diff_agl to i4, !dbg !214  ; [#uses=1 type=i4] [debug line = 287:17]
  %tmp.32 = call fastcc zeroext i7 @bin2char(i4 zeroext %tmp.31) nounwind, !dbg !214 ; [#uses=1 type=i7] [debug line = 287:17]
  %.trunc125.ext = zext i7 %tmp.32 to i8, !dbg !214 ; [#uses=1 type=i8] [debug line = 287:17]
  call fastcc void @write_mem(i8 zeroext 19, i8 zeroext %.trunc125.ext), !dbg !214 ; [debug line = 287:17]
  %tmp.33 = icmp slt i21 %diff_agl, -249, !dbg !215 ; [#uses=1 type=i1] [debug line = 289:3]
  br i1 %tmp.33, label %5, label %6, !dbg !215    ; [debug line = 289:3]

; <label>:5                                       ; preds = %4
  call fastcc void @write_mem(i8 zeroext 21, i8 zeroext 76), !dbg !216 ; [debug line = 291:4]
  br label %7, !dbg !218                          ; [debug line = 292:3]

; <label>:6                                       ; preds = %4
  call fastcc void @write_mem(i8 zeroext 21, i8 zeroext 45), !dbg !219 ; [debug line = 295:4]
  br label %7

; <label>:7                                       ; preds = %6, %5
  %too_left = phi i1 [ true, %5 ], [ false, %6 ]  ; [#uses=3 type=i1]
  %tmp.34 = icmp sgt i21 %diff_agl, 249, !dbg !221 ; [#uses=1 type=i1] [debug line = 298:3]
  br i1 %tmp.34, label %8, label %9, !dbg !221    ; [debug line = 298:3]

; <label>:8                                       ; preds = %7
  call fastcc void @write_mem(i8 zeroext 22, i8 zeroext 82), !dbg !222 ; [debug line = 300:4]
  br label %_ifconv, !dbg !224                    ; [debug line = 301:3]

; <label>:9                                       ; preds = %7
  call fastcc void @write_mem(i8 zeroext 22, i8 zeroext 45), !dbg !225 ; [debug line = 304:4]
  br label %_ifconv

_ifconv:                                          ; preds = %9, %8
  %too_right = phi i1 [ true, %8 ], [ false, %9 ] ; [#uses=9 type=i1]
  %mode = call fastcc zeroext i8 @read_mem(i8 zeroext -128), !dbg !227 ; [#uses=2 type=i8] [debug line = 308:10]
  call void @llvm.dbg.value(metadata !{i8 %mode}, i64 0, metadata !228), !dbg !227 ; [debug line = 308:10] [debug variable = mode]
  %tmp.35 = lshr i8 %mode, 3, !dbg !229           ; [#uses=1 type=i8] [debug line = 310:3]
  %tmp.35.cast = trunc i8 %tmp.35 to i5, !dbg !229 ; [#uses=1 type=i5] [debug line = 310:3]
  %chR_pwm = udiv i5 %tmp.35.cast, 3, !dbg !229   ; [#uses=1 type=i5] [debug line = 310:3]
  %chR_pwm.cast = trunc i5 %chR_pwm to i4, !dbg !229 ; [#uses=2 type=i4] [debug line = 310:3]
  call void @llvm.dbg.value(metadata !{i5 %chR_pwm}, i64 0, metadata !230), !dbg !229 ; [debug line = 310:3] [debug variable = chR_pwm]
  call void @llvm.dbg.value(metadata !{i5 %chR_pwm}, i64 0, metadata !231), !dbg !229 ; [debug line = 310:3] [debug variable = chL_pwm]
  %tmp.36.cast = trunc i8 %mode to i3, !dbg !232  ; [#uses=5 type=i3] [debug line = 312:3]
  %brmerge = or i1 %too_right, %too_left, !dbg !233 ; [#uses=3 type=i1] [debug line = 342:4]
  %. = select i1 %too_left, i4 0, i4 %chR_pwm.cast, !dbg !235 ; [#uses=2 type=i4] [debug line = 321:9]
  %.1 = select i1 %too_left, i4 -6, i4 %chR_pwm.cast, !dbg !235 ; [#uses=2 type=i4] [debug line = 321:9]
  %chL_pwm.4 = select i1 %too_right, i4 0, i4 -6, !dbg !236 ; [#uses=1 type=i4] [debug line = 347:5]
  %not.too_right. = xor i1 %too_right, true, !dbg !236 ; [#uses=4 type=i1] [debug line = 347:5]
  %chR_pwm.6 = select i1 %too_right, i4 -6, i4 0, !dbg !238 ; [#uses=1 type=i4] [debug line = 368:5]
  %sel_tmp = icmp eq i3 %tmp.36.cast, 1           ; [#uses=2 type=i1]
  %sel_tmp2 = and i1 %sel_tmp, %not.too_right.    ; [#uses=2 type=i1]
  %sel_tmp3 = select i1 %sel_tmp2, i4 %., i4 0    ; [#uses=1 type=i4]
  %sel_tmp4 = icmp eq i3 %tmp.36.cast, 3          ; [#uses=2 type=i1]
  %sel_tmp6 = and i1 %sel_tmp4, %not.too_right.   ; [#uses=4 type=i1]
  %sel_tmp7 = select i1 %sel_tmp6, i4 %.1, i4 %sel_tmp3 ; [#uses=1 type=i4]
  %sel_tmp8 = icmp eq i3 %tmp.36.cast, -3         ; [#uses=2 type=i1]
  %sel_tmp9 = and i1 %sel_tmp8, %brmerge          ; [#uses=4 type=i1]
  %sel_tmp1 = select i1 %sel_tmp9, i4 -6, i4 %sel_tmp7 ; [#uses=1 type=i4]
  %sel_tmp5 = icmp eq i3 %tmp.36.cast, -1         ; [#uses=2 type=i1]
  %sel_tmp10 = and i1 %sel_tmp5, %brmerge         ; [#uses=4 type=i1]
  %sel_tmp11 = select i1 %sel_tmp10, i4 %chR_pwm.6, i4 %sel_tmp1 ; [#uses=1 type=i4]
  %sel_tmp12 = and i1 %sel_tmp, %too_right        ; [#uses=3 type=i1]
  %sel_tmp13 = and i1 %sel_tmp4, %too_right       ; [#uses=6 type=i1]
  %sel_tmp14 = select i1 %sel_tmp13, i4 0, i4 -6  ; [#uses=1 type=i4]
  %tmp = or i1 %sel_tmp13, %sel_tmp12             ; [#uses=1 type=i1]
  %sel_tmp15 = select i1 %tmp, i4 %sel_tmp14, i4 %sel_tmp11 ; [#uses=1 type=i4]
  %sel_tmp16 = xor i1 %brmerge, true, !dbg !233   ; [#uses=2 type=i1] [debug line = 342:4]
  %sel_tmp17 = and i1 %sel_tmp8, %sel_tmp16       ; [#uses=4 type=i1]
  %sel_tmp18 = and i1 %sel_tmp5, %sel_tmp16       ; [#uses=4 type=i1]
  %tmp.36 = or i1 %sel_tmp18, %sel_tmp17          ; [#uses=1 type=i1]
  %chR_pwm.8 = select i1 %tmp.36, i4 0, i4 %sel_tmp15 ; [#uses=2 type=i4]
  %sel_tmp19 = select i1 %sel_tmp2, i4 %.1, i4 0  ; [#uses=1 type=i4]
  %sel_tmp20 = select i1 %sel_tmp6, i4 %., i4 %sel_tmp19 ; [#uses=1 type=i4]
  %sel_tmp21 = select i1 %sel_tmp9, i4 %chL_pwm.4, i4 %sel_tmp20 ; [#uses=1 type=i4]
  %sel_tmp22 = select i1 %sel_tmp10, i4 -6, i4 %sel_tmp21 ; [#uses=1 type=i4]
  %sel_tmp23 = select i1 %sel_tmp13, i4 -6, i4 0  ; [#uses=1 type=i4]
  %tmp.37 = or i1 %sel_tmp13, %sel_tmp12          ; [#uses=1 type=i1]
  %sel_tmp24 = select i1 %tmp.37, i4 %sel_tmp23, i4 %sel_tmp22 ; [#uses=1 type=i4]
  %tmp.43 = or i1 %sel_tmp18, %sel_tmp17          ; [#uses=1 type=i1]
  %chL_pwm.8 = select i1 %tmp.43, i4 0, i4 %sel_tmp24 ; [#uses=2 type=i4]
  %sel_tmp56.not = icmp ne i3 %tmp.36.cast, 1     ; [#uses=1 type=i1]
  %not.sel_tmp = or i1 %too_right, %sel_tmp56.not ; [#uses=2 type=i1]
  %sel_tmp25 = and i1 %chR_dir, %not.sel_tmp      ; [#uses=1 type=i1]
  %sel_tmp26 = or i1 %sel_tmp6, %sel_tmp25        ; [#uses=1 type=i1]
  %sel_tmp27 = select i1 %sel_tmp9, i1 %not.too_right., i1 %sel_tmp26 ; [#uses=1 type=i1]
  %sel_tmp28 = select i1 %sel_tmp10, i1 %not.too_right., i1 %sel_tmp27 ; [#uses=1 type=i1]
  %not.sel_tmp1 = xor i1 %sel_tmp12, true         ; [#uses=2 type=i1]
  %sel_tmp29 = and i1 %sel_tmp28, %not.sel_tmp1   ; [#uses=1 type=i1]
  %sel_tmp30 = or i1 %sel_tmp13, %sel_tmp29       ; [#uses=1 type=i1]
  %sel_tmp31 = select i1 %sel_tmp17, i1 %chR_dir, i1 %sel_tmp30 ; [#uses=1 type=i1]
  %chR_dir.5 = select i1 %sel_tmp18, i1 %chR_dir, i1 %sel_tmp31 ; [#uses=2 type=i1]
  %sel_tmp32 = and i1 %chL_dir, %not.sel_tmp      ; [#uses=1 type=i1]
  %sel_tmp33 = or i1 %sel_tmp6, %sel_tmp32        ; [#uses=1 type=i1]
  %sel_tmp34 = select i1 %sel_tmp9, i1 %too_right, i1 %sel_tmp33 ; [#uses=1 type=i1]
  %sel_tmp35 = select i1 %sel_tmp10, i1 %too_right, i1 %sel_tmp34 ; [#uses=1 type=i1]
  %sel_tmp36 = and i1 %sel_tmp35, %not.sel_tmp1   ; [#uses=1 type=i1]
  %sel_tmp37 = or i1 %sel_tmp13, %sel_tmp36       ; [#uses=1 type=i1]
  %sel_tmp38 = select i1 %sel_tmp17, i1 %chL_dir, i1 %sel_tmp37 ; [#uses=1 type=i1]
  %chL_dir.5 = select i1 %sel_tmp18, i1 %chL_dir, i1 %sel_tmp38 ; [#uses=2 type=i1]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !240 ; [debug line = 390:3]
  store volatile i1 %chR_dir.5, i1* @r_dir, align 1, !dbg !241 ; [debug line = 391:3]
  call void @llvm.dbg.value(metadata !{i32* %mtr_pwm_cnt}, i64 0, metadata !242), !dbg !245 ; [debug line = 392:3] [debug variable = mtr_pwm_cnt]
  %mtr_pwm_cnt.load = load volatile i32* %mtr_pwm_cnt, align 4, !dbg !245 ; [#uses=1 type=i32] [debug line = 392:3]
  %tmp.38 = zext i4 %chR_pwm.8 to i32, !dbg !245  ; [#uses=1 type=i32] [debug line = 392:3]
  %tmp.39 = icmp slt i32 %mtr_pwm_cnt.load, %tmp.38, !dbg !245 ; [#uses=1 type=i1] [debug line = 392:3]
  br i1 %tmp.39, label %10, label %11, !dbg !245  ; [debug line = 392:3]

; <label>:10                                      ; preds = %_ifconv
  store volatile i1 true, i1* @r_pwm, align 1, !dbg !246 ; [debug line = 393:4]
  br label %12, !dbg !246                         ; [debug line = 393:4]

; <label>:11                                      ; preds = %_ifconv
  store volatile i1 false, i1* @r_pwm, align 1, !dbg !247 ; [debug line = 395:4]
  br label %12

; <label>:12                                      ; preds = %11, %10
  store volatile i1 %chL_dir.5, i1* @l_dir, align 1, !dbg !248 ; [debug line = 397:3]
  call void @llvm.dbg.value(metadata !{i32* %mtr_pwm_cnt}, i64 0, metadata !242), !dbg !249 ; [debug line = 398:3] [debug variable = mtr_pwm_cnt]
  %mtr_pwm_cnt.load.1 = load volatile i32* %mtr_pwm_cnt, align 4, !dbg !249 ; [#uses=1 type=i32] [debug line = 398:3]
  %tmp.40 = zext i4 %chL_pwm.8 to i32, !dbg !249  ; [#uses=1 type=i32] [debug line = 398:3]
  %tmp.41 = icmp slt i32 %mtr_pwm_cnt.load.1, %tmp.40, !dbg !249 ; [#uses=1 type=i1] [debug line = 398:3]
  br i1 %tmp.41, label %13, label %14, !dbg !249  ; [debug line = 398:3]

; <label>:13                                      ; preds = %12
  store volatile i1 true, i1* @l_pwm, align 1, !dbg !250 ; [debug line = 399:4]
  br label %15, !dbg !250                         ; [debug line = 399:4]

; <label>:14                                      ; preds = %12
  store volatile i1 false, i1* @l_pwm, align 1, !dbg !251 ; [debug line = 401:4]
  br label %15

; <label>:15                                      ; preds = %14, %13
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !252 ; [debug line = 402:3]
  call void @llvm.dbg.value(metadata !{i32* %mtr_pwm_cnt}, i64 0, metadata !242), !dbg !253 ; [debug line = 405:3] [debug variable = mtr_pwm_cnt]
  %mtr_pwm_cnt.load.2 = load volatile i32* %mtr_pwm_cnt, align 4, !dbg !253 ; [#uses=1 type=i32] [debug line = 405:3]
  %mtr_pwm_cnt.1 = add nsw i32 %mtr_pwm_cnt.load.2, 1, !dbg !253 ; [#uses=1 type=i32] [debug line = 405:3]
  call void @llvm.dbg.value(metadata !{i32 %mtr_pwm_cnt.1}, i64 0, metadata !242), !dbg !253 ; [debug line = 405:3] [debug variable = mtr_pwm_cnt]
  store volatile i32 %mtr_pwm_cnt.1, i32* %mtr_pwm_cnt, align 4, !dbg !253 ; [debug line = 405:3]
  %mtr_pwm_cnt.load.3 = load volatile i32* %mtr_pwm_cnt, align 4, !dbg !254 ; [#uses=1 type=i32] [debug line = 406:3]
  %tmp.42 = icmp sgt i32 %mtr_pwm_cnt.load.3, 9, !dbg !254 ; [#uses=1 type=i1] [debug line = 406:3]
  br i1 %tmp.42, label %16, label %._crit_edge11, !dbg !254 ; [debug line = 406:3]

; <label>:16                                      ; preds = %15
  store volatile i32 0, i32* %mtr_pwm_cnt, align 4, !dbg !255 ; [debug line = 407:4]
  br label %._crit_edge11, !dbg !257              ; [debug line = 408:3]

._crit_edge11:                                    ; preds = %16, %15
  %tmp.44 = call fastcc zeroext i7 @bin2char(i4 zeroext 0) nounwind, !dbg !258 ; [#uses=1 type=i7] [debug line = 411:17]
  %.trunc126.ext = zext i7 %tmp.44 to i8, !dbg !258 ; [#uses=1 type=i8] [debug line = 411:17]
  call fastcc void @write_mem(i8 zeroext 24, i8 zeroext %.trunc126.ext), !dbg !258 ; [debug line = 411:17]
  %tmp.45 = call fastcc zeroext i7 @bin2char(i4 zeroext %chL_pwm.8) nounwind, !dbg !259 ; [#uses=1 type=i7] [debug line = 412:17]
  %.trunc127.ext = zext i7 %tmp.45 to i8, !dbg !259 ; [#uses=1 type=i8] [debug line = 412:17]
  call fastcc void @write_mem(i8 zeroext 25, i8 zeroext %.trunc127.ext), !dbg !259 ; [debug line = 412:17]
  %tmp.46 = call fastcc zeroext i7 @bin2char(i4 zeroext 0) nounwind, !dbg !260 ; [#uses=1 type=i7] [debug line = 414:17]
  %.trunc128.ext = zext i7 %tmp.46 to i8, !dbg !260 ; [#uses=1 type=i8] [debug line = 414:17]
  call fastcc void @write_mem(i8 zeroext 27, i8 zeroext %.trunc128.ext), !dbg !260 ; [debug line = 414:17]
  %tmp.47 = call fastcc zeroext i7 @bin2char(i4 zeroext %chR_pwm.8) nounwind, !dbg !261 ; [#uses=1 type=i7] [debug line = 415:17]
  %.trunc129.ext = zext i7 %tmp.47 to i8, !dbg !261 ; [#uses=1 type=i8] [debug line = 415:17]
  call fastcc void @write_mem(i8 zeroext 28, i8 zeroext %.trunc129.ext), !dbg !261 ; [debug line = 415:17]
  call fastcc void @wait_tmr(), !dbg !262         ; [debug line = 420:3]
  br label %.preheader, !dbg !263                 ; [debug line = 421:2]
}

; [#uses=23]
declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

; [#uses=1]
define internal fastcc i21 @diff_angle(i16 zeroext %target, i16 %value) readnone {
  call void @llvm.dbg.value(metadata !{i16 %target}, i64 0, metadata !264), !dbg !270 ; [debug line = 133:25] [debug variable = target]
  call void @llvm.dbg.value(metadata !{i16 %value}, i64 0, metadata !271), !dbg !272 ; [debug line = 133:40] [debug variable = value]
  %tmp.cast3 = zext i16 %value to i18, !dbg !273  ; [#uses=2 type=i18] [debug line = 138:2]
  %tmp.cast = zext i16 %value to i17, !dbg !273   ; [#uses=2 type=i17] [debug line = 138:2]
  %tmp..cast = zext i16 %target to i17, !dbg !273 ; [#uses=4 type=i17] [debug line = 138:2]
  %tmp. = sub i17 %tmp.cast, %tmp..cast, !dbg !273 ; [#uses=1 type=i17] [debug line = 138:2]
  %tmp.53.cast = sext i17 %tmp. to i19, !dbg !273 ; [#uses=1 type=i19] [debug line = 138:2]
  %tmp.48 = add i17 %tmp..cast, -1, !dbg !275     ; [#uses=1 type=i17] [debug line = 139:2]
  %tmp.49 = sub i17 %tmp.48, %tmp.cast, !dbg !275 ; [#uses=2 type=i17] [debug line = 139:2]
  %tmp.50 = icmp sgt i17 %tmp.49, -18001          ; [#uses=1 type=i1]
  %smax2 = select i1 %tmp.50, i17 %tmp.49, i17 -18001 ; [#uses=1 type=i17]
  %smax2.cast = sext i17 %smax2 to i18            ; [#uses=1 type=i18]
  %tmp.51 = sub i17 36000, %tmp..cast, !dbg !275  ; [#uses=1 type=i17] [debug line = 139:2]
  %tmp.57.cast.cast = sext i17 %tmp.51 to i19, !dbg !275 ; [#uses=1 type=i19] [debug line = 139:2]
  %tmp1 = add i18 %tmp.cast3, %smax2.cast, !dbg !275 ; [#uses=1 type=i18] [debug line = 139:2]
  %tmp1.cast.cast = sext i18 %tmp1 to i19, !dbg !275 ; [#uses=1 type=i19] [debug line = 139:2]
  %tmp.52 = add i19 %tmp.57.cast.cast, %tmp1.cast.cast, !dbg !275 ; [#uses=2 type=i19] [debug line = 139:2]
  %tmp.53 = urem i19 %tmp.52, 36000, !dbg !275    ; [#uses=1 type=i19] [debug line = 139:2]
  %tmp.54 = sub i19 %tmp.52, %tmp.53, !dbg !276   ; [#uses=2 type=i19] [debug line = 141:2]
  %tmp.61.cast = sext i19 %tmp.54 to i20, !dbg !276 ; [#uses=1 type=i20] [debug line = 141:2]
  %tmp.55 = sub i19 %tmp.53.cast, %tmp.54, !dbg !276 ; [#uses=3 type=i19] [debug line = 141:2]
  %tmp.62.cast = sext i19 %tmp.55 to i20, !dbg !276 ; [#uses=1 type=i20] [debug line = 141:2]
  %tmp.56 = icmp sgt i19 %tmp.55, -18000          ; [#uses=1 type=i1]
  %smax1 = select i1 %tmp.56, i19 %tmp.55, i19 -18000 ; [#uses=1 type=i19]
  %smax1.cast = sext i19 %smax1 to i20            ; [#uses=1 type=i20]
  %tmp.57 = add i17 %tmp..cast, 35999, !dbg !276  ; [#uses=1 type=i17] [debug line = 141:2]
  %tmp.64.cast = zext i17 %tmp.57 to i18, !dbg !276 ; [#uses=1 type=i18] [debug line = 141:2]
  %tmp.58 = sub i18 %tmp.64.cast, %tmp.cast3, !dbg !276 ; [#uses=1 type=i18] [debug line = 141:2]
  %tmp.65.cast.cast = sext i18 %tmp.58 to i21, !dbg !276 ; [#uses=1 type=i21] [debug line = 141:2]
  %tmp2 = add i20 %smax1.cast, %tmp.61.cast, !dbg !276 ; [#uses=1 type=i20] [debug line = 141:2]
  %tmp2.cast.cast = sext i20 %tmp2 to i21, !dbg !276 ; [#uses=1 type=i21] [debug line = 141:2]
  %tmp.59 = add i21 %tmp.65.cast.cast, %tmp2.cast.cast, !dbg !276 ; [#uses=2 type=i21] [debug line = 141:2]
  %tmp.60 = urem i21 %tmp.59, 36000, !dbg !276    ; [#uses=1 type=i21] [debug line = 141:2]
  %tmp.68.cast = trunc i21 %tmp.60 to i20, !dbg !276 ; [#uses=1 type=i20] [debug line = 141:2]
  %tmp.61 = sub i20 %tmp.62.cast, %tmp.68.cast, !dbg !276 ; [#uses=1 type=i20] [debug line = 141:2]
  %tmp.69.cast = sext i20 %tmp.61 to i21, !dbg !276 ; [#uses=1 type=i21] [debug line = 141:2]
  %retval = add i21 %tmp.59, %tmp.69.cast, !dbg !273 ; [#uses=1 type=i21] [debug line = 138:2]
  call void @llvm.dbg.value(metadata !{i21 %retval}, i64 0, metadata !277), !dbg !273 ; [debug line = 138:2] [debug variable = retval]
  ret i21 %retval, !dbg !278                      ; [debug line = 144:2]
}

; [#uses=16]
define internal fastcc zeroext i7 @bin2char(i4 zeroext %val) readnone {
  call void @llvm.dbg.value(metadata !{i4 %val}, i64 0, metadata !279), !dbg !285 ; [debug line = 148:22] [debug variable = val]
  switch i4 %val, label %15 [
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
  ], !dbg !286                                    ; [debug line = 153:2]

; <label>:1                                       ; preds = %0
  br label %._crit_edge, !dbg !288                ; [debug line = 155:24]

; <label>:2                                       ; preds = %0
  br label %._crit_edge, !dbg !290                ; [debug line = 156:24]

; <label>:3                                       ; preds = %0
  br label %._crit_edge, !dbg !291                ; [debug line = 157:24]

; <label>:4                                       ; preds = %0
  br label %._crit_edge, !dbg !292                ; [debug line = 158:24]

; <label>:5                                       ; preds = %0
  br label %._crit_edge, !dbg !293                ; [debug line = 159:24]

; <label>:6                                       ; preds = %0
  br label %._crit_edge, !dbg !294                ; [debug line = 160:24]

; <label>:7                                       ; preds = %0
  br label %._crit_edge, !dbg !295                ; [debug line = 161:24]

; <label>:8                                       ; preds = %0
  br label %._crit_edge, !dbg !296                ; [debug line = 162:24]

; <label>:9                                       ; preds = %0
  br label %._crit_edge, !dbg !297                ; [debug line = 163:24]

; <label>:10                                      ; preds = %0
  br label %._crit_edge, !dbg !298                ; [debug line = 164:25]

; <label>:11                                      ; preds = %0
  br label %._crit_edge, !dbg !299                ; [debug line = 165:25]

; <label>:12                                      ; preds = %0
  br label %._crit_edge, !dbg !300                ; [debug line = 166:25]

; <label>:13                                      ; preds = %0
  br label %._crit_edge, !dbg !301                ; [debug line = 167:25]

; <label>:14                                      ; preds = %0
  br label %._crit_edge, !dbg !302                ; [debug line = 168:25]

; <label>:15                                      ; preds = %0
  br label %._crit_edge, !dbg !303                ; [debug line = 170:2]

._crit_edge:                                      ; preds = %15, %14, %13, %12, %11, %10, %9, %8, %7, %6, %5, %4, %3, %2, %1, %0
  %retval = phi i7 [ -58, %15 ], [ -59, %14 ], [ -60, %13 ], [ -61, %12 ], [ -62, %11 ], [ -63, %10 ], [ 57, %9 ], [ 56, %8 ], [ 55, %7 ], [ 54, %6 ], [ 53, %5 ], [ 52, %4 ], [ 51, %3 ], [ 50, %2 ], [ 49, %1 ], [ 48, %0 ] ; [#uses=1 type=i7]
  ret i7 %retval, !dbg !304                       ; [debug line = 172:2]
}

; [#uses=18]
declare void @_ssdm_op_Wait(...) nounwind

; [#uses=1]
declare void @_ssdm_op_SpecTopModule(...) nounwind

; [#uses=2]
declare i32 @_ssdm_op_SpecLoopTripCount(...)

; [#uses=1]
declare i32 @_ssdm_op_SpecLoopBegin(...)

; [#uses=12]
declare void @_ssdm_op_SpecInterface(...) nounwind

; [#uses=2]
declare i16 @_ssdm_op_BitConcatenate.i16.i8.i8(i8, i8) nounwind readnone

; [#uses=2]
declare void @_ssdm_SpecKeepAssert(...)

!hls.encrypted.func = !{}
!llvm.map.gv = !{!0, !7, !12, !17, !22, !27, !32, !37, !42, !47, !52, !57}
!llvm.dbg.cu = !{!62}

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
!62 = metadata !{i32 786449, i32 0, i32 1, metadata !"D:/21_streamer_car5_artix7/fpga_arty/motor_ctrl/solution1/.autopilot/db/motor_ctrl.pragma.2.c", metadata !"D:\5C21_streamer_car5_artix7\5Cfpga_arty", metadata !"clang version 3.1 ", i1 true, i1 false, metadata !"", i32 0, null, null, null, metadata !63} ; [ DW_TAG_compile_unit ]
!63 = metadata !{metadata !64}
!64 = metadata !{metadata !65, metadata !70, metadata !71, metadata !72, metadata !73, metadata !74, metadata !75, metadata !79, metadata !80, metadata !81, metadata !82, metadata !83}
!65 = metadata !{i32 786484, i32 0, null, metadata !"mem_rreq", metadata !"mem_rreq", metadata !"", metadata !66, i32 63, metadata !67, i32 0, i32 1, i1* @mem_rreq} ; [ DW_TAG_variable ]
!66 = metadata !{i32 786473, metadata !"motor_ctrl.c", metadata !"D:\5C21_streamer_car5_artix7\5Cfpga_arty", null} ; [ DW_TAG_file_type ]
!67 = metadata !{i32 786485, null, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !68} ; [ DW_TAG_volatile_type ]
!68 = metadata !{i32 786454, null, metadata !"uint1", metadata !66, i32 3, i64 0, i64 0, i64 0, i32 0, metadata !69} ; [ DW_TAG_typedef ]
!69 = metadata !{i32 786468, null, metadata !"uint1", null, i32 0, i64 1, i64 1, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!70 = metadata !{i32 786484, i32 0, null, metadata !"r_pwm", metadata !"r_pwm", metadata !"", metadata !66, i32 55, metadata !67, i32 0, i32 1, i1* @r_pwm} ; [ DW_TAG_variable ]
!71 = metadata !{i32 786484, i32 0, null, metadata !"dummy_tmr_out", metadata !"dummy_tmr_out", metadata !"", metadata !66, i32 66, metadata !67, i32 0, i32 1, i1* @dummy_tmr_out} ; [ DW_TAG_variable ]
!72 = metadata !{i32 786484, i32 0, null, metadata !"l_pwm", metadata !"l_pwm", metadata !"", metadata !66, i32 53, metadata !67, i32 0, i32 1, i1* @l_pwm} ; [ DW_TAG_variable ]
!73 = metadata !{i32 786484, i32 0, null, metadata !"l_dir", metadata !"l_dir", metadata !"", metadata !66, i32 52, metadata !67, i32 0, i32 1, i1* @l_dir} ; [ DW_TAG_variable ]
!74 = metadata !{i32 786484, i32 0, null, metadata !"r_dir", metadata !"r_dir", metadata !"", metadata !66, i32 54, metadata !67, i32 0, i32 1, i1* @r_dir} ; [ DW_TAG_variable ]
!75 = metadata !{i32 786484, i32 0, null, metadata !"mem_addr", metadata !"mem_addr", metadata !"", metadata !66, i32 58, metadata !76, i32 0, i32 1, i8* @mem_addr} ; [ DW_TAG_variable ]
!76 = metadata !{i32 786485, null, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !77} ; [ DW_TAG_volatile_type ]
!77 = metadata !{i32 786454, null, metadata !"uint8", metadata !66, i32 10, i64 0, i64 0, i64 0, i32 0, metadata !78} ; [ DW_TAG_typedef ]
!78 = metadata !{i32 786468, null, metadata !"uint8", null, i32 0, i64 8, i64 8, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!79 = metadata !{i32 786484, i32 0, null, metadata !"mem_wreq", metadata !"mem_wreq", metadata !"", metadata !66, i32 61, metadata !67, i32 0, i32 1, i1* @mem_wreq} ; [ DW_TAG_variable ]
!80 = metadata !{i32 786484, i32 0, null, metadata !"mem_dout", metadata !"mem_dout", metadata !"", metadata !66, i32 60, metadata !76, i32 0, i32 1, i8* @mem_dout} ; [ DW_TAG_variable ]
!81 = metadata !{i32 786484, i32 0, null, metadata !"mem_wack", metadata !"mem_wack", metadata !"", metadata !66, i32 62, metadata !67, i32 0, i32 1, i1* @mem_wack} ; [ DW_TAG_variable ]
!82 = metadata !{i32 786484, i32 0, null, metadata !"mem_rack", metadata !"mem_rack", metadata !"", metadata !66, i32 64, metadata !67, i32 0, i32 1, i1* @mem_rack} ; [ DW_TAG_variable ]
!83 = metadata !{i32 786484, i32 0, null, metadata !"mem_din", metadata !"mem_din", metadata !"", metadata !66, i32 59, metadata !76, i32 0, i32 1, i8* @mem_din} ; [ DW_TAG_variable ]
!84 = metadata !{i8 -128, i8 31, i8 0, i8 -1}     
!85 = metadata !{i32 786689, metadata !86, metadata !"addr", metadata !66, i32 16777301, metadata !77, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!86 = metadata !{i32 786478, i32 0, metadata !66, metadata !"write_mem", metadata !"write_mem", metadata !"", metadata !66, i32 85, metadata !87, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (i8, i8)* @write_mem, null, null, metadata !89, i32 86} ; [ DW_TAG_subprogram ]
!87 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !88, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!88 = metadata !{null, metadata !77, metadata !77}
!89 = metadata !{metadata !90}
!90 = metadata !{i32 786468}                      ; [ DW_TAG_base_type ]
!91 = metadata !{i32 85, i32 22, metadata !86, null}
!92 = metadata !{i32 786689, metadata !86, metadata !"data", metadata !66, i32 33554517, metadata !77, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!93 = metadata !{i32 85, i32 34, metadata !86, null}
!94 = metadata !{i32 88, i32 2, metadata !95, null}
!95 = metadata !{i32 786443, metadata !86, i32 86, i32 1, metadata !66, i32 3} ; [ DW_TAG_lexical_block ]
!96 = metadata !{i32 89, i32 2, metadata !95, null}
!97 = metadata !{i32 90, i32 2, metadata !95, null}
!98 = metadata !{i32 91, i32 2, metadata !95, null}
!99 = metadata !{i32 92, i32 2, metadata !95, null}
!100 = metadata !{i32 94, i32 2, metadata !95, null}
!101 = metadata !{i32 95, i32 3, metadata !102, null}
!102 = metadata !{i32 786443, metadata !95, i32 94, i32 5, metadata !66, i32 4} ; [ DW_TAG_lexical_block ]
!103 = metadata !{i32 96, i32 3, metadata !102, null}
!104 = metadata !{i32 97, i32 3, metadata !102, null}
!105 = metadata !{i32 98, i32 2, metadata !102, null}
!106 = metadata !{i32 99, i32 2, metadata !95, null}
!107 = metadata !{i32 101, i32 2, metadata !95, null}
!108 = metadata !{i32 102, i32 2, metadata !95, null}
!109 = metadata !{i32 103, i32 2, metadata !95, null}
!110 = metadata !{i32 104, i32 2, metadata !95, null}
!111 = metadata !{i32 105, i32 1, metadata !95, null}
!112 = metadata !{i32 77, i32 2, metadata !113, null}
!113 = metadata !{i32 786443, metadata !114, i32 74, i32 1, metadata !66, i32 0} ; [ DW_TAG_lexical_block ]
!114 = metadata !{i32 786478, i32 0, metadata !66, metadata !"wait_tmr", metadata !"wait_tmr", metadata !"", metadata !66, i32 73, metadata !115, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, metadata !89, i32 74} ; [ DW_TAG_subprogram ]
!115 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !116, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!116 = metadata !{null, metadata !117}
!117 = metadata !{i32 786454, null, metadata !"uint32", metadata !66, i32 34, i64 0, i64 0, i64 0, i32 0, metadata !118} ; [ DW_TAG_typedef ]
!118 = metadata !{i32 786468, null, metadata !"uint32", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!119 = metadata !{i32 78, i32 7, metadata !120, null}
!120 = metadata !{i32 786443, metadata !113, i32 78, i32 2, metadata !66, i32 1} ; [ DW_TAG_lexical_block ]
!121 = metadata !{i32 79, i32 3, metadata !122, null}
!122 = metadata !{i32 786443, metadata !120, i32 78, i32 28, metadata !66, i32 2} ; [ DW_TAG_lexical_block ]
!123 = metadata !{i32 78, i32 23, metadata !120, null}
!124 = metadata !{i32 786688, metadata !113, metadata !"t", metadata !66, i32 76, metadata !117, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!125 = metadata !{i32 81, i32 2, metadata !113, null}
!126 = metadata !{i32 82, i32 1, metadata !113, null}
!127 = metadata !{i8 -128, i8 -123, i8 -128, i8 -123} 
!128 = metadata !{i32 786689, metadata !129, metadata !"addr", metadata !66, i32 16777324, metadata !77, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!129 = metadata !{i32 786478, i32 0, metadata !66, metadata !"read_mem", metadata !"read_mem", metadata !"", metadata !66, i32 108, metadata !130, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i8 (i8)* @read_mem, null, null, metadata !89, i32 109} ; [ DW_TAG_subprogram ]
!130 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !131, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!131 = metadata !{metadata !77, metadata !77}
!132 = metadata !{i32 108, i32 22, metadata !129, null}
!133 = metadata !{i32 113, i32 2, metadata !134, null}
!134 = metadata !{i32 786443, metadata !129, i32 109, i32 1, metadata !66, i32 5} ; [ DW_TAG_lexical_block ]
!135 = metadata !{i32 114, i32 2, metadata !134, null}
!136 = metadata !{i32 115, i32 2, metadata !134, null}
!137 = metadata !{i32 116, i32 2, metadata !134, null}
!138 = metadata !{i32 118, i32 2, metadata !134, null}
!139 = metadata !{i32 119, i32 3, metadata !140, null}
!140 = metadata !{i32 786443, metadata !134, i32 118, i32 5, metadata !66, i32 6} ; [ DW_TAG_lexical_block ]
!141 = metadata !{i32 120, i32 3, metadata !140, null}
!142 = metadata !{i32 121, i32 3, metadata !140, null}
!143 = metadata !{i32 786688, metadata !134, metadata !"dt", metadata !66, i32 111, metadata !77, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!144 = metadata !{i32 122, i32 2, metadata !140, null}
!145 = metadata !{i32 123, i32 2, metadata !134, null}
!146 = metadata !{i32 125, i32 2, metadata !134, null}
!147 = metadata !{i32 126, i32 2, metadata !134, null}
!148 = metadata !{i32 127, i32 2, metadata !134, null}
!149 = metadata !{i32 129, i32 2, metadata !134, null}
!150 = metadata !{i32 178, i32 1, metadata !151, null}
!151 = metadata !{i32 786443, metadata !152, i32 177, i32 1, metadata !66, i32 10} ; [ DW_TAG_lexical_block ]
!152 = metadata !{i32 786478, i32 0, metadata !66, metadata !"motor_ctrl", metadata !"motor_ctrl", metadata !"", metadata !66, i32 176, metadata !153, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 false, void ()* @motor_ctrl, null, null, metadata !89, i32 177} ; [ DW_TAG_subprogram ]
!153 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !154, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!154 = metadata !{null}
!155 = metadata !{i32 179, i32 1, metadata !151, null}
!156 = metadata !{i32 181, i32 1, metadata !151, null}
!157 = metadata !{i32 182, i32 1, metadata !151, null}
!158 = metadata !{i32 183, i32 1, metadata !151, null}
!159 = metadata !{i32 184, i32 1, metadata !151, null}
!160 = metadata !{i32 186, i32 1, metadata !151, null}
!161 = metadata !{i32 187, i32 1, metadata !151, null}
!162 = metadata !{i32 188, i32 1, metadata !151, null}
!163 = metadata !{i32 189, i32 1, metadata !151, null}
!164 = metadata !{i32 190, i32 1, metadata !151, null}
!165 = metadata !{i32 191, i32 1, metadata !151, null}
!166 = metadata !{i32 192, i32 1, metadata !151, null}
!167 = metadata !{i32 207, i32 2, metadata !151, null}
!168 = metadata !{i32 208, i32 2, metadata !151, null}
!169 = metadata !{i32 209, i32 2, metadata !151, null}
!170 = metadata !{i32 210, i32 2, metadata !151, null}
!171 = metadata !{i32 211, i32 2, metadata !151, null}
!172 = metadata !{i32 218, i32 2, metadata !151, null}
!173 = metadata !{i32 220, i32 2, metadata !151, null}
!174 = metadata !{i32 221, i32 2, metadata !151, null}
!175 = metadata !{i32 222, i32 2, metadata !151, null}
!176 = metadata !{i32 223, i32 2, metadata !151, null}
!177 = metadata !{i32 224, i32 2, metadata !151, null}
!178 = metadata !{i32 226, i32 7, metadata !179, null}
!179 = metadata !{i32 786443, metadata !151, i32 226, i32 2, metadata !66, i32 11} ; [ DW_TAG_lexical_block ]
!180 = metadata !{i32 227, i32 3, metadata !181, null}
!181 = metadata !{i32 786443, metadata !179, i32 226, i32 27, metadata !66, i32 12} ; [ DW_TAG_lexical_block ]
!182 = metadata !{i32 226, i32 22, metadata !179, null}
!183 = metadata !{i32 786688, metadata !151, metadata !"i", metadata !66, i32 205, metadata !77, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!184 = metadata !{i32 251, i32 8, metadata !185, null}
!185 = metadata !{i32 786443, metadata !151, i32 246, i32 12, metadata !66, i32 13} ; [ DW_TAG_lexical_block ]
!186 = metadata !{i32 786688, metadata !151, metadata !"eh", metadata !66, i32 203, metadata !77, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!187 = metadata !{i32 252, i32 8, metadata !185, null}
!188 = metadata !{i32 786688, metadata !151, metadata !"el", metadata !66, i32 203, metadata !77, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!189 = metadata !{i32 786688, metadata !151, metadata !"et", metadata !66, i32 204, metadata !190, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!190 = metadata !{i32 786454, null, metadata !"uint16", metadata !66, i32 18, i64 0, i64 0, i64 0, i32 0, metadata !191} ; [ DW_TAG_typedef ]
!191 = metadata !{i32 786468, null, metadata !"uint16", null, i32 0, i64 16, i64 16, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!192 = metadata !{i32 253, i32 3, metadata !185, null}
!193 = metadata !{i32 255, i32 3, metadata !185, null}
!194 = metadata !{i32 256, i32 3, metadata !185, null}
!195 = metadata !{i32 256, i32 10, metadata !185, null}
!196 = metadata !{i32 258, i32 3, metadata !185, null}
!197 = metadata !{i32 259, i32 8, metadata !185, null}
!198 = metadata !{i32 260, i32 8, metadata !185, null}
!199 = metadata !{i32 786688, metadata !151, metadata !"e", metadata !66, i32 204, metadata !190, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!200 = metadata !{i32 261, i32 3, metadata !185, null}
!201 = metadata !{i32 262, i32 3, metadata !185, null}
!202 = metadata !{i32 265, i32 16, metadata !185, null}
!203 = metadata !{i32 266, i32 16, metadata !185, null}
!204 = metadata !{i32 267, i32 16, metadata !185, null}
!205 = metadata !{i32 268, i32 16, metadata !185, null}
!206 = metadata !{i32 270, i32 16, metadata !185, null}
!207 = metadata !{i32 271, i32 16, metadata !185, null}
!208 = metadata !{i32 272, i32 17, metadata !185, null}
!209 = metadata !{i32 273, i32 17, metadata !185, null}
!210 = metadata !{i32 281, i32 14, metadata !185, null}
!211 = metadata !{i32 284, i32 17, metadata !185, null}
!212 = metadata !{i32 285, i32 17, metadata !185, null}
!213 = metadata !{i32 286, i32 17, metadata !185, null}
!214 = metadata !{i32 287, i32 17, metadata !185, null}
!215 = metadata !{i32 289, i32 3, metadata !185, null}
!216 = metadata !{i32 291, i32 4, metadata !217, null}
!217 = metadata !{i32 786443, metadata !185, i32 289, i32 72, metadata !66, i32 14} ; [ DW_TAG_lexical_block ]
!218 = metadata !{i32 292, i32 3, metadata !217, null}
!219 = metadata !{i32 295, i32 4, metadata !220, null}
!220 = metadata !{i32 786443, metadata !185, i32 293, i32 8, metadata !66, i32 15} ; [ DW_TAG_lexical_block ]
!221 = metadata !{i32 298, i32 3, metadata !185, null}
!222 = metadata !{i32 300, i32 4, metadata !223, null}
!223 = metadata !{i32 786443, metadata !185, i32 298, i32 58, metadata !66, i32 16} ; [ DW_TAG_lexical_block ]
!224 = metadata !{i32 301, i32 3, metadata !223, null}
!225 = metadata !{i32 304, i32 4, metadata !226, null}
!226 = metadata !{i32 786443, metadata !185, i32 302, i32 8, metadata !66, i32 17} ; [ DW_TAG_lexical_block ]
!227 = metadata !{i32 308, i32 10, metadata !185, null}
!228 = metadata !{i32 786688, metadata !151, metadata !"mode", metadata !66, i32 202, metadata !77, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!229 = metadata !{i32 310, i32 3, metadata !185, null}
!230 = metadata !{i32 786688, metadata !151, metadata !"chR_pwm", metadata !66, i32 200, metadata !77, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!231 = metadata !{i32 786688, metadata !151, metadata !"chL_pwm", metadata !66, i32 200, metadata !77, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!232 = metadata !{i32 312, i32 3, metadata !185, null}
!233 = metadata !{i32 342, i32 4, metadata !234, null}
!234 = metadata !{i32 786443, metadata !185, i32 312, i32 24, metadata !66, i32 18} ; [ DW_TAG_lexical_block ]
!235 = metadata !{i32 321, i32 9, metadata !234, null}
!236 = metadata !{i32 347, i32 5, metadata !237, null}
!237 = metadata !{i32 786443, metadata !234, i32 346, i32 9, metadata !66, i32 24} ; [ DW_TAG_lexical_block ]
!238 = metadata !{i32 368, i32 5, metadata !239, null}
!239 = metadata !{i32 786443, metadata !234, i32 367, i32 9, metadata !66, i32 28} ; [ DW_TAG_lexical_block ]
!240 = metadata !{i32 390, i32 3, metadata !185, null}
!241 = metadata !{i32 391, i32 3, metadata !185, null}
!242 = metadata !{i32 786688, metadata !151, metadata !"mtr_pwm_cnt", metadata !66, i32 195, metadata !243, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!243 = metadata !{i32 786485, null, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !244} ; [ DW_TAG_volatile_type ]
!244 = metadata !{i32 786468, null, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!245 = metadata !{i32 392, i32 3, metadata !185, null}
!246 = metadata !{i32 393, i32 4, metadata !185, null}
!247 = metadata !{i32 395, i32 4, metadata !185, null}
!248 = metadata !{i32 397, i32 3, metadata !185, null}
!249 = metadata !{i32 398, i32 3, metadata !185, null}
!250 = metadata !{i32 399, i32 4, metadata !185, null}
!251 = metadata !{i32 401, i32 4, metadata !185, null}
!252 = metadata !{i32 402, i32 3, metadata !185, null}
!253 = metadata !{i32 405, i32 3, metadata !185, null}
!254 = metadata !{i32 406, i32 3, metadata !185, null}
!255 = metadata !{i32 407, i32 4, metadata !256, null}
!256 = metadata !{i32 786443, metadata !185, i32 406, i32 26, metadata !66, i32 31} ; [ DW_TAG_lexical_block ]
!257 = metadata !{i32 408, i32 3, metadata !256, null}
!258 = metadata !{i32 411, i32 17, metadata !185, null}
!259 = metadata !{i32 412, i32 17, metadata !185, null}
!260 = metadata !{i32 414, i32 17, metadata !185, null}
!261 = metadata !{i32 415, i32 17, metadata !185, null}
!262 = metadata !{i32 420, i32 3, metadata !185, null}
!263 = metadata !{i32 421, i32 2, metadata !185, null}
!264 = metadata !{i32 786689, metadata !265, metadata !"target", metadata !66, i32 16777349, metadata !190, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!265 = metadata !{i32 786478, i32 0, metadata !66, metadata !"diff_angle", metadata !"diff_angle", metadata !"", metadata !66, i32 133, metadata !266, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, metadata !89, i32 134} ; [ DW_TAG_subprogram ]
!266 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !267, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!267 = metadata !{metadata !268, metadata !190, metadata !190}
!268 = metadata !{i32 786454, null, metadata !"int32", metadata !66, i32 34, i64 0, i64 0, i64 0, i32 0, metadata !269} ; [ DW_TAG_typedef ]
!269 = metadata !{i32 786468, null, metadata !"int32", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!270 = metadata !{i32 133, i32 25, metadata !265, null}
!271 = metadata !{i32 786689, metadata !265, metadata !"value", metadata !66, i32 33554565, metadata !190, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!272 = metadata !{i32 133, i32 40, metadata !265, null}
!273 = metadata !{i32 138, i32 2, metadata !274, null}
!274 = metadata !{i32 786443, metadata !265, i32 134, i32 1, metadata !66, i32 7} ; [ DW_TAG_lexical_block ]
!275 = metadata !{i32 139, i32 2, metadata !274, null}
!276 = metadata !{i32 141, i32 2, metadata !274, null}
!277 = metadata !{i32 786688, metadata !274, metadata !"retval", metadata !66, i32 136, metadata !268, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!278 = metadata !{i32 144, i32 2, metadata !274, null}
!279 = metadata !{i32 786689, metadata !280, metadata !"val", metadata !66, i32 16777364, metadata !283, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!280 = metadata !{i32 786478, i32 0, metadata !66, metadata !"bin2char", metadata !"bin2char", metadata !"", metadata !66, i32 148, metadata !281, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, metadata !89, i32 149} ; [ DW_TAG_subprogram ]
!281 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !282, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!282 = metadata !{metadata !77, metadata !283}
!283 = metadata !{i32 786454, null, metadata !"uint4", metadata !66, i32 6, i64 0, i64 0, i64 0, i32 0, metadata !284} ; [ DW_TAG_typedef ]
!284 = metadata !{i32 786468, null, metadata !"uint4", null, i32 0, i64 4, i64 4, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!285 = metadata !{i32 148, i32 22, metadata !280, null}
!286 = metadata !{i32 153, i32 2, metadata !287, null}
!287 = metadata !{i32 786443, metadata !280, i32 149, i32 1, metadata !66, i32 8} ; [ DW_TAG_lexical_block ]
!288 = metadata !{i32 155, i32 24, metadata !289, null}
!289 = metadata !{i32 786443, metadata !287, i32 153, i32 15, metadata !66, i32 9} ; [ DW_TAG_lexical_block ]
!290 = metadata !{i32 156, i32 24, metadata !289, null}
!291 = metadata !{i32 157, i32 24, metadata !289, null}
!292 = metadata !{i32 158, i32 24, metadata !289, null}
!293 = metadata !{i32 159, i32 24, metadata !289, null}
!294 = metadata !{i32 160, i32 24, metadata !289, null}
!295 = metadata !{i32 161, i32 24, metadata !289, null}
!296 = metadata !{i32 162, i32 24, metadata !289, null}
!297 = metadata !{i32 163, i32 24, metadata !289, null}
!298 = metadata !{i32 164, i32 25, metadata !289, null}
!299 = metadata !{i32 165, i32 25, metadata !289, null}
!300 = metadata !{i32 166, i32 25, metadata !289, null}
!301 = metadata !{i32 167, i32 25, metadata !289, null}
!302 = metadata !{i32 168, i32 25, metadata !289, null}
!303 = metadata !{i32 170, i32 2, metadata !289, null}
!304 = metadata !{i32 172, i32 2, metadata !287, null}
