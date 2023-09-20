; ModuleID = 'D:/21_streamer_car5_artix7/fpga_arty/motor_ctrl/solution1/.autopilot/db/a.g.1.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-w64-mingw32"

@r_pwm = global i1 false, align 1                 ; [#uses=4 type=i1*]
@r_dir = global i1 false, align 1                 ; [#uses=3 type=i1*]
@mem_wreq = global i1 false, align 1              ; [#uses=4 type=i1*]
@mem_wack = common global i1 false, align 1       ; [#uses=2 type=i1*]
@mem_rreq = global i1 false, align 1              ; [#uses=4 type=i1*]
@mem_rack = common global i1 false, align 1       ; [#uses=2 type=i1*]
@mem_dout = global i8 0, align 1                  ; [#uses=4 type=i8*]
@mem_din = common global i8 0, align 1            ; [#uses=2 type=i8*]
@mem_addr = global i8 0, align 1                  ; [#uses=7 type=i8*]
@l_pwm = global i1 false, align 1                 ; [#uses=4 type=i1*]
@l_dir = global i1 false, align 1                 ; [#uses=3 type=i1*]
@dummy_tmr_out = global i1 false, align 1         ; [#uses=3 type=i1*]
@.str2 = private unnamed_addr constant [12 x i8] c"hls_label_0\00", align 1 ; [#uses=1 type=[12 x i8]*]
@.str1 = private unnamed_addr constant [8 x i8] c"ap_none\00", align 1 ; [#uses=1 type=[8 x i8]*]
@.str = private unnamed_addr constant [1 x i8] zeroinitializer, align 1 ; [#uses=1 type=[1 x i8]*]

; [#uses=24]
define internal fastcc void @write_mem(i8 zeroext %addr, i8 zeroext %data) nounwind uwtable {
  call void @llvm.dbg.value(metadata !{i8 %addr}, i64 0, metadata !54), !dbg !55 ; [debug line = 85:22] [debug variable = addr]
  call void @llvm.dbg.value(metadata !{i8 %data}, i64 0, metadata !56), !dbg !57 ; [debug line = 85:34] [debug variable = data]
  call void (...)* @_ssdm_InlineSelf(i32 2, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !58 ; [debug line = 87:1]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !60 ; [debug line = 88:2]
  store volatile i8 %addr, i8* @mem_addr, align 1, !dbg !61 ; [debug line = 89:2]
  store volatile i8 %data, i8* @mem_dout, align 1, !dbg !62 ; [debug line = 90:2]
  store volatile i1 false, i1* @mem_wreq, align 1, !dbg !63 ; [debug line = 91:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !64 ; [debug line = 92:2]
  br label %._crit_edge, !dbg !65                 ; [debug line = 94:2]

._crit_edge:                                      ; preds = %._crit_edge, %0
  store volatile i8 %addr, i8* @mem_addr, align 1, !dbg !66 ; [debug line = 95:3]
  store volatile i8 %data, i8* @mem_dout, align 1, !dbg !68 ; [debug line = 96:3]
  store volatile i1 true, i1* @mem_wreq, align 1, !dbg !69 ; [debug line = 97:3]
  %mem_wack.load = load volatile i1* @mem_wack, align 1, !dbg !70 ; [#uses=1 type=i1] [debug line = 98:2]
  br i1 %mem_wack.load, label %1, label %._crit_edge, !dbg !70 ; [debug line = 98:2]

; <label>:1                                       ; preds = %._crit_edge
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !71 ; [debug line = 99:2]
  store volatile i8 %addr, i8* @mem_addr, align 1, !dbg !72 ; [debug line = 101:2]
  store volatile i8 %data, i8* @mem_dout, align 1, !dbg !73 ; [debug line = 102:2]
  store volatile i1 false, i1* @mem_wreq, align 1, !dbg !74 ; [debug line = 103:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !75 ; [debug line = 104:2]
  ret void, !dbg !76                              ; [debug line = 105:1]
}

; [#uses=1]
define internal fastcc void @wait_tmr() nounwind uwtable {
  call void (...)* @_ssdm_InlineSelf(i32 2, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !77 ; [debug line = 75:1]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !79 ; [debug line = 77:2]
  br label %1, !dbg !80                           ; [debug line = 78:7]

; <label>:1                                       ; preds = %2, %0
  %t = phi i32 [ 0, %0 ], [ %t.1, %2 ]            ; [#uses=2 type=i32]
  %exitcond = icmp eq i32 %t, 100000, !dbg !80    ; [#uses=1 type=i1] [debug line = 78:7]
  br i1 %exitcond, label %3, label %2, !dbg !80   ; [debug line = 78:7]

; <label>:2                                       ; preds = %1
  %dummy_tmr_out.load = load volatile i1* @dummy_tmr_out, align 1, !dbg !82 ; [#uses=1 type=i1] [debug line = 79:3]
  %not. = xor i1 %dummy_tmr_out.load, true, !dbg !82 ; [#uses=1 type=i1] [debug line = 79:3]
  store volatile i1 %not., i1* @dummy_tmr_out, align 1, !dbg !82 ; [debug line = 79:3]
  %t.1 = add i32 %t, 1, !dbg !84                  ; [#uses=1 type=i32] [debug line = 78:23]
  call void @llvm.dbg.value(metadata !{i32 %t.1}, i64 0, metadata !85), !dbg !84 ; [debug line = 78:23] [debug variable = t]
  br label %1, !dbg !84                           ; [debug line = 78:23]

; <label>:3                                       ; preds = %1
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !86 ; [debug line = 81:2]
  ret void, !dbg !87                              ; [debug line = 82:1]
}

; [#uses=6]
define internal fastcc zeroext i8 @read_mem(i8 zeroext %addr) nounwind uwtable {
  call void @llvm.dbg.value(metadata !{i8 %addr}, i64 0, metadata !88), !dbg !89 ; [debug line = 108:22] [debug variable = addr]
  call void (...)* @_ssdm_InlineSelf(i32 2, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !90 ; [debug line = 110:1]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !92 ; [debug line = 113:2]
  store volatile i8 %addr, i8* @mem_addr, align 1, !dbg !93 ; [debug line = 114:2]
  store volatile i1 true, i1* @mem_rreq, align 1, !dbg !94 ; [debug line = 115:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !95 ; [debug line = 116:2]
  br label %._crit_edge, !dbg !96                 ; [debug line = 118:2]

._crit_edge:                                      ; preds = %._crit_edge, %0
  store volatile i8 %addr, i8* @mem_addr, align 1, !dbg !97 ; [debug line = 119:3]
  store volatile i1 true, i1* @mem_rreq, align 1, !dbg !99 ; [debug line = 120:3]
  %dt = load volatile i8* @mem_din, align 1, !dbg !100 ; [#uses=1 type=i8] [debug line = 121:3]
  call void @llvm.dbg.value(metadata !{i8 %dt}, i64 0, metadata !101), !dbg !100 ; [debug line = 121:3] [debug variable = dt]
  %mem_rack.load = load volatile i1* @mem_rack, align 1, !dbg !102 ; [#uses=1 type=i1] [debug line = 122:2]
  br i1 %mem_rack.load, label %1, label %._crit_edge, !dbg !102 ; [debug line = 122:2]

; <label>:1                                       ; preds = %._crit_edge
  %dt.0.lcssa = phi i8 [ %dt, %._crit_edge ]      ; [#uses=1 type=i8]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !103 ; [debug line = 123:2]
  store volatile i8 %addr, i8* @mem_addr, align 1, !dbg !104 ; [debug line = 125:2]
  store volatile i1 false, i1* @mem_rreq, align 1, !dbg !105 ; [debug line = 126:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !106 ; [debug line = 127:2]
  ret i8 %dt.0.lcssa, !dbg !107                   ; [debug line = 129:2]
}

; [#uses=0]
define void @motor_ctrl() noreturn nounwind uwtable {
  %mtr_pwm_cnt = alloca i32, align 4              ; [#uses=7 type=i32*]
  call void (...)* @_ssdm_op_SpecTopModule(), !dbg !108 ; [debug line = 178:1]
  %dummy_tmr_out.load = load volatile i1* @dummy_tmr_out, align 1, !dbg !110 ; [#uses=1 type=i1] [debug line = 179:1]
  %tmp = zext i1 %dummy_tmr_out.load to i32, !dbg !110 ; [#uses=1 type=i32] [debug line = 179:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp, i8* getelementptr inbounds ([8 x i8]* @.str1, i64 0, i64 0), i32 1, i32 1, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !110 ; [debug line = 179:1]
  %l_dir.load = load volatile i1* @l_dir, align 1, !dbg !111 ; [#uses=1 type=i1] [debug line = 181:1]
  %tmp.1 = zext i1 %l_dir.load to i32, !dbg !111  ; [#uses=1 type=i32] [debug line = 181:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.1, i8* getelementptr inbounds ([8 x i8]* @.str1, i64 0, i64 0), i32 1, i32 1, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !111 ; [debug line = 181:1]
  %l_pwm.load = load volatile i1* @l_pwm, align 1, !dbg !112 ; [#uses=1 type=i1] [debug line = 182:1]
  %tmp.2 = zext i1 %l_pwm.load to i32, !dbg !112  ; [#uses=1 type=i32] [debug line = 182:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.2, i8* getelementptr inbounds ([8 x i8]* @.str1, i64 0, i64 0), i32 1, i32 1, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !112 ; [debug line = 182:1]
  %r_dir.load = load volatile i1* @r_dir, align 1, !dbg !113 ; [#uses=1 type=i1] [debug line = 183:1]
  %tmp.3 = zext i1 %r_dir.load to i32, !dbg !113  ; [#uses=1 type=i32] [debug line = 183:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.3, i8* getelementptr inbounds ([8 x i8]* @.str1, i64 0, i64 0), i32 1, i32 1, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !113 ; [debug line = 183:1]
  %r_pwm.load = load volatile i1* @r_pwm, align 1, !dbg !114 ; [#uses=1 type=i1] [debug line = 184:1]
  %tmp.4 = zext i1 %r_pwm.load to i32, !dbg !114  ; [#uses=1 type=i32] [debug line = 184:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.4, i8* getelementptr inbounds ([8 x i8]* @.str1, i64 0, i64 0), i32 1, i32 1, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !114 ; [debug line = 184:1]
  %mem_addr.load = load volatile i8* @mem_addr, align 1, !dbg !115 ; [#uses=1 type=i8] [debug line = 186:1]
  %tmp.5 = zext i8 %mem_addr.load to i32, !dbg !115 ; [#uses=1 type=i32] [debug line = 186:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.5, i8* getelementptr inbounds ([8 x i8]* @.str1, i64 0, i64 0), i32 1, i32 1, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !115 ; [debug line = 186:1]
  %mem_din.load = load volatile i8* @mem_din, align 1, !dbg !116 ; [#uses=1 type=i8] [debug line = 187:1]
  %tmp.6 = zext i8 %mem_din.load to i32, !dbg !116 ; [#uses=1 type=i32] [debug line = 187:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.6, i8* getelementptr inbounds ([8 x i8]* @.str1, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !116 ; [debug line = 187:1]
  %mem_dout.load = load volatile i8* @mem_dout, align 1, !dbg !117 ; [#uses=1 type=i8] [debug line = 188:1]
  %tmp.7 = zext i8 %mem_dout.load to i32, !dbg !117 ; [#uses=1 type=i32] [debug line = 188:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.7, i8* getelementptr inbounds ([8 x i8]* @.str1, i64 0, i64 0), i32 1, i32 1, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !117 ; [debug line = 188:1]
  %mem_wreq.load = load volatile i1* @mem_wreq, align 1, !dbg !118 ; [#uses=1 type=i1] [debug line = 189:1]
  %tmp.8 = zext i1 %mem_wreq.load to i32, !dbg !118 ; [#uses=1 type=i32] [debug line = 189:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.8, i8* getelementptr inbounds ([8 x i8]* @.str1, i64 0, i64 0), i32 1, i32 1, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !118 ; [debug line = 189:1]
  %mem_wack.load = load volatile i1* @mem_wack, align 1, !dbg !119 ; [#uses=1 type=i1] [debug line = 190:1]
  %tmp.9 = zext i1 %mem_wack.load to i32, !dbg !119 ; [#uses=1 type=i32] [debug line = 190:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.9, i8* getelementptr inbounds ([8 x i8]* @.str1, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !119 ; [debug line = 190:1]
  %mem_rreq.load = load volatile i1* @mem_rreq, align 1, !dbg !120 ; [#uses=1 type=i1] [debug line = 191:1]
  %tmp.10 = zext i1 %mem_rreq.load to i32, !dbg !120 ; [#uses=1 type=i32] [debug line = 191:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.10, i8* getelementptr inbounds ([8 x i8]* @.str1, i64 0, i64 0), i32 1, i32 1, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !120 ; [debug line = 191:1]
  %mem_rack.load = load volatile i1* @mem_rack, align 1, !dbg !121 ; [#uses=1 type=i1] [debug line = 192:1]
  %tmp.11 = zext i1 %mem_rack.load to i32, !dbg !121 ; [#uses=1 type=i32] [debug line = 192:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.11, i8* getelementptr inbounds ([8 x i8]* @.str1, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !121 ; [debug line = 192:1]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !122 ; [debug line = 207:2]
  store volatile i1 false, i1* @l_dir, align 1, !dbg !123 ; [debug line = 208:2]
  store volatile i1 false, i1* @l_pwm, align 1, !dbg !124 ; [debug line = 209:2]
  store volatile i1 false, i1* @r_dir, align 1, !dbg !125 ; [debug line = 210:2]
  store volatile i1 false, i1* @r_pwm, align 1, !dbg !126 ; [debug line = 211:2]
  store volatile i32 0, i32* %mtr_pwm_cnt, align 4, !dbg !127 ; [debug line = 218:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !128 ; [debug line = 220:2]
  call fastcc void @write_mem(i8 zeroext -128, i8 zeroext 0), !dbg !129 ; [debug line = 221:2]
  call fastcc void @write_mem(i8 zeroext -123, i8 zeroext 0), !dbg !130 ; [debug line = 222:2]
  call fastcc void @write_mem(i8 zeroext -122, i8 zeroext 0), !dbg !131 ; [debug line = 223:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !132 ; [debug line = 224:2]
  br label %1, !dbg !133                          ; [debug line = 226:7]

; <label>:1                                       ; preds = %2, %0
  %i = phi i8 [ 0, %0 ], [ %i.1, %2 ]             ; [#uses=3 type=i8]
  %exitcond = icmp eq i8 %i, 32, !dbg !133        ; [#uses=1 type=i1] [debug line = 226:7]
  br i1 %exitcond, label %.preheader.preheader, label %2, !dbg !133 ; [debug line = 226:7]

.preheader.preheader:                             ; preds = %1
  br label %.preheader

; <label>:2                                       ; preds = %1
  call fastcc void @write_mem(i8 zeroext %i, i8 zeroext 32), !dbg !135 ; [debug line = 227:3]
  %i.1 = add i8 %i, 1, !dbg !137                  ; [#uses=1 type=i8] [debug line = 226:22]
  call void @llvm.dbg.value(metadata !{i8 %i.1}, i64 0, metadata !138), !dbg !137 ; [debug line = 226:22] [debug variable = i]
  br label %1, !dbg !137                          ; [debug line = 226:22]

.preheader:                                       ; preds = %._crit_edge11, %.preheader.preheader
  %chR_dir = phi i1 [ %chR_dir.5, %._crit_edge11 ], [ false, %.preheader.preheader ] ; [#uses=3 type=i1]
  %chL_dir = phi i1 [ %chL_dir.5, %._crit_edge11 ], [ false, %.preheader.preheader ] ; [#uses=3 type=i1]
  %loop_begin = call i32 (...)* @_ssdm_op_SpecLoopBegin() nounwind ; [#uses=0 type=i32]
  %rbegin = call i32 (...)* @_ssdm_op_SpecRegionBegin(i8* getelementptr inbounds ([12 x i8]* @.str2, i64 0, i64 0)) nounwind, !dbg !139 ; [#uses=1 type=i32] [debug line = 246:13]
  %eh = call fastcc zeroext i8 @read_mem(i8 zeroext -127), !dbg !141 ; [#uses=3 type=i8] [debug line = 251:8]
  call void @llvm.dbg.value(metadata !{i8 %eh}, i64 0, metadata !142), !dbg !141 ; [debug line = 251:8] [debug variable = eh]
  %el = call fastcc zeroext i8 @read_mem(i8 zeroext -126), !dbg !143 ; [#uses=3 type=i8] [debug line = 252:8]
  call void @llvm.dbg.value(metadata !{i8 %el}, i64 0, metadata !144), !dbg !143 ; [debug line = 252:8] [debug variable = el]
  %et = call i16 @_ssdm_op_BitConcatenate.i16.i8.i8(i8 %eh, i8 %el) ; [#uses=1 type=i16]
  call void @llvm.dbg.value(metadata !{i16 %et}, i64 0, metadata !145), !dbg !146 ; [debug line = 253:3] [debug variable = et]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !147 ; [debug line = 255:3]
  br label %._crit_edge, !dbg !148                ; [debug line = 256:3]

._crit_edge:                                      ; preds = %._crit_edge, %.preheader
  %tmp.16 = call fastcc zeroext i8 @read_mem(i8 zeroext -123), !dbg !149 ; [#uses=1 type=i8] [debug line = 256:10]
  %tmp.17 = icmp eq i8 %tmp.16, 0, !dbg !149      ; [#uses=1 type=i1] [debug line = 256:10]
  br i1 %tmp.17, label %._crit_edge, label %3, !dbg !149 ; [debug line = 256:10]

; <label>:3                                       ; preds = %._crit_edge
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !150 ; [debug line = 258:3]
  %eh.1 = call fastcc zeroext i8 @read_mem(i8 zeroext -125), !dbg !151 ; [#uses=3 type=i8] [debug line = 259:8]
  call void @llvm.dbg.value(metadata !{i8 %eh.1}, i64 0, metadata !142), !dbg !151 ; [debug line = 259:8] [debug variable = eh]
  %el.1 = call fastcc zeroext i8 @read_mem(i8 zeroext -124), !dbg !152 ; [#uses=3 type=i8] [debug line = 260:8]
  call void @llvm.dbg.value(metadata !{i8 %el.1}, i64 0, metadata !144), !dbg !152 ; [debug line = 260:8] [debug variable = el]
  %e = call i16 @_ssdm_op_BitConcatenate.i16.i8.i8(i8 %eh.1, i8 %el.1) ; [#uses=1 type=i16]
  call void @llvm.dbg.value(metadata !{i16 %e}, i64 0, metadata !153), !dbg !154 ; [debug line = 261:3] [debug variable = e]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !155 ; [debug line = 262:3]
  %tmp.21 = lshr i8 %eh.1, 4, !dbg !156           ; [#uses=1 type=i8] [debug line = 265:16]
  %tmp.22 = trunc i8 %tmp.21 to i4, !dbg !156     ; [#uses=1 type=i4] [debug line = 265:16]
  %tmp.23 = call fastcc zeroext i8 @bin2char(i4 zeroext %tmp.22), !dbg !156 ; [#uses=1 type=i8] [debug line = 265:16]
  call fastcc void @write_mem(i8 zeroext 3, i8 zeroext %tmp.23), !dbg !156 ; [debug line = 265:16]
  %tmp.24 = trunc i8 %eh.1 to i4, !dbg !157       ; [#uses=1 type=i4] [debug line = 266:16]
  %tmp.25 = call fastcc zeroext i8 @bin2char(i4 zeroext %tmp.24), !dbg !157 ; [#uses=1 type=i8] [debug line = 266:16]
  call fastcc void @write_mem(i8 zeroext 4, i8 zeroext %tmp.25), !dbg !157 ; [debug line = 266:16]
  %tmp.26 = lshr i8 %el.1, 4, !dbg !158           ; [#uses=1 type=i8] [debug line = 267:16]
  %tmp.27 = trunc i8 %tmp.26 to i4, !dbg !158     ; [#uses=1 type=i4] [debug line = 267:16]
  %tmp.28 = call fastcc zeroext i8 @bin2char(i4 zeroext %tmp.27), !dbg !158 ; [#uses=1 type=i8] [debug line = 267:16]
  call fastcc void @write_mem(i8 zeroext 5, i8 zeroext %tmp.28), !dbg !158 ; [debug line = 267:16]
  %tmp.29 = trunc i8 %el.1 to i4, !dbg !159       ; [#uses=1 type=i4] [debug line = 268:16]
  %tmp.30 = call fastcc zeroext i8 @bin2char(i4 zeroext %tmp.29), !dbg !159 ; [#uses=1 type=i8] [debug line = 268:16]
  call fastcc void @write_mem(i8 zeroext 6, i8 zeroext %tmp.30), !dbg !159 ; [debug line = 268:16]
  %tmp.31 = lshr i8 %eh, 4, !dbg !160             ; [#uses=1 type=i8] [debug line = 270:16]
  %tmp.32 = trunc i8 %tmp.31 to i4, !dbg !160     ; [#uses=1 type=i4] [debug line = 270:16]
  %tmp.33 = call fastcc zeroext i8 @bin2char(i4 zeroext %tmp.32), !dbg !160 ; [#uses=1 type=i8] [debug line = 270:16]
  call fastcc void @write_mem(i8 zeroext 8, i8 zeroext %tmp.33), !dbg !160 ; [debug line = 270:16]
  %tmp.34 = trunc i8 %eh to i4, !dbg !161         ; [#uses=1 type=i4] [debug line = 271:16]
  %tmp.35 = call fastcc zeroext i8 @bin2char(i4 zeroext %tmp.34), !dbg !161 ; [#uses=1 type=i8] [debug line = 271:16]
  call fastcc void @write_mem(i8 zeroext 9, i8 zeroext %tmp.35), !dbg !161 ; [debug line = 271:16]
  %tmp.36 = lshr i8 %el, 4, !dbg !162             ; [#uses=1 type=i8] [debug line = 272:17]
  %tmp.37 = trunc i8 %tmp.36 to i4, !dbg !162     ; [#uses=1 type=i4] [debug line = 272:17]
  %tmp.38 = call fastcc zeroext i8 @bin2char(i4 zeroext %tmp.37), !dbg !162 ; [#uses=1 type=i8] [debug line = 272:17]
  call fastcc void @write_mem(i8 zeroext 10, i8 zeroext %tmp.38), !dbg !162 ; [debug line = 272:17]
  %tmp.39 = trunc i8 %el to i4, !dbg !163         ; [#uses=1 type=i4] [debug line = 273:17]
  %tmp.40 = call fastcc zeroext i8 @bin2char(i4 zeroext %tmp.39), !dbg !163 ; [#uses=1 type=i8] [debug line = 273:17]
  call fastcc void @write_mem(i8 zeroext 11, i8 zeroext %tmp.40), !dbg !163 ; [debug line = 273:17]
  %diff_agl = call fastcc i32 @diff_angle(i16 zeroext %et, i16 zeroext %e), !dbg !164 ; [#uses=6 type=i32] [debug line = 281:14]
  call void @llvm.dbg.value(metadata !{i32 %diff_agl}, i64 0, metadata !165), !dbg !164 ; [debug line = 281:14] [debug variable = diff_agl]
  %tmp.41 = lshr i32 %diff_agl, 12, !dbg !166     ; [#uses=1 type=i32] [debug line = 284:17]
  %tmp.42 = trunc i32 %tmp.41 to i4, !dbg !166    ; [#uses=1 type=i4] [debug line = 284:17]
  %tmp.43 = call fastcc zeroext i8 @bin2char(i4 zeroext %tmp.42), !dbg !166 ; [#uses=1 type=i8] [debug line = 284:17]
  call fastcc void @write_mem(i8 zeroext 16, i8 zeroext %tmp.43), !dbg !166 ; [debug line = 284:17]
  %tmp.44 = lshr i32 %diff_agl, 8, !dbg !167      ; [#uses=1 type=i32] [debug line = 285:17]
  %tmp.45 = trunc i32 %tmp.44 to i4, !dbg !167    ; [#uses=1 type=i4] [debug line = 285:17]
  %tmp.46 = call fastcc zeroext i8 @bin2char(i4 zeroext %tmp.45), !dbg !167 ; [#uses=1 type=i8] [debug line = 285:17]
  call fastcc void @write_mem(i8 zeroext 17, i8 zeroext %tmp.46), !dbg !167 ; [debug line = 285:17]
  %tmp.47 = lshr i32 %diff_agl, 4, !dbg !168      ; [#uses=1 type=i32] [debug line = 286:17]
  %tmp.48 = trunc i32 %tmp.47 to i4, !dbg !168    ; [#uses=1 type=i4] [debug line = 286:17]
  %tmp.49 = call fastcc zeroext i8 @bin2char(i4 zeroext %tmp.48), !dbg !168 ; [#uses=1 type=i8] [debug line = 286:17]
  call fastcc void @write_mem(i8 zeroext 18, i8 zeroext %tmp.49), !dbg !168 ; [debug line = 286:17]
  %tmp.50 = trunc i32 %diff_agl to i4, !dbg !169  ; [#uses=1 type=i4] [debug line = 287:17]
  %tmp.51 = call fastcc zeroext i8 @bin2char(i4 zeroext %tmp.50), !dbg !169 ; [#uses=1 type=i8] [debug line = 287:17]
  call fastcc void @write_mem(i8 zeroext 19, i8 zeroext %tmp.51), !dbg !169 ; [debug line = 287:17]
  %tmp.52 = icmp slt i32 %diff_agl, -249, !dbg !170 ; [#uses=1 type=i1] [debug line = 289:3]
  br i1 %tmp.52, label %4, label %5, !dbg !170    ; [debug line = 289:3]

; <label>:4                                       ; preds = %3
  call fastcc void @write_mem(i8 zeroext 21, i8 zeroext 76), !dbg !171 ; [debug line = 291:4]
  br label %6, !dbg !173                          ; [debug line = 292:3]

; <label>:5                                       ; preds = %3
  call fastcc void @write_mem(i8 zeroext 21, i8 zeroext 45), !dbg !174 ; [debug line = 295:4]
  br label %6

; <label>:6                                       ; preds = %5, %4
  %too_left = phi i1 [ true, %4 ], [ false, %5 ]  ; [#uses=6 type=i1]
  %tmp.53 = icmp sgt i32 %diff_agl, 249, !dbg !176 ; [#uses=1 type=i1] [debug line = 298:3]
  br i1 %tmp.53, label %7, label %8, !dbg !176    ; [debug line = 298:3]

; <label>:7                                       ; preds = %6
  call fastcc void @write_mem(i8 zeroext 22, i8 zeroext 82), !dbg !177 ; [debug line = 300:4]
  br label %9, !dbg !179                          ; [debug line = 301:3]

; <label>:8                                       ; preds = %6
  call fastcc void @write_mem(i8 zeroext 22, i8 zeroext 45), !dbg !180 ; [debug line = 304:4]
  br label %9

; <label>:9                                       ; preds = %8, %7
  %too_right = phi i1 [ true, %7 ], [ false, %8 ] ; [#uses=10 type=i1]
  %mode = call fastcc zeroext i8 @read_mem(i8 zeroext -128), !dbg !182 ; [#uses=2 type=i8] [debug line = 308:10]
  call void @llvm.dbg.value(metadata !{i8 %mode}, i64 0, metadata !183), !dbg !182 ; [debug line = 308:10] [debug variable = mode]
  %tmp.54 = lshr i8 %mode, 3, !dbg !184           ; [#uses=1 type=i8] [debug line = 310:3]
  %chR_pwm = udiv i8 %tmp.54, 3, !dbg !184        ; [#uses=4 type=i8] [debug line = 310:3]
  call void @llvm.dbg.value(metadata !{i8 %chR_pwm}, i64 0, metadata !185), !dbg !184 ; [debug line = 310:3] [debug variable = chR_pwm]
  call void @llvm.dbg.value(metadata !{i8 %chR_pwm}, i64 0, metadata !186), !dbg !184 ; [debug line = 310:3] [debug variable = chL_pwm]
  %tmp.56 = zext i8 %mode to i32, !dbg !187       ; [#uses=1 type=i32] [debug line = 312:3]
  %tmp.57 = and i32 %tmp.56, 7, !dbg !187         ; [#uses=1 type=i32] [debug line = 312:3]
  switch i32 %tmp.57, label %._crit_edge6 [
    i32 1, label %10
    i32 3, label %12
    i32 5, label %14
    i32 7, label %16
  ], !dbg !187                                    ; [debug line = 312:3]

; <label>:10                                      ; preds = %9
  br i1 %too_right, label %._crit_edge6, label %11, !dbg !188 ; [debug line = 317:4]

; <label>:11                                      ; preds = %10
  %. = select i1 %too_left, i8 0, i8 %chR_pwm, !dbg !190 ; [#uses=1 type=i8] [debug line = 321:9]
  %.1 = select i1 %too_left, i8 10, i8 %chR_pwm, !dbg !190 ; [#uses=1 type=i8] [debug line = 321:9]
  br label %._crit_edge6

; <label>:12                                      ; preds = %9
  br i1 %too_right, label %._crit_edge6, label %13, !dbg !191 ; [debug line = 331:4]

; <label>:13                                      ; preds = %12
  %.2 = select i1 %too_left, i8 10, i8 %chR_pwm, !dbg !192 ; [#uses=1 type=i8] [debug line = 335:9]
  %.3 = select i1 %too_left, i8 0, i8 %chR_pwm, !dbg !192 ; [#uses=1 type=i8] [debug line = 335:9]
  br label %._crit_edge6

; <label>:14                                      ; preds = %9
  %brmerge = or i1 %too_right, %too_left, !dbg !193 ; [#uses=1 type=i1] [debug line = 342:4]
  br i1 %brmerge, label %15, label %._crit_edge6, !dbg !193 ; [debug line = 342:4]

; <label>:15                                      ; preds = %14
  %chL_pwm.4 = select i1 %too_right, i8 0, i8 10, !dbg !194 ; [#uses=1 type=i8] [debug line = 347:5]
  %not.too_right. = xor i1 %too_right, true, !dbg !194 ; [#uses=1 type=i1] [debug line = 347:5]
  br label %._crit_edge6

; <label>:16                                      ; preds = %9
  %brmerge4 = or i1 %too_right, %too_left, !dbg !196 ; [#uses=1 type=i1] [debug line = 363:4]
  br i1 %brmerge4, label %17, label %._crit_edge6, !dbg !196 ; [debug line = 363:4]

; <label>:17                                      ; preds = %16
  %chR_pwm.6 = select i1 %too_right, i8 10, i8 0, !dbg !197 ; [#uses=1 type=i8] [debug line = 368:5]
  %not.too_right = xor i1 %too_right, true, !dbg !197 ; [#uses=1 type=i1] [debug line = 368:5]
  br label %._crit_edge6

._crit_edge6:                                     ; preds = %17, %16, %15, %14, %13, %12, %11, %10, %9
  %chR_pwm.8 = phi i8 [ %., %11 ], [ %.2, %13 ], [ 10, %15 ], [ %chR_pwm.6, %17 ], [ 0, %9 ], [ 10, %10 ], [ 0, %12 ], [ 0, %14 ], [ 0, %16 ] ; [#uses=3 type=i8]
  %chL_pwm.8 = phi i8 [ %.1, %11 ], [ %.3, %13 ], [ %chL_pwm.4, %15 ], [ 10, %17 ], [ 0, %9 ], [ 0, %10 ], [ 10, %12 ], [ 0, %14 ], [ 0, %16 ] ; [#uses=3 type=i8]
  %chR_dir.5 = phi i1 [ false, %11 ], [ true, %13 ], [ %not.too_right., %15 ], [ %not.too_right, %17 ], [ %chR_dir, %9 ], [ false, %10 ], [ true, %12 ], [ %chR_dir, %14 ], [ %chR_dir, %16 ] ; [#uses=2 type=i1]
  %chL_dir.5 = phi i1 [ false, %11 ], [ true, %13 ], [ %too_right, %15 ], [ %too_right, %17 ], [ %chL_dir, %9 ], [ false, %10 ], [ true, %12 ], [ %chL_dir, %14 ], [ %chL_dir, %16 ] ; [#uses=2 type=i1]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !199 ; [debug line = 390:3]
  store volatile i1 %chR_dir.5, i1* @r_dir, align 1, !dbg !200 ; [debug line = 391:3]
  call void @llvm.dbg.value(metadata !{i32* %mtr_pwm_cnt}, i64 0, metadata !201), !dbg !204 ; [debug line = 392:3] [debug variable = mtr_pwm_cnt]
  %mtr_pwm_cnt.load = load volatile i32* %mtr_pwm_cnt, align 4, !dbg !204 ; [#uses=1 type=i32] [debug line = 392:3]
  %tmp.58 = zext i8 %chR_pwm.8 to i32, !dbg !204  ; [#uses=1 type=i32] [debug line = 392:3]
  %tmp.59 = icmp slt i32 %mtr_pwm_cnt.load, %tmp.58, !dbg !204 ; [#uses=1 type=i1] [debug line = 392:3]
  br i1 %tmp.59, label %18, label %19, !dbg !204  ; [debug line = 392:3]

; <label>:18                                      ; preds = %._crit_edge6
  store volatile i1 true, i1* @r_pwm, align 1, !dbg !205 ; [debug line = 393:4]
  br label %20, !dbg !205                         ; [debug line = 393:4]

; <label>:19                                      ; preds = %._crit_edge6
  store volatile i1 false, i1* @r_pwm, align 1, !dbg !206 ; [debug line = 395:4]
  br label %20

; <label>:20                                      ; preds = %19, %18
  store volatile i1 %chL_dir.5, i1* @l_dir, align 1, !dbg !207 ; [debug line = 397:3]
  call void @llvm.dbg.value(metadata !{i32* %mtr_pwm_cnt}, i64 0, metadata !201), !dbg !208 ; [debug line = 398:3] [debug variable = mtr_pwm_cnt]
  %mtr_pwm_cnt.load.1 = load volatile i32* %mtr_pwm_cnt, align 4, !dbg !208 ; [#uses=1 type=i32] [debug line = 398:3]
  %tmp.60 = zext i8 %chL_pwm.8 to i32, !dbg !208  ; [#uses=1 type=i32] [debug line = 398:3]
  %tmp.61 = icmp slt i32 %mtr_pwm_cnt.load.1, %tmp.60, !dbg !208 ; [#uses=1 type=i1] [debug line = 398:3]
  br i1 %tmp.61, label %21, label %22, !dbg !208  ; [debug line = 398:3]

; <label>:21                                      ; preds = %20
  store volatile i1 true, i1* @l_pwm, align 1, !dbg !209 ; [debug line = 399:4]
  br label %23, !dbg !209                         ; [debug line = 399:4]

; <label>:22                                      ; preds = %20
  store volatile i1 false, i1* @l_pwm, align 1, !dbg !210 ; [debug line = 401:4]
  br label %23

; <label>:23                                      ; preds = %22, %21
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !211 ; [debug line = 402:3]
  call void @llvm.dbg.value(metadata !{i32* %mtr_pwm_cnt}, i64 0, metadata !201), !dbg !212 ; [debug line = 405:3] [debug variable = mtr_pwm_cnt]
  %mtr_pwm_cnt.load.2 = load volatile i32* %mtr_pwm_cnt, align 4, !dbg !212 ; [#uses=1 type=i32] [debug line = 405:3]
  %mtr_pwm_cnt.1 = add nsw i32 %mtr_pwm_cnt.load.2, 1, !dbg !212 ; [#uses=1 type=i32] [debug line = 405:3]
  call void @llvm.dbg.value(metadata !{i32 %mtr_pwm_cnt.1}, i64 0, metadata !201), !dbg !212 ; [debug line = 405:3] [debug variable = mtr_pwm_cnt]
  store volatile i32 %mtr_pwm_cnt.1, i32* %mtr_pwm_cnt, align 4, !dbg !212 ; [debug line = 405:3]
  call void @llvm.dbg.value(metadata !{i32* %mtr_pwm_cnt}, i64 0, metadata !201), !dbg !213 ; [debug line = 406:3] [debug variable = mtr_pwm_cnt]
  %mtr_pwm_cnt.load.3 = load volatile i32* %mtr_pwm_cnt, align 4, !dbg !213 ; [#uses=1 type=i32] [debug line = 406:3]
  %tmp.63 = icmp sgt i32 %mtr_pwm_cnt.load.3, 9, !dbg !213 ; [#uses=1 type=i1] [debug line = 406:3]
  br i1 %tmp.63, label %24, label %._crit_edge11, !dbg !213 ; [debug line = 406:3]

; <label>:24                                      ; preds = %23
  store volatile i32 0, i32* %mtr_pwm_cnt, align 4, !dbg !214 ; [debug line = 407:4]
  br label %._crit_edge11, !dbg !216              ; [debug line = 408:3]

._crit_edge11:                                    ; preds = %24, %23
  %tmp.64 = lshr i8 %chL_pwm.8, 4, !dbg !217      ; [#uses=1 type=i8] [debug line = 411:17]
  %tmp.65 = trunc i8 %tmp.64 to i4, !dbg !217     ; [#uses=1 type=i4] [debug line = 411:17]
  %tmp.66 = call fastcc zeroext i8 @bin2char(i4 zeroext %tmp.65), !dbg !217 ; [#uses=1 type=i8] [debug line = 411:17]
  call fastcc void @write_mem(i8 zeroext 24, i8 zeroext %tmp.66), !dbg !217 ; [debug line = 411:17]
  %tmp.67 = trunc i8 %chL_pwm.8 to i4, !dbg !218  ; [#uses=1 type=i4] [debug line = 412:17]
  %tmp.68 = call fastcc zeroext i8 @bin2char(i4 zeroext %tmp.67), !dbg !218 ; [#uses=1 type=i8] [debug line = 412:17]
  call fastcc void @write_mem(i8 zeroext 25, i8 zeroext %tmp.68), !dbg !218 ; [debug line = 412:17]
  %tmp.69 = lshr i8 %chR_pwm.8, 4, !dbg !219      ; [#uses=1 type=i8] [debug line = 414:17]
  %tmp.70 = trunc i8 %tmp.69 to i4, !dbg !219     ; [#uses=1 type=i4] [debug line = 414:17]
  %tmp.71 = call fastcc zeroext i8 @bin2char(i4 zeroext %tmp.70), !dbg !219 ; [#uses=1 type=i8] [debug line = 414:17]
  call fastcc void @write_mem(i8 zeroext 27, i8 zeroext %tmp.71), !dbg !219 ; [debug line = 414:17]
  %tmp.72 = trunc i8 %chR_pwm.8 to i4, !dbg !220  ; [#uses=1 type=i4] [debug line = 415:17]
  %tmp.73 = call fastcc zeroext i8 @bin2char(i4 zeroext %tmp.72), !dbg !220 ; [#uses=1 type=i8] [debug line = 415:17]
  call fastcc void @write_mem(i8 zeroext 28, i8 zeroext %tmp.73), !dbg !220 ; [debug line = 415:17]
  call fastcc void @wait_tmr(), !dbg !221         ; [debug line = 420:3]
  %rend = call i32 (...)* @_ssdm_op_SpecRegionEnd(i8* getelementptr inbounds ([12 x i8]* @.str2, i64 0, i64 0), i32 %rbegin) nounwind, !dbg !222 ; [#uses=0 type=i32] [debug line = 421:2]
  br label %.preheader, !dbg !222                 ; [debug line = 421:2]
}

; [#uses=25]
declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

; [#uses=1]
define internal fastcc i32 @diff_angle(i16 zeroext %target, i16 zeroext %value) nounwind uwtable {
  call void @llvm.dbg.value(metadata !{i16 %target}, i64 0, metadata !223), !dbg !224 ; [debug line = 133:25] [debug variable = target]
  call void @llvm.dbg.value(metadata !{i16 %value}, i64 0, metadata !225), !dbg !226 ; [debug line = 133:40] [debug variable = value]
  call void (...)* @_ssdm_InlineSelf(i32 2, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !227 ; [debug line = 135:1]
  %tmp = zext i16 %value to i32, !dbg !229        ; [#uses=3 type=i32] [debug line = 138:2]
  %.neg = sub i32 0, %tmp                         ; [#uses=2 type=i32]
  %tmp.74 = zext i16 %target to i32, !dbg !229    ; [#uses=4 type=i32] [debug line = 138:2]
  %tmp.75 = sub nsw i32 %tmp, %tmp.74, !dbg !229  ; [#uses=1 type=i32] [debug line = 138:2]
  %tmp.76 = add i32 %tmp.74, -1, !dbg !230        ; [#uses=1 type=i32] [debug line = 139:2]
  %tmp.77 = add i32 %tmp.76, %.neg, !dbg !230     ; [#uses=2 type=i32] [debug line = 139:2]
  %tmp.78 = icmp sgt i32 %tmp.77, -18001          ; [#uses=1 type=i1]
  %smax2 = select i1 %tmp.78, i32 %tmp.77, i32 -18001 ; [#uses=1 type=i32]
  %.neg9 = sub i32 0, %tmp.74                     ; [#uses=1 type=i32]
  %tmp.79 = add i32 %.neg9, 36000, !dbg !230      ; [#uses=1 type=i32] [debug line = 139:2]
  %tmp.80 = add i32 %tmp.79, %tmp, !dbg !230      ; [#uses=1 type=i32] [debug line = 139:2]
  %tmp.81 = add i32 %tmp.80, %smax2, !dbg !230    ; [#uses=2 type=i32] [debug line = 139:2]
  %tmp.82 = urem i32 %tmp.81, 36000, !dbg !230    ; [#uses=1 type=i32] [debug line = 139:2]
  %tmp.83 = sub i32 %tmp.81, %tmp.82, !dbg !231   ; [#uses=2 type=i32] [debug line = 141:2]
  %.neg1 = sub i32 0, %tmp.83                     ; [#uses=1 type=i32]
  %tmp.84 = add i32 %.neg1, %tmp.75, !dbg !231    ; [#uses=3 type=i32] [debug line = 141:2]
  %tmp.85 = icmp sgt i32 %tmp.84, -18000          ; [#uses=1 type=i1]
  %smax1 = select i1 %tmp.85, i32 %tmp.84, i32 -18000 ; [#uses=1 type=i32]
  %tmp.86 = add i32 %tmp.74, 35999, !dbg !231     ; [#uses=1 type=i32] [debug line = 141:2]
  %tmp.87 = add i32 %tmp.86, %.neg, !dbg !231     ; [#uses=1 type=i32] [debug line = 141:2]
  %tmp.88 = add i32 %tmp.87, %tmp.83, !dbg !231   ; [#uses=1 type=i32] [debug line = 141:2]
  %tmp.89 = add i32 %tmp.88, %smax1, !dbg !231    ; [#uses=2 type=i32] [debug line = 141:2]
  %tmp.90 = urem i32 %tmp.89, 36000, !dbg !231    ; [#uses=1 type=i32] [debug line = 141:2]
  %.neg2 = sub i32 0, %tmp.90                     ; [#uses=1 type=i32]
  %tmp.91 = add i32 %tmp.84, %.neg2, !dbg !231    ; [#uses=1 type=i32] [debug line = 141:2]
  %retval = add i32 %tmp.91, %tmp.89, !dbg !229   ; [#uses=1 type=i32] [debug line = 138:2]
  call void @llvm.dbg.value(metadata !{i32 %retval}, i64 0, metadata !232), !dbg !229 ; [debug line = 138:2] [debug variable = retval]
  ret i32 %retval, !dbg !233                      ; [debug line = 144:2]
}

; [#uses=16]
define internal fastcc zeroext i8 @bin2char(i4 zeroext %val) nounwind uwtable {
  call void @llvm.dbg.value(metadata !{i4 %val}, i64 0, metadata !234), !dbg !235 ; [debug line = 148:22] [debug variable = val]
  call void (...)* @_ssdm_InlineSelf(i32 2, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !236 ; [debug line = 150:1]
  %tmp = zext i4 %val to i32, !dbg !238           ; [#uses=1 type=i32] [debug line = 153:2]
  switch i32 %tmp, label %15 [
    i32 0, label %._crit_edge
    i32 1, label %1
    i32 2, label %2
    i32 3, label %3
    i32 4, label %4
    i32 5, label %5
    i32 6, label %6
    i32 7, label %7
    i32 8, label %8
    i32 9, label %9
    i32 10, label %10
    i32 11, label %11
    i32 12, label %12
    i32 13, label %13
    i32 14, label %14
  ], !dbg !238                                    ; [debug line = 153:2]

; <label>:1                                       ; preds = %0
  br label %._crit_edge, !dbg !239                ; [debug line = 155:24]

; <label>:2                                       ; preds = %0
  br label %._crit_edge, !dbg !241                ; [debug line = 156:24]

; <label>:3                                       ; preds = %0
  br label %._crit_edge, !dbg !242                ; [debug line = 157:24]

; <label>:4                                       ; preds = %0
  br label %._crit_edge, !dbg !243                ; [debug line = 158:24]

; <label>:5                                       ; preds = %0
  br label %._crit_edge, !dbg !244                ; [debug line = 159:24]

; <label>:6                                       ; preds = %0
  br label %._crit_edge, !dbg !245                ; [debug line = 160:24]

; <label>:7                                       ; preds = %0
  br label %._crit_edge, !dbg !246                ; [debug line = 161:24]

; <label>:8                                       ; preds = %0
  br label %._crit_edge, !dbg !247                ; [debug line = 162:24]

; <label>:9                                       ; preds = %0
  br label %._crit_edge, !dbg !248                ; [debug line = 163:24]

; <label>:10                                      ; preds = %0
  br label %._crit_edge, !dbg !249                ; [debug line = 164:25]

; <label>:11                                      ; preds = %0
  br label %._crit_edge, !dbg !250                ; [debug line = 165:25]

; <label>:12                                      ; preds = %0
  br label %._crit_edge, !dbg !251                ; [debug line = 166:25]

; <label>:13                                      ; preds = %0
  br label %._crit_edge, !dbg !252                ; [debug line = 167:25]

; <label>:14                                      ; preds = %0
  br label %._crit_edge, !dbg !253                ; [debug line = 168:25]

; <label>:15                                      ; preds = %0
  br label %._crit_edge, !dbg !254                ; [debug line = 170:2]

._crit_edge:                                      ; preds = %15, %14, %13, %12, %11, %10, %9, %8, %7, %6, %5, %4, %3, %2, %1, %0
  %retval = phi i8 [ 70, %15 ], [ 69, %14 ], [ 68, %13 ], [ 67, %12 ], [ 66, %11 ], [ 65, %10 ], [ 57, %9 ], [ 56, %8 ], [ 55, %7 ], [ 54, %6 ], [ 53, %5 ], [ 52, %4 ], [ 51, %3 ], [ 50, %2 ], [ 49, %1 ], [ 48, %0 ] ; [#uses=1 type=i8]
  ret i8 %retval, !dbg !255                       ; [debug line = 172:2]
}

; [#uses=18]
declare void @_ssdm_op_Wait(...) nounwind

; [#uses=1]
declare void @_ssdm_op_SpecTopModule(...) nounwind

; [#uses=1]
declare i32 @_ssdm_op_SpecRegionEnd(...)

; [#uses=1]
declare i32 @_ssdm_op_SpecRegionBegin(...)

; [#uses=1]
declare i32 @_ssdm_op_SpecLoopBegin(...)

; [#uses=12]
declare void @_ssdm_op_SpecInterface(...) nounwind

; [#uses=2]
declare i16 @_ssdm_op_BitConcatenate.i16.i8.i8(i8, i8) nounwind readnone

; [#uses=5]
declare void @_ssdm_InlineSelf(...) nounwind

!llvm.dbg.cu = !{!0}
!hls.encrypted.func = !{}

!0 = metadata !{i32 786449, i32 0, i32 1, metadata !"D:/21_streamer_car5_artix7/fpga_arty/motor_ctrl/solution1/.autopilot/db/motor_ctrl.pragma.2.c", metadata !"D:\5C21_streamer_car5_artix7\5Cfpga_arty", metadata !"clang version 3.1 ", i1 true, i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !36} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !5, metadata !13, metadata !18, metadata !21, metadata !28, metadata !33}
!5 = metadata !{i32 786478, i32 0, metadata !6, metadata !"wait_tmr", metadata !"wait_tmr", metadata !"", metadata !6, i32 73, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, metadata !11, i32 74} ; [ DW_TAG_subprogram ]
!6 = metadata !{i32 786473, metadata !"motor_ctrl.c", metadata !"D:\5C21_streamer_car5_artix7\5Cfpga_arty", null} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!8 = metadata !{null, metadata !9}
!9 = metadata !{i32 786454, null, metadata !"uint32", metadata !6, i32 34, i64 0, i64 0, i64 0, i32 0, metadata !10} ; [ DW_TAG_typedef ]
!10 = metadata !{i32 786468, null, metadata !"uint32", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!11 = metadata !{metadata !12}
!12 = metadata !{i32 786468}                      ; [ DW_TAG_base_type ]
!13 = metadata !{i32 786478, i32 0, metadata !6, metadata !"write_mem", metadata !"write_mem", metadata !"", metadata !6, i32 85, metadata !14, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (i8, i8)* @write_mem, null, null, metadata !11, i32 86} ; [ DW_TAG_subprogram ]
!14 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !15, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!15 = metadata !{null, metadata !16, metadata !16}
!16 = metadata !{i32 786454, null, metadata !"uint8", metadata !6, i32 10, i64 0, i64 0, i64 0, i32 0, metadata !17} ; [ DW_TAG_typedef ]
!17 = metadata !{i32 786468, null, metadata !"uint8", null, i32 0, i64 8, i64 8, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!18 = metadata !{i32 786478, i32 0, metadata !6, metadata !"read_mem", metadata !"read_mem", metadata !"", metadata !6, i32 108, metadata !19, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i8 (i8)* @read_mem, null, null, metadata !11, i32 109} ; [ DW_TAG_subprogram ]
!19 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !20, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!20 = metadata !{metadata !16, metadata !16}
!21 = metadata !{i32 786478, i32 0, metadata !6, metadata !"diff_angle", metadata !"diff_angle", metadata !"", metadata !6, i32 133, metadata !22, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (i16, i16)* @diff_angle, null, null, metadata !11, i32 134} ; [ DW_TAG_subprogram ]
!22 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !23, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!23 = metadata !{metadata !24, metadata !26, metadata !26}
!24 = metadata !{i32 786454, null, metadata !"int32", metadata !6, i32 34, i64 0, i64 0, i64 0, i32 0, metadata !25} ; [ DW_TAG_typedef ]
!25 = metadata !{i32 786468, null, metadata !"int32", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!26 = metadata !{i32 786454, null, metadata !"uint16", metadata !6, i32 18, i64 0, i64 0, i64 0, i32 0, metadata !27} ; [ DW_TAG_typedef ]
!27 = metadata !{i32 786468, null, metadata !"uint16", null, i32 0, i64 16, i64 16, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!28 = metadata !{i32 786478, i32 0, metadata !6, metadata !"bin2char", metadata !"bin2char", metadata !"", metadata !6, i32 148, metadata !29, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i8 (i4)* @bin2char, null, null, metadata !11, i32 149} ; [ DW_TAG_subprogram ]
!29 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !30, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!30 = metadata !{metadata !16, metadata !31}
!31 = metadata !{i32 786454, null, metadata !"uint4", metadata !6, i32 6, i64 0, i64 0, i64 0, i32 0, metadata !32} ; [ DW_TAG_typedef ]
!32 = metadata !{i32 786468, null, metadata !"uint4", null, i32 0, i64 4, i64 4, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!33 = metadata !{i32 786478, i32 0, metadata !6, metadata !"motor_ctrl", metadata !"motor_ctrl", metadata !"", metadata !6, i32 176, metadata !34, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 false, void ()* @motor_ctrl, null, null, metadata !11, i32 177} ; [ DW_TAG_subprogram ]
!34 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !35, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!35 = metadata !{null}
!36 = metadata !{metadata !37}
!37 = metadata !{metadata !38, metadata !42, metadata !43, metadata !44, metadata !45, metadata !47, metadata !48, metadata !49, metadata !50, metadata !51, metadata !52, metadata !53}
!38 = metadata !{i32 786484, i32 0, null, metadata !"l_dir", metadata !"l_dir", metadata !"", metadata !6, i32 52, metadata !39, i32 0, i32 1, i1* @l_dir} ; [ DW_TAG_variable ]
!39 = metadata !{i32 786485, null, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !40} ; [ DW_TAG_volatile_type ]
!40 = metadata !{i32 786454, null, metadata !"uint1", metadata !6, i32 3, i64 0, i64 0, i64 0, i32 0, metadata !41} ; [ DW_TAG_typedef ]
!41 = metadata !{i32 786468, null, metadata !"uint1", null, i32 0, i64 1, i64 1, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!42 = metadata !{i32 786484, i32 0, null, metadata !"l_pwm", metadata !"l_pwm", metadata !"", metadata !6, i32 53, metadata !39, i32 0, i32 1, i1* @l_pwm} ; [ DW_TAG_variable ]
!43 = metadata !{i32 786484, i32 0, null, metadata !"r_dir", metadata !"r_dir", metadata !"", metadata !6, i32 54, metadata !39, i32 0, i32 1, i1* @r_dir} ; [ DW_TAG_variable ]
!44 = metadata !{i32 786484, i32 0, null, metadata !"r_pwm", metadata !"r_pwm", metadata !"", metadata !6, i32 55, metadata !39, i32 0, i32 1, i1* @r_pwm} ; [ DW_TAG_variable ]
!45 = metadata !{i32 786484, i32 0, null, metadata !"mem_addr", metadata !"mem_addr", metadata !"", metadata !6, i32 58, metadata !46, i32 0, i32 1, i8* @mem_addr} ; [ DW_TAG_variable ]
!46 = metadata !{i32 786485, null, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !16} ; [ DW_TAG_volatile_type ]
!47 = metadata !{i32 786484, i32 0, null, metadata !"mem_dout", metadata !"mem_dout", metadata !"", metadata !6, i32 60, metadata !46, i32 0, i32 1, i8* @mem_dout} ; [ DW_TAG_variable ]
!48 = metadata !{i32 786484, i32 0, null, metadata !"mem_wreq", metadata !"mem_wreq", metadata !"", metadata !6, i32 61, metadata !39, i32 0, i32 1, i1* @mem_wreq} ; [ DW_TAG_variable ]
!49 = metadata !{i32 786484, i32 0, null, metadata !"mem_rreq", metadata !"mem_rreq", metadata !"", metadata !6, i32 63, metadata !39, i32 0, i32 1, i1* @mem_rreq} ; [ DW_TAG_variable ]
!50 = metadata !{i32 786484, i32 0, null, metadata !"dummy_tmr_out", metadata !"dummy_tmr_out", metadata !"", metadata !6, i32 66, metadata !39, i32 0, i32 1, i1* @dummy_tmr_out} ; [ DW_TAG_variable ]
!51 = metadata !{i32 786484, i32 0, null, metadata !"mem_din", metadata !"mem_din", metadata !"", metadata !6, i32 59, metadata !46, i32 0, i32 1, i8* @mem_din} ; [ DW_TAG_variable ]
!52 = metadata !{i32 786484, i32 0, null, metadata !"mem_wack", metadata !"mem_wack", metadata !"", metadata !6, i32 62, metadata !39, i32 0, i32 1, i1* @mem_wack} ; [ DW_TAG_variable ]
!53 = metadata !{i32 786484, i32 0, null, metadata !"mem_rack", metadata !"mem_rack", metadata !"", metadata !6, i32 64, metadata !39, i32 0, i32 1, i1* @mem_rack} ; [ DW_TAG_variable ]
!54 = metadata !{i32 786689, metadata !13, metadata !"addr", metadata !6, i32 16777301, metadata !16, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!55 = metadata !{i32 85, i32 22, metadata !13, null}
!56 = metadata !{i32 786689, metadata !13, metadata !"data", metadata !6, i32 33554517, metadata !16, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!57 = metadata !{i32 85, i32 34, metadata !13, null}
!58 = metadata !{i32 87, i32 1, metadata !59, null}
!59 = metadata !{i32 786443, metadata !13, i32 86, i32 1, metadata !6, i32 3} ; [ DW_TAG_lexical_block ]
!60 = metadata !{i32 88, i32 2, metadata !59, null}
!61 = metadata !{i32 89, i32 2, metadata !59, null}
!62 = metadata !{i32 90, i32 2, metadata !59, null}
!63 = metadata !{i32 91, i32 2, metadata !59, null}
!64 = metadata !{i32 92, i32 2, metadata !59, null}
!65 = metadata !{i32 94, i32 2, metadata !59, null}
!66 = metadata !{i32 95, i32 3, metadata !67, null}
!67 = metadata !{i32 786443, metadata !59, i32 94, i32 5, metadata !6, i32 4} ; [ DW_TAG_lexical_block ]
!68 = metadata !{i32 96, i32 3, metadata !67, null}
!69 = metadata !{i32 97, i32 3, metadata !67, null}
!70 = metadata !{i32 98, i32 2, metadata !67, null}
!71 = metadata !{i32 99, i32 2, metadata !59, null}
!72 = metadata !{i32 101, i32 2, metadata !59, null}
!73 = metadata !{i32 102, i32 2, metadata !59, null}
!74 = metadata !{i32 103, i32 2, metadata !59, null}
!75 = metadata !{i32 104, i32 2, metadata !59, null}
!76 = metadata !{i32 105, i32 1, metadata !59, null}
!77 = metadata !{i32 75, i32 1, metadata !78, null}
!78 = metadata !{i32 786443, metadata !5, i32 74, i32 1, metadata !6, i32 0} ; [ DW_TAG_lexical_block ]
!79 = metadata !{i32 77, i32 2, metadata !78, null}
!80 = metadata !{i32 78, i32 7, metadata !81, null}
!81 = metadata !{i32 786443, metadata !78, i32 78, i32 2, metadata !6, i32 1} ; [ DW_TAG_lexical_block ]
!82 = metadata !{i32 79, i32 3, metadata !83, null}
!83 = metadata !{i32 786443, metadata !81, i32 78, i32 28, metadata !6, i32 2} ; [ DW_TAG_lexical_block ]
!84 = metadata !{i32 78, i32 23, metadata !81, null}
!85 = metadata !{i32 786688, metadata !78, metadata !"t", metadata !6, i32 76, metadata !9, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!86 = metadata !{i32 81, i32 2, metadata !78, null}
!87 = metadata !{i32 82, i32 1, metadata !78, null}
!88 = metadata !{i32 786689, metadata !18, metadata !"addr", metadata !6, i32 16777324, metadata !16, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!89 = metadata !{i32 108, i32 22, metadata !18, null}
!90 = metadata !{i32 110, i32 1, metadata !91, null}
!91 = metadata !{i32 786443, metadata !18, i32 109, i32 1, metadata !6, i32 5} ; [ DW_TAG_lexical_block ]
!92 = metadata !{i32 113, i32 2, metadata !91, null}
!93 = metadata !{i32 114, i32 2, metadata !91, null}
!94 = metadata !{i32 115, i32 2, metadata !91, null}
!95 = metadata !{i32 116, i32 2, metadata !91, null}
!96 = metadata !{i32 118, i32 2, metadata !91, null}
!97 = metadata !{i32 119, i32 3, metadata !98, null}
!98 = metadata !{i32 786443, metadata !91, i32 118, i32 5, metadata !6, i32 6} ; [ DW_TAG_lexical_block ]
!99 = metadata !{i32 120, i32 3, metadata !98, null}
!100 = metadata !{i32 121, i32 3, metadata !98, null}
!101 = metadata !{i32 786688, metadata !91, metadata !"dt", metadata !6, i32 111, metadata !16, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!102 = metadata !{i32 122, i32 2, metadata !98, null}
!103 = metadata !{i32 123, i32 2, metadata !91, null}
!104 = metadata !{i32 125, i32 2, metadata !91, null}
!105 = metadata !{i32 126, i32 2, metadata !91, null}
!106 = metadata !{i32 127, i32 2, metadata !91, null}
!107 = metadata !{i32 129, i32 2, metadata !91, null}
!108 = metadata !{i32 178, i32 1, metadata !109, null}
!109 = metadata !{i32 786443, metadata !33, i32 177, i32 1, metadata !6, i32 10} ; [ DW_TAG_lexical_block ]
!110 = metadata !{i32 179, i32 1, metadata !109, null}
!111 = metadata !{i32 181, i32 1, metadata !109, null}
!112 = metadata !{i32 182, i32 1, metadata !109, null}
!113 = metadata !{i32 183, i32 1, metadata !109, null}
!114 = metadata !{i32 184, i32 1, metadata !109, null}
!115 = metadata !{i32 186, i32 1, metadata !109, null}
!116 = metadata !{i32 187, i32 1, metadata !109, null}
!117 = metadata !{i32 188, i32 1, metadata !109, null}
!118 = metadata !{i32 189, i32 1, metadata !109, null}
!119 = metadata !{i32 190, i32 1, metadata !109, null}
!120 = metadata !{i32 191, i32 1, metadata !109, null}
!121 = metadata !{i32 192, i32 1, metadata !109, null}
!122 = metadata !{i32 207, i32 2, metadata !109, null}
!123 = metadata !{i32 208, i32 2, metadata !109, null}
!124 = metadata !{i32 209, i32 2, metadata !109, null}
!125 = metadata !{i32 210, i32 2, metadata !109, null}
!126 = metadata !{i32 211, i32 2, metadata !109, null}
!127 = metadata !{i32 218, i32 2, metadata !109, null}
!128 = metadata !{i32 220, i32 2, metadata !109, null}
!129 = metadata !{i32 221, i32 2, metadata !109, null}
!130 = metadata !{i32 222, i32 2, metadata !109, null}
!131 = metadata !{i32 223, i32 2, metadata !109, null}
!132 = metadata !{i32 224, i32 2, metadata !109, null}
!133 = metadata !{i32 226, i32 7, metadata !134, null}
!134 = metadata !{i32 786443, metadata !109, i32 226, i32 2, metadata !6, i32 11} ; [ DW_TAG_lexical_block ]
!135 = metadata !{i32 227, i32 3, metadata !136, null}
!136 = metadata !{i32 786443, metadata !134, i32 226, i32 27, metadata !6, i32 12} ; [ DW_TAG_lexical_block ]
!137 = metadata !{i32 226, i32 22, metadata !134, null}
!138 = metadata !{i32 786688, metadata !109, metadata !"i", metadata !6, i32 205, metadata !16, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!139 = metadata !{i32 246, i32 13, metadata !140, null}
!140 = metadata !{i32 786443, metadata !109, i32 246, i32 12, metadata !6, i32 13} ; [ DW_TAG_lexical_block ]
!141 = metadata !{i32 251, i32 8, metadata !140, null}
!142 = metadata !{i32 786688, metadata !109, metadata !"eh", metadata !6, i32 203, metadata !16, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!143 = metadata !{i32 252, i32 8, metadata !140, null}
!144 = metadata !{i32 786688, metadata !109, metadata !"el", metadata !6, i32 203, metadata !16, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!145 = metadata !{i32 786688, metadata !109, metadata !"et", metadata !6, i32 204, metadata !26, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!146 = metadata !{i32 253, i32 3, metadata !140, null}
!147 = metadata !{i32 255, i32 3, metadata !140, null}
!148 = metadata !{i32 256, i32 3, metadata !140, null}
!149 = metadata !{i32 256, i32 10, metadata !140, null}
!150 = metadata !{i32 258, i32 3, metadata !140, null}
!151 = metadata !{i32 259, i32 8, metadata !140, null}
!152 = metadata !{i32 260, i32 8, metadata !140, null}
!153 = metadata !{i32 786688, metadata !109, metadata !"e", metadata !6, i32 204, metadata !26, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!154 = metadata !{i32 261, i32 3, metadata !140, null}
!155 = metadata !{i32 262, i32 3, metadata !140, null}
!156 = metadata !{i32 265, i32 16, metadata !140, null}
!157 = metadata !{i32 266, i32 16, metadata !140, null}
!158 = metadata !{i32 267, i32 16, metadata !140, null}
!159 = metadata !{i32 268, i32 16, metadata !140, null}
!160 = metadata !{i32 270, i32 16, metadata !140, null}
!161 = metadata !{i32 271, i32 16, metadata !140, null}
!162 = metadata !{i32 272, i32 17, metadata !140, null}
!163 = metadata !{i32 273, i32 17, metadata !140, null}
!164 = metadata !{i32 281, i32 14, metadata !140, null}
!165 = metadata !{i32 786688, metadata !109, metadata !"diff_agl", metadata !6, i32 197, metadata !24, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!166 = metadata !{i32 284, i32 17, metadata !140, null}
!167 = metadata !{i32 285, i32 17, metadata !140, null}
!168 = metadata !{i32 286, i32 17, metadata !140, null}
!169 = metadata !{i32 287, i32 17, metadata !140, null}
!170 = metadata !{i32 289, i32 3, metadata !140, null}
!171 = metadata !{i32 291, i32 4, metadata !172, null}
!172 = metadata !{i32 786443, metadata !140, i32 289, i32 72, metadata !6, i32 14} ; [ DW_TAG_lexical_block ]
!173 = metadata !{i32 292, i32 3, metadata !172, null}
!174 = metadata !{i32 295, i32 4, metadata !175, null}
!175 = metadata !{i32 786443, metadata !140, i32 293, i32 8, metadata !6, i32 15} ; [ DW_TAG_lexical_block ]
!176 = metadata !{i32 298, i32 3, metadata !140, null}
!177 = metadata !{i32 300, i32 4, metadata !178, null}
!178 = metadata !{i32 786443, metadata !140, i32 298, i32 58, metadata !6, i32 16} ; [ DW_TAG_lexical_block ]
!179 = metadata !{i32 301, i32 3, metadata !178, null}
!180 = metadata !{i32 304, i32 4, metadata !181, null}
!181 = metadata !{i32 786443, metadata !140, i32 302, i32 8, metadata !6, i32 17} ; [ DW_TAG_lexical_block ]
!182 = metadata !{i32 308, i32 10, metadata !140, null}
!183 = metadata !{i32 786688, metadata !109, metadata !"mode", metadata !6, i32 202, metadata !16, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!184 = metadata !{i32 310, i32 3, metadata !140, null}
!185 = metadata !{i32 786688, metadata !109, metadata !"chR_pwm", metadata !6, i32 200, metadata !16, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!186 = metadata !{i32 786688, metadata !109, metadata !"chL_pwm", metadata !6, i32 200, metadata !16, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!187 = metadata !{i32 312, i32 3, metadata !140, null}
!188 = metadata !{i32 317, i32 4, metadata !189, null}
!189 = metadata !{i32 786443, metadata !140, i32 312, i32 24, metadata !6, i32 18} ; [ DW_TAG_lexical_block ]
!190 = metadata !{i32 321, i32 9, metadata !189, null}
!191 = metadata !{i32 331, i32 4, metadata !189, null}
!192 = metadata !{i32 335, i32 9, metadata !189, null}
!193 = metadata !{i32 342, i32 4, metadata !189, null}
!194 = metadata !{i32 347, i32 5, metadata !195, null}
!195 = metadata !{i32 786443, metadata !189, i32 346, i32 9, metadata !6, i32 24} ; [ DW_TAG_lexical_block ]
!196 = metadata !{i32 363, i32 4, metadata !189, null}
!197 = metadata !{i32 368, i32 5, metadata !198, null}
!198 = metadata !{i32 786443, metadata !189, i32 367, i32 9, metadata !6, i32 28} ; [ DW_TAG_lexical_block ]
!199 = metadata !{i32 390, i32 3, metadata !140, null}
!200 = metadata !{i32 391, i32 3, metadata !140, null}
!201 = metadata !{i32 786688, metadata !109, metadata !"mtr_pwm_cnt", metadata !6, i32 195, metadata !202, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!202 = metadata !{i32 786485, null, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !203} ; [ DW_TAG_volatile_type ]
!203 = metadata !{i32 786468, null, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!204 = metadata !{i32 392, i32 3, metadata !140, null}
!205 = metadata !{i32 393, i32 4, metadata !140, null}
!206 = metadata !{i32 395, i32 4, metadata !140, null}
!207 = metadata !{i32 397, i32 3, metadata !140, null}
!208 = metadata !{i32 398, i32 3, metadata !140, null}
!209 = metadata !{i32 399, i32 4, metadata !140, null}
!210 = metadata !{i32 401, i32 4, metadata !140, null}
!211 = metadata !{i32 402, i32 3, metadata !140, null}
!212 = metadata !{i32 405, i32 3, metadata !140, null}
!213 = metadata !{i32 406, i32 3, metadata !140, null}
!214 = metadata !{i32 407, i32 4, metadata !215, null}
!215 = metadata !{i32 786443, metadata !140, i32 406, i32 26, metadata !6, i32 31} ; [ DW_TAG_lexical_block ]
!216 = metadata !{i32 408, i32 3, metadata !215, null}
!217 = metadata !{i32 411, i32 17, metadata !140, null}
!218 = metadata !{i32 412, i32 17, metadata !140, null}
!219 = metadata !{i32 414, i32 17, metadata !140, null}
!220 = metadata !{i32 415, i32 17, metadata !140, null}
!221 = metadata !{i32 420, i32 3, metadata !140, null}
!222 = metadata !{i32 421, i32 2, metadata !140, null}
!223 = metadata !{i32 786689, metadata !21, metadata !"target", metadata !6, i32 16777349, metadata !26, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!224 = metadata !{i32 133, i32 25, metadata !21, null}
!225 = metadata !{i32 786689, metadata !21, metadata !"value", metadata !6, i32 33554565, metadata !26, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!226 = metadata !{i32 133, i32 40, metadata !21, null}
!227 = metadata !{i32 135, i32 1, metadata !228, null}
!228 = metadata !{i32 786443, metadata !21, i32 134, i32 1, metadata !6, i32 7} ; [ DW_TAG_lexical_block ]
!229 = metadata !{i32 138, i32 2, metadata !228, null}
!230 = metadata !{i32 139, i32 2, metadata !228, null}
!231 = metadata !{i32 141, i32 2, metadata !228, null}
!232 = metadata !{i32 786688, metadata !228, metadata !"retval", metadata !6, i32 136, metadata !24, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!233 = metadata !{i32 144, i32 2, metadata !228, null}
!234 = metadata !{i32 786689, metadata !28, metadata !"val", metadata !6, i32 16777364, metadata !31, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!235 = metadata !{i32 148, i32 22, metadata !28, null}
!236 = metadata !{i32 150, i32 1, metadata !237, null}
!237 = metadata !{i32 786443, metadata !28, i32 149, i32 1, metadata !6, i32 8} ; [ DW_TAG_lexical_block ]
!238 = metadata !{i32 153, i32 2, metadata !237, null}
!239 = metadata !{i32 155, i32 24, metadata !240, null}
!240 = metadata !{i32 786443, metadata !237, i32 153, i32 15, metadata !6, i32 9} ; [ DW_TAG_lexical_block ]
!241 = metadata !{i32 156, i32 24, metadata !240, null}
!242 = metadata !{i32 157, i32 24, metadata !240, null}
!243 = metadata !{i32 158, i32 24, metadata !240, null}
!244 = metadata !{i32 159, i32 24, metadata !240, null}
!245 = metadata !{i32 160, i32 24, metadata !240, null}
!246 = metadata !{i32 161, i32 24, metadata !240, null}
!247 = metadata !{i32 162, i32 24, metadata !240, null}
!248 = metadata !{i32 163, i32 24, metadata !240, null}
!249 = metadata !{i32 164, i32 25, metadata !240, null}
!250 = metadata !{i32 165, i32 25, metadata !240, null}
!251 = metadata !{i32 166, i32 25, metadata !240, null}
!252 = metadata !{i32 167, i32 25, metadata !240, null}
!253 = metadata !{i32 168, i32 25, metadata !240, null}
!254 = metadata !{i32 170, i32 2, metadata !240, null}
!255 = metadata !{i32 172, i32 2, metadata !237, null}
