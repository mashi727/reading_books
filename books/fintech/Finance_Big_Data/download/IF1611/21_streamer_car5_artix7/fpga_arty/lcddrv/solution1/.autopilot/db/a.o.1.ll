; ModuleID = 'D:/21_streamer_car5_artix7/fpga_arty/lcddrv/solution1/.autopilot/db/a.g.1.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-w64-mingw32"

@rs = global i1 false, align 1                    ; [#uses=4 type=i1*]
@mem_req = global i1 false, align 1               ; [#uses=4 type=i1*]
@mem_din = common global i8 0, align 1            ; [#uses=2 type=i8*]
@mem_addr = global i5 0, align 1                  ; [#uses=3 type=i5*]
@mem_ack = common global i1 false, align 1        ; [#uses=2 type=i1*]
@ind = global i1 false, align 1                   ; [#uses=1 type=i1*]
@en = global i1 false, align 1                    ; [#uses=5 type=i1*]
@dummy_tmr_out = global i1 false, align 1         ; [#uses=3 type=i1*]
@data = global i4 0, align 1                      ; [#uses=4 type=i4*]
@.str2 = private unnamed_addr constant [12 x i8] c"hls_label_0\00", align 1 ; [#uses=1 type=[12 x i8]*]
@.str1 = private unnamed_addr constant [8 x i8] c"ap_none\00", align 1 ; [#uses=1 type=[8 x i8]*]
@.str = private unnamed_addr constant [1 x i8] zeroinitializer, align 1 ; [#uses=1 type=[1 x i8]*]

; [#uses=13]
define internal fastcc void @wait_tmr(i32 %tmr) nounwind uwtable {
  call void @llvm.dbg.value(metadata !{i32 %tmr}, i64 0, metadata !46), !dbg !47 ; [debug line = 65:22] [debug variable = tmr]
  call void (...)* @_ssdm_InlineSelf(i32 2, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !48 ; [debug line = 67:1]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !50 ; [debug line = 69:2]
  br label %1, !dbg !51                           ; [debug line = 70:7]

; <label>:1                                       ; preds = %2, %0
  %t = phi i32 [ 0, %0 ], [ %t.1, %2 ]            ; [#uses=2 type=i32]
  %exitcond = icmp eq i32 %t, %tmr, !dbg !51      ; [#uses=1 type=i1] [debug line = 70:7]
  br i1 %exitcond, label %3, label %2, !dbg !51   ; [debug line = 70:7]

; <label>:2                                       ; preds = %1
  %dummy_tmr_out.load = load volatile i1* @dummy_tmr_out, align 1, !dbg !53 ; [#uses=1 type=i1] [debug line = 71:3]
  %not. = xor i1 %dummy_tmr_out.load, true, !dbg !53 ; [#uses=1 type=i1] [debug line = 71:3]
  store volatile i1 %not., i1* @dummy_tmr_out, align 1, !dbg !53 ; [debug line = 71:3]
  %t.1 = add i32 %t, 1, !dbg !55                  ; [#uses=1 type=i32] [debug line = 70:23]
  call void @llvm.dbg.value(metadata !{i32 %t.1}, i64 0, metadata !56), !dbg !55 ; [debug line = 70:23] [debug variable = t]
  br label %1, !dbg !55                           ; [debug line = 70:23]

; <label>:3                                       ; preds = %1
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !57 ; [debug line = 73:2]
  ret void, !dbg !58                              ; [debug line = 74:1]
}

; [#uses=2]
define internal fastcc zeroext i8 @read_mem(i5 zeroext %addr) nounwind uwtable {
  call void @llvm.dbg.value(metadata !{i5 %addr}, i64 0, metadata !59), !dbg !60 ; [debug line = 138:22] [debug variable = addr]
  call void (...)* @_ssdm_InlineSelf(i32 2, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !61 ; [debug line = 140:1]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !63 ; [debug line = 143:2]
  store volatile i5 %addr, i5* @mem_addr, align 1, !dbg !64 ; [debug line = 144:2]
  store volatile i1 true, i1* @mem_req, align 1, !dbg !65 ; [debug line = 145:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !66 ; [debug line = 146:2]
  br label %._crit_edge, !dbg !67                 ; [debug line = 147:2]

._crit_edge:                                      ; preds = %._crit_edge, %0
  store volatile i5 %addr, i5* @mem_addr, align 1, !dbg !68 ; [debug line = 148:3]
  store volatile i1 true, i1* @mem_req, align 1, !dbg !70 ; [debug line = 149:3]
  %dt = load volatile i8* @mem_din, align 1, !dbg !71 ; [#uses=1 type=i8] [debug line = 150:3]
  call void @llvm.dbg.value(metadata !{i8 %dt}, i64 0, metadata !72), !dbg !71 ; [debug line = 150:3] [debug variable = dt]
  %mem_ack.load = load volatile i1* @mem_ack, align 1, !dbg !73 ; [#uses=1 type=i1] [debug line = 151:2]
  br i1 %mem_ack.load, label %1, label %._crit_edge, !dbg !73 ; [debug line = 151:2]

; <label>:1                                       ; preds = %._crit_edge
  %dt.0.lcssa = phi i8 [ %dt, %._crit_edge ]      ; [#uses=1 type=i8]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !74 ; [debug line = 152:2]
  store volatile i1 false, i1* @mem_req, align 1, !dbg !75 ; [debug line = 153:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !76 ; [debug line = 154:2]
  ret i8 %dt.0.lcssa, !dbg !77                    ; [debug line = 156:2]
}

; [#uses=10]
declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

; [#uses=0]
define void @lcddrv() noreturn nounwind uwtable {
  call void (...)* @_ssdm_op_SpecTopModule(), !dbg !78 ; [debug line = 162:1]
  %dummy_tmr_out.load = load volatile i1* @dummy_tmr_out, align 1, !dbg !80 ; [#uses=1 type=i1] [debug line = 163:1]
  %tmp = zext i1 %dummy_tmr_out.load to i32, !dbg !80 ; [#uses=1 type=i32] [debug line = 163:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp, i8* getelementptr inbounds ([8 x i8]* @.str1, i64 0, i64 0), i32 1, i32 1, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !80 ; [debug line = 163:1]
  %rs.load = load volatile i1* @rs, align 1, !dbg !81 ; [#uses=1 type=i1] [debug line = 165:1]
  %tmp.1 = zext i1 %rs.load to i32, !dbg !81      ; [#uses=1 type=i32] [debug line = 165:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.1, i8* getelementptr inbounds ([8 x i8]* @.str1, i64 0, i64 0), i32 1, i32 1, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !81 ; [debug line = 165:1]
  %en.load = load volatile i1* @en, align 1, !dbg !82 ; [#uses=1 type=i1] [debug line = 166:1]
  %tmp.2 = zext i1 %en.load to i32, !dbg !82      ; [#uses=1 type=i32] [debug line = 166:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.2, i8* getelementptr inbounds ([8 x i8]* @.str1, i64 0, i64 0), i32 1, i32 1, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !82 ; [debug line = 166:1]
  %data.load = load volatile i4* @data, align 1, !dbg !83 ; [#uses=1 type=i4] [debug line = 167:1]
  %tmp.3 = zext i4 %data.load to i32, !dbg !83    ; [#uses=1 type=i32] [debug line = 167:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.3, i8* getelementptr inbounds ([8 x i8]* @.str1, i64 0, i64 0), i32 1, i32 1, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !83 ; [debug line = 167:1]
  %ind.load = load volatile i1* @ind, align 1, !dbg !84 ; [#uses=1 type=i1] [debug line = 168:1]
  %tmp.4 = zext i1 %ind.load to i32, !dbg !84     ; [#uses=1 type=i32] [debug line = 168:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.4, i8* getelementptr inbounds ([8 x i8]* @.str1, i64 0, i64 0), i32 1, i32 1, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !84 ; [debug line = 168:1]
  %mem_addr.load = load volatile i5* @mem_addr, align 1, !dbg !85 ; [#uses=1 type=i5] [debug line = 170:1]
  %tmp.5 = zext i5 %mem_addr.load to i32, !dbg !85 ; [#uses=1 type=i32] [debug line = 170:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.5, i8* getelementptr inbounds ([8 x i8]* @.str1, i64 0, i64 0), i32 1, i32 1, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !85 ; [debug line = 170:1]
  %mem_din.load = load volatile i8* @mem_din, align 1, !dbg !86 ; [#uses=1 type=i8] [debug line = 171:1]
  %tmp.6 = zext i8 %mem_din.load to i32, !dbg !86 ; [#uses=1 type=i32] [debug line = 171:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.6, i8* getelementptr inbounds ([8 x i8]* @.str1, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !86 ; [debug line = 171:1]
  %mem_req.load = load volatile i1* @mem_req, align 1, !dbg !87 ; [#uses=1 type=i1] [debug line = 172:1]
  %tmp.7 = zext i1 %mem_req.load to i32, !dbg !87 ; [#uses=1 type=i32] [debug line = 172:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.7, i8* getelementptr inbounds ([8 x i8]* @.str1, i64 0, i64 0), i32 1, i32 1, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !87 ; [debug line = 172:1]
  %mem_ack.load = load volatile i1* @mem_ack, align 1, !dbg !88 ; [#uses=1 type=i1] [debug line = 173:1]
  %tmp.8 = zext i1 %mem_ack.load to i32, !dbg !88 ; [#uses=1 type=i32] [debug line = 173:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.8, i8* getelementptr inbounds ([8 x i8]* @.str1, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !88 ; [debug line = 173:1]
  call fastcc void @init_lcd(), !dbg !89          ; [debug line = 178:2]
  br label %1, !dbg !90                           ; [debug line = 180:2]

; <label>:1                                       ; preds = %7, %0
  %loop_begin = call i32 (...)* @_ssdm_op_SpecLoopBegin() nounwind ; [#uses=0 type=i32]
  %rbegin = call i32 (...)* @_ssdm_op_SpecRegionBegin(i8* getelementptr inbounds ([12 x i8]* @.str2, i64 0, i64 0)) nounwind, !dbg !91 ; [#uses=1 type=i32] [debug line = 180:13]
  call fastcc void @lcd_send_cmd(i1 zeroext false, i4 zeroext -8), !dbg !93 ; [debug line = 182:3]
  call fastcc void @lcd_send_cmd(i1 zeroext false, i4 zeroext 0), !dbg !94 ; [debug line = 183:3]
  br label %2, !dbg !95                           ; [debug line = 185:8]

; <label>:2                                       ; preds = %3, %1
  %pos = phi i8 [ 0, %1 ], [ %pos.2, %3 ]         ; [#uses=3 type=i8]
  %exitcond1 = icmp eq i8 %pos, 16, !dbg !95      ; [#uses=1 type=i1] [debug line = 185:8]
  br i1 %exitcond1, label %4, label %3, !dbg !95  ; [debug line = 185:8]

; <label>:3                                       ; preds = %2
  %tmp.9 = trunc i8 %pos to i5, !dbg !97          ; [#uses=1 type=i5] [debug line = 186:9]
  %dt = call fastcc zeroext i8 @read_mem(i5 zeroext %tmp.9), !dbg !97 ; [#uses=2 type=i8] [debug line = 186:9]
  call void @llvm.dbg.value(metadata !{i8 %dt}, i64 0, metadata !99), !dbg !97 ; [debug line = 186:9] [debug variable = dt]
  %tmp.11 = lshr i8 %dt, 4, !dbg !100             ; [#uses=1 type=i8] [debug line = 187:4]
  %tmp.12 = trunc i8 %tmp.11 to i4, !dbg !100     ; [#uses=1 type=i4] [debug line = 187:4]
  call fastcc void @lcd_send_cmd(i1 zeroext true, i4 zeroext %tmp.12), !dbg !100 ; [debug line = 187:4]
  %tmp.13 = trunc i8 %dt to i4, !dbg !101         ; [#uses=1 type=i4] [debug line = 188:4]
  call fastcc void @lcd_send_cmd(i1 zeroext true, i4 zeroext %tmp.13), !dbg !101 ; [debug line = 188:4]
  %pos.2 = add i8 %pos, 1, !dbg !102              ; [#uses=1 type=i8] [debug line = 185:27]
  call void @llvm.dbg.value(metadata !{i8 %pos.2}, i64 0, metadata !103), !dbg !102 ; [debug line = 185:27] [debug variable = pos]
  br label %2, !dbg !102                          ; [debug line = 185:27]

; <label>:4                                       ; preds = %2
  call fastcc void @lcd_send_cmd(i1 zeroext false, i4 zeroext -4), !dbg !104 ; [debug line = 192:3]
  call fastcc void @lcd_send_cmd(i1 zeroext false, i4 zeroext 0), !dbg !105 ; [debug line = 193:3]
  br label %5, !dbg !106                          ; [debug line = 195:8]

; <label>:5                                       ; preds = %6, %4
  %pos.1 = phi i8 [ 16, %4 ], [ %pos.3, %6 ]      ; [#uses=3 type=i8]
  %exitcond = icmp eq i8 %pos.1, 32, !dbg !106    ; [#uses=1 type=i1] [debug line = 195:8]
  br i1 %exitcond, label %7, label %6, !dbg !106  ; [debug line = 195:8]

; <label>:6                                       ; preds = %5
  %tmp.15 = trunc i8 %pos.1 to i5, !dbg !108      ; [#uses=1 type=i5] [debug line = 196:9]
  %dt.1 = call fastcc zeroext i8 @read_mem(i5 zeroext %tmp.15), !dbg !108 ; [#uses=2 type=i8] [debug line = 196:9]
  call void @llvm.dbg.value(metadata !{i8 %dt.1}, i64 0, metadata !99), !dbg !108 ; [debug line = 196:9] [debug variable = dt]
  %tmp.17 = lshr i8 %dt.1, 4, !dbg !110           ; [#uses=1 type=i8] [debug line = 197:4]
  %tmp.18 = trunc i8 %tmp.17 to i4, !dbg !110     ; [#uses=1 type=i4] [debug line = 197:4]
  call fastcc void @lcd_send_cmd(i1 zeroext true, i4 zeroext %tmp.18), !dbg !110 ; [debug line = 197:4]
  %tmp.19 = trunc i8 %dt.1 to i4, !dbg !111       ; [#uses=1 type=i4] [debug line = 198:4]
  call fastcc void @lcd_send_cmd(i1 zeroext true, i4 zeroext %tmp.19), !dbg !111 ; [debug line = 198:4]
  %pos.3 = add i8 %pos.1, 1, !dbg !112            ; [#uses=1 type=i8] [debug line = 195:28]
  call void @llvm.dbg.value(metadata !{i8 %pos.3}, i64 0, metadata !103), !dbg !112 ; [debug line = 195:28] [debug variable = pos]
  br label %5, !dbg !112                          ; [debug line = 195:28]

; <label>:7                                       ; preds = %5
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !113 ; [debug line = 202:3]
  call fastcc void @wait_tmr(i32 10000000), !dbg !114 ; [debug line = 203:3]
  %rend = call i32 (...)* @_ssdm_op_SpecRegionEnd(i8* getelementptr inbounds ([12 x i8]* @.str2, i64 0, i64 0), i32 %rbegin) nounwind, !dbg !115 ; [#uses=0 type=i32] [debug line = 204:2]
  br label %1, !dbg !115                          ; [debug line = 204:2]
}

; [#uses=20]
define internal fastcc void @lcd_send_cmd(i1 zeroext %mode, i4 zeroext %wd) nounwind uwtable {
  call void @llvm.dbg.value(metadata !{i1 %mode}, i64 0, metadata !116), !dbg !117 ; [debug line = 77:25] [debug variable = mode]
  call void @llvm.dbg.value(metadata !{i4 %wd}, i64 0, metadata !118), !dbg !119 ; [debug line = 77:37] [debug variable = wd]
  call void (...)* @_ssdm_InlineSelf(i32 2, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !120 ; [debug line = 79:1]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !122 ; [debug line = 80:2]
  store volatile i1 false, i1* @en, align 1, !dbg !123 ; [debug line = 81:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !124 ; [debug line = 82:2]
  call fastcc void @wait_tmr(i32 1000), !dbg !125 ; [debug line = 83:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !126 ; [debug line = 85:2]
  store volatile i1 false, i1* @en, align 1, !dbg !127 ; [debug line = 86:2]
  store volatile i1 %mode, i1* @rs, align 1, !dbg !128 ; [debug line = 87:2]
  store volatile i4 %wd, i4* @data, align 1, !dbg !129 ; [debug line = 88:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !130 ; [debug line = 89:2]
  call fastcc void @wait_tmr(i32 1000), !dbg !131 ; [debug line = 90:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !132 ; [debug line = 92:2]
  store volatile i1 true, i1* @en, align 1, !dbg !133 ; [debug line = 93:2]
  store volatile i1 %mode, i1* @rs, align 1, !dbg !134 ; [debug line = 94:2]
  store volatile i4 %wd, i4* @data, align 1, !dbg !135 ; [debug line = 95:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !136 ; [debug line = 96:2]
  call fastcc void @wait_tmr(i32 1000), !dbg !137 ; [debug line = 97:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !138 ; [debug line = 99:2]
  store volatile i1 false, i1* @en, align 1, !dbg !139 ; [debug line = 100:2]
  store volatile i1 %mode, i1* @rs, align 1, !dbg !140 ; [debug line = 101:2]
  store volatile i4 %wd, i4* @data, align 1, !dbg !141 ; [debug line = 102:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !142 ; [debug line = 103:2]
  call fastcc void @wait_tmr(i32 1000), !dbg !143 ; [debug line = 104:2]
  ret void, !dbg !144                             ; [debug line = 105:1]
}

; [#uses=1]
define internal fastcc void @init_lcd() nounwind uwtable {
  call void (...)* @_ssdm_InlineSelf(i32 2, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !145 ; [debug line = 110:1]
  call fastcc void @wait_tmr(i32 2000000), !dbg !147 ; [debug line = 111:2]
  call fastcc void @lcd_send_cmd(i1 zeroext false, i4 zeroext 3), !dbg !148 ; [debug line = 112:2]
  call fastcc void @wait_tmr(i32 500000), !dbg !149 ; [debug line = 113:2]
  call fastcc void @lcd_send_cmd(i1 zeroext false, i4 zeroext 3), !dbg !150 ; [debug line = 114:2]
  call fastcc void @wait_tmr(i32 50000), !dbg !151 ; [debug line = 115:2]
  call fastcc void @lcd_send_cmd(i1 zeroext false, i4 zeroext 3), !dbg !152 ; [debug line = 116:2]
  call fastcc void @wait_tmr(i32 50000), !dbg !153 ; [debug line = 117:2]
  call fastcc void @lcd_send_cmd(i1 zeroext false, i4 zeroext 2), !dbg !154 ; [debug line = 119:2]
  call fastcc void @lcd_send_cmd(i1 zeroext false, i4 zeroext 2), !dbg !155 ; [debug line = 120:2]
  call fastcc void @lcd_send_cmd(i1 zeroext false, i4 zeroext -8), !dbg !156 ; [debug line = 121:2]
  call fastcc void @wait_tmr(i32 10000), !dbg !157 ; [debug line = 122:2]
  call fastcc void @lcd_send_cmd(i1 zeroext false, i4 zeroext 0), !dbg !158 ; [debug line = 124:2]
  call fastcc void @lcd_send_cmd(i1 zeroext false, i4 zeroext -4), !dbg !159 ; [debug line = 125:2]
  call fastcc void @wait_tmr(i32 10000), !dbg !160 ; [debug line = 126:2]
  call fastcc void @lcd_send_cmd(i1 zeroext false, i4 zeroext 0), !dbg !161 ; [debug line = 128:2]
  call fastcc void @lcd_send_cmd(i1 zeroext false, i4 zeroext 1), !dbg !162 ; [debug line = 129:2]
  call fastcc void @wait_tmr(i32 200000), !dbg !163 ; [debug line = 130:2]
  call fastcc void @lcd_send_cmd(i1 zeroext false, i4 zeroext 0), !dbg !164 ; [debug line = 132:2]
  call fastcc void @lcd_send_cmd(i1 zeroext false, i4 zeroext 2), !dbg !165 ; [debug line = 133:2]
  call fastcc void @wait_tmr(i32 10000), !dbg !166 ; [debug line = 134:2]
  ret void, !dbg !167                             ; [debug line = 135:1]
}

; [#uses=15]
declare void @_ssdm_op_Wait(...) nounwind

; [#uses=1]
declare void @_ssdm_op_SpecTopModule(...) nounwind

; [#uses=1]
declare i32 @_ssdm_op_SpecRegionEnd(...)

; [#uses=1]
declare i32 @_ssdm_op_SpecRegionBegin(...)

; [#uses=1]
declare i32 @_ssdm_op_SpecLoopBegin(...)

; [#uses=9]
declare void @_ssdm_op_SpecInterface(...) nounwind

; [#uses=4]
declare void @_ssdm_InlineSelf(...) nounwind

!llvm.dbg.cu = !{!0}
!hls.encrypted.func = !{}

!0 = metadata !{i32 786449, i32 0, i32 1, metadata !"D:/21_streamer_car5_artix7/fpga_arty/lcddrv/solution1/.autopilot/db/lcddrv.pragma.2.c", metadata !"D:\5C21_streamer_car5_artix7\5Cfpga_arty", metadata !"clang version 3.1 ", i1 true, i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !31} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !5, metadata !13, metadata !20, metadata !23, metadata !30}
!5 = metadata !{i32 786478, i32 0, metadata !6, metadata !"wait_tmr", metadata !"wait_tmr", metadata !"", metadata !6, i32 65, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (i32)* @wait_tmr, null, null, metadata !11, i32 66} ; [ DW_TAG_subprogram ]
!6 = metadata !{i32 786473, metadata !"lcddrv.c", metadata !"D:\5C21_streamer_car5_artix7\5Cfpga_arty", null} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!8 = metadata !{null, metadata !9}
!9 = metadata !{i32 786454, null, metadata !"uint32", metadata !6, i32 34, i64 0, i64 0, i64 0, i32 0, metadata !10} ; [ DW_TAG_typedef ]
!10 = metadata !{i32 786468, null, metadata !"uint32", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!11 = metadata !{metadata !12}
!12 = metadata !{i32 786468}                      ; [ DW_TAG_base_type ]
!13 = metadata !{i32 786478, i32 0, metadata !6, metadata !"lcd_send_cmd", metadata !"lcd_send_cmd", metadata !"", metadata !6, i32 77, metadata !14, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (i1, i4)* @lcd_send_cmd, null, null, metadata !11, i32 78} ; [ DW_TAG_subprogram ]
!14 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !15, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!15 = metadata !{null, metadata !16, metadata !18}
!16 = metadata !{i32 786454, null, metadata !"uint1", metadata !6, i32 3, i64 0, i64 0, i64 0, i32 0, metadata !17} ; [ DW_TAG_typedef ]
!17 = metadata !{i32 786468, null, metadata !"uint1", null, i32 0, i64 1, i64 1, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!18 = metadata !{i32 786454, null, metadata !"uint4", metadata !6, i32 6, i64 0, i64 0, i64 0, i32 0, metadata !19} ; [ DW_TAG_typedef ]
!19 = metadata !{i32 786468, null, metadata !"uint4", null, i32 0, i64 4, i64 4, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!20 = metadata !{i32 786478, i32 0, metadata !6, metadata !"init_lcd", metadata !"init_lcd", metadata !"", metadata !6, i32 108, metadata !21, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 false, void ()* @init_lcd, null, null, metadata !11, i32 109} ; [ DW_TAG_subprogram ]
!21 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !22, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!22 = metadata !{null}
!23 = metadata !{i32 786478, i32 0, metadata !6, metadata !"read_mem", metadata !"read_mem", metadata !"", metadata !6, i32 138, metadata !24, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i8 (i5)* @read_mem, null, null, metadata !11, i32 139} ; [ DW_TAG_subprogram ]
!24 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !25, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!25 = metadata !{metadata !26, metadata !28}
!26 = metadata !{i32 786454, null, metadata !"uint8", metadata !6, i32 10, i64 0, i64 0, i64 0, i32 0, metadata !27} ; [ DW_TAG_typedef ]
!27 = metadata !{i32 786468, null, metadata !"uint8", null, i32 0, i64 8, i64 8, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!28 = metadata !{i32 786454, null, metadata !"uint5", metadata !6, i32 7, i64 0, i64 0, i64 0, i32 0, metadata !29} ; [ DW_TAG_typedef ]
!29 = metadata !{i32 786468, null, metadata !"uint5", null, i32 0, i64 5, i64 8, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!30 = metadata !{i32 786478, i32 0, metadata !6, metadata !"lcddrv", metadata !"lcddrv", metadata !"", metadata !6, i32 160, metadata !21, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 false, void ()* @lcddrv, null, null, metadata !11, i32 161} ; [ DW_TAG_subprogram ]
!31 = metadata !{metadata !32}
!32 = metadata !{metadata !33, metadata !35, metadata !36, metadata !38, metadata !39, metadata !41, metadata !42, metadata !43, metadata !45}
!33 = metadata !{i32 786484, i32 0, null, metadata !"rs", metadata !"rs", metadata !"", metadata !6, i32 47, metadata !34, i32 0, i32 1, i1* @rs} ; [ DW_TAG_variable ]
!34 = metadata !{i32 786485, null, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !16} ; [ DW_TAG_volatile_type ]
!35 = metadata !{i32 786484, i32 0, null, metadata !"en", metadata !"en", metadata !"", metadata !6, i32 48, metadata !34, i32 0, i32 1, i1* @en} ; [ DW_TAG_variable ]
!36 = metadata !{i32 786484, i32 0, null, metadata !"data", metadata !"data", metadata !"", metadata !6, i32 49, metadata !37, i32 0, i32 1, i4* @data} ; [ DW_TAG_variable ]
!37 = metadata !{i32 786485, null, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !18} ; [ DW_TAG_volatile_type ]
!38 = metadata !{i32 786484, i32 0, null, metadata !"ind", metadata !"ind", metadata !"", metadata !6, i32 50, metadata !34, i32 0, i32 1, i1* @ind} ; [ DW_TAG_variable ]
!39 = metadata !{i32 786484, i32 0, null, metadata !"mem_addr", metadata !"mem_addr", metadata !"", metadata !6, i32 53, metadata !40, i32 0, i32 1, i5* @mem_addr} ; [ DW_TAG_variable ]
!40 = metadata !{i32 786485, null, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !28} ; [ DW_TAG_volatile_type ]
!41 = metadata !{i32 786484, i32 0, null, metadata !"mem_req", metadata !"mem_req", metadata !"", metadata !6, i32 55, metadata !34, i32 0, i32 1, i1* @mem_req} ; [ DW_TAG_variable ]
!42 = metadata !{i32 786484, i32 0, null, metadata !"dummy_tmr_out", metadata !"dummy_tmr_out", metadata !"", metadata !6, i32 59, metadata !34, i32 0, i32 1, i1* @dummy_tmr_out} ; [ DW_TAG_variable ]
!43 = metadata !{i32 786484, i32 0, null, metadata !"mem_din", metadata !"mem_din", metadata !"", metadata !6, i32 54, metadata !44, i32 0, i32 1, i8* @mem_din} ; [ DW_TAG_variable ]
!44 = metadata !{i32 786485, null, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !26} ; [ DW_TAG_volatile_type ]
!45 = metadata !{i32 786484, i32 0, null, metadata !"mem_ack", metadata !"mem_ack", metadata !"", metadata !6, i32 56, metadata !34, i32 0, i32 1, i1* @mem_ack} ; [ DW_TAG_variable ]
!46 = metadata !{i32 786689, metadata !5, metadata !"tmr", metadata !6, i32 16777281, metadata !9, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!47 = metadata !{i32 65, i32 22, metadata !5, null}
!48 = metadata !{i32 67, i32 1, metadata !49, null}
!49 = metadata !{i32 786443, metadata !5, i32 66, i32 1, metadata !6, i32 0} ; [ DW_TAG_lexical_block ]
!50 = metadata !{i32 69, i32 2, metadata !49, null}
!51 = metadata !{i32 70, i32 7, metadata !52, null}
!52 = metadata !{i32 786443, metadata !49, i32 70, i32 2, metadata !6, i32 1} ; [ DW_TAG_lexical_block ]
!53 = metadata !{i32 71, i32 3, metadata !54, null}
!54 = metadata !{i32 786443, metadata !52, i32 70, i32 28, metadata !6, i32 2} ; [ DW_TAG_lexical_block ]
!55 = metadata !{i32 70, i32 23, metadata !52, null}
!56 = metadata !{i32 786688, metadata !49, metadata !"t", metadata !6, i32 68, metadata !9, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!57 = metadata !{i32 73, i32 2, metadata !49, null}
!58 = metadata !{i32 74, i32 1, metadata !49, null}
!59 = metadata !{i32 786689, metadata !23, metadata !"addr", metadata !6, i32 16777354, metadata !28, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!60 = metadata !{i32 138, i32 22, metadata !23, null}
!61 = metadata !{i32 140, i32 1, metadata !62, null}
!62 = metadata !{i32 786443, metadata !23, i32 139, i32 1, metadata !6, i32 5} ; [ DW_TAG_lexical_block ]
!63 = metadata !{i32 143, i32 2, metadata !62, null}
!64 = metadata !{i32 144, i32 2, metadata !62, null}
!65 = metadata !{i32 145, i32 2, metadata !62, null}
!66 = metadata !{i32 146, i32 2, metadata !62, null}
!67 = metadata !{i32 147, i32 2, metadata !62, null}
!68 = metadata !{i32 148, i32 3, metadata !69, null}
!69 = metadata !{i32 786443, metadata !62, i32 147, i32 5, metadata !6, i32 6} ; [ DW_TAG_lexical_block ]
!70 = metadata !{i32 149, i32 3, metadata !69, null}
!71 = metadata !{i32 150, i32 3, metadata !69, null}
!72 = metadata !{i32 786688, metadata !62, metadata !"dt", metadata !6, i32 141, metadata !26, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!73 = metadata !{i32 151, i32 2, metadata !69, null}
!74 = metadata !{i32 152, i32 2, metadata !62, null}
!75 = metadata !{i32 153, i32 2, metadata !62, null}
!76 = metadata !{i32 154, i32 2, metadata !62, null}
!77 = metadata !{i32 156, i32 2, metadata !62, null}
!78 = metadata !{i32 162, i32 1, metadata !79, null}
!79 = metadata !{i32 786443, metadata !30, i32 161, i32 1, metadata !6, i32 7} ; [ DW_TAG_lexical_block ]
!80 = metadata !{i32 163, i32 1, metadata !79, null}
!81 = metadata !{i32 165, i32 1, metadata !79, null}
!82 = metadata !{i32 166, i32 1, metadata !79, null}
!83 = metadata !{i32 167, i32 1, metadata !79, null}
!84 = metadata !{i32 168, i32 1, metadata !79, null}
!85 = metadata !{i32 170, i32 1, metadata !79, null}
!86 = metadata !{i32 171, i32 1, metadata !79, null}
!87 = metadata !{i32 172, i32 1, metadata !79, null}
!88 = metadata !{i32 173, i32 1, metadata !79, null}
!89 = metadata !{i32 178, i32 2, metadata !79, null}
!90 = metadata !{i32 180, i32 2, metadata !79, null}
!91 = metadata !{i32 180, i32 13, metadata !92, null}
!92 = metadata !{i32 786443, metadata !79, i32 180, i32 12, metadata !6, i32 8} ; [ DW_TAG_lexical_block ]
!93 = metadata !{i32 182, i32 3, metadata !92, null}
!94 = metadata !{i32 183, i32 3, metadata !92, null}
!95 = metadata !{i32 185, i32 8, metadata !96, null}
!96 = metadata !{i32 786443, metadata !92, i32 185, i32 3, metadata !6, i32 9} ; [ DW_TAG_lexical_block ]
!97 = metadata !{i32 186, i32 9, metadata !98, null}
!98 = metadata !{i32 786443, metadata !96, i32 185, i32 34, metadata !6, i32 10} ; [ DW_TAG_lexical_block ]
!99 = metadata !{i32 786688, metadata !79, metadata !"dt", metadata !6, i32 176, metadata !26, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!100 = metadata !{i32 187, i32 4, metadata !98, null}
!101 = metadata !{i32 188, i32 4, metadata !98, null}
!102 = metadata !{i32 185, i32 27, metadata !96, null}
!103 = metadata !{i32 786688, metadata !79, metadata !"pos", metadata !6, i32 175, metadata !26, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!104 = metadata !{i32 192, i32 3, metadata !92, null}
!105 = metadata !{i32 193, i32 3, metadata !92, null}
!106 = metadata !{i32 195, i32 8, metadata !107, null}
!107 = metadata !{i32 786443, metadata !92, i32 195, i32 3, metadata !6, i32 11} ; [ DW_TAG_lexical_block ]
!108 = metadata !{i32 196, i32 9, metadata !109, null}
!109 = metadata !{i32 786443, metadata !107, i32 195, i32 35, metadata !6, i32 12} ; [ DW_TAG_lexical_block ]
!110 = metadata !{i32 197, i32 4, metadata !109, null}
!111 = metadata !{i32 198, i32 4, metadata !109, null}
!112 = metadata !{i32 195, i32 28, metadata !107, null}
!113 = metadata !{i32 202, i32 3, metadata !92, null}
!114 = metadata !{i32 203, i32 3, metadata !92, null}
!115 = metadata !{i32 204, i32 2, metadata !92, null}
!116 = metadata !{i32 786689, metadata !13, metadata !"mode", metadata !6, i32 16777293, metadata !16, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!117 = metadata !{i32 77, i32 25, metadata !13, null}
!118 = metadata !{i32 786689, metadata !13, metadata !"wd", metadata !6, i32 33554509, metadata !18, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!119 = metadata !{i32 77, i32 37, metadata !13, null}
!120 = metadata !{i32 79, i32 1, metadata !121, null}
!121 = metadata !{i32 786443, metadata !13, i32 78, i32 1, metadata !6, i32 3} ; [ DW_TAG_lexical_block ]
!122 = metadata !{i32 80, i32 2, metadata !121, null}
!123 = metadata !{i32 81, i32 2, metadata !121, null}
!124 = metadata !{i32 82, i32 2, metadata !121, null}
!125 = metadata !{i32 83, i32 2, metadata !121, null}
!126 = metadata !{i32 85, i32 2, metadata !121, null}
!127 = metadata !{i32 86, i32 2, metadata !121, null}
!128 = metadata !{i32 87, i32 2, metadata !121, null}
!129 = metadata !{i32 88, i32 2, metadata !121, null}
!130 = metadata !{i32 89, i32 2, metadata !121, null}
!131 = metadata !{i32 90, i32 2, metadata !121, null}
!132 = metadata !{i32 92, i32 2, metadata !121, null}
!133 = metadata !{i32 93, i32 2, metadata !121, null}
!134 = metadata !{i32 94, i32 2, metadata !121, null}
!135 = metadata !{i32 95, i32 2, metadata !121, null}
!136 = metadata !{i32 96, i32 2, metadata !121, null}
!137 = metadata !{i32 97, i32 2, metadata !121, null}
!138 = metadata !{i32 99, i32 2, metadata !121, null}
!139 = metadata !{i32 100, i32 2, metadata !121, null}
!140 = metadata !{i32 101, i32 2, metadata !121, null}
!141 = metadata !{i32 102, i32 2, metadata !121, null}
!142 = metadata !{i32 103, i32 2, metadata !121, null}
!143 = metadata !{i32 104, i32 2, metadata !121, null}
!144 = metadata !{i32 105, i32 1, metadata !121, null}
!145 = metadata !{i32 110, i32 1, metadata !146, null}
!146 = metadata !{i32 786443, metadata !20, i32 109, i32 1, metadata !6, i32 4} ; [ DW_TAG_lexical_block ]
!147 = metadata !{i32 111, i32 2, metadata !146, null}
!148 = metadata !{i32 112, i32 2, metadata !146, null}
!149 = metadata !{i32 113, i32 2, metadata !146, null}
!150 = metadata !{i32 114, i32 2, metadata !146, null}
!151 = metadata !{i32 115, i32 2, metadata !146, null}
!152 = metadata !{i32 116, i32 2, metadata !146, null}
!153 = metadata !{i32 117, i32 2, metadata !146, null}
!154 = metadata !{i32 119, i32 2, metadata !146, null}
!155 = metadata !{i32 120, i32 2, metadata !146, null}
!156 = metadata !{i32 121, i32 2, metadata !146, null}
!157 = metadata !{i32 122, i32 2, metadata !146, null}
!158 = metadata !{i32 124, i32 2, metadata !146, null}
!159 = metadata !{i32 125, i32 2, metadata !146, null}
!160 = metadata !{i32 126, i32 2, metadata !146, null}
!161 = metadata !{i32 128, i32 2, metadata !146, null}
!162 = metadata !{i32 129, i32 2, metadata !146, null}
!163 = metadata !{i32 130, i32 2, metadata !146, null}
!164 = metadata !{i32 132, i32 2, metadata !146, null}
!165 = metadata !{i32 133, i32 2, metadata !146, null}
!166 = metadata !{i32 134, i32 2, metadata !146, null}
!167 = metadata !{i32 135, i32 1, metadata !146, null}
