; ModuleID = 'D:/21_streamer_car5_artix7/fpga_arty/lcddrv/solution1/.autopilot/db/a.o.2.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-w64-mingw32"

@rs = global i1 false, align 1                    ; [#uses=5 type=i1*]
@mem_req = global i1 false, align 1               ; [#uses=5 type=i1*]
@mem_din = common global i8 0, align 1            ; [#uses=3 type=i8*]
@mem_addr = global i5 0, align 1                  ; [#uses=4 type=i5*]
@mem_ack = common global i1 false, align 1        ; [#uses=3 type=i1*]
@ind = global i1 false, align 1                   ; [#uses=2 type=i1*]
@en = global i1 false, align 1                    ; [#uses=6 type=i1*]
@dummy_tmr_out = global i1 false, align 1         ; [#uses=4 type=i1*]
@data = global i4 0, align 1                      ; [#uses=5 type=i4*]
@.str2 = private unnamed_addr constant [12 x i8] c"hls_label_0\00", align 1 ; [#uses=2 type=[12 x i8]*]
@.str1 = private unnamed_addr constant [8 x i8] c"ap_none\00", align 1 ; [#uses=9 type=[8 x i8]*]
@.str = private unnamed_addr constant [1 x i8] zeroinitializer, align 1 ; [#uses=36 type=[1 x i8]*]

; [#uses=13]
define internal fastcc void @wait_tmr(i25 %tmr) {
  call void @llvm.dbg.value(metadata !{i25 %tmr}, i64 0, metadata !72), !dbg !80 ; [debug line = 65:22] [debug variable = tmr]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !81 ; [debug line = 69:2]
  br label %1, !dbg !83                           ; [debug line = 70:7]

; <label>:1                                       ; preds = %3, %0
  %t = phi i24 [ 0, %0 ], [ %t.1, %3 ]            ; [#uses=2 type=i24]
  %t.cast = zext i24 %t to i25, !dbg !83          ; [#uses=1 type=i25] [debug line = 70:7]
  %exitcond = icmp eq i25 %t.cast, %tmr, !dbg !83 ; [#uses=1 type=i1] [debug line = 70:7]
  %2 = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 1000, i64 10000000, i64 0) nounwind ; [#uses=0 type=i32]
  br i1 %exitcond, label %4, label %3, !dbg !83   ; [debug line = 70:7]

; <label>:3                                       ; preds = %1
  %dummy_tmr_out.load = load volatile i1* @dummy_tmr_out, align 1, !dbg !85 ; [#uses=1 type=i1] [debug line = 71:3]
  %not. = xor i1 %dummy_tmr_out.load, true, !dbg !85 ; [#uses=1 type=i1] [debug line = 71:3]
  store volatile i1 %not., i1* @dummy_tmr_out, align 1, !dbg !85 ; [debug line = 71:3]
  %t.1 = add i24 %t, 1, !dbg !87                  ; [#uses=1 type=i24] [debug line = 70:23]
  call void @llvm.dbg.value(metadata !{i24 %t.1}, i64 0, metadata !88), !dbg !87 ; [debug line = 70:23] [debug variable = t]
  br label %1, !dbg !87                           ; [debug line = 70:23]

; <label>:4                                       ; preds = %1
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !89 ; [debug line = 73:2]
  ret void, !dbg !90                              ; [debug line = 74:1]
}

; [#uses=2]
define internal fastcc zeroext i8 @read_mem(i5 zeroext %addr) nounwind uwtable {
  call void @llvm.dbg.value(metadata !{i5 %addr}, i64 0, metadata !91), !dbg !95 ; [debug line = 138:22] [debug variable = addr]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !96 ; [debug line = 143:2]
  store volatile i5 %addr, i5* @mem_addr, align 1, !dbg !98 ; [debug line = 144:2]
  store volatile i1 true, i1* @mem_req, align 1, !dbg !99 ; [debug line = 145:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !100 ; [debug line = 146:2]
  br label %._crit_edge, !dbg !101                ; [debug line = 147:2]

._crit_edge:                                      ; preds = %._crit_edge, %0
  store volatile i5 %addr, i5* @mem_addr, align 1, !dbg !102 ; [debug line = 148:3]
  store volatile i1 true, i1* @mem_req, align 1, !dbg !104 ; [debug line = 149:3]
  %dt = load volatile i8* @mem_din, align 1, !dbg !105 ; [#uses=1 type=i8] [debug line = 150:3]
  call void @llvm.dbg.value(metadata !{i8 %dt}, i64 0, metadata !106), !dbg !105 ; [debug line = 150:3] [debug variable = dt]
  %mem_ack.load = load volatile i1* @mem_ack, align 1, !dbg !107 ; [#uses=1 type=i1] [debug line = 151:2]
  br i1 %mem_ack.load, label %1, label %._crit_edge, !dbg !107 ; [debug line = 151:2]

; <label>:1                                       ; preds = %._crit_edge
  %dt.lcssa = phi i8 [ %dt, %._crit_edge ]        ; [#uses=1 type=i8]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !108 ; [debug line = 152:2]
  store volatile i1 false, i1* @mem_req, align 1, !dbg !109 ; [debug line = 153:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !110 ; [debug line = 154:2]
  ret i8 %dt.lcssa, !dbg !111                     ; [debug line = 156:2]
}

; [#uses=10]
declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

; [#uses=0]
define void @lcddrv() noreturn nounwind uwtable {
  call void (...)* @_ssdm_op_SpecTopModule(), !dbg !112 ; [debug line = 162:1]
  %dummy_tmr_out.load = load volatile i1* @dummy_tmr_out, align 1, !dbg !117 ; [#uses=0 type=i1] [debug line = 163:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @dummy_tmr_out, [8 x i8]* @.str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !117 ; [debug line = 163:1]
  %rs.load = load volatile i1* @rs, align 1, !dbg !118 ; [#uses=0 type=i1] [debug line = 165:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @rs, [8 x i8]* @.str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !118 ; [debug line = 165:1]
  %en.load = load volatile i1* @en, align 1, !dbg !119 ; [#uses=0 type=i1] [debug line = 166:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @en, [8 x i8]* @.str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !119 ; [debug line = 166:1]
  %data.load = load volatile i4* @data, align 1, !dbg !120 ; [#uses=0 type=i4] [debug line = 167:1]
  call void (...)* @_ssdm_op_SpecInterface(i4* @data, [8 x i8]* @.str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !120 ; [debug line = 167:1]
  %ind.load = load volatile i1* @ind, align 1, !dbg !121 ; [#uses=0 type=i1] [debug line = 168:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @ind, [8 x i8]* @.str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !121 ; [debug line = 168:1]
  %mem_addr.load = load volatile i5* @mem_addr, align 1, !dbg !122 ; [#uses=0 type=i5] [debug line = 170:1]
  call void (...)* @_ssdm_op_SpecInterface(i5* @mem_addr, [8 x i8]* @.str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !122 ; [debug line = 170:1]
  %mem_din.load = load volatile i8* @mem_din, align 1, !dbg !123 ; [#uses=0 type=i8] [debug line = 171:1]
  call void (...)* @_ssdm_op_SpecInterface(i8* @mem_din, [8 x i8]* @.str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !123 ; [debug line = 171:1]
  %mem_req.load = load volatile i1* @mem_req, align 1, !dbg !124 ; [#uses=0 type=i1] [debug line = 172:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @mem_req, [8 x i8]* @.str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !124 ; [debug line = 172:1]
  %mem_ack.load = load volatile i1* @mem_ack, align 1, !dbg !125 ; [#uses=0 type=i1] [debug line = 173:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @mem_ack, [8 x i8]* @.str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !125 ; [debug line = 173:1]
  call fastcc void @init_lcd(), !dbg !126         ; [debug line = 178:2]
  br label %1, !dbg !127                          ; [debug line = 180:2]

; <label>:1                                       ; preds = %9, %0
  %loop_begin = call i32 (...)* @_ssdm_op_SpecLoopBegin() nounwind ; [#uses=0 type=i32]
  %tmp = call i32 (...)* @_ssdm_op_SpecRegionBegin([12 x i8]* @.str2) nounwind, !dbg !128 ; [#uses=1 type=i32] [debug line = 180:13]
  call fastcc void @lcd_send_cmd(i1 zeroext false, i4 zeroext -8), !dbg !130 ; [debug line = 182:3]
  call fastcc void @lcd_send_cmd(i1 zeroext false, i4 zeroext 0), !dbg !131 ; [debug line = 183:3]
  br label %2, !dbg !132                          ; [debug line = 185:8]

; <label>:2                                       ; preds = %4, %1
  %pos = phi i5 [ 0, %1 ], [ %pos.2, %4 ]         ; [#uses=3 type=i5]
  %exitcond1 = icmp eq i5 %pos, -16, !dbg !132    ; [#uses=1 type=i1] [debug line = 185:8]
  %3 = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 16, i64 16, i64 16) nounwind ; [#uses=0 type=i32]
  br i1 %exitcond1, label %5, label %4, !dbg !132 ; [debug line = 185:8]

; <label>:4                                       ; preds = %2
  %dt = call fastcc zeroext i8 @read_mem(i5 zeroext %pos), !dbg !134 ; [#uses=2 type=i8] [debug line = 186:9]
  call void @llvm.dbg.value(metadata !{i8 %dt}, i64 0, metadata !136), !dbg !134 ; [debug line = 186:9] [debug variable = dt]
  %tmp. = lshr i8 %dt, 4, !dbg !137               ; [#uses=1 type=i8] [debug line = 187:4]
  %tmp.1 = trunc i8 %tmp. to i4, !dbg !137        ; [#uses=1 type=i4] [debug line = 187:4]
  call fastcc void @lcd_send_cmd(i1 zeroext true, i4 zeroext %tmp.1), !dbg !137 ; [debug line = 187:4]
  %tmp.2 = trunc i8 %dt to i4, !dbg !138          ; [#uses=1 type=i4] [debug line = 188:4]
  call fastcc void @lcd_send_cmd(i1 zeroext true, i4 zeroext %tmp.2), !dbg !138 ; [debug line = 188:4]
  %pos.2 = add i5 %pos, 1, !dbg !139              ; [#uses=1 type=i5] [debug line = 185:27]
  call void @llvm.dbg.value(metadata !{i5 %pos.2}, i64 0, metadata !140), !dbg !139 ; [debug line = 185:27] [debug variable = pos]
  br label %2, !dbg !139                          ; [debug line = 185:27]

; <label>:5                                       ; preds = %2
  call fastcc void @lcd_send_cmd(i1 zeroext false, i4 zeroext -4), !dbg !141 ; [debug line = 192:3]
  call fastcc void @lcd_send_cmd(i1 zeroext false, i4 zeroext 0), !dbg !142 ; [debug line = 193:3]
  br label %6, !dbg !143                          ; [debug line = 195:8]

; <label>:6                                       ; preds = %8, %5
  %pos.1 = phi i6 [ 16, %5 ], [ %pos.3, %8 ]      ; [#uses=3 type=i6]
  %exitcond = icmp eq i6 %pos.1, -32, !dbg !143   ; [#uses=1 type=i1] [debug line = 195:8]
  %7 = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 16, i64 16, i64 16) nounwind ; [#uses=0 type=i32]
  br i1 %exitcond, label %9, label %8, !dbg !143  ; [debug line = 195:8]

; <label>:8                                       ; preds = %6
  %tmp.3 = trunc i6 %pos.1 to i5, !dbg !145       ; [#uses=1 type=i5] [debug line = 196:9]
  %dt.1 = call fastcc zeroext i8 @read_mem(i5 zeroext %tmp.3), !dbg !145 ; [#uses=2 type=i8] [debug line = 196:9]
  call void @llvm.dbg.value(metadata !{i8 %dt.1}, i64 0, metadata !136), !dbg !145 ; [debug line = 196:9] [debug variable = dt]
  %tmp.4 = lshr i8 %dt.1, 4, !dbg !147            ; [#uses=1 type=i8] [debug line = 197:4]
  %tmp.5 = trunc i8 %tmp.4 to i4, !dbg !147       ; [#uses=1 type=i4] [debug line = 197:4]
  call fastcc void @lcd_send_cmd(i1 zeroext true, i4 zeroext %tmp.5), !dbg !147 ; [debug line = 197:4]
  %tmp.6 = trunc i8 %dt.1 to i4, !dbg !148        ; [#uses=1 type=i4] [debug line = 198:4]
  call fastcc void @lcd_send_cmd(i1 zeroext true, i4 zeroext %tmp.6), !dbg !148 ; [debug line = 198:4]
  %pos.3 = add i6 %pos.1, 1, !dbg !149            ; [#uses=1 type=i6] [debug line = 195:28]
  call void @llvm.dbg.value(metadata !{i6 %pos.3}, i64 0, metadata !140), !dbg !149 ; [debug line = 195:28] [debug variable = pos]
  br label %6, !dbg !149                          ; [debug line = 195:28]

; <label>:9                                       ; preds = %6
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !150 ; [debug line = 202:3]
  call fastcc void @wait_tmr(i25 10000000) nounwind, !dbg !151 ; [debug line = 203:3]
  %10 = call i32 (...)* @_ssdm_op_SpecRegionEnd([12 x i8]* @.str2, i32 %tmp) nounwind, !dbg !152 ; [#uses=0 type=i32] [debug line = 204:2]
  br label %1, !dbg !152                          ; [debug line = 204:2]
}

; [#uses=20]
define internal fastcc void @lcd_send_cmd(i1 zeroext %mode, i4 zeroext %wd) nounwind uwtable {
  call void @llvm.dbg.value(metadata !{i1 %mode}, i64 0, metadata !153), !dbg !157 ; [debug line = 77:25] [debug variable = mode]
  call void @llvm.dbg.value(metadata !{i4 %wd}, i64 0, metadata !158), !dbg !159 ; [debug line = 77:37] [debug variable = wd]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !160 ; [debug line = 80:2]
  store volatile i1 false, i1* @en, align 1, !dbg !162 ; [debug line = 81:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !163 ; [debug line = 82:2]
  call fastcc void @wait_tmr(i25 1000) nounwind, !dbg !164 ; [debug line = 83:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !165 ; [debug line = 85:2]
  store volatile i1 false, i1* @en, align 1, !dbg !166 ; [debug line = 86:2]
  store volatile i1 %mode, i1* @rs, align 1, !dbg !167 ; [debug line = 87:2]
  store volatile i4 %wd, i4* @data, align 1, !dbg !168 ; [debug line = 88:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !169 ; [debug line = 89:2]
  call fastcc void @wait_tmr(i25 1000) nounwind, !dbg !170 ; [debug line = 90:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !171 ; [debug line = 92:2]
  store volatile i1 true, i1* @en, align 1, !dbg !172 ; [debug line = 93:2]
  store volatile i1 %mode, i1* @rs, align 1, !dbg !173 ; [debug line = 94:2]
  store volatile i4 %wd, i4* @data, align 1, !dbg !174 ; [debug line = 95:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !175 ; [debug line = 96:2]
  call fastcc void @wait_tmr(i25 1000) nounwind, !dbg !176 ; [debug line = 97:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !177 ; [debug line = 99:2]
  store volatile i1 false, i1* @en, align 1, !dbg !178 ; [debug line = 100:2]
  store volatile i1 %mode, i1* @rs, align 1, !dbg !179 ; [debug line = 101:2]
  store volatile i4 %wd, i4* @data, align 1, !dbg !180 ; [debug line = 102:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !181 ; [debug line = 103:2]
  call fastcc void @wait_tmr(i25 1000) nounwind, !dbg !182 ; [debug line = 104:2]
  ret void, !dbg !183                             ; [debug line = 105:1]
}

; [#uses=1]
define internal fastcc void @init_lcd() nounwind uwtable {
  call fastcc void @wait_tmr(i25 2000000) nounwind, !dbg !184 ; [debug line = 111:2]
  call fastcc void @lcd_send_cmd(i1 zeroext false, i4 zeroext 3), !dbg !187 ; [debug line = 112:2]
  call fastcc void @wait_tmr(i25 500000) nounwind, !dbg !188 ; [debug line = 113:2]
  call fastcc void @lcd_send_cmd(i1 zeroext false, i4 zeroext 3), !dbg !189 ; [debug line = 114:2]
  call fastcc void @wait_tmr(i25 50000) nounwind, !dbg !190 ; [debug line = 115:2]
  call fastcc void @lcd_send_cmd(i1 zeroext false, i4 zeroext 3), !dbg !191 ; [debug line = 116:2]
  call fastcc void @wait_tmr(i25 50000) nounwind, !dbg !192 ; [debug line = 117:2]
  call fastcc void @lcd_send_cmd(i1 zeroext false, i4 zeroext 2), !dbg !193 ; [debug line = 119:2]
  call fastcc void @lcd_send_cmd(i1 zeroext false, i4 zeroext 2), !dbg !194 ; [debug line = 120:2]
  call fastcc void @lcd_send_cmd(i1 zeroext false, i4 zeroext -8), !dbg !195 ; [debug line = 121:2]
  call fastcc void @wait_tmr(i25 10000) nounwind, !dbg !196 ; [debug line = 122:2]
  call fastcc void @lcd_send_cmd(i1 zeroext false, i4 zeroext 0), !dbg !197 ; [debug line = 124:2]
  call fastcc void @lcd_send_cmd(i1 zeroext false, i4 zeroext -4), !dbg !198 ; [debug line = 125:2]
  call fastcc void @wait_tmr(i25 10000) nounwind, !dbg !199 ; [debug line = 126:2]
  call fastcc void @lcd_send_cmd(i1 zeroext false, i4 zeroext 0), !dbg !200 ; [debug line = 128:2]
  call fastcc void @lcd_send_cmd(i1 zeroext false, i4 zeroext 1), !dbg !201 ; [debug line = 129:2]
  call fastcc void @wait_tmr(i25 200000) nounwind, !dbg !202 ; [debug line = 130:2]
  call fastcc void @lcd_send_cmd(i1 zeroext false, i4 zeroext 0), !dbg !203 ; [debug line = 132:2]
  call fastcc void @lcd_send_cmd(i1 zeroext false, i4 zeroext 2), !dbg !204 ; [debug line = 133:2]
  call fastcc void @wait_tmr(i25 10000) nounwind, !dbg !205 ; [debug line = 134:2]
  ret void, !dbg !206                             ; [debug line = 135:1]
}

; [#uses=15]
declare void @_ssdm_op_Wait(...) nounwind

; [#uses=1]
declare void @_ssdm_op_SpecTopModule(...) nounwind

; [#uses=1]
declare i32 @_ssdm_op_SpecRegionEnd(...)

; [#uses=1]
declare i32 @_ssdm_op_SpecRegionBegin(...)

; [#uses=3]
declare i32 @_ssdm_op_SpecLoopTripCount(...)

; [#uses=1]
declare i32 @_ssdm_op_SpecLoopBegin(...)

; [#uses=9]
declare void @_ssdm_op_SpecInterface(...) nounwind

; [#uses=0]
declare void @_ssdm_SpecKeepAssert(...)

!hls.encrypted.func = !{}
!llvm.map.gv = !{!0, !7, !12, !17, !22, !27, !32, !37, !42}
!llvm.dbg.cu = !{!47}

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
!47 = metadata !{i32 786449, i32 0, i32 1, metadata !"D:/21_streamer_car5_artix7/fpga_arty/lcddrv/solution1/.autopilot/db/lcddrv.pragma.2.c", metadata !"D:\5C21_streamer_car5_artix7\5Cfpga_arty", metadata !"clang version 3.1 ", i1 true, i1 false, metadata !"", i32 0, null, null, null, metadata !48} ; [ DW_TAG_compile_unit ]
!48 = metadata !{metadata !49}
!49 = metadata !{metadata !50, metadata !55, metadata !56, metadata !57, metadata !58, metadata !62, metadata !63, metadata !64, metadata !68}
!50 = metadata !{i32 786484, i32 0, null, metadata !"en", metadata !"en", metadata !"", metadata !51, i32 48, metadata !52, i32 0, i32 1, i1* @en} ; [ DW_TAG_variable ]
!51 = metadata !{i32 786473, metadata !"lcddrv.c", metadata !"D:\5C21_streamer_car5_artix7\5Cfpga_arty", null} ; [ DW_TAG_file_type ]
!52 = metadata !{i32 786485, null, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !53} ; [ DW_TAG_volatile_type ]
!53 = metadata !{i32 786454, null, metadata !"uint1", metadata !51, i32 3, i64 0, i64 0, i64 0, i32 0, metadata !54} ; [ DW_TAG_typedef ]
!54 = metadata !{i32 786468, null, metadata !"uint1", null, i32 0, i64 1, i64 1, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!55 = metadata !{i32 786484, i32 0, null, metadata !"mem_req", metadata !"mem_req", metadata !"", metadata !51, i32 55, metadata !52, i32 0, i32 1, i1* @mem_req} ; [ DW_TAG_variable ]
!56 = metadata !{i32 786484, i32 0, null, metadata !"dummy_tmr_out", metadata !"dummy_tmr_out", metadata !"", metadata !51, i32 59, metadata !52, i32 0, i32 1, i1* @dummy_tmr_out} ; [ DW_TAG_variable ]
!57 = metadata !{i32 786484, i32 0, null, metadata !"mem_ack", metadata !"mem_ack", metadata !"", metadata !51, i32 56, metadata !52, i32 0, i32 1, i1* @mem_ack} ; [ DW_TAG_variable ]
!58 = metadata !{i32 786484, i32 0, null, metadata !"mem_addr", metadata !"mem_addr", metadata !"", metadata !51, i32 53, metadata !59, i32 0, i32 1, i5* @mem_addr} ; [ DW_TAG_variable ]
!59 = metadata !{i32 786485, null, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !60} ; [ DW_TAG_volatile_type ]
!60 = metadata !{i32 786454, null, metadata !"uint5", metadata !51, i32 7, i64 0, i64 0, i64 0, i32 0, metadata !61} ; [ DW_TAG_typedef ]
!61 = metadata !{i32 786468, null, metadata !"uint5", null, i32 0, i64 5, i64 8, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!62 = metadata !{i32 786484, i32 0, null, metadata !"ind", metadata !"ind", metadata !"", metadata !51, i32 50, metadata !52, i32 0, i32 1, i1* @ind} ; [ DW_TAG_variable ]
!63 = metadata !{i32 786484, i32 0, null, metadata !"rs", metadata !"rs", metadata !"", metadata !51, i32 47, metadata !52, i32 0, i32 1, i1* @rs} ; [ DW_TAG_variable ]
!64 = metadata !{i32 786484, i32 0, null, metadata !"mem_din", metadata !"mem_din", metadata !"", metadata !51, i32 54, metadata !65, i32 0, i32 1, i8* @mem_din} ; [ DW_TAG_variable ]
!65 = metadata !{i32 786485, null, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !66} ; [ DW_TAG_volatile_type ]
!66 = metadata !{i32 786454, null, metadata !"uint8", metadata !51, i32 10, i64 0, i64 0, i64 0, i32 0, metadata !67} ; [ DW_TAG_typedef ]
!67 = metadata !{i32 786468, null, metadata !"uint8", null, i32 0, i64 8, i64 8, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!68 = metadata !{i32 786484, i32 0, null, metadata !"data", metadata !"data", metadata !"", metadata !51, i32 49, metadata !69, i32 0, i32 1, i4* @data} ; [ DW_TAG_variable ]
!69 = metadata !{i32 786485, null, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !70} ; [ DW_TAG_volatile_type ]
!70 = metadata !{i32 786454, null, metadata !"uint4", metadata !51, i32 6, i64 0, i64 0, i64 0, i32 0, metadata !71} ; [ DW_TAG_typedef ]
!71 = metadata !{i32 786468, null, metadata !"uint4", null, i32 0, i64 4, i64 4, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!72 = metadata !{i32 786689, metadata !73, metadata !"tmr", metadata !51, i32 16777281, metadata !76, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!73 = metadata !{i32 786478, i32 0, metadata !51, metadata !"wait_tmr", metadata !"wait_tmr", metadata !"", metadata !51, i32 65, metadata !74, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, metadata !78, i32 66} ; [ DW_TAG_subprogram ]
!74 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !75, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!75 = metadata !{null, metadata !76}
!76 = metadata !{i32 786454, null, metadata !"uint32", metadata !51, i32 34, i64 0, i64 0, i64 0, i32 0, metadata !77} ; [ DW_TAG_typedef ]
!77 = metadata !{i32 786468, null, metadata !"uint32", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!78 = metadata !{metadata !79}
!79 = metadata !{i32 786468}                      ; [ DW_TAG_base_type ]
!80 = metadata !{i32 65, i32 22, metadata !73, null}
!81 = metadata !{i32 69, i32 2, metadata !82, null}
!82 = metadata !{i32 786443, metadata !73, i32 66, i32 1, metadata !51, i32 0} ; [ DW_TAG_lexical_block ]
!83 = metadata !{i32 70, i32 7, metadata !84, null}
!84 = metadata !{i32 786443, metadata !82, i32 70, i32 2, metadata !51, i32 1} ; [ DW_TAG_lexical_block ]
!85 = metadata !{i32 71, i32 3, metadata !86, null}
!86 = metadata !{i32 786443, metadata !84, i32 70, i32 28, metadata !51, i32 2} ; [ DW_TAG_lexical_block ]
!87 = metadata !{i32 70, i32 23, metadata !84, null}
!88 = metadata !{i32 786688, metadata !82, metadata !"t", metadata !51, i32 68, metadata !76, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!89 = metadata !{i32 73, i32 2, metadata !82, null}
!90 = metadata !{i32 74, i32 1, metadata !82, null}
!91 = metadata !{i32 786689, metadata !92, metadata !"addr", metadata !51, i32 16777354, metadata !60, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!92 = metadata !{i32 786478, i32 0, metadata !51, metadata !"read_mem", metadata !"read_mem", metadata !"", metadata !51, i32 138, metadata !93, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i8 (i5)* @read_mem, null, null, metadata !78, i32 139} ; [ DW_TAG_subprogram ]
!93 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !94, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!94 = metadata !{metadata !66, metadata !60}
!95 = metadata !{i32 138, i32 22, metadata !92, null}
!96 = metadata !{i32 143, i32 2, metadata !97, null}
!97 = metadata !{i32 786443, metadata !92, i32 139, i32 1, metadata !51, i32 5} ; [ DW_TAG_lexical_block ]
!98 = metadata !{i32 144, i32 2, metadata !97, null}
!99 = metadata !{i32 145, i32 2, metadata !97, null}
!100 = metadata !{i32 146, i32 2, metadata !97, null}
!101 = metadata !{i32 147, i32 2, metadata !97, null}
!102 = metadata !{i32 148, i32 3, metadata !103, null}
!103 = metadata !{i32 786443, metadata !97, i32 147, i32 5, metadata !51, i32 6} ; [ DW_TAG_lexical_block ]
!104 = metadata !{i32 149, i32 3, metadata !103, null}
!105 = metadata !{i32 150, i32 3, metadata !103, null}
!106 = metadata !{i32 786688, metadata !97, metadata !"dt", metadata !51, i32 141, metadata !66, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!107 = metadata !{i32 151, i32 2, metadata !103, null}
!108 = metadata !{i32 152, i32 2, metadata !97, null}
!109 = metadata !{i32 153, i32 2, metadata !97, null}
!110 = metadata !{i32 154, i32 2, metadata !97, null}
!111 = metadata !{i32 156, i32 2, metadata !97, null}
!112 = metadata !{i32 162, i32 1, metadata !113, null}
!113 = metadata !{i32 786443, metadata !114, i32 161, i32 1, metadata !51, i32 7} ; [ DW_TAG_lexical_block ]
!114 = metadata !{i32 786478, i32 0, metadata !51, metadata !"lcddrv", metadata !"lcddrv", metadata !"", metadata !51, i32 160, metadata !115, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 false, void ()* @lcddrv, null, null, metadata !78, i32 161} ; [ DW_TAG_subprogram ]
!115 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !116, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!116 = metadata !{null}
!117 = metadata !{i32 163, i32 1, metadata !113, null}
!118 = metadata !{i32 165, i32 1, metadata !113, null}
!119 = metadata !{i32 166, i32 1, metadata !113, null}
!120 = metadata !{i32 167, i32 1, metadata !113, null}
!121 = metadata !{i32 168, i32 1, metadata !113, null}
!122 = metadata !{i32 170, i32 1, metadata !113, null}
!123 = metadata !{i32 171, i32 1, metadata !113, null}
!124 = metadata !{i32 172, i32 1, metadata !113, null}
!125 = metadata !{i32 173, i32 1, metadata !113, null}
!126 = metadata !{i32 178, i32 2, metadata !113, null}
!127 = metadata !{i32 180, i32 2, metadata !113, null}
!128 = metadata !{i32 180, i32 13, metadata !129, null}
!129 = metadata !{i32 786443, metadata !113, i32 180, i32 12, metadata !51, i32 8} ; [ DW_TAG_lexical_block ]
!130 = metadata !{i32 182, i32 3, metadata !129, null}
!131 = metadata !{i32 183, i32 3, metadata !129, null}
!132 = metadata !{i32 185, i32 8, metadata !133, null}
!133 = metadata !{i32 786443, metadata !129, i32 185, i32 3, metadata !51, i32 9} ; [ DW_TAG_lexical_block ]
!134 = metadata !{i32 186, i32 9, metadata !135, null}
!135 = metadata !{i32 786443, metadata !133, i32 185, i32 34, metadata !51, i32 10} ; [ DW_TAG_lexical_block ]
!136 = metadata !{i32 786688, metadata !113, metadata !"dt", metadata !51, i32 176, metadata !66, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!137 = metadata !{i32 187, i32 4, metadata !135, null}
!138 = metadata !{i32 188, i32 4, metadata !135, null}
!139 = metadata !{i32 185, i32 27, metadata !133, null}
!140 = metadata !{i32 786688, metadata !113, metadata !"pos", metadata !51, i32 175, metadata !66, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!141 = metadata !{i32 192, i32 3, metadata !129, null}
!142 = metadata !{i32 193, i32 3, metadata !129, null}
!143 = metadata !{i32 195, i32 8, metadata !144, null}
!144 = metadata !{i32 786443, metadata !129, i32 195, i32 3, metadata !51, i32 11} ; [ DW_TAG_lexical_block ]
!145 = metadata !{i32 196, i32 9, metadata !146, null}
!146 = metadata !{i32 786443, metadata !144, i32 195, i32 35, metadata !51, i32 12} ; [ DW_TAG_lexical_block ]
!147 = metadata !{i32 197, i32 4, metadata !146, null}
!148 = metadata !{i32 198, i32 4, metadata !146, null}
!149 = metadata !{i32 195, i32 28, metadata !144, null}
!150 = metadata !{i32 202, i32 3, metadata !129, null}
!151 = metadata !{i32 203, i32 3, metadata !129, null}
!152 = metadata !{i32 204, i32 2, metadata !129, null}
!153 = metadata !{i32 786689, metadata !154, metadata !"mode", metadata !51, i32 16777293, metadata !53, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!154 = metadata !{i32 786478, i32 0, metadata !51, metadata !"lcd_send_cmd", metadata !"lcd_send_cmd", metadata !"", metadata !51, i32 77, metadata !155, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (i1, i4)* @lcd_send_cmd, null, null, metadata !78, i32 78} ; [ DW_TAG_subprogram ]
!155 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !156, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!156 = metadata !{null, metadata !53, metadata !70}
!157 = metadata !{i32 77, i32 25, metadata !154, null}
!158 = metadata !{i32 786689, metadata !154, metadata !"wd", metadata !51, i32 33554509, metadata !70, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!159 = metadata !{i32 77, i32 37, metadata !154, null}
!160 = metadata !{i32 80, i32 2, metadata !161, null}
!161 = metadata !{i32 786443, metadata !154, i32 78, i32 1, metadata !51, i32 3} ; [ DW_TAG_lexical_block ]
!162 = metadata !{i32 81, i32 2, metadata !161, null}
!163 = metadata !{i32 82, i32 2, metadata !161, null}
!164 = metadata !{i32 83, i32 2, metadata !161, null}
!165 = metadata !{i32 85, i32 2, metadata !161, null}
!166 = metadata !{i32 86, i32 2, metadata !161, null}
!167 = metadata !{i32 87, i32 2, metadata !161, null}
!168 = metadata !{i32 88, i32 2, metadata !161, null}
!169 = metadata !{i32 89, i32 2, metadata !161, null}
!170 = metadata !{i32 90, i32 2, metadata !161, null}
!171 = metadata !{i32 92, i32 2, metadata !161, null}
!172 = metadata !{i32 93, i32 2, metadata !161, null}
!173 = metadata !{i32 94, i32 2, metadata !161, null}
!174 = metadata !{i32 95, i32 2, metadata !161, null}
!175 = metadata !{i32 96, i32 2, metadata !161, null}
!176 = metadata !{i32 97, i32 2, metadata !161, null}
!177 = metadata !{i32 99, i32 2, metadata !161, null}
!178 = metadata !{i32 100, i32 2, metadata !161, null}
!179 = metadata !{i32 101, i32 2, metadata !161, null}
!180 = metadata !{i32 102, i32 2, metadata !161, null}
!181 = metadata !{i32 103, i32 2, metadata !161, null}
!182 = metadata !{i32 104, i32 2, metadata !161, null}
!183 = metadata !{i32 105, i32 1, metadata !161, null}
!184 = metadata !{i32 111, i32 2, metadata !185, null}
!185 = metadata !{i32 786443, metadata !186, i32 109, i32 1, metadata !51, i32 4} ; [ DW_TAG_lexical_block ]
!186 = metadata !{i32 786478, i32 0, metadata !51, metadata !"init_lcd", metadata !"init_lcd", metadata !"", metadata !51, i32 108, metadata !115, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 false, void ()* @init_lcd, null, null, metadata !78, i32 109} ; [ DW_TAG_subprogram ]
!187 = metadata !{i32 112, i32 2, metadata !185, null}
!188 = metadata !{i32 113, i32 2, metadata !185, null}
!189 = metadata !{i32 114, i32 2, metadata !185, null}
!190 = metadata !{i32 115, i32 2, metadata !185, null}
!191 = metadata !{i32 116, i32 2, metadata !185, null}
!192 = metadata !{i32 117, i32 2, metadata !185, null}
!193 = metadata !{i32 119, i32 2, metadata !185, null}
!194 = metadata !{i32 120, i32 2, metadata !185, null}
!195 = metadata !{i32 121, i32 2, metadata !185, null}
!196 = metadata !{i32 122, i32 2, metadata !185, null}
!197 = metadata !{i32 124, i32 2, metadata !185, null}
!198 = metadata !{i32 125, i32 2, metadata !185, null}
!199 = metadata !{i32 126, i32 2, metadata !185, null}
!200 = metadata !{i32 128, i32 2, metadata !185, null}
!201 = metadata !{i32 129, i32 2, metadata !185, null}
!202 = metadata !{i32 130, i32 2, metadata !185, null}
!203 = metadata !{i32 132, i32 2, metadata !185, null}
!204 = metadata !{i32 133, i32 2, metadata !185, null}
!205 = metadata !{i32 134, i32 2, metadata !185, null}
!206 = metadata !{i32 135, i32 1, metadata !185, null}
