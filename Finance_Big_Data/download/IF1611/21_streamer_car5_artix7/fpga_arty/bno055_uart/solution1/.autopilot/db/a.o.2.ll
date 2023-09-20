; ModuleID = 'D:/21_streamer_car5_artix7/fpga_arty/bno055_uart/solution1/.autopilot/db/a.o.2.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-w64-mingw32"

@uart_tx = common global i1 false, align 1        ; [#uses=6 type=i1*]
@uart_rx = common global i1 false, align 1        ; [#uses=3 type=i1*]
@mem_wreq = global i1 false, align 1              ; [#uses=5 type=i1*]
@mem_wack = common global i1 false, align 1       ; [#uses=3 type=i1*]
@mem_rreq = global i1 false, align 1              ; [#uses=2 type=i1*]
@mem_rack = common global i1 false, align 1       ; [#uses=2 type=i1*]
@mem_dout = global i8 0, align 1                  ; [#uses=5 type=i8*]
@mem_din = common global i8 0, align 1            ; [#uses=2 type=i8*]
@mem_addr = global i8 0, align 1                  ; [#uses=5 type=i8*]
@dummy_tmr_out = global i1 false, align 1         ; [#uses=4 type=i1*]
@.str6 = private unnamed_addr constant [8 x i8] c"ap_none\00", align 1 ; [#uses=10 type=[8 x i8]*]
@.str = private unnamed_addr constant [1 x i8] zeroinitializer, align 1 ; [#uses=40 type=[1 x i8]*]

; [#uses=8]
define internal fastcc void @write_mem(i8 zeroext %addr, i8 zeroext %data) nounwind uwtable {
  call void (...)* @_ssdm_SpecKeepAssert(i8 %addr) nounwind, !hlsrange !72
  call void @llvm.dbg.value(metadata !{i8 %addr}, i64 0, metadata !73), !dbg !79 ; [debug line = 272:22] [debug variable = addr]
  call void @llvm.dbg.value(metadata !{i8 %data}, i64 0, metadata !80), !dbg !81 ; [debug line = 272:34] [debug variable = data]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !82 ; [debug line = 275:2]
  store volatile i8 %addr, i8* @mem_addr, align 1, !dbg !84 ; [debug line = 276:2]
  store volatile i8 %data, i8* @mem_dout, align 1, !dbg !85 ; [debug line = 277:2]
  store volatile i1 false, i1* @mem_wreq, align 1, !dbg !86 ; [debug line = 278:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !87 ; [debug line = 279:2]
  br label %._crit_edge, !dbg !88                 ; [debug line = 281:2]

._crit_edge:                                      ; preds = %._crit_edge, %0
  store volatile i8 %addr, i8* @mem_addr, align 1, !dbg !89 ; [debug line = 282:3]
  store volatile i8 %data, i8* @mem_dout, align 1, !dbg !91 ; [debug line = 283:3]
  store volatile i1 true, i1* @mem_wreq, align 1, !dbg !92 ; [debug line = 284:3]
  %mem_wack.load = load volatile i1* @mem_wack, align 1, !dbg !93 ; [#uses=1 type=i1] [debug line = 285:2]
  br i1 %mem_wack.load, label %1, label %._crit_edge, !dbg !93 ; [debug line = 285:2]

; <label>:1                                       ; preds = %._crit_edge
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !94 ; [debug line = 286:2]
  store volatile i8 %addr, i8* @mem_addr, align 1, !dbg !95 ; [debug line = 288:2]
  store volatile i8 %data, i8* @mem_dout, align 1, !dbg !96 ; [debug line = 289:2]
  store volatile i1 false, i1* @mem_wreq, align 1, !dbg !97 ; [debug line = 290:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !98 ; [debug line = 291:2]
  ret void, !dbg !99                              ; [debug line = 292:1]
}

; [#uses=15]
define internal fastcc void @wait_tmr(i28 %tmr) {
  call void @llvm.dbg.value(metadata !{i28 %tmr}, i64 0, metadata !100), !dbg !106 ; [debug line = 74:22] [debug variable = tmr]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !107 ; [debug line = 78:2]
  br label %1, !dbg !109                          ; [debug line = 79:7]

; <label>:1                                       ; preds = %3, %0
  %t = phi i27 [ 0, %0 ], [ %t.1, %3 ]            ; [#uses=2 type=i27]
  %t.cast = zext i27 %t to i28, !dbg !109         ; [#uses=1 type=i28] [debug line = 79:7]
  %exitcond = icmp eq i28 %t.cast, %tmr, !dbg !109 ; [#uses=1 type=i1] [debug line = 79:7]
  %2 = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 1, i64 100000000, i64 0) nounwind ; [#uses=0 type=i32]
  br i1 %exitcond, label %4, label %3, !dbg !109  ; [debug line = 79:7]

; <label>:3                                       ; preds = %1
  %dummy_tmr_out.load = load volatile i1* @dummy_tmr_out, align 1, !dbg !111 ; [#uses=1 type=i1] [debug line = 80:3]
  %not. = xor i1 %dummy_tmr_out.load, true, !dbg !111 ; [#uses=1 type=i1] [debug line = 80:3]
  store volatile i1 %not., i1* @dummy_tmr_out, align 1, !dbg !111 ; [debug line = 80:3]
  %t.1 = add i27 %t, 1, !dbg !113                 ; [#uses=1 type=i27] [debug line = 79:23]
  call void @llvm.dbg.value(metadata !{i27 %t.1}, i64 0, metadata !114), !dbg !113 ; [debug line = 79:23] [debug variable = t]
  br label %1, !dbg !113                          ; [debug line = 79:23]

; <label>:4                                       ; preds = %1
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !115 ; [debug line = 82:2]
  ret void, !dbg !116                             ; [debug line = 83:1]
}

; [#uses=6]
define internal fastcc void @uart_write_reg(i7 zeroext %reg_addr, i8 %data) {
  %reg_addr.cast = zext i7 %reg_addr to i8        ; [#uses=2 type=i8]
  call void (...)* @_ssdm_SpecKeepAssert(i8 %reg_addr.cast) nounwind, !hlsrange !117
  call void @llvm.dbg.value(metadata !{i7 %reg_addr}, i64 0, metadata !118), !dbg !120 ; [debug line = 174:27] [debug variable = reg_addr]
  call void @llvm.dbg.value(metadata !{i8 %data}, i64 0, metadata !121), !dbg !122 ; [debug line = 174:43] [debug variable = data]
  call fastcc void @uart_send_byte(i8 zeroext -86), !dbg !123 ; [debug line = 177:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !125 ; [debug line = 178:2]
  call fastcc void @uart_send_byte(i8 zeroext 0), !dbg !126 ; [debug line = 179:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !127 ; [debug line = 180:2]
  call fastcc void @uart_send_byte(i8 zeroext %reg_addr.cast), !dbg !128 ; [debug line = 181:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !129 ; [debug line = 182:2]
  call fastcc void @uart_send_byte(i8 zeroext 1), !dbg !130 ; [debug line = 183:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !131 ; [debug line = 184:2]
  call fastcc void @uart_send_byte(i8 zeroext %data), !dbg !132 ; [debug line = 185:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !133 ; [debug line = 186:2]
  call fastcc void @wait_tmr(i28 1), !dbg !134    ; [debug line = 188:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !135 ; [debug line = 190:2]
  %1 = call fastcc zeroext i8 @uart_receive_byte(), !dbg !136 ; [#uses=0 type=i8] [debug line = 191:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !137 ; [debug line = 192:2]
  %2 = call fastcc zeroext i8 @uart_receive_byte(), !dbg !138 ; [#uses=0 type=i8] [debug line = 193:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !139 ; [debug line = 194:2]
  ret void, !dbg !140                             ; [debug line = 195:1]
}

; [#uses=13]
define internal fastcc void @uart_send_byte(i8 zeroext %data) nounwind uwtable {
  call void @llvm.dbg.value(metadata !{i8 %data}, i64 0, metadata !141), !dbg !145 ; [debug line = 93:27] [debug variable = data]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !146 ; [debug line = 100:2]
  store volatile i1 false, i1* @uart_tx, align 1, !dbg !148 ; [debug line = 101:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !149 ; [debug line = 102:2]
  call fastcc void @wait_tmr(i28 868) nounwind, !dbg !150 ; [debug line = 103:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !151 ; [debug line = 104:2]
  br label %1, !dbg !152                          ; [debug line = 107:7]

; <label>:1                                       ; preds = %3, %0
  %i = phi i4 [ 0, %0 ], [ %i.1, %3 ]             ; [#uses=2 type=i4]
  %__Val2__ = phi i8 [ %data, %0 ], [ %data.assign, %3 ] ; [#uses=2 type=i8]
  %exitcond = icmp eq i4 %i, -8, !dbg !152        ; [#uses=1 type=i1] [debug line = 107:7]
  %2 = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 8, i64 8, i64 8) nounwind ; [#uses=0 type=i32]
  br i1 %exitcond, label %4, label %3, !dbg !152  ; [debug line = 107:7]

; <label>:3                                       ; preds = %1
  call void @llvm.dbg.value(metadata !{i8 %__Val2__}, i64 0, metadata !154), !dbg !157 ; [debug line = 108:41] [debug variable = __Val2__]
  %dt = call i1 @_ssdm_op_PartSelect.i1.i8.i32.i32(i8 %__Val2__, i32 0, i32 0), !dbg !158 ; [#uses=1 type=i1] [debug line = 108:72]
  call void @llvm.dbg.value(metadata !{i1 %dt}, i64 0, metadata !159), !dbg !160 ; [debug line = 108:161] [debug variable = dt]
  %data.assign = lshr i8 %__Val2__, 1, !dbg !161  ; [#uses=1 type=i8] [debug line = 109:3]
  call void @llvm.dbg.value(metadata !{i8 %data.assign}, i64 0, metadata !141), !dbg !161 ; [debug line = 109:3] [debug variable = data]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !162 ; [debug line = 110:3]
  store volatile i1 %dt, i1* @uart_tx, align 1, !dbg !163 ; [debug line = 111:3]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !164 ; [debug line = 112:3]
  call fastcc void @wait_tmr(i28 868) nounwind, !dbg !165 ; [debug line = 113:3]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !166 ; [debug line = 114:3]
  %i.1 = add i4 %i, 1, !dbg !167                  ; [#uses=1 type=i4] [debug line = 107:21]
  call void @llvm.dbg.value(metadata !{i4 %i.1}, i64 0, metadata !168), !dbg !167 ; [debug line = 107:21] [debug variable = i]
  br label %1, !dbg !167                          ; [debug line = 107:21]

; <label>:4                                       ; preds = %1
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !171 ; [debug line = 118:2]
  store volatile i1 true, i1* @uart_tx, align 1, !dbg !172 ; [debug line = 119:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !173 ; [debug line = 120:2]
  call fastcc void @wait_tmr(i28 868) nounwind, !dbg !174 ; [debug line = 121:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !175 ; [debug line = 122:2]
  ret void, !dbg !176                             ; [debug line = 123:1]
}

; [#uses=11]
define internal fastcc zeroext i8 @uart_receive_byte() nounwind uwtable {
  br label %._crit_edge, !dbg !177                ; [debug line = 134:2]

._crit_edge:                                      ; preds = %._crit_edge, %0
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !182 ; [debug line = 135:3]
  %tmp = call fastcc zeroext i1 @read_uart_rx(), !dbg !184 ; [#uses=1 type=i1] [debug line = 136:11]
  br i1 %tmp, label %.preheader.preheader, label %._crit_edge, !dbg !184 ; [debug line = 136:11]

.preheader.preheader:                             ; preds = %._crit_edge
  br label %.preheader, !dbg !185                 ; [debug line = 139:7]

.preheader:                                       ; preds = %._crit_edge1, %.preheader.preheader
  %timer = phi i24 [ %timer.1, %._crit_edge1 ], [ 0, %.preheader.preheader ] ; [#uses=2 type=i24]
  %tmp.1 = icmp ult i24 %timer, -6777216, !dbg !185 ; [#uses=1 type=i1] [debug line = 139:7]
  %1 = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 1, i64 10000000, i64 5000000) nounwind ; [#uses=0 type=i32]
  br i1 %tmp.1, label %2, label %.loopexit, !dbg !185 ; [debug line = 139:7]

; <label>:2                                       ; preds = %.preheader
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !187 ; [debug line = 140:3]
  %tmp.2 = call fastcc zeroext i1 @read_uart_rx(), !dbg !189 ; [#uses=1 type=i1] [debug line = 141:7]
  br i1 %tmp.2, label %._crit_edge1, label %3, !dbg !189 ; [debug line = 141:7]

; <label>:3                                       ; preds = %2
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !190 ; [debug line = 142:4]
  call fastcc void @wait_tmr(i28 217) nounwind, !dbg !192 ; [debug line = 143:4]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !193 ; [debug line = 144:4]
  %tmp.3 = call fastcc zeroext i1 @read_uart_rx(), !dbg !194 ; [#uses=1 type=i1] [debug line = 145:8]
  br i1 %tmp.3, label %._crit_edge1, label %.loopexit, !dbg !194 ; [debug line = 145:8]

._crit_edge1:                                     ; preds = %3, %2
  %timer.1 = add i24 %timer, 1, !dbg !195         ; [#uses=1 type=i24] [debug line = 139:51]
  call void @llvm.dbg.value(metadata !{i24 %timer.1}, i64 0, metadata !196), !dbg !195 ; [debug line = 139:51] [debug variable = timer]
  br label %.preheader, !dbg !195                 ; [debug line = 139:51]

.loopexit:                                        ; preds = %3, %.preheader
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !197 ; [debug line = 151:2]
  call fastcc void @wait_tmr(i28 651) nounwind, !dbg !198 ; [debug line = 152:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !199 ; [debug line = 153:2]
  br label %4, !dbg !200                          ; [debug line = 155:7]

; <label>:4                                       ; preds = %6, %.loopexit
  %data = phi i8 [ 0, %.loopexit ], [ %data.1, %6 ] ; [#uses=2 type=i8]
  %i = phi i4 [ 0, %.loopexit ], [ %i.2, %6 ]     ; [#uses=2 type=i4]
  %exitcond = icmp eq i4 %i, -8, !dbg !200        ; [#uses=1 type=i1] [debug line = 155:7]
  %5 = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 8, i64 8, i64 8) nounwind ; [#uses=0 type=i32]
  br i1 %exitcond, label %7, label %6, !dbg !200  ; [debug line = 155:7]

; <label>:6                                       ; preds = %4
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !202 ; [debug line = 156:3]
  call fastcc void @wait_tmr(i28 434) nounwind, !dbg !204 ; [debug line = 157:3]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !205 ; [debug line = 158:3]
  %tmp.5 = call fastcc zeroext i1 @read_uart_rx(), !dbg !206 ; [#uses=1 type=i1] [debug line = 159:7]
  %dt = select i1 %tmp.5, i8 -128, i8 0, !dbg !206 ; [#uses=1 type=i8] [debug line = 159:7]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !207 ; [debug line = 163:3]
  %tmp.6 = lshr i8 %data, 1, !dbg !208            ; [#uses=1 type=i8] [debug line = 164:3]
  %data.1 = or i8 %dt, %tmp.6, !dbg !208          ; [#uses=1 type=i8] [debug line = 164:3]
  call void @llvm.dbg.value(metadata !{i8 %data.1}, i64 0, metadata !209), !dbg !208 ; [debug line = 164:3] [debug variable = data]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !210 ; [debug line = 165:3]
  call fastcc void @wait_tmr(i28 434) nounwind, !dbg !211 ; [debug line = 166:3]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !212 ; [debug line = 167:3]
  %i.2 = add i4 %i, 1, !dbg !213                  ; [#uses=1 type=i4] [debug line = 155:21]
  call void @llvm.dbg.value(metadata !{i4 %i.2}, i64 0, metadata !214), !dbg !213 ; [debug line = 155:21] [debug variable = i]
  br label %4, !dbg !213                          ; [debug line = 155:21]

; <label>:7                                       ; preds = %4
  %data.lcssa = phi i8 [ %data, %4 ]              ; [#uses=1 type=i8]
  ret i8 %data.lcssa, !dbg !215                   ; [debug line = 170:2]
}

; [#uses=1]
define internal fastcc zeroext i16 @uart_read_reg16() nounwind uwtable {
  call fastcc void @uart_send_byte(i8 zeroext -86), !dbg !216 ; [debug line = 239:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !223 ; [debug line = 240:2]
  call fastcc void @uart_send_byte(i8 zeroext 1), !dbg !224 ; [debug line = 241:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !225 ; [debug line = 242:2]
  call fastcc void @uart_send_byte(i8 zeroext 26), !dbg !226 ; [debug line = 243:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !227 ; [debug line = 244:2]
  call fastcc void @uart_send_byte(i8 zeroext 2), !dbg !228 ; [debug line = 245:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !229 ; [debug line = 246:2]
  call fastcc void @wait_tmr(i28 1) nounwind, !dbg !230 ; [debug line = 248:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !231 ; [debug line = 250:2]
  %buf = call fastcc zeroext i8 @uart_receive_byte(), !dbg !232 ; [#uses=3 type=i8] [debug line = 251:8]
  call void @llvm.dbg.value(metadata !{i8 %buf}, i64 0, metadata !233), !dbg !232 ; [debug line = 251:8] [debug variable = buf]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !234 ; [debug line = 252:2]
  %tmp = icmp eq i8 %buf, -69, !dbg !235          ; [#uses=1 type=i1] [debug line = 253:2]
  br i1 %tmp, label %1, label %3, !dbg !235       ; [debug line = 253:2]

; <label>:1                                       ; preds = %0
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !236 ; [debug line = 254:3]
  %2 = call fastcc zeroext i8 @uart_receive_byte(), !dbg !238 ; [#uses=0 type=i8] [debug line = 255:9]
  call void @llvm.dbg.value(metadata !{i8 %2}, i64 0, metadata !233), !dbg !238 ; [debug line = 255:9] [debug variable = buf]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !239 ; [debug line = 256:3]
  %bh = call fastcc zeroext i8 @uart_receive_byte(), !dbg !240 ; [#uses=1 type=i8] [debug line = 257:8]
  call void @llvm.dbg.value(metadata !{i8 %bh}, i64 0, metadata !241), !dbg !240 ; [debug line = 257:8] [debug variable = bh]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !242 ; [debug line = 258:3]
  %bl = call fastcc zeroext i8 @uart_receive_byte(), !dbg !243 ; [#uses=1 type=i8] [debug line = 259:8]
  call void @llvm.dbg.value(metadata !{i8 %bl}, i64 0, metadata !244), !dbg !243 ; [debug line = 259:8] [debug variable = bl]
  %tmp.9 = call i16 @_ssdm_op_BitConcatenate.i16.i8.i8(i8 %bh, i8 %bl) ; [#uses=1 type=i16]
  br label %6, !dbg !245                          ; [debug line = 260:3]

; <label>:3                                       ; preds = %0
  %tmp. = icmp eq i8 %buf, -18, !dbg !246         ; [#uses=1 type=i1] [debug line = 262:7]
  br i1 %tmp., label %4, label %5, !dbg !246      ; [debug line = 262:7]

; <label>:4                                       ; preds = %3
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !247 ; [debug line = 263:3]
  %buf.1 = call fastcc zeroext i8 @uart_receive_byte(), !dbg !249 ; [#uses=1 type=i8] [debug line = 264:9]
  call void @llvm.dbg.value(metadata !{i8 %buf.1}, i64 0, metadata !233), !dbg !249 ; [debug line = 264:9] [debug variable = buf]
  %tmp.1 = zext i8 %buf.1 to i16, !dbg !250       ; [#uses=1 type=i16] [debug line = 265:3]
  br label %6, !dbg !250                          ; [debug line = 265:3]

; <label>:5                                       ; preds = %3
  %tmp.2 = zext i8 %buf to i16, !dbg !251         ; [#uses=1 type=i16] [debug line = 268:2]
  br label %6, !dbg !251                          ; [debug line = 268:2]

; <label>:6                                       ; preds = %5, %4, %1
  %.0 = phi i16 [ %tmp.9, %1 ], [ %tmp.1, %4 ], [ %tmp.2, %5 ] ; [#uses=1 type=i16]
  ret i16 %.0, !dbg !252                          ; [debug line = 269:1]
}

; [#uses=4]
define internal fastcc zeroext i8 @uart_read_reg(i8 zeroext %reg_addr) nounwind uwtable {
  call void (...)* @_ssdm_SpecKeepAssert(i8 %reg_addr) nounwind, !hlsrange !253
  call void @llvm.dbg.value(metadata !{i8 %reg_addr}, i64 0, metadata !254), !dbg !258 ; [debug line = 198:27] [debug variable = reg_addr]
  call fastcc void @uart_send_byte(i8 zeroext -86), !dbg !259 ; [debug line = 203:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !261 ; [debug line = 204:2]
  call fastcc void @uart_send_byte(i8 zeroext 1), !dbg !262 ; [debug line = 205:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !263 ; [debug line = 206:2]
  call fastcc void @uart_send_byte(i8 zeroext %reg_addr), !dbg !264 ; [debug line = 207:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !265 ; [debug line = 208:2]
  call fastcc void @uart_send_byte(i8 zeroext 1), !dbg !266 ; [debug line = 209:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !267 ; [debug line = 210:2]
  call fastcc void @wait_tmr(i28 1) nounwind, !dbg !268 ; [debug line = 212:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !269 ; [debug line = 214:2]
  %buf = call fastcc zeroext i8 @uart_receive_byte(), !dbg !270 ; [#uses=3 type=i8] [debug line = 215:8]
  call void @llvm.dbg.value(metadata !{i8 %buf}, i64 0, metadata !271), !dbg !270 ; [debug line = 215:8] [debug variable = buf]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !272 ; [debug line = 216:2]
  %tmp = icmp eq i8 %buf, -69, !dbg !273          ; [#uses=1 type=i1] [debug line = 217:2]
  br i1 %tmp, label %1, label %3, !dbg !273       ; [debug line = 217:2]

; <label>:1                                       ; preds = %0
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !274 ; [debug line = 218:3]
  %2 = call fastcc zeroext i8 @uart_receive_byte(), !dbg !276 ; [#uses=0 type=i8] [debug line = 219:9]
  call void @llvm.dbg.value(metadata !{i8 %2}, i64 0, metadata !271), !dbg !276 ; [debug line = 219:9] [debug variable = buf]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !277 ; [debug line = 220:3]
  %buf.2 = call fastcc zeroext i8 @uart_receive_byte(), !dbg !278 ; [#uses=1 type=i8] [debug line = 221:9]
  call void @llvm.dbg.value(metadata !{i8 %buf.2}, i64 0, metadata !271), !dbg !278 ; [debug line = 221:9] [debug variable = buf]
  br label %._crit_edge, !dbg !279                ; [debug line = 222:3]

; <label>:3                                       ; preds = %0
  %tmp. = icmp eq i8 %buf, -18, !dbg !280         ; [#uses=1 type=i1] [debug line = 224:7]
  br i1 %tmp., label %4, label %._crit_edge, !dbg !280 ; [debug line = 224:7]

; <label>:4                                       ; preds = %3
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !281 ; [debug line = 225:3]
  %buf.3 = call fastcc zeroext i8 @uart_receive_byte(), !dbg !283 ; [#uses=1 type=i8] [debug line = 226:9]
  call void @llvm.dbg.value(metadata !{i8 %buf.3}, i64 0, metadata !271), !dbg !283 ; [debug line = 226:9] [debug variable = buf]
  br label %._crit_edge, !dbg !284                ; [debug line = 227:3]

._crit_edge:                                      ; preds = %4, %3, %1
  %.0 = phi i8 [ %buf.2, %1 ], [ %buf.3, %4 ], [ %buf, %3 ] ; [#uses=1 type=i8]
  ret i8 %.0, !dbg !285                           ; [debug line = 231:1]
}

; [#uses=4]
define internal fastcc zeroext i1 @read_uart_rx() nounwind uwtable {
  %uart_rx.load = load volatile i1* @uart_rx, align 1, !dbg !286 ; [#uses=1 type=i1] [debug line = 89:2]
  ret i1 %uart_rx.load, !dbg !286                 ; [debug line = 89:2]
}

; [#uses=33]
declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

; [#uses=1]
declare i16 @llvm.bswap.i16(i16) nounwind readnone

; [#uses=0]
define void @bno055_uart() noreturn nounwind uwtable {
  call void (...)* @_ssdm_op_SpecTopModule(), !dbg !291 ; [debug line = 365:1]
  %dummy_tmr_out.load = load volatile i1* @dummy_tmr_out, align 1, !dbg !296 ; [#uses=0 type=i1] [debug line = 366:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @dummy_tmr_out, [8 x i8]* @.str6, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !296 ; [debug line = 366:1]
  %uart_rx.load = load volatile i1* @uart_rx, align 1, !dbg !297 ; [#uses=0 type=i1] [debug line = 368:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @uart_rx, [8 x i8]* @.str6, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !297 ; [debug line = 368:1]
  %uart_tx.load = load volatile i1* @uart_tx, align 1, !dbg !298 ; [#uses=0 type=i1] [debug line = 369:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @uart_tx, [8 x i8]* @.str6, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !298 ; [debug line = 369:1]
  %mem_addr.load = load volatile i8* @mem_addr, align 1, !dbg !299 ; [#uses=0 type=i8] [debug line = 373:1]
  call void (...)* @_ssdm_op_SpecInterface(i8* @mem_addr, [8 x i8]* @.str6, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !299 ; [debug line = 373:1]
  %mem_din.load = load volatile i8* @mem_din, align 1, !dbg !300 ; [#uses=0 type=i8] [debug line = 374:1]
  call void (...)* @_ssdm_op_SpecInterface(i8* @mem_din, [8 x i8]* @.str6, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !300 ; [debug line = 374:1]
  %mem_dout.load = load volatile i8* @mem_dout, align 1, !dbg !301 ; [#uses=0 type=i8] [debug line = 375:1]
  call void (...)* @_ssdm_op_SpecInterface(i8* @mem_dout, [8 x i8]* @.str6, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !301 ; [debug line = 375:1]
  %mem_wreq.load = load volatile i1* @mem_wreq, align 1, !dbg !302 ; [#uses=0 type=i1] [debug line = 376:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @mem_wreq, [8 x i8]* @.str6, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !302 ; [debug line = 376:1]
  %mem_wack.load = load volatile i1* @mem_wack, align 1, !dbg !303 ; [#uses=0 type=i1] [debug line = 377:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @mem_wack, [8 x i8]* @.str6, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !303 ; [debug line = 377:1]
  %mem_rreq.load = load volatile i1* @mem_rreq, align 1, !dbg !304 ; [#uses=0 type=i1] [debug line = 378:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @mem_rreq, [8 x i8]* @.str6, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !304 ; [debug line = 378:1]
  %mem_rack.load = load volatile i1* @mem_rack, align 1, !dbg !305 ; [#uses=0 type=i1] [debug line = 379:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @mem_rack, [8 x i8]* @.str6, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !305 ; [debug line = 379:1]
  call fastcc void @write_mem(i8 zeroext 21, i8 zeroext 0), !dbg !306 ; [debug line = 387:2]
  call fastcc void @write_mem(i8 zeroext 22, i8 zeroext 0), !dbg !307 ; [debug line = 388:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !308 ; [debug line = 390:2]
  store volatile i1 true, i1* @uart_tx, align 1, !dbg !309 ; [debug line = 391:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !310 ; [debug line = 392:2]
  call fastcc void @uart_write_reg(i7 zeroext 7, i8 zeroext 0) nounwind, !dbg !311 ; [debug line = 402:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !312 ; [debug line = 403:2]
  call fastcc void @uart_write_reg(i7 zeroext 63, i8 zeroext -64) nounwind, !dbg !313 ; [debug line = 404:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !314 ; [debug line = 405:2]
  call fastcc void @wait_tmr(i28 100000000) nounwind, !dbg !315 ; [debug line = 406:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !316 ; [debug line = 407:2]
  call fastcc void @uart_write_reg(i7 zeroext 7, i8 zeroext 0) nounwind, !dbg !317 ; [debug line = 409:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !318 ; [debug line = 410:2]
  %dt = call fastcc zeroext i8 @uart_read_reg(i8 zeroext 62), !dbg !319 ; [#uses=1 type=i8] [debug line = 412:7]
  call void @llvm.dbg.value(metadata !{i8 %dt}, i64 0, metadata !320), !dbg !319 ; [debug line = 412:7] [debug variable = dt]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !321 ; [debug line = 413:2]
  %tmp. = and i8 %dt, -4, !dbg !322               ; [#uses=1 type=i8] [debug line = 414:2]
  call fastcc void @uart_write_reg(i7 zeroext 62, i8 zeroext %tmp.) nounwind, !dbg !322 ; [debug line = 414:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !323 ; [debug line = 415:2]
  call fastcc void @wait_tmr(i28 10000000) nounwind, !dbg !324 ; [debug line = 416:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !325 ; [debug line = 417:2]
  %dt.1 = call fastcc zeroext i8 @uart_read_reg(i8 zeroext 61), !dbg !326 ; [#uses=1 type=i8] [debug line = 419:7]
  call void @llvm.dbg.value(metadata !{i8 %dt.1}, i64 0, metadata !320), !dbg !326 ; [debug line = 419:7] [debug variable = dt]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !327 ; [debug line = 420:2]
  %tmp.3 = and i8 %dt.1, -16, !dbg !328           ; [#uses=1 type=i8] [debug line = 421:2]
  %tmp.4 = or i8 %tmp.3, 12, !dbg !328            ; [#uses=1 type=i8] [debug line = 421:2]
  call fastcc void @uart_write_reg(i7 zeroext 61, i8 zeroext %tmp.4) nounwind, !dbg !328 ; [debug line = 421:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !329 ; [debug line = 422:2]
  call fastcc void @wait_tmr(i28 100000000) nounwind, !dbg !330 ; [debug line = 423:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !331 ; [debug line = 424:2]
  %dt.2 = call fastcc zeroext i8 @uart_read_reg(i8 zeroext 59), !dbg !332 ; [#uses=1 type=i8] [debug line = 426:7]
  call void @llvm.dbg.value(metadata !{i8 %dt.2}, i64 0, metadata !320), !dbg !332 ; [debug line = 426:7] [debug variable = dt]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !333 ; [debug line = 427:2]
  %tmp.5 = or i8 %dt.2, -128, !dbg !334           ; [#uses=1 type=i8] [debug line = 428:2]
  call fastcc void @uart_write_reg(i7 zeroext 59, i8 zeroext %tmp.5) nounwind, !dbg !334 ; [debug line = 428:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !335 ; [debug line = 429:2]
  call fastcc void @wait_tmr(i28 10000000) nounwind, !dbg !336 ; [debug line = 430:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !337 ; [debug line = 431:2]
  br label %1, !dbg !338                          ; [debug line = 434:2]

; <label>:1                                       ; preds = %1, %0
  %dt.3 = call fastcc zeroext i8 @uart_read_reg(i8 zeroext 0), !dbg !339 ; [#uses=2 type=i8] [debug line = 435:8]
  call void @llvm.dbg.value(metadata !{i8 %dt.3}, i64 0, metadata !320), !dbg !339 ; [debug line = 435:8] [debug variable = dt]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !341 ; [debug line = 436:3]
  %tmp.6 = lshr i8 %dt.3, 4, !dbg !342            ; [#uses=1 type=i8] [debug line = 437:16]
  %tmp.7 = trunc i8 %tmp.6 to i4, !dbg !342       ; [#uses=1 type=i4] [debug line = 437:16]
  %tmp.8 = call fastcc zeroext i7 @bin2char(i4 zeroext %tmp.7) nounwind, !dbg !342 ; [#uses=1 type=i7] [debug line = 437:16]
  %.trunc.ext = zext i7 %tmp.8 to i8, !dbg !342   ; [#uses=1 type=i8] [debug line = 437:16]
  call fastcc void @write_mem(i8 zeroext 0, i8 zeroext %.trunc.ext), !dbg !342 ; [debug line = 437:16]
  %tmp.9 = trunc i8 %dt.3 to i4, !dbg !343        ; [#uses=1 type=i4] [debug line = 438:16]
  %tmp.1 = call fastcc zeroext i7 @bin2char(i4 zeroext %tmp.9) nounwind, !dbg !343 ; [#uses=1 type=i7] [debug line = 438:16]
  %.trunc2.ext = zext i7 %tmp.1 to i8, !dbg !343  ; [#uses=1 type=i8] [debug line = 438:16]
  call fastcc void @write_mem(i8 zeroext 1, i8 zeroext %.trunc2.ext), !dbg !343 ; [debug line = 438:16]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !344 ; [debug line = 440:3]
  %e_tmp = call fastcc zeroext i16 @uart_read_reg16(), !dbg !345 ; [#uses=1 type=i16] [debug line = 441:11]
  call void @llvm.dbg.value(metadata !{i16 %e_tmp}, i64 0, metadata !346), !dbg !345 ; [debug line = 441:11] [debug variable = e_tmp]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !347 ; [debug line = 442:3]
  %e = call i16 @llvm.bswap.i16(i16 %e_tmp), !dbg !348 ; [#uses=1 type=i16] [debug line = 444:3]
  %e.cast = zext i16 %e to i20, !dbg !348         ; [#uses=1 type=i20] [debug line = 444:3]
  call void @llvm.dbg.value(metadata !{i16 %e}, i64 0, metadata !349), !dbg !348 ; [debug line = 444:3] [debug variable = e]
  %e.1 = mul i20 %e.cast, 100, !dbg !350          ; [#uses=2 type=i20] [debug line = 445:3]
  %e.1.cast = trunc i20 %e.1 to i12, !dbg !350    ; [#uses=1 type=i12] [debug line = 445:3]
  call void @llvm.dbg.value(metadata !{i20 %e.1}, i64 0, metadata !349), !dbg !350 ; [debug line = 445:3] [debug variable = e]
  %e.2 = lshr i12 %e.1.cast, 4, !dbg !351         ; [#uses=1 type=i12] [debug line = 446:3]
  call void @llvm.dbg.value(metadata !{i12 %e.2}, i64 0, metadata !349), !dbg !351 ; [debug line = 446:3] [debug variable = e]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !352 ; [debug line = 460:3]
  call fastcc void @write_mem(i8 zeroext -123, i8 zeroext 0), !dbg !353 ; [debug line = 461:3]
  %tmp.2 = lshr i20 %e.1, 12, !dbg !354           ; [#uses=1 type=i20] [debug line = 462:3]
  %tmp.10 = trunc i20 %tmp.2 to i8, !dbg !354     ; [#uses=1 type=i8] [debug line = 462:3]
  call fastcc void @write_mem(i8 zeroext -125, i8 zeroext %tmp.10), !dbg !354 ; [debug line = 462:3]
  %tmp.11 = trunc i12 %e.2 to i8, !dbg !355       ; [#uses=1 type=i8] [debug line = 463:3]
  call fastcc void @write_mem(i8 zeroext -124, i8 zeroext %tmp.11), !dbg !355 ; [debug line = 463:3]
  call fastcc void @write_mem(i8 zeroext -123, i8 zeroext 1), !dbg !356 ; [debug line = 464:3]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !357 ; [debug line = 465:3]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !358 ; [debug line = 468:3]
  call fastcc void @wait_tmr(i28 1000000) nounwind, !dbg !359 ; [debug line = 469:3]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !360 ; [debug line = 470:3]
  br label %1, !dbg !361                          ; [debug line = 471:2]
}

; [#uses=2]
define internal fastcc zeroext i7 @bin2char(i4 zeroext %val) readnone {
  call void @llvm.dbg.value(metadata !{i4 %val}, i64 0, metadata !362), !dbg !366 ; [debug line = 332:22] [debug variable = val]
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
  ], !dbg !367                                    ; [debug line = 337:2]

; <label>:1                                       ; preds = %0
  br label %._crit_edge, !dbg !369                ; [debug line = 339:24]

; <label>:2                                       ; preds = %0
  br label %._crit_edge, !dbg !371                ; [debug line = 340:24]

; <label>:3                                       ; preds = %0
  br label %._crit_edge, !dbg !372                ; [debug line = 341:24]

; <label>:4                                       ; preds = %0
  br label %._crit_edge, !dbg !373                ; [debug line = 342:24]

; <label>:5                                       ; preds = %0
  br label %._crit_edge, !dbg !374                ; [debug line = 343:24]

; <label>:6                                       ; preds = %0
  br label %._crit_edge, !dbg !375                ; [debug line = 344:24]

; <label>:7                                       ; preds = %0
  br label %._crit_edge, !dbg !376                ; [debug line = 345:24]

; <label>:8                                       ; preds = %0
  br label %._crit_edge, !dbg !377                ; [debug line = 346:24]

; <label>:9                                       ; preds = %0
  br label %._crit_edge, !dbg !378                ; [debug line = 347:24]

; <label>:10                                      ; preds = %0
  br label %._crit_edge, !dbg !379                ; [debug line = 348:25]

; <label>:11                                      ; preds = %0
  br label %._crit_edge, !dbg !380                ; [debug line = 349:25]

; <label>:12                                      ; preds = %0
  br label %._crit_edge, !dbg !381                ; [debug line = 350:25]

; <label>:13                                      ; preds = %0
  br label %._crit_edge, !dbg !382                ; [debug line = 351:25]

; <label>:14                                      ; preds = %0
  br label %._crit_edge, !dbg !383                ; [debug line = 352:25]

; <label>:15                                      ; preds = %0
  br label %._crit_edge, !dbg !384                ; [debug line = 354:2]

._crit_edge:                                      ; preds = %15, %14, %13, %12, %11, %10, %9, %8, %7, %6, %5, %4, %3, %2, %1, %0
  %retval = phi i7 [ -58, %15 ], [ -59, %14 ], [ -60, %13 ], [ -61, %12 ], [ -62, %11 ], [ -63, %10 ], [ 57, %9 ], [ 56, %8 ], [ 55, %7 ], [ 54, %6 ], [ 53, %5 ], [ 52, %4 ], [ 51, %3 ], [ 50, %2 ], [ 49, %1 ], [ 48, %0 ] ; [#uses=1 type=i7]
  ret i7 %retval, !dbg !385                       ; [debug line = 356:2]
}

; [#uses=75]
declare void @_ssdm_op_Wait(...) nounwind

; [#uses=1]
declare void @_ssdm_op_SpecTopModule(...) nounwind

; [#uses=4]
declare i32 @_ssdm_op_SpecLoopTripCount(...)

; [#uses=10]
declare void @_ssdm_op_SpecInterface(...) nounwind

; [#uses=1]
declare i1 @_ssdm_op_PartSelect.i1.i8.i32.i32(i8, i32, i32) nounwind readnone

; [#uses=1]
declare i16 @_ssdm_op_BitConcatenate.i16.i8.i8(i8, i8) nounwind readnone

; [#uses=3]
declare void @_ssdm_SpecKeepAssert(...)

!hls.encrypted.func = !{}
!llvm.map.gv = !{!0, !7, !12, !17, !22, !27, !32, !37, !42, !47}
!llvm.dbg.cu = !{!52}

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
!52 = metadata !{i32 786449, i32 0, i32 1, metadata !"D:/21_streamer_car5_artix7/fpga_arty/bno055_uart/solution1/.autopilot/db/bno055_uart.pragma.2.c", metadata !"D:\5C21_streamer_car5_artix7\5Cfpga_arty", metadata !"clang version 3.1 ", i1 true, i1 false, metadata !"", i32 0, null, null, null, metadata !53} ; [ DW_TAG_compile_unit ]
!53 = metadata !{metadata !54}
!54 = metadata !{metadata !55, metadata !60, metadata !61, metadata !62, metadata !63, metadata !67, metadata !68, metadata !69, metadata !70, metadata !71}
!55 = metadata !{i32 786484, i32 0, null, metadata !"mem_rreq", metadata !"mem_rreq", metadata !"", metadata !56, i32 56, metadata !57, i32 0, i32 1, i1* @mem_rreq} ; [ DW_TAG_variable ]
!56 = metadata !{i32 786473, metadata !"bno055_uart.c", metadata !"D:\5C21_streamer_car5_artix7\5Cfpga_arty", null} ; [ DW_TAG_file_type ]
!57 = metadata !{i32 786485, null, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !58} ; [ DW_TAG_volatile_type ]
!58 = metadata !{i32 786454, null, metadata !"uint1", metadata !56, i32 3, i64 0, i64 0, i64 0, i32 0, metadata !59} ; [ DW_TAG_typedef ]
!59 = metadata !{i32 786468, null, metadata !"uint1", null, i32 0, i64 1, i64 1, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!60 = metadata !{i32 786484, i32 0, null, metadata !"mem_rack", metadata !"mem_rack", metadata !"", metadata !56, i32 57, metadata !57, i32 0, i32 1, i1* @mem_rack} ; [ DW_TAG_variable ]
!61 = metadata !{i32 786484, i32 0, null, metadata !"uart_rx", metadata !"uart_rx", metadata !"", metadata !56, i32 47, metadata !57, i32 0, i32 1, i1* @uart_rx} ; [ DW_TAG_variable ]
!62 = metadata !{i32 786484, i32 0, null, metadata !"mem_wack", metadata !"mem_wack", metadata !"", metadata !56, i32 55, metadata !57, i32 0, i32 1, i1* @mem_wack} ; [ DW_TAG_variable ]
!63 = metadata !{i32 786484, i32 0, null, metadata !"mem_dout", metadata !"mem_dout", metadata !"", metadata !56, i32 53, metadata !64, i32 0, i32 1, i8* @mem_dout} ; [ DW_TAG_variable ]
!64 = metadata !{i32 786485, null, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !65} ; [ DW_TAG_volatile_type ]
!65 = metadata !{i32 786454, null, metadata !"uint8", metadata !56, i32 10, i64 0, i64 0, i64 0, i32 0, metadata !66} ; [ DW_TAG_typedef ]
!66 = metadata !{i32 786468, null, metadata !"uint8", null, i32 0, i64 8, i64 8, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!67 = metadata !{i32 786484, i32 0, null, metadata !"mem_din", metadata !"mem_din", metadata !"", metadata !56, i32 52, metadata !64, i32 0, i32 1, i8* @mem_din} ; [ DW_TAG_variable ]
!68 = metadata !{i32 786484, i32 0, null, metadata !"mem_addr", metadata !"mem_addr", metadata !"", metadata !56, i32 51, metadata !64, i32 0, i32 1, i8* @mem_addr} ; [ DW_TAG_variable ]
!69 = metadata !{i32 786484, i32 0, null, metadata !"mem_wreq", metadata !"mem_wreq", metadata !"", metadata !56, i32 54, metadata !57, i32 0, i32 1, i1* @mem_wreq} ; [ DW_TAG_variable ]
!70 = metadata !{i32 786484, i32 0, null, metadata !"dummy_tmr_out", metadata !"dummy_tmr_out", metadata !"", metadata !56, i32 62, metadata !57, i32 0, i32 1, i1* @dummy_tmr_out} ; [ DW_TAG_variable ]
!71 = metadata !{i32 786484, i32 0, null, metadata !"uart_tx", metadata !"uart_tx", metadata !"", metadata !56, i32 48, metadata !57, i32 0, i32 1, i1* @uart_tx} ; [ DW_TAG_variable ]
!72 = metadata !{i8 -125, i8 22, i8 0, i8 -1}     
!73 = metadata !{i32 786689, metadata !74, metadata !"addr", metadata !56, i32 16777488, metadata !65, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!74 = metadata !{i32 786478, i32 0, metadata !56, metadata !"write_mem", metadata !"write_mem", metadata !"", metadata !56, i32 272, metadata !75, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (i8, i8)* @write_mem, null, null, metadata !77, i32 273} ; [ DW_TAG_subprogram ]
!75 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !76, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!76 = metadata !{null, metadata !65, metadata !65}
!77 = metadata !{metadata !78}
!78 = metadata !{i32 786468}                      ; [ DW_TAG_base_type ]
!79 = metadata !{i32 272, i32 22, metadata !74, null}
!80 = metadata !{i32 786689, metadata !74, metadata !"data", metadata !56, i32 33554704, metadata !65, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!81 = metadata !{i32 272, i32 34, metadata !74, null}
!82 = metadata !{i32 275, i32 2, metadata !83, null}
!83 = metadata !{i32 786443, metadata !74, i32 273, i32 1, metadata !56, i32 23} ; [ DW_TAG_lexical_block ]
!84 = metadata !{i32 276, i32 2, metadata !83, null}
!85 = metadata !{i32 277, i32 2, metadata !83, null}
!86 = metadata !{i32 278, i32 2, metadata !83, null}
!87 = metadata !{i32 279, i32 2, metadata !83, null}
!88 = metadata !{i32 281, i32 2, metadata !83, null}
!89 = metadata !{i32 282, i32 3, metadata !90, null}
!90 = metadata !{i32 786443, metadata !83, i32 281, i32 5, metadata !56, i32 24} ; [ DW_TAG_lexical_block ]
!91 = metadata !{i32 283, i32 3, metadata !90, null}
!92 = metadata !{i32 284, i32 3, metadata !90, null}
!93 = metadata !{i32 285, i32 2, metadata !90, null}
!94 = metadata !{i32 286, i32 2, metadata !83, null}
!95 = metadata !{i32 288, i32 2, metadata !83, null}
!96 = metadata !{i32 289, i32 2, metadata !83, null}
!97 = metadata !{i32 290, i32 2, metadata !83, null}
!98 = metadata !{i32 291, i32 2, metadata !83, null}
!99 = metadata !{i32 292, i32 1, metadata !83, null}
!100 = metadata !{i32 786689, metadata !101, metadata !"tmr", metadata !56, i32 16777290, metadata !104, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!101 = metadata !{i32 786478, i32 0, metadata !56, metadata !"wait_tmr", metadata !"wait_tmr", metadata !"", metadata !56, i32 74, metadata !102, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, metadata !77, i32 75} ; [ DW_TAG_subprogram ]
!102 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !103, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!103 = metadata !{null, metadata !104}
!104 = metadata !{i32 786454, null, metadata !"uint32", metadata !56, i32 34, i64 0, i64 0, i64 0, i32 0, metadata !105} ; [ DW_TAG_typedef ]
!105 = metadata !{i32 786468, null, metadata !"uint32", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!106 = metadata !{i32 74, i32 22, metadata !101, null}
!107 = metadata !{i32 78, i32 2, metadata !108, null}
!108 = metadata !{i32 786443, metadata !101, i32 75, i32 1, metadata !56, i32 0} ; [ DW_TAG_lexical_block ]
!109 = metadata !{i32 79, i32 7, metadata !110, null}
!110 = metadata !{i32 786443, metadata !108, i32 79, i32 2, metadata !56, i32 1} ; [ DW_TAG_lexical_block ]
!111 = metadata !{i32 80, i32 3, metadata !112, null}
!112 = metadata !{i32 786443, metadata !110, i32 79, i32 28, metadata !56, i32 2} ; [ DW_TAG_lexical_block ]
!113 = metadata !{i32 79, i32 23, metadata !110, null}
!114 = metadata !{i32 786688, metadata !108, metadata !"t", metadata !56, i32 77, metadata !104, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!115 = metadata !{i32 82, i32 2, metadata !108, null}
!116 = metadata !{i32 83, i32 1, metadata !108, null}
!117 = metadata !{i7 7, i7 63, i7 7, i7 63}       
!118 = metadata !{i32 786689, metadata !119, metadata !"reg_addr", metadata !56, i32 16777390, metadata !65, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!119 = metadata !{i32 786478, i32 0, metadata !56, metadata !"uart_write_reg", metadata !"uart_write_reg", metadata !"", metadata !56, i32 174, metadata !75, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, metadata !77, i32 175} ; [ DW_TAG_subprogram ]
!120 = metadata !{i32 174, i32 27, metadata !119, null}
!121 = metadata !{i32 786689, metadata !119, metadata !"data", metadata !56, i32 33554606, metadata !65, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!122 = metadata !{i32 174, i32 43, metadata !119, null}
!123 = metadata !{i32 177, i32 2, metadata !124, null}
!124 = metadata !{i32 786443, metadata !119, i32 175, i32 1, metadata !56, i32 16} ; [ DW_TAG_lexical_block ]
!125 = metadata !{i32 178, i32 2, metadata !124, null}
!126 = metadata !{i32 179, i32 2, metadata !124, null}
!127 = metadata !{i32 180, i32 2, metadata !124, null}
!128 = metadata !{i32 181, i32 2, metadata !124, null}
!129 = metadata !{i32 182, i32 2, metadata !124, null}
!130 = metadata !{i32 183, i32 2, metadata !124, null}
!131 = metadata !{i32 184, i32 2, metadata !124, null}
!132 = metadata !{i32 185, i32 2, metadata !124, null}
!133 = metadata !{i32 186, i32 2, metadata !124, null}
!134 = metadata !{i32 188, i32 2, metadata !124, null}
!135 = metadata !{i32 190, i32 2, metadata !124, null}
!136 = metadata !{i32 191, i32 2, metadata !124, null}
!137 = metadata !{i32 192, i32 2, metadata !124, null}
!138 = metadata !{i32 193, i32 2, metadata !124, null}
!139 = metadata !{i32 194, i32 2, metadata !124, null}
!140 = metadata !{i32 195, i32 1, metadata !124, null}
!141 = metadata !{i32 786689, metadata !142, metadata !"data", metadata !56, i32 16777309, metadata !65, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!142 = metadata !{i32 786478, i32 0, metadata !56, metadata !"uart_send_byte", metadata !"uart_send_byte", metadata !"", metadata !56, i32 93, metadata !143, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (i8)* @uart_send_byte, null, null, metadata !77, i32 94} ; [ DW_TAG_subprogram ]
!143 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !144, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!144 = metadata !{null, metadata !65}
!145 = metadata !{i32 93, i32 27, metadata !142, null}
!146 = metadata !{i32 100, i32 2, metadata !147, null}
!147 = metadata !{i32 786443, metadata !142, i32 94, i32 1, metadata !56, i32 4} ; [ DW_TAG_lexical_block ]
!148 = metadata !{i32 101, i32 2, metadata !147, null}
!149 = metadata !{i32 102, i32 2, metadata !147, null}
!150 = metadata !{i32 103, i32 2, metadata !147, null}
!151 = metadata !{i32 104, i32 2, metadata !147, null}
!152 = metadata !{i32 107, i32 7, metadata !153, null}
!153 = metadata !{i32 786443, metadata !147, i32 107, i32 2, metadata !56, i32 5} ; [ DW_TAG_lexical_block ]
!154 = metadata !{i32 786688, metadata !155, metadata !"__Val2__", metadata !56, i32 108, metadata !65, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!155 = metadata !{i32 786443, metadata !156, i32 108, i32 9, metadata !56, i32 7} ; [ DW_TAG_lexical_block ]
!156 = metadata !{i32 786443, metadata !153, i32 107, i32 26, metadata !56, i32 6} ; [ DW_TAG_lexical_block ]
!157 = metadata !{i32 108, i32 41, metadata !155, null}
!158 = metadata !{i32 108, i32 72, metadata !155, null}
!159 = metadata !{i32 786688, metadata !147, metadata !"dt", metadata !56, i32 97, metadata !58, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!160 = metadata !{i32 108, i32 161, metadata !155, null}
!161 = metadata !{i32 109, i32 3, metadata !156, null}
!162 = metadata !{i32 110, i32 3, metadata !156, null}
!163 = metadata !{i32 111, i32 3, metadata !156, null}
!164 = metadata !{i32 112, i32 3, metadata !156, null}
!165 = metadata !{i32 113, i32 3, metadata !156, null}
!166 = metadata !{i32 114, i32 3, metadata !156, null}
!167 = metadata !{i32 107, i32 21, metadata !153, null}
!168 = metadata !{i32 786688, metadata !147, metadata !"i", metadata !56, i32 96, metadata !169, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!169 = metadata !{i32 786454, null, metadata !"uint4", metadata !56, i32 6, i64 0, i64 0, i64 0, i32 0, metadata !170} ; [ DW_TAG_typedef ]
!170 = metadata !{i32 786468, null, metadata !"uint4", null, i32 0, i64 4, i64 4, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!171 = metadata !{i32 118, i32 2, metadata !147, null}
!172 = metadata !{i32 119, i32 2, metadata !147, null}
!173 = metadata !{i32 120, i32 2, metadata !147, null}
!174 = metadata !{i32 121, i32 2, metadata !147, null}
!175 = metadata !{i32 122, i32 2, metadata !147, null}
!176 = metadata !{i32 123, i32 1, metadata !147, null}
!177 = metadata !{i32 134, i32 2, metadata !178, null}
!178 = metadata !{i32 786443, metadata !179, i32 127, i32 1, metadata !56, i32 8} ; [ DW_TAG_lexical_block ]
!179 = metadata !{i32 786478, i32 0, metadata !56, metadata !"uart_receive_byte", metadata !"uart_receive_byte", metadata !"", metadata !56, i32 126, metadata !180, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i8 ()* @uart_receive_byte, null, null, metadata !77, i32 127} ; [ DW_TAG_subprogram ]
!180 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !181, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!181 = metadata !{metadata !65}
!182 = metadata !{i32 135, i32 3, metadata !183, null}
!183 = metadata !{i32 786443, metadata !178, i32 134, i32 5, metadata !56, i32 9} ; [ DW_TAG_lexical_block ]
!184 = metadata !{i32 136, i32 11, metadata !178, null}
!185 = metadata !{i32 139, i32 7, metadata !186, null}
!186 = metadata !{i32 786443, metadata !178, i32 139, i32 2, metadata !56, i32 10} ; [ DW_TAG_lexical_block ]
!187 = metadata !{i32 140, i32 3, metadata !188, null}
!188 = metadata !{i32 786443, metadata !186, i32 139, i32 60, metadata !56, i32 11} ; [ DW_TAG_lexical_block ]
!189 = metadata !{i32 141, i32 7, metadata !188, null}
!190 = metadata !{i32 142, i32 4, metadata !191, null}
!191 = metadata !{i32 786443, metadata !188, i32 141, i32 28, metadata !56, i32 12} ; [ DW_TAG_lexical_block ]
!192 = metadata !{i32 143, i32 4, metadata !191, null}
!193 = metadata !{i32 144, i32 4, metadata !191, null}
!194 = metadata !{i32 145, i32 8, metadata !191, null}
!195 = metadata !{i32 139, i32 51, metadata !186, null}
!196 = metadata !{i32 786688, metadata !178, metadata !"timer", metadata !56, i32 132, metadata !104, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!197 = metadata !{i32 151, i32 2, metadata !178, null}
!198 = metadata !{i32 152, i32 2, metadata !178, null}
!199 = metadata !{i32 153, i32 2, metadata !178, null}
!200 = metadata !{i32 155, i32 7, metadata !201, null}
!201 = metadata !{i32 786443, metadata !178, i32 155, i32 2, metadata !56, i32 14} ; [ DW_TAG_lexical_block ]
!202 = metadata !{i32 156, i32 3, metadata !203, null}
!203 = metadata !{i32 786443, metadata !201, i32 155, i32 26, metadata !56, i32 15} ; [ DW_TAG_lexical_block ]
!204 = metadata !{i32 157, i32 3, metadata !203, null}
!205 = metadata !{i32 158, i32 3, metadata !203, null}
!206 = metadata !{i32 159, i32 7, metadata !203, null}
!207 = metadata !{i32 163, i32 3, metadata !203, null}
!208 = metadata !{i32 164, i32 3, metadata !203, null}
!209 = metadata !{i32 786688, metadata !178, metadata !"data", metadata !56, i32 130, metadata !65, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!210 = metadata !{i32 165, i32 3, metadata !203, null}
!211 = metadata !{i32 166, i32 3, metadata !203, null}
!212 = metadata !{i32 167, i32 3, metadata !203, null}
!213 = metadata !{i32 155, i32 21, metadata !201, null}
!214 = metadata !{i32 786688, metadata !178, metadata !"i", metadata !56, i32 129, metadata !169, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!215 = metadata !{i32 170, i32 2, metadata !178, null}
!216 = metadata !{i32 239, i32 2, metadata !217, null}
!217 = metadata !{i32 786443, metadata !218, i32 235, i32 1, metadata !56, i32 20} ; [ DW_TAG_lexical_block ]
!218 = metadata !{i32 786478, i32 0, metadata !56, metadata !"uart_read_reg16", metadata !"uart_read_reg16", metadata !"", metadata !56, i32 234, metadata !219, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, metadata !77, i32 235} ; [ DW_TAG_subprogram ]
!219 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !220, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!220 = metadata !{metadata !221, metadata !65}
!221 = metadata !{i32 786454, null, metadata !"uint16", metadata !56, i32 18, i64 0, i64 0, i64 0, i32 0, metadata !222} ; [ DW_TAG_typedef ]
!222 = metadata !{i32 786468, null, metadata !"uint16", null, i32 0, i64 16, i64 16, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!223 = metadata !{i32 240, i32 2, metadata !217, null}
!224 = metadata !{i32 241, i32 2, metadata !217, null}
!225 = metadata !{i32 242, i32 2, metadata !217, null}
!226 = metadata !{i32 243, i32 2, metadata !217, null}
!227 = metadata !{i32 244, i32 2, metadata !217, null}
!228 = metadata !{i32 245, i32 2, metadata !217, null}
!229 = metadata !{i32 246, i32 2, metadata !217, null}
!230 = metadata !{i32 248, i32 2, metadata !217, null}
!231 = metadata !{i32 250, i32 2, metadata !217, null}
!232 = metadata !{i32 251, i32 8, metadata !217, null}
!233 = metadata !{i32 786688, metadata !217, metadata !"buf", metadata !56, i32 237, metadata !65, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!234 = metadata !{i32 252, i32 2, metadata !217, null}
!235 = metadata !{i32 253, i32 2, metadata !217, null}
!236 = metadata !{i32 254, i32 3, metadata !237, null}
!237 = metadata !{i32 786443, metadata !217, i32 253, i32 19, metadata !56, i32 21} ; [ DW_TAG_lexical_block ]
!238 = metadata !{i32 255, i32 9, metadata !237, null}
!239 = metadata !{i32 256, i32 3, metadata !237, null}
!240 = metadata !{i32 257, i32 8, metadata !237, null}
!241 = metadata !{i32 786688, metadata !217, metadata !"bh", metadata !56, i32 237, metadata !65, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!242 = metadata !{i32 258, i32 3, metadata !237, null}
!243 = metadata !{i32 259, i32 8, metadata !237, null}
!244 = metadata !{i32 786688, metadata !217, metadata !"bl", metadata !56, i32 237, metadata !65, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!245 = metadata !{i32 260, i32 3, metadata !237, null}
!246 = metadata !{i32 262, i32 7, metadata !217, null}
!247 = metadata !{i32 263, i32 3, metadata !248, null}
!248 = metadata !{i32 786443, metadata !217, i32 262, i32 24, metadata !56, i32 22} ; [ DW_TAG_lexical_block ]
!249 = metadata !{i32 264, i32 9, metadata !248, null}
!250 = metadata !{i32 265, i32 3, metadata !248, null}
!251 = metadata !{i32 268, i32 2, metadata !217, null}
!252 = metadata !{i32 269, i32 1, metadata !217, null}
!253 = metadata !{i8 0, i8 62, i8 0, i8 62}       
!254 = metadata !{i32 786689, metadata !255, metadata !"reg_addr", metadata !56, i32 16777414, metadata !65, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!255 = metadata !{i32 786478, i32 0, metadata !56, metadata !"uart_read_reg", metadata !"uart_read_reg", metadata !"", metadata !56, i32 198, metadata !256, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i8 (i8)* @uart_read_reg, null, null, metadata !77, i32 199} ; [ DW_TAG_subprogram ]
!256 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !257, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!257 = metadata !{metadata !65, metadata !65}
!258 = metadata !{i32 198, i32 27, metadata !255, null}
!259 = metadata !{i32 203, i32 2, metadata !260, null}
!260 = metadata !{i32 786443, metadata !255, i32 199, i32 1, metadata !56, i32 17} ; [ DW_TAG_lexical_block ]
!261 = metadata !{i32 204, i32 2, metadata !260, null}
!262 = metadata !{i32 205, i32 2, metadata !260, null}
!263 = metadata !{i32 206, i32 2, metadata !260, null}
!264 = metadata !{i32 207, i32 2, metadata !260, null}
!265 = metadata !{i32 208, i32 2, metadata !260, null}
!266 = metadata !{i32 209, i32 2, metadata !260, null}
!267 = metadata !{i32 210, i32 2, metadata !260, null}
!268 = metadata !{i32 212, i32 2, metadata !260, null}
!269 = metadata !{i32 214, i32 2, metadata !260, null}
!270 = metadata !{i32 215, i32 8, metadata !260, null}
!271 = metadata !{i32 786688, metadata !260, metadata !"buf", metadata !56, i32 201, metadata !65, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!272 = metadata !{i32 216, i32 2, metadata !260, null}
!273 = metadata !{i32 217, i32 2, metadata !260, null}
!274 = metadata !{i32 218, i32 3, metadata !275, null}
!275 = metadata !{i32 786443, metadata !260, i32 217, i32 19, metadata !56, i32 18} ; [ DW_TAG_lexical_block ]
!276 = metadata !{i32 219, i32 9, metadata !275, null}
!277 = metadata !{i32 220, i32 3, metadata !275, null}
!278 = metadata !{i32 221, i32 9, metadata !275, null}
!279 = metadata !{i32 222, i32 3, metadata !275, null}
!280 = metadata !{i32 224, i32 7, metadata !260, null}
!281 = metadata !{i32 225, i32 3, metadata !282, null}
!282 = metadata !{i32 786443, metadata !260, i32 224, i32 24, metadata !56, i32 19} ; [ DW_TAG_lexical_block ]
!283 = metadata !{i32 226, i32 9, metadata !282, null}
!284 = metadata !{i32 227, i32 3, metadata !282, null}
!285 = metadata !{i32 231, i32 1, metadata !260, null}
!286 = metadata !{i32 89, i32 2, metadata !287, null}
!287 = metadata !{i32 786443, metadata !288, i32 87, i32 1, metadata !56, i32 3} ; [ DW_TAG_lexical_block ]
!288 = metadata !{i32 786478, i32 0, metadata !56, metadata !"read_uart_rx", metadata !"read_uart_rx", metadata !"", metadata !56, i32 86, metadata !289, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i1 ()* @read_uart_rx, null, null, metadata !77, i32 87} ; [ DW_TAG_subprogram ]
!289 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !290, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!290 = metadata !{metadata !58}
!291 = metadata !{i32 365, i32 1, metadata !292, null}
!292 = metadata !{i32 786443, metadata !293, i32 364, i32 1, metadata !56, i32 30} ; [ DW_TAG_lexical_block ]
!293 = metadata !{i32 786478, i32 0, metadata !56, metadata !"bno055_uart", metadata !"bno055_uart", metadata !"", metadata !56, i32 363, metadata !294, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @bno055_uart, null, null, metadata !77, i32 364} ; [ DW_TAG_subprogram ]
!294 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !295, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!295 = metadata !{null}
!296 = metadata !{i32 366, i32 1, metadata !292, null}
!297 = metadata !{i32 368, i32 1, metadata !292, null}
!298 = metadata !{i32 369, i32 1, metadata !292, null}
!299 = metadata !{i32 373, i32 1, metadata !292, null}
!300 = metadata !{i32 374, i32 1, metadata !292, null}
!301 = metadata !{i32 375, i32 1, metadata !292, null}
!302 = metadata !{i32 376, i32 1, metadata !292, null}
!303 = metadata !{i32 377, i32 1, metadata !292, null}
!304 = metadata !{i32 378, i32 1, metadata !292, null}
!305 = metadata !{i32 379, i32 1, metadata !292, null}
!306 = metadata !{i32 387, i32 2, metadata !292, null}
!307 = metadata !{i32 388, i32 2, metadata !292, null}
!308 = metadata !{i32 390, i32 2, metadata !292, null}
!309 = metadata !{i32 391, i32 2, metadata !292, null}
!310 = metadata !{i32 392, i32 2, metadata !292, null}
!311 = metadata !{i32 402, i32 2, metadata !292, null}
!312 = metadata !{i32 403, i32 2, metadata !292, null}
!313 = metadata !{i32 404, i32 2, metadata !292, null}
!314 = metadata !{i32 405, i32 2, metadata !292, null}
!315 = metadata !{i32 406, i32 2, metadata !292, null}
!316 = metadata !{i32 407, i32 2, metadata !292, null}
!317 = metadata !{i32 409, i32 2, metadata !292, null}
!318 = metadata !{i32 410, i32 2, metadata !292, null}
!319 = metadata !{i32 412, i32 7, metadata !292, null}
!320 = metadata !{i32 786688, metadata !292, metadata !"dt", metadata !56, i32 381, metadata !65, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!321 = metadata !{i32 413, i32 2, metadata !292, null}
!322 = metadata !{i32 414, i32 2, metadata !292, null}
!323 = metadata !{i32 415, i32 2, metadata !292, null}
!324 = metadata !{i32 416, i32 2, metadata !292, null}
!325 = metadata !{i32 417, i32 2, metadata !292, null}
!326 = metadata !{i32 419, i32 7, metadata !292, null}
!327 = metadata !{i32 420, i32 2, metadata !292, null}
!328 = metadata !{i32 421, i32 2, metadata !292, null}
!329 = metadata !{i32 422, i32 2, metadata !292, null}
!330 = metadata !{i32 423, i32 2, metadata !292, null}
!331 = metadata !{i32 424, i32 2, metadata !292, null}
!332 = metadata !{i32 426, i32 7, metadata !292, null}
!333 = metadata !{i32 427, i32 2, metadata !292, null}
!334 = metadata !{i32 428, i32 2, metadata !292, null}
!335 = metadata !{i32 429, i32 2, metadata !292, null}
!336 = metadata !{i32 430, i32 2, metadata !292, null}
!337 = metadata !{i32 431, i32 2, metadata !292, null}
!338 = metadata !{i32 434, i32 2, metadata !292, null}
!339 = metadata !{i32 435, i32 8, metadata !340, null}
!340 = metadata !{i32 786443, metadata !292, i32 434, i32 12, metadata !56, i32 31} ; [ DW_TAG_lexical_block ]
!341 = metadata !{i32 436, i32 3, metadata !340, null}
!342 = metadata !{i32 437, i32 16, metadata !340, null}
!343 = metadata !{i32 438, i32 16, metadata !340, null}
!344 = metadata !{i32 440, i32 3, metadata !340, null}
!345 = metadata !{i32 441, i32 11, metadata !340, null}
!346 = metadata !{i32 786688, metadata !292, metadata !"e_tmp", metadata !56, i32 383, metadata !221, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!347 = metadata !{i32 442, i32 3, metadata !340, null}
!348 = metadata !{i32 444, i32 3, metadata !340, null}
!349 = metadata !{i32 786688, metadata !292, metadata !"e", metadata !56, i32 384, metadata !104, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!350 = metadata !{i32 445, i32 3, metadata !340, null}
!351 = metadata !{i32 446, i32 3, metadata !340, null}
!352 = metadata !{i32 460, i32 3, metadata !340, null}
!353 = metadata !{i32 461, i32 3, metadata !340, null}
!354 = metadata !{i32 462, i32 3, metadata !340, null}
!355 = metadata !{i32 463, i32 3, metadata !340, null}
!356 = metadata !{i32 464, i32 3, metadata !340, null}
!357 = metadata !{i32 465, i32 3, metadata !340, null}
!358 = metadata !{i32 468, i32 3, metadata !340, null}
!359 = metadata !{i32 469, i32 3, metadata !340, null}
!360 = metadata !{i32 470, i32 3, metadata !340, null}
!361 = metadata !{i32 471, i32 2, metadata !340, null}
!362 = metadata !{i32 786689, metadata !363, metadata !"val", metadata !56, i32 16777548, metadata !169, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!363 = metadata !{i32 786478, i32 0, metadata !56, metadata !"bin2char", metadata !"bin2char", metadata !"", metadata !56, i32 332, metadata !364, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, metadata !77, i32 333} ; [ DW_TAG_subprogram ]
!364 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !365, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!365 = metadata !{metadata !65, metadata !169}
!366 = metadata !{i32 332, i32 22, metadata !363, null}
!367 = metadata !{i32 337, i32 2, metadata !368, null}
!368 = metadata !{i32 786443, metadata !363, i32 333, i32 1, metadata !56, i32 28} ; [ DW_TAG_lexical_block ]
!369 = metadata !{i32 339, i32 24, metadata !370, null}
!370 = metadata !{i32 786443, metadata !368, i32 337, i32 15, metadata !56, i32 29} ; [ DW_TAG_lexical_block ]
!371 = metadata !{i32 340, i32 24, metadata !370, null}
!372 = metadata !{i32 341, i32 24, metadata !370, null}
!373 = metadata !{i32 342, i32 24, metadata !370, null}
!374 = metadata !{i32 343, i32 24, metadata !370, null}
!375 = metadata !{i32 344, i32 24, metadata !370, null}
!376 = metadata !{i32 345, i32 24, metadata !370, null}
!377 = metadata !{i32 346, i32 24, metadata !370, null}
!378 = metadata !{i32 347, i32 24, metadata !370, null}
!379 = metadata !{i32 348, i32 25, metadata !370, null}
!380 = metadata !{i32 349, i32 25, metadata !370, null}
!381 = metadata !{i32 350, i32 25, metadata !370, null}
!382 = metadata !{i32 351, i32 25, metadata !370, null}
!383 = metadata !{i32 352, i32 25, metadata !370, null}
!384 = metadata !{i32 354, i32 2, metadata !370, null}
!385 = metadata !{i32 356, i32 2, metadata !368, null}
