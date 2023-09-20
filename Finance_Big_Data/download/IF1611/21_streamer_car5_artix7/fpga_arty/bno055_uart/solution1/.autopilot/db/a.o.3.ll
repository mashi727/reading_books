; ModuleID = 'D:/21_streamer_car5_artix7/fpga_arty/bno055_uart/solution1/.autopilot/db/a.o.3.bc'
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
@p_str6 = private unnamed_addr constant [8 x i8] c"ap_none\00", align 1 ; [#uses=10 type=[8 x i8]*]
@p_str = private unnamed_addr constant [1 x i8] zeroinitializer, align 1 ; [#uses=40 type=[1 x i8]*]

; [#uses=8]
define internal fastcc void @bno055_uart_write_mem(i8 zeroext %addr, i8 zeroext %data) nounwind uwtable {
  %data_read = call i8 @_ssdm_op_Read.ap_auto.i8(i8 %data) nounwind ; [#uses=3 type=i8]
  call void @llvm.dbg.value(metadata !{i8 %data_read}, i64 0, metadata !52), !dbg !61 ; [debug line = 272:34] [debug variable = data]
  %addr_read = call i8 @_ssdm_op_Read.ap_auto.i8(i8 %addr) nounwind ; [#uses=3 type=i8]
  call void @llvm.dbg.value(metadata !{i8 %addr_read}, i64 0, metadata !62), !dbg !63 ; [debug line = 272:22] [debug variable = addr]
  call void @llvm.dbg.value(metadata !{i8 %addr}, i64 0, metadata !62), !dbg !63 ; [debug line = 272:22] [debug variable = addr]
  call void @llvm.dbg.value(metadata !{i8 %data}, i64 0, metadata !52), !dbg !61 ; [debug line = 272:34] [debug variable = data]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !64 ; [debug line = 275:2]
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @mem_addr, i8 %addr_read) nounwind, !dbg !66 ; [debug line = 276:2]
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @mem_dout, i8 %data_read) nounwind, !dbg !67 ; [debug line = 277:2]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @mem_wreq, i1 false) nounwind, !dbg !68 ; [debug line = 278:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !69 ; [debug line = 279:2]
  br label %._crit_edge, !dbg !70                 ; [debug line = 281:2]

._crit_edge:                                      ; preds = %._crit_edge, %0
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @mem_addr, i8 %addr_read) nounwind, !dbg !71 ; [debug line = 282:3]
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @mem_dout, i8 %data_read) nounwind, !dbg !73 ; [debug line = 283:3]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @mem_wreq, i1 true) nounwind, !dbg !74 ; [debug line = 284:3]
  %mem_wack_read = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @mem_wack) nounwind, !dbg !75 ; [#uses=1 type=i1] [debug line = 285:2]
  br i1 %mem_wack_read, label %1, label %._crit_edge, !dbg !75 ; [debug line = 285:2]

; <label>:1                                       ; preds = %._crit_edge
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !76 ; [debug line = 286:2]
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @mem_addr, i8 %addr_read) nounwind, !dbg !77 ; [debug line = 288:2]
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @mem_dout, i8 %data_read) nounwind, !dbg !78 ; [debug line = 289:2]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @mem_wreq, i1 false) nounwind, !dbg !79 ; [debug line = 290:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !80 ; [debug line = 291:2]
  ret void, !dbg !81                              ; [debug line = 292:1]
}

; [#uses=15]
define internal fastcc void @bno055_uart_wait_tmr(i28 %tmr) {
  %tmr_read = call i28 @_ssdm_op_Read.ap_auto.i28(i28 %tmr) ; [#uses=1 type=i28]
  call void @llvm.dbg.value(metadata !{i28 %tmr_read}, i64 0, metadata !82), !dbg !88 ; [debug line = 74:22] [debug variable = tmr]
  call void @llvm.dbg.value(metadata !{i28 %tmr}, i64 0, metadata !82), !dbg !88 ; [debug line = 74:22] [debug variable = tmr]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !89 ; [debug line = 78:2]
  br label %1, !dbg !91                           ; [debug line = 79:7]

; <label>:1                                       ; preds = %2, %0
  %t = phi i27 [ 0, %0 ], [ %t_1, %2 ]            ; [#uses=2 type=i27]
  %t_cast = zext i27 %t to i28, !dbg !91          ; [#uses=1 type=i28] [debug line = 79:7]
  %exitcond = icmp eq i28 %t_cast, %tmr_read, !dbg !91 ; [#uses=1 type=i1] [debug line = 79:7]
  %empty = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 1, i64 100000000, i64 0) nounwind ; [#uses=0 type=i32]
  %t_1 = add i27 %t, 1, !dbg !93                  ; [#uses=1 type=i27] [debug line = 79:23]
  br i1 %exitcond, label %3, label %2, !dbg !91   ; [debug line = 79:7]

; <label>:2                                       ; preds = %1
  %dummy_tmr_out_read = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @dummy_tmr_out), !dbg !94 ; [#uses=1 type=i1] [debug line = 80:3]
  %not_s = xor i1 %dummy_tmr_out_read, true, !dbg !94 ; [#uses=1 type=i1] [debug line = 80:3]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @dummy_tmr_out, i1 %not_s), !dbg !94 ; [debug line = 80:3]
  call void @llvm.dbg.value(metadata !{i27 %t_1}, i64 0, metadata !96), !dbg !93 ; [debug line = 79:23] [debug variable = t]
  br label %1, !dbg !93                           ; [debug line = 79:23]

; <label>:3                                       ; preds = %1
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !97 ; [debug line = 82:2]
  ret void, !dbg !98                              ; [debug line = 83:1]
}

; [#uses=6]
define internal fastcc void @bno055_uart_uart_write_reg(i7 zeroext %reg_addr, i8 %data) {
  %data_read = call i8 @_ssdm_op_Read.ap_auto.i8(i8 %data) ; [#uses=1 type=i8]
  call void @llvm.dbg.value(metadata !{i8 %data_read}, i64 0, metadata !99), !dbg !101 ; [debug line = 174:43] [debug variable = data]
  %reg_addr_read = call i7 @_ssdm_op_Read.ap_auto.i7(i7 %reg_addr) ; [#uses=1 type=i7]
  call void @llvm.dbg.value(metadata !{i7 %reg_addr_read}, i64 0, metadata !102), !dbg !103 ; [debug line = 174:27] [debug variable = reg_addr]
  %reg_addr_cast = zext i7 %reg_addr_read to i8   ; [#uses=1 type=i8]
  call void @llvm.dbg.value(metadata !{i7 %reg_addr}, i64 0, metadata !102), !dbg !103 ; [debug line = 174:27] [debug variable = reg_addr]
  call void @llvm.dbg.value(metadata !{i8 %data}, i64 0, metadata !99), !dbg !101 ; [debug line = 174:43] [debug variable = data]
  call fastcc void @bno055_uart_uart_send_byte(i8 zeroext -86), !dbg !104 ; [debug line = 177:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !106 ; [debug line = 178:2]
  call fastcc void @bno055_uart_uart_send_byte(i8 zeroext 0), !dbg !107 ; [debug line = 179:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !108 ; [debug line = 180:2]
  call fastcc void @bno055_uart_uart_send_byte(i8 zeroext %reg_addr_cast), !dbg !109 ; [debug line = 181:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !110 ; [debug line = 182:2]
  call fastcc void @bno055_uart_uart_send_byte(i8 zeroext 1), !dbg !111 ; [debug line = 183:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !112 ; [debug line = 184:2]
  call fastcc void @bno055_uart_uart_send_byte(i8 zeroext %data_read), !dbg !113 ; [debug line = 185:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !114 ; [debug line = 186:2]
  call fastcc void @bno055_uart_wait_tmr(i28 1), !dbg !115 ; [debug line = 188:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !116 ; [debug line = 190:2]
  %empty = call fastcc zeroext i8 @bno055_uart_uart_receive_byte(), !dbg !117 ; [#uses=0 type=i8] [debug line = 191:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !118 ; [debug line = 192:2]
  %empty_8 = call fastcc zeroext i8 @bno055_uart_uart_receive_byte(), !dbg !119 ; [#uses=0 type=i8] [debug line = 193:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !120 ; [debug line = 194:2]
  ret void, !dbg !121                             ; [debug line = 195:1]
}

; [#uses=13]
define internal fastcc void @bno055_uart_uart_send_byte(i8 zeroext %data) nounwind uwtable {
  %data_read = call i8 @_ssdm_op_Read.ap_auto.i8(i8 %data) nounwind ; [#uses=1 type=i8]
  call void @llvm.dbg.value(metadata !{i8 %data_read}, i64 0, metadata !122), !dbg !126 ; [debug line = 93:27] [debug variable = data]
  call void @llvm.dbg.value(metadata !{i8 %data}, i64 0, metadata !122), !dbg !126 ; [debug line = 93:27] [debug variable = data]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !127 ; [debug line = 100:2]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @uart_tx, i1 false) nounwind, !dbg !129 ; [debug line = 101:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !130 ; [debug line = 102:2]
  call fastcc void @bno055_uart_wait_tmr(i28 868) nounwind, !dbg !131 ; [debug line = 103:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !132 ; [debug line = 104:2]
  br label %1, !dbg !133                          ; [debug line = 107:7]

; <label>:1                                       ; preds = %2, %0
  %i = phi i4 [ 0, %0 ], [ %i_1, %2 ]             ; [#uses=2 type=i4]
  %p_Val2_s = phi i8 [ %data_read, %0 ], [ %tmp, %2 ] ; [#uses=2 type=i8]
  %exitcond = icmp eq i4 %i, -8, !dbg !133        ; [#uses=1 type=i1] [debug line = 107:7]
  %empty = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 8, i64 8, i64 8) nounwind ; [#uses=0 type=i32]
  %i_1 = add i4 %i, 1, !dbg !135                  ; [#uses=1 type=i4] [debug line = 107:21]
  br i1 %exitcond, label %3, label %2, !dbg !133  ; [debug line = 107:7]

; <label>:2                                       ; preds = %1
  call void @llvm.dbg.value(metadata !{i8 %p_Val2_s}, i64 0, metadata !136), !dbg !139 ; [debug line = 108:41] [debug variable = __Val2__]
  %dt = trunc i8 %p_Val2_s to i1, !dbg !140       ; [#uses=1 type=i1] [debug line = 108:72]
  call void @llvm.dbg.value(metadata !{i1 %dt}, i64 0, metadata !141), !dbg !144 ; [debug line = 108:161] [debug variable = dt]
  %data_assign = call i7 @_ssdm_op_PartSelect.i7.i8.i32.i32(i8 %p_Val2_s, i32 1, i32 7), !dbg !145 ; [#uses=1 type=i7] [debug line = 109:3]
  %tmp = zext i7 %data_assign to i8, !dbg !145    ; [#uses=1 type=i8] [debug line = 109:3]
  call void @llvm.dbg.value(metadata !{i8 %tmp}, i64 0, metadata !122), !dbg !145 ; [debug line = 109:3] [debug variable = data]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !146 ; [debug line = 110:3]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @uart_tx, i1 %dt) nounwind, !dbg !147 ; [debug line = 111:3]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !148 ; [debug line = 112:3]
  call fastcc void @bno055_uart_wait_tmr(i28 868) nounwind, !dbg !149 ; [debug line = 113:3]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !150 ; [debug line = 114:3]
  call void @llvm.dbg.value(metadata !{i4 %i_1}, i64 0, metadata !151), !dbg !135 ; [debug line = 107:21] [debug variable = i]
  br label %1, !dbg !135                          ; [debug line = 107:21]

; <label>:3                                       ; preds = %1
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !154 ; [debug line = 118:2]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @uart_tx, i1 true) nounwind, !dbg !155 ; [debug line = 119:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !156 ; [debug line = 120:2]
  call fastcc void @bno055_uart_wait_tmr(i28 868) nounwind, !dbg !157 ; [debug line = 121:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !158 ; [debug line = 122:2]
  ret void, !dbg !159                             ; [debug line = 123:1]
}

; [#uses=11]
define internal fastcc zeroext i8 @bno055_uart_uart_receive_byte() nounwind uwtable {
  br label %._crit_edge, !dbg !160                ; [debug line = 134:2]

._crit_edge:                                      ; preds = %._crit_edge, %0
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !165 ; [debug line = 135:3]
  %tmp = call fastcc zeroext i1 @bno055_uart_read_uart_rx(), !dbg !167 ; [#uses=1 type=i1] [debug line = 136:11]
  br i1 %tmp, label %.preheader, label %._crit_edge, !dbg !167 ; [debug line = 136:11]

.preheader:                                       ; preds = %._crit_edge1, %._crit_edge
  %timer = phi i24 [ %timer_1, %._crit_edge1 ], [ 0, %._crit_edge ] ; [#uses=2 type=i24]
  %tmp_1 = icmp ult i24 %timer, -6777216, !dbg !168 ; [#uses=1 type=i1] [debug line = 139:7]
  %empty = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 1, i64 10000000, i64 5000000) nounwind ; [#uses=0 type=i32]
  %timer_1 = add i24 %timer, 1, !dbg !170         ; [#uses=1 type=i24] [debug line = 139:51]
  br i1 %tmp_1, label %1, label %.loopexit, !dbg !168 ; [debug line = 139:7]

; <label>:1                                       ; preds = %.preheader
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !171 ; [debug line = 140:3]
  %tmp_2 = call fastcc zeroext i1 @bno055_uart_read_uart_rx(), !dbg !173 ; [#uses=1 type=i1] [debug line = 141:7]
  br i1 %tmp_2, label %._crit_edge1, label %2, !dbg !173 ; [debug line = 141:7]

; <label>:2                                       ; preds = %1
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !174 ; [debug line = 142:4]
  call fastcc void @bno055_uart_wait_tmr(i28 217) nounwind, !dbg !176 ; [debug line = 143:4]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !177 ; [debug line = 144:4]
  %tmp_3 = call fastcc zeroext i1 @bno055_uart_read_uart_rx(), !dbg !178 ; [#uses=1 type=i1] [debug line = 145:8]
  br i1 %tmp_3, label %._crit_edge1, label %.loopexit, !dbg !178 ; [debug line = 145:8]

._crit_edge1:                                     ; preds = %2, %1
  call void @llvm.dbg.value(metadata !{i24 %timer_1}, i64 0, metadata !179), !dbg !170 ; [debug line = 139:51] [debug variable = timer]
  br label %.preheader, !dbg !170                 ; [debug line = 139:51]

.loopexit:                                        ; preds = %2, %.preheader
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !180 ; [debug line = 151:2]
  call fastcc void @bno055_uart_wait_tmr(i28 651) nounwind, !dbg !181 ; [debug line = 152:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !182 ; [debug line = 153:2]
  br label %3, !dbg !183                          ; [debug line = 155:7]

; <label>:3                                       ; preds = %4, %.loopexit
  %data = phi i8 [ 0, %.loopexit ], [ %data_1, %4 ] ; [#uses=2 type=i8]
  %i = phi i4 [ 0, %.loopexit ], [ %i_2, %4 ]     ; [#uses=2 type=i4]
  %exitcond = icmp eq i4 %i, -8, !dbg !183        ; [#uses=1 type=i1] [debug line = 155:7]
  %empty_9 = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 8, i64 8, i64 8) nounwind ; [#uses=0 type=i32]
  %i_2 = add i4 %i, 1, !dbg !185                  ; [#uses=1 type=i4] [debug line = 155:21]
  br i1 %exitcond, label %5, label %4, !dbg !183  ; [debug line = 155:7]

; <label>:4                                       ; preds = %3
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !186 ; [debug line = 156:3]
  call fastcc void @bno055_uart_wait_tmr(i28 434) nounwind, !dbg !188 ; [debug line = 157:3]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !189 ; [debug line = 158:3]
  %tmp_5 = call fastcc zeroext i1 @bno055_uart_read_uart_rx(), !dbg !190 ; [#uses=1 type=i1] [debug line = 159:7]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !191 ; [debug line = 163:3]
  %tmp_6 = call i7 @_ssdm_op_PartSelect.i7.i8.i32.i32(i8 %data, i32 1, i32 7), !dbg !192 ; [#uses=1 type=i7] [debug line = 164:3]
  %tmp_4 = select i1 %tmp_5, i1 true, i1 false, !dbg !190 ; [#uses=1 type=i1] [debug line = 159:7]
  %data_1 = call i8 @_ssdm_op_BitConcatenate.i8.i1.i7(i1 %tmp_4, i7 %tmp_6), !dbg !192 ; [#uses=1 type=i8] [debug line = 164:3]
  call void @llvm.dbg.value(metadata !{i8 %data_1}, i64 0, metadata !193), !dbg !192 ; [debug line = 164:3] [debug variable = data]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !194 ; [debug line = 165:3]
  call fastcc void @bno055_uart_wait_tmr(i28 434) nounwind, !dbg !195 ; [debug line = 166:3]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !196 ; [debug line = 167:3]
  call void @llvm.dbg.value(metadata !{i4 %i_2}, i64 0, metadata !197), !dbg !185 ; [debug line = 155:21] [debug variable = i]
  br label %3, !dbg !185                          ; [debug line = 155:21]

; <label>:5                                       ; preds = %3
  ret i8 %data, !dbg !198                         ; [debug line = 170:2]
}

; [#uses=1]
define internal fastcc zeroext i16 @bno055_uart_uart_read_reg16() nounwind uwtable {
  call fastcc void @bno055_uart_uart_send_byte(i8 zeroext -86), !dbg !199 ; [debug line = 239:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !206 ; [debug line = 240:2]
  call fastcc void @bno055_uart_uart_send_byte(i8 zeroext 1), !dbg !207 ; [debug line = 241:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !208 ; [debug line = 242:2]
  call fastcc void @bno055_uart_uart_send_byte(i8 zeroext 26), !dbg !209 ; [debug line = 243:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !210 ; [debug line = 244:2]
  call fastcc void @bno055_uart_uart_send_byte(i8 zeroext 2), !dbg !211 ; [debug line = 245:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !212 ; [debug line = 246:2]
  call fastcc void @bno055_uart_wait_tmr(i28 1) nounwind, !dbg !213 ; [debug line = 248:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !214 ; [debug line = 250:2]
  %buf = call fastcc zeroext i8 @bno055_uart_uart_receive_byte(), !dbg !215 ; [#uses=3 type=i8] [debug line = 251:8]
  call void @llvm.dbg.value(metadata !{i8 %buf}, i64 0, metadata !216), !dbg !215 ; [debug line = 251:8] [debug variable = buf]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !217 ; [debug line = 252:2]
  %tmp = icmp eq i8 %buf, -69, !dbg !218          ; [#uses=1 type=i1] [debug line = 253:2]
  br i1 %tmp, label %1, label %2, !dbg !218       ; [debug line = 253:2]

; <label>:1                                       ; preds = %0
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !219 ; [debug line = 254:3]
  %empty = call fastcc zeroext i8 @bno055_uart_uart_receive_byte(), !dbg !221 ; [#uses=0 type=i8] [debug line = 255:9]
  call void @llvm.dbg.value(metadata !{i8 %empty}, i64 0, metadata !216), !dbg !221 ; [debug line = 255:9] [debug variable = buf]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !222 ; [debug line = 256:3]
  %bh = call fastcc zeroext i8 @bno055_uart_uart_receive_byte(), !dbg !223 ; [#uses=1 type=i8] [debug line = 257:8]
  call void @llvm.dbg.value(metadata !{i8 %bh}, i64 0, metadata !224), !dbg !223 ; [debug line = 257:8] [debug variable = bh]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !225 ; [debug line = 258:3]
  %bl = call fastcc zeroext i8 @bno055_uart_uart_receive_byte(), !dbg !226 ; [#uses=1 type=i8] [debug line = 259:8]
  call void @llvm.dbg.value(metadata !{i8 %bl}, i64 0, metadata !227), !dbg !226 ; [debug line = 259:8] [debug variable = bl]
  %tmp_9 = call i16 @_ssdm_op_BitConcatenate.i16.i8.i8(i8 %bh, i8 %bl) ; [#uses=1 type=i16]
  br label %5, !dbg !228                          ; [debug line = 260:3]

; <label>:2                                       ; preds = %0
  %tmp_s = icmp eq i8 %buf, -18, !dbg !229        ; [#uses=1 type=i1] [debug line = 262:7]
  br i1 %tmp_s, label %3, label %4, !dbg !229     ; [debug line = 262:7]

; <label>:3                                       ; preds = %2
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !230 ; [debug line = 263:3]
  %buf_1 = call fastcc zeroext i8 @bno055_uart_uart_receive_byte(), !dbg !232 ; [#uses=1 type=i8] [debug line = 264:9]
  call void @llvm.dbg.value(metadata !{i8 %buf_1}, i64 0, metadata !216), !dbg !232 ; [debug line = 264:9] [debug variable = buf]
  %tmp_1 = zext i8 %buf_1 to i16, !dbg !233       ; [#uses=1 type=i16] [debug line = 265:3]
  br label %5, !dbg !233                          ; [debug line = 265:3]

; <label>:4                                       ; preds = %2
  %tmp_2 = zext i8 %buf to i16, !dbg !234         ; [#uses=1 type=i16] [debug line = 268:2]
  br label %5, !dbg !234                          ; [debug line = 268:2]

; <label>:5                                       ; preds = %4, %3, %1
  %p_0 = phi i16 [ %tmp_9, %1 ], [ %tmp_1, %3 ], [ %tmp_2, %4 ] ; [#uses=1 type=i16]
  ret i16 %p_0, !dbg !235                         ; [debug line = 269:1]
}

; [#uses=4]
define internal fastcc zeroext i8 @bno055_uart_uart_read_reg(i8 zeroext %reg_addr) nounwind uwtable {
  %reg_addr_read = call i8 @_ssdm_op_Read.ap_auto.i8(i8 %reg_addr) nounwind ; [#uses=1 type=i8]
  call void @llvm.dbg.value(metadata !{i8 %reg_addr_read}, i64 0, metadata !236), !dbg !240 ; [debug line = 198:27] [debug variable = reg_addr]
  call void @llvm.dbg.value(metadata !{i8 %reg_addr}, i64 0, metadata !236), !dbg !240 ; [debug line = 198:27] [debug variable = reg_addr]
  call fastcc void @bno055_uart_uart_send_byte(i8 zeroext -86), !dbg !241 ; [debug line = 203:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !243 ; [debug line = 204:2]
  call fastcc void @bno055_uart_uart_send_byte(i8 zeroext 1), !dbg !244 ; [debug line = 205:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !245 ; [debug line = 206:2]
  call fastcc void @bno055_uart_uart_send_byte(i8 zeroext %reg_addr_read), !dbg !246 ; [debug line = 207:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !247 ; [debug line = 208:2]
  call fastcc void @bno055_uart_uart_send_byte(i8 zeroext 1), !dbg !248 ; [debug line = 209:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !249 ; [debug line = 210:2]
  call fastcc void @bno055_uart_wait_tmr(i28 1) nounwind, !dbg !250 ; [debug line = 212:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !251 ; [debug line = 214:2]
  %buf = call fastcc zeroext i8 @bno055_uart_uart_receive_byte(), !dbg !252 ; [#uses=3 type=i8] [debug line = 215:8]
  call void @llvm.dbg.value(metadata !{i8 %buf}, i64 0, metadata !253), !dbg !252 ; [debug line = 215:8] [debug variable = buf]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !254 ; [debug line = 216:2]
  %tmp = icmp eq i8 %buf, -69, !dbg !255          ; [#uses=1 type=i1] [debug line = 217:2]
  br i1 %tmp, label %1, label %2, !dbg !255       ; [debug line = 217:2]

; <label>:1                                       ; preds = %0
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !256 ; [debug line = 218:3]
  %empty = call fastcc zeroext i8 @bno055_uart_uart_receive_byte(), !dbg !258 ; [#uses=0 type=i8] [debug line = 219:9]
  call void @llvm.dbg.value(metadata !{i8 %empty}, i64 0, metadata !253), !dbg !258 ; [debug line = 219:9] [debug variable = buf]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !259 ; [debug line = 220:3]
  %buf_2 = call fastcc zeroext i8 @bno055_uart_uart_receive_byte(), !dbg !260 ; [#uses=1 type=i8] [debug line = 221:9]
  call void @llvm.dbg.value(metadata !{i8 %buf_2}, i64 0, metadata !253), !dbg !260 ; [debug line = 221:9] [debug variable = buf]
  br label %._crit_edge, !dbg !261                ; [debug line = 222:3]

; <label>:2                                       ; preds = %0
  %tmp_s = icmp eq i8 %buf, -18, !dbg !262        ; [#uses=1 type=i1] [debug line = 224:7]
  br i1 %tmp_s, label %3, label %._crit_edge, !dbg !262 ; [debug line = 224:7]

; <label>:3                                       ; preds = %2
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !263 ; [debug line = 225:3]
  %buf_3 = call fastcc zeroext i8 @bno055_uart_uart_receive_byte(), !dbg !265 ; [#uses=1 type=i8] [debug line = 226:9]
  call void @llvm.dbg.value(metadata !{i8 %buf_3}, i64 0, metadata !253), !dbg !265 ; [debug line = 226:9] [debug variable = buf]
  br label %._crit_edge, !dbg !266                ; [debug line = 227:3]

._crit_edge:                                      ; preds = %3, %2, %1
  %p_0 = phi i8 [ %buf_2, %1 ], [ %buf_3, %3 ], [ %buf, %2 ] ; [#uses=1 type=i8]
  ret i8 %p_0, !dbg !267                          ; [debug line = 231:1]
}

; [#uses=4]
define internal fastcc zeroext i1 @bno055_uart_read_uart_rx() nounwind uwtable {
  %uart_rx_read = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @uart_rx) nounwind, !dbg !268 ; [#uses=1 type=i1] [debug line = 89:2]
  ret i1 %uart_rx_read, !dbg !268                 ; [debug line = 89:2]
}

; [#uses=3]
declare i8 @llvm.part.select.i8(i8, i32, i32) nounwind readnone

; [#uses=1]
declare i20 @llvm.part.select.i20(i20, i32, i32) nounwind readnone

; [#uses=1]
declare i16 @llvm.part.select.i16(i16, i32, i32) nounwind readnone

; [#uses=40]
declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

; [#uses=0]
define void @bno055_uart() noreturn nounwind uwtable {
  call void (...)* @_ssdm_op_SpecTopModule(), !dbg !273 ; [debug line = 365:1]
  %dummy_tmr_out_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @dummy_tmr_out) nounwind, !dbg !278 ; [#uses=0 type=i1] [debug line = 366:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @dummy_tmr_out, [8 x i8]* @p_str6, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !278 ; [debug line = 366:1]
  %uart_rx_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @uart_rx) nounwind, !dbg !279 ; [#uses=0 type=i1] [debug line = 368:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @uart_rx, [8 x i8]* @p_str6, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !279 ; [debug line = 368:1]
  %uart_tx_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @uart_tx) nounwind, !dbg !280 ; [#uses=0 type=i1] [debug line = 369:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @uart_tx, [8 x i8]* @p_str6, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !280 ; [debug line = 369:1]
  %mem_addr_load = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @mem_addr) nounwind, !dbg !281 ; [#uses=0 type=i8] [debug line = 373:1]
  call void (...)* @_ssdm_op_SpecInterface(i8* @mem_addr, [8 x i8]* @p_str6, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !281 ; [debug line = 373:1]
  %mem_din_load = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @mem_din) nounwind, !dbg !282 ; [#uses=0 type=i8] [debug line = 374:1]
  call void (...)* @_ssdm_op_SpecInterface(i8* @mem_din, [8 x i8]* @p_str6, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !282 ; [debug line = 374:1]
  %mem_dout_load = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @mem_dout) nounwind, !dbg !283 ; [#uses=0 type=i8] [debug line = 375:1]
  call void (...)* @_ssdm_op_SpecInterface(i8* @mem_dout, [8 x i8]* @p_str6, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !283 ; [debug line = 375:1]
  %mem_wreq_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @mem_wreq) nounwind, !dbg !284 ; [#uses=0 type=i1] [debug line = 376:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @mem_wreq, [8 x i8]* @p_str6, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !284 ; [debug line = 376:1]
  %mem_wack_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @mem_wack) nounwind, !dbg !285 ; [#uses=0 type=i1] [debug line = 377:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @mem_wack, [8 x i8]* @p_str6, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !285 ; [debug line = 377:1]
  %mem_rreq_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @mem_rreq) nounwind, !dbg !286 ; [#uses=0 type=i1] [debug line = 378:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @mem_rreq, [8 x i8]* @p_str6, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !286 ; [debug line = 378:1]
  %mem_rack_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @mem_rack) nounwind, !dbg !287 ; [#uses=0 type=i1] [debug line = 379:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @mem_rack, [8 x i8]* @p_str6, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !287 ; [debug line = 379:1]
  call fastcc void @bno055_uart_write_mem(i8 zeroext 21, i8 zeroext 0), !dbg !288 ; [debug line = 387:2]
  call fastcc void @bno055_uart_write_mem(i8 zeroext 22, i8 zeroext 0), !dbg !289 ; [debug line = 388:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !290 ; [debug line = 390:2]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @uart_tx, i1 true) nounwind, !dbg !291 ; [debug line = 391:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !292 ; [debug line = 392:2]
  call fastcc void @bno055_uart_uart_write_reg(i7 zeroext 7, i8 zeroext 0) nounwind, !dbg !293 ; [debug line = 402:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !294 ; [debug line = 403:2]
  call fastcc void @bno055_uart_uart_write_reg(i7 zeroext 63, i8 zeroext -64) nounwind, !dbg !295 ; [debug line = 404:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !296 ; [debug line = 405:2]
  call fastcc void @bno055_uart_wait_tmr(i28 100000000) nounwind, !dbg !297 ; [debug line = 406:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !298 ; [debug line = 407:2]
  call fastcc void @bno055_uart_uart_write_reg(i7 zeroext 7, i8 zeroext 0) nounwind, !dbg !299 ; [debug line = 409:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !300 ; [debug line = 410:2]
  %dt = call fastcc zeroext i8 @bno055_uart_uart_read_reg(i8 zeroext 62), !dbg !301 ; [#uses=1 type=i8] [debug line = 412:7]
  call void @llvm.dbg.value(metadata !{i8 %dt}, i64 0, metadata !302), !dbg !301 ; [debug line = 412:7] [debug variable = dt]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !303 ; [debug line = 413:2]
  %tmp = call i6 @_ssdm_op_PartSelect.i6.i8.i32.i32(i8 %dt, i32 2, i32 7) ; [#uses=1 type=i6]
  %tmp_s = call i8 @_ssdm_op_BitConcatenate.i8.i6.i2(i6 %tmp, i2 0), !dbg !304 ; [#uses=1 type=i8] [debug line = 414:2]
  call fastcc void @bno055_uart_uart_write_reg(i7 zeroext 62, i8 zeroext %tmp_s) nounwind, !dbg !304 ; [debug line = 414:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !305 ; [debug line = 415:2]
  call fastcc void @bno055_uart_wait_tmr(i28 10000000) nounwind, !dbg !306 ; [debug line = 416:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !307 ; [debug line = 417:2]
  %dt_1 = call fastcc zeroext i8 @bno055_uart_uart_read_reg(i8 zeroext 61), !dbg !308 ; [#uses=1 type=i8] [debug line = 419:7]
  call void @llvm.dbg.value(metadata !{i8 %dt_1}, i64 0, metadata !302), !dbg !308 ; [debug line = 419:7] [debug variable = dt]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !309 ; [debug line = 420:2]
  %tmp_9 = call i4 @_ssdm_op_PartSelect.i4.i8.i32.i32(i8 %dt_1, i32 4, i32 7) ; [#uses=1 type=i4]
  %tmp_3 = call i8 @_ssdm_op_BitConcatenate.i8.i4.i4(i4 %tmp_9, i4 0), !dbg !310 ; [#uses=1 type=i8] [debug line = 421:2]
  %tmp_4 = or i8 %tmp_3, 12, !dbg !310            ; [#uses=1 type=i8] [debug line = 421:2]
  call fastcc void @bno055_uart_uart_write_reg(i7 zeroext 61, i8 zeroext %tmp_4) nounwind, !dbg !310 ; [debug line = 421:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !311 ; [debug line = 422:2]
  call fastcc void @bno055_uart_wait_tmr(i28 100000000) nounwind, !dbg !312 ; [debug line = 423:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !313 ; [debug line = 424:2]
  %dt_2 = call fastcc zeroext i8 @bno055_uart_uart_read_reg(i8 zeroext 59), !dbg !314 ; [#uses=1 type=i8] [debug line = 426:7]
  call void @llvm.dbg.value(metadata !{i8 %dt_2}, i64 0, metadata !302), !dbg !314 ; [debug line = 426:7] [debug variable = dt]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !315 ; [debug line = 427:2]
  %tmp_5 = or i8 %dt_2, -128, !dbg !316           ; [#uses=1 type=i8] [debug line = 428:2]
  call fastcc void @bno055_uart_uart_write_reg(i7 zeroext 59, i8 zeroext %tmp_5) nounwind, !dbg !316 ; [debug line = 428:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !317 ; [debug line = 429:2]
  call fastcc void @bno055_uart_wait_tmr(i28 10000000) nounwind, !dbg !318 ; [debug line = 430:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !319 ; [debug line = 431:2]
  br label %1, !dbg !320                          ; [debug line = 434:2]

; <label>:1                                       ; preds = %1, %0
  %dt_3 = call fastcc zeroext i8 @bno055_uart_uart_read_reg(i8 zeroext 0), !dbg !321 ; [#uses=2 type=i8] [debug line = 435:8]
  call void @llvm.dbg.value(metadata !{i8 %dt_3}, i64 0, metadata !302), !dbg !321 ; [debug line = 435:8] [debug variable = dt]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !323 ; [debug line = 436:3]
  %tmp_7 = call i4 @_ssdm_op_PartSelect.i4.i8.i32.i32(i8 %dt_3, i32 4, i32 7), !dbg !324 ; [#uses=1 type=i4] [debug line = 437:16]
  %tmp_8 = call fastcc zeroext i7 @bno055_uart_bin2char(i4 zeroext %tmp_7) nounwind, !dbg !324 ; [#uses=1 type=i7] [debug line = 437:16]
  %p_trunc_ext = zext i7 %tmp_8 to i8, !dbg !324  ; [#uses=1 type=i8] [debug line = 437:16]
  call fastcc void @bno055_uart_write_mem(i8 zeroext 0, i8 zeroext %p_trunc_ext), !dbg !324 ; [debug line = 437:16]
  %tmp_10 = trunc i8 %dt_3 to i4, !dbg !325       ; [#uses=1 type=i4] [debug line = 438:16]
  %tmp_1 = call fastcc zeroext i7 @bno055_uart_bin2char(i4 zeroext %tmp_10) nounwind, !dbg !325 ; [#uses=1 type=i7] [debug line = 438:16]
  %p_trunc2_ext = zext i7 %tmp_1 to i8, !dbg !325 ; [#uses=1 type=i8] [debug line = 438:16]
  call fastcc void @bno055_uart_write_mem(i8 zeroext 1, i8 zeroext %p_trunc2_ext), !dbg !325 ; [debug line = 438:16]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !326 ; [debug line = 440:3]
  %e_tmp = call fastcc zeroext i16 @bno055_uart_uart_read_reg16(), !dbg !327 ; [#uses=2 type=i16] [debug line = 441:11]
  call void @llvm.dbg.value(metadata !{i16 %e_tmp}, i64 0, metadata !328), !dbg !327 ; [debug line = 441:11] [debug variable = e_tmp]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !329 ; [debug line = 442:3]
  %tmp_11 = trunc i16 %e_tmp to i8                ; [#uses=1 type=i8]
  %e_tmp_1 = call i8 @_ssdm_op_PartSelect.i8.i16.i32.i32(i16 %e_tmp, i32 8, i32 15) ; [#uses=1 type=i8]
  %e = call i16 @_ssdm_op_BitConcatenate.i16.i8.i8(i8 %tmp_11, i8 %e_tmp_1) ; [#uses=1 type=i16]
  %e_cast = zext i16 %e to i20, !dbg !330         ; [#uses=1 type=i20] [debug line = 444:3]
  call void @llvm.dbg.value(metadata !{i16 %e}, i64 0, metadata !331), !dbg !330 ; [debug line = 444:3] [debug variable = e]
  %e_1 = mul i20 100, %e_cast, !dbg !332          ; [#uses=2 type=i20] [debug line = 445:3]
  call void @llvm.dbg.value(metadata !{i20 %e_1}, i64 0, metadata !331), !dbg !332 ; [debug line = 445:3] [debug variable = e]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !333 ; [debug line = 460:3]
  call fastcc void @bno055_uart_write_mem(i8 zeroext -123, i8 zeroext 0), !dbg !334 ; [debug line = 461:3]
  %tmp_2 = call i8 @_ssdm_op_PartSelect.i8.i20.i32.i32(i20 %e_1, i32 12, i32 19), !dbg !335 ; [#uses=1 type=i8] [debug line = 462:3]
  call fastcc void @bno055_uart_write_mem(i8 zeroext -125, i8 zeroext %tmp_2), !dbg !335 ; [debug line = 462:3]
  %tmp_6 = call i8 @_ssdm_op_PartSelect.i8.i20.i32.i32(i20 %e_1, i32 4, i32 11), !dbg !336 ; [#uses=1 type=i8] [debug line = 463:3]
  call fastcc void @bno055_uart_write_mem(i8 zeroext -124, i8 zeroext %tmp_6), !dbg !336 ; [debug line = 463:3]
  call fastcc void @bno055_uart_write_mem(i8 zeroext -123, i8 zeroext 1), !dbg !337 ; [debug line = 464:3]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !338 ; [debug line = 465:3]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !339 ; [debug line = 468:3]
  call fastcc void @bno055_uart_wait_tmr(i28 1000000) nounwind, !dbg !340 ; [debug line = 469:3]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !341 ; [debug line = 470:3]
  br label %1, !dbg !342                          ; [debug line = 471:2]
}

; [#uses=2]
define internal fastcc zeroext i7 @bno055_uart_bin2char(i4 zeroext %val) readnone {
  %val_read = call i4 @_ssdm_op_Read.ap_auto.i4(i4 %val) ; [#uses=1 type=i4]
  call void @llvm.dbg.value(metadata !{i4 %val_read}, i64 0, metadata !343), !dbg !347 ; [debug line = 332:22] [debug variable = val]
  call void @llvm.dbg.value(metadata !{i4 %val}, i64 0, metadata !343), !dbg !347 ; [debug line = 332:22] [debug variable = val]
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
  ], !dbg !348                                    ; [debug line = 337:2]

; <label>:1                                       ; preds = %0
  br label %._crit_edge, !dbg !350                ; [debug line = 339:24]

; <label>:2                                       ; preds = %0
  br label %._crit_edge, !dbg !352                ; [debug line = 340:24]

; <label>:3                                       ; preds = %0
  br label %._crit_edge, !dbg !353                ; [debug line = 341:24]

; <label>:4                                       ; preds = %0
  br label %._crit_edge, !dbg !354                ; [debug line = 342:24]

; <label>:5                                       ; preds = %0
  br label %._crit_edge, !dbg !355                ; [debug line = 343:24]

; <label>:6                                       ; preds = %0
  br label %._crit_edge, !dbg !356                ; [debug line = 344:24]

; <label>:7                                       ; preds = %0
  br label %._crit_edge, !dbg !357                ; [debug line = 345:24]

; <label>:8                                       ; preds = %0
  br label %._crit_edge, !dbg !358                ; [debug line = 346:24]

; <label>:9                                       ; preds = %0
  br label %._crit_edge, !dbg !359                ; [debug line = 347:24]

; <label>:10                                      ; preds = %0
  br label %._crit_edge, !dbg !360                ; [debug line = 348:25]

; <label>:11                                      ; preds = %0
  br label %._crit_edge, !dbg !361                ; [debug line = 349:25]

; <label>:12                                      ; preds = %0
  br label %._crit_edge, !dbg !362                ; [debug line = 350:25]

; <label>:13                                      ; preds = %0
  br label %._crit_edge, !dbg !363                ; [debug line = 351:25]

; <label>:14                                      ; preds = %0
  br label %._crit_edge, !dbg !364                ; [debug line = 352:25]

; <label>:15                                      ; preds = %0
  br label %._crit_edge, !dbg !365                ; [debug line = 354:2]

._crit_edge:                                      ; preds = %15, %14, %13, %12, %11, %10, %9, %8, %7, %6, %5, %4, %3, %2, %1, %0
  %retval = phi i7 [ -58, %15 ], [ -59, %14 ], [ -60, %13 ], [ -61, %12 ], [ -62, %11 ], [ -63, %10 ], [ 57, %9 ], [ 56, %8 ], [ 55, %7 ], [ 54, %6 ], [ 53, %5 ], [ 52, %4 ], [ 51, %3 ], [ 50, %2 ], [ 49, %1 ], [ 48, %0 ] ; [#uses=1 type=i7]
  ret i7 %retval, !dbg !366                       ; [debug line = 356:2]
}

; [#uses=6]
define weak void @_ssdm_op_Write.ap_none.volatile.i8P(i8*, i8) {
entry:
  store i8 %1, i8* %0
  ret void
}

; [#uses=8]
define weak void @_ssdm_op_Write.ap_none.volatile.i1P(i1*, i1) {
entry:
  store i1 %1, i1* %0
  ret void
}

; [#uses=75]
define weak void @_ssdm_op_Wait(...) nounwind {
entry:
  ret void
}

; [#uses=1]
define weak void @_ssdm_op_SpecTopModule(...) nounwind {
entry:
  ret void
}

; [#uses=4]
define weak i32 @_ssdm_op_SpecLoopTripCount(...) {
entry:
  ret i32 0
}

; [#uses=10]
define weak void @_ssdm_op_SpecInterface(...) nounwind {
entry:
  ret void
}

; [#uses=3]
define weak i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8*) {
entry:
  %empty = load i8* %0                            ; [#uses=1 type=i8]
  ret i8 %empty
}

; [#uses=10]
define weak i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1*) {
entry:
  %empty = load i1* %0                            ; [#uses=1 type=i1]
  ret i1 %empty
}

; [#uses=5]
define weak i8 @_ssdm_op_Read.ap_auto.i8(i8) {
entry:
  ret i8 %0
}

; [#uses=1]
define weak i7 @_ssdm_op_Read.ap_auto.i7(i7) {
entry:
  ret i7 %0
}

; [#uses=1]
define weak i4 @_ssdm_op_Read.ap_auto.i4(i4) {
entry:
  ret i4 %0
}

; [#uses=1]
define weak i28 @_ssdm_op_Read.ap_auto.i28(i28) {
entry:
  ret i28 %0
}

; [#uses=2]
define weak i8 @_ssdm_op_PartSelect.i8.i20.i32.i32(i20, i32, i32) nounwind readnone {
entry:
  %empty = call i20 @llvm.part.select.i20(i20 %0, i32 %1, i32 %2) ; [#uses=1 type=i20]
  %empty_10 = trunc i20 %empty to i8              ; [#uses=1 type=i8]
  ret i8 %empty_10
}

; [#uses=1]
define weak i8 @_ssdm_op_PartSelect.i8.i16.i32.i32(i16, i32, i32) nounwind readnone {
entry:
  %empty = call i16 @llvm.part.select.i16(i16 %0, i32 %1, i32 %2) ; [#uses=1 type=i16]
  %empty_11 = trunc i16 %empty to i8              ; [#uses=1 type=i8]
  ret i8 %empty_11
}

; [#uses=2]
define weak i7 @_ssdm_op_PartSelect.i7.i8.i32.i32(i8, i32, i32) nounwind readnone {
entry:
  %empty = call i8 @llvm.part.select.i8(i8 %0, i32 %1, i32 %2) ; [#uses=1 type=i8]
  %empty_12 = trunc i8 %empty to i7               ; [#uses=1 type=i7]
  ret i7 %empty_12
}

; [#uses=1]
define weak i6 @_ssdm_op_PartSelect.i6.i8.i32.i32(i8, i32, i32) nounwind readnone {
entry:
  %empty = call i8 @llvm.part.select.i8(i8 %0, i32 %1, i32 %2) ; [#uses=1 type=i8]
  %empty_13 = trunc i8 %empty to i6               ; [#uses=1 type=i6]
  ret i6 %empty_13
}

; [#uses=2]
define weak i4 @_ssdm_op_PartSelect.i4.i8.i32.i32(i8, i32, i32) nounwind readnone {
entry:
  %empty = call i8 @llvm.part.select.i8(i8 %0, i32 %1, i32 %2) ; [#uses=1 type=i8]
  %empty_14 = trunc i8 %empty to i4               ; [#uses=1 type=i4]
  ret i4 %empty_14
}

; [#uses=0]
declare i1 @_ssdm_op_PartSelect.i1.i8.i32.i32(i8, i32, i32) nounwind readnone

; [#uses=0]
define weak i1 @_ssdm_op_BitSelect.i1.i8.i32(i8, i32) nounwind readnone {
entry:
  %empty = trunc i32 %1 to i8                     ; [#uses=1 type=i8]
  %empty_15 = shl i8 1, %empty                    ; [#uses=1 type=i8]
  %empty_16 = and i8 %0, %empty_15                ; [#uses=1 type=i8]
  %empty_17 = icmp ne i8 %empty_16, 0             ; [#uses=1 type=i1]
  ret i1 %empty_17
}

; [#uses=1]
define weak i8 @_ssdm_op_BitConcatenate.i8.i6.i2(i6, i2) nounwind readnone {
entry:
  %empty = zext i6 %0 to i8                       ; [#uses=1 type=i8]
  %empty_18 = zext i2 %1 to i8                    ; [#uses=1 type=i8]
  %empty_19 = shl i8 %empty, 2                    ; [#uses=1 type=i8]
  %empty_20 = or i8 %empty_19, %empty_18          ; [#uses=1 type=i8]
  ret i8 %empty_20
}

; [#uses=1]
define weak i8 @_ssdm_op_BitConcatenate.i8.i4.i4(i4, i4) nounwind readnone {
entry:
  %empty = zext i4 %0 to i8                       ; [#uses=1 type=i8]
  %empty_21 = zext i4 %1 to i8                    ; [#uses=1 type=i8]
  %empty_22 = shl i8 %empty, 4                    ; [#uses=1 type=i8]
  %empty_23 = or i8 %empty_22, %empty_21          ; [#uses=1 type=i8]
  ret i8 %empty_23
}

; [#uses=1]
define weak i8 @_ssdm_op_BitConcatenate.i8.i1.i7(i1, i7) nounwind readnone {
entry:
  %empty = zext i1 %0 to i8                       ; [#uses=1 type=i8]
  %empty_24 = zext i7 %1 to i8                    ; [#uses=1 type=i8]
  %empty_25 = shl i8 %empty, 7                    ; [#uses=1 type=i8]
  %empty_26 = or i8 %empty_25, %empty_24          ; [#uses=1 type=i8]
  ret i8 %empty_26
}

; [#uses=2]
define weak i16 @_ssdm_op_BitConcatenate.i16.i8.i8(i8, i8) nounwind readnone {
entry:
  %empty = zext i8 %0 to i16                      ; [#uses=1 type=i16]
  %empty_27 = zext i8 %1 to i16                   ; [#uses=1 type=i16]
  %empty_28 = shl i16 %empty, 8                   ; [#uses=1 type=i16]
  %empty_29 = or i16 %empty_28, %empty_27         ; [#uses=1 type=i16]
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
!52 = metadata !{i32 786689, metadata !53, metadata !"data", metadata !54, i32 33554704, metadata !57, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!53 = metadata !{i32 786478, i32 0, metadata !54, metadata !"write_mem", metadata !"write_mem", metadata !"", metadata !54, i32 272, metadata !55, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (i8, i8)* @bno055_uart_write_mem, null, null, metadata !59, i32 273} ; [ DW_TAG_subprogram ]
!54 = metadata !{i32 786473, metadata !"bno055_uart.c", metadata !"D:\5C21_streamer_car5_artix7\5Cfpga_arty", null} ; [ DW_TAG_file_type ]
!55 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !56, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!56 = metadata !{null, metadata !57, metadata !57}
!57 = metadata !{i32 786454, null, metadata !"uint8", metadata !54, i32 10, i64 0, i64 0, i64 0, i32 0, metadata !58} ; [ DW_TAG_typedef ]
!58 = metadata !{i32 786468, null, metadata !"uint8", null, i32 0, i64 8, i64 8, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!59 = metadata !{metadata !60}
!60 = metadata !{i32 786468}                      ; [ DW_TAG_base_type ]
!61 = metadata !{i32 272, i32 34, metadata !53, null}
!62 = metadata !{i32 786689, metadata !53, metadata !"addr", metadata !54, i32 16777488, metadata !57, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!63 = metadata !{i32 272, i32 22, metadata !53, null}
!64 = metadata !{i32 275, i32 2, metadata !65, null}
!65 = metadata !{i32 786443, metadata !53, i32 273, i32 1, metadata !54, i32 23} ; [ DW_TAG_lexical_block ]
!66 = metadata !{i32 276, i32 2, metadata !65, null}
!67 = metadata !{i32 277, i32 2, metadata !65, null}
!68 = metadata !{i32 278, i32 2, metadata !65, null}
!69 = metadata !{i32 279, i32 2, metadata !65, null}
!70 = metadata !{i32 281, i32 2, metadata !65, null}
!71 = metadata !{i32 282, i32 3, metadata !72, null}
!72 = metadata !{i32 786443, metadata !65, i32 281, i32 5, metadata !54, i32 24} ; [ DW_TAG_lexical_block ]
!73 = metadata !{i32 283, i32 3, metadata !72, null}
!74 = metadata !{i32 284, i32 3, metadata !72, null}
!75 = metadata !{i32 285, i32 2, metadata !72, null}
!76 = metadata !{i32 286, i32 2, metadata !65, null}
!77 = metadata !{i32 288, i32 2, metadata !65, null}
!78 = metadata !{i32 289, i32 2, metadata !65, null}
!79 = metadata !{i32 290, i32 2, metadata !65, null}
!80 = metadata !{i32 291, i32 2, metadata !65, null}
!81 = metadata !{i32 292, i32 1, metadata !65, null}
!82 = metadata !{i32 786689, metadata !83, metadata !"tmr", metadata !54, i32 16777290, metadata !86, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!83 = metadata !{i32 786478, i32 0, metadata !54, metadata !"wait_tmr", metadata !"wait_tmr", metadata !"", metadata !54, i32 74, metadata !84, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, metadata !59, i32 75} ; [ DW_TAG_subprogram ]
!84 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !85, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!85 = metadata !{null, metadata !86}
!86 = metadata !{i32 786454, null, metadata !"uint32", metadata !54, i32 34, i64 0, i64 0, i64 0, i32 0, metadata !87} ; [ DW_TAG_typedef ]
!87 = metadata !{i32 786468, null, metadata !"uint32", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!88 = metadata !{i32 74, i32 22, metadata !83, null}
!89 = metadata !{i32 78, i32 2, metadata !90, null}
!90 = metadata !{i32 786443, metadata !83, i32 75, i32 1, metadata !54, i32 0} ; [ DW_TAG_lexical_block ]
!91 = metadata !{i32 79, i32 7, metadata !92, null}
!92 = metadata !{i32 786443, metadata !90, i32 79, i32 2, metadata !54, i32 1} ; [ DW_TAG_lexical_block ]
!93 = metadata !{i32 79, i32 23, metadata !92, null}
!94 = metadata !{i32 80, i32 3, metadata !95, null}
!95 = metadata !{i32 786443, metadata !92, i32 79, i32 28, metadata !54, i32 2} ; [ DW_TAG_lexical_block ]
!96 = metadata !{i32 786688, metadata !90, metadata !"t", metadata !54, i32 77, metadata !86, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!97 = metadata !{i32 82, i32 2, metadata !90, null}
!98 = metadata !{i32 83, i32 1, metadata !90, null}
!99 = metadata !{i32 786689, metadata !100, metadata !"data", metadata !54, i32 33554606, metadata !57, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!100 = metadata !{i32 786478, i32 0, metadata !54, metadata !"uart_write_reg", metadata !"uart_write_reg", metadata !"", metadata !54, i32 174, metadata !55, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, metadata !59, i32 175} ; [ DW_TAG_subprogram ]
!101 = metadata !{i32 174, i32 43, metadata !100, null}
!102 = metadata !{i32 786689, metadata !100, metadata !"reg_addr", metadata !54, i32 16777390, metadata !57, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!103 = metadata !{i32 174, i32 27, metadata !100, null}
!104 = metadata !{i32 177, i32 2, metadata !105, null}
!105 = metadata !{i32 786443, metadata !100, i32 175, i32 1, metadata !54, i32 16} ; [ DW_TAG_lexical_block ]
!106 = metadata !{i32 178, i32 2, metadata !105, null}
!107 = metadata !{i32 179, i32 2, metadata !105, null}
!108 = metadata !{i32 180, i32 2, metadata !105, null}
!109 = metadata !{i32 181, i32 2, metadata !105, null}
!110 = metadata !{i32 182, i32 2, metadata !105, null}
!111 = metadata !{i32 183, i32 2, metadata !105, null}
!112 = metadata !{i32 184, i32 2, metadata !105, null}
!113 = metadata !{i32 185, i32 2, metadata !105, null}
!114 = metadata !{i32 186, i32 2, metadata !105, null}
!115 = metadata !{i32 188, i32 2, metadata !105, null}
!116 = metadata !{i32 190, i32 2, metadata !105, null}
!117 = metadata !{i32 191, i32 2, metadata !105, null}
!118 = metadata !{i32 192, i32 2, metadata !105, null}
!119 = metadata !{i32 193, i32 2, metadata !105, null}
!120 = metadata !{i32 194, i32 2, metadata !105, null}
!121 = metadata !{i32 195, i32 1, metadata !105, null}
!122 = metadata !{i32 786689, metadata !123, metadata !"data", metadata !54, i32 16777309, metadata !57, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!123 = metadata !{i32 786478, i32 0, metadata !54, metadata !"uart_send_byte", metadata !"uart_send_byte", metadata !"", metadata !54, i32 93, metadata !124, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (i8)* @bno055_uart_uart_send_byte, null, null, metadata !59, i32 94} ; [ DW_TAG_subprogram ]
!124 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !125, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!125 = metadata !{null, metadata !57}
!126 = metadata !{i32 93, i32 27, metadata !123, null}
!127 = metadata !{i32 100, i32 2, metadata !128, null}
!128 = metadata !{i32 786443, metadata !123, i32 94, i32 1, metadata !54, i32 4} ; [ DW_TAG_lexical_block ]
!129 = metadata !{i32 101, i32 2, metadata !128, null}
!130 = metadata !{i32 102, i32 2, metadata !128, null}
!131 = metadata !{i32 103, i32 2, metadata !128, null}
!132 = metadata !{i32 104, i32 2, metadata !128, null}
!133 = metadata !{i32 107, i32 7, metadata !134, null}
!134 = metadata !{i32 786443, metadata !128, i32 107, i32 2, metadata !54, i32 5} ; [ DW_TAG_lexical_block ]
!135 = metadata !{i32 107, i32 21, metadata !134, null}
!136 = metadata !{i32 786688, metadata !137, metadata !"__Val2__", metadata !54, i32 108, metadata !57, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!137 = metadata !{i32 786443, metadata !138, i32 108, i32 9, metadata !54, i32 7} ; [ DW_TAG_lexical_block ]
!138 = metadata !{i32 786443, metadata !134, i32 107, i32 26, metadata !54, i32 6} ; [ DW_TAG_lexical_block ]
!139 = metadata !{i32 108, i32 41, metadata !137, null}
!140 = metadata !{i32 108, i32 72, metadata !137, null}
!141 = metadata !{i32 786688, metadata !128, metadata !"dt", metadata !54, i32 97, metadata !142, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!142 = metadata !{i32 786454, null, metadata !"uint1", metadata !54, i32 3, i64 0, i64 0, i64 0, i32 0, metadata !143} ; [ DW_TAG_typedef ]
!143 = metadata !{i32 786468, null, metadata !"uint1", null, i32 0, i64 1, i64 1, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!144 = metadata !{i32 108, i32 161, metadata !137, null}
!145 = metadata !{i32 109, i32 3, metadata !138, null}
!146 = metadata !{i32 110, i32 3, metadata !138, null}
!147 = metadata !{i32 111, i32 3, metadata !138, null}
!148 = metadata !{i32 112, i32 3, metadata !138, null}
!149 = metadata !{i32 113, i32 3, metadata !138, null}
!150 = metadata !{i32 114, i32 3, metadata !138, null}
!151 = metadata !{i32 786688, metadata !128, metadata !"i", metadata !54, i32 96, metadata !152, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!152 = metadata !{i32 786454, null, metadata !"uint4", metadata !54, i32 6, i64 0, i64 0, i64 0, i32 0, metadata !153} ; [ DW_TAG_typedef ]
!153 = metadata !{i32 786468, null, metadata !"uint4", null, i32 0, i64 4, i64 4, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!154 = metadata !{i32 118, i32 2, metadata !128, null}
!155 = metadata !{i32 119, i32 2, metadata !128, null}
!156 = metadata !{i32 120, i32 2, metadata !128, null}
!157 = metadata !{i32 121, i32 2, metadata !128, null}
!158 = metadata !{i32 122, i32 2, metadata !128, null}
!159 = metadata !{i32 123, i32 1, metadata !128, null}
!160 = metadata !{i32 134, i32 2, metadata !161, null}
!161 = metadata !{i32 786443, metadata !162, i32 127, i32 1, metadata !54, i32 8} ; [ DW_TAG_lexical_block ]
!162 = metadata !{i32 786478, i32 0, metadata !54, metadata !"uart_receive_byte", metadata !"uart_receive_byte", metadata !"", metadata !54, i32 126, metadata !163, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i8 ()* @bno055_uart_uart_receive_byte, null, null, metadata !59, i32 127} ; [ DW_TAG_subprogram ]
!163 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !164, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!164 = metadata !{metadata !57}
!165 = metadata !{i32 135, i32 3, metadata !166, null}
!166 = metadata !{i32 786443, metadata !161, i32 134, i32 5, metadata !54, i32 9} ; [ DW_TAG_lexical_block ]
!167 = metadata !{i32 136, i32 11, metadata !161, null}
!168 = metadata !{i32 139, i32 7, metadata !169, null}
!169 = metadata !{i32 786443, metadata !161, i32 139, i32 2, metadata !54, i32 10} ; [ DW_TAG_lexical_block ]
!170 = metadata !{i32 139, i32 51, metadata !169, null}
!171 = metadata !{i32 140, i32 3, metadata !172, null}
!172 = metadata !{i32 786443, metadata !169, i32 139, i32 60, metadata !54, i32 11} ; [ DW_TAG_lexical_block ]
!173 = metadata !{i32 141, i32 7, metadata !172, null}
!174 = metadata !{i32 142, i32 4, metadata !175, null}
!175 = metadata !{i32 786443, metadata !172, i32 141, i32 28, metadata !54, i32 12} ; [ DW_TAG_lexical_block ]
!176 = metadata !{i32 143, i32 4, metadata !175, null}
!177 = metadata !{i32 144, i32 4, metadata !175, null}
!178 = metadata !{i32 145, i32 8, metadata !175, null}
!179 = metadata !{i32 786688, metadata !161, metadata !"timer", metadata !54, i32 132, metadata !86, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!180 = metadata !{i32 151, i32 2, metadata !161, null}
!181 = metadata !{i32 152, i32 2, metadata !161, null}
!182 = metadata !{i32 153, i32 2, metadata !161, null}
!183 = metadata !{i32 155, i32 7, metadata !184, null}
!184 = metadata !{i32 786443, metadata !161, i32 155, i32 2, metadata !54, i32 14} ; [ DW_TAG_lexical_block ]
!185 = metadata !{i32 155, i32 21, metadata !184, null}
!186 = metadata !{i32 156, i32 3, metadata !187, null}
!187 = metadata !{i32 786443, metadata !184, i32 155, i32 26, metadata !54, i32 15} ; [ DW_TAG_lexical_block ]
!188 = metadata !{i32 157, i32 3, metadata !187, null}
!189 = metadata !{i32 158, i32 3, metadata !187, null}
!190 = metadata !{i32 159, i32 7, metadata !187, null}
!191 = metadata !{i32 163, i32 3, metadata !187, null}
!192 = metadata !{i32 164, i32 3, metadata !187, null}
!193 = metadata !{i32 786688, metadata !161, metadata !"data", metadata !54, i32 130, metadata !57, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!194 = metadata !{i32 165, i32 3, metadata !187, null}
!195 = metadata !{i32 166, i32 3, metadata !187, null}
!196 = metadata !{i32 167, i32 3, metadata !187, null}
!197 = metadata !{i32 786688, metadata !161, metadata !"i", metadata !54, i32 129, metadata !152, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!198 = metadata !{i32 170, i32 2, metadata !161, null}
!199 = metadata !{i32 239, i32 2, metadata !200, null}
!200 = metadata !{i32 786443, metadata !201, i32 235, i32 1, metadata !54, i32 20} ; [ DW_TAG_lexical_block ]
!201 = metadata !{i32 786478, i32 0, metadata !54, metadata !"uart_read_reg16", metadata !"uart_read_reg16", metadata !"", metadata !54, i32 234, metadata !202, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, metadata !59, i32 235} ; [ DW_TAG_subprogram ]
!202 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !203, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!203 = metadata !{metadata !204, metadata !57}
!204 = metadata !{i32 786454, null, metadata !"uint16", metadata !54, i32 18, i64 0, i64 0, i64 0, i32 0, metadata !205} ; [ DW_TAG_typedef ]
!205 = metadata !{i32 786468, null, metadata !"uint16", null, i32 0, i64 16, i64 16, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!206 = metadata !{i32 240, i32 2, metadata !200, null}
!207 = metadata !{i32 241, i32 2, metadata !200, null}
!208 = metadata !{i32 242, i32 2, metadata !200, null}
!209 = metadata !{i32 243, i32 2, metadata !200, null}
!210 = metadata !{i32 244, i32 2, metadata !200, null}
!211 = metadata !{i32 245, i32 2, metadata !200, null}
!212 = metadata !{i32 246, i32 2, metadata !200, null}
!213 = metadata !{i32 248, i32 2, metadata !200, null}
!214 = metadata !{i32 250, i32 2, metadata !200, null}
!215 = metadata !{i32 251, i32 8, metadata !200, null}
!216 = metadata !{i32 786688, metadata !200, metadata !"buf", metadata !54, i32 237, metadata !57, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!217 = metadata !{i32 252, i32 2, metadata !200, null}
!218 = metadata !{i32 253, i32 2, metadata !200, null}
!219 = metadata !{i32 254, i32 3, metadata !220, null}
!220 = metadata !{i32 786443, metadata !200, i32 253, i32 19, metadata !54, i32 21} ; [ DW_TAG_lexical_block ]
!221 = metadata !{i32 255, i32 9, metadata !220, null}
!222 = metadata !{i32 256, i32 3, metadata !220, null}
!223 = metadata !{i32 257, i32 8, metadata !220, null}
!224 = metadata !{i32 786688, metadata !200, metadata !"bh", metadata !54, i32 237, metadata !57, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!225 = metadata !{i32 258, i32 3, metadata !220, null}
!226 = metadata !{i32 259, i32 8, metadata !220, null}
!227 = metadata !{i32 786688, metadata !200, metadata !"bl", metadata !54, i32 237, metadata !57, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!228 = metadata !{i32 260, i32 3, metadata !220, null}
!229 = metadata !{i32 262, i32 7, metadata !200, null}
!230 = metadata !{i32 263, i32 3, metadata !231, null}
!231 = metadata !{i32 786443, metadata !200, i32 262, i32 24, metadata !54, i32 22} ; [ DW_TAG_lexical_block ]
!232 = metadata !{i32 264, i32 9, metadata !231, null}
!233 = metadata !{i32 265, i32 3, metadata !231, null}
!234 = metadata !{i32 268, i32 2, metadata !200, null}
!235 = metadata !{i32 269, i32 1, metadata !200, null}
!236 = metadata !{i32 786689, metadata !237, metadata !"reg_addr", metadata !54, i32 16777414, metadata !57, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!237 = metadata !{i32 786478, i32 0, metadata !54, metadata !"uart_read_reg", metadata !"uart_read_reg", metadata !"", metadata !54, i32 198, metadata !238, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i8 (i8)* @bno055_uart_uart_read_reg, null, null, metadata !59, i32 199} ; [ DW_TAG_subprogram ]
!238 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !239, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!239 = metadata !{metadata !57, metadata !57}
!240 = metadata !{i32 198, i32 27, metadata !237, null}
!241 = metadata !{i32 203, i32 2, metadata !242, null}
!242 = metadata !{i32 786443, metadata !237, i32 199, i32 1, metadata !54, i32 17} ; [ DW_TAG_lexical_block ]
!243 = metadata !{i32 204, i32 2, metadata !242, null}
!244 = metadata !{i32 205, i32 2, metadata !242, null}
!245 = metadata !{i32 206, i32 2, metadata !242, null}
!246 = metadata !{i32 207, i32 2, metadata !242, null}
!247 = metadata !{i32 208, i32 2, metadata !242, null}
!248 = metadata !{i32 209, i32 2, metadata !242, null}
!249 = metadata !{i32 210, i32 2, metadata !242, null}
!250 = metadata !{i32 212, i32 2, metadata !242, null}
!251 = metadata !{i32 214, i32 2, metadata !242, null}
!252 = metadata !{i32 215, i32 8, metadata !242, null}
!253 = metadata !{i32 786688, metadata !242, metadata !"buf", metadata !54, i32 201, metadata !57, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!254 = metadata !{i32 216, i32 2, metadata !242, null}
!255 = metadata !{i32 217, i32 2, metadata !242, null}
!256 = metadata !{i32 218, i32 3, metadata !257, null}
!257 = metadata !{i32 786443, metadata !242, i32 217, i32 19, metadata !54, i32 18} ; [ DW_TAG_lexical_block ]
!258 = metadata !{i32 219, i32 9, metadata !257, null}
!259 = metadata !{i32 220, i32 3, metadata !257, null}
!260 = metadata !{i32 221, i32 9, metadata !257, null}
!261 = metadata !{i32 222, i32 3, metadata !257, null}
!262 = metadata !{i32 224, i32 7, metadata !242, null}
!263 = metadata !{i32 225, i32 3, metadata !264, null}
!264 = metadata !{i32 786443, metadata !242, i32 224, i32 24, metadata !54, i32 19} ; [ DW_TAG_lexical_block ]
!265 = metadata !{i32 226, i32 9, metadata !264, null}
!266 = metadata !{i32 227, i32 3, metadata !264, null}
!267 = metadata !{i32 231, i32 1, metadata !242, null}
!268 = metadata !{i32 89, i32 2, metadata !269, null}
!269 = metadata !{i32 786443, metadata !270, i32 87, i32 1, metadata !54, i32 3} ; [ DW_TAG_lexical_block ]
!270 = metadata !{i32 786478, i32 0, metadata !54, metadata !"read_uart_rx", metadata !"read_uart_rx", metadata !"", metadata !54, i32 86, metadata !271, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i1 ()* @bno055_uart_read_uart_rx, null, null, metadata !59, i32 87} ; [ DW_TAG_subprogram ]
!271 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !272, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!272 = metadata !{metadata !142}
!273 = metadata !{i32 365, i32 1, metadata !274, null}
!274 = metadata !{i32 786443, metadata !275, i32 364, i32 1, metadata !54, i32 30} ; [ DW_TAG_lexical_block ]
!275 = metadata !{i32 786478, i32 0, metadata !54, metadata !"bno055_uart", metadata !"bno055_uart", metadata !"", metadata !54, i32 363, metadata !276, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @bno055_uart, null, null, metadata !59, i32 364} ; [ DW_TAG_subprogram ]
!276 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !277, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!277 = metadata !{null}
!278 = metadata !{i32 366, i32 1, metadata !274, null}
!279 = metadata !{i32 368, i32 1, metadata !274, null}
!280 = metadata !{i32 369, i32 1, metadata !274, null}
!281 = metadata !{i32 373, i32 1, metadata !274, null}
!282 = metadata !{i32 374, i32 1, metadata !274, null}
!283 = metadata !{i32 375, i32 1, metadata !274, null}
!284 = metadata !{i32 376, i32 1, metadata !274, null}
!285 = metadata !{i32 377, i32 1, metadata !274, null}
!286 = metadata !{i32 378, i32 1, metadata !274, null}
!287 = metadata !{i32 379, i32 1, metadata !274, null}
!288 = metadata !{i32 387, i32 2, metadata !274, null}
!289 = metadata !{i32 388, i32 2, metadata !274, null}
!290 = metadata !{i32 390, i32 2, metadata !274, null}
!291 = metadata !{i32 391, i32 2, metadata !274, null}
!292 = metadata !{i32 392, i32 2, metadata !274, null}
!293 = metadata !{i32 402, i32 2, metadata !274, null}
!294 = metadata !{i32 403, i32 2, metadata !274, null}
!295 = metadata !{i32 404, i32 2, metadata !274, null}
!296 = metadata !{i32 405, i32 2, metadata !274, null}
!297 = metadata !{i32 406, i32 2, metadata !274, null}
!298 = metadata !{i32 407, i32 2, metadata !274, null}
!299 = metadata !{i32 409, i32 2, metadata !274, null}
!300 = metadata !{i32 410, i32 2, metadata !274, null}
!301 = metadata !{i32 412, i32 7, metadata !274, null}
!302 = metadata !{i32 786688, metadata !274, metadata !"dt", metadata !54, i32 381, metadata !57, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!303 = metadata !{i32 413, i32 2, metadata !274, null}
!304 = metadata !{i32 414, i32 2, metadata !274, null}
!305 = metadata !{i32 415, i32 2, metadata !274, null}
!306 = metadata !{i32 416, i32 2, metadata !274, null}
!307 = metadata !{i32 417, i32 2, metadata !274, null}
!308 = metadata !{i32 419, i32 7, metadata !274, null}
!309 = metadata !{i32 420, i32 2, metadata !274, null}
!310 = metadata !{i32 421, i32 2, metadata !274, null}
!311 = metadata !{i32 422, i32 2, metadata !274, null}
!312 = metadata !{i32 423, i32 2, metadata !274, null}
!313 = metadata !{i32 424, i32 2, metadata !274, null}
!314 = metadata !{i32 426, i32 7, metadata !274, null}
!315 = metadata !{i32 427, i32 2, metadata !274, null}
!316 = metadata !{i32 428, i32 2, metadata !274, null}
!317 = metadata !{i32 429, i32 2, metadata !274, null}
!318 = metadata !{i32 430, i32 2, metadata !274, null}
!319 = metadata !{i32 431, i32 2, metadata !274, null}
!320 = metadata !{i32 434, i32 2, metadata !274, null}
!321 = metadata !{i32 435, i32 8, metadata !322, null}
!322 = metadata !{i32 786443, metadata !274, i32 434, i32 12, metadata !54, i32 31} ; [ DW_TAG_lexical_block ]
!323 = metadata !{i32 436, i32 3, metadata !322, null}
!324 = metadata !{i32 437, i32 16, metadata !322, null}
!325 = metadata !{i32 438, i32 16, metadata !322, null}
!326 = metadata !{i32 440, i32 3, metadata !322, null}
!327 = metadata !{i32 441, i32 11, metadata !322, null}
!328 = metadata !{i32 786688, metadata !274, metadata !"e_tmp", metadata !54, i32 383, metadata !204, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!329 = metadata !{i32 442, i32 3, metadata !322, null}
!330 = metadata !{i32 444, i32 3, metadata !322, null}
!331 = metadata !{i32 786688, metadata !274, metadata !"e", metadata !54, i32 384, metadata !86, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!332 = metadata !{i32 445, i32 3, metadata !322, null}
!333 = metadata !{i32 460, i32 3, metadata !322, null}
!334 = metadata !{i32 461, i32 3, metadata !322, null}
!335 = metadata !{i32 462, i32 3, metadata !322, null}
!336 = metadata !{i32 463, i32 3, metadata !322, null}
!337 = metadata !{i32 464, i32 3, metadata !322, null}
!338 = metadata !{i32 465, i32 3, metadata !322, null}
!339 = metadata !{i32 468, i32 3, metadata !322, null}
!340 = metadata !{i32 469, i32 3, metadata !322, null}
!341 = metadata !{i32 470, i32 3, metadata !322, null}
!342 = metadata !{i32 471, i32 2, metadata !322, null}
!343 = metadata !{i32 786689, metadata !344, metadata !"val", metadata !54, i32 16777548, metadata !152, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!344 = metadata !{i32 786478, i32 0, metadata !54, metadata !"bin2char", metadata !"bin2char", metadata !"", metadata !54, i32 332, metadata !345, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, metadata !59, i32 333} ; [ DW_TAG_subprogram ]
!345 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !346, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!346 = metadata !{metadata !57, metadata !152}
!347 = metadata !{i32 332, i32 22, metadata !344, null}
!348 = metadata !{i32 337, i32 2, metadata !349, null}
!349 = metadata !{i32 786443, metadata !344, i32 333, i32 1, metadata !54, i32 28} ; [ DW_TAG_lexical_block ]
!350 = metadata !{i32 339, i32 24, metadata !351, null}
!351 = metadata !{i32 786443, metadata !349, i32 337, i32 15, metadata !54, i32 29} ; [ DW_TAG_lexical_block ]
!352 = metadata !{i32 340, i32 24, metadata !351, null}
!353 = metadata !{i32 341, i32 24, metadata !351, null}
!354 = metadata !{i32 342, i32 24, metadata !351, null}
!355 = metadata !{i32 343, i32 24, metadata !351, null}
!356 = metadata !{i32 344, i32 24, metadata !351, null}
!357 = metadata !{i32 345, i32 24, metadata !351, null}
!358 = metadata !{i32 346, i32 24, metadata !351, null}
!359 = metadata !{i32 347, i32 24, metadata !351, null}
!360 = metadata !{i32 348, i32 25, metadata !351, null}
!361 = metadata !{i32 349, i32 25, metadata !351, null}
!362 = metadata !{i32 350, i32 25, metadata !351, null}
!363 = metadata !{i32 351, i32 25, metadata !351, null}
!364 = metadata !{i32 352, i32 25, metadata !351, null}
!365 = metadata !{i32 354, i32 2, metadata !351, null}
!366 = metadata !{i32 356, i32 2, metadata !349, null}
