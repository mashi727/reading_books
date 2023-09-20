; ModuleID = 'D:/21_streamer_car5_artix7/fpga_arty/lcddrv/solution1/.autopilot/db/a.o.3.bc'
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
@p_str2 = private unnamed_addr constant [12 x i8] c"hls_label_0\00", align 1 ; [#uses=2 type=[12 x i8]*]
@p_str1 = private unnamed_addr constant [8 x i8] c"ap_none\00", align 1 ; [#uses=9 type=[8 x i8]*]
@p_str = private unnamed_addr constant [1 x i8] zeroinitializer, align 1 ; [#uses=36 type=[1 x i8]*]

; [#uses=13]
define internal fastcc void @lcddrv_wait_tmr(i25 %tmr) {
  %tmr_read = call i25 @_ssdm_op_Read.ap_auto.i25(i25 %tmr) ; [#uses=1 type=i25]
  call void @llvm.dbg.value(metadata !{i25 %tmr_read}, i64 0, metadata !47), !dbg !56 ; [debug line = 65:22] [debug variable = tmr]
  call void @llvm.dbg.value(metadata !{i25 %tmr}, i64 0, metadata !47), !dbg !56 ; [debug line = 65:22] [debug variable = tmr]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !57 ; [debug line = 69:2]
  br label %1, !dbg !59                           ; [debug line = 70:7]

; <label>:1                                       ; preds = %2, %0
  %t = phi i24 [ 0, %0 ], [ %t_1, %2 ]            ; [#uses=2 type=i24]
  %t_cast = zext i24 %t to i25, !dbg !59          ; [#uses=1 type=i25] [debug line = 70:7]
  %exitcond = icmp eq i25 %t_cast, %tmr_read, !dbg !59 ; [#uses=1 type=i1] [debug line = 70:7]
  %empty = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 1000, i64 10000000, i64 0) nounwind ; [#uses=0 type=i32]
  %t_1 = add i24 %t, 1, !dbg !61                  ; [#uses=1 type=i24] [debug line = 70:23]
  br i1 %exitcond, label %3, label %2, !dbg !59   ; [debug line = 70:7]

; <label>:2                                       ; preds = %1
  %dummy_tmr_out_read = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @dummy_tmr_out), !dbg !62 ; [#uses=1 type=i1] [debug line = 71:3]
  %not_s = xor i1 %dummy_tmr_out_read, true, !dbg !62 ; [#uses=1 type=i1] [debug line = 71:3]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @dummy_tmr_out, i1 %not_s), !dbg !62 ; [debug line = 71:3]
  call void @llvm.dbg.value(metadata !{i24 %t_1}, i64 0, metadata !64), !dbg !61 ; [debug line = 70:23] [debug variable = t]
  br label %1, !dbg !61                           ; [debug line = 70:23]

; <label>:3                                       ; preds = %1
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !65 ; [debug line = 73:2]
  ret void, !dbg !66                              ; [debug line = 74:1]
}

; [#uses=2]
define internal fastcc zeroext i8 @lcddrv_read_mem(i5 zeroext %addr) nounwind uwtable {
  %addr_read = call i5 @_ssdm_op_Read.ap_auto.i5(i5 %addr) nounwind ; [#uses=2 type=i5]
  call void @llvm.dbg.value(metadata !{i5 %addr_read}, i64 0, metadata !67), !dbg !75 ; [debug line = 138:22] [debug variable = addr]
  call void @llvm.dbg.value(metadata !{i5 %addr}, i64 0, metadata !67), !dbg !75 ; [debug line = 138:22] [debug variable = addr]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !76 ; [debug line = 143:2]
  call void @_ssdm_op_Write.ap_none.volatile.i5P(i5* @mem_addr, i5 %addr_read) nounwind, !dbg !78 ; [debug line = 144:2]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @mem_req, i1 true) nounwind, !dbg !79 ; [debug line = 145:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !80 ; [debug line = 146:2]
  br label %._crit_edge, !dbg !81                 ; [debug line = 147:2]

._crit_edge:                                      ; preds = %._crit_edge, %0
  call void @_ssdm_op_Write.ap_none.volatile.i5P(i5* @mem_addr, i5 %addr_read) nounwind, !dbg !82 ; [debug line = 148:3]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @mem_req, i1 true) nounwind, !dbg !84 ; [debug line = 149:3]
  %dt = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @mem_din) nounwind, !dbg !85 ; [#uses=1 type=i8] [debug line = 150:3]
  call void @llvm.dbg.value(metadata !{i8 %dt}, i64 0, metadata !86), !dbg !85 ; [debug line = 150:3] [debug variable = dt]
  %mem_ack_read = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @mem_ack) nounwind, !dbg !87 ; [#uses=1 type=i1] [debug line = 151:2]
  br i1 %mem_ack_read, label %1, label %._crit_edge, !dbg !87 ; [debug line = 151:2]

; <label>:1                                       ; preds = %._crit_edge
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !88 ; [debug line = 152:2]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @mem_req, i1 false) nounwind, !dbg !89 ; [debug line = 153:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !90 ; [debug line = 154:2]
  ret i8 %dt, !dbg !91                            ; [debug line = 156:2]
}

; [#uses=1]
declare i8 @llvm.part.select.i8(i8, i32, i32) nounwind readnone

; [#uses=14]
declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

; [#uses=0]
define void @lcddrv() noreturn nounwind uwtable {
  call void (...)* @_ssdm_op_SpecTopModule(), !dbg !92 ; [debug line = 162:1]
  %dummy_tmr_out_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @dummy_tmr_out) nounwind, !dbg !97 ; [#uses=0 type=i1] [debug line = 163:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @dummy_tmr_out, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !97 ; [debug line = 163:1]
  %rs_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @rs) nounwind, !dbg !98 ; [#uses=0 type=i1] [debug line = 165:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @rs, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !98 ; [debug line = 165:1]
  %en_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @en) nounwind, !dbg !99 ; [#uses=0 type=i1] [debug line = 166:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @en, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !99 ; [debug line = 166:1]
  %data_load = call i4 @_ssdm_op_Read.ap_none.volatile.i4P(i4* @data) nounwind, !dbg !100 ; [#uses=0 type=i4] [debug line = 167:1]
  call void (...)* @_ssdm_op_SpecInterface(i4* @data, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !100 ; [debug line = 167:1]
  %ind_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @ind) nounwind, !dbg !101 ; [#uses=0 type=i1] [debug line = 168:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @ind, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !101 ; [debug line = 168:1]
  %mem_addr_load = call i5 @_ssdm_op_Read.ap_none.volatile.i5P(i5* @mem_addr) nounwind, !dbg !102 ; [#uses=0 type=i5] [debug line = 170:1]
  call void (...)* @_ssdm_op_SpecInterface(i5* @mem_addr, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !102 ; [debug line = 170:1]
  %mem_din_load = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @mem_din) nounwind, !dbg !103 ; [#uses=0 type=i8] [debug line = 171:1]
  call void (...)* @_ssdm_op_SpecInterface(i8* @mem_din, [8 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !103 ; [debug line = 171:1]
  %mem_req_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @mem_req) nounwind, !dbg !104 ; [#uses=0 type=i1] [debug line = 172:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @mem_req, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !104 ; [debug line = 172:1]
  %mem_ack_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @mem_ack) nounwind, !dbg !105 ; [#uses=0 type=i1] [debug line = 173:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @mem_ack, [8 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !105 ; [debug line = 173:1]
  call fastcc void @lcddrv_init_lcd(), !dbg !106  ; [debug line = 178:2]
  br label %1, !dbg !107                          ; [debug line = 180:2]

; <label>:1                                       ; preds = %7, %0
  %loop_begin = call i32 (...)* @_ssdm_op_SpecLoopBegin() nounwind ; [#uses=0 type=i32]
  %tmp = call i32 (...)* @_ssdm_op_SpecRegionBegin([12 x i8]* @p_str2) nounwind, !dbg !108 ; [#uses=1 type=i32] [debug line = 180:13]
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext false, i4 zeroext -8), !dbg !110 ; [debug line = 182:3]
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext false, i4 zeroext 0), !dbg !111 ; [debug line = 183:3]
  br label %2, !dbg !112                          ; [debug line = 185:8]

; <label>:2                                       ; preds = %3, %1
  %pos = phi i5 [ 0, %1 ], [ %pos_2, %3 ]         ; [#uses=3 type=i5]
  %exitcond1 = icmp eq i5 %pos, -16, !dbg !112    ; [#uses=1 type=i1] [debug line = 185:8]
  %empty = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 16, i64 16, i64 16) nounwind ; [#uses=0 type=i32]
  %pos_2 = add i5 %pos, 1, !dbg !114              ; [#uses=1 type=i5] [debug line = 185:27]
  br i1 %exitcond1, label %4, label %3, !dbg !112 ; [debug line = 185:8]

; <label>:3                                       ; preds = %2
  %dt = call fastcc zeroext i8 @lcddrv_read_mem(i5 zeroext %pos), !dbg !115 ; [#uses=2 type=i8] [debug line = 186:9]
  call void @llvm.dbg.value(metadata !{i8 %dt}, i64 0, metadata !117), !dbg !115 ; [debug line = 186:9] [debug variable = dt]
  %tmp_1 = call i4 @_ssdm_op_PartSelect.i4.i8.i32.i32(i8 %dt, i32 4, i32 7), !dbg !118 ; [#uses=1 type=i4] [debug line = 187:4]
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext true, i4 zeroext %tmp_1), !dbg !118 ; [debug line = 187:4]
  %tmp_2 = trunc i8 %dt to i4, !dbg !119          ; [#uses=1 type=i4] [debug line = 188:4]
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext true, i4 zeroext %tmp_2), !dbg !119 ; [debug line = 188:4]
  call void @llvm.dbg.value(metadata !{i5 %pos_2}, i64 0, metadata !120), !dbg !114 ; [debug line = 185:27] [debug variable = pos]
  br label %2, !dbg !114                          ; [debug line = 185:27]

; <label>:4                                       ; preds = %2
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext false, i4 zeroext -4), !dbg !121 ; [debug line = 192:3]
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext false, i4 zeroext 0), !dbg !122 ; [debug line = 193:3]
  br label %5, !dbg !123                          ; [debug line = 195:8]

; <label>:5                                       ; preds = %6, %4
  %pos_1 = phi i6 [ 16, %4 ], [ %pos_3, %6 ]      ; [#uses=3 type=i6]
  %exitcond = icmp eq i6 %pos_1, -32, !dbg !123   ; [#uses=1 type=i1] [debug line = 195:8]
  %empty_4 = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 16, i64 16, i64 16) nounwind ; [#uses=0 type=i32]
  br i1 %exitcond, label %7, label %6, !dbg !123  ; [debug line = 195:8]

; <label>:6                                       ; preds = %5
  %tmp_3 = trunc i6 %pos_1 to i5, !dbg !125       ; [#uses=1 type=i5] [debug line = 196:9]
  %dt_1 = call fastcc zeroext i8 @lcddrv_read_mem(i5 zeroext %tmp_3), !dbg !125 ; [#uses=2 type=i8] [debug line = 196:9]
  call void @llvm.dbg.value(metadata !{i8 %dt_1}, i64 0, metadata !117), !dbg !125 ; [debug line = 196:9] [debug variable = dt]
  %tmp_5 = call i4 @_ssdm_op_PartSelect.i4.i8.i32.i32(i8 %dt_1, i32 4, i32 7), !dbg !127 ; [#uses=1 type=i4] [debug line = 197:4]
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext true, i4 zeroext %tmp_5), !dbg !127 ; [debug line = 197:4]
  %tmp_4 = trunc i8 %dt_1 to i4, !dbg !128        ; [#uses=1 type=i4] [debug line = 198:4]
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext true, i4 zeroext %tmp_4), !dbg !128 ; [debug line = 198:4]
  %pos_3 = add i6 1, %pos_1, !dbg !129            ; [#uses=1 type=i6] [debug line = 195:28]
  call void @llvm.dbg.value(metadata !{i6 %pos_3}, i64 0, metadata !120), !dbg !129 ; [debug line = 195:28] [debug variable = pos]
  br label %5, !dbg !129                          ; [debug line = 195:28]

; <label>:7                                       ; preds = %5
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !130 ; [debug line = 202:3]
  call fastcc void @lcddrv_wait_tmr(i25 10000000) nounwind, !dbg !131 ; [debug line = 203:3]
  %empty_5 = call i32 (...)* @_ssdm_op_SpecRegionEnd([12 x i8]* @p_str2, i32 %tmp) nounwind, !dbg !132 ; [#uses=0 type=i32] [debug line = 204:2]
  br label %1, !dbg !132                          ; [debug line = 204:2]
}

; [#uses=20]
define internal fastcc void @lcddrv_lcd_send_cmd(i1 zeroext %mode, i4 zeroext %wd) nounwind uwtable {
  %wd_read = call i4 @_ssdm_op_Read.ap_auto.i4(i4 %wd) nounwind ; [#uses=3 type=i4]
  call void @llvm.dbg.value(metadata !{i4 %wd_read}, i64 0, metadata !133), !dbg !141 ; [debug line = 77:37] [debug variable = wd]
  %mode_read = call i1 @_ssdm_op_Read.ap_auto.i1(i1 %mode) nounwind ; [#uses=3 type=i1]
  call void @llvm.dbg.value(metadata !{i1 %mode_read}, i64 0, metadata !142), !dbg !143 ; [debug line = 77:25] [debug variable = mode]
  call void @llvm.dbg.value(metadata !{i1 %mode}, i64 0, metadata !142), !dbg !143 ; [debug line = 77:25] [debug variable = mode]
  call void @llvm.dbg.value(metadata !{i4 %wd}, i64 0, metadata !133), !dbg !141 ; [debug line = 77:37] [debug variable = wd]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !144 ; [debug line = 80:2]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @en, i1 false) nounwind, !dbg !146 ; [debug line = 81:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !147 ; [debug line = 82:2]
  call fastcc void @lcddrv_wait_tmr(i25 1000) nounwind, !dbg !148 ; [debug line = 83:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !149 ; [debug line = 85:2]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @en, i1 false) nounwind, !dbg !150 ; [debug line = 86:2]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @rs, i1 %mode_read) nounwind, !dbg !151 ; [debug line = 87:2]
  call void @_ssdm_op_Write.ap_none.volatile.i4P(i4* @data, i4 %wd_read) nounwind, !dbg !152 ; [debug line = 88:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !153 ; [debug line = 89:2]
  call fastcc void @lcddrv_wait_tmr(i25 1000) nounwind, !dbg !154 ; [debug line = 90:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !155 ; [debug line = 92:2]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @en, i1 true) nounwind, !dbg !156 ; [debug line = 93:2]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @rs, i1 %mode_read) nounwind, !dbg !157 ; [debug line = 94:2]
  call void @_ssdm_op_Write.ap_none.volatile.i4P(i4* @data, i4 %wd_read) nounwind, !dbg !158 ; [debug line = 95:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !159 ; [debug line = 96:2]
  call fastcc void @lcddrv_wait_tmr(i25 1000) nounwind, !dbg !160 ; [debug line = 97:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !161 ; [debug line = 99:2]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @en, i1 false) nounwind, !dbg !162 ; [debug line = 100:2]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @rs, i1 %mode_read) nounwind, !dbg !163 ; [debug line = 101:2]
  call void @_ssdm_op_Write.ap_none.volatile.i4P(i4* @data, i4 %wd_read) nounwind, !dbg !164 ; [debug line = 102:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !165 ; [debug line = 103:2]
  call fastcc void @lcddrv_wait_tmr(i25 1000) nounwind, !dbg !166 ; [debug line = 104:2]
  ret void, !dbg !167                             ; [debug line = 105:1]
}

; [#uses=1]
define internal fastcc void @lcddrv_init_lcd() nounwind uwtable {
  call fastcc void @lcddrv_wait_tmr(i25 2000000) nounwind, !dbg !168 ; [debug line = 111:2]
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext false, i4 zeroext 3), !dbg !171 ; [debug line = 112:2]
  call fastcc void @lcddrv_wait_tmr(i25 500000) nounwind, !dbg !172 ; [debug line = 113:2]
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext false, i4 zeroext 3), !dbg !173 ; [debug line = 114:2]
  call fastcc void @lcddrv_wait_tmr(i25 50000) nounwind, !dbg !174 ; [debug line = 115:2]
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext false, i4 zeroext 3), !dbg !175 ; [debug line = 116:2]
  call fastcc void @lcddrv_wait_tmr(i25 50000) nounwind, !dbg !176 ; [debug line = 117:2]
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext false, i4 zeroext 2), !dbg !177 ; [debug line = 119:2]
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext false, i4 zeroext 2), !dbg !178 ; [debug line = 120:2]
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext false, i4 zeroext -8), !dbg !179 ; [debug line = 121:2]
  call fastcc void @lcddrv_wait_tmr(i25 10000) nounwind, !dbg !180 ; [debug line = 122:2]
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext false, i4 zeroext 0), !dbg !181 ; [debug line = 124:2]
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext false, i4 zeroext -4), !dbg !182 ; [debug line = 125:2]
  call fastcc void @lcddrv_wait_tmr(i25 10000) nounwind, !dbg !183 ; [debug line = 126:2]
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext false, i4 zeroext 0), !dbg !184 ; [debug line = 128:2]
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext false, i4 zeroext 1), !dbg !185 ; [debug line = 129:2]
  call fastcc void @lcddrv_wait_tmr(i25 200000) nounwind, !dbg !186 ; [debug line = 130:2]
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext false, i4 zeroext 0), !dbg !187 ; [debug line = 132:2]
  call fastcc void @lcddrv_lcd_send_cmd(i1 zeroext false, i4 zeroext 2), !dbg !188 ; [debug line = 133:2]
  call fastcc void @lcddrv_wait_tmr(i25 10000) nounwind, !dbg !189 ; [debug line = 134:2]
  ret void, !dbg !190                             ; [debug line = 135:1]
}

; [#uses=2]
define weak void @_ssdm_op_Write.ap_none.volatile.i5P(i5*, i5) {
entry:
  store i5 %1, i5* %0
  ret void
}

; [#uses=3]
define weak void @_ssdm_op_Write.ap_none.volatile.i4P(i4*, i4) {
entry:
  store i4 %1, i4* %0
  ret void
}

; [#uses=11]
define weak void @_ssdm_op_Write.ap_none.volatile.i1P(i1*, i1) {
entry:
  store i1 %1, i1* %0
  ret void
}

; [#uses=15]
define weak void @_ssdm_op_Wait(...) nounwind {
entry:
  ret void
}

; [#uses=1]
define weak void @_ssdm_op_SpecTopModule(...) nounwind {
entry:
  ret void
}

; [#uses=1]
define weak i32 @_ssdm_op_SpecRegionEnd(...) {
entry:
  ret i32 0
}

; [#uses=1]
define weak i32 @_ssdm_op_SpecRegionBegin(...) {
entry:
  ret i32 0
}

; [#uses=3]
define weak i32 @_ssdm_op_SpecLoopTripCount(...) {
entry:
  ret i32 0
}

; [#uses=1]
define weak i32 @_ssdm_op_SpecLoopBegin(...) {
entry:
  ret i32 0
}

; [#uses=9]
define weak void @_ssdm_op_SpecInterface(...) nounwind {
entry:
  ret void
}

; [#uses=2]
define weak i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8*) {
entry:
  %empty = load i8* %0                            ; [#uses=1 type=i8]
  ret i8 %empty
}

; [#uses=1]
define weak i5 @_ssdm_op_Read.ap_none.volatile.i5P(i5*) {
entry:
  %empty = load i5* %0                            ; [#uses=1 type=i5]
  ret i5 %empty
}

; [#uses=1]
define weak i4 @_ssdm_op_Read.ap_none.volatile.i4P(i4*) {
entry:
  %empty = load i4* %0                            ; [#uses=1 type=i4]
  ret i4 %empty
}

; [#uses=8]
define weak i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1*) {
entry:
  %empty = load i1* %0                            ; [#uses=1 type=i1]
  ret i1 %empty
}

; [#uses=1]
define weak i5 @_ssdm_op_Read.ap_auto.i5(i5) {
entry:
  ret i5 %0
}

; [#uses=1]
define weak i4 @_ssdm_op_Read.ap_auto.i4(i4) {
entry:
  ret i4 %0
}

; [#uses=1]
define weak i25 @_ssdm_op_Read.ap_auto.i25(i25) {
entry:
  ret i25 %0
}

; [#uses=1]
define weak i1 @_ssdm_op_Read.ap_auto.i1(i1) {
entry:
  ret i1 %0
}

; [#uses=0]
declare i5 @_ssdm_op_PartSelect.i5.i6.i32.i32(i6, i32, i32) nounwind readnone

; [#uses=2]
define weak i4 @_ssdm_op_PartSelect.i4.i8.i32.i32(i8, i32, i32) nounwind readnone {
entry:
  %empty = call i8 @llvm.part.select.i8(i8 %0, i32 %1, i32 %2) ; [#uses=1 type=i8]
  %empty_6 = trunc i8 %empty to i4                ; [#uses=1 type=i4]
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
!47 = metadata !{i32 786689, metadata !48, metadata !"tmr", metadata !49, i32 16777281, metadata !52, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!48 = metadata !{i32 786478, i32 0, metadata !49, metadata !"wait_tmr", metadata !"wait_tmr", metadata !"", metadata !49, i32 65, metadata !50, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, metadata !54, i32 66} ; [ DW_TAG_subprogram ]
!49 = metadata !{i32 786473, metadata !"lcddrv.c", metadata !"D:\5C21_streamer_car5_artix7\5Cfpga_arty", null} ; [ DW_TAG_file_type ]
!50 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !51, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!51 = metadata !{null, metadata !52}
!52 = metadata !{i32 786454, null, metadata !"uint32", metadata !49, i32 34, i64 0, i64 0, i64 0, i32 0, metadata !53} ; [ DW_TAG_typedef ]
!53 = metadata !{i32 786468, null, metadata !"uint32", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!54 = metadata !{metadata !55}
!55 = metadata !{i32 786468}                      ; [ DW_TAG_base_type ]
!56 = metadata !{i32 65, i32 22, metadata !48, null}
!57 = metadata !{i32 69, i32 2, metadata !58, null}
!58 = metadata !{i32 786443, metadata !48, i32 66, i32 1, metadata !49, i32 0} ; [ DW_TAG_lexical_block ]
!59 = metadata !{i32 70, i32 7, metadata !60, null}
!60 = metadata !{i32 786443, metadata !58, i32 70, i32 2, metadata !49, i32 1} ; [ DW_TAG_lexical_block ]
!61 = metadata !{i32 70, i32 23, metadata !60, null}
!62 = metadata !{i32 71, i32 3, metadata !63, null}
!63 = metadata !{i32 786443, metadata !60, i32 70, i32 28, metadata !49, i32 2} ; [ DW_TAG_lexical_block ]
!64 = metadata !{i32 786688, metadata !58, metadata !"t", metadata !49, i32 68, metadata !52, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!65 = metadata !{i32 73, i32 2, metadata !58, null}
!66 = metadata !{i32 74, i32 1, metadata !58, null}
!67 = metadata !{i32 786689, metadata !68, metadata !"addr", metadata !49, i32 16777354, metadata !73, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!68 = metadata !{i32 786478, i32 0, metadata !49, metadata !"read_mem", metadata !"read_mem", metadata !"", metadata !49, i32 138, metadata !69, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i8 (i5)* @lcddrv_read_mem, null, null, metadata !54, i32 139} ; [ DW_TAG_subprogram ]
!69 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !70, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!70 = metadata !{metadata !71, metadata !73}
!71 = metadata !{i32 786454, null, metadata !"uint8", metadata !49, i32 10, i64 0, i64 0, i64 0, i32 0, metadata !72} ; [ DW_TAG_typedef ]
!72 = metadata !{i32 786468, null, metadata !"uint8", null, i32 0, i64 8, i64 8, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!73 = metadata !{i32 786454, null, metadata !"uint5", metadata !49, i32 7, i64 0, i64 0, i64 0, i32 0, metadata !74} ; [ DW_TAG_typedef ]
!74 = metadata !{i32 786468, null, metadata !"uint5", null, i32 0, i64 5, i64 8, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!75 = metadata !{i32 138, i32 22, metadata !68, null}
!76 = metadata !{i32 143, i32 2, metadata !77, null}
!77 = metadata !{i32 786443, metadata !68, i32 139, i32 1, metadata !49, i32 5} ; [ DW_TAG_lexical_block ]
!78 = metadata !{i32 144, i32 2, metadata !77, null}
!79 = metadata !{i32 145, i32 2, metadata !77, null}
!80 = metadata !{i32 146, i32 2, metadata !77, null}
!81 = metadata !{i32 147, i32 2, metadata !77, null}
!82 = metadata !{i32 148, i32 3, metadata !83, null}
!83 = metadata !{i32 786443, metadata !77, i32 147, i32 5, metadata !49, i32 6} ; [ DW_TAG_lexical_block ]
!84 = metadata !{i32 149, i32 3, metadata !83, null}
!85 = metadata !{i32 150, i32 3, metadata !83, null}
!86 = metadata !{i32 786688, metadata !77, metadata !"dt", metadata !49, i32 141, metadata !71, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!87 = metadata !{i32 151, i32 2, metadata !83, null}
!88 = metadata !{i32 152, i32 2, metadata !77, null}
!89 = metadata !{i32 153, i32 2, metadata !77, null}
!90 = metadata !{i32 154, i32 2, metadata !77, null}
!91 = metadata !{i32 156, i32 2, metadata !77, null}
!92 = metadata !{i32 162, i32 1, metadata !93, null}
!93 = metadata !{i32 786443, metadata !94, i32 161, i32 1, metadata !49, i32 7} ; [ DW_TAG_lexical_block ]
!94 = metadata !{i32 786478, i32 0, metadata !49, metadata !"lcddrv", metadata !"lcddrv", metadata !"", metadata !49, i32 160, metadata !95, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 false, void ()* @lcddrv, null, null, metadata !54, i32 161} ; [ DW_TAG_subprogram ]
!95 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !96, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!96 = metadata !{null}
!97 = metadata !{i32 163, i32 1, metadata !93, null}
!98 = metadata !{i32 165, i32 1, metadata !93, null}
!99 = metadata !{i32 166, i32 1, metadata !93, null}
!100 = metadata !{i32 167, i32 1, metadata !93, null}
!101 = metadata !{i32 168, i32 1, metadata !93, null}
!102 = metadata !{i32 170, i32 1, metadata !93, null}
!103 = metadata !{i32 171, i32 1, metadata !93, null}
!104 = metadata !{i32 172, i32 1, metadata !93, null}
!105 = metadata !{i32 173, i32 1, metadata !93, null}
!106 = metadata !{i32 178, i32 2, metadata !93, null}
!107 = metadata !{i32 180, i32 2, metadata !93, null}
!108 = metadata !{i32 180, i32 13, metadata !109, null}
!109 = metadata !{i32 786443, metadata !93, i32 180, i32 12, metadata !49, i32 8} ; [ DW_TAG_lexical_block ]
!110 = metadata !{i32 182, i32 3, metadata !109, null}
!111 = metadata !{i32 183, i32 3, metadata !109, null}
!112 = metadata !{i32 185, i32 8, metadata !113, null}
!113 = metadata !{i32 786443, metadata !109, i32 185, i32 3, metadata !49, i32 9} ; [ DW_TAG_lexical_block ]
!114 = metadata !{i32 185, i32 27, metadata !113, null}
!115 = metadata !{i32 186, i32 9, metadata !116, null}
!116 = metadata !{i32 786443, metadata !113, i32 185, i32 34, metadata !49, i32 10} ; [ DW_TAG_lexical_block ]
!117 = metadata !{i32 786688, metadata !93, metadata !"dt", metadata !49, i32 176, metadata !71, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!118 = metadata !{i32 187, i32 4, metadata !116, null}
!119 = metadata !{i32 188, i32 4, metadata !116, null}
!120 = metadata !{i32 786688, metadata !93, metadata !"pos", metadata !49, i32 175, metadata !71, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!121 = metadata !{i32 192, i32 3, metadata !109, null}
!122 = metadata !{i32 193, i32 3, metadata !109, null}
!123 = metadata !{i32 195, i32 8, metadata !124, null}
!124 = metadata !{i32 786443, metadata !109, i32 195, i32 3, metadata !49, i32 11} ; [ DW_TAG_lexical_block ]
!125 = metadata !{i32 196, i32 9, metadata !126, null}
!126 = metadata !{i32 786443, metadata !124, i32 195, i32 35, metadata !49, i32 12} ; [ DW_TAG_lexical_block ]
!127 = metadata !{i32 197, i32 4, metadata !126, null}
!128 = metadata !{i32 198, i32 4, metadata !126, null}
!129 = metadata !{i32 195, i32 28, metadata !124, null}
!130 = metadata !{i32 202, i32 3, metadata !109, null}
!131 = metadata !{i32 203, i32 3, metadata !109, null}
!132 = metadata !{i32 204, i32 2, metadata !109, null}
!133 = metadata !{i32 786689, metadata !134, metadata !"wd", metadata !49, i32 33554509, metadata !139, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!134 = metadata !{i32 786478, i32 0, metadata !49, metadata !"lcd_send_cmd", metadata !"lcd_send_cmd", metadata !"", metadata !49, i32 77, metadata !135, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (i1, i4)* @lcddrv_lcd_send_cmd, null, null, metadata !54, i32 78} ; [ DW_TAG_subprogram ]
!135 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !136, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!136 = metadata !{null, metadata !137, metadata !139}
!137 = metadata !{i32 786454, null, metadata !"uint1", metadata !49, i32 3, i64 0, i64 0, i64 0, i32 0, metadata !138} ; [ DW_TAG_typedef ]
!138 = metadata !{i32 786468, null, metadata !"uint1", null, i32 0, i64 1, i64 1, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!139 = metadata !{i32 786454, null, metadata !"uint4", metadata !49, i32 6, i64 0, i64 0, i64 0, i32 0, metadata !140} ; [ DW_TAG_typedef ]
!140 = metadata !{i32 786468, null, metadata !"uint4", null, i32 0, i64 4, i64 4, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!141 = metadata !{i32 77, i32 37, metadata !134, null}
!142 = metadata !{i32 786689, metadata !134, metadata !"mode", metadata !49, i32 16777293, metadata !137, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!143 = metadata !{i32 77, i32 25, metadata !134, null}
!144 = metadata !{i32 80, i32 2, metadata !145, null}
!145 = metadata !{i32 786443, metadata !134, i32 78, i32 1, metadata !49, i32 3} ; [ DW_TAG_lexical_block ]
!146 = metadata !{i32 81, i32 2, metadata !145, null}
!147 = metadata !{i32 82, i32 2, metadata !145, null}
!148 = metadata !{i32 83, i32 2, metadata !145, null}
!149 = metadata !{i32 85, i32 2, metadata !145, null}
!150 = metadata !{i32 86, i32 2, metadata !145, null}
!151 = metadata !{i32 87, i32 2, metadata !145, null}
!152 = metadata !{i32 88, i32 2, metadata !145, null}
!153 = metadata !{i32 89, i32 2, metadata !145, null}
!154 = metadata !{i32 90, i32 2, metadata !145, null}
!155 = metadata !{i32 92, i32 2, metadata !145, null}
!156 = metadata !{i32 93, i32 2, metadata !145, null}
!157 = metadata !{i32 94, i32 2, metadata !145, null}
!158 = metadata !{i32 95, i32 2, metadata !145, null}
!159 = metadata !{i32 96, i32 2, metadata !145, null}
!160 = metadata !{i32 97, i32 2, metadata !145, null}
!161 = metadata !{i32 99, i32 2, metadata !145, null}
!162 = metadata !{i32 100, i32 2, metadata !145, null}
!163 = metadata !{i32 101, i32 2, metadata !145, null}
!164 = metadata !{i32 102, i32 2, metadata !145, null}
!165 = metadata !{i32 103, i32 2, metadata !145, null}
!166 = metadata !{i32 104, i32 2, metadata !145, null}
!167 = metadata !{i32 105, i32 1, metadata !145, null}
!168 = metadata !{i32 111, i32 2, metadata !169, null}
!169 = metadata !{i32 786443, metadata !170, i32 109, i32 1, metadata !49, i32 4} ; [ DW_TAG_lexical_block ]
!170 = metadata !{i32 786478, i32 0, metadata !49, metadata !"init_lcd", metadata !"init_lcd", metadata !"", metadata !49, i32 108, metadata !95, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 false, void ()* @lcddrv_init_lcd, null, null, metadata !54, i32 109} ; [ DW_TAG_subprogram ]
!171 = metadata !{i32 112, i32 2, metadata !169, null}
!172 = metadata !{i32 113, i32 2, metadata !169, null}
!173 = metadata !{i32 114, i32 2, metadata !169, null}
!174 = metadata !{i32 115, i32 2, metadata !169, null}
!175 = metadata !{i32 116, i32 2, metadata !169, null}
!176 = metadata !{i32 117, i32 2, metadata !169, null}
!177 = metadata !{i32 119, i32 2, metadata !169, null}
!178 = metadata !{i32 120, i32 2, metadata !169, null}
!179 = metadata !{i32 121, i32 2, metadata !169, null}
!180 = metadata !{i32 122, i32 2, metadata !169, null}
!181 = metadata !{i32 124, i32 2, metadata !169, null}
!182 = metadata !{i32 125, i32 2, metadata !169, null}
!183 = metadata !{i32 126, i32 2, metadata !169, null}
!184 = metadata !{i32 128, i32 2, metadata !169, null}
!185 = metadata !{i32 129, i32 2, metadata !169, null}
!186 = metadata !{i32 130, i32 2, metadata !169, null}
!187 = metadata !{i32 132, i32 2, metadata !169, null}
!188 = metadata !{i32 133, i32 2, metadata !169, null}
!189 = metadata !{i32 134, i32 2, metadata !169, null}
!190 = metadata !{i32 135, i32 1, metadata !169, null}
