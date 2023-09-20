; ModuleID = 'D:/21_streamer_car5_artix7/fpga_arty/i2c_slave_core/solution1/.autopilot/db/a.o.3.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-w64-mingw32"

@mem_wreq = global i1 false, align 1              ; [#uses=6 type=i1*]
@mem_wack = common global i1 false, align 1       ; [#uses=3 type=i1*]
@mem_rreq = global i1 false, align 1              ; [#uses=6 type=i1*]
@mem_rack = common global i1 false, align 1       ; [#uses=3 type=i1*]
@mem_dout = global i8 0, align 1                  ; [#uses=6 type=i8*]
@mem_din = common global i8 0, align 1            ; [#uses=3 type=i8*]
@mem_addr = global i8 0, align 1                  ; [#uses=9 type=i8*]
@i2c_val = common global i2 0, align 1            ; [#uses=35 type=i2*]
@i2c_sda_out = global i1 true, align 1            ; [#uses=10 type=i1*]
@i2c_sda_oe = global i1 false, align 1            ; [#uses=13 type=i1*]
@i2c_in = common global i2 0, align 1             ; [#uses=32 type=i2*]
@dev_addr_in = common global i7 0, align 1        ; [#uses=4 type=i7*]
@auto_inc_regad_in = common global i1 false, align 1 ; [#uses=5 type=i1*]
@p_str6 = private unnamed_addr constant [12 x i8] c"hls_label_4\00", align 1 ; [#uses=2 type=[12 x i8]*]
@p_str5 = private unnamed_addr constant [17 x i8] c"label_read_start\00", align 1 ; [#uses=1 type=[17 x i8]*]
@p_str1 = private unnamed_addr constant [8 x i8] c"ap_none\00", align 1 ; [#uses=12 type=[8 x i8]*]
@p_str = private unnamed_addr constant [1 x i8] zeroinitializer, align 1 ; [#uses=48 type=[1 x i8]*]

; [#uses=1]
define internal fastcc void @i2c_slave_core_write_mem(i8 zeroext %addr, i8 zeroext %data) nounwind uwtable {
  %data_read = call i8 @_ssdm_op_Read.ap_auto.i8(i8 %data) nounwind ; [#uses=3 type=i8]
  call void @llvm.dbg.value(metadata !{i8 %data_read}, i64 0, metadata !67), !dbg !76 ; [debug line = 80:34] [debug variable = data]
  %addr_read = call i8 @_ssdm_op_Read.ap_auto.i8(i8 %addr) nounwind ; [#uses=3 type=i8]
  call void @llvm.dbg.value(metadata !{i8 %addr_read}, i64 0, metadata !77), !dbg !78 ; [debug line = 80:22] [debug variable = addr]
  call void @llvm.dbg.value(metadata !{i8 %addr}, i64 0, metadata !77), !dbg !78 ; [debug line = 80:22] [debug variable = addr]
  call void @llvm.dbg.value(metadata !{i8 %data}, i64 0, metadata !67), !dbg !76 ; [debug line = 80:34] [debug variable = data]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !79 ; [debug line = 83:2]
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @mem_addr, i8 %addr_read) nounwind, !dbg !81 ; [debug line = 84:2]
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @mem_dout, i8 %data_read) nounwind, !dbg !82 ; [debug line = 85:2]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @mem_wreq, i1 false) nounwind, !dbg !83 ; [debug line = 86:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !84 ; [debug line = 87:2]
  br label %._crit_edge, !dbg !85                 ; [debug line = 89:2]

._crit_edge:                                      ; preds = %._crit_edge, %0
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @mem_addr, i8 %addr_read) nounwind, !dbg !86 ; [debug line = 90:3]
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @mem_dout, i8 %data_read) nounwind, !dbg !88 ; [debug line = 91:3]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @mem_wreq, i1 true) nounwind, !dbg !89 ; [debug line = 92:3]
  %mem_wack_read = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @mem_wack) nounwind, !dbg !90 ; [#uses=1 type=i1] [debug line = 93:2]
  br i1 %mem_wack_read, label %1, label %._crit_edge, !dbg !90 ; [debug line = 93:2]

; <label>:1                                       ; preds = %._crit_edge
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !91 ; [debug line = 94:2]
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @mem_addr, i8 %addr_read) nounwind, !dbg !92 ; [debug line = 96:2]
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @mem_dout, i8 %data_read) nounwind, !dbg !93 ; [debug line = 97:2]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @mem_wreq, i1 false) nounwind, !dbg !94 ; [debug line = 98:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !95 ; [debug line = 99:2]
  ret void, !dbg !96                              ; [debug line = 100:1]
}

; [#uses=2]
define internal fastcc zeroext i8 @i2c_slave_core_read_mem(i8 zeroext %addr) nounwind uwtable {
  %addr_read = call i8 @_ssdm_op_Read.ap_auto.i8(i8 %addr) nounwind ; [#uses=3 type=i8]
  call void @llvm.dbg.value(metadata !{i8 %addr_read}, i64 0, metadata !97), !dbg !101 ; [debug line = 103:22] [debug variable = addr]
  call void @llvm.dbg.value(metadata !{i8 %addr}, i64 0, metadata !97), !dbg !101 ; [debug line = 103:22] [debug variable = addr]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !102 ; [debug line = 108:2]
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @mem_addr, i8 %addr_read) nounwind, !dbg !104 ; [debug line = 109:2]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @mem_rreq, i1 true) nounwind, !dbg !105 ; [debug line = 110:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !106 ; [debug line = 111:2]
  br label %._crit_edge, !dbg !107                ; [debug line = 113:2]

._crit_edge:                                      ; preds = %._crit_edge, %0
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @mem_addr, i8 %addr_read) nounwind, !dbg !108 ; [debug line = 114:3]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @mem_rreq, i1 true) nounwind, !dbg !110 ; [debug line = 115:3]
  %dt = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @mem_din) nounwind, !dbg !111 ; [#uses=1 type=i8] [debug line = 116:3]
  call void @llvm.dbg.value(metadata !{i8 %dt}, i64 0, metadata !112), !dbg !111 ; [debug line = 116:3] [debug variable = dt]
  %mem_rack_read = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @mem_rack) nounwind, !dbg !113 ; [#uses=1 type=i1] [debug line = 117:2]
  br i1 %mem_rack_read, label %1, label %._crit_edge, !dbg !113 ; [debug line = 117:2]

; <label>:1                                       ; preds = %._crit_edge
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !114 ; [debug line = 118:2]
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @mem_addr, i8 %addr_read) nounwind, !dbg !115 ; [debug line = 120:2]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @mem_rreq, i1 false) nounwind, !dbg !116 ; [debug line = 121:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !117 ; [debug line = 122:2]
  ret i8 %dt, !dbg !118                           ; [debug line = 124:2]
}

; [#uses=73]
declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

; [#uses=3]
declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

; [#uses=0]
define void @i2c_slave_core() noreturn nounwind uwtable {
  %dev_addr_2 = alloca i7                         ; [#uses=4 type=i7*]
  call void @llvm.dbg.declare(metadata !{i7* %dev_addr_2}, metadata !119) ; [debug variable = dev_addr]
  %reg_addr_4 = alloca i8                         ; [#uses=5 type=i8*]
  call void @llvm.dbg.declare(metadata !{i8* %reg_addr_4}, metadata !126) ; [debug variable = reg_addr]
  %p_Val2_33 = alloca i8                          ; [#uses=5 type=i8*]
  call void @llvm.dbg.declare(metadata !{i8* %p_Val2_33}, metadata !127) ; [debug variable = __Val2__]
  call void (...)* @_ssdm_op_SpecTopModule(), !dbg !134 ; [debug line = 133:1]
  %i2c_in_load = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind, !dbg !135 ; [#uses=0 type=i2] [debug line = 134:1]
  call void (...)* @_ssdm_op_SpecInterface(i2* @i2c_in, [8 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !135 ; [debug line = 134:1]
  %i2c_sda_out_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @i2c_sda_out) nounwind, !dbg !136 ; [#uses=0 type=i1] [debug line = 135:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @i2c_sda_out, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !136 ; [debug line = 135:1]
  %i2c_sda_oe_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @i2c_sda_oe) nounwind, !dbg !137 ; [#uses=0 type=i1] [debug line = 136:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @i2c_sda_oe, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !137 ; [debug line = 136:1]
  %dev_addr_in_load = call i7 @_ssdm_op_Read.ap_none.volatile.i7P(i7* @dev_addr_in) nounwind, !dbg !138 ; [#uses=0 type=i7] [debug line = 138:1]
  call void (...)* @_ssdm_op_SpecInterface(i7* @dev_addr_in, [8 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !138 ; [debug line = 138:1]
  %auto_inc_regad_in_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @auto_inc_regad_in) nounwind, !dbg !139 ; [#uses=0 type=i1] [debug line = 139:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @auto_inc_regad_in, [8 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !139 ; [debug line = 139:1]
  %mem_addr_load = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @mem_addr) nounwind, !dbg !140 ; [#uses=0 type=i8] [debug line = 141:1]
  call void (...)* @_ssdm_op_SpecInterface(i8* @mem_addr, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !140 ; [debug line = 141:1]
  %mem_din_load = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @mem_din) nounwind, !dbg !141 ; [#uses=0 type=i8] [debug line = 142:1]
  call void (...)* @_ssdm_op_SpecInterface(i8* @mem_din, [8 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !141 ; [debug line = 142:1]
  %mem_dout_load = call i8 @_ssdm_op_Read.ap_none.volatile.i8P(i8* @mem_dout) nounwind, !dbg !142 ; [#uses=0 type=i8] [debug line = 143:1]
  call void (...)* @_ssdm_op_SpecInterface(i8* @mem_dout, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !142 ; [debug line = 143:1]
  %mem_wreq_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @mem_wreq) nounwind, !dbg !143 ; [#uses=0 type=i1] [debug line = 144:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @mem_wreq, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !143 ; [debug line = 144:1]
  %mem_wack_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @mem_wack) nounwind, !dbg !144 ; [#uses=0 type=i1] [debug line = 145:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @mem_wack, [8 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !144 ; [debug line = 145:1]
  %mem_rreq_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @mem_rreq) nounwind, !dbg !145 ; [#uses=0 type=i1] [debug line = 146:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @mem_rreq, [8 x i8]* @p_str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !145 ; [debug line = 146:1]
  %mem_rack_load = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @mem_rack) nounwind, !dbg !146 ; [#uses=0 type=i1] [debug line = 147:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @mem_rack, [8 x i8]* @p_str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str, [1 x i8]* @p_str, [1 x i8]* @p_str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @p_str) nounwind, !dbg !146 ; [debug line = 147:1]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !147 ; [debug line = 158:2]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @i2c_sda_out, i1 true) nounwind, !dbg !148 ; [debug line = 159:2]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @i2c_sda_oe, i1 false) nounwind, !dbg !149 ; [debug line = 160:2]
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @mem_addr, i8 0) nounwind, !dbg !150 ; [debug line = 161:2]
  call void @_ssdm_op_Write.ap_none.volatile.i8P(i8* @mem_dout, i8 0) nounwind, !dbg !151 ; [debug line = 162:2]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @mem_wreq, i1 false) nounwind, !dbg !152 ; [debug line = 163:2]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @mem_rreq, i1 false) nounwind, !dbg !153 ; [debug line = 164:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !154 ; [debug line = 165:2]
  br label %.backedge, !dbg !155                  ; [debug line = 168:13]

.backedge.loopexit:                               ; preds = %._crit_edge101
  store i2 %p_Val2_65, i2* @i2c_val, align 1, !dbg !156 ; [debug line = 76:2@438:6]
  store i8 %reg_data_8, i8* %p_Val2_33, !dbg !161 ; [debug line = 435:5]
  store i8 %re_7, i8* %reg_addr_4
  store i7 %de_2, i7* %dev_addr_2, !dbg !162      ; [debug line = 205:195]
  br label %.backedge.backedge

.backedge.loopexit83:                             ; preds = %.preheader
  store i2 %p_Val2_63, i2* @i2c_val, align 1, !dbg !166 ; [debug line = 76:2@430:6]
  store i8 %reg_data_4, i8* %p_Val2_33, !dbg !169 ; [debug line = 404:14]
  store i8 %re_7, i8* %reg_addr_4
  store i7 %de_2, i7* %dev_addr_2, !dbg !162      ; [debug line = 205:195]
  br label %.backedge.backedge

.backedge.loopexit84:                             ; preds = %._crit_edge92
  store i2 %p_Val2_53, i2* @i2c_val, align 1, !dbg !170 ; [debug line = 76:2@325:6]
  store i8 %reg_data_3, i8* %p_Val2_33, !dbg !175 ; [debug line = 320:196]
  store i8 %re_2, i8* %reg_addr_4, !dbg !177      ; [debug line = 256:195]
  br label %.backedge.backedge

.backedge.loopexit88:                             ; preds = %15
  store i2 %p_Val2_48, i2* @i2c_val, align 1, !dbg !181 ; [debug line = 76:2@293:4]
  store i8 %reg_data, i8* %p_Val2_33, !dbg !184   ; [debug line = 289:194]
  br label %.backedge.backedge

.backedge.loopexit102:                            ; preds = %.preheader50
  store i2 %p_Val2_35, i2* @i2c_val, align 1, !dbg !186 ; [debug line = 76:2@191:4]
  br label %.backedge.backedge

.backedge.loopexit104:                            ; preds = %.preheader52
  store i2 %p_Val2_34, i2* @i2c_val, align 1, !dbg !189 ; [debug line = 76:2@184:4]
  br label %.backedge.backedge

.backedge.backedge:                               ; preds = %.backedge.loopexit104, %.backedge.loopexit102, %.backedge.loopexit88, %.backedge.loopexit84, %.backedge.loopexit83, %.backedge.loopexit
  br label %.backedge

.backedge:                                        ; preds = %.backedge.backedge, %0
  %loop_begin = call i32 (...)* @_ssdm_op_SpecLoopBegin() nounwind ; [#uses=0 type=i32]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @i2c_sda_out, i1 true) nounwind, !dbg !192 ; [debug line = 173:3]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @i2c_sda_oe, i1 false) nounwind, !dbg !193 ; [debug line = 174:3]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !194 ; [debug line = 175:3]
  br label %.critedge, !dbg !195                  ; [debug line = 178:3]

.critedge:                                        ; preds = %.critedge.backedge, %.backedge
  %p_Val2_s = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind, !dbg !196 ; [#uses=3 type=i2] [debug line = 76:2@179:4]
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_s}, i64 0, metadata !199), !dbg !203 ; [debug line = 180:51] [debug variable = __Val2__]
  %tmp = trunc i2 %p_Val2_s to i1, !dbg !204      ; [#uses=1 type=i1] [debug line = 180:85]
  br i1 %tmp, label %1, label %.critedge.backedge, !dbg !205 ; [debug line = 180:174]

; <label>:1                                       ; preds = %.critedge
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_s}, i64 0, metadata !206), !dbg !208 ; [debug line = 180:224] [debug variable = __Val2__]
  %tmp_1 = call i1 @_ssdm_op_BitSelect.i1.i2.i32(i2 %p_Val2_s, i32 1), !dbg !209 ; [#uses=1 type=i1] [debug line = 180:0]
  br i1 %tmp_1, label %.preheader52.preheader, label %.critedge.backedge, !dbg !209 ; [debug line = 180:0]

.preheader52.preheader:                           ; preds = %1
  store i2 %p_Val2_s, i2* @i2c_val, align 1, !dbg !196 ; [debug line = 76:2@179:4]
  br label %.preheader52, !dbg !189               ; [debug line = 76:2@184:4]

.critedge.backedge:                               ; preds = %1, %.critedge
  br label %.critedge

.preheader52:                                     ; preds = %2, %.preheader52.preheader
  %p_Val2_34 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind, !dbg !189 ; [#uses=4 type=i2] [debug line = 76:2@184:4]
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_34}, i64 0, metadata !210), !dbg !212 ; [debug line = 185:47] [debug variable = __Val2__]
  %tmp_2 = trunc i2 %p_Val2_34 to i1, !dbg !213   ; [#uses=1 type=i1] [debug line = 185:81]
  br i1 %tmp_2, label %2, label %.backedge.loopexit104, !dbg !214 ; [debug line = 185:170]

; <label>:2                                       ; preds = %.preheader52
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_34}, i64 0, metadata !215), !dbg !217 ; [debug line = 187:51] [debug variable = __Val2__]
  %tmp_4 = call i1 @_ssdm_op_BitSelect.i1.i2.i32(i2 %p_Val2_34, i32 1), !dbg !218 ; [#uses=1 type=i1] [debug line = 187:85]
  br i1 %tmp_4, label %.preheader52, label %.preheader50.preheader, !dbg !219 ; [debug line = 187:174]

.preheader50.preheader:                           ; preds = %2
  store i2 %p_Val2_34, i2* @i2c_val, align 1, !dbg !189 ; [debug line = 76:2@184:4]
  br label %.preheader50, !dbg !186               ; [debug line = 76:2@191:4]

.preheader50:                                     ; preds = %3, %.preheader50.preheader
  %p_Val2_35 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind, !dbg !186 ; [#uses=4 type=i2] [debug line = 76:2@191:4]
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_35}, i64 0, metadata !220), !dbg !222 ; [debug line = 192:47] [debug variable = __Val2__]
  %tmp_5 = call i1 @_ssdm_op_BitSelect.i1.i2.i32(i2 %p_Val2_35, i32 1), !dbg !223 ; [#uses=1 type=i1] [debug line = 192:81]
  br i1 %tmp_5, label %.backedge.loopexit102, label %3, !dbg !224 ; [debug line = 192:170]

; <label>:3                                       ; preds = %.preheader50
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_35}, i64 0, metadata !225), !dbg !227 ; [debug line = 194:51] [debug variable = __Val2__]
  %tmp_8 = trunc i2 %p_Val2_35 to i1, !dbg !228   ; [#uses=1 type=i1] [debug line = 194:85]
  br i1 %tmp_8, label %.preheader50, label %.preheader49, !dbg !229 ; [debug line = 194:174]

.preheader49.loopexit:                            ; preds = %._crit_edge
  store i7 %dev_addr, i7* %dev_addr_2, !dbg !162  ; [debug line = 205:195]
  br label %.preheader49

.preheader49:                                     ; preds = %.preheader49.loopexit, %3
  %storemerge = phi i2 [ %p_Val2_38, %.preheader49.loopexit ], [ %p_Val2_35, %3 ] ; [#uses=1 type=i2]
  %bit_cnt = phi i3 [ %bit_cnt_6, %.preheader49.loopexit ], [ 0, %3 ] ; [#uses=2 type=i3]
  %dev_addr_2_load = load i7* %dev_addr_2         ; [#uses=3 type=i7]
  store i2 %storemerge, i2* @i2c_val, align 1, !dbg !230 ; [debug line = 76:2@209:5]
  %exitcond1 = icmp eq i3 %bit_cnt, -1, !dbg !233 ; [#uses=1 type=i1] [debug line = 199:8]
  %empty = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 7, i64 7, i64 7) nounwind ; [#uses=0 type=i32]
  %bit_cnt_6 = add i3 %bit_cnt, 1, !dbg !234      ; [#uses=1 type=i3] [debug line = 199:34]
  call void @llvm.dbg.value(metadata !{i3 %bit_cnt_6}, i64 0, metadata !235), !dbg !234 ; [debug line = 199:34] [debug variable = bit_cnt]
  br i1 %exitcond1, label %5, label %.preheader48, !dbg !233 ; [debug line = 199:8]

.preheader48:                                     ; preds = %.preheader48, %.preheader49
  %p_Val2_36 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind, !dbg !238 ; [#uses=3 type=i2] [debug line = 76:2@202:5]
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_36}, i64 0, metadata !241), !dbg !243 ; [debug line = 203:52] [debug variable = __Val2__]
  %tmp_9 = trunc i2 %p_Val2_36 to i1, !dbg !244   ; [#uses=1 type=i1] [debug line = 203:86]
  br i1 %tmp_9, label %4, label %.preheader48, !dbg !245 ; [debug line = 203:175]

; <label>:4                                       ; preds = %.preheader48
  store i2 %p_Val2_36, i2* @i2c_val, align 1, !dbg !238 ; [debug line = 76:2@202:5]
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_36}, i64 0, metadata !246), !dbg !247 ; [debug line = 205:72] [debug variable = __Val2__]
  %tmp_13 = call i1 @_ssdm_op_BitSelect.i1.i2.i32(i2 %p_Val2_36, i32 1), !dbg !248 ; [#uses=1 type=i1] [debug line = 205:106]
  %tmp_14 = trunc i7 %dev_addr_2_load to i6       ; [#uses=1 type=i6]
  %dev_addr = call i7 @_ssdm_op_BitConcatenate.i7.i6.i1(i6 %tmp_14, i1 %tmp_13), !dbg !162 ; [#uses=1 type=i7] [debug line = 205:195]
  call void @llvm.dbg.value(metadata !{i7 %dev_addr}, i64 0, metadata !119), !dbg !162 ; [debug line = 205:195] [debug variable = dev_addr]
  br label %._crit_edge, !dbg !249                ; [debug line = 208:4]

._crit_edge:                                      ; preds = %._crit_edge, %4
  %p_Val2_38 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind, !dbg !230 ; [#uses=2 type=i2] [debug line = 76:2@209:5]
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_38}, i64 0, metadata !250), !dbg !252 ; [debug line = 210:52] [debug variable = __Val2__]
  %tmp_16 = trunc i2 %p_Val2_38 to i1, !dbg !253  ; [#uses=1 type=i1] [debug line = 210:86]
  br i1 %tmp_16, label %._crit_edge, label %.preheader49.loopexit, !dbg !254 ; [debug line = 210:175]

; <label>:5                                       ; preds = %.preheader49
  %dev_addr_in_read = call i7 @_ssdm_op_Read.ap_none.volatile.i7P(i7* @dev_addr_in) nounwind, !dbg !255 ; [#uses=1 type=i7] [debug line = 214:3]
  %not_s = icmp ne i7 %dev_addr_2_load, %dev_addr_in_read, !dbg !255 ; [#uses=1 type=i1] [debug line = 214:3]
  br label %._crit_edge84, !dbg !256              ; [debug line = 220:3]

._crit_edge84:                                    ; preds = %._crit_edge84, %5
  %p_Val2_37 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind, !dbg !257 ; [#uses=3 type=i2] [debug line = 76:2@221:4]
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_37}, i64 0, metadata !260), !dbg !262 ; [debug line = 222:51] [debug variable = __Val2__]
  %tmp_12 = trunc i2 %p_Val2_37 to i1, !dbg !263  ; [#uses=1 type=i1] [debug line = 222:85]
  br i1 %tmp_12, label %6, label %._crit_edge84, !dbg !264 ; [debug line = 222:174]

; <label>:6                                       ; preds = %._crit_edge84
  store i2 %p_Val2_37, i2* @i2c_val, align 1, !dbg !257 ; [debug line = 76:2@221:4]
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_37}, i64 0, metadata !265), !dbg !267 ; [debug line = 224:46] [debug variable = __Val2__]
  %tmp_15 = call i1 @_ssdm_op_BitSelect.i1.i2.i32(i2 %p_Val2_37, i32 1), !dbg !268 ; [#uses=1 type=i1] [debug line = 224:80]
  %ignore_0_s = or i1 %not_s, %tmp_15, !dbg !269  ; [#uses=6 type=i1] [debug line = 224:169]
  br label %._crit_edge85, !dbg !270              ; [debug line = 227:3]

._crit_edge85:                                    ; preds = %._crit_edge85, %6
  %p_Val2_39 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind, !dbg !271 ; [#uses=2 type=i2] [debug line = 76:2@228:4]
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_39}, i64 0, metadata !274), !dbg !276 ; [debug line = 229:51] [debug variable = __Val2__]
  %tmp_17 = trunc i2 %p_Val2_39 to i1, !dbg !277  ; [#uses=1 type=i1] [debug line = 229:85]
  br i1 %tmp_17, label %._crit_edge85, label %7, !dbg !278 ; [debug line = 229:174]

; <label>:7                                       ; preds = %._crit_edge85
  store i2 %p_Val2_39, i2* @i2c_val, align 1, !dbg !271 ; [debug line = 76:2@228:4]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !279 ; [debug line = 232:3]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @i2c_sda_out, i1 %ignore_0_s) nounwind, !dbg !280 ; [debug line = 233:3]
  %not_ignore_1 = xor i1 %ignore_0_s, true, !dbg !281 ; [#uses=3 type=i1] [debug line = 234:3]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @i2c_sda_oe, i1 %not_ignore_1) nounwind, !dbg !281 ; [debug line = 234:3]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !282 ; [debug line = 235:3]
  br label %._crit_edge86, !dbg !283              ; [debug line = 237:3]

._crit_edge86:                                    ; preds = %._crit_edge86, %7
  %p_Val2_40 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind, !dbg !284 ; [#uses=2 type=i2] [debug line = 76:2@238:4]
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_40}, i64 0, metadata !287), !dbg !289 ; [debug line = 239:51] [debug variable = __Val2__]
  %tmp_18 = trunc i2 %p_Val2_40 to i1, !dbg !290  ; [#uses=1 type=i1] [debug line = 239:85]
  br i1 %tmp_18, label %.preheader47.preheader, label %._crit_edge86, !dbg !291 ; [debug line = 239:174]

.preheader47.preheader:                           ; preds = %._crit_edge86
  store i2 %p_Val2_40, i2* @i2c_val, align 1, !dbg !284 ; [debug line = 76:2@238:4]
  br label %.preheader47, !dbg !292               ; [debug line = 76:2@242:4]

.preheader47:                                     ; preds = %.preheader47, %.preheader47.preheader
  %p_Val2_41 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind, !dbg !292 ; [#uses=2 type=i2] [debug line = 76:2@242:4]
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_41}, i64 0, metadata !295), !dbg !297 ; [debug line = 243:51] [debug variable = __Val2__]
  %tmp_19 = trunc i2 %p_Val2_41 to i1, !dbg !298  ; [#uses=1 type=i1] [debug line = 243:85]
  br i1 %tmp_19, label %.preheader47, label %8, !dbg !299 ; [debug line = 243:174]

; <label>:8                                       ; preds = %.preheader47
  store i2 %p_Val2_41, i2* @i2c_val, align 1, !dbg !292 ; [debug line = 76:2@242:4]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @i2c_sda_oe, i1 false) nounwind, !dbg !300 ; [debug line = 245:3]
  br label %9, !dbg !301                          ; [debug line = 250:8]

; <label>:9                                       ; preds = %11, %8
  %bit_cnt_1 = phi i4 [ 0, %8 ], [ %bit_cnt_7, %11 ] ; [#uses=2 type=i4]
  %reg_addr_4_load = load i8* %reg_addr_4, !dbg !302 ; [#uses=6 type=i8] [debug line = 407:4]
  %exitcond2 = icmp eq i4 %bit_cnt_1, -8, !dbg !301 ; [#uses=1 type=i1] [debug line = 250:8]
  %empty_3 = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 8, i64 8, i64 8) nounwind ; [#uses=0 type=i32]
  %bit_cnt_7 = add i4 %bit_cnt_1, 1, !dbg !303    ; [#uses=1 type=i4] [debug line = 250:34]
  br i1 %exitcond2, label %12, label %.preheader46, !dbg !301 ; [debug line = 250:8]

.preheader46:                                     ; preds = %.preheader46, %9
  %p_Val2_42 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind, !dbg !304 ; [#uses=3 type=i2] [debug line = 76:2@253:5]
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_42}, i64 0, metadata !307), !dbg !309 ; [debug line = 254:52] [debug variable = __Val2__]
  %tmp_20 = trunc i2 %p_Val2_42 to i1, !dbg !310  ; [#uses=1 type=i1] [debug line = 254:86]
  br i1 %tmp_20, label %10, label %.preheader46, !dbg !311 ; [debug line = 254:175]

; <label>:10                                      ; preds = %.preheader46
  store i2 %p_Val2_42, i2* @i2c_val, align 1, !dbg !304 ; [debug line = 76:2@253:5]
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_42}, i64 0, metadata !312), !dbg !313 ; [debug line = 256:72] [debug variable = __Val2__]
  %tmp_22 = call i1 @_ssdm_op_BitSelect.i1.i2.i32(i2 %p_Val2_42, i32 1), !dbg !314 ; [#uses=1 type=i1] [debug line = 256:106]
  %tmp_23 = trunc i8 %reg_addr_4_load to i7       ; [#uses=1 type=i7]
  %reg_addr = call i8 @_ssdm_op_BitConcatenate.i8.i7.i1(i7 %tmp_23, i1 %tmp_22), !dbg !177 ; [#uses=1 type=i8] [debug line = 256:195]
  call void @llvm.dbg.value(metadata !{i8 %reg_addr}, i64 0, metadata !126), !dbg !177 ; [debug line = 256:195] [debug variable = reg_addr]
  br label %._crit_edge87, !dbg !315              ; [debug line = 259:4]

._crit_edge87:                                    ; preds = %._crit_edge87, %10
  %p_Val2_44 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind, !dbg !316 ; [#uses=2 type=i2] [debug line = 76:2@260:5]
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_44}, i64 0, metadata !319), !dbg !321 ; [debug line = 261:52] [debug variable = __Val2__]
  %tmp_24 = trunc i2 %p_Val2_44 to i1, !dbg !322  ; [#uses=1 type=i1] [debug line = 261:86]
  br i1 %tmp_24, label %._crit_edge87, label %11, !dbg !323 ; [debug line = 261:175]

; <label>:11                                      ; preds = %._crit_edge87
  store i2 %p_Val2_44, i2* @i2c_val, align 1, !dbg !316 ; [debug line = 76:2@260:5]
  call void @llvm.dbg.value(metadata !{i4 %bit_cnt_7}, i64 0, metadata !235), !dbg !303 ; [debug line = 250:34] [debug variable = bit_cnt]
  store i8 %reg_addr, i8* %reg_addr_4, !dbg !177  ; [debug line = 256:195]
  br label %9, !dbg !303                          ; [debug line = 250:34]

; <label>:12                                      ; preds = %9
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !324 ; [debug line = 265:3]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @i2c_sda_out, i1 %ignore_0_s) nounwind, !dbg !325 ; [debug line = 266:3]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @i2c_sda_oe, i1 %not_ignore_1) nounwind, !dbg !326 ; [debug line = 267:3]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !327 ; [debug line = 268:3]
  br label %._crit_edge88, !dbg !328              ; [debug line = 270:3]

._crit_edge88:                                    ; preds = %._crit_edge88, %12
  %p_Val2_43 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind, !dbg !329 ; [#uses=2 type=i2] [debug line = 76:2@271:4]
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_43}, i64 0, metadata !332), !dbg !334 ; [debug line = 272:51] [debug variable = __Val2__]
  %tmp_21 = trunc i2 %p_Val2_43 to i1, !dbg !335  ; [#uses=1 type=i1] [debug line = 272:85]
  br i1 %tmp_21, label %.preheader45.preheader, label %._crit_edge88, !dbg !336 ; [debug line = 272:174]

.preheader45.preheader:                           ; preds = %._crit_edge88
  store i2 %p_Val2_43, i2* @i2c_val, align 1, !dbg !329 ; [debug line = 76:2@271:4]
  br label %.preheader45, !dbg !337               ; [debug line = 76:2@275:4]

.preheader45:                                     ; preds = %.preheader45, %.preheader45.preheader
  %p_Val2_45 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind, !dbg !337 ; [#uses=2 type=i2] [debug line = 76:2@275:4]
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_45}, i64 0, metadata !340), !dbg !342 ; [debug line = 276:51] [debug variable = __Val2__]
  %tmp_25 = trunc i2 %p_Val2_45 to i1, !dbg !343  ; [#uses=1 type=i1] [debug line = 276:85]
  br i1 %tmp_25, label %.preheader45, label %13, !dbg !344 ; [debug line = 276:174]

; <label>:13                                      ; preds = %.preheader45
  store i2 %p_Val2_45, i2* @i2c_val, align 1, !dbg !337 ; [debug line = 76:2@275:4]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @i2c_sda_oe, i1 false) nounwind, !dbg !345 ; [debug line = 278:3]
  br label %._crit_edge89, !dbg !346              ; [debug line = 285:3]

._crit_edge89:                                    ; preds = %._crit_edge89, %13
  %p_Val2_46 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind, !dbg !347 ; [#uses=4 type=i2] [debug line = 76:2@286:4]
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_46}, i64 0, metadata !350), !dbg !352 ; [debug line = 287:51] [debug variable = __Val2__]
  %tmp_26 = trunc i2 %p_Val2_46 to i1, !dbg !353  ; [#uses=1 type=i1] [debug line = 287:85]
  br i1 %tmp_26, label %14, label %._crit_edge89, !dbg !354 ; [debug line = 287:174]

; <label>:14                                      ; preds = %._crit_edge89
  %p_Val2_33_load = load i8* %p_Val2_33           ; [#uses=1 type=i8]
  store i2 %p_Val2_46, i2* @i2c_val, align 1, !dbg !347 ; [debug line = 76:2@286:4]
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_46}, i64 0, metadata !355), !dbg !356 ; [debug line = 289:71] [debug variable = __Val2__]
  %tmp_27 = call i1 @_ssdm_op_BitSelect.i1.i2.i32(i2 %p_Val2_46, i32 1), !dbg !357 ; [#uses=1 type=i1] [debug line = 289:105]
  %tmp_28 = trunc i8 %p_Val2_33_load to i7        ; [#uses=1 type=i7]
  %reg_data = call i8 @_ssdm_op_BitConcatenate.i8.i7.i1(i7 %tmp_28, i1 %tmp_27), !dbg !184 ; [#uses=2 type=i8] [debug line = 289:194]
  call void @llvm.dbg.value(metadata !{i8 %reg_data}, i64 0, metadata !358), !dbg !184 ; [debug line = 289:194] [debug variable = reg_data]
  br label %._crit_edge91, !dbg !359              ; [debug line = 291:3]

._crit_edge91:                                    ; preds = %16, %14
  %p_Val2_47 = phi i2 [ %p_Val2_48, %16 ], [ %p_Val2_46, %14 ], !dbg !360 ; [#uses=1 type=i2] [debug line = 292:61]
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_47}, i64 0, metadata !362), !dbg !360 ; [debug line = 292:61] [debug variable = __Val2__]
  %pre_i2c_sda_val = call i1 @_ssdm_op_BitSelect.i1.i2.i32(i2 %p_Val2_47, i32 1), !dbg !363 ; [#uses=2 type=i1] [debug line = 292:95]
  call void @llvm.dbg.value(metadata !{i1 %pre_i2c_sda_val}, i64 0, metadata !364), !dbg !367 ; [debug line = 292:184] [debug variable = pre_i2c_sda_val]
  %p_Val2_48 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind, !dbg !181 ; [#uses=6 type=i2] [debug line = 76:2@293:4]
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_48}, i64 0, metadata !368), !dbg !370 ; [debug line = 295:47] [debug variable = __Val2__]
  %tmp_30 = trunc i2 %p_Val2_48 to i1, !dbg !371  ; [#uses=1 type=i1] [debug line = 295:81]
  br i1 %tmp_30, label %15, label %.preheader40.preheader, !dbg !372 ; [debug line = 295:170]

.preheader40.preheader:                           ; preds = %._crit_edge91
  store i2 %p_Val2_48, i2* @i2c_val, align 1, !dbg !181 ; [debug line = 76:2@293:4]
  br label %.preheader40, !dbg !373               ; [debug line = 314:4]

; <label>:15                                      ; preds = %._crit_edge91
  br i1 %ignore_0_s, label %.backedge.loopexit88, label %16, !dbg !374 ; [debug line = 297:4]

; <label>:16                                      ; preds = %15
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_48}, i64 0, metadata !375), !dbg !377 ; [debug line = 297:62] [debug variable = __Val2__]
  %tmp_31 = call i1 @_ssdm_op_BitSelect.i1.i2.i32(i2 %p_Val2_48, i32 1), !dbg !378 ; [#uses=1 type=i1] [debug line = 297:96]
  %p_not1 = xor i1 %pre_i2c_sda_val, true, !dbg !379 ; [#uses=1 type=i1] [debug line = 297:185]
  %brmerge = or i1 %tmp_31, %p_not1, !dbg !379    ; [#uses=1 type=i1] [debug line = 297:185]
  br i1 %brmerge, label %._crit_edge91, label %.preheader41.preheader, !dbg !379 ; [debug line = 297:185]

.preheader41.preheader:                           ; preds = %16
  store i2 %p_Val2_48, i2* @i2c_val, align 1, !dbg !181 ; [debug line = 76:2@293:4]
  br label %.preheader41, !dbg !380               ; [debug line = 76:2@299:6]

.preheader41:                                     ; preds = %.preheader41, %.preheader41.preheader
  %p_Val2_49 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind, !dbg !380 ; [#uses=2 type=i2] [debug line = 76:2@299:6]
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_49}, i64 0, metadata !384), !dbg !386 ; [debug line = 300:53] [debug variable = __Val2__]
  %tmp_33 = trunc i2 %p_Val2_49 to i1, !dbg !387  ; [#uses=1 type=i1] [debug line = 300:87]
  br i1 %tmp_33, label %.preheader41, label %.preheader34, !dbg !388 ; [debug line = 300:176]

.preheader40:                                     ; preds = %23, %22, %.preheader40.preheader
  %bit_cnt_2 = phi i1 [ true, %.preheader40.preheader ], [ false, %23 ], [ false, %22 ] ; [#uses=1 type=i1]
  %reg_data_1 = phi i8 [ %reg_data, %.preheader40.preheader ], [ %reg_data_2, %23 ], [ %reg_data_2, %22 ] ; [#uses=1 type=i8]
  %re_2 = phi i8 [ %reg_addr_4_load, %.preheader40.preheader ], [ %p_re_2, %23 ], [ %re_2, %22 ] ; [#uses=5 type=i8]
  %bit_cnt_2_cast = zext i1 %bit_cnt_2 to i4, !dbg !373 ; [#uses=1 type=i4] [debug line = 314:4]
  br label %17, !dbg !373                         ; [debug line = 314:4]

; <label>:17                                      ; preds = %20, %.preheader40
  %bit_cnt_3 = phi i4 [ %bit_cnt_2_cast, %.preheader40 ], [ %bit_cnt_8, %20 ] ; [#uses=2 type=i4]
  %reg_data_2 = phi i8 [ %reg_data_1, %.preheader40 ], [ %reg_data_3, %20 ] ; [#uses=4 type=i8]
  %tmp_32 = call i1 @_ssdm_op_BitSelect.i1.i4.i32(i4 %bit_cnt_3, i32 3), !dbg !373 ; [#uses=1 type=i1] [debug line = 314:4]
  br i1 %tmp_32, label %21, label %.preheader37, !dbg !373 ; [debug line = 314:4]

.preheader37:                                     ; preds = %.preheader37, %17
  %p_Val2_50 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind, !dbg !389 ; [#uses=4 type=i2] [debug line = 76:2@317:6]
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_50}, i64 0, metadata !392), !dbg !394 ; [debug line = 318:53] [debug variable = __Val2__]
  %tmp_34 = trunc i2 %p_Val2_50 to i1, !dbg !395  ; [#uses=1 type=i1] [debug line = 318:87]
  br i1 %tmp_34, label %18, label %.preheader37, !dbg !396 ; [debug line = 318:176]

; <label>:18                                      ; preds = %.preheader37
  store i2 %p_Val2_50, i2* @i2c_val, align 1, !dbg !389 ; [debug line = 76:2@317:6]
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_50}, i64 0, metadata !397), !dbg !398 ; [debug line = 320:73] [debug variable = __Val2__]
  %tmp_36 = call i1 @_ssdm_op_BitSelect.i1.i2.i32(i2 %p_Val2_50, i32 1), !dbg !399 ; [#uses=1 type=i1] [debug line = 320:107]
  %tmp_37 = trunc i8 %reg_data_2 to i7            ; [#uses=1 type=i7]
  %reg_data_3 = call i8 @_ssdm_op_BitConcatenate.i8.i7.i1(i7 %tmp_37, i1 %tmp_36), !dbg !175 ; [#uses=2 type=i8] [debug line = 320:196]
  call void @llvm.dbg.value(metadata !{i8 %reg_data_3}, i64 0, metadata !358), !dbg !175 ; [debug line = 320:196] [debug variable = reg_data]
  br label %._crit_edge92, !dbg !400              ; [debug line = 323:5]

._crit_edge92:                                    ; preds = %19, %18
  %p_Val2_52 = phi i2 [ %p_Val2_53, %19 ], [ %p_Val2_50, %18 ], !dbg !401 ; [#uses=1 type=i2] [debug line = 324:63]
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_52}, i64 0, metadata !403), !dbg !401 ; [debug line = 324:63] [debug variable = __Val2__]
  %tmp_38 = call i1 @_ssdm_op_BitSelect.i1.i2.i32(i2 %p_Val2_52, i32 1), !dbg !404 ; [#uses=1 type=i1] [debug line = 324:97]
  %p_Val2_53 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind, !dbg !170 ; [#uses=5 type=i2] [debug line = 76:2@325:6]
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_53}, i64 0, metadata !405), !dbg !407 ; [debug line = 326:49] [debug variable = __Val2__]
  %tmp_39 = call i1 @_ssdm_op_BitSelect.i1.i2.i32(i2 %p_Val2_53, i32 1), !dbg !408 ; [#uses=1 type=i1] [debug line = 326:83]
  %tmp_6 = xor i1 %tmp_39, true, !dbg !408        ; [#uses=1 type=i1] [debug line = 326:83]
  %brmerge1 = or i1 %tmp_38, %tmp_6, !dbg !409    ; [#uses=1 type=i1] [debug line = 326:172]
  br i1 %brmerge1, label %19, label %.backedge.loopexit84, !dbg !409 ; [debug line = 326:172]

; <label>:19                                      ; preds = %._crit_edge92
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_53}, i64 0, metadata !410), !dbg !412 ; [debug line = 328:53] [debug variable = __Val2__]
  %tmp_43 = trunc i2 %p_Val2_53 to i1, !dbg !413  ; [#uses=1 type=i1] [debug line = 328:87]
  br i1 %tmp_43, label %._crit_edge92, label %20, !dbg !414 ; [debug line = 328:176]

; <label>:20                                      ; preds = %19
  store i2 %p_Val2_53, i2* @i2c_val, align 1, !dbg !170 ; [debug line = 76:2@325:6]
  %bit_cnt_8 = add i4 %bit_cnt_3, 1, !dbg !415    ; [#uses=1 type=i4] [debug line = 330:5]
  call void @llvm.dbg.value(metadata !{i4 %bit_cnt_8}, i64 0, metadata !235), !dbg !415 ; [debug line = 330:5] [debug variable = bit_cnt]
  br label %17, !dbg !416                         ; [debug line = 331:4]

; <label>:21                                      ; preds = %17
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !417 ; [debug line = 334:4]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @i2c_sda_out, i1 %ignore_0_s) nounwind, !dbg !418 ; [debug line = 335:4]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @i2c_sda_oe, i1 %not_ignore_1) nounwind, !dbg !419 ; [debug line = 336:4]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !420 ; [debug line = 337:4]
  br label %._crit_edge93, !dbg !421              ; [debug line = 339:4]

._crit_edge93:                                    ; preds = %._crit_edge93, %21
  %p_Val2_51 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind, !dbg !422 ; [#uses=2 type=i2] [debug line = 76:2@340:5]
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_51}, i64 0, metadata !425), !dbg !427 ; [debug line = 341:52] [debug variable = __Val2__]
  %tmp_35 = trunc i2 %p_Val2_51 to i1, !dbg !428  ; [#uses=1 type=i1] [debug line = 341:86]
  br i1 %tmp_35, label %.preheader35.preheader, label %._crit_edge93, !dbg !429 ; [debug line = 341:175]

.preheader35.preheader:                           ; preds = %._crit_edge93
  store i2 %p_Val2_51, i2* @i2c_val, align 1, !dbg !422 ; [debug line = 76:2@340:5]
  br label %.preheader35, !dbg !430               ; [debug line = 76:2@344:5]

.preheader35:                                     ; preds = %.preheader35, %.preheader35.preheader
  %p_Val2_56 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind, !dbg !430 ; [#uses=2 type=i2] [debug line = 76:2@344:5]
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_56}, i64 0, metadata !433), !dbg !435 ; [debug line = 345:52] [debug variable = __Val2__]
  %tmp_42 = trunc i2 %p_Val2_56 to i1, !dbg !436  ; [#uses=1 type=i1] [debug line = 345:86]
  br i1 %tmp_42, label %.preheader35, label %22, !dbg !437 ; [debug line = 345:175]

; <label>:22                                      ; preds = %.preheader35
  store i2 %p_Val2_56, i2* @i2c_val, align 1, !dbg !430 ; [debug line = 76:2@344:5]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @i2c_sda_oe, i1 false) nounwind, !dbg !438 ; [debug line = 347:4]
  br i1 %ignore_0_s, label %.preheader40, label %23, !dbg !439 ; [debug line = 350:4]

; <label>:23                                      ; preds = %22
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !440 ; [debug line = 351:5]
  call fastcc void @i2c_slave_core_write_mem(i8 zeroext %re_2, i8 zeroext %reg_data_2), !dbg !442 ; [debug line = 352:5]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !443 ; [debug line = 353:5]
  %auto_inc_regad_in_read = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @auto_inc_regad_in) nounwind, !dbg !444 ; [#uses=1 type=i1] [debug line = 354:5]
  %reg_addr_1 = add i8 %re_2, 1, !dbg !445        ; [#uses=1 type=i8] [debug line = 355:6]
  call void @llvm.dbg.value(metadata !{i8 %reg_addr_1}, i64 0, metadata !126), !dbg !445 ; [debug line = 355:6] [debug variable = reg_addr]
  %p_re_2 = select i1 %auto_inc_regad_in_read, i8 %reg_addr_1, i8 %re_2, !dbg !444 ; [#uses=1 type=i8] [debug line = 354:5]
  br label %.preheader40, !dbg !446               ; [debug line = 356:4]

.preheader34:                                     ; preds = %._crit_edge96, %.preheader41
  %storemerge1 = phi i2 [ %p_Val2_49, %.preheader41 ], [ %p_Val2_58, %._crit_edge96 ] ; [#uses=1 type=i2]
  %bit_cnt_4 = phi i3 [ 0, %.preheader41 ], [ %bit_cnt_9, %._crit_edge96 ] ; [#uses=2 type=i3]
  %de_2 = phi i7 [ %dev_addr_2_load, %.preheader41 ], [ %dev_addr_1, %._crit_edge96 ] ; [#uses=4 type=i7]
  store i2 %storemerge1, i2* @i2c_val, align 1, !dbg !447 ; [debug line = 76:2@377:5]
  %exitcond = icmp eq i3 %bit_cnt_4, -1, !dbg !452 ; [#uses=1 type=i1] [debug line = 367:8]
  %empty_4 = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 7, i64 7, i64 7) nounwind ; [#uses=0 type=i32]
  %bit_cnt_9 = add i3 %bit_cnt_4, 1, !dbg !453    ; [#uses=1 type=i3] [debug line = 367:34]
  call void @llvm.dbg.value(metadata !{i3 %bit_cnt_9}, i64 0, metadata !235), !dbg !453 ; [debug line = 367:34] [debug variable = bit_cnt]
  br i1 %exitcond, label %26, label %24, !dbg !452 ; [debug line = 367:8]

; <label>:24                                      ; preds = %.preheader34
  call void (...)* @_ssdm_op_SpecLoopName([17 x i8]* @p_str5) nounwind, !dbg !454 ; [debug line = 367:46]
  br label %._crit_edge95, !dbg !455              ; [debug line = 369:4]

._crit_edge95:                                    ; preds = %._crit_edge95, %24
  %p_Val2_55 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind, !dbg !456 ; [#uses=3 type=i2] [debug line = 76:2@370:5]
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_55}, i64 0, metadata !459), !dbg !461 ; [debug line = 371:52] [debug variable = __Val2__]
  %tmp_41 = trunc i2 %p_Val2_55 to i1, !dbg !462  ; [#uses=1 type=i1] [debug line = 371:86]
  br i1 %tmp_41, label %25, label %._crit_edge95, !dbg !463 ; [debug line = 371:175]

; <label>:25                                      ; preds = %._crit_edge95
  store i2 %p_Val2_55, i2* @i2c_val, align 1, !dbg !456 ; [debug line = 76:2@370:5]
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_55}, i64 0, metadata !464), !dbg !466 ; [debug line = 373:72] [debug variable = __Val2__]
  %tmp_45 = call i1 @_ssdm_op_BitSelect.i1.i2.i32(i2 %p_Val2_55, i32 1), !dbg !467 ; [#uses=1 type=i1] [debug line = 373:106]
  %tmp_46 = trunc i7 %de_2 to i6                  ; [#uses=1 type=i6]
  %dev_addr_1 = call i7 @_ssdm_op_BitConcatenate.i7.i6.i1(i6 %tmp_46, i1 %tmp_45), !dbg !468 ; [#uses=1 type=i7] [debug line = 373:195]
  call void @llvm.dbg.value(metadata !{i7 %dev_addr_1}, i64 0, metadata !119), !dbg !468 ; [debug line = 373:195] [debug variable = dev_addr]
  br label %._crit_edge96, !dbg !469              ; [debug line = 376:4]

._crit_edge96:                                    ; preds = %._crit_edge96, %25
  %p_Val2_58 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind, !dbg !447 ; [#uses=2 type=i2] [debug line = 76:2@377:5]
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_58}, i64 0, metadata !470), !dbg !472 ; [debug line = 378:52] [debug variable = __Val2__]
  %tmp_48 = trunc i2 %p_Val2_58 to i1, !dbg !473  ; [#uses=1 type=i1] [debug line = 378:86]
  br i1 %tmp_48, label %._crit_edge96, label %.preheader34, !dbg !474 ; [debug line = 378:175]

; <label>:26                                      ; preds = %.preheader34
  %dev_addr_in_read_1 = call i7 @_ssdm_op_Read.ap_none.volatile.i7P(i7* @dev_addr_in) nounwind, !dbg !475 ; [#uses=1 type=i7] [debug line = 382:3]
  %not_2 = icmp ne i7 %de_2, %dev_addr_in_read_1, !dbg !475 ; [#uses=2 type=i1] [debug line = 382:3]
  br label %._crit_edge97, !dbg !476              ; [debug line = 388:3]

._crit_edge97:                                    ; preds = %._crit_edge97, %26
  %p_Val2_54 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind, !dbg !477 ; [#uses=3 type=i2] [debug line = 76:2@389:4]
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_54}, i64 0, metadata !480), !dbg !482 ; [debug line = 390:51] [debug variable = __Val2__]
  %tmp_40 = trunc i2 %p_Val2_54 to i1, !dbg !483  ; [#uses=1 type=i1] [debug line = 390:85]
  br i1 %tmp_40, label %27, label %._crit_edge97, !dbg !484 ; [debug line = 390:174]

; <label>:27                                      ; preds = %._crit_edge97
  store i2 %p_Val2_54, i2* @i2c_val, align 1, !dbg !477 ; [debug line = 76:2@389:4]
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_54}, i64 0, metadata !485), !dbg !487 ; [debug line = 392:46] [debug variable = __Val2__]
  %tmp_44 = call i1 @_ssdm_op_BitSelect.i1.i2.i32(i2 %p_Val2_54, i32 1), !dbg !488 ; [#uses=2 type=i1] [debug line = 392:80]
  %tmp_7 = xor i1 %tmp_44, true, !dbg !488        ; [#uses=1 type=i1] [debug line = 392:80]
  %p_ignore_2 = or i1 %not_2, %tmp_7, !dbg !489   ; [#uses=4 type=i1] [debug line = 392:169]
  br label %._crit_edge98, !dbg !490              ; [debug line = 395:3]

._crit_edge98:                                    ; preds = %._crit_edge98, %27
  %p_Val2_57 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind, !dbg !491 ; [#uses=2 type=i2] [debug line = 76:2@396:4]
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_57}, i64 0, metadata !494), !dbg !496 ; [debug line = 397:51] [debug variable = __Val2__]
  %tmp_47 = trunc i2 %p_Val2_57 to i1, !dbg !497  ; [#uses=1 type=i1] [debug line = 397:85]
  br i1 %tmp_47, label %._crit_edge98, label %_ifconv1, !dbg !498 ; [debug line = 397:174]

_ifconv1:                                         ; preds = %._crit_edge98
  store i2 %p_Val2_57, i2* @i2c_val, align 1, !dbg !491 ; [debug line = 76:2@396:4]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !499 ; [debug line = 400:3]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @i2c_sda_out, i1 %p_ignore_2) nounwind, !dbg !500 ; [debug line = 401:3]
  %not_2_not = xor i1 %not_2, true, !dbg !501     ; [#uses=1 type=i1] [debug line = 402:3]
  %not_ignore_3 = and i1 %tmp_44, %not_2_not, !dbg !501 ; [#uses=2 type=i1] [debug line = 402:3]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @i2c_sda_oe, i1 %not_ignore_3) nounwind, !dbg !501 ; [debug line = 402:3]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !502 ; [debug line = 403:3]
  %reg_data_5 = call fastcc zeroext i8 @i2c_slave_core_read_mem(i8 zeroext %reg_addr_4_load), !dbg !169 ; [#uses=1 type=i8] [debug line = 404:14]
  call void @llvm.dbg.value(metadata !{i8 %reg_data_5}, i64 0, metadata !358), !dbg !169 ; [debug line = 404:14] [debug variable = reg_data]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !503 ; [debug line = 405:3]
  %auto_inc_regad_in_read_1 = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @auto_inc_regad_in) nounwind, !dbg !504 ; [#uses=1 type=i1] [debug line = 406:3]
  %reg_addr_2 = add i8 %reg_addr_4_load, 1, !dbg !302 ; [#uses=1 type=i8] [debug line = 407:4]
  call void @llvm.dbg.value(metadata !{i8 %reg_addr_2}, i64 0, metadata !126), !dbg !302 ; [debug line = 407:4] [debug variable = reg_addr]
  %re_1_s = select i1 %p_ignore_2, i8 %reg_addr_4_load, i8 %reg_addr_2, !dbg !504 ; [#uses=1 type=i8] [debug line = 406:3]
  %re_6 = select i1 %auto_inc_regad_in_read_1, i8 %re_1_s, i8 %reg_addr_4_load ; [#uses=1 type=i8]
  br label %._crit_edge100, !dbg !505             ; [debug line = 410:3]

._crit_edge100:                                   ; preds = %._crit_edge100, %_ifconv1
  %p_Val2_59 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind, !dbg !506 ; [#uses=2 type=i2] [debug line = 76:2@411:4]
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_59}, i64 0, metadata !509), !dbg !511 ; [debug line = 412:51] [debug variable = __Val2__]
  %tmp_49 = trunc i2 %p_Val2_59 to i1, !dbg !512  ; [#uses=1 type=i1] [debug line = 412:85]
  br i1 %tmp_49, label %.preheader33.preheader, label %._crit_edge100, !dbg !513 ; [debug line = 412:174]

.preheader33.preheader:                           ; preds = %._crit_edge100
  store i2 %p_Val2_59, i2* @i2c_val, align 1, !dbg !506 ; [debug line = 76:2@411:4]
  br label %.preheader33, !dbg !514               ; [debug line = 76:2@415:4]

.preheader33:                                     ; preds = %.preheader33, %.preheader33.preheader
  %p_Val2_60 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind, !dbg !514 ; [#uses=2 type=i2] [debug line = 76:2@415:4]
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_60}, i64 0, metadata !517), !dbg !519 ; [debug line = 416:51] [debug variable = __Val2__]
  %tmp_50 = trunc i2 %p_Val2_60 to i1, !dbg !520  ; [#uses=1 type=i1] [debug line = 416:85]
  br i1 %tmp_50, label %.preheader33, label %.preheader31.preheader, !dbg !521 ; [debug line = 416:174]

.preheader31.preheader:                           ; preds = %.preheader33
  store i2 %p_Val2_60, i2* @i2c_val, align 1, !dbg !514 ; [debug line = 76:2@415:4]
  br label %.preheader31, !dbg !522               ; [debug line = 421:14]

.preheader31:                                     ; preds = %35, %.preheader31.preheader
  %terminate_read = phi i1 [ %terminate_read_1, %35 ], [ false, %.preheader31.preheader ] ; [#uses=1 type=i1]
  %p_Val2_61 = phi i8 [ %reg_data_6, %35 ], [ %reg_data_5, %.preheader31.preheader ] ; [#uses=2 type=i8]
  %re_7 = phi i8 [ %re_8, %35 ], [ %re_6, %.preheader31.preheader ] ; [#uses=6 type=i8]
  %tmp_3 = call i32 (...)* @_ssdm_op_SpecRegionBegin([12 x i8]* @p_str6) nounwind, !dbg !522 ; [#uses=1 type=i32] [debug line = 421:14]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !523 ; [debug line = 422:4]
  call void @llvm.dbg.value(metadata !{i8 %p_Val2_61}, i64 0, metadata !524), !dbg !526 ; [debug line = 423:59] [debug variable = __Val2__]
  %tmp_51 = call i1 @_ssdm_op_BitSelect.i1.i8.i32(i8 %p_Val2_61, i32 7), !dbg !527 ; [#uses=1 type=i1] [debug line = 423:94]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @i2c_sda_out, i1 %tmp_51) nounwind, !dbg !528 ; [debug line = 423:183]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @i2c_sda_oe, i1 %not_ignore_3) nounwind, !dbg !529 ; [debug line = 424:4]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !530 ; [debug line = 425:4]
  br label %28, !dbg !531                         ; [debug line = 428:9]

; <label>:28                                      ; preds = %._crit_edge102, %.preheader31
  %bit_cnt_5 = phi i4 [ 0, %.preheader31 ], [ %bit_cnt_10, %._crit_edge102 ] ; [#uses=3 type=i4]
  %reg_data_4 = phi i8 [ %p_Val2_61, %.preheader31 ], [ %reg_data_8, %._crit_edge102 ] ; [#uses=2 type=i8]
  %tmp_52 = call i1 @_ssdm_op_BitSelect.i1.i4.i32(i4 %bit_cnt_5, i32 3), !dbg !531 ; [#uses=1 type=i1] [debug line = 428:9]
  %empty_5 = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 1, i64 8, i64 4) nounwind ; [#uses=0 type=i32]
  %bit_cnt_10 = add i4 %bit_cnt_5, 1, !dbg !532   ; [#uses=1 type=i4] [debug line = 428:35]
  br i1 %tmp_52, label %_ifconv, label %.preheader, !dbg !531 ; [debug line = 428:9]

.preheader:                                       ; preds = %29, %28
  %p_Val2_63 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind, !dbg !166 ; [#uses=3 type=i2] [debug line = 76:2@430:6]
  %brmerge2 = or i1 %p_ignore_2, %terminate_read, !dbg !533 ; [#uses=1 type=i1] [debug line = 431:6]
  br i1 %brmerge2, label %.backedge.loopexit83, label %29, !dbg !533 ; [debug line = 431:6]

; <label>:29                                      ; preds = %.preheader
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_63}, i64 0, metadata !534), !dbg !536 ; [debug line = 433:53] [debug variable = __Val2__]
  %tmp_54 = trunc i2 %p_Val2_63 to i1, !dbg !537  ; [#uses=1 type=i1] [debug line = 433:87]
  br i1 %tmp_54, label %30, label %.preheader, !dbg !538 ; [debug line = 433:176]

; <label>:30                                      ; preds = %29
  store i2 %p_Val2_63, i2* @i2c_val, align 1, !dbg !166 ; [debug line = 76:2@430:6]
  %tmp_56 = shl i8 %reg_data_4, 1, !dbg !161      ; [#uses=1 type=i8] [debug line = 435:5]
  %reg_data_8 = or i8 %tmp_56, 1, !dbg !161       ; [#uses=3 type=i8] [debug line = 435:5]
  call void @llvm.dbg.value(metadata !{i8 %reg_data_8}, i64 0, metadata !358), !dbg !161 ; [debug line = 435:5] [debug variable = reg_data]
  br label %._crit_edge101, !dbg !539             ; [debug line = 437:5]

._crit_edge101:                                   ; preds = %31, %30
  %p_Val2_65 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind, !dbg !156 ; [#uses=4 type=i2] [debug line = 76:2@438:6]
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_65}, i64 0, metadata !540), !dbg !542 ; [debug line = 439:49] [debug variable = __Val2__]
  %tmp_58 = call i1 @_ssdm_op_BitSelect.i1.i2.i32(i2 %p_Val2_65, i32 1), !dbg !543 ; [#uses=1 type=i1] [debug line = 439:83]
  %tmp_s = xor i1 %tmp_58, true, !dbg !543        ; [#uses=1 type=i1] [debug line = 439:83]
  %brmerge3 = or i1 %pre_i2c_sda_val, %tmp_s, !dbg !544 ; [#uses=1 type=i1] [debug line = 439:172]
  br i1 %brmerge3, label %31, label %.backedge.loopexit, !dbg !544 ; [debug line = 439:172]

; <label>:31                                      ; preds = %._crit_edge101
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_65}, i64 0, metadata !545), !dbg !547 ; [debug line = 441:53] [debug variable = __Val2__]
  %tmp_59 = trunc i2 %p_Val2_65 to i1, !dbg !548  ; [#uses=1 type=i1] [debug line = 441:87]
  br i1 %tmp_59, label %._crit_edge101, label %32, !dbg !549 ; [debug line = 441:176]

; <label>:32                                      ; preds = %31
  store i2 %p_Val2_65, i2* @i2c_val, align 1, !dbg !156 ; [debug line = 76:2@438:6]
  %tmp_10 = icmp ult i4 %bit_cnt_5, 7, !dbg !550  ; [#uses=1 type=i1] [debug line = 443:5]
  br i1 %tmp_10, label %33, label %._crit_edge102, !dbg !550 ; [debug line = 443:5]

; <label>:33                                      ; preds = %32
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !551 ; [debug line = 444:6]
  call void @llvm.dbg.value(metadata !{i8 %reg_data_8}, i64 0, metadata !127), !dbg !552 ; [debug line = 445:61] [debug variable = __Val2__]
  %tmp_60 = call i1 @_ssdm_op_BitSelect.i1.i8.i32(i8 %reg_data_8, i32 7), !dbg !553 ; [#uses=1 type=i1] [debug line = 445:96]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @i2c_sda_out, i1 %tmp_60) nounwind, !dbg !554 ; [debug line = 445:185]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !555 ; [debug line = 446:6]
  br label %._crit_edge102, !dbg !556             ; [debug line = 447:5]

._crit_edge102:                                   ; preds = %33, %32
  call void @llvm.dbg.value(metadata !{i4 %bit_cnt_10}, i64 0, metadata !235), !dbg !532 ; [debug line = 428:35] [debug variable = bit_cnt]
  br label %28, !dbg !532                         ; [debug line = 428:35]

_ifconv:                                          ; preds = %28
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !557 ; [debug line = 450:4]
  call void @_ssdm_op_Write.ap_none.volatile.i1P(i1* @i2c_sda_oe, i1 false) nounwind, !dbg !558 ; [debug line = 451:4]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !559 ; [debug line = 452:4]
  %reg_data_6 = call fastcc zeroext i8 @i2c_slave_core_read_mem(i8 zeroext %re_7), !dbg !560 ; [#uses=1 type=i8] [debug line = 453:15]
  call void @llvm.dbg.value(metadata !{i8 %reg_data_6}, i64 0, metadata !358), !dbg !560 ; [debug line = 453:15] [debug variable = reg_data]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !561 ; [debug line = 454:4]
  %auto_inc_regad_in_read_2 = call i1 @_ssdm_op_Read.ap_none.volatile.i1P(i1* @auto_inc_regad_in) nounwind, !dbg !562 ; [#uses=1 type=i1] [debug line = 455:4]
  %reg_addr_3 = add i8 %re_7, 1, !dbg !563        ; [#uses=1 type=i8] [debug line = 456:5]
  call void @llvm.dbg.value(metadata !{i8 %reg_addr_3}, i64 0, metadata !126), !dbg !563 ; [debug line = 456:5] [debug variable = reg_addr]
  %re_7_s = select i1 %p_ignore_2, i8 %re_7, i8 %reg_addr_3, !dbg !562 ; [#uses=1 type=i8] [debug line = 455:4]
  %re_8 = select i1 %auto_inc_regad_in_read_2, i8 %re_7_s, i8 %re_7 ; [#uses=1 type=i8]
  br label %._crit_edge104, !dbg !564             ; [debug line = 459:4]

._crit_edge104:                                   ; preds = %._crit_edge104, %_ifconv
  %p_Val2_62 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind, !dbg !565 ; [#uses=3 type=i2] [debug line = 76:2@460:5]
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_62}, i64 0, metadata !568), !dbg !570 ; [debug line = 461:52] [debug variable = __Val2__]
  %tmp_53 = trunc i2 %p_Val2_62 to i1, !dbg !571  ; [#uses=1 type=i1] [debug line = 461:86]
  br i1 %tmp_53, label %34, label %._crit_edge104, !dbg !572 ; [debug line = 461:175]

; <label>:34                                      ; preds = %._crit_edge104
  store i2 %p_Val2_62, i2* @i2c_val, align 1, !dbg !565 ; [debug line = 76:2@460:5]
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_62}, i64 0, metadata !573), !dbg !575 ; [debug line = 463:60] [debug variable = __Val2__]
  %terminate_read_1 = call i1 @_ssdm_op_BitSelect.i1.i2.i32(i2 %p_Val2_62, i32 1), !dbg !576 ; [#uses=1 type=i1] [debug line = 463:94]
  call void @llvm.dbg.value(metadata !{i1 %terminate_read_1}, i64 0, metadata !577), !dbg !578 ; [debug line = 463:183] [debug variable = terminate_read]
  br label %._crit_edge105, !dbg !579             ; [debug line = 465:4]

._crit_edge105:                                   ; preds = %._crit_edge105, %34
  %p_Val2_64 = call i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2* @i2c_in) nounwind, !dbg !580 ; [#uses=2 type=i2] [debug line = 76:2@466:5]
  call void @llvm.dbg.value(metadata !{i2 %p_Val2_64}, i64 0, metadata !583), !dbg !585 ; [debug line = 467:52] [debug variable = __Val2__]
  %tmp_57 = trunc i2 %p_Val2_64 to i1, !dbg !586  ; [#uses=1 type=i1] [debug line = 467:86]
  br i1 %tmp_57, label %._crit_edge105, label %35, !dbg !587 ; [debug line = 467:175]

; <label>:35                                      ; preds = %._crit_edge105
  store i2 %p_Val2_64, i2* @i2c_val, align 1, !dbg !580 ; [debug line = 76:2@466:5]
  %empty_6 = call i32 (...)* @_ssdm_op_SpecRegionEnd([12 x i8]* @p_str6, i32 %tmp_3) nounwind, !dbg !588 ; [#uses=0 type=i32] [debug line = 468:3]
  br label %.preheader31, !dbg !588               ; [debug line = 468:3]
}

; [#uses=11]
define weak void @_ssdm_op_Write.ap_none.volatile.i8P(i8*, i8) {
entry:
  store i8 %1, i8* %0
  ret void
}

; [#uses=27]
define weak void @_ssdm_op_Write.ap_none.volatile.i1P(i1*, i1) {
entry:
  store i1 %1, i1* %0
  ret void
}

; [#uses=29]
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

; [#uses=4]
define weak i32 @_ssdm_op_SpecLoopTripCount(...) {
entry:
  ret i32 0
}

; [#uses=1]
define weak void @_ssdm_op_SpecLoopName(...) nounwind {
entry:
  ret void
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

; [#uses=3]
define weak i7 @_ssdm_op_Read.ap_none.volatile.i7P(i7*) {
entry:
  %empty = load i7* %0                            ; [#uses=1 type=i7]
  ret i7 %empty
}

; [#uses=31]
define weak i2 @_ssdm_op_Read.ap_none.volatile.i2P(i2*) {
entry:
  %empty = load i2* %0                            ; [#uses=1 type=i2]
  ret i2 %empty
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

; [#uses=0]
declare i7 @_ssdm_op_PartSelect.i7.i8.i32.i32(i8, i32, i32) nounwind readnone

; [#uses=0]
declare i6 @_ssdm_op_PartSelect.i6.i7.i32.i32(i7, i32, i32) nounwind readnone

; [#uses=0]
declare i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2, i32, i32) nounwind readnone

; [#uses=2]
define weak i1 @_ssdm_op_BitSelect.i1.i8.i32(i8, i32) nounwind readnone {
entry:
  %empty = trunc i32 %1 to i8                     ; [#uses=1 type=i8]
  %empty_7 = shl i8 1, %empty                     ; [#uses=1 type=i8]
  %empty_8 = and i8 %0, %empty_7                  ; [#uses=1 type=i8]
  %empty_9 = icmp ne i8 %empty_8, 0               ; [#uses=1 type=i1]
  ret i1 %empty_9
}

; [#uses=2]
define weak i1 @_ssdm_op_BitSelect.i1.i4.i32(i4, i32) nounwind readnone {
entry:
  %empty = trunc i32 %1 to i4                     ; [#uses=1 type=i4]
  %empty_10 = shl i4 1, %empty                    ; [#uses=1 type=i4]
  %empty_11 = and i4 %0, %empty_10                ; [#uses=1 type=i4]
  %empty_12 = icmp ne i4 %empty_11, 0             ; [#uses=1 type=i1]
  ret i1 %empty_12
}

; [#uses=16]
define weak i1 @_ssdm_op_BitSelect.i1.i2.i32(i2, i32) nounwind readnone {
entry:
  %empty = trunc i32 %1 to i2                     ; [#uses=1 type=i2]
  %empty_13 = shl i2 1, %empty                    ; [#uses=1 type=i2]
  %empty_14 = and i2 %0, %empty_13                ; [#uses=1 type=i2]
  %empty_15 = icmp ne i2 %empty_14, 0             ; [#uses=1 type=i1]
  ret i1 %empty_15
}

; [#uses=3]
define weak i8 @_ssdm_op_BitConcatenate.i8.i7.i1(i7, i1) nounwind readnone {
entry:
  %empty = zext i7 %0 to i8                       ; [#uses=1 type=i8]
  %empty_16 = zext i1 %1 to i8                    ; [#uses=1 type=i8]
  %empty_17 = shl i8 %empty, 1                    ; [#uses=1 type=i8]
  %empty_18 = or i8 %empty_17, %empty_16          ; [#uses=1 type=i8]
  ret i8 %empty_18
}

; [#uses=2]
define weak i7 @_ssdm_op_BitConcatenate.i7.i6.i1(i6, i1) nounwind readnone {
entry:
  %empty = zext i6 %0 to i7                       ; [#uses=1 type=i7]
  %empty_19 = zext i1 %1 to i7                    ; [#uses=1 type=i7]
  %empty_20 = shl i7 %empty, 1                    ; [#uses=1 type=i7]
  %empty_21 = or i7 %empty_20, %empty_19          ; [#uses=1 type=i7]
  ret i7 %empty_21
}

!hls.encrypted.func = !{}
!llvm.map.gv = !{!0, !7, !12, !17, !22, !27, !32, !37, !42, !47, !52, !57, !62}

!0 = metadata !{metadata !1, i1* @mem_wreq}
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0, i32 0, metadata !3}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"mem_wreq", metadata !5, metadata !"uint1", i32 0, i32 0}
!5 = metadata !{metadata !6}
!6 = metadata !{i32 0, i32 0, i32 1}
!7 = metadata !{metadata !8, i1* @mem_wack}
!8 = metadata !{metadata !9}
!9 = metadata !{i32 0, i32 0, metadata !10}
!10 = metadata !{metadata !11}
!11 = metadata !{metadata !"mem_wack", metadata !5, metadata !"uint1", i32 0, i32 0}
!12 = metadata !{metadata !13, i1* @mem_rreq}
!13 = metadata !{metadata !14}
!14 = metadata !{i32 0, i32 0, metadata !15}
!15 = metadata !{metadata !16}
!16 = metadata !{metadata !"mem_rreq", metadata !5, metadata !"uint1", i32 0, i32 0}
!17 = metadata !{metadata !18, i1* @mem_rack}
!18 = metadata !{metadata !19}
!19 = metadata !{i32 0, i32 0, metadata !20}
!20 = metadata !{metadata !21}
!21 = metadata !{metadata !"mem_rack", metadata !5, metadata !"uint1", i32 0, i32 0}
!22 = metadata !{metadata !23, i8* @mem_dout}
!23 = metadata !{metadata !24}
!24 = metadata !{i32 0, i32 7, metadata !25}
!25 = metadata !{metadata !26}
!26 = metadata !{metadata !"mem_dout", metadata !5, metadata !"uint8", i32 0, i32 7}
!27 = metadata !{metadata !28, i8* @mem_din}
!28 = metadata !{metadata !29}
!29 = metadata !{i32 0, i32 7, metadata !30}
!30 = metadata !{metadata !31}
!31 = metadata !{metadata !"mem_din", metadata !5, metadata !"uint8", i32 0, i32 7}
!32 = metadata !{metadata !33, i8* @mem_addr}
!33 = metadata !{metadata !34}
!34 = metadata !{i32 0, i32 7, metadata !35}
!35 = metadata !{metadata !36}
!36 = metadata !{metadata !"mem_addr", metadata !5, metadata !"uint8", i32 0, i32 7}
!37 = metadata !{metadata !38, i2* @i2c_val}
!38 = metadata !{metadata !39}
!39 = metadata !{i32 0, i32 1, metadata !40}
!40 = metadata !{metadata !41}
!41 = metadata !{metadata !"i2c_val", metadata !5, metadata !"uint2", i32 0, i32 1}
!42 = metadata !{metadata !43, i1* @i2c_sda_out}
!43 = metadata !{metadata !44}
!44 = metadata !{i32 0, i32 0, metadata !45}
!45 = metadata !{metadata !46}
!46 = metadata !{metadata !"i2c_sda_out", metadata !5, metadata !"uint1", i32 0, i32 0}
!47 = metadata !{metadata !48, i1* @i2c_sda_oe}
!48 = metadata !{metadata !49}
!49 = metadata !{i32 0, i32 0, metadata !50}
!50 = metadata !{metadata !51}
!51 = metadata !{metadata !"i2c_sda_oe", metadata !5, metadata !"uint1", i32 0, i32 0}
!52 = metadata !{metadata !53, i2* @i2c_in}
!53 = metadata !{metadata !54}
!54 = metadata !{i32 0, i32 1, metadata !55}
!55 = metadata !{metadata !56}
!56 = metadata !{metadata !"i2c_in", metadata !5, metadata !"uint2", i32 0, i32 1}
!57 = metadata !{metadata !58, i7* @dev_addr_in}
!58 = metadata !{metadata !59}
!59 = metadata !{i32 0, i32 6, metadata !60}
!60 = metadata !{metadata !61}
!61 = metadata !{metadata !"dev_addr_in", metadata !5, metadata !"uint7", i32 0, i32 6}
!62 = metadata !{metadata !63, i1* @auto_inc_regad_in}
!63 = metadata !{metadata !64}
!64 = metadata !{i32 0, i32 0, metadata !65}
!65 = metadata !{metadata !66}
!66 = metadata !{metadata !"auto_inc_regad_in", metadata !5, metadata !"uint1", i32 0, i32 0}
!67 = metadata !{i32 786689, metadata !68, metadata !"data", metadata !69, i32 33554512, metadata !72, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!68 = metadata !{i32 786478, i32 0, metadata !69, metadata !"write_mem", metadata !"write_mem", metadata !"", metadata !69, i32 80, metadata !70, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (i8, i8)* @i2c_slave_core_write_mem, null, null, metadata !74, i32 81} ; [ DW_TAG_subprogram ]
!69 = metadata !{i32 786473, metadata !"i2c_slave_core.c", metadata !"D:\5C21_streamer_car5_artix7\5Cfpga_arty", null} ; [ DW_TAG_file_type ]
!70 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !71, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!71 = metadata !{null, metadata !72, metadata !72}
!72 = metadata !{i32 786454, null, metadata !"uint8", metadata !69, i32 10, i64 0, i64 0, i64 0, i32 0, metadata !73} ; [ DW_TAG_typedef ]
!73 = metadata !{i32 786468, null, metadata !"uint8", null, i32 0, i64 8, i64 8, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!74 = metadata !{metadata !75}
!75 = metadata !{i32 786468}                      ; [ DW_TAG_base_type ]
!76 = metadata !{i32 80, i32 34, metadata !68, null}
!77 = metadata !{i32 786689, metadata !68, metadata !"addr", metadata !69, i32 16777296, metadata !72, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!78 = metadata !{i32 80, i32 22, metadata !68, null}
!79 = metadata !{i32 83, i32 2, metadata !80, null}
!80 = metadata !{i32 786443, metadata !68, i32 81, i32 1, metadata !69, i32 1} ; [ DW_TAG_lexical_block ]
!81 = metadata !{i32 84, i32 2, metadata !80, null}
!82 = metadata !{i32 85, i32 2, metadata !80, null}
!83 = metadata !{i32 86, i32 2, metadata !80, null}
!84 = metadata !{i32 87, i32 2, metadata !80, null}
!85 = metadata !{i32 89, i32 2, metadata !80, null}
!86 = metadata !{i32 90, i32 3, metadata !87, null}
!87 = metadata !{i32 786443, metadata !80, i32 89, i32 5, metadata !69, i32 2} ; [ DW_TAG_lexical_block ]
!88 = metadata !{i32 91, i32 3, metadata !87, null}
!89 = metadata !{i32 92, i32 3, metadata !87, null}
!90 = metadata !{i32 93, i32 2, metadata !87, null}
!91 = metadata !{i32 94, i32 2, metadata !80, null}
!92 = metadata !{i32 96, i32 2, metadata !80, null}
!93 = metadata !{i32 97, i32 2, metadata !80, null}
!94 = metadata !{i32 98, i32 2, metadata !80, null}
!95 = metadata !{i32 99, i32 2, metadata !80, null}
!96 = metadata !{i32 100, i32 1, metadata !80, null}
!97 = metadata !{i32 786689, metadata !98, metadata !"addr", metadata !69, i32 16777319, metadata !72, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!98 = metadata !{i32 786478, i32 0, metadata !69, metadata !"read_mem", metadata !"read_mem", metadata !"", metadata !69, i32 103, metadata !99, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i8 (i8)* @i2c_slave_core_read_mem, null, null, metadata !74, i32 104} ; [ DW_TAG_subprogram ]
!99 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !100, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!100 = metadata !{metadata !72, metadata !72}
!101 = metadata !{i32 103, i32 22, metadata !98, null}
!102 = metadata !{i32 108, i32 2, metadata !103, null}
!103 = metadata !{i32 786443, metadata !98, i32 104, i32 1, metadata !69, i32 3} ; [ DW_TAG_lexical_block ]
!104 = metadata !{i32 109, i32 2, metadata !103, null}
!105 = metadata !{i32 110, i32 2, metadata !103, null}
!106 = metadata !{i32 111, i32 2, metadata !103, null}
!107 = metadata !{i32 113, i32 2, metadata !103, null}
!108 = metadata !{i32 114, i32 3, metadata !109, null}
!109 = metadata !{i32 786443, metadata !103, i32 113, i32 5, metadata !69, i32 4} ; [ DW_TAG_lexical_block ]
!110 = metadata !{i32 115, i32 3, metadata !109, null}
!111 = metadata !{i32 116, i32 3, metadata !109, null}
!112 = metadata !{i32 786688, metadata !103, metadata !"dt", metadata !69, i32 106, metadata !72, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!113 = metadata !{i32 117, i32 2, metadata !109, null}
!114 = metadata !{i32 118, i32 2, metadata !103, null}
!115 = metadata !{i32 120, i32 2, metadata !103, null}
!116 = metadata !{i32 121, i32 2, metadata !103, null}
!117 = metadata !{i32 122, i32 2, metadata !103, null}
!118 = metadata !{i32 124, i32 2, metadata !103, null}
!119 = metadata !{i32 786688, metadata !120, metadata !"dev_addr", metadata !69, i32 150, metadata !124, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!120 = metadata !{i32 786443, metadata !121, i32 132, i32 1, metadata !69, i32 5} ; [ DW_TAG_lexical_block ]
!121 = metadata !{i32 786478, i32 0, metadata !69, metadata !"i2c_slave_core", metadata !"i2c_slave_core", metadata !"", metadata !69, i32 131, metadata !122, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @i2c_slave_core, null, null, metadata !74, i32 132} ; [ DW_TAG_subprogram ]
!122 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !123, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!123 = metadata !{null}
!124 = metadata !{i32 786454, null, metadata !"uint7", metadata !69, i32 9, i64 0, i64 0, i64 0, i32 0, metadata !125} ; [ DW_TAG_typedef ]
!125 = metadata !{i32 786468, null, metadata !"uint7", null, i32 0, i64 7, i64 8, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!126 = metadata !{i32 786688, metadata !120, metadata !"reg_addr", metadata !69, i32 151, metadata !72, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!127 = metadata !{i32 786688, metadata !128, metadata !"__Val2__", metadata !69, i32 445, metadata !72, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!128 = metadata !{i32 786443, metadata !129, i32 445, i32 21, metadata !69, i32 93} ; [ DW_TAG_lexical_block ]
!129 = metadata !{i32 786443, metadata !130, i32 443, i32 23, metadata !69, i32 92} ; [ DW_TAG_lexical_block ]
!130 = metadata !{i32 786443, metadata !131, i32 428, i32 46, metadata !69, i32 86} ; [ DW_TAG_lexical_block ]
!131 = metadata !{i32 786443, metadata !132, i32 428, i32 4, metadata !69, i32 85} ; [ DW_TAG_lexical_block ]
!132 = metadata !{i32 786443, metadata !133, i32 421, i32 13, metadata !69, i32 83} ; [ DW_TAG_lexical_block ]
!133 = metadata !{i32 786443, metadata !120, i32 168, i32 12, metadata !69, i32 6} ; [ DW_TAG_lexical_block ]
!134 = metadata !{i32 133, i32 1, metadata !120, null}
!135 = metadata !{i32 134, i32 1, metadata !120, null}
!136 = metadata !{i32 135, i32 1, metadata !120, null}
!137 = metadata !{i32 136, i32 1, metadata !120, null}
!138 = metadata !{i32 138, i32 1, metadata !120, null}
!139 = metadata !{i32 139, i32 1, metadata !120, null}
!140 = metadata !{i32 141, i32 1, metadata !120, null}
!141 = metadata !{i32 142, i32 1, metadata !120, null}
!142 = metadata !{i32 143, i32 1, metadata !120, null}
!143 = metadata !{i32 144, i32 1, metadata !120, null}
!144 = metadata !{i32 145, i32 1, metadata !120, null}
!145 = metadata !{i32 146, i32 1, metadata !120, null}
!146 = metadata !{i32 147, i32 1, metadata !120, null}
!147 = metadata !{i32 158, i32 2, metadata !120, null}
!148 = metadata !{i32 159, i32 2, metadata !120, null}
!149 = metadata !{i32 160, i32 2, metadata !120, null}
!150 = metadata !{i32 161, i32 2, metadata !120, null}
!151 = metadata !{i32 162, i32 2, metadata !120, null}
!152 = metadata !{i32 163, i32 2, metadata !120, null}
!153 = metadata !{i32 164, i32 2, metadata !120, null}
!154 = metadata !{i32 165, i32 2, metadata !120, null}
!155 = metadata !{i32 168, i32 13, metadata !133, null}
!156 = metadata !{i32 76, i32 2, metadata !157, metadata !159}
!157 = metadata !{i32 786443, metadata !158, i32 74, i32 1, metadata !69, i32 0} ; [ DW_TAG_lexical_block ]
!158 = metadata !{i32 786478, i32 0, metadata !69, metadata !"read_i2c", metadata !"read_i2c", metadata !"", metadata !69, i32 73, metadata !122, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, metadata !74, i32 74} ; [ DW_TAG_subprogram ]
!159 = metadata !{i32 438, i32 6, metadata !160, null}
!160 = metadata !{i32 786443, metadata !130, i32 437, i32 8, metadata !69, i32 89} ; [ DW_TAG_lexical_block ]
!161 = metadata !{i32 435, i32 5, metadata !130, null}
!162 = metadata !{i32 205, i32 195, metadata !163, null}
!163 = metadata !{i32 786443, metadata !164, i32 205, i32 34, metadata !69, i32 20} ; [ DW_TAG_lexical_block ]
!164 = metadata !{i32 786443, metadata !165, i32 199, i32 45, metadata !69, i32 17} ; [ DW_TAG_lexical_block ]
!165 = metadata !{i32 786443, metadata !133, i32 199, i32 3, metadata !69, i32 16} ; [ DW_TAG_lexical_block ]
!166 = metadata !{i32 76, i32 2, metadata !157, metadata !167}
!167 = metadata !{i32 430, i32 6, metadata !168, null}
!168 = metadata !{i32 786443, metadata !130, i32 429, i32 8, metadata !69, i32 87} ; [ DW_TAG_lexical_block ]
!169 = metadata !{i32 404, i32 14, metadata !133, null}
!170 = metadata !{i32 76, i32 2, metadata !157, metadata !171}
!171 = metadata !{i32 325, i32 6, metadata !172, null}
!172 = metadata !{i32 786443, metadata !173, i32 323, i32 8, metadata !69, i32 58} ; [ DW_TAG_lexical_block ]
!173 = metadata !{i32 786443, metadata !174, i32 314, i32 24, metadata !69, i32 54} ; [ DW_TAG_lexical_block ]
!174 = metadata !{i32 786443, metadata !133, i32 313, i32 6, metadata !69, i32 53} ; [ DW_TAG_lexical_block ]
!175 = metadata !{i32 320, i32 196, metadata !176, null}
!176 = metadata !{i32 786443, metadata !173, i32 320, i32 35, metadata !69, i32 57} ; [ DW_TAG_lexical_block ]
!177 = metadata !{i32 256, i32 195, metadata !178, null}
!178 = metadata !{i32 786443, metadata !179, i32 256, i32 34, metadata !69, i32 36} ; [ DW_TAG_lexical_block ]
!179 = metadata !{i32 786443, metadata !180, i32 250, i32 45, metadata !69, i32 33} ; [ DW_TAG_lexical_block ]
!180 = metadata !{i32 786443, metadata !133, i32 250, i32 3, metadata !69, i32 32} ; [ DW_TAG_lexical_block ]
!181 = metadata !{i32 76, i32 2, metadata !157, metadata !182}
!182 = metadata !{i32 293, i32 4, metadata !183, null}
!183 = metadata !{i32 786443, metadata !133, i32 291, i32 6, metadata !69, i32 46} ; [ DW_TAG_lexical_block ]
!184 = metadata !{i32 289, i32 194, metadata !185, null}
!185 = metadata !{i32 786443, metadata !133, i32 289, i32 33, metadata !69, i32 45} ; [ DW_TAG_lexical_block ]
!186 = metadata !{i32 76, i32 2, metadata !157, metadata !187}
!187 = metadata !{i32 191, i32 4, metadata !188, null}
!188 = metadata !{i32 786443, metadata !133, i32 190, i32 6, metadata !69, i32 13} ; [ DW_TAG_lexical_block ]
!189 = metadata !{i32 76, i32 2, metadata !157, metadata !190}
!190 = metadata !{i32 184, i32 4, metadata !191, null}
!191 = metadata !{i32 786443, metadata !133, i32 183, i32 6, metadata !69, i32 10} ; [ DW_TAG_lexical_block ]
!192 = metadata !{i32 173, i32 3, metadata !133, null}
!193 = metadata !{i32 174, i32 3, metadata !133, null}
!194 = metadata !{i32 175, i32 3, metadata !133, null}
!195 = metadata !{i32 178, i32 3, metadata !133, null}
!196 = metadata !{i32 76, i32 2, metadata !157, metadata !197}
!197 = metadata !{i32 179, i32 4, metadata !198, null}
!198 = metadata !{i32 786443, metadata !133, i32 178, i32 6, metadata !69, i32 7} ; [ DW_TAG_lexical_block ]
!199 = metadata !{i32 786688, metadata !200, metadata !"__Val2__", metadata !69, i32 180, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!200 = metadata !{i32 786443, metadata !133, i32 180, i32 13, metadata !69, i32 8} ; [ DW_TAG_lexical_block ]
!201 = metadata !{i32 786454, null, metadata !"uint2", metadata !69, i32 4, i64 0, i64 0, i64 0, i32 0, metadata !202} ; [ DW_TAG_typedef ]
!202 = metadata !{i32 786468, null, metadata !"uint2", null, i32 0, i64 2, i64 2, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!203 = metadata !{i32 180, i32 51, metadata !200, null}
!204 = metadata !{i32 180, i32 85, metadata !200, null}
!205 = metadata !{i32 180, i32 174, metadata !200, null}
!206 = metadata !{i32 786688, metadata !207, metadata !"__Val2__", metadata !69, i32 180, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!207 = metadata !{i32 786443, metadata !133, i32 180, i32 186, metadata !69, i32 9} ; [ DW_TAG_lexical_block ]
!208 = metadata !{i32 180, i32 224, metadata !207, null}
!209 = metadata !{i32 180, i32 0, metadata !207, null}
!210 = metadata !{i32 786688, metadata !211, metadata !"__Val2__", metadata !69, i32 185, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!211 = metadata !{i32 786443, metadata !191, i32 185, i32 9, metadata !69, i32 11} ; [ DW_TAG_lexical_block ]
!212 = metadata !{i32 185, i32 47, metadata !211, null}
!213 = metadata !{i32 185, i32 81, metadata !211, null}
!214 = metadata !{i32 185, i32 170, metadata !211, null}
!215 = metadata !{i32 786688, metadata !216, metadata !"__Val2__", metadata !69, i32 187, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!216 = metadata !{i32 786443, metadata !133, i32 187, i32 13, metadata !69, i32 12} ; [ DW_TAG_lexical_block ]
!217 = metadata !{i32 187, i32 51, metadata !216, null}
!218 = metadata !{i32 187, i32 85, metadata !216, null}
!219 = metadata !{i32 187, i32 174, metadata !216, null}
!220 = metadata !{i32 786688, metadata !221, metadata !"__Val2__", metadata !69, i32 192, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!221 = metadata !{i32 786443, metadata !188, i32 192, i32 9, metadata !69, i32 14} ; [ DW_TAG_lexical_block ]
!222 = metadata !{i32 192, i32 47, metadata !221, null}
!223 = metadata !{i32 192, i32 81, metadata !221, null}
!224 = metadata !{i32 192, i32 170, metadata !221, null}
!225 = metadata !{i32 786688, metadata !226, metadata !"__Val2__", metadata !69, i32 194, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!226 = metadata !{i32 786443, metadata !133, i32 194, i32 13, metadata !69, i32 15} ; [ DW_TAG_lexical_block ]
!227 = metadata !{i32 194, i32 51, metadata !226, null}
!228 = metadata !{i32 194, i32 85, metadata !226, null}
!229 = metadata !{i32 194, i32 174, metadata !226, null}
!230 = metadata !{i32 76, i32 2, metadata !157, metadata !231}
!231 = metadata !{i32 209, i32 5, metadata !232, null}
!232 = metadata !{i32 786443, metadata !164, i32 208, i32 7, metadata !69, i32 21} ; [ DW_TAG_lexical_block ]
!233 = metadata !{i32 199, i32 8, metadata !165, null}
!234 = metadata !{i32 199, i32 34, metadata !165, null}
!235 = metadata !{i32 786688, metadata !120, metadata !"bit_cnt", metadata !69, i32 153, metadata !236, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!236 = metadata !{i32 786454, null, metadata !"uint4", metadata !69, i32 6, i64 0, i64 0, i64 0, i32 0, metadata !237} ; [ DW_TAG_typedef ]
!237 = metadata !{i32 786468, null, metadata !"uint4", null, i32 0, i64 4, i64 4, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!238 = metadata !{i32 76, i32 2, metadata !157, metadata !239}
!239 = metadata !{i32 202, i32 5, metadata !240, null}
!240 = metadata !{i32 786443, metadata !164, i32 201, i32 7, metadata !69, i32 18} ; [ DW_TAG_lexical_block ]
!241 = metadata !{i32 786688, metadata !242, metadata !"__Val2__", metadata !69, i32 203, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!242 = metadata !{i32 786443, metadata !164, i32 203, i32 14, metadata !69, i32 19} ; [ DW_TAG_lexical_block ]
!243 = metadata !{i32 203, i32 52, metadata !242, null}
!244 = metadata !{i32 203, i32 86, metadata !242, null}
!245 = metadata !{i32 203, i32 175, metadata !242, null}
!246 = metadata !{i32 786688, metadata !163, metadata !"__Val2__", metadata !69, i32 205, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!247 = metadata !{i32 205, i32 72, metadata !163, null}
!248 = metadata !{i32 205, i32 106, metadata !163, null}
!249 = metadata !{i32 208, i32 4, metadata !164, null}
!250 = metadata !{i32 786688, metadata !251, metadata !"__Val2__", metadata !69, i32 210, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!251 = metadata !{i32 786443, metadata !164, i32 210, i32 14, metadata !69, i32 22} ; [ DW_TAG_lexical_block ]
!252 = metadata !{i32 210, i32 52, metadata !251, null}
!253 = metadata !{i32 210, i32 86, metadata !251, null}
!254 = metadata !{i32 210, i32 175, metadata !251, null}
!255 = metadata !{i32 214, i32 3, metadata !133, null}
!256 = metadata !{i32 220, i32 3, metadata !133, null}
!257 = metadata !{i32 76, i32 2, metadata !157, metadata !258}
!258 = metadata !{i32 221, i32 4, metadata !259, null}
!259 = metadata !{i32 786443, metadata !133, i32 220, i32 6, metadata !69, i32 23} ; [ DW_TAG_lexical_block ]
!260 = metadata !{i32 786688, metadata !261, metadata !"__Val2__", metadata !69, i32 222, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!261 = metadata !{i32 786443, metadata !133, i32 222, i32 13, metadata !69, i32 24} ; [ DW_TAG_lexical_block ]
!262 = metadata !{i32 222, i32 51, metadata !261, null}
!263 = metadata !{i32 222, i32 85, metadata !261, null}
!264 = metadata !{i32 222, i32 174, metadata !261, null}
!265 = metadata !{i32 786688, metadata !266, metadata !"__Val2__", metadata !69, i32 224, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!266 = metadata !{i32 786443, metadata !133, i32 224, i32 8, metadata !69, i32 25} ; [ DW_TAG_lexical_block ]
!267 = metadata !{i32 224, i32 46, metadata !266, null}
!268 = metadata !{i32 224, i32 80, metadata !266, null}
!269 = metadata !{i32 224, i32 169, metadata !266, null}
!270 = metadata !{i32 227, i32 3, metadata !133, null}
!271 = metadata !{i32 76, i32 2, metadata !157, metadata !272}
!272 = metadata !{i32 228, i32 4, metadata !273, null}
!273 = metadata !{i32 786443, metadata !133, i32 227, i32 6, metadata !69, i32 26} ; [ DW_TAG_lexical_block ]
!274 = metadata !{i32 786688, metadata !275, metadata !"__Val2__", metadata !69, i32 229, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!275 = metadata !{i32 786443, metadata !133, i32 229, i32 13, metadata !69, i32 27} ; [ DW_TAG_lexical_block ]
!276 = metadata !{i32 229, i32 51, metadata !275, null}
!277 = metadata !{i32 229, i32 85, metadata !275, null}
!278 = metadata !{i32 229, i32 174, metadata !275, null}
!279 = metadata !{i32 232, i32 3, metadata !133, null}
!280 = metadata !{i32 233, i32 3, metadata !133, null}
!281 = metadata !{i32 234, i32 3, metadata !133, null}
!282 = metadata !{i32 235, i32 3, metadata !133, null}
!283 = metadata !{i32 237, i32 3, metadata !133, null}
!284 = metadata !{i32 76, i32 2, metadata !157, metadata !285}
!285 = metadata !{i32 238, i32 4, metadata !286, null}
!286 = metadata !{i32 786443, metadata !133, i32 237, i32 6, metadata !69, i32 28} ; [ DW_TAG_lexical_block ]
!287 = metadata !{i32 786688, metadata !288, metadata !"__Val2__", metadata !69, i32 239, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!288 = metadata !{i32 786443, metadata !133, i32 239, i32 13, metadata !69, i32 29} ; [ DW_TAG_lexical_block ]
!289 = metadata !{i32 239, i32 51, metadata !288, null}
!290 = metadata !{i32 239, i32 85, metadata !288, null}
!291 = metadata !{i32 239, i32 174, metadata !288, null}
!292 = metadata !{i32 76, i32 2, metadata !157, metadata !293}
!293 = metadata !{i32 242, i32 4, metadata !294, null}
!294 = metadata !{i32 786443, metadata !133, i32 241, i32 6, metadata !69, i32 30} ; [ DW_TAG_lexical_block ]
!295 = metadata !{i32 786688, metadata !296, metadata !"__Val2__", metadata !69, i32 243, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!296 = metadata !{i32 786443, metadata !133, i32 243, i32 13, metadata !69, i32 31} ; [ DW_TAG_lexical_block ]
!297 = metadata !{i32 243, i32 51, metadata !296, null}
!298 = metadata !{i32 243, i32 85, metadata !296, null}
!299 = metadata !{i32 243, i32 174, metadata !296, null}
!300 = metadata !{i32 245, i32 3, metadata !133, null}
!301 = metadata !{i32 250, i32 8, metadata !180, null}
!302 = metadata !{i32 407, i32 4, metadata !133, null}
!303 = metadata !{i32 250, i32 34, metadata !180, null}
!304 = metadata !{i32 76, i32 2, metadata !157, metadata !305}
!305 = metadata !{i32 253, i32 5, metadata !306, null}
!306 = metadata !{i32 786443, metadata !179, i32 252, i32 7, metadata !69, i32 34} ; [ DW_TAG_lexical_block ]
!307 = metadata !{i32 786688, metadata !308, metadata !"__Val2__", metadata !69, i32 254, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!308 = metadata !{i32 786443, metadata !179, i32 254, i32 14, metadata !69, i32 35} ; [ DW_TAG_lexical_block ]
!309 = metadata !{i32 254, i32 52, metadata !308, null}
!310 = metadata !{i32 254, i32 86, metadata !308, null}
!311 = metadata !{i32 254, i32 175, metadata !308, null}
!312 = metadata !{i32 786688, metadata !178, metadata !"__Val2__", metadata !69, i32 256, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!313 = metadata !{i32 256, i32 72, metadata !178, null}
!314 = metadata !{i32 256, i32 106, metadata !178, null}
!315 = metadata !{i32 259, i32 4, metadata !179, null}
!316 = metadata !{i32 76, i32 2, metadata !157, metadata !317}
!317 = metadata !{i32 260, i32 5, metadata !318, null}
!318 = metadata !{i32 786443, metadata !179, i32 259, i32 7, metadata !69, i32 37} ; [ DW_TAG_lexical_block ]
!319 = metadata !{i32 786688, metadata !320, metadata !"__Val2__", metadata !69, i32 261, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!320 = metadata !{i32 786443, metadata !179, i32 261, i32 14, metadata !69, i32 38} ; [ DW_TAG_lexical_block ]
!321 = metadata !{i32 261, i32 52, metadata !320, null}
!322 = metadata !{i32 261, i32 86, metadata !320, null}
!323 = metadata !{i32 261, i32 175, metadata !320, null}
!324 = metadata !{i32 265, i32 3, metadata !133, null}
!325 = metadata !{i32 266, i32 3, metadata !133, null}
!326 = metadata !{i32 267, i32 3, metadata !133, null}
!327 = metadata !{i32 268, i32 3, metadata !133, null}
!328 = metadata !{i32 270, i32 3, metadata !133, null}
!329 = metadata !{i32 76, i32 2, metadata !157, metadata !330}
!330 = metadata !{i32 271, i32 4, metadata !331, null}
!331 = metadata !{i32 786443, metadata !133, i32 270, i32 6, metadata !69, i32 39} ; [ DW_TAG_lexical_block ]
!332 = metadata !{i32 786688, metadata !333, metadata !"__Val2__", metadata !69, i32 272, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!333 = metadata !{i32 786443, metadata !133, i32 272, i32 13, metadata !69, i32 40} ; [ DW_TAG_lexical_block ]
!334 = metadata !{i32 272, i32 51, metadata !333, null}
!335 = metadata !{i32 272, i32 85, metadata !333, null}
!336 = metadata !{i32 272, i32 174, metadata !333, null}
!337 = metadata !{i32 76, i32 2, metadata !157, metadata !338}
!338 = metadata !{i32 275, i32 4, metadata !339, null}
!339 = metadata !{i32 786443, metadata !133, i32 274, i32 6, metadata !69, i32 41} ; [ DW_TAG_lexical_block ]
!340 = metadata !{i32 786688, metadata !341, metadata !"__Val2__", metadata !69, i32 276, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!341 = metadata !{i32 786443, metadata !133, i32 276, i32 13, metadata !69, i32 42} ; [ DW_TAG_lexical_block ]
!342 = metadata !{i32 276, i32 51, metadata !341, null}
!343 = metadata !{i32 276, i32 85, metadata !341, null}
!344 = metadata !{i32 276, i32 174, metadata !341, null}
!345 = metadata !{i32 278, i32 3, metadata !133, null}
!346 = metadata !{i32 285, i32 3, metadata !133, null}
!347 = metadata !{i32 76, i32 2, metadata !157, metadata !348}
!348 = metadata !{i32 286, i32 4, metadata !349, null}
!349 = metadata !{i32 786443, metadata !133, i32 285, i32 6, metadata !69, i32 43} ; [ DW_TAG_lexical_block ]
!350 = metadata !{i32 786688, metadata !351, metadata !"__Val2__", metadata !69, i32 287, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!351 = metadata !{i32 786443, metadata !133, i32 287, i32 13, metadata !69, i32 44} ; [ DW_TAG_lexical_block ]
!352 = metadata !{i32 287, i32 51, metadata !351, null}
!353 = metadata !{i32 287, i32 85, metadata !351, null}
!354 = metadata !{i32 287, i32 174, metadata !351, null}
!355 = metadata !{i32 786688, metadata !185, metadata !"__Val2__", metadata !69, i32 289, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!356 = metadata !{i32 289, i32 71, metadata !185, null}
!357 = metadata !{i32 289, i32 105, metadata !185, null}
!358 = metadata !{i32 786688, metadata !120, metadata !"reg_data", metadata !69, i32 152, metadata !72, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!359 = metadata !{i32 291, i32 3, metadata !133, null}
!360 = metadata !{i32 292, i32 61, metadata !361, null}
!361 = metadata !{i32 786443, metadata !183, i32 292, i32 23, metadata !69, i32 47} ; [ DW_TAG_lexical_block ]
!362 = metadata !{i32 786688, metadata !361, metadata !"__Val2__", metadata !69, i32 292, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!363 = metadata !{i32 292, i32 95, metadata !361, null}
!364 = metadata !{i32 786688, metadata !120, metadata !"pre_i2c_sda_val", metadata !69, i32 156, metadata !365, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!365 = metadata !{i32 786454, null, metadata !"uint1", metadata !69, i32 3, i64 0, i64 0, i64 0, i32 0, metadata !366} ; [ DW_TAG_typedef ]
!366 = metadata !{i32 786468, null, metadata !"uint1", null, i32 0, i64 1, i64 1, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!367 = metadata !{i32 292, i32 184, metadata !361, null}
!368 = metadata !{i32 786688, metadata !369, metadata !"__Val2__", metadata !69, i32 295, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!369 = metadata !{i32 786443, metadata !183, i32 295, i32 9, metadata !69, i32 48} ; [ DW_TAG_lexical_block ]
!370 = metadata !{i32 295, i32 47, metadata !369, null}
!371 = metadata !{i32 295, i32 81, metadata !369, null}
!372 = metadata !{i32 295, i32 170, metadata !369, null}
!373 = metadata !{i32 314, i32 4, metadata !174, null}
!374 = metadata !{i32 297, i32 4, metadata !183, null}
!375 = metadata !{i32 786688, metadata !376, metadata !"__Val2__", metadata !69, i32 297, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!376 = metadata !{i32 786443, metadata !183, i32 297, i32 24, metadata !69, i32 49} ; [ DW_TAG_lexical_block ]
!377 = metadata !{i32 297, i32 62, metadata !376, null}
!378 = metadata !{i32 297, i32 96, metadata !376, null}
!379 = metadata !{i32 297, i32 185, metadata !376, null}
!380 = metadata !{i32 76, i32 2, metadata !157, metadata !381}
!381 = metadata !{i32 299, i32 6, metadata !382, null}
!382 = metadata !{i32 786443, metadata !383, i32 298, i32 8, metadata !69, i32 51} ; [ DW_TAG_lexical_block ]
!383 = metadata !{i32 786443, metadata !183, i32 297, i32 218, metadata !69, i32 50} ; [ DW_TAG_lexical_block ]
!384 = metadata !{i32 786688, metadata !385, metadata !"__Val2__", metadata !69, i32 300, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!385 = metadata !{i32 786443, metadata !383, i32 300, i32 15, metadata !69, i32 52} ; [ DW_TAG_lexical_block ]
!386 = metadata !{i32 300, i32 53, metadata !385, null}
!387 = metadata !{i32 300, i32 87, metadata !385, null}
!388 = metadata !{i32 300, i32 176, metadata !385, null}
!389 = metadata !{i32 76, i32 2, metadata !157, metadata !390}
!390 = metadata !{i32 317, i32 6, metadata !391, null}
!391 = metadata !{i32 786443, metadata !173, i32 316, i32 8, metadata !69, i32 55} ; [ DW_TAG_lexical_block ]
!392 = metadata !{i32 786688, metadata !393, metadata !"__Val2__", metadata !69, i32 318, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!393 = metadata !{i32 786443, metadata !173, i32 318, i32 15, metadata !69, i32 56} ; [ DW_TAG_lexical_block ]
!394 = metadata !{i32 318, i32 53, metadata !393, null}
!395 = metadata !{i32 318, i32 87, metadata !393, null}
!396 = metadata !{i32 318, i32 176, metadata !393, null}
!397 = metadata !{i32 786688, metadata !176, metadata !"__Val2__", metadata !69, i32 320, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!398 = metadata !{i32 320, i32 73, metadata !176, null}
!399 = metadata !{i32 320, i32 107, metadata !176, null}
!400 = metadata !{i32 323, i32 5, metadata !173, null}
!401 = metadata !{i32 324, i32 63, metadata !402, null}
!402 = metadata !{i32 786443, metadata !172, i32 324, i32 25, metadata !69, i32 59} ; [ DW_TAG_lexical_block ]
!403 = metadata !{i32 786688, metadata !402, metadata !"__Val2__", metadata !69, i32 324, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!404 = metadata !{i32 324, i32 97, metadata !402, null}
!405 = metadata !{i32 786688, metadata !406, metadata !"__Val2__", metadata !69, i32 326, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!406 = metadata !{i32 786443, metadata !172, i32 326, i32 11, metadata !69, i32 60} ; [ DW_TAG_lexical_block ]
!407 = metadata !{i32 326, i32 49, metadata !406, null}
!408 = metadata !{i32 326, i32 83, metadata !406, null}
!409 = metadata !{i32 326, i32 172, metadata !406, null}
!410 = metadata !{i32 786688, metadata !411, metadata !"__Val2__", metadata !69, i32 328, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!411 = metadata !{i32 786443, metadata !173, i32 328, i32 15, metadata !69, i32 61} ; [ DW_TAG_lexical_block ]
!412 = metadata !{i32 328, i32 53, metadata !411, null}
!413 = metadata !{i32 328, i32 87, metadata !411, null}
!414 = metadata !{i32 328, i32 176, metadata !411, null}
!415 = metadata !{i32 330, i32 5, metadata !173, null}
!416 = metadata !{i32 331, i32 4, metadata !173, null}
!417 = metadata !{i32 334, i32 4, metadata !174, null}
!418 = metadata !{i32 335, i32 4, metadata !174, null}
!419 = metadata !{i32 336, i32 4, metadata !174, null}
!420 = metadata !{i32 337, i32 4, metadata !174, null}
!421 = metadata !{i32 339, i32 4, metadata !174, null}
!422 = metadata !{i32 76, i32 2, metadata !157, metadata !423}
!423 = metadata !{i32 340, i32 5, metadata !424, null}
!424 = metadata !{i32 786443, metadata !174, i32 339, i32 7, metadata !69, i32 62} ; [ DW_TAG_lexical_block ]
!425 = metadata !{i32 786688, metadata !426, metadata !"__Val2__", metadata !69, i32 341, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!426 = metadata !{i32 786443, metadata !174, i32 341, i32 14, metadata !69, i32 63} ; [ DW_TAG_lexical_block ]
!427 = metadata !{i32 341, i32 52, metadata !426, null}
!428 = metadata !{i32 341, i32 86, metadata !426, null}
!429 = metadata !{i32 341, i32 175, metadata !426, null}
!430 = metadata !{i32 76, i32 2, metadata !157, metadata !431}
!431 = metadata !{i32 344, i32 5, metadata !432, null}
!432 = metadata !{i32 786443, metadata !174, i32 343, i32 7, metadata !69, i32 64} ; [ DW_TAG_lexical_block ]
!433 = metadata !{i32 786688, metadata !434, metadata !"__Val2__", metadata !69, i32 345, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!434 = metadata !{i32 786443, metadata !174, i32 345, i32 14, metadata !69, i32 65} ; [ DW_TAG_lexical_block ]
!435 = metadata !{i32 345, i32 52, metadata !434, null}
!436 = metadata !{i32 345, i32 86, metadata !434, null}
!437 = metadata !{i32 345, i32 175, metadata !434, null}
!438 = metadata !{i32 347, i32 4, metadata !174, null}
!439 = metadata !{i32 350, i32 4, metadata !174, null}
!440 = metadata !{i32 351, i32 5, metadata !441, null}
!441 = metadata !{i32 786443, metadata !174, i32 350, i32 21, metadata !69, i32 66} ; [ DW_TAG_lexical_block ]
!442 = metadata !{i32 352, i32 5, metadata !441, null}
!443 = metadata !{i32 353, i32 5, metadata !441, null}
!444 = metadata !{i32 354, i32 5, metadata !441, null}
!445 = metadata !{i32 355, i32 6, metadata !441, null}
!446 = metadata !{i32 356, i32 4, metadata !441, null}
!447 = metadata !{i32 76, i32 2, metadata !157, metadata !448}
!448 = metadata !{i32 377, i32 5, metadata !449, null}
!449 = metadata !{i32 786443, metadata !450, i32 376, i32 7, metadata !69, i32 72} ; [ DW_TAG_lexical_block ]
!450 = metadata !{i32 786443, metadata !451, i32 367, i32 45, metadata !69, i32 68} ; [ DW_TAG_lexical_block ]
!451 = metadata !{i32 786443, metadata !133, i32 367, i32 3, metadata !69, i32 67} ; [ DW_TAG_lexical_block ]
!452 = metadata !{i32 367, i32 8, metadata !451, null}
!453 = metadata !{i32 367, i32 34, metadata !451, null}
!454 = metadata !{i32 367, i32 46, metadata !450, null}
!455 = metadata !{i32 369, i32 4, metadata !450, null}
!456 = metadata !{i32 76, i32 2, metadata !157, metadata !457}
!457 = metadata !{i32 370, i32 5, metadata !458, null}
!458 = metadata !{i32 786443, metadata !450, i32 369, i32 7, metadata !69, i32 69} ; [ DW_TAG_lexical_block ]
!459 = metadata !{i32 786688, metadata !460, metadata !"__Val2__", metadata !69, i32 371, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!460 = metadata !{i32 786443, metadata !450, i32 371, i32 14, metadata !69, i32 70} ; [ DW_TAG_lexical_block ]
!461 = metadata !{i32 371, i32 52, metadata !460, null}
!462 = metadata !{i32 371, i32 86, metadata !460, null}
!463 = metadata !{i32 371, i32 175, metadata !460, null}
!464 = metadata !{i32 786688, metadata !465, metadata !"__Val2__", metadata !69, i32 373, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!465 = metadata !{i32 786443, metadata !450, i32 373, i32 34, metadata !69, i32 71} ; [ DW_TAG_lexical_block ]
!466 = metadata !{i32 373, i32 72, metadata !465, null}
!467 = metadata !{i32 373, i32 106, metadata !465, null}
!468 = metadata !{i32 373, i32 195, metadata !465, null}
!469 = metadata !{i32 376, i32 4, metadata !450, null}
!470 = metadata !{i32 786688, metadata !471, metadata !"__Val2__", metadata !69, i32 378, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!471 = metadata !{i32 786443, metadata !450, i32 378, i32 14, metadata !69, i32 73} ; [ DW_TAG_lexical_block ]
!472 = metadata !{i32 378, i32 52, metadata !471, null}
!473 = metadata !{i32 378, i32 86, metadata !471, null}
!474 = metadata !{i32 378, i32 175, metadata !471, null}
!475 = metadata !{i32 382, i32 3, metadata !133, null}
!476 = metadata !{i32 388, i32 3, metadata !133, null}
!477 = metadata !{i32 76, i32 2, metadata !157, metadata !478}
!478 = metadata !{i32 389, i32 4, metadata !479, null}
!479 = metadata !{i32 786443, metadata !133, i32 388, i32 6, metadata !69, i32 74} ; [ DW_TAG_lexical_block ]
!480 = metadata !{i32 786688, metadata !481, metadata !"__Val2__", metadata !69, i32 390, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!481 = metadata !{i32 786443, metadata !133, i32 390, i32 13, metadata !69, i32 75} ; [ DW_TAG_lexical_block ]
!482 = metadata !{i32 390, i32 51, metadata !481, null}
!483 = metadata !{i32 390, i32 85, metadata !481, null}
!484 = metadata !{i32 390, i32 174, metadata !481, null}
!485 = metadata !{i32 786688, metadata !486, metadata !"__Val2__", metadata !69, i32 392, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!486 = metadata !{i32 786443, metadata !133, i32 392, i32 8, metadata !69, i32 76} ; [ DW_TAG_lexical_block ]
!487 = metadata !{i32 392, i32 46, metadata !486, null}
!488 = metadata !{i32 392, i32 80, metadata !486, null}
!489 = metadata !{i32 392, i32 169, metadata !486, null}
!490 = metadata !{i32 395, i32 3, metadata !133, null}
!491 = metadata !{i32 76, i32 2, metadata !157, metadata !492}
!492 = metadata !{i32 396, i32 4, metadata !493, null}
!493 = metadata !{i32 786443, metadata !133, i32 395, i32 6, metadata !69, i32 77} ; [ DW_TAG_lexical_block ]
!494 = metadata !{i32 786688, metadata !495, metadata !"__Val2__", metadata !69, i32 397, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!495 = metadata !{i32 786443, metadata !133, i32 397, i32 13, metadata !69, i32 78} ; [ DW_TAG_lexical_block ]
!496 = metadata !{i32 397, i32 51, metadata !495, null}
!497 = metadata !{i32 397, i32 85, metadata !495, null}
!498 = metadata !{i32 397, i32 174, metadata !495, null}
!499 = metadata !{i32 400, i32 3, metadata !133, null}
!500 = metadata !{i32 401, i32 3, metadata !133, null}
!501 = metadata !{i32 402, i32 3, metadata !133, null}
!502 = metadata !{i32 403, i32 3, metadata !133, null}
!503 = metadata !{i32 405, i32 3, metadata !133, null}
!504 = metadata !{i32 406, i32 3, metadata !133, null}
!505 = metadata !{i32 410, i32 3, metadata !133, null}
!506 = metadata !{i32 76, i32 2, metadata !157, metadata !507}
!507 = metadata !{i32 411, i32 4, metadata !508, null}
!508 = metadata !{i32 786443, metadata !133, i32 410, i32 6, metadata !69, i32 79} ; [ DW_TAG_lexical_block ]
!509 = metadata !{i32 786688, metadata !510, metadata !"__Val2__", metadata !69, i32 412, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!510 = metadata !{i32 786443, metadata !133, i32 412, i32 13, metadata !69, i32 80} ; [ DW_TAG_lexical_block ]
!511 = metadata !{i32 412, i32 51, metadata !510, null}
!512 = metadata !{i32 412, i32 85, metadata !510, null}
!513 = metadata !{i32 412, i32 174, metadata !510, null}
!514 = metadata !{i32 76, i32 2, metadata !157, metadata !515}
!515 = metadata !{i32 415, i32 4, metadata !516, null}
!516 = metadata !{i32 786443, metadata !133, i32 414, i32 6, metadata !69, i32 81} ; [ DW_TAG_lexical_block ]
!517 = metadata !{i32 786688, metadata !518, metadata !"__Val2__", metadata !69, i32 416, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!518 = metadata !{i32 786443, metadata !133, i32 416, i32 13, metadata !69, i32 82} ; [ DW_TAG_lexical_block ]
!519 = metadata !{i32 416, i32 51, metadata !518, null}
!520 = metadata !{i32 416, i32 85, metadata !518, null}
!521 = metadata !{i32 416, i32 174, metadata !518, null}
!522 = metadata !{i32 421, i32 14, metadata !132, null}
!523 = metadata !{i32 422, i32 4, metadata !132, null}
!524 = metadata !{i32 786688, metadata !525, metadata !"__Val2__", metadata !69, i32 423, metadata !72, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!525 = metadata !{i32 786443, metadata !132, i32 423, i32 19, metadata !69, i32 84} ; [ DW_TAG_lexical_block ]
!526 = metadata !{i32 423, i32 59, metadata !525, null}
!527 = metadata !{i32 423, i32 94, metadata !525, null}
!528 = metadata !{i32 423, i32 183, metadata !525, null}
!529 = metadata !{i32 424, i32 4, metadata !132, null}
!530 = metadata !{i32 425, i32 4, metadata !132, null}
!531 = metadata !{i32 428, i32 9, metadata !131, null}
!532 = metadata !{i32 428, i32 35, metadata !131, null}
!533 = metadata !{i32 431, i32 6, metadata !168, null}
!534 = metadata !{i32 786688, metadata !535, metadata !"__Val2__", metadata !69, i32 433, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!535 = metadata !{i32 786443, metadata !130, i32 433, i32 15, metadata !69, i32 88} ; [ DW_TAG_lexical_block ]
!536 = metadata !{i32 433, i32 53, metadata !535, null}
!537 = metadata !{i32 433, i32 87, metadata !535, null}
!538 = metadata !{i32 433, i32 176, metadata !535, null}
!539 = metadata !{i32 437, i32 5, metadata !130, null}
!540 = metadata !{i32 786688, metadata !541, metadata !"__Val2__", metadata !69, i32 439, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!541 = metadata !{i32 786443, metadata !160, i32 439, i32 11, metadata !69, i32 90} ; [ DW_TAG_lexical_block ]
!542 = metadata !{i32 439, i32 49, metadata !541, null}
!543 = metadata !{i32 439, i32 83, metadata !541, null}
!544 = metadata !{i32 439, i32 172, metadata !541, null}
!545 = metadata !{i32 786688, metadata !546, metadata !"__Val2__", metadata !69, i32 441, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!546 = metadata !{i32 786443, metadata !130, i32 441, i32 15, metadata !69, i32 91} ; [ DW_TAG_lexical_block ]
!547 = metadata !{i32 441, i32 53, metadata !546, null}
!548 = metadata !{i32 441, i32 87, metadata !546, null}
!549 = metadata !{i32 441, i32 176, metadata !546, null}
!550 = metadata !{i32 443, i32 5, metadata !130, null}
!551 = metadata !{i32 444, i32 6, metadata !129, null}
!552 = metadata !{i32 445, i32 61, metadata !128, null}
!553 = metadata !{i32 445, i32 96, metadata !128, null}
!554 = metadata !{i32 445, i32 185, metadata !128, null}
!555 = metadata !{i32 446, i32 6, metadata !129, null}
!556 = metadata !{i32 447, i32 5, metadata !129, null}
!557 = metadata !{i32 450, i32 4, metadata !132, null}
!558 = metadata !{i32 451, i32 4, metadata !132, null}
!559 = metadata !{i32 452, i32 4, metadata !132, null}
!560 = metadata !{i32 453, i32 15, metadata !132, null}
!561 = metadata !{i32 454, i32 4, metadata !132, null}
!562 = metadata !{i32 455, i32 4, metadata !132, null}
!563 = metadata !{i32 456, i32 5, metadata !132, null}
!564 = metadata !{i32 459, i32 4, metadata !132, null}
!565 = metadata !{i32 76, i32 2, metadata !157, metadata !566}
!566 = metadata !{i32 460, i32 5, metadata !567, null}
!567 = metadata !{i32 786443, metadata !132, i32 459, i32 7, metadata !69, i32 94} ; [ DW_TAG_lexical_block ]
!568 = metadata !{i32 786688, metadata !569, metadata !"__Val2__", metadata !69, i32 461, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!569 = metadata !{i32 786443, metadata !132, i32 461, i32 14, metadata !69, i32 95} ; [ DW_TAG_lexical_block ]
!570 = metadata !{i32 461, i32 52, metadata !569, null}
!571 = metadata !{i32 461, i32 86, metadata !569, null}
!572 = metadata !{i32 461, i32 175, metadata !569, null}
!573 = metadata !{i32 786688, metadata !574, metadata !"__Val2__", metadata !69, i32 463, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!574 = metadata !{i32 786443, metadata !132, i32 463, i32 22, metadata !69, i32 96} ; [ DW_TAG_lexical_block ]
!575 = metadata !{i32 463, i32 60, metadata !574, null}
!576 = metadata !{i32 463, i32 94, metadata !574, null}
!577 = metadata !{i32 786688, metadata !120, metadata !"terminate_read", metadata !69, i32 155, metadata !365, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!578 = metadata !{i32 463, i32 183, metadata !574, null}
!579 = metadata !{i32 465, i32 4, metadata !132, null}
!580 = metadata !{i32 76, i32 2, metadata !157, metadata !581}
!581 = metadata !{i32 466, i32 5, metadata !582, null}
!582 = metadata !{i32 786443, metadata !132, i32 465, i32 7, metadata !69, i32 97} ; [ DW_TAG_lexical_block ]
!583 = metadata !{i32 786688, metadata !584, metadata !"__Val2__", metadata !69, i32 467, metadata !201, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!584 = metadata !{i32 786443, metadata !132, i32 467, i32 14, metadata !69, i32 98} ; [ DW_TAG_lexical_block ]
!585 = metadata !{i32 467, i32 52, metadata !584, null}
!586 = metadata !{i32 467, i32 86, metadata !584, null}
!587 = metadata !{i32 467, i32 175, metadata !584, null}
!588 = metadata !{i32 468, i32 3, metadata !132, null}
