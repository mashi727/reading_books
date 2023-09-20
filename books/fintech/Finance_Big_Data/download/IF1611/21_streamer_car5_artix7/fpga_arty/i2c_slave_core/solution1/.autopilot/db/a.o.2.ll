; ModuleID = 'D:/21_streamer_car5_artix7/fpga_arty/i2c_slave_core/solution1/.autopilot/db/a.o.2.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-w64-mingw32"

@mem_wreq = global i1 false, align 1              ; [#uses=6 type=i1*]
@mem_wack = common global i1 false, align 1       ; [#uses=3 type=i1*]
@mem_rreq = global i1 false, align 1              ; [#uses=6 type=i1*]
@mem_rack = common global i1 false, align 1       ; [#uses=3 type=i1*]
@mem_dout = global i8 0, align 1                  ; [#uses=6 type=i8*]
@mem_din = common global i8 0, align 1            ; [#uses=3 type=i8*]
@mem_addr = global i8 0, align 1                  ; [#uses=9 type=i8*]
@i2c_val = common global i2 0, align 1            ; [#uses=37 type=i2*]
@i2c_sda_out = global i1 true, align 1            ; [#uses=10 type=i1*]
@i2c_sda_oe = global i1 false, align 1            ; [#uses=13 type=i1*]
@i2c_in = common global i2 0, align 1             ; [#uses=32 type=i2*]
@dev_addr_in = common global i7 0, align 1        ; [#uses=4 type=i7*]
@auto_inc_regad_in = common global i1 false, align 1 ; [#uses=5 type=i1*]
@.str6 = private unnamed_addr constant [12 x i8] c"hls_label_4\00", align 1 ; [#uses=2 type=[12 x i8]*]
@.str5 = private unnamed_addr constant [17 x i8] c"label_read_start\00", align 1 ; [#uses=1 type=[17 x i8]*]
@.str1 = private unnamed_addr constant [8 x i8] c"ap_none\00", align 1 ; [#uses=12 type=[8 x i8]*]
@.str = private unnamed_addr constant [1 x i8] zeroinitializer, align 1 ; [#uses=48 type=[1 x i8]*]

; [#uses=1]
define internal fastcc void @write_mem(i8 zeroext %addr, i8 zeroext %data) nounwind uwtable {
  call void @llvm.dbg.value(metadata !{i8 %addr}, i64 0, metadata !96), !dbg !102 ; [debug line = 80:22] [debug variable = addr]
  call void @llvm.dbg.value(metadata !{i8 %data}, i64 0, metadata !103), !dbg !104 ; [debug line = 80:34] [debug variable = data]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !105 ; [debug line = 83:2]
  store volatile i8 %addr, i8* @mem_addr, align 1, !dbg !107 ; [debug line = 84:2]
  store volatile i8 %data, i8* @mem_dout, align 1, !dbg !108 ; [debug line = 85:2]
  store volatile i1 false, i1* @mem_wreq, align 1, !dbg !109 ; [debug line = 86:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !110 ; [debug line = 87:2]
  br label %._crit_edge, !dbg !111                ; [debug line = 89:2]

._crit_edge:                                      ; preds = %._crit_edge, %0
  store volatile i8 %addr, i8* @mem_addr, align 1, !dbg !112 ; [debug line = 90:3]
  store volatile i8 %data, i8* @mem_dout, align 1, !dbg !114 ; [debug line = 91:3]
  store volatile i1 true, i1* @mem_wreq, align 1, !dbg !115 ; [debug line = 92:3]
  %mem_wack.load = load volatile i1* @mem_wack, align 1, !dbg !116 ; [#uses=1 type=i1] [debug line = 93:2]
  br i1 %mem_wack.load, label %1, label %._crit_edge, !dbg !116 ; [debug line = 93:2]

; <label>:1                                       ; preds = %._crit_edge
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !117 ; [debug line = 94:2]
  store volatile i8 %addr, i8* @mem_addr, align 1, !dbg !118 ; [debug line = 96:2]
  store volatile i8 %data, i8* @mem_dout, align 1, !dbg !119 ; [debug line = 97:2]
  store volatile i1 false, i1* @mem_wreq, align 1, !dbg !120 ; [debug line = 98:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !121 ; [debug line = 99:2]
  ret void, !dbg !122                             ; [debug line = 100:1]
}

; [#uses=2]
define internal fastcc zeroext i8 @read_mem(i8 zeroext %addr) nounwind uwtable {
  call void @llvm.dbg.value(metadata !{i8 %addr}, i64 0, metadata !123), !dbg !127 ; [debug line = 103:22] [debug variable = addr]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !128 ; [debug line = 108:2]
  store volatile i8 %addr, i8* @mem_addr, align 1, !dbg !130 ; [debug line = 109:2]
  store volatile i1 true, i1* @mem_rreq, align 1, !dbg !131 ; [debug line = 110:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !132 ; [debug line = 111:2]
  br label %._crit_edge, !dbg !133                ; [debug line = 113:2]

._crit_edge:                                      ; preds = %._crit_edge, %0
  store volatile i8 %addr, i8* @mem_addr, align 1, !dbg !134 ; [debug line = 114:3]
  store volatile i1 true, i1* @mem_rreq, align 1, !dbg !136 ; [debug line = 115:3]
  %dt = load volatile i8* @mem_din, align 1, !dbg !137 ; [#uses=1 type=i8] [debug line = 116:3]
  call void @llvm.dbg.value(metadata !{i8 %dt}, i64 0, metadata !138), !dbg !137 ; [debug line = 116:3] [debug variable = dt]
  %mem_rack.load = load volatile i1* @mem_rack, align 1, !dbg !139 ; [#uses=1 type=i1] [debug line = 117:2]
  br i1 %mem_rack.load, label %1, label %._crit_edge, !dbg !139 ; [debug line = 117:2]

; <label>:1                                       ; preds = %._crit_edge
  %dt.lcssa = phi i8 [ %dt, %._crit_edge ]        ; [#uses=1 type=i8]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !140 ; [debug line = 118:2]
  store volatile i8 %addr, i8* @mem_addr, align 1, !dbg !141 ; [debug line = 120:2]
  store volatile i1 false, i1* @mem_rreq, align 1, !dbg !142 ; [debug line = 121:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !143 ; [debug line = 122:2]
  ret i8 %dt.lcssa, !dbg !144                     ; [debug line = 124:2]
}

; [#uses=70]
declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

; [#uses=0]
define void @i2c_slave_core() noreturn nounwind uwtable {
  call void (...)* @_ssdm_op_SpecTopModule(), !dbg !145 ; [debug line = 133:1]
  %i2c_in.load = load volatile i2* @i2c_in, align 1, !dbg !150 ; [#uses=0 type=i2] [debug line = 134:1]
  call void (...)* @_ssdm_op_SpecInterface(i2* @i2c_in, [8 x i8]* @.str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !150 ; [debug line = 134:1]
  %i2c_sda_out.load = load volatile i1* @i2c_sda_out, align 1, !dbg !151 ; [#uses=0 type=i1] [debug line = 135:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @i2c_sda_out, [8 x i8]* @.str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !151 ; [debug line = 135:1]
  %i2c_sda_oe.load = load volatile i1* @i2c_sda_oe, align 1, !dbg !152 ; [#uses=0 type=i1] [debug line = 136:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @i2c_sda_oe, [8 x i8]* @.str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !152 ; [debug line = 136:1]
  %dev_addr_in.load = load volatile i7* @dev_addr_in, align 1, !dbg !153 ; [#uses=0 type=i7] [debug line = 138:1]
  call void (...)* @_ssdm_op_SpecInterface(i7* @dev_addr_in, [8 x i8]* @.str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !153 ; [debug line = 138:1]
  %auto_inc_regad_in.load = load volatile i1* @auto_inc_regad_in, align 1, !dbg !154 ; [#uses=0 type=i1] [debug line = 139:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @auto_inc_regad_in, [8 x i8]* @.str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !154 ; [debug line = 139:1]
  %mem_addr.load = load volatile i8* @mem_addr, align 1, !dbg !155 ; [#uses=0 type=i8] [debug line = 141:1]
  call void (...)* @_ssdm_op_SpecInterface(i8* @mem_addr, [8 x i8]* @.str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !155 ; [debug line = 141:1]
  %mem_din.load = load volatile i8* @mem_din, align 1, !dbg !156 ; [#uses=0 type=i8] [debug line = 142:1]
  call void (...)* @_ssdm_op_SpecInterface(i8* @mem_din, [8 x i8]* @.str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !156 ; [debug line = 142:1]
  %mem_dout.load = load volatile i8* @mem_dout, align 1, !dbg !157 ; [#uses=0 type=i8] [debug line = 143:1]
  call void (...)* @_ssdm_op_SpecInterface(i8* @mem_dout, [8 x i8]* @.str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !157 ; [debug line = 143:1]
  %mem_wreq.load = load volatile i1* @mem_wreq, align 1, !dbg !158 ; [#uses=0 type=i1] [debug line = 144:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @mem_wreq, [8 x i8]* @.str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !158 ; [debug line = 144:1]
  %mem_wack.load = load volatile i1* @mem_wack, align 1, !dbg !159 ; [#uses=0 type=i1] [debug line = 145:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @mem_wack, [8 x i8]* @.str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !159 ; [debug line = 145:1]
  %mem_rreq.load = load volatile i1* @mem_rreq, align 1, !dbg !160 ; [#uses=0 type=i1] [debug line = 146:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @mem_rreq, [8 x i8]* @.str1, i32 1, i32 1, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !160 ; [debug line = 146:1]
  %mem_rack.load = load volatile i1* @mem_rack, align 1, !dbg !161 ; [#uses=0 type=i1] [debug line = 147:1]
  call void (...)* @_ssdm_op_SpecInterface(i1* @mem_rack, [8 x i8]* @.str1, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str, [1 x i8]* @.str, [1 x i8]* @.str, i32 0, i32 0, i32 0, i32 0, [1 x i8]* @.str) nounwind, !dbg !161 ; [debug line = 147:1]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !162 ; [debug line = 158:2]
  store volatile i1 true, i1* @i2c_sda_out, align 1, !dbg !163 ; [debug line = 159:2]
  store volatile i1 false, i1* @i2c_sda_oe, align 1, !dbg !164 ; [debug line = 160:2]
  store volatile i8 0, i8* @mem_addr, align 1, !dbg !165 ; [debug line = 161:2]
  store volatile i8 0, i8* @mem_dout, align 1, !dbg !166 ; [debug line = 162:2]
  store volatile i1 false, i1* @mem_wreq, align 1, !dbg !167 ; [debug line = 163:2]
  store volatile i1 false, i1* @mem_rreq, align 1, !dbg !168 ; [debug line = 164:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !169 ; [debug line = 165:2]
  br label %.backedge, !dbg !170                  ; [debug line = 168:13]

.backedge.loopexit:                               ; preds = %._crit_edge101
  %re.7.lcssa = phi i8 [ %re.7, %._crit_edge101 ] ; [#uses=1 type=i8]
  %reg_data.9.lcssa = phi i8 [ %reg_data.9, %._crit_edge101 ] ; [#uses=1 type=i8]
  %__Val2__.31.lcssa = phi i2 [ %__Val2__.33, %._crit_edge101 ] ; [#uses=1 type=i2]
  store i2 %__Val2__.31.lcssa, i2* @i2c_val, align 1, !dbg !172 ; [debug line = 76:2@438:6]
  br label %.backedge.backedge

.backedge.loopexit83:                             ; preds = %.preheader
  %re.7.lcssa8 = phi i8 [ %re.7, %.preheader ]    ; [#uses=1 type=i8]
  %reg_data.4.lcssa7 = phi i8 [ %reg_data.4, %.preheader ] ; [#uses=1 type=i8]
  %__Val2__.29.lcssa = phi i2 [ %__Val2__.30, %.preheader ] ; [#uses=1 type=i2]
  store i2 %__Val2__.29.lcssa, i2* @i2c_val, align 1, !dbg !180 ; [debug line = 76:2@430:6]
  br label %.backedge.backedge

.backedge.loopexit84:                             ; preds = %._crit_edge92
  %re.2.lcssa = phi i8 [ %re.2, %._crit_edge92 ]  ; [#uses=1 type=i8]
  %reg_data.5.lcssa = phi i8 [ %reg_data.5, %._crit_edge92 ] ; [#uses=1 type=i8]
  %__Val2__.20.lcssa = phi i2 [ %__Val2__.20, %._crit_edge92 ] ; [#uses=1 type=i2]
  store i2 %__Val2__.20.lcssa, i2* @i2c_val, align 1, !dbg !183 ; [debug line = 76:2@325:6]
  br label %.backedge.backedge

.backedge.loopexit88:                             ; preds = %18
  %__Val2__.15.lcssa1 = phi i2 [ %__Val2__.15, %18 ] ; [#uses=1 type=i2]
  store i2 %__Val2__.15.lcssa1, i2* @i2c_val, align 1, !dbg !188 ; [debug line = 76:2@293:4]
  br label %.backedge.backedge

.backedge.loopexit102:                            ; preds = %.preheader50
  %__Val2__.2.lcssa = phi i2 [ %__Val2__.2, %.preheader50 ] ; [#uses=1 type=i2]
  store i2 %__Val2__.2.lcssa, i2* @i2c_val, align 1, !dbg !191 ; [debug line = 76:2@191:4]
  br label %.backedge.backedge

.backedge.loopexit104:                            ; preds = %.preheader52
  %__Val2__.1.lcssa = phi i2 [ %__Val2__.1, %.preheader52 ] ; [#uses=1 type=i2]
  store i2 %__Val2__.1.lcssa, i2* @i2c_val, align 1, !dbg !194 ; [debug line = 76:2@184:4]
  br label %.backedge.backedge

.backedge.backedge:                               ; preds = %.backedge.loopexit104, %.backedge.loopexit102, %.backedge.loopexit88, %.backedge.loopexit84, %.backedge.loopexit83, %.backedge.loopexit
  %reg_data.be = phi i8 [ %reg_data.9.lcssa, %.backedge.loopexit ], [ %reg_data.4.lcssa7, %.backedge.loopexit83 ], [ %reg_data.5.lcssa, %.backedge.loopexit84 ], [ %reg_data.3, %.backedge.loopexit88 ], [ %reg_data, %.backedge.loopexit102 ], [ %reg_data, %.backedge.loopexit104 ] ; [#uses=1 type=i8]
  %re.be = phi i8 [ %re.7.lcssa, %.backedge.loopexit ], [ %re.7.lcssa8, %.backedge.loopexit83 ], [ %re.2.lcssa, %.backedge.loopexit84 ], [ %re.1.lcssa, %.backedge.loopexit88 ], [ %re, %.backedge.loopexit102 ], [ %re, %.backedge.loopexit104 ] ; [#uses=1 type=i8]
  %de.be = phi i7 [ %de.2.lcssa, %.backedge.loopexit ], [ %de.2.lcssa, %.backedge.loopexit83 ], [ %de.1.lcssa, %.backedge.loopexit84 ], [ %de.1.lcssa, %.backedge.loopexit88 ], [ %de, %.backedge.loopexit102 ], [ %de, %.backedge.loopexit104 ] ; [#uses=1 type=i7]
  br label %.backedge

.backedge:                                        ; preds = %.backedge.backedge, %0
  %reg_data = phi i8 [ undef, %0 ], [ %reg_data.be, %.backedge.backedge ] ; [#uses=3 type=i8]
  %re = phi i8 [ undef, %0 ], [ %re.be, %.backedge.backedge ] ; [#uses=3 type=i8]
  %de = phi i7 [ undef, %0 ], [ %de.be, %.backedge.backedge ] ; [#uses=3 type=i7]
  %loop_begin = call i32 (...)* @_ssdm_op_SpecLoopBegin() nounwind ; [#uses=0 type=i32]
  store volatile i1 true, i1* @i2c_sda_out, align 1, !dbg !197 ; [debug line = 173:3]
  store volatile i1 false, i1* @i2c_sda_oe, align 1, !dbg !198 ; [debug line = 174:3]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !199 ; [debug line = 175:3]
  br label %.critedge, !dbg !200                  ; [debug line = 178:3]

.critedge:                                        ; preds = %.critedge.backedge, %.backedge
  %__Val2__ = load volatile i2* @i2c_in, align 1, !dbg !201 ; [#uses=3 type=i2] [debug line = 76:2@179:4]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__}, i64 0, metadata !204), !dbg !206 ; [debug line = 180:51] [debug variable = __Val2__]
  %__Result__ = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__, i32 0, i32 0), !dbg !207 ; [#uses=1 type=i1] [debug line = 180:85]
  br i1 %__Result__, label %1, label %.critedge.backedge, !dbg !208 ; [debug line = 180:174]

; <label>:1                                       ; preds = %.critedge
  call void @llvm.dbg.value(metadata !{i2 %__Val2__}, i64 0, metadata !209), !dbg !211 ; [debug line = 180:224] [debug variable = __Val2__]
  %__Result__.1 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__, i32 1, i32 1), !dbg !212 ; [#uses=1 type=i1] [debug line = 180:0]
  br i1 %__Result__.1, label %.preheader52.preheader, label %.critedge.backedge, !dbg !212 ; [debug line = 180:0]

.preheader52.preheader:                           ; preds = %1
  %__Val2__.lcssa = phi i2 [ %__Val2__, %1 ]      ; [#uses=1 type=i2]
  store i2 %__Val2__.lcssa, i2* @i2c_val, align 1, !dbg !201 ; [debug line = 76:2@179:4]
  br label %.preheader52, !dbg !194               ; [debug line = 76:2@184:4]

.critedge.backedge:                               ; preds = %1, %.critedge
  br label %.critedge

.preheader52:                                     ; preds = %2, %.preheader52.preheader
  %__Val2__.1 = load volatile i2* @i2c_in, align 1, !dbg !194 ; [#uses=4 type=i2] [debug line = 76:2@184:4]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.1}, i64 0, metadata !213), !dbg !215 ; [debug line = 185:47] [debug variable = __Val2__]
  %__Result__.2 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.1, i32 0, i32 0), !dbg !216 ; [#uses=1 type=i1] [debug line = 185:81]
  br i1 %__Result__.2, label %2, label %.backedge.loopexit104, !dbg !217 ; [debug line = 185:170]

; <label>:2                                       ; preds = %.preheader52
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.1}, i64 0, metadata !218), !dbg !220 ; [debug line = 187:51] [debug variable = __Val2__]
  %__Result__.3 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.1, i32 1, i32 1), !dbg !221 ; [#uses=1 type=i1] [debug line = 187:85]
  br i1 %__Result__.3, label %.preheader52, label %.preheader50.preheader, !dbg !222 ; [debug line = 187:174]

.preheader50.preheader:                           ; preds = %2
  %__Val2__.1.lcssa1 = phi i2 [ %__Val2__.1, %2 ] ; [#uses=1 type=i2]
  store i2 %__Val2__.1.lcssa1, i2* @i2c_val, align 1, !dbg !194 ; [debug line = 76:2@184:4]
  br label %.preheader50, !dbg !191               ; [debug line = 76:2@191:4]

.preheader50:                                     ; preds = %3, %.preheader50.preheader
  %__Val2__.2 = load volatile i2* @i2c_in, align 1, !dbg !191 ; [#uses=4 type=i2] [debug line = 76:2@191:4]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.2}, i64 0, metadata !223), !dbg !225 ; [debug line = 192:47] [debug variable = __Val2__]
  %__Result__.4 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.2, i32 1, i32 1), !dbg !226 ; [#uses=1 type=i1] [debug line = 192:81]
  br i1 %__Result__.4, label %.backedge.loopexit102, label %3, !dbg !227 ; [debug line = 192:170]

; <label>:3                                       ; preds = %.preheader50
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.2}, i64 0, metadata !228), !dbg !230 ; [debug line = 194:51] [debug variable = __Val2__]
  %__Result__.5 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.2, i32 0, i32 0), !dbg !231 ; [#uses=1 type=i1] [debug line = 194:85]
  br i1 %__Result__.5, label %.preheader50, label %.preheader49.preheader, !dbg !232 ; [debug line = 194:174]

.preheader49.preheader:                           ; preds = %3
  %__Val2__.2.lcssa1 = phi i2 [ %__Val2__.2, %3 ] ; [#uses=1 type=i2]
  store i2 %__Val2__.2.lcssa1, i2* @i2c_val, align 1, !dbg !191 ; [debug line = 76:2@191:4]
  br label %.preheader49, !dbg !233               ; [debug line = 199:8]

.preheader49:                                     ; preds = %6, %.preheader49.preheader
  %bit_cnt = phi i3 [ %bit_cnt.6, %6 ], [ 0, %.preheader49.preheader ] ; [#uses=2 type=i3]
  %de.1 = phi i7 [ %dev_addr, %6 ], [ %de, %.preheader49.preheader ] ; [#uses=2 type=i7]
  %exitcond1 = icmp eq i3 %bit_cnt, -1, !dbg !233 ; [#uses=1 type=i1] [debug line = 199:8]
  %4 = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 7, i64 7, i64 7) nounwind ; [#uses=0 type=i32]
  br i1 %exitcond1, label %7, label %.preheader48.preheader, !dbg !233 ; [debug line = 199:8]

.preheader48.preheader:                           ; preds = %.preheader49
  br label %.preheader48, !dbg !235               ; [debug line = 76:2@202:5]

.preheader48:                                     ; preds = %.preheader48, %.preheader48.preheader
  %__Val2__.4 = load volatile i2* @i2c_in, align 1, !dbg !235 ; [#uses=2 type=i2] [debug line = 76:2@202:5]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.4}, i64 0, metadata !239), !dbg !241 ; [debug line = 203:52] [debug variable = __Val2__]
  %__Result__. = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.4, i32 0, i32 0), !dbg !242 ; [#uses=1 type=i1] [debug line = 203:86]
  br i1 %__Result__., label %5, label %.preheader48, !dbg !243 ; [debug line = 203:175]

; <label>:5                                       ; preds = %.preheader48
  %__Val2__.4.lcssa = phi i2 [ %__Val2__.4, %.preheader48 ] ; [#uses=2 type=i2]
  store i2 %__Val2__.4.lcssa, i2* @i2c_val, align 1, !dbg !235 ; [debug line = 76:2@202:5]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.4}, i64 0, metadata !244), !dbg !246 ; [debug line = 205:72] [debug variable = __Val2__]
  %__Result__1 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.4.lcssa, i32 1, i32 1), !dbg !247 ; [#uses=1 type=i1] [debug line = 205:106]
  %tmp = call i6 @_ssdm_op_PartSelect.i6.i7.i32.i32(i7 %de.1, i32 0, i32 5) ; [#uses=1 type=i6]
  %dev_addr = call i7 @_ssdm_op_BitConcatenate.i7.i6.i1(i6 %tmp, i1 %__Result__1), !dbg !248 ; [#uses=1 type=i7] [debug line = 205:195]
  call void @llvm.dbg.value(metadata !{i7 %dev_addr}, i64 0, metadata !249), !dbg !248 ; [debug line = 205:195] [debug variable = dev_addr]
  br label %._crit_edge, !dbg !250                ; [debug line = 208:4]

._crit_edge:                                      ; preds = %._crit_edge, %5
  %__Val2__.6 = load volatile i2* @i2c_in, align 1, !dbg !251 ; [#uses=2 type=i2] [debug line = 76:2@209:5]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.6}, i64 0, metadata !254), !dbg !256 ; [debug line = 210:52] [debug variable = __Val2__]
  %__Result__2 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.6, i32 0, i32 0), !dbg !257 ; [#uses=1 type=i1] [debug line = 210:86]
  br i1 %__Result__2, label %._crit_edge, label %6, !dbg !258 ; [debug line = 210:175]

; <label>:6                                       ; preds = %._crit_edge
  %__Val2__.6.lcssa = phi i2 [ %__Val2__.6, %._crit_edge ] ; [#uses=1 type=i2]
  store i2 %__Val2__.6.lcssa, i2* @i2c_val, align 1, !dbg !251 ; [debug line = 76:2@209:5]
  %bit_cnt.6 = add i3 %bit_cnt, 1, !dbg !259      ; [#uses=1 type=i3] [debug line = 199:34]
  call void @llvm.dbg.value(metadata !{i3 %bit_cnt.6}, i64 0, metadata !260), !dbg !259 ; [debug line = 199:34] [debug variable = bit_cnt]
  br label %.preheader49, !dbg !259               ; [debug line = 199:34]

; <label>:7                                       ; preds = %.preheader49
  %de.1.lcssa = phi i7 [ %de.1, %.preheader49 ]   ; [#uses=4 type=i7]
  %dev_addr_in.load.1 = load volatile i7* @dev_addr_in, align 1, !dbg !263 ; [#uses=1 type=i7] [debug line = 214:3]
  %not. = icmp ne i7 %de.1.lcssa, %dev_addr_in.load.1, !dbg !263 ; [#uses=1 type=i1] [debug line = 214:3]
  br label %._crit_edge84, !dbg !264              ; [debug line = 220:3]

._crit_edge84:                                    ; preds = %._crit_edge84, %7
  %__Val2__.3 = load volatile i2* @i2c_in, align 1, !dbg !265 ; [#uses=2 type=i2] [debug line = 76:2@221:4]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.3}, i64 0, metadata !268), !dbg !270 ; [debug line = 222:51] [debug variable = __Val2__]
  %__Result__.6 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.3, i32 0, i32 0), !dbg !271 ; [#uses=1 type=i1] [debug line = 222:85]
  br i1 %__Result__.6, label %8, label %._crit_edge84, !dbg !272 ; [debug line = 222:174]

; <label>:8                                       ; preds = %._crit_edge84
  %__Val2__.3.lcssa = phi i2 [ %__Val2__.3, %._crit_edge84 ] ; [#uses=2 type=i2]
  store i2 %__Val2__.3.lcssa, i2* @i2c_val, align 1, !dbg !265 ; [debug line = 76:2@221:4]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.3}, i64 0, metadata !273), !dbg !275 ; [debug line = 224:46] [debug variable = __Val2__]
  %__Result__.7 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.3.lcssa, i32 1, i32 1), !dbg !276 ; [#uses=1 type=i1] [debug line = 224:80]
  %ignore.0. = or i1 %not., %__Result__.7, !dbg !277 ; [#uses=6 type=i1] [debug line = 224:169]
  br label %._crit_edge85, !dbg !278              ; [debug line = 227:3]

._crit_edge85:                                    ; preds = %._crit_edge85, %8
  %__Val2__.5 = load volatile i2* @i2c_in, align 1, !dbg !279 ; [#uses=2 type=i2] [debug line = 76:2@228:4]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.5}, i64 0, metadata !282), !dbg !284 ; [debug line = 229:51] [debug variable = __Val2__]
  %__Result__3 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.5, i32 0, i32 0), !dbg !285 ; [#uses=1 type=i1] [debug line = 229:85]
  br i1 %__Result__3, label %._crit_edge85, label %9, !dbg !286 ; [debug line = 229:174]

; <label>:9                                       ; preds = %._crit_edge85
  %__Val2__.5.lcssa = phi i2 [ %__Val2__.5, %._crit_edge85 ] ; [#uses=1 type=i2]
  store i2 %__Val2__.5.lcssa, i2* @i2c_val, align 1, !dbg !279 ; [debug line = 76:2@228:4]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !287 ; [debug line = 232:3]
  store volatile i1 %ignore.0., i1* @i2c_sda_out, align 1, !dbg !288 ; [debug line = 233:3]
  %not.ignore.1 = xor i1 %ignore.0., true, !dbg !289 ; [#uses=3 type=i1] [debug line = 234:3]
  store volatile i1 %not.ignore.1, i1* @i2c_sda_oe, align 1, !dbg !289 ; [debug line = 234:3]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !290 ; [debug line = 235:3]
  br label %._crit_edge86, !dbg !291              ; [debug line = 237:3]

._crit_edge86:                                    ; preds = %._crit_edge86, %9
  %__Val2__.7 = load volatile i2* @i2c_in, align 1, !dbg !292 ; [#uses=2 type=i2] [debug line = 76:2@238:4]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.7}, i64 0, metadata !295), !dbg !297 ; [debug line = 239:51] [debug variable = __Val2__]
  %__Result__4 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.7, i32 0, i32 0), !dbg !298 ; [#uses=1 type=i1] [debug line = 239:85]
  br i1 %__Result__4, label %.preheader47.preheader, label %._crit_edge86, !dbg !299 ; [debug line = 239:174]

.preheader47.preheader:                           ; preds = %._crit_edge86
  %__Val2__.7.lcssa = phi i2 [ %__Val2__.7, %._crit_edge86 ] ; [#uses=1 type=i2]
  store i2 %__Val2__.7.lcssa, i2* @i2c_val, align 1, !dbg !292 ; [debug line = 76:2@238:4]
  br label %.preheader47, !dbg !300               ; [debug line = 76:2@242:4]

.preheader47:                                     ; preds = %.preheader47, %.preheader47.preheader
  %__Val2__.8 = load volatile i2* @i2c_in, align 1, !dbg !300 ; [#uses=2 type=i2] [debug line = 76:2@242:4]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.8}, i64 0, metadata !303), !dbg !305 ; [debug line = 243:51] [debug variable = __Val2__]
  %__Result__5 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.8, i32 0, i32 0), !dbg !306 ; [#uses=1 type=i1] [debug line = 243:85]
  br i1 %__Result__5, label %.preheader47, label %10, !dbg !307 ; [debug line = 243:174]

; <label>:10                                      ; preds = %.preheader47
  %__Val2__.8.lcssa = phi i2 [ %__Val2__.8, %.preheader47 ] ; [#uses=1 type=i2]
  store i2 %__Val2__.8.lcssa, i2* @i2c_val, align 1, !dbg !300 ; [debug line = 76:2@242:4]
  store volatile i1 false, i1* @i2c_sda_oe, align 1, !dbg !308 ; [debug line = 245:3]
  br label %11, !dbg !309                         ; [debug line = 250:8]

; <label>:11                                      ; preds = %14, %10
  %bit_cnt.1 = phi i4 [ 0, %10 ], [ %bit_cnt.7, %14 ] ; [#uses=2 type=i4]
  %re.1 = phi i8 [ %re, %10 ], [ %reg_addr, %14 ] ; [#uses=2 type=i8]
  %exitcond2 = icmp eq i4 %bit_cnt.1, -8, !dbg !309 ; [#uses=1 type=i1] [debug line = 250:8]
  %12 = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 8, i64 8, i64 8) nounwind ; [#uses=0 type=i32]
  br i1 %exitcond2, label %15, label %.preheader46.preheader, !dbg !309 ; [debug line = 250:8]

.preheader46.preheader:                           ; preds = %11
  br label %.preheader46, !dbg !311               ; [debug line = 76:2@253:5]

.preheader46:                                     ; preds = %.preheader46, %.preheader46.preheader
  %__Val2__.10 = load volatile i2* @i2c_in, align 1, !dbg !311 ; [#uses=2 type=i2] [debug line = 76:2@253:5]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.10}, i64 0, metadata !315), !dbg !317 ; [debug line = 254:52] [debug variable = __Val2__]
  %__Result__.8 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.10, i32 0, i32 0), !dbg !318 ; [#uses=1 type=i1] [debug line = 254:86]
  br i1 %__Result__.8, label %13, label %.preheader46, !dbg !319 ; [debug line = 254:175]

; <label>:13                                      ; preds = %.preheader46
  %__Val2__.10.lcssa = phi i2 [ %__Val2__.10, %.preheader46 ] ; [#uses=2 type=i2]
  store i2 %__Val2__.10.lcssa, i2* @i2c_val, align 1, !dbg !311 ; [debug line = 76:2@253:5]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.10}, i64 0, metadata !320), !dbg !322 ; [debug line = 256:72] [debug variable = __Val2__]
  %__Result__.9 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.10.lcssa, i32 1, i32 1), !dbg !323 ; [#uses=1 type=i1] [debug line = 256:106]
  %tmp.1 = call i7 @_ssdm_op_PartSelect.i7.i8.i32.i32(i8 %re.1, i32 0, i32 6) ; [#uses=1 type=i7]
  %reg_addr = call i8 @_ssdm_op_BitConcatenate.i8.i7.i1(i7 %tmp.1, i1 %__Result__.9), !dbg !324 ; [#uses=1 type=i8] [debug line = 256:195]
  call void @llvm.dbg.value(metadata !{i8 %reg_addr}, i64 0, metadata !325), !dbg !324 ; [debug line = 256:195] [debug variable = reg_addr]
  br label %._crit_edge87, !dbg !326              ; [debug line = 259:4]

._crit_edge87:                                    ; preds = %._crit_edge87, %13
  %__Val2__.12 = load volatile i2* @i2c_in, align 1, !dbg !327 ; [#uses=2 type=i2] [debug line = 76:2@260:5]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.12}, i64 0, metadata !330), !dbg !332 ; [debug line = 261:52] [debug variable = __Val2__]
  %__Result__7 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.12, i32 0, i32 0), !dbg !333 ; [#uses=1 type=i1] [debug line = 261:86]
  br i1 %__Result__7, label %._crit_edge87, label %14, !dbg !334 ; [debug line = 261:175]

; <label>:14                                      ; preds = %._crit_edge87
  %__Val2__.12.lcssa = phi i2 [ %__Val2__.12, %._crit_edge87 ] ; [#uses=1 type=i2]
  store i2 %__Val2__.12.lcssa, i2* @i2c_val, align 1, !dbg !327 ; [debug line = 76:2@260:5]
  %bit_cnt.7 = add i4 %bit_cnt.1, 1, !dbg !335    ; [#uses=1 type=i4] [debug line = 250:34]
  call void @llvm.dbg.value(metadata !{i4 %bit_cnt.7}, i64 0, metadata !260), !dbg !335 ; [debug line = 250:34] [debug variable = bit_cnt]
  br label %11, !dbg !335                         ; [debug line = 250:34]

; <label>:15                                      ; preds = %11
  %re.1.lcssa = phi i8 [ %re.1, %11 ]             ; [#uses=6 type=i8]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !336 ; [debug line = 265:3]
  store volatile i1 %ignore.0., i1* @i2c_sda_out, align 1, !dbg !337 ; [debug line = 266:3]
  store volatile i1 %not.ignore.1, i1* @i2c_sda_oe, align 1, !dbg !338 ; [debug line = 267:3]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !339 ; [debug line = 268:3]
  br label %._crit_edge88, !dbg !340              ; [debug line = 270:3]

._crit_edge88:                                    ; preds = %._crit_edge88, %15
  %__Val2__.9 = load volatile i2* @i2c_in, align 1, !dbg !341 ; [#uses=2 type=i2] [debug line = 76:2@271:4]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.9}, i64 0, metadata !344), !dbg !346 ; [debug line = 272:51] [debug variable = __Val2__]
  %__Result__.10 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.9, i32 0, i32 0), !dbg !347 ; [#uses=1 type=i1] [debug line = 272:85]
  br i1 %__Result__.10, label %.preheader45.preheader, label %._crit_edge88, !dbg !348 ; [debug line = 272:174]

.preheader45.preheader:                           ; preds = %._crit_edge88
  %__Val2__.9.lcssa = phi i2 [ %__Val2__.9, %._crit_edge88 ] ; [#uses=1 type=i2]
  store i2 %__Val2__.9.lcssa, i2* @i2c_val, align 1, !dbg !341 ; [debug line = 76:2@271:4]
  br label %.preheader45, !dbg !349               ; [debug line = 76:2@275:4]

.preheader45:                                     ; preds = %.preheader45, %.preheader45.preheader
  %__Val2__.11 = load volatile i2* @i2c_in, align 1, !dbg !349 ; [#uses=2 type=i2] [debug line = 76:2@275:4]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.11}, i64 0, metadata !352), !dbg !354 ; [debug line = 276:51] [debug variable = __Val2__]
  %__Result__6 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.11, i32 0, i32 0), !dbg !355 ; [#uses=1 type=i1] [debug line = 276:85]
  br i1 %__Result__6, label %.preheader45, label %16, !dbg !356 ; [debug line = 276:174]

; <label>:16                                      ; preds = %.preheader45
  %__Val2__.11.lcssa = phi i2 [ %__Val2__.11, %.preheader45 ] ; [#uses=1 type=i2]
  store i2 %__Val2__.11.lcssa, i2* @i2c_val, align 1, !dbg !349 ; [debug line = 76:2@275:4]
  store volatile i1 false, i1* @i2c_sda_oe, align 1, !dbg !357 ; [debug line = 278:3]
  br label %._crit_edge89, !dbg !358              ; [debug line = 285:3]

._crit_edge89:                                    ; preds = %._crit_edge89, %16
  %__Val2__.13 = load volatile i2* @i2c_in, align 1, !dbg !359 ; [#uses=2 type=i2] [debug line = 76:2@286:4]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.13}, i64 0, metadata !362), !dbg !364 ; [debug line = 287:51] [debug variable = __Val2__]
  %__Result__8 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.13, i32 0, i32 0), !dbg !365 ; [#uses=1 type=i1] [debug line = 287:85]
  br i1 %__Result__8, label %17, label %._crit_edge89, !dbg !366 ; [debug line = 287:174]

; <label>:17                                      ; preds = %._crit_edge89
  %__Val2__.13.lcssa = phi i2 [ %__Val2__.13, %._crit_edge89 ] ; [#uses=3 type=i2]
  store i2 %__Val2__.13.lcssa, i2* @i2c_val, align 1, !dbg !359 ; [debug line = 76:2@286:4]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.13}, i64 0, metadata !367), !dbg !369 ; [debug line = 289:71] [debug variable = __Val2__]
  %__Result__.11 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.13.lcssa, i32 1, i32 1), !dbg !370 ; [#uses=1 type=i1] [debug line = 289:105]
  %tmp.2 = call i7 @_ssdm_op_PartSelect.i7.i8.i32.i32(i8 %reg_data, i32 0, i32 6) ; [#uses=1 type=i7]
  %reg_data.3 = call i8 @_ssdm_op_BitConcatenate.i8.i7.i1(i7 %tmp.2, i1 %__Result__.11), !dbg !371 ; [#uses=2 type=i8] [debug line = 289:194]
  call void @llvm.dbg.value(metadata !{i8 %reg_data.3}, i64 0, metadata !372), !dbg !371 ; [debug line = 289:194] [debug variable = reg_data]
  br label %._crit_edge91, !dbg !373              ; [debug line = 291:3]

._crit_edge91:                                    ; preds = %19, %17
  %__Val2__.14 = phi i2 [ %__Val2__.15, %19 ], [ %__Val2__.13.lcssa, %17 ], !dbg !374 ; [#uses=1 type=i2] [debug line = 292:61]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.14}, i64 0, metadata !376), !dbg !374 ; [debug line = 292:61] [debug variable = __Val2__]
  %pre_i2c_sda_val = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.14, i32 1, i32 1), !dbg !377 ; [#uses=2 type=i1] [debug line = 292:95]
  call void @llvm.dbg.value(metadata !{i1 %pre_i2c_sda_val}, i64 0, metadata !378), !dbg !379 ; [debug line = 292:184] [debug variable = pre_i2c_sda_val]
  %__Val2__.15 = load volatile i2* @i2c_in, align 1, !dbg !188 ; [#uses=6 type=i2] [debug line = 76:2@293:4]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.15}, i64 0, metadata !380), !dbg !382 ; [debug line = 295:47] [debug variable = __Val2__]
  %__Result__9 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.15, i32 0, i32 0), !dbg !383 ; [#uses=1 type=i1] [debug line = 295:81]
  br i1 %__Result__9, label %18, label %.preheader40.preheader, !dbg !384 ; [debug line = 295:170]

.preheader40.preheader:                           ; preds = %._crit_edge91
  %__Val2__.15.lcssa = phi i2 [ %__Val2__.15, %._crit_edge91 ] ; [#uses=1 type=i2]
  store i2 %__Val2__.15.lcssa, i2* @i2c_val, align 1, !dbg !188 ; [debug line = 76:2@293:4]
  br label %.preheader40, !dbg !385               ; [debug line = 314:4]

; <label>:18                                      ; preds = %._crit_edge91
  br i1 %ignore.0., label %.backedge.loopexit88, label %19, !dbg !386 ; [debug line = 297:4]

; <label>:19                                      ; preds = %18
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.15}, i64 0, metadata !387), !dbg !389 ; [debug line = 297:62] [debug variable = __Val2__]
  %__Result__10 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.15, i32 1, i32 1), !dbg !390 ; [#uses=1 type=i1] [debug line = 297:96]
  %.not1 = xor i1 %pre_i2c_sda_val, true, !dbg !391 ; [#uses=1 type=i1] [debug line = 297:185]
  %brmerge = or i1 %__Result__10, %.not1, !dbg !391 ; [#uses=1 type=i1] [debug line = 297:185]
  br i1 %brmerge, label %._crit_edge91, label %.preheader41.preheader, !dbg !391 ; [debug line = 297:185]

.preheader41.preheader:                           ; preds = %19
  %__Val2__.15.lcssa2 = phi i2 [ %__Val2__.15, %19 ] ; [#uses=1 type=i2]
  %pre_i2c_sda_val.lcssa9 = phi i1 [ %pre_i2c_sda_val, %19 ] ; [#uses=1 type=i1]
  store i2 %__Val2__.15.lcssa2, i2* @i2c_val, align 1, !dbg !188 ; [debug line = 76:2@293:4]
  br label %.preheader41, !dbg !392               ; [debug line = 76:2@299:6]

.preheader41:                                     ; preds = %.preheader41, %.preheader41.preheader
  %__Val2__.16 = load volatile i2* @i2c_in, align 1, !dbg !392 ; [#uses=2 type=i2] [debug line = 76:2@299:6]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.16}, i64 0, metadata !396), !dbg !398 ; [debug line = 300:53] [debug variable = __Val2__]
  %__Result__11 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.16, i32 0, i32 0), !dbg !399 ; [#uses=1 type=i1] [debug line = 300:87]
  br i1 %__Result__11, label %.preheader41, label %.preheader34.preheader, !dbg !400 ; [debug line = 300:176]

.preheader34.preheader:                           ; preds = %.preheader41
  %__Val2__.16.lcssa = phi i2 [ %__Val2__.16, %.preheader41 ] ; [#uses=1 type=i2]
  store i2 %__Val2__.16.lcssa, i2* @i2c_val, align 1, !dbg !392 ; [debug line = 76:2@299:6]
  br label %.preheader34, !dbg !401               ; [debug line = 367:8]

.preheader40:                                     ; preds = %.preheader40.backedge, %.preheader40.preheader
  %bit_cnt.2 = phi i1 [ true, %.preheader40.preheader ], [ false, %.preheader40.backedge ] ; [#uses=1 type=i1]
  %reg_data.1 = phi i8 [ %reg_data.3, %.preheader40.preheader ], [ %reg_data.2.lcssa, %.preheader40.backedge ] ; [#uses=1 type=i8]
  %re.2 = phi i8 [ %re.1.lcssa, %.preheader40.preheader ], [ %re.2.be, %.preheader40.backedge ] ; [#uses=5 type=i8]
  %bit_cnt.2.cast = zext i1 %bit_cnt.2 to i4, !dbg !385 ; [#uses=1 type=i4] [debug line = 314:4]
  br label %20, !dbg !385                         ; [debug line = 314:4]

; <label>:20                                      ; preds = %23, %.preheader40
  %bit_cnt.3 = phi i4 [ %bit_cnt.2.cast, %.preheader40 ], [ %bit_cnt.8, %23 ] ; [#uses=2 type=i4]
  %reg_data.2 = phi i8 [ %reg_data.1, %.preheader40 ], [ %reg_data.5, %23 ] ; [#uses=2 type=i8]
  %tmp. = icmp sgt i4 %bit_cnt.3, -1, !dbg !385   ; [#uses=1 type=i1] [debug line = 314:4]
  br i1 %tmp., label %.preheader37.preheader, label %24, !dbg !385 ; [debug line = 314:4]

.preheader37.preheader:                           ; preds = %20
  br label %.preheader37, !dbg !403               ; [debug line = 76:2@317:6]

.preheader37:                                     ; preds = %.preheader37, %.preheader37.preheader
  %__Val2__.17 = load volatile i2* @i2c_in, align 1, !dbg !403 ; [#uses=2 type=i2] [debug line = 76:2@317:6]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.17}, i64 0, metadata !406), !dbg !408 ; [debug line = 318:53] [debug variable = __Val2__]
  %__Result__.12 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.17, i32 0, i32 0), !dbg !409 ; [#uses=1 type=i1] [debug line = 318:87]
  br i1 %__Result__.12, label %21, label %.preheader37, !dbg !410 ; [debug line = 318:176]

; <label>:21                                      ; preds = %.preheader37
  %__Val2__.17.lcssa = phi i2 [ %__Val2__.17, %.preheader37 ] ; [#uses=3 type=i2]
  store i2 %__Val2__.17.lcssa, i2* @i2c_val, align 1, !dbg !403 ; [debug line = 76:2@317:6]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.17}, i64 0, metadata !411), !dbg !413 ; [debug line = 320:73] [debug variable = __Val2__]
  %__Result__12 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.17.lcssa, i32 1, i32 1), !dbg !414 ; [#uses=1 type=i1] [debug line = 320:107]
  %tmp.4 = call i7 @_ssdm_op_PartSelect.i7.i8.i32.i32(i8 %reg_data.2, i32 0, i32 6) ; [#uses=1 type=i7]
  %reg_data.5 = call i8 @_ssdm_op_BitConcatenate.i8.i7.i1(i7 %tmp.4, i1 %__Result__12), !dbg !415 ; [#uses=2 type=i8] [debug line = 320:196]
  call void @llvm.dbg.value(metadata !{i8 %reg_data.5}, i64 0, metadata !372), !dbg !415 ; [debug line = 320:196] [debug variable = reg_data]
  br label %._crit_edge92, !dbg !416              ; [debug line = 323:5]

._crit_edge92:                                    ; preds = %22, %21
  %__Val2__.19 = phi i2 [ %__Val2__.20, %22 ], [ %__Val2__.17.lcssa, %21 ], !dbg !417 ; [#uses=1 type=i2] [debug line = 324:63]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.19}, i64 0, metadata !419), !dbg !417 ; [debug line = 324:63] [debug variable = __Val2__]
  %__Result__13 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.19, i32 1, i32 1), !dbg !420 ; [#uses=1 type=i1] [debug line = 324:97]
  %__Val2__.20 = load volatile i2* @i2c_in, align 1, !dbg !183 ; [#uses=5 type=i2] [debug line = 76:2@325:6]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.20}, i64 0, metadata !421), !dbg !423 ; [debug line = 326:49] [debug variable = __Val2__]
  %__Result__14 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.20, i32 1, i32 1), !dbg !424 ; [#uses=1 type=i1] [debug line = 326:83]
  %tmp.6 = xor i1 %__Result__14, true, !dbg !424  ; [#uses=1 type=i1] [debug line = 326:83]
  %brmerge1 = or i1 %__Result__13, %tmp.6, !dbg !425 ; [#uses=1 type=i1] [debug line = 326:172]
  br i1 %brmerge1, label %22, label %.backedge.loopexit84, !dbg !425 ; [debug line = 326:172]

; <label>:22                                      ; preds = %._crit_edge92
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.20}, i64 0, metadata !426), !dbg !428 ; [debug line = 328:53] [debug variable = __Val2__]
  %__Result__17 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.20, i32 0, i32 0), !dbg !429 ; [#uses=1 type=i1] [debug line = 328:87]
  br i1 %__Result__17, label %._crit_edge92, label %23, !dbg !430 ; [debug line = 328:176]

; <label>:23                                      ; preds = %22
  %__Val2__.20.lcssa2 = phi i2 [ %__Val2__.20, %22 ] ; [#uses=1 type=i2]
  store i2 %__Val2__.20.lcssa2, i2* @i2c_val, align 1, !dbg !183 ; [debug line = 76:2@325:6]
  %bit_cnt.8 = add i4 %bit_cnt.3, 1, !dbg !431    ; [#uses=1 type=i4] [debug line = 330:5]
  call void @llvm.dbg.value(metadata !{i4 %bit_cnt.8}, i64 0, metadata !260), !dbg !431 ; [debug line = 330:5] [debug variable = bit_cnt]
  br label %20, !dbg !432                         ; [debug line = 331:4]

; <label>:24                                      ; preds = %20
  %reg_data.2.lcssa = phi i8 [ %reg_data.2, %20 ] ; [#uses=2 type=i8]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !433 ; [debug line = 334:4]
  store volatile i1 %ignore.0., i1* @i2c_sda_out, align 1, !dbg !434 ; [debug line = 335:4]
  store volatile i1 %not.ignore.1, i1* @i2c_sda_oe, align 1, !dbg !435 ; [debug line = 336:4]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !436 ; [debug line = 337:4]
  br label %._crit_edge93, !dbg !437              ; [debug line = 339:4]

._crit_edge93:                                    ; preds = %._crit_edge93, %24
  %__Val2__.18 = load volatile i2* @i2c_in, align 1, !dbg !438 ; [#uses=2 type=i2] [debug line = 76:2@340:5]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.18}, i64 0, metadata !441), !dbg !443 ; [debug line = 341:52] [debug variable = __Val2__]
  %__Result__.13 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.18, i32 0, i32 0), !dbg !444 ; [#uses=1 type=i1] [debug line = 341:86]
  br i1 %__Result__.13, label %.preheader35.preheader, label %._crit_edge93, !dbg !445 ; [debug line = 341:175]

.preheader35.preheader:                           ; preds = %._crit_edge93
  %__Val2__.18.lcssa = phi i2 [ %__Val2__.18, %._crit_edge93 ] ; [#uses=1 type=i2]
  store i2 %__Val2__.18.lcssa, i2* @i2c_val, align 1, !dbg !438 ; [debug line = 76:2@340:5]
  br label %.preheader35, !dbg !446               ; [debug line = 76:2@344:5]

.preheader35:                                     ; preds = %.preheader35, %.preheader35.preheader
  %__Val2__.21 = load volatile i2* @i2c_in, align 1, !dbg !446 ; [#uses=2 type=i2] [debug line = 76:2@344:5]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.21}, i64 0, metadata !449), !dbg !451 ; [debug line = 345:52] [debug variable = __Val2__]
  %__Result__15 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.21, i32 0, i32 0), !dbg !452 ; [#uses=1 type=i1] [debug line = 345:86]
  br i1 %__Result__15, label %.preheader35, label %25, !dbg !453 ; [debug line = 345:175]

; <label>:25                                      ; preds = %.preheader35
  %__Val2__.21.lcssa = phi i2 [ %__Val2__.21, %.preheader35 ] ; [#uses=1 type=i2]
  store i2 %__Val2__.21.lcssa, i2* @i2c_val, align 1, !dbg !446 ; [debug line = 76:2@344:5]
  store volatile i1 false, i1* @i2c_sda_oe, align 1, !dbg !454 ; [debug line = 347:4]
  br i1 %ignore.0., label %.preheader40.backedge, label %26, !dbg !455 ; [debug line = 350:4]

; <label>:26                                      ; preds = %25
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !456 ; [debug line = 351:5]
  call fastcc void @write_mem(i8 zeroext %re.2, i8 zeroext %reg_data.2.lcssa), !dbg !458 ; [debug line = 352:5]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !459 ; [debug line = 353:5]
  %auto_inc_regad_in.load.1 = load volatile i1* @auto_inc_regad_in, align 1, !dbg !460 ; [#uses=1 type=i1] [debug line = 354:5]
  %reg_addr.1 = add i8 %re.2, 1, !dbg !461        ; [#uses=1 type=i8] [debug line = 355:6]
  call void @llvm.dbg.value(metadata !{i8 %reg_addr.1}, i64 0, metadata !325), !dbg !461 ; [debug line = 355:6] [debug variable = reg_addr]
  %.re.2 = select i1 %auto_inc_regad_in.load.1, i8 %reg_addr.1, i8 %re.2, !dbg !460 ; [#uses=1 type=i8] [debug line = 354:5]
  br label %.preheader40.backedge, !dbg !462      ; [debug line = 356:4]

.preheader40.backedge:                            ; preds = %26, %25
  %re.2.be = phi i8 [ %.re.2, %26 ], [ %re.2, %25 ] ; [#uses=1 type=i8]
  br label %.preheader40

.preheader34:                                     ; preds = %30, %.preheader34.preheader
  %bit_cnt.4 = phi i3 [ %bit_cnt.9, %30 ], [ 0, %.preheader34.preheader ] ; [#uses=2 type=i3]
  %de.2 = phi i7 [ %dev_addr.1, %30 ], [ %de.1.lcssa, %.preheader34.preheader ] ; [#uses=2 type=i7]
  %exitcond = icmp eq i3 %bit_cnt.4, -1, !dbg !401 ; [#uses=1 type=i1] [debug line = 367:8]
  %27 = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 7, i64 7, i64 7) nounwind ; [#uses=0 type=i32]
  br i1 %exitcond, label %31, label %28, !dbg !401 ; [debug line = 367:8]

; <label>:28                                      ; preds = %.preheader34
  call void (...)* @_ssdm_op_SpecLoopName([17 x i8]* @.str5) nounwind, !dbg !463 ; [debug line = 367:46]
  br label %._crit_edge95, !dbg !465              ; [debug line = 369:4]

._crit_edge95:                                    ; preds = %._crit_edge95, %28
  %__Val2__.23 = load volatile i2* @i2c_in, align 1, !dbg !466 ; [#uses=2 type=i2] [debug line = 76:2@370:5]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.23}, i64 0, metadata !469), !dbg !471 ; [debug line = 371:52] [debug variable = __Val2__]
  %__Result__16 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.23, i32 0, i32 0), !dbg !472 ; [#uses=1 type=i1] [debug line = 371:86]
  br i1 %__Result__16, label %29, label %._crit_edge95, !dbg !473 ; [debug line = 371:175]

; <label>:29                                      ; preds = %._crit_edge95
  %__Val2__.23.lcssa = phi i2 [ %__Val2__.23, %._crit_edge95 ] ; [#uses=2 type=i2]
  store i2 %__Val2__.23.lcssa, i2* @i2c_val, align 1, !dbg !466 ; [debug line = 76:2@370:5]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.23}, i64 0, metadata !474), !dbg !476 ; [debug line = 373:72] [debug variable = __Val2__]
  %__Result__.15 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.23.lcssa, i32 1, i32 1), !dbg !477 ; [#uses=1 type=i1] [debug line = 373:106]
  %tmp.8 = call i6 @_ssdm_op_PartSelect.i6.i7.i32.i32(i7 %de.2, i32 0, i32 5) ; [#uses=1 type=i6]
  %dev_addr.1 = call i7 @_ssdm_op_BitConcatenate.i7.i6.i1(i6 %tmp.8, i1 %__Result__.15), !dbg !478 ; [#uses=1 type=i7] [debug line = 373:195]
  call void @llvm.dbg.value(metadata !{i7 %dev_addr.1}, i64 0, metadata !249), !dbg !478 ; [debug line = 373:195] [debug variable = dev_addr]
  br label %._crit_edge96, !dbg !479              ; [debug line = 376:4]

._crit_edge96:                                    ; preds = %._crit_edge96, %29
  %__Val2__.25 = load volatile i2* @i2c_in, align 1, !dbg !480 ; [#uses=2 type=i2] [debug line = 76:2@377:5]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.25}, i64 0, metadata !483), !dbg !485 ; [debug line = 378:52] [debug variable = __Val2__]
  %__Result__20 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.25, i32 0, i32 0), !dbg !486 ; [#uses=1 type=i1] [debug line = 378:86]
  br i1 %__Result__20, label %._crit_edge96, label %30, !dbg !487 ; [debug line = 378:175]

; <label>:30                                      ; preds = %._crit_edge96
  %__Val2__.25.lcssa = phi i2 [ %__Val2__.25, %._crit_edge96 ] ; [#uses=1 type=i2]
  store i2 %__Val2__.25.lcssa, i2* @i2c_val, align 1, !dbg !480 ; [debug line = 76:2@377:5]
  %bit_cnt.9 = add i3 %bit_cnt.4, 1, !dbg !488    ; [#uses=1 type=i3] [debug line = 367:34]
  call void @llvm.dbg.value(metadata !{i3 %bit_cnt.9}, i64 0, metadata !260), !dbg !488 ; [debug line = 367:34] [debug variable = bit_cnt]
  br label %.preheader34, !dbg !488               ; [debug line = 367:34]

; <label>:31                                      ; preds = %.preheader34
  %de.2.lcssa = phi i7 [ %de.2, %.preheader34 ]   ; [#uses=3 type=i7]
  %dev_addr_in.load.2 = load volatile i7* @dev_addr_in, align 1, !dbg !489 ; [#uses=1 type=i7] [debug line = 382:3]
  %not.2 = icmp ne i7 %de.2.lcssa, %dev_addr_in.load.2, !dbg !489 ; [#uses=2 type=i1] [debug line = 382:3]
  br label %._crit_edge97, !dbg !490              ; [debug line = 388:3]

._crit_edge97:                                    ; preds = %._crit_edge97, %31
  %__Val2__.22 = load volatile i2* @i2c_in, align 1, !dbg !491 ; [#uses=2 type=i2] [debug line = 76:2@389:4]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.22}, i64 0, metadata !494), !dbg !496 ; [debug line = 390:51] [debug variable = __Val2__]
  %__Result__.14 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.22, i32 0, i32 0), !dbg !497 ; [#uses=1 type=i1] [debug line = 390:85]
  br i1 %__Result__.14, label %32, label %._crit_edge97, !dbg !498 ; [debug line = 390:174]

; <label>:32                                      ; preds = %._crit_edge97
  %__Val2__.22.lcssa = phi i2 [ %__Val2__.22, %._crit_edge97 ] ; [#uses=2 type=i2]
  store i2 %__Val2__.22.lcssa, i2* @i2c_val, align 1, !dbg !491 ; [debug line = 76:2@389:4]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.22}, i64 0, metadata !499), !dbg !501 ; [debug line = 392:46] [debug variable = __Val2__]
  %__Result__18 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.22.lcssa, i32 1, i32 1), !dbg !502 ; [#uses=2 type=i1] [debug line = 392:80]
  %tmp.7 = xor i1 %__Result__18, true, !dbg !502  ; [#uses=1 type=i1] [debug line = 392:80]
  %.ignore.2 = or i1 %not.2, %tmp.7, !dbg !503    ; [#uses=4 type=i1] [debug line = 392:169]
  br label %._crit_edge98, !dbg !504              ; [debug line = 395:3]

._crit_edge98:                                    ; preds = %._crit_edge98, %32
  %__Val2__.24 = load volatile i2* @i2c_in, align 1, !dbg !505 ; [#uses=2 type=i2] [debug line = 76:2@396:4]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.24}, i64 0, metadata !508), !dbg !510 ; [debug line = 397:51] [debug variable = __Val2__]
  %__Result__19 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.24, i32 0, i32 0), !dbg !511 ; [#uses=1 type=i1] [debug line = 397:85]
  br i1 %__Result__19, label %._crit_edge98, label %_ifconv1, !dbg !512 ; [debug line = 397:174]

_ifconv1:                                         ; preds = %._crit_edge98
  %__Val2__.24.lcssa = phi i2 [ %__Val2__.24, %._crit_edge98 ] ; [#uses=1 type=i2]
  store i2 %__Val2__.24.lcssa, i2* @i2c_val, align 1, !dbg !505 ; [debug line = 76:2@396:4]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !513 ; [debug line = 400:3]
  store volatile i1 %.ignore.2, i1* @i2c_sda_out, align 1, !dbg !514 ; [debug line = 401:3]
  %not.2.not = xor i1 %not.2, true, !dbg !515     ; [#uses=1 type=i1] [debug line = 402:3]
  %not.ignore.3 = and i1 %__Result__18, %not.2.not, !dbg !515 ; [#uses=2 type=i1] [debug line = 402:3]
  store volatile i1 %not.ignore.3, i1* @i2c_sda_oe, align 1, !dbg !515 ; [debug line = 402:3]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !516 ; [debug line = 403:3]
  %reg_data.6 = call fastcc zeroext i8 @read_mem(i8 zeroext %re.1.lcssa), !dbg !517 ; [#uses=1 type=i8] [debug line = 404:14]
  call void @llvm.dbg.value(metadata !{i8 %reg_data.6}, i64 0, metadata !372), !dbg !517 ; [debug line = 404:14] [debug variable = reg_data]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !518 ; [debug line = 405:3]
  %auto_inc_regad_in.load.2 = load volatile i1* @auto_inc_regad_in, align 1, !dbg !519 ; [#uses=1 type=i1] [debug line = 406:3]
  %reg_addr.2 = add i8 %re.1.lcssa, 1, !dbg !520  ; [#uses=1 type=i8] [debug line = 407:4]
  call void @llvm.dbg.value(metadata !{i8 %reg_addr.2}, i64 0, metadata !325), !dbg !520 ; [debug line = 407:4] [debug variable = reg_addr]
  %re.1. = select i1 %.ignore.2, i8 %re.1.lcssa, i8 %reg_addr.2, !dbg !519 ; [#uses=1 type=i8] [debug line = 406:3]
  %re.6 = select i1 %auto_inc_regad_in.load.2, i8 %re.1., i8 %re.1.lcssa ; [#uses=1 type=i8]
  br label %._crit_edge100, !dbg !521             ; [debug line = 410:3]

._crit_edge100:                                   ; preds = %._crit_edge100, %_ifconv1
  %__Val2__.26 = load volatile i2* @i2c_in, align 1, !dbg !522 ; [#uses=2 type=i2] [debug line = 76:2@411:4]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.26}, i64 0, metadata !525), !dbg !527 ; [debug line = 412:51] [debug variable = __Val2__]
  %__Result__21 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.26, i32 0, i32 0), !dbg !528 ; [#uses=1 type=i1] [debug line = 412:85]
  br i1 %__Result__21, label %.preheader33.preheader, label %._crit_edge100, !dbg !529 ; [debug line = 412:174]

.preheader33.preheader:                           ; preds = %._crit_edge100
  %__Val2__.26.lcssa = phi i2 [ %__Val2__.26, %._crit_edge100 ] ; [#uses=1 type=i2]
  store i2 %__Val2__.26.lcssa, i2* @i2c_val, align 1, !dbg !522 ; [debug line = 76:2@411:4]
  br label %.preheader33, !dbg !530               ; [debug line = 76:2@415:4]

.preheader33:                                     ; preds = %.preheader33, %.preheader33.preheader
  %__Val2__.27 = load volatile i2* @i2c_in, align 1, !dbg !530 ; [#uses=2 type=i2] [debug line = 76:2@415:4]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.27}, i64 0, metadata !533), !dbg !535 ; [debug line = 416:51] [debug variable = __Val2__]
  %__Result__22 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.27, i32 0, i32 0), !dbg !536 ; [#uses=1 type=i1] [debug line = 416:85]
  br i1 %__Result__22, label %.preheader33, label %.preheader31.preheader, !dbg !537 ; [debug line = 416:174]

.preheader31.preheader:                           ; preds = %.preheader33
  %__Val2__.27.lcssa = phi i2 [ %__Val2__.27, %.preheader33 ] ; [#uses=1 type=i2]
  store i2 %__Val2__.27.lcssa, i2* @i2c_val, align 1, !dbg !530 ; [debug line = 76:2@415:4]
  br label %.preheader31, !dbg !538               ; [debug line = 421:14]

.preheader31:                                     ; preds = %41, %.preheader31.preheader
  %terminate_read = phi i1 [ %terminate_read.1, %41 ], [ false, %.preheader31.preheader ] ; [#uses=1 type=i1]
  %__Val2__.28 = phi i8 [ %reg_data.7, %41 ], [ %reg_data.6, %.preheader31.preheader ] ; [#uses=2 type=i8]
  %re.7 = phi i8 [ %re.8, %41 ], [ %re.6, %.preheader31.preheader ] ; [#uses=6 type=i8]
  %tmp.3 = call i32 (...)* @_ssdm_op_SpecRegionBegin([12 x i8]* @.str6) nounwind, !dbg !538 ; [#uses=1 type=i32] [debug line = 421:14]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !539 ; [debug line = 422:4]
  call void @llvm.dbg.value(metadata !{i8 %__Val2__.28}, i64 0, metadata !540), !dbg !542 ; [debug line = 423:59] [debug variable = __Val2__]
  %__Result__.16 = call i1 @_ssdm_op_PartSelect.i1.i8.i32.i32(i8 %__Val2__.28, i32 7, i32 7), !dbg !543 ; [#uses=1 type=i1] [debug line = 423:94]
  store volatile i1 %__Result__.16, i1* @i2c_sda_out, align 1, !dbg !544 ; [debug line = 423:183]
  store volatile i1 %not.ignore.3, i1* @i2c_sda_oe, align 1, !dbg !545 ; [debug line = 424:4]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !546 ; [debug line = 425:4]
  %brmerge2 = or i1 %.ignore.2, %terminate_read, !dbg !547 ; [#uses=1 type=i1] [debug line = 431:6]
  br label %33, !dbg !548                         ; [debug line = 428:9]

; <label>:33                                      ; preds = %._crit_edge102, %.preheader31
  %bit_cnt.5 = phi i4 [ 0, %.preheader31 ], [ %bit_cnt.10, %._crit_edge102 ] ; [#uses=3 type=i4]
  %reg_data.4 = phi i8 [ %__Val2__.28, %.preheader31 ], [ %reg_data.9, %._crit_edge102 ] ; [#uses=2 type=i8]
  %tmp.5 = icmp sgt i4 %bit_cnt.5, -1, !dbg !548  ; [#uses=1 type=i1] [debug line = 428:9]
  %34 = call i32 (...)* @_ssdm_op_SpecLoopTripCount(i64 1, i64 8, i64 4) nounwind ; [#uses=0 type=i32]
  br i1 %tmp.5, label %.preheader.preheader, label %_ifconv, !dbg !548 ; [debug line = 428:9]

.preheader.preheader:                             ; preds = %33
  br label %.preheader, !dbg !180                 ; [debug line = 76:2@430:6]

.preheader:                                       ; preds = %35, %.preheader.preheader
  %__Val2__.30 = load volatile i2* @i2c_in, align 1, !dbg !180 ; [#uses=3 type=i2] [debug line = 76:2@430:6]
  br i1 %brmerge2, label %.backedge.loopexit83, label %35, !dbg !547 ; [debug line = 431:6]

; <label>:35                                      ; preds = %.preheader
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.30}, i64 0, metadata !549), !dbg !551 ; [debug line = 433:53] [debug variable = __Val2__]
  %__Result__23 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.30, i32 0, i32 0), !dbg !552 ; [#uses=1 type=i1] [debug line = 433:87]
  br i1 %__Result__23, label %36, label %.preheader, !dbg !553 ; [debug line = 433:176]

; <label>:36                                      ; preds = %35
  %__Val2__.29.lcssa5 = phi i2 [ %__Val2__.30, %35 ] ; [#uses=1 type=i2]
  store i2 %__Val2__.29.lcssa5, i2* @i2c_val, align 1, !dbg !180 ; [debug line = 76:2@430:6]
  %tmp.9 = shl i8 %reg_data.4, 1, !dbg !554       ; [#uses=1 type=i8] [debug line = 435:5]
  %reg_data.9 = or i8 %tmp.9, 1, !dbg !554        ; [#uses=3 type=i8] [debug line = 435:5]
  call void @llvm.dbg.value(metadata !{i8 %reg_data.9}, i64 0, metadata !372), !dbg !554 ; [debug line = 435:5] [debug variable = reg_data]
  br label %._crit_edge101, !dbg !555             ; [debug line = 437:5]

._crit_edge101:                                   ; preds = %37, %36
  %__Val2__.33 = load volatile i2* @i2c_in, align 1, !dbg !172 ; [#uses=4 type=i2] [debug line = 76:2@438:6]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.33}, i64 0, metadata !556), !dbg !558 ; [debug line = 439:49] [debug variable = __Val2__]
  %__Result__.18 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.33, i32 1, i32 1), !dbg !559 ; [#uses=1 type=i1] [debug line = 439:83]
  %tmp.10 = xor i1 %__Result__.18, true, !dbg !559 ; [#uses=1 type=i1] [debug line = 439:83]
  %brmerge3 = or i1 %pre_i2c_sda_val.lcssa9, %tmp.10, !dbg !560 ; [#uses=1 type=i1] [debug line = 439:172]
  br i1 %brmerge3, label %37, label %.backedge.loopexit, !dbg !560 ; [debug line = 439:172]

; <label>:37                                      ; preds = %._crit_edge101
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.33}, i64 0, metadata !561), !dbg !563 ; [debug line = 441:53] [debug variable = __Val2__]
  %__Result__25 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.33, i32 0, i32 0), !dbg !564 ; [#uses=1 type=i1] [debug line = 441:87]
  br i1 %__Result__25, label %._crit_edge101, label %38, !dbg !565 ; [debug line = 441:176]

; <label>:38                                      ; preds = %37
  %__Val2__.31.lcssa4 = phi i2 [ %__Val2__.33, %37 ] ; [#uses=1 type=i2]
  store i2 %__Val2__.31.lcssa4, i2* @i2c_val, align 1, !dbg !172 ; [debug line = 76:2@438:6]
  %tmp.11 = icmp ult i4 %bit_cnt.5, 7, !dbg !566  ; [#uses=1 type=i1] [debug line = 443:5]
  br i1 %tmp.11, label %39, label %._crit_edge102, !dbg !566 ; [debug line = 443:5]

; <label>:39                                      ; preds = %38
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !567 ; [debug line = 444:6]
  call void @llvm.dbg.value(metadata !{i8 %reg_data.9}, i64 0, metadata !569), !dbg !571 ; [debug line = 445:61] [debug variable = __Val2__]
  %__Result__26 = call i1 @_ssdm_op_PartSelect.i1.i8.i32.i32(i8 %reg_data.9, i32 7, i32 7), !dbg !572 ; [#uses=1 type=i1] [debug line = 445:96]
  store volatile i1 %__Result__26, i1* @i2c_sda_out, align 1, !dbg !573 ; [debug line = 445:185]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !574 ; [debug line = 446:6]
  br label %._crit_edge102, !dbg !575             ; [debug line = 447:5]

._crit_edge102:                                   ; preds = %39, %38
  %bit_cnt.10 = add i4 %bit_cnt.5, 1, !dbg !576   ; [#uses=1 type=i4] [debug line = 428:35]
  call void @llvm.dbg.value(metadata !{i4 %bit_cnt.10}, i64 0, metadata !260), !dbg !576 ; [debug line = 428:35] [debug variable = bit_cnt]
  br label %33, !dbg !576                         ; [debug line = 428:35]

_ifconv:                                          ; preds = %33
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !577 ; [debug line = 450:4]
  store volatile i1 false, i1* @i2c_sda_oe, align 1, !dbg !578 ; [debug line = 451:4]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !579 ; [debug line = 452:4]
  %reg_data.7 = call fastcc zeroext i8 @read_mem(i8 zeroext %re.7), !dbg !580 ; [#uses=1 type=i8] [debug line = 453:15]
  call void @llvm.dbg.value(metadata !{i8 %reg_data.7}, i64 0, metadata !372), !dbg !580 ; [debug line = 453:15] [debug variable = reg_data]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !581 ; [debug line = 454:4]
  %auto_inc_regad_in.load.3 = load volatile i1* @auto_inc_regad_in, align 1, !dbg !582 ; [#uses=1 type=i1] [debug line = 455:4]
  %reg_addr.3 = add i8 %re.7, 1, !dbg !583        ; [#uses=1 type=i8] [debug line = 456:5]
  call void @llvm.dbg.value(metadata !{i8 %reg_addr.3}, i64 0, metadata !325), !dbg !583 ; [debug line = 456:5] [debug variable = reg_addr]
  %re.7. = select i1 %.ignore.2, i8 %re.7, i8 %reg_addr.3, !dbg !582 ; [#uses=1 type=i8] [debug line = 455:4]
  %re.8 = select i1 %auto_inc_regad_in.load.3, i8 %re.7., i8 %re.7 ; [#uses=1 type=i8]
  br label %._crit_edge104, !dbg !584             ; [debug line = 459:4]

._crit_edge104:                                   ; preds = %._crit_edge104, %_ifconv
  %__Val2__.29 = load volatile i2* @i2c_in, align 1, !dbg !585 ; [#uses=2 type=i2] [debug line = 76:2@460:5]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.29}, i64 0, metadata !588), !dbg !590 ; [debug line = 461:52] [debug variable = __Val2__]
  %__Result__.17 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.29, i32 0, i32 0), !dbg !591 ; [#uses=1 type=i1] [debug line = 461:86]
  br i1 %__Result__.17, label %40, label %._crit_edge104, !dbg !592 ; [debug line = 461:175]

; <label>:40                                      ; preds = %._crit_edge104
  %__Val2__.30.lcssa = phi i2 [ %__Val2__.29, %._crit_edge104 ] ; [#uses=2 type=i2]
  store i2 %__Val2__.30.lcssa, i2* @i2c_val, align 1, !dbg !585 ; [debug line = 76:2@460:5]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.29}, i64 0, metadata !593), !dbg !595 ; [debug line = 463:60] [debug variable = __Val2__]
  %terminate_read.1 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.30.lcssa, i32 1, i32 1), !dbg !596 ; [#uses=1 type=i1] [debug line = 463:94]
  call void @llvm.dbg.value(metadata !{i1 %terminate_read.1}, i64 0, metadata !597), !dbg !598 ; [debug line = 463:183] [debug variable = terminate_read]
  br label %._crit_edge105, !dbg !599             ; [debug line = 465:4]

._crit_edge105:                                   ; preds = %._crit_edge105, %40
  %__Val2__.32 = load volatile i2* @i2c_in, align 1, !dbg !600 ; [#uses=2 type=i2] [debug line = 76:2@466:5]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.32}, i64 0, metadata !603), !dbg !605 ; [debug line = 467:52] [debug variable = __Val2__]
  %__Result__24 = call i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2 %__Val2__.32, i32 0, i32 0), !dbg !606 ; [#uses=1 type=i1] [debug line = 467:86]
  br i1 %__Result__24, label %._crit_edge105, label %41, !dbg !607 ; [debug line = 467:175]

; <label>:41                                      ; preds = %._crit_edge105
  %__Val2__.32.lcssa = phi i2 [ %__Val2__.32, %._crit_edge105 ] ; [#uses=1 type=i2]
  store i2 %__Val2__.32.lcssa, i2* @i2c_val, align 1, !dbg !600 ; [debug line = 76:2@466:5]
  %42 = call i32 (...)* @_ssdm_op_SpecRegionEnd([12 x i8]* @.str6, i32 %tmp.3) nounwind, !dbg !608 ; [#uses=0 type=i32] [debug line = 468:3]
  br label %.preheader31, !dbg !608               ; [debug line = 468:3]
}

; [#uses=29]
declare void @_ssdm_op_Wait(...) nounwind

; [#uses=1]
declare void @_ssdm_op_SpecTopModule(...) nounwind

; [#uses=1]
declare i32 @_ssdm_op_SpecRegionEnd(...)

; [#uses=1]
declare i32 @_ssdm_op_SpecRegionBegin(...)

; [#uses=4]
declare i32 @_ssdm_op_SpecLoopTripCount(...)

; [#uses=1]
declare void @_ssdm_op_SpecLoopName(...) nounwind

; [#uses=1]
declare i32 @_ssdm_op_SpecLoopBegin(...)

; [#uses=12]
declare void @_ssdm_op_SpecInterface(...) nounwind

; [#uses=3]
declare i7 @_ssdm_op_PartSelect.i7.i8.i32.i32(i8, i32, i32) nounwind readnone

; [#uses=2]
declare i6 @_ssdm_op_PartSelect.i6.i7.i32.i32(i7, i32, i32) nounwind readnone

; [#uses=2]
declare i1 @_ssdm_op_PartSelect.i1.i8.i32.i32(i8, i32, i32) nounwind readnone

; [#uses=46]
declare i1 @_ssdm_op_PartSelect.i1.i2.i32.i32(i2, i32, i32) nounwind readnone

; [#uses=3]
declare i8 @_ssdm_op_BitConcatenate.i8.i7.i1(i7, i1) nounwind readnone

; [#uses=2]
declare i7 @_ssdm_op_BitConcatenate.i7.i6.i1(i6, i1) nounwind readnone

!hls.encrypted.func = !{}
!llvm.map.gv = !{!0, !7, !12, !17, !22, !27, !32, !37, !42, !47, !52, !57, !62}
!llvm.dbg.cu = !{!67}

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
!67 = metadata !{i32 786449, i32 0, i32 1, metadata !"D:/21_streamer_car5_artix7/fpga_arty/i2c_slave_core/solution1/.autopilot/db/i2c_slave_core.pragma.2.c", metadata !"D:\5C21_streamer_car5_artix7\5Cfpga_arty", metadata !"clang version 3.1 ", i1 true, i1 false, metadata !"", i32 0, null, null, null, metadata !68} ; [ DW_TAG_compile_unit ]
!68 = metadata !{metadata !69}
!69 = metadata !{metadata !70, metadata !75, metadata !79, metadata !82, metadata !83, metadata !84, metadata !85, metadata !86, metadata !87, metadata !88, metadata !89, metadata !91, metadata !95}
!70 = metadata !{i32 786484, i32 0, null, metadata !"mem_rreq", metadata !"mem_rreq", metadata !"", metadata !71, i32 57, metadata !72, i32 0, i32 1, i1* @mem_rreq} ; [ DW_TAG_variable ]
!71 = metadata !{i32 786473, metadata !"i2c_slave_core.c", metadata !"D:\5C21_streamer_car5_artix7\5Cfpga_arty", null} ; [ DW_TAG_file_type ]
!72 = metadata !{i32 786485, null, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !73} ; [ DW_TAG_volatile_type ]
!73 = metadata !{i32 786454, null, metadata !"uint1", metadata !71, i32 3, i64 0, i64 0, i64 0, i32 0, metadata !74} ; [ DW_TAG_typedef ]
!74 = metadata !{i32 786468, null, metadata !"uint1", null, i32 0, i64 1, i64 1, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!75 = metadata !{i32 786484, i32 0, null, metadata !"mem_addr", metadata !"mem_addr", metadata !"", metadata !71, i32 52, metadata !76, i32 0, i32 1, i8* @mem_addr} ; [ DW_TAG_variable ]
!76 = metadata !{i32 786485, null, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !77} ; [ DW_TAG_volatile_type ]
!77 = metadata !{i32 786454, null, metadata !"uint8", metadata !71, i32 10, i64 0, i64 0, i64 0, i32 0, metadata !78} ; [ DW_TAG_typedef ]
!78 = metadata !{i32 786468, null, metadata !"uint8", null, i32 0, i64 8, i64 8, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!79 = metadata !{i32 786484, i32 0, null, metadata !"i2c_val", metadata !"i2c_val", metadata !"", metadata !71, i32 67, metadata !80, i32 0, i32 1, i2* @i2c_val} ; [ DW_TAG_variable ]
!80 = metadata !{i32 786454, null, metadata !"uint2", metadata !71, i32 4, i64 0, i64 0, i64 0, i32 0, metadata !81} ; [ DW_TAG_typedef ]
!81 = metadata !{i32 786468, null, metadata !"uint2", null, i32 0, i64 2, i64 2, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!82 = metadata !{i32 786484, i32 0, null, metadata !"i2c_sda_out", metadata !"i2c_sda_out", metadata !"", metadata !71, i32 48, metadata !72, i32 0, i32 1, i1* @i2c_sda_out} ; [ DW_TAG_variable ]
!83 = metadata !{i32 786484, i32 0, null, metadata !"i2c_sda_oe", metadata !"i2c_sda_oe", metadata !"", metadata !71, i32 49, metadata !72, i32 0, i32 1, i1* @i2c_sda_oe} ; [ DW_TAG_variable ]
!84 = metadata !{i32 786484, i32 0, null, metadata !"auto_inc_regad_in", metadata !"auto_inc_regad_in", metadata !"", metadata !71, i32 61, metadata !72, i32 0, i32 1, i1* @auto_inc_regad_in} ; [ DW_TAG_variable ]
!85 = metadata !{i32 786484, i32 0, null, metadata !"mem_din", metadata !"mem_din", metadata !"", metadata !71, i32 53, metadata !76, i32 0, i32 1, i8* @mem_din} ; [ DW_TAG_variable ]
!86 = metadata !{i32 786484, i32 0, null, metadata !"mem_wreq", metadata !"mem_wreq", metadata !"", metadata !71, i32 55, metadata !72, i32 0, i32 1, i1* @mem_wreq} ; [ DW_TAG_variable ]
!87 = metadata !{i32 786484, i32 0, null, metadata !"mem_wack", metadata !"mem_wack", metadata !"", metadata !71, i32 56, metadata !72, i32 0, i32 1, i1* @mem_wack} ; [ DW_TAG_variable ]
!88 = metadata !{i32 786484, i32 0, null, metadata !"mem_rack", metadata !"mem_rack", metadata !"", metadata !71, i32 58, metadata !72, i32 0, i32 1, i1* @mem_rack} ; [ DW_TAG_variable ]
!89 = metadata !{i32 786484, i32 0, null, metadata !"i2c_in", metadata !"i2c_in", metadata !"", metadata !71, i32 47, metadata !90, i32 0, i32 1, i2* @i2c_in} ; [ DW_TAG_variable ]
!90 = metadata !{i32 786485, null, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !80} ; [ DW_TAG_volatile_type ]
!91 = metadata !{i32 786484, i32 0, null, metadata !"dev_addr_in", metadata !"dev_addr_in", metadata !"", metadata !71, i32 60, metadata !92, i32 0, i32 1, i7* @dev_addr_in} ; [ DW_TAG_variable ]
!92 = metadata !{i32 786485, null, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !93} ; [ DW_TAG_volatile_type ]
!93 = metadata !{i32 786454, null, metadata !"uint7", metadata !71, i32 9, i64 0, i64 0, i64 0, i32 0, metadata !94} ; [ DW_TAG_typedef ]
!94 = metadata !{i32 786468, null, metadata !"uint7", null, i32 0, i64 7, i64 8, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!95 = metadata !{i32 786484, i32 0, null, metadata !"mem_dout", metadata !"mem_dout", metadata !"", metadata !71, i32 54, metadata !76, i32 0, i32 1, i8* @mem_dout} ; [ DW_TAG_variable ]
!96 = metadata !{i32 786689, metadata !97, metadata !"addr", metadata !71, i32 16777296, metadata !77, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!97 = metadata !{i32 786478, i32 0, metadata !71, metadata !"write_mem", metadata !"write_mem", metadata !"", metadata !71, i32 80, metadata !98, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (i8, i8)* @write_mem, null, null, metadata !100, i32 81} ; [ DW_TAG_subprogram ]
!98 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !99, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!99 = metadata !{null, metadata !77, metadata !77}
!100 = metadata !{metadata !101}
!101 = metadata !{i32 786468}                     ; [ DW_TAG_base_type ]
!102 = metadata !{i32 80, i32 22, metadata !97, null}
!103 = metadata !{i32 786689, metadata !97, metadata !"data", metadata !71, i32 33554512, metadata !77, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!104 = metadata !{i32 80, i32 34, metadata !97, null}
!105 = metadata !{i32 83, i32 2, metadata !106, null}
!106 = metadata !{i32 786443, metadata !97, i32 81, i32 1, metadata !71, i32 1} ; [ DW_TAG_lexical_block ]
!107 = metadata !{i32 84, i32 2, metadata !106, null}
!108 = metadata !{i32 85, i32 2, metadata !106, null}
!109 = metadata !{i32 86, i32 2, metadata !106, null}
!110 = metadata !{i32 87, i32 2, metadata !106, null}
!111 = metadata !{i32 89, i32 2, metadata !106, null}
!112 = metadata !{i32 90, i32 3, metadata !113, null}
!113 = metadata !{i32 786443, metadata !106, i32 89, i32 5, metadata !71, i32 2} ; [ DW_TAG_lexical_block ]
!114 = metadata !{i32 91, i32 3, metadata !113, null}
!115 = metadata !{i32 92, i32 3, metadata !113, null}
!116 = metadata !{i32 93, i32 2, metadata !113, null}
!117 = metadata !{i32 94, i32 2, metadata !106, null}
!118 = metadata !{i32 96, i32 2, metadata !106, null}
!119 = metadata !{i32 97, i32 2, metadata !106, null}
!120 = metadata !{i32 98, i32 2, metadata !106, null}
!121 = metadata !{i32 99, i32 2, metadata !106, null}
!122 = metadata !{i32 100, i32 1, metadata !106, null}
!123 = metadata !{i32 786689, metadata !124, metadata !"addr", metadata !71, i32 16777319, metadata !77, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!124 = metadata !{i32 786478, i32 0, metadata !71, metadata !"read_mem", metadata !"read_mem", metadata !"", metadata !71, i32 103, metadata !125, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i8 (i8)* @read_mem, null, null, metadata !100, i32 104} ; [ DW_TAG_subprogram ]
!125 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !126, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!126 = metadata !{metadata !77, metadata !77}
!127 = metadata !{i32 103, i32 22, metadata !124, null}
!128 = metadata !{i32 108, i32 2, metadata !129, null}
!129 = metadata !{i32 786443, metadata !124, i32 104, i32 1, metadata !71, i32 3} ; [ DW_TAG_lexical_block ]
!130 = metadata !{i32 109, i32 2, metadata !129, null}
!131 = metadata !{i32 110, i32 2, metadata !129, null}
!132 = metadata !{i32 111, i32 2, metadata !129, null}
!133 = metadata !{i32 113, i32 2, metadata !129, null}
!134 = metadata !{i32 114, i32 3, metadata !135, null}
!135 = metadata !{i32 786443, metadata !129, i32 113, i32 5, metadata !71, i32 4} ; [ DW_TAG_lexical_block ]
!136 = metadata !{i32 115, i32 3, metadata !135, null}
!137 = metadata !{i32 116, i32 3, metadata !135, null}
!138 = metadata !{i32 786688, metadata !129, metadata !"dt", metadata !71, i32 106, metadata !77, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!139 = metadata !{i32 117, i32 2, metadata !135, null}
!140 = metadata !{i32 118, i32 2, metadata !129, null}
!141 = metadata !{i32 120, i32 2, metadata !129, null}
!142 = metadata !{i32 121, i32 2, metadata !129, null}
!143 = metadata !{i32 122, i32 2, metadata !129, null}
!144 = metadata !{i32 124, i32 2, metadata !129, null}
!145 = metadata !{i32 133, i32 1, metadata !146, null}
!146 = metadata !{i32 786443, metadata !147, i32 132, i32 1, metadata !71, i32 5} ; [ DW_TAG_lexical_block ]
!147 = metadata !{i32 786478, i32 0, metadata !71, metadata !"i2c_slave_core", metadata !"i2c_slave_core", metadata !"", metadata !71, i32 131, metadata !148, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @i2c_slave_core, null, null, metadata !100, i32 132} ; [ DW_TAG_subprogram ]
!148 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !149, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!149 = metadata !{null}
!150 = metadata !{i32 134, i32 1, metadata !146, null}
!151 = metadata !{i32 135, i32 1, metadata !146, null}
!152 = metadata !{i32 136, i32 1, metadata !146, null}
!153 = metadata !{i32 138, i32 1, metadata !146, null}
!154 = metadata !{i32 139, i32 1, metadata !146, null}
!155 = metadata !{i32 141, i32 1, metadata !146, null}
!156 = metadata !{i32 142, i32 1, metadata !146, null}
!157 = metadata !{i32 143, i32 1, metadata !146, null}
!158 = metadata !{i32 144, i32 1, metadata !146, null}
!159 = metadata !{i32 145, i32 1, metadata !146, null}
!160 = metadata !{i32 146, i32 1, metadata !146, null}
!161 = metadata !{i32 147, i32 1, metadata !146, null}
!162 = metadata !{i32 158, i32 2, metadata !146, null}
!163 = metadata !{i32 159, i32 2, metadata !146, null}
!164 = metadata !{i32 160, i32 2, metadata !146, null}
!165 = metadata !{i32 161, i32 2, metadata !146, null}
!166 = metadata !{i32 162, i32 2, metadata !146, null}
!167 = metadata !{i32 163, i32 2, metadata !146, null}
!168 = metadata !{i32 164, i32 2, metadata !146, null}
!169 = metadata !{i32 165, i32 2, metadata !146, null}
!170 = metadata !{i32 168, i32 13, metadata !171, null}
!171 = metadata !{i32 786443, metadata !146, i32 168, i32 12, metadata !71, i32 6} ; [ DW_TAG_lexical_block ]
!172 = metadata !{i32 76, i32 2, metadata !173, metadata !175}
!173 = metadata !{i32 786443, metadata !174, i32 74, i32 1, metadata !71, i32 0} ; [ DW_TAG_lexical_block ]
!174 = metadata !{i32 786478, i32 0, metadata !71, metadata !"read_i2c", metadata !"read_i2c", metadata !"", metadata !71, i32 73, metadata !148, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, metadata !100, i32 74} ; [ DW_TAG_subprogram ]
!175 = metadata !{i32 438, i32 6, metadata !176, null}
!176 = metadata !{i32 786443, metadata !177, i32 437, i32 8, metadata !71, i32 89} ; [ DW_TAG_lexical_block ]
!177 = metadata !{i32 786443, metadata !178, i32 428, i32 46, metadata !71, i32 86} ; [ DW_TAG_lexical_block ]
!178 = metadata !{i32 786443, metadata !179, i32 428, i32 4, metadata !71, i32 85} ; [ DW_TAG_lexical_block ]
!179 = metadata !{i32 786443, metadata !171, i32 421, i32 13, metadata !71, i32 83} ; [ DW_TAG_lexical_block ]
!180 = metadata !{i32 76, i32 2, metadata !173, metadata !181}
!181 = metadata !{i32 430, i32 6, metadata !182, null}
!182 = metadata !{i32 786443, metadata !177, i32 429, i32 8, metadata !71, i32 87} ; [ DW_TAG_lexical_block ]
!183 = metadata !{i32 76, i32 2, metadata !173, metadata !184}
!184 = metadata !{i32 325, i32 6, metadata !185, null}
!185 = metadata !{i32 786443, metadata !186, i32 323, i32 8, metadata !71, i32 58} ; [ DW_TAG_lexical_block ]
!186 = metadata !{i32 786443, metadata !187, i32 314, i32 24, metadata !71, i32 54} ; [ DW_TAG_lexical_block ]
!187 = metadata !{i32 786443, metadata !171, i32 313, i32 6, metadata !71, i32 53} ; [ DW_TAG_lexical_block ]
!188 = metadata !{i32 76, i32 2, metadata !173, metadata !189}
!189 = metadata !{i32 293, i32 4, metadata !190, null}
!190 = metadata !{i32 786443, metadata !171, i32 291, i32 6, metadata !71, i32 46} ; [ DW_TAG_lexical_block ]
!191 = metadata !{i32 76, i32 2, metadata !173, metadata !192}
!192 = metadata !{i32 191, i32 4, metadata !193, null}
!193 = metadata !{i32 786443, metadata !171, i32 190, i32 6, metadata !71, i32 13} ; [ DW_TAG_lexical_block ]
!194 = metadata !{i32 76, i32 2, metadata !173, metadata !195}
!195 = metadata !{i32 184, i32 4, metadata !196, null}
!196 = metadata !{i32 786443, metadata !171, i32 183, i32 6, metadata !71, i32 10} ; [ DW_TAG_lexical_block ]
!197 = metadata !{i32 173, i32 3, metadata !171, null}
!198 = metadata !{i32 174, i32 3, metadata !171, null}
!199 = metadata !{i32 175, i32 3, metadata !171, null}
!200 = metadata !{i32 178, i32 3, metadata !171, null}
!201 = metadata !{i32 76, i32 2, metadata !173, metadata !202}
!202 = metadata !{i32 179, i32 4, metadata !203, null}
!203 = metadata !{i32 786443, metadata !171, i32 178, i32 6, metadata !71, i32 7} ; [ DW_TAG_lexical_block ]
!204 = metadata !{i32 786688, metadata !205, metadata !"__Val2__", metadata !71, i32 180, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!205 = metadata !{i32 786443, metadata !171, i32 180, i32 13, metadata !71, i32 8} ; [ DW_TAG_lexical_block ]
!206 = metadata !{i32 180, i32 51, metadata !205, null}
!207 = metadata !{i32 180, i32 85, metadata !205, null}
!208 = metadata !{i32 180, i32 174, metadata !205, null}
!209 = metadata !{i32 786688, metadata !210, metadata !"__Val2__", metadata !71, i32 180, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!210 = metadata !{i32 786443, metadata !171, i32 180, i32 186, metadata !71, i32 9} ; [ DW_TAG_lexical_block ]
!211 = metadata !{i32 180, i32 224, metadata !210, null}
!212 = metadata !{i32 180, i32 0, metadata !210, null}
!213 = metadata !{i32 786688, metadata !214, metadata !"__Val2__", metadata !71, i32 185, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!214 = metadata !{i32 786443, metadata !196, i32 185, i32 9, metadata !71, i32 11} ; [ DW_TAG_lexical_block ]
!215 = metadata !{i32 185, i32 47, metadata !214, null}
!216 = metadata !{i32 185, i32 81, metadata !214, null}
!217 = metadata !{i32 185, i32 170, metadata !214, null}
!218 = metadata !{i32 786688, metadata !219, metadata !"__Val2__", metadata !71, i32 187, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!219 = metadata !{i32 786443, metadata !171, i32 187, i32 13, metadata !71, i32 12} ; [ DW_TAG_lexical_block ]
!220 = metadata !{i32 187, i32 51, metadata !219, null}
!221 = metadata !{i32 187, i32 85, metadata !219, null}
!222 = metadata !{i32 187, i32 174, metadata !219, null}
!223 = metadata !{i32 786688, metadata !224, metadata !"__Val2__", metadata !71, i32 192, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!224 = metadata !{i32 786443, metadata !193, i32 192, i32 9, metadata !71, i32 14} ; [ DW_TAG_lexical_block ]
!225 = metadata !{i32 192, i32 47, metadata !224, null}
!226 = metadata !{i32 192, i32 81, metadata !224, null}
!227 = metadata !{i32 192, i32 170, metadata !224, null}
!228 = metadata !{i32 786688, metadata !229, metadata !"__Val2__", metadata !71, i32 194, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!229 = metadata !{i32 786443, metadata !171, i32 194, i32 13, metadata !71, i32 15} ; [ DW_TAG_lexical_block ]
!230 = metadata !{i32 194, i32 51, metadata !229, null}
!231 = metadata !{i32 194, i32 85, metadata !229, null}
!232 = metadata !{i32 194, i32 174, metadata !229, null}
!233 = metadata !{i32 199, i32 8, metadata !234, null}
!234 = metadata !{i32 786443, metadata !171, i32 199, i32 3, metadata !71, i32 16} ; [ DW_TAG_lexical_block ]
!235 = metadata !{i32 76, i32 2, metadata !173, metadata !236}
!236 = metadata !{i32 202, i32 5, metadata !237, null}
!237 = metadata !{i32 786443, metadata !238, i32 201, i32 7, metadata !71, i32 18} ; [ DW_TAG_lexical_block ]
!238 = metadata !{i32 786443, metadata !234, i32 199, i32 45, metadata !71, i32 17} ; [ DW_TAG_lexical_block ]
!239 = metadata !{i32 786688, metadata !240, metadata !"__Val2__", metadata !71, i32 203, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!240 = metadata !{i32 786443, metadata !238, i32 203, i32 14, metadata !71, i32 19} ; [ DW_TAG_lexical_block ]
!241 = metadata !{i32 203, i32 52, metadata !240, null}
!242 = metadata !{i32 203, i32 86, metadata !240, null}
!243 = metadata !{i32 203, i32 175, metadata !240, null}
!244 = metadata !{i32 786688, metadata !245, metadata !"__Val2__", metadata !71, i32 205, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!245 = metadata !{i32 786443, metadata !238, i32 205, i32 34, metadata !71, i32 20} ; [ DW_TAG_lexical_block ]
!246 = metadata !{i32 205, i32 72, metadata !245, null}
!247 = metadata !{i32 205, i32 106, metadata !245, null}
!248 = metadata !{i32 205, i32 195, metadata !245, null}
!249 = metadata !{i32 786688, metadata !146, metadata !"dev_addr", metadata !71, i32 150, metadata !93, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!250 = metadata !{i32 208, i32 4, metadata !238, null}
!251 = metadata !{i32 76, i32 2, metadata !173, metadata !252}
!252 = metadata !{i32 209, i32 5, metadata !253, null}
!253 = metadata !{i32 786443, metadata !238, i32 208, i32 7, metadata !71, i32 21} ; [ DW_TAG_lexical_block ]
!254 = metadata !{i32 786688, metadata !255, metadata !"__Val2__", metadata !71, i32 210, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!255 = metadata !{i32 786443, metadata !238, i32 210, i32 14, metadata !71, i32 22} ; [ DW_TAG_lexical_block ]
!256 = metadata !{i32 210, i32 52, metadata !255, null}
!257 = metadata !{i32 210, i32 86, metadata !255, null}
!258 = metadata !{i32 210, i32 175, metadata !255, null}
!259 = metadata !{i32 199, i32 34, metadata !234, null}
!260 = metadata !{i32 786688, metadata !146, metadata !"bit_cnt", metadata !71, i32 153, metadata !261, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!261 = metadata !{i32 786454, null, metadata !"uint4", metadata !71, i32 6, i64 0, i64 0, i64 0, i32 0, metadata !262} ; [ DW_TAG_typedef ]
!262 = metadata !{i32 786468, null, metadata !"uint4", null, i32 0, i64 4, i64 4, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!263 = metadata !{i32 214, i32 3, metadata !171, null}
!264 = metadata !{i32 220, i32 3, metadata !171, null}
!265 = metadata !{i32 76, i32 2, metadata !173, metadata !266}
!266 = metadata !{i32 221, i32 4, metadata !267, null}
!267 = metadata !{i32 786443, metadata !171, i32 220, i32 6, metadata !71, i32 23} ; [ DW_TAG_lexical_block ]
!268 = metadata !{i32 786688, metadata !269, metadata !"__Val2__", metadata !71, i32 222, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!269 = metadata !{i32 786443, metadata !171, i32 222, i32 13, metadata !71, i32 24} ; [ DW_TAG_lexical_block ]
!270 = metadata !{i32 222, i32 51, metadata !269, null}
!271 = metadata !{i32 222, i32 85, metadata !269, null}
!272 = metadata !{i32 222, i32 174, metadata !269, null}
!273 = metadata !{i32 786688, metadata !274, metadata !"__Val2__", metadata !71, i32 224, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!274 = metadata !{i32 786443, metadata !171, i32 224, i32 8, metadata !71, i32 25} ; [ DW_TAG_lexical_block ]
!275 = metadata !{i32 224, i32 46, metadata !274, null}
!276 = metadata !{i32 224, i32 80, metadata !274, null}
!277 = metadata !{i32 224, i32 169, metadata !274, null}
!278 = metadata !{i32 227, i32 3, metadata !171, null}
!279 = metadata !{i32 76, i32 2, metadata !173, metadata !280}
!280 = metadata !{i32 228, i32 4, metadata !281, null}
!281 = metadata !{i32 786443, metadata !171, i32 227, i32 6, metadata !71, i32 26} ; [ DW_TAG_lexical_block ]
!282 = metadata !{i32 786688, metadata !283, metadata !"__Val2__", metadata !71, i32 229, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!283 = metadata !{i32 786443, metadata !171, i32 229, i32 13, metadata !71, i32 27} ; [ DW_TAG_lexical_block ]
!284 = metadata !{i32 229, i32 51, metadata !283, null}
!285 = metadata !{i32 229, i32 85, metadata !283, null}
!286 = metadata !{i32 229, i32 174, metadata !283, null}
!287 = metadata !{i32 232, i32 3, metadata !171, null}
!288 = metadata !{i32 233, i32 3, metadata !171, null}
!289 = metadata !{i32 234, i32 3, metadata !171, null}
!290 = metadata !{i32 235, i32 3, metadata !171, null}
!291 = metadata !{i32 237, i32 3, metadata !171, null}
!292 = metadata !{i32 76, i32 2, metadata !173, metadata !293}
!293 = metadata !{i32 238, i32 4, metadata !294, null}
!294 = metadata !{i32 786443, metadata !171, i32 237, i32 6, metadata !71, i32 28} ; [ DW_TAG_lexical_block ]
!295 = metadata !{i32 786688, metadata !296, metadata !"__Val2__", metadata !71, i32 239, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!296 = metadata !{i32 786443, metadata !171, i32 239, i32 13, metadata !71, i32 29} ; [ DW_TAG_lexical_block ]
!297 = metadata !{i32 239, i32 51, metadata !296, null}
!298 = metadata !{i32 239, i32 85, metadata !296, null}
!299 = metadata !{i32 239, i32 174, metadata !296, null}
!300 = metadata !{i32 76, i32 2, metadata !173, metadata !301}
!301 = metadata !{i32 242, i32 4, metadata !302, null}
!302 = metadata !{i32 786443, metadata !171, i32 241, i32 6, metadata !71, i32 30} ; [ DW_TAG_lexical_block ]
!303 = metadata !{i32 786688, metadata !304, metadata !"__Val2__", metadata !71, i32 243, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!304 = metadata !{i32 786443, metadata !171, i32 243, i32 13, metadata !71, i32 31} ; [ DW_TAG_lexical_block ]
!305 = metadata !{i32 243, i32 51, metadata !304, null}
!306 = metadata !{i32 243, i32 85, metadata !304, null}
!307 = metadata !{i32 243, i32 174, metadata !304, null}
!308 = metadata !{i32 245, i32 3, metadata !171, null}
!309 = metadata !{i32 250, i32 8, metadata !310, null}
!310 = metadata !{i32 786443, metadata !171, i32 250, i32 3, metadata !71, i32 32} ; [ DW_TAG_lexical_block ]
!311 = metadata !{i32 76, i32 2, metadata !173, metadata !312}
!312 = metadata !{i32 253, i32 5, metadata !313, null}
!313 = metadata !{i32 786443, metadata !314, i32 252, i32 7, metadata !71, i32 34} ; [ DW_TAG_lexical_block ]
!314 = metadata !{i32 786443, metadata !310, i32 250, i32 45, metadata !71, i32 33} ; [ DW_TAG_lexical_block ]
!315 = metadata !{i32 786688, metadata !316, metadata !"__Val2__", metadata !71, i32 254, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!316 = metadata !{i32 786443, metadata !314, i32 254, i32 14, metadata !71, i32 35} ; [ DW_TAG_lexical_block ]
!317 = metadata !{i32 254, i32 52, metadata !316, null}
!318 = metadata !{i32 254, i32 86, metadata !316, null}
!319 = metadata !{i32 254, i32 175, metadata !316, null}
!320 = metadata !{i32 786688, metadata !321, metadata !"__Val2__", metadata !71, i32 256, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!321 = metadata !{i32 786443, metadata !314, i32 256, i32 34, metadata !71, i32 36} ; [ DW_TAG_lexical_block ]
!322 = metadata !{i32 256, i32 72, metadata !321, null}
!323 = metadata !{i32 256, i32 106, metadata !321, null}
!324 = metadata !{i32 256, i32 195, metadata !321, null}
!325 = metadata !{i32 786688, metadata !146, metadata !"reg_addr", metadata !71, i32 151, metadata !77, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!326 = metadata !{i32 259, i32 4, metadata !314, null}
!327 = metadata !{i32 76, i32 2, metadata !173, metadata !328}
!328 = metadata !{i32 260, i32 5, metadata !329, null}
!329 = metadata !{i32 786443, metadata !314, i32 259, i32 7, metadata !71, i32 37} ; [ DW_TAG_lexical_block ]
!330 = metadata !{i32 786688, metadata !331, metadata !"__Val2__", metadata !71, i32 261, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!331 = metadata !{i32 786443, metadata !314, i32 261, i32 14, metadata !71, i32 38} ; [ DW_TAG_lexical_block ]
!332 = metadata !{i32 261, i32 52, metadata !331, null}
!333 = metadata !{i32 261, i32 86, metadata !331, null}
!334 = metadata !{i32 261, i32 175, metadata !331, null}
!335 = metadata !{i32 250, i32 34, metadata !310, null}
!336 = metadata !{i32 265, i32 3, metadata !171, null}
!337 = metadata !{i32 266, i32 3, metadata !171, null}
!338 = metadata !{i32 267, i32 3, metadata !171, null}
!339 = metadata !{i32 268, i32 3, metadata !171, null}
!340 = metadata !{i32 270, i32 3, metadata !171, null}
!341 = metadata !{i32 76, i32 2, metadata !173, metadata !342}
!342 = metadata !{i32 271, i32 4, metadata !343, null}
!343 = metadata !{i32 786443, metadata !171, i32 270, i32 6, metadata !71, i32 39} ; [ DW_TAG_lexical_block ]
!344 = metadata !{i32 786688, metadata !345, metadata !"__Val2__", metadata !71, i32 272, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!345 = metadata !{i32 786443, metadata !171, i32 272, i32 13, metadata !71, i32 40} ; [ DW_TAG_lexical_block ]
!346 = metadata !{i32 272, i32 51, metadata !345, null}
!347 = metadata !{i32 272, i32 85, metadata !345, null}
!348 = metadata !{i32 272, i32 174, metadata !345, null}
!349 = metadata !{i32 76, i32 2, metadata !173, metadata !350}
!350 = metadata !{i32 275, i32 4, metadata !351, null}
!351 = metadata !{i32 786443, metadata !171, i32 274, i32 6, metadata !71, i32 41} ; [ DW_TAG_lexical_block ]
!352 = metadata !{i32 786688, metadata !353, metadata !"__Val2__", metadata !71, i32 276, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!353 = metadata !{i32 786443, metadata !171, i32 276, i32 13, metadata !71, i32 42} ; [ DW_TAG_lexical_block ]
!354 = metadata !{i32 276, i32 51, metadata !353, null}
!355 = metadata !{i32 276, i32 85, metadata !353, null}
!356 = metadata !{i32 276, i32 174, metadata !353, null}
!357 = metadata !{i32 278, i32 3, metadata !171, null}
!358 = metadata !{i32 285, i32 3, metadata !171, null}
!359 = metadata !{i32 76, i32 2, metadata !173, metadata !360}
!360 = metadata !{i32 286, i32 4, metadata !361, null}
!361 = metadata !{i32 786443, metadata !171, i32 285, i32 6, metadata !71, i32 43} ; [ DW_TAG_lexical_block ]
!362 = metadata !{i32 786688, metadata !363, metadata !"__Val2__", metadata !71, i32 287, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!363 = metadata !{i32 786443, metadata !171, i32 287, i32 13, metadata !71, i32 44} ; [ DW_TAG_lexical_block ]
!364 = metadata !{i32 287, i32 51, metadata !363, null}
!365 = metadata !{i32 287, i32 85, metadata !363, null}
!366 = metadata !{i32 287, i32 174, metadata !363, null}
!367 = metadata !{i32 786688, metadata !368, metadata !"__Val2__", metadata !71, i32 289, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!368 = metadata !{i32 786443, metadata !171, i32 289, i32 33, metadata !71, i32 45} ; [ DW_TAG_lexical_block ]
!369 = metadata !{i32 289, i32 71, metadata !368, null}
!370 = metadata !{i32 289, i32 105, metadata !368, null}
!371 = metadata !{i32 289, i32 194, metadata !368, null}
!372 = metadata !{i32 786688, metadata !146, metadata !"reg_data", metadata !71, i32 152, metadata !77, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!373 = metadata !{i32 291, i32 3, metadata !171, null}
!374 = metadata !{i32 292, i32 61, metadata !375, null}
!375 = metadata !{i32 786443, metadata !190, i32 292, i32 23, metadata !71, i32 47} ; [ DW_TAG_lexical_block ]
!376 = metadata !{i32 786688, metadata !375, metadata !"__Val2__", metadata !71, i32 292, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!377 = metadata !{i32 292, i32 95, metadata !375, null}
!378 = metadata !{i32 786688, metadata !146, metadata !"pre_i2c_sda_val", metadata !71, i32 156, metadata !73, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!379 = metadata !{i32 292, i32 184, metadata !375, null}
!380 = metadata !{i32 786688, metadata !381, metadata !"__Val2__", metadata !71, i32 295, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!381 = metadata !{i32 786443, metadata !190, i32 295, i32 9, metadata !71, i32 48} ; [ DW_TAG_lexical_block ]
!382 = metadata !{i32 295, i32 47, metadata !381, null}
!383 = metadata !{i32 295, i32 81, metadata !381, null}
!384 = metadata !{i32 295, i32 170, metadata !381, null}
!385 = metadata !{i32 314, i32 4, metadata !187, null}
!386 = metadata !{i32 297, i32 4, metadata !190, null}
!387 = metadata !{i32 786688, metadata !388, metadata !"__Val2__", metadata !71, i32 297, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!388 = metadata !{i32 786443, metadata !190, i32 297, i32 24, metadata !71, i32 49} ; [ DW_TAG_lexical_block ]
!389 = metadata !{i32 297, i32 62, metadata !388, null}
!390 = metadata !{i32 297, i32 96, metadata !388, null}
!391 = metadata !{i32 297, i32 185, metadata !388, null}
!392 = metadata !{i32 76, i32 2, metadata !173, metadata !393}
!393 = metadata !{i32 299, i32 6, metadata !394, null}
!394 = metadata !{i32 786443, metadata !395, i32 298, i32 8, metadata !71, i32 51} ; [ DW_TAG_lexical_block ]
!395 = metadata !{i32 786443, metadata !190, i32 297, i32 218, metadata !71, i32 50} ; [ DW_TAG_lexical_block ]
!396 = metadata !{i32 786688, metadata !397, metadata !"__Val2__", metadata !71, i32 300, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!397 = metadata !{i32 786443, metadata !395, i32 300, i32 15, metadata !71, i32 52} ; [ DW_TAG_lexical_block ]
!398 = metadata !{i32 300, i32 53, metadata !397, null}
!399 = metadata !{i32 300, i32 87, metadata !397, null}
!400 = metadata !{i32 300, i32 176, metadata !397, null}
!401 = metadata !{i32 367, i32 8, metadata !402, null}
!402 = metadata !{i32 786443, metadata !171, i32 367, i32 3, metadata !71, i32 67} ; [ DW_TAG_lexical_block ]
!403 = metadata !{i32 76, i32 2, metadata !173, metadata !404}
!404 = metadata !{i32 317, i32 6, metadata !405, null}
!405 = metadata !{i32 786443, metadata !186, i32 316, i32 8, metadata !71, i32 55} ; [ DW_TAG_lexical_block ]
!406 = metadata !{i32 786688, metadata !407, metadata !"__Val2__", metadata !71, i32 318, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!407 = metadata !{i32 786443, metadata !186, i32 318, i32 15, metadata !71, i32 56} ; [ DW_TAG_lexical_block ]
!408 = metadata !{i32 318, i32 53, metadata !407, null}
!409 = metadata !{i32 318, i32 87, metadata !407, null}
!410 = metadata !{i32 318, i32 176, metadata !407, null}
!411 = metadata !{i32 786688, metadata !412, metadata !"__Val2__", metadata !71, i32 320, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!412 = metadata !{i32 786443, metadata !186, i32 320, i32 35, metadata !71, i32 57} ; [ DW_TAG_lexical_block ]
!413 = metadata !{i32 320, i32 73, metadata !412, null}
!414 = metadata !{i32 320, i32 107, metadata !412, null}
!415 = metadata !{i32 320, i32 196, metadata !412, null}
!416 = metadata !{i32 323, i32 5, metadata !186, null}
!417 = metadata !{i32 324, i32 63, metadata !418, null}
!418 = metadata !{i32 786443, metadata !185, i32 324, i32 25, metadata !71, i32 59} ; [ DW_TAG_lexical_block ]
!419 = metadata !{i32 786688, metadata !418, metadata !"__Val2__", metadata !71, i32 324, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!420 = metadata !{i32 324, i32 97, metadata !418, null}
!421 = metadata !{i32 786688, metadata !422, metadata !"__Val2__", metadata !71, i32 326, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!422 = metadata !{i32 786443, metadata !185, i32 326, i32 11, metadata !71, i32 60} ; [ DW_TAG_lexical_block ]
!423 = metadata !{i32 326, i32 49, metadata !422, null}
!424 = metadata !{i32 326, i32 83, metadata !422, null}
!425 = metadata !{i32 326, i32 172, metadata !422, null}
!426 = metadata !{i32 786688, metadata !427, metadata !"__Val2__", metadata !71, i32 328, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!427 = metadata !{i32 786443, metadata !186, i32 328, i32 15, metadata !71, i32 61} ; [ DW_TAG_lexical_block ]
!428 = metadata !{i32 328, i32 53, metadata !427, null}
!429 = metadata !{i32 328, i32 87, metadata !427, null}
!430 = metadata !{i32 328, i32 176, metadata !427, null}
!431 = metadata !{i32 330, i32 5, metadata !186, null}
!432 = metadata !{i32 331, i32 4, metadata !186, null}
!433 = metadata !{i32 334, i32 4, metadata !187, null}
!434 = metadata !{i32 335, i32 4, metadata !187, null}
!435 = metadata !{i32 336, i32 4, metadata !187, null}
!436 = metadata !{i32 337, i32 4, metadata !187, null}
!437 = metadata !{i32 339, i32 4, metadata !187, null}
!438 = metadata !{i32 76, i32 2, metadata !173, metadata !439}
!439 = metadata !{i32 340, i32 5, metadata !440, null}
!440 = metadata !{i32 786443, metadata !187, i32 339, i32 7, metadata !71, i32 62} ; [ DW_TAG_lexical_block ]
!441 = metadata !{i32 786688, metadata !442, metadata !"__Val2__", metadata !71, i32 341, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!442 = metadata !{i32 786443, metadata !187, i32 341, i32 14, metadata !71, i32 63} ; [ DW_TAG_lexical_block ]
!443 = metadata !{i32 341, i32 52, metadata !442, null}
!444 = metadata !{i32 341, i32 86, metadata !442, null}
!445 = metadata !{i32 341, i32 175, metadata !442, null}
!446 = metadata !{i32 76, i32 2, metadata !173, metadata !447}
!447 = metadata !{i32 344, i32 5, metadata !448, null}
!448 = metadata !{i32 786443, metadata !187, i32 343, i32 7, metadata !71, i32 64} ; [ DW_TAG_lexical_block ]
!449 = metadata !{i32 786688, metadata !450, metadata !"__Val2__", metadata !71, i32 345, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!450 = metadata !{i32 786443, metadata !187, i32 345, i32 14, metadata !71, i32 65} ; [ DW_TAG_lexical_block ]
!451 = metadata !{i32 345, i32 52, metadata !450, null}
!452 = metadata !{i32 345, i32 86, metadata !450, null}
!453 = metadata !{i32 345, i32 175, metadata !450, null}
!454 = metadata !{i32 347, i32 4, metadata !187, null}
!455 = metadata !{i32 350, i32 4, metadata !187, null}
!456 = metadata !{i32 351, i32 5, metadata !457, null}
!457 = metadata !{i32 786443, metadata !187, i32 350, i32 21, metadata !71, i32 66} ; [ DW_TAG_lexical_block ]
!458 = metadata !{i32 352, i32 5, metadata !457, null}
!459 = metadata !{i32 353, i32 5, metadata !457, null}
!460 = metadata !{i32 354, i32 5, metadata !457, null}
!461 = metadata !{i32 355, i32 6, metadata !457, null}
!462 = metadata !{i32 356, i32 4, metadata !457, null}
!463 = metadata !{i32 367, i32 46, metadata !464, null}
!464 = metadata !{i32 786443, metadata !402, i32 367, i32 45, metadata !71, i32 68} ; [ DW_TAG_lexical_block ]
!465 = metadata !{i32 369, i32 4, metadata !464, null}
!466 = metadata !{i32 76, i32 2, metadata !173, metadata !467}
!467 = metadata !{i32 370, i32 5, metadata !468, null}
!468 = metadata !{i32 786443, metadata !464, i32 369, i32 7, metadata !71, i32 69} ; [ DW_TAG_lexical_block ]
!469 = metadata !{i32 786688, metadata !470, metadata !"__Val2__", metadata !71, i32 371, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!470 = metadata !{i32 786443, metadata !464, i32 371, i32 14, metadata !71, i32 70} ; [ DW_TAG_lexical_block ]
!471 = metadata !{i32 371, i32 52, metadata !470, null}
!472 = metadata !{i32 371, i32 86, metadata !470, null}
!473 = metadata !{i32 371, i32 175, metadata !470, null}
!474 = metadata !{i32 786688, metadata !475, metadata !"__Val2__", metadata !71, i32 373, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!475 = metadata !{i32 786443, metadata !464, i32 373, i32 34, metadata !71, i32 71} ; [ DW_TAG_lexical_block ]
!476 = metadata !{i32 373, i32 72, metadata !475, null}
!477 = metadata !{i32 373, i32 106, metadata !475, null}
!478 = metadata !{i32 373, i32 195, metadata !475, null}
!479 = metadata !{i32 376, i32 4, metadata !464, null}
!480 = metadata !{i32 76, i32 2, metadata !173, metadata !481}
!481 = metadata !{i32 377, i32 5, metadata !482, null}
!482 = metadata !{i32 786443, metadata !464, i32 376, i32 7, metadata !71, i32 72} ; [ DW_TAG_lexical_block ]
!483 = metadata !{i32 786688, metadata !484, metadata !"__Val2__", metadata !71, i32 378, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!484 = metadata !{i32 786443, metadata !464, i32 378, i32 14, metadata !71, i32 73} ; [ DW_TAG_lexical_block ]
!485 = metadata !{i32 378, i32 52, metadata !484, null}
!486 = metadata !{i32 378, i32 86, metadata !484, null}
!487 = metadata !{i32 378, i32 175, metadata !484, null}
!488 = metadata !{i32 367, i32 34, metadata !402, null}
!489 = metadata !{i32 382, i32 3, metadata !171, null}
!490 = metadata !{i32 388, i32 3, metadata !171, null}
!491 = metadata !{i32 76, i32 2, metadata !173, metadata !492}
!492 = metadata !{i32 389, i32 4, metadata !493, null}
!493 = metadata !{i32 786443, metadata !171, i32 388, i32 6, metadata !71, i32 74} ; [ DW_TAG_lexical_block ]
!494 = metadata !{i32 786688, metadata !495, metadata !"__Val2__", metadata !71, i32 390, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!495 = metadata !{i32 786443, metadata !171, i32 390, i32 13, metadata !71, i32 75} ; [ DW_TAG_lexical_block ]
!496 = metadata !{i32 390, i32 51, metadata !495, null}
!497 = metadata !{i32 390, i32 85, metadata !495, null}
!498 = metadata !{i32 390, i32 174, metadata !495, null}
!499 = metadata !{i32 786688, metadata !500, metadata !"__Val2__", metadata !71, i32 392, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!500 = metadata !{i32 786443, metadata !171, i32 392, i32 8, metadata !71, i32 76} ; [ DW_TAG_lexical_block ]
!501 = metadata !{i32 392, i32 46, metadata !500, null}
!502 = metadata !{i32 392, i32 80, metadata !500, null}
!503 = metadata !{i32 392, i32 169, metadata !500, null}
!504 = metadata !{i32 395, i32 3, metadata !171, null}
!505 = metadata !{i32 76, i32 2, metadata !173, metadata !506}
!506 = metadata !{i32 396, i32 4, metadata !507, null}
!507 = metadata !{i32 786443, metadata !171, i32 395, i32 6, metadata !71, i32 77} ; [ DW_TAG_lexical_block ]
!508 = metadata !{i32 786688, metadata !509, metadata !"__Val2__", metadata !71, i32 397, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!509 = metadata !{i32 786443, metadata !171, i32 397, i32 13, metadata !71, i32 78} ; [ DW_TAG_lexical_block ]
!510 = metadata !{i32 397, i32 51, metadata !509, null}
!511 = metadata !{i32 397, i32 85, metadata !509, null}
!512 = metadata !{i32 397, i32 174, metadata !509, null}
!513 = metadata !{i32 400, i32 3, metadata !171, null}
!514 = metadata !{i32 401, i32 3, metadata !171, null}
!515 = metadata !{i32 402, i32 3, metadata !171, null}
!516 = metadata !{i32 403, i32 3, metadata !171, null}
!517 = metadata !{i32 404, i32 14, metadata !171, null}
!518 = metadata !{i32 405, i32 3, metadata !171, null}
!519 = metadata !{i32 406, i32 3, metadata !171, null}
!520 = metadata !{i32 407, i32 4, metadata !171, null}
!521 = metadata !{i32 410, i32 3, metadata !171, null}
!522 = metadata !{i32 76, i32 2, metadata !173, metadata !523}
!523 = metadata !{i32 411, i32 4, metadata !524, null}
!524 = metadata !{i32 786443, metadata !171, i32 410, i32 6, metadata !71, i32 79} ; [ DW_TAG_lexical_block ]
!525 = metadata !{i32 786688, metadata !526, metadata !"__Val2__", metadata !71, i32 412, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!526 = metadata !{i32 786443, metadata !171, i32 412, i32 13, metadata !71, i32 80} ; [ DW_TAG_lexical_block ]
!527 = metadata !{i32 412, i32 51, metadata !526, null}
!528 = metadata !{i32 412, i32 85, metadata !526, null}
!529 = metadata !{i32 412, i32 174, metadata !526, null}
!530 = metadata !{i32 76, i32 2, metadata !173, metadata !531}
!531 = metadata !{i32 415, i32 4, metadata !532, null}
!532 = metadata !{i32 786443, metadata !171, i32 414, i32 6, metadata !71, i32 81} ; [ DW_TAG_lexical_block ]
!533 = metadata !{i32 786688, metadata !534, metadata !"__Val2__", metadata !71, i32 416, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!534 = metadata !{i32 786443, metadata !171, i32 416, i32 13, metadata !71, i32 82} ; [ DW_TAG_lexical_block ]
!535 = metadata !{i32 416, i32 51, metadata !534, null}
!536 = metadata !{i32 416, i32 85, metadata !534, null}
!537 = metadata !{i32 416, i32 174, metadata !534, null}
!538 = metadata !{i32 421, i32 14, metadata !179, null}
!539 = metadata !{i32 422, i32 4, metadata !179, null}
!540 = metadata !{i32 786688, metadata !541, metadata !"__Val2__", metadata !71, i32 423, metadata !77, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!541 = metadata !{i32 786443, metadata !179, i32 423, i32 19, metadata !71, i32 84} ; [ DW_TAG_lexical_block ]
!542 = metadata !{i32 423, i32 59, metadata !541, null}
!543 = metadata !{i32 423, i32 94, metadata !541, null}
!544 = metadata !{i32 423, i32 183, metadata !541, null}
!545 = metadata !{i32 424, i32 4, metadata !179, null}
!546 = metadata !{i32 425, i32 4, metadata !179, null}
!547 = metadata !{i32 431, i32 6, metadata !182, null}
!548 = metadata !{i32 428, i32 9, metadata !178, null}
!549 = metadata !{i32 786688, metadata !550, metadata !"__Val2__", metadata !71, i32 433, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!550 = metadata !{i32 786443, metadata !177, i32 433, i32 15, metadata !71, i32 88} ; [ DW_TAG_lexical_block ]
!551 = metadata !{i32 433, i32 53, metadata !550, null}
!552 = metadata !{i32 433, i32 87, metadata !550, null}
!553 = metadata !{i32 433, i32 176, metadata !550, null}
!554 = metadata !{i32 435, i32 5, metadata !177, null}
!555 = metadata !{i32 437, i32 5, metadata !177, null}
!556 = metadata !{i32 786688, metadata !557, metadata !"__Val2__", metadata !71, i32 439, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!557 = metadata !{i32 786443, metadata !176, i32 439, i32 11, metadata !71, i32 90} ; [ DW_TAG_lexical_block ]
!558 = metadata !{i32 439, i32 49, metadata !557, null}
!559 = metadata !{i32 439, i32 83, metadata !557, null}
!560 = metadata !{i32 439, i32 172, metadata !557, null}
!561 = metadata !{i32 786688, metadata !562, metadata !"__Val2__", metadata !71, i32 441, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!562 = metadata !{i32 786443, metadata !177, i32 441, i32 15, metadata !71, i32 91} ; [ DW_TAG_lexical_block ]
!563 = metadata !{i32 441, i32 53, metadata !562, null}
!564 = metadata !{i32 441, i32 87, metadata !562, null}
!565 = metadata !{i32 441, i32 176, metadata !562, null}
!566 = metadata !{i32 443, i32 5, metadata !177, null}
!567 = metadata !{i32 444, i32 6, metadata !568, null}
!568 = metadata !{i32 786443, metadata !177, i32 443, i32 23, metadata !71, i32 92} ; [ DW_TAG_lexical_block ]
!569 = metadata !{i32 786688, metadata !570, metadata !"__Val2__", metadata !71, i32 445, metadata !77, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!570 = metadata !{i32 786443, metadata !568, i32 445, i32 21, metadata !71, i32 93} ; [ DW_TAG_lexical_block ]
!571 = metadata !{i32 445, i32 61, metadata !570, null}
!572 = metadata !{i32 445, i32 96, metadata !570, null}
!573 = metadata !{i32 445, i32 185, metadata !570, null}
!574 = metadata !{i32 446, i32 6, metadata !568, null}
!575 = metadata !{i32 447, i32 5, metadata !568, null}
!576 = metadata !{i32 428, i32 35, metadata !178, null}
!577 = metadata !{i32 450, i32 4, metadata !179, null}
!578 = metadata !{i32 451, i32 4, metadata !179, null}
!579 = metadata !{i32 452, i32 4, metadata !179, null}
!580 = metadata !{i32 453, i32 15, metadata !179, null}
!581 = metadata !{i32 454, i32 4, metadata !179, null}
!582 = metadata !{i32 455, i32 4, metadata !179, null}
!583 = metadata !{i32 456, i32 5, metadata !179, null}
!584 = metadata !{i32 459, i32 4, metadata !179, null}
!585 = metadata !{i32 76, i32 2, metadata !173, metadata !586}
!586 = metadata !{i32 460, i32 5, metadata !587, null}
!587 = metadata !{i32 786443, metadata !179, i32 459, i32 7, metadata !71, i32 94} ; [ DW_TAG_lexical_block ]
!588 = metadata !{i32 786688, metadata !589, metadata !"__Val2__", metadata !71, i32 461, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!589 = metadata !{i32 786443, metadata !179, i32 461, i32 14, metadata !71, i32 95} ; [ DW_TAG_lexical_block ]
!590 = metadata !{i32 461, i32 52, metadata !589, null}
!591 = metadata !{i32 461, i32 86, metadata !589, null}
!592 = metadata !{i32 461, i32 175, metadata !589, null}
!593 = metadata !{i32 786688, metadata !594, metadata !"__Val2__", metadata !71, i32 463, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!594 = metadata !{i32 786443, metadata !179, i32 463, i32 22, metadata !71, i32 96} ; [ DW_TAG_lexical_block ]
!595 = metadata !{i32 463, i32 60, metadata !594, null}
!596 = metadata !{i32 463, i32 94, metadata !594, null}
!597 = metadata !{i32 786688, metadata !146, metadata !"terminate_read", metadata !71, i32 155, metadata !73, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!598 = metadata !{i32 463, i32 183, metadata !594, null}
!599 = metadata !{i32 465, i32 4, metadata !179, null}
!600 = metadata !{i32 76, i32 2, metadata !173, metadata !601}
!601 = metadata !{i32 466, i32 5, metadata !602, null}
!602 = metadata !{i32 786443, metadata !179, i32 465, i32 7, metadata !71, i32 97} ; [ DW_TAG_lexical_block ]
!603 = metadata !{i32 786688, metadata !604, metadata !"__Val2__", metadata !71, i32 467, metadata !80, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!604 = metadata !{i32 786443, metadata !179, i32 467, i32 14, metadata !71, i32 98} ; [ DW_TAG_lexical_block ]
!605 = metadata !{i32 467, i32 52, metadata !604, null}
!606 = metadata !{i32 467, i32 86, metadata !604, null}
!607 = metadata !{i32 467, i32 175, metadata !604, null}
!608 = metadata !{i32 468, i32 3, metadata !179, null}
