; ModuleID = 'D:/21_streamer_car5_artix7/fpga_arty/i2c_slave_core/solution1/.autopilot/db/a.g.1.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-w64-mingw32"

@mem_wreq = global i1 false, align 1              ; [#uses=5 type=i1*]
@mem_wack = common global i1 false, align 1       ; [#uses=2 type=i1*]
@mem_rreq = global i1 false, align 1              ; [#uses=5 type=i1*]
@mem_rack = common global i1 false, align 1       ; [#uses=2 type=i1*]
@mem_dout = global i8 0, align 1                  ; [#uses=5 type=i8*]
@mem_din = common global i8 0, align 1            ; [#uses=2 type=i8*]
@mem_addr = global i8 0, align 1                  ; [#uses=8 type=i8*]
@i2c_val = common global i2 0, align 1            ; [#uses=32 type=i2*]
@i2c_sda_out = global i1 true, align 1            ; [#uses=9 type=i1*]
@i2c_sda_oe = global i1 false, align 1            ; [#uses=12 type=i1*]
@i2c_in = common global i2 0, align 1             ; [#uses=31 type=i2*]
@dev_addr_in = common global i7 0, align 1        ; [#uses=3 type=i7*]
@auto_inc_regad_in = common global i1 false, align 1 ; [#uses=4 type=i1*]
@.str7 = private unnamed_addr constant [12 x i8] c"hls_label_1\00", align 1 ; [#uses=1 type=[12 x i8]*]
@.str6 = private unnamed_addr constant [12 x i8] c"hls_label_4\00", align 1 ; [#uses=1 type=[12 x i8]*]
@.str5 = private unnamed_addr constant [17 x i8] c"label_read_start\00", align 1 ; [#uses=1 type=[17 x i8]*]
@.str4 = private unnamed_addr constant [12 x i8] c"hls_label_0\00", align 1 ; [#uses=1 type=[12 x i8]*]
@.str3 = private unnamed_addr constant [12 x i8] c"hls_label_2\00", align 1 ; [#uses=1 type=[12 x i8]*]
@.str2 = private unnamed_addr constant [12 x i8] c"hls_label_3\00", align 1 ; [#uses=1 type=[12 x i8]*]
@.str1 = private unnamed_addr constant [8 x i8] c"ap_none\00", align 1 ; [#uses=1 type=[8 x i8]*]
@.str = private unnamed_addr constant [1 x i8] zeroinitializer, align 1 ; [#uses=1 type=[1 x i8]*]

; [#uses=1]
define internal fastcc void @write_mem(i8 zeroext %addr, i8 zeroext %data) nounwind uwtable {
  call void @llvm.dbg.value(metadata !{i8 %addr}, i64 0, metadata !45), !dbg !46 ; [debug line = 80:22] [debug variable = addr]
  call void @llvm.dbg.value(metadata !{i8 %data}, i64 0, metadata !47), !dbg !48 ; [debug line = 80:34] [debug variable = data]
  call void (...)* @_ssdm_InlineSelf(i32 2, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !49 ; [debug line = 82:1]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !51 ; [debug line = 83:2]
  store volatile i8 %addr, i8* @mem_addr, align 1, !dbg !52 ; [debug line = 84:2]
  store volatile i8 %data, i8* @mem_dout, align 1, !dbg !53 ; [debug line = 85:2]
  store volatile i1 false, i1* @mem_wreq, align 1, !dbg !54 ; [debug line = 86:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !55 ; [debug line = 87:2]
  br label %._crit_edge, !dbg !56                 ; [debug line = 89:2]

._crit_edge:                                      ; preds = %._crit_edge, %0
  store volatile i8 %addr, i8* @mem_addr, align 1, !dbg !57 ; [debug line = 90:3]
  store volatile i8 %data, i8* @mem_dout, align 1, !dbg !59 ; [debug line = 91:3]
  store volatile i1 true, i1* @mem_wreq, align 1, !dbg !60 ; [debug line = 92:3]
  %mem_wack.load = load volatile i1* @mem_wack, align 1, !dbg !61 ; [#uses=1 type=i1] [debug line = 93:2]
  br i1 %mem_wack.load, label %1, label %._crit_edge, !dbg !61 ; [debug line = 93:2]

; <label>:1                                       ; preds = %._crit_edge
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !62 ; [debug line = 94:2]
  store volatile i8 %addr, i8* @mem_addr, align 1, !dbg !63 ; [debug line = 96:2]
  store volatile i8 %data, i8* @mem_dout, align 1, !dbg !64 ; [debug line = 97:2]
  store volatile i1 false, i1* @mem_wreq, align 1, !dbg !65 ; [debug line = 98:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !66 ; [debug line = 99:2]
  ret void, !dbg !67                              ; [debug line = 100:1]
}

; [#uses=2]
define internal fastcc zeroext i8 @read_mem(i8 zeroext %addr) nounwind uwtable {
  call void @llvm.dbg.value(metadata !{i8 %addr}, i64 0, metadata !68), !dbg !69 ; [debug line = 103:22] [debug variable = addr]
  call void (...)* @_ssdm_InlineSelf(i32 2, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !70 ; [debug line = 105:1]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !72 ; [debug line = 108:2]
  store volatile i8 %addr, i8* @mem_addr, align 1, !dbg !73 ; [debug line = 109:2]
  store volatile i1 true, i1* @mem_rreq, align 1, !dbg !74 ; [debug line = 110:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !75 ; [debug line = 111:2]
  br label %._crit_edge, !dbg !76                 ; [debug line = 113:2]

._crit_edge:                                      ; preds = %._crit_edge, %0
  store volatile i8 %addr, i8* @mem_addr, align 1, !dbg !77 ; [debug line = 114:3]
  store volatile i1 true, i1* @mem_rreq, align 1, !dbg !79 ; [debug line = 115:3]
  %dt = load volatile i8* @mem_din, align 1, !dbg !80 ; [#uses=1 type=i8] [debug line = 116:3]
  call void @llvm.dbg.value(metadata !{i8 %dt}, i64 0, metadata !81), !dbg !80 ; [debug line = 116:3] [debug variable = dt]
  %mem_rack.load = load volatile i1* @mem_rack, align 1, !dbg !82 ; [#uses=1 type=i1] [debug line = 117:2]
  br i1 %mem_rack.load, label %1, label %._crit_edge, !dbg !82 ; [debug line = 117:2]

; <label>:1                                       ; preds = %._crit_edge
  %dt.0.lcssa = phi i8 [ %dt, %._crit_edge ]      ; [#uses=1 type=i8]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !83 ; [debug line = 118:2]
  store volatile i8 %addr, i8* @mem_addr, align 1, !dbg !84 ; [debug line = 120:2]
  store volatile i1 false, i1* @mem_rreq, align 1, !dbg !85 ; [debug line = 121:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !86 ; [debug line = 122:2]
  ret i8 %dt.0.lcssa, !dbg !87                    ; [debug line = 124:2]
}

; [#uses=2]
declare i8 @llvm.part.select.i8(i8, i32, i32) nounwind readnone

; [#uses=46]
declare i2 @llvm.part.select.i2(i2, i32, i32) nounwind readnone

; [#uses=118]
declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

; [#uses=0]
define void @i2c_slave_core() noreturn nounwind uwtable {
  call void (...)* @_ssdm_op_SpecTopModule(), !dbg !88 ; [debug line = 133:1]
  %i2c_in.load = load volatile i2* @i2c_in, align 1, !dbg !90 ; [#uses=1 type=i2] [debug line = 134:1]
  %tmp = zext i2 %i2c_in.load to i32, !dbg !90    ; [#uses=1 type=i32] [debug line = 134:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp, i8* getelementptr inbounds ([8 x i8]* @.str1, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !90 ; [debug line = 134:1]
  %i2c_sda_out.load = load volatile i1* @i2c_sda_out, align 1, !dbg !91 ; [#uses=1 type=i1] [debug line = 135:1]
  %tmp.1 = zext i1 %i2c_sda_out.load to i32, !dbg !91 ; [#uses=1 type=i32] [debug line = 135:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.1, i8* getelementptr inbounds ([8 x i8]* @.str1, i64 0, i64 0), i32 1, i32 1, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !91 ; [debug line = 135:1]
  %i2c_sda_oe.load = load volatile i1* @i2c_sda_oe, align 1, !dbg !92 ; [#uses=1 type=i1] [debug line = 136:1]
  %tmp.2 = zext i1 %i2c_sda_oe.load to i32, !dbg !92 ; [#uses=1 type=i32] [debug line = 136:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.2, i8* getelementptr inbounds ([8 x i8]* @.str1, i64 0, i64 0), i32 1, i32 1, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !92 ; [debug line = 136:1]
  %dev_addr_in.load = load volatile i7* @dev_addr_in, align 1, !dbg !93 ; [#uses=1 type=i7] [debug line = 138:1]
  %tmp.3 = zext i7 %dev_addr_in.load to i32, !dbg !93 ; [#uses=1 type=i32] [debug line = 138:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.3, i8* getelementptr inbounds ([8 x i8]* @.str1, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !93 ; [debug line = 138:1]
  %auto_inc_regad_in.load = load volatile i1* @auto_inc_regad_in, align 1, !dbg !94 ; [#uses=1 type=i1] [debug line = 139:1]
  %tmp.4 = zext i1 %auto_inc_regad_in.load to i32, !dbg !94 ; [#uses=1 type=i32] [debug line = 139:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.4, i8* getelementptr inbounds ([8 x i8]* @.str1, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !94 ; [debug line = 139:1]
  %mem_addr.load = load volatile i8* @mem_addr, align 1, !dbg !95 ; [#uses=1 type=i8] [debug line = 141:1]
  %tmp.5 = zext i8 %mem_addr.load to i32, !dbg !95 ; [#uses=1 type=i32] [debug line = 141:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.5, i8* getelementptr inbounds ([8 x i8]* @.str1, i64 0, i64 0), i32 1, i32 1, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !95 ; [debug line = 141:1]
  %mem_din.load = load volatile i8* @mem_din, align 1, !dbg !96 ; [#uses=1 type=i8] [debug line = 142:1]
  %tmp.6 = zext i8 %mem_din.load to i32, !dbg !96 ; [#uses=1 type=i32] [debug line = 142:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.6, i8* getelementptr inbounds ([8 x i8]* @.str1, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !96 ; [debug line = 142:1]
  %mem_dout.load = load volatile i8* @mem_dout, align 1, !dbg !97 ; [#uses=1 type=i8] [debug line = 143:1]
  %tmp.7 = zext i8 %mem_dout.load to i32, !dbg !97 ; [#uses=1 type=i32] [debug line = 143:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.7, i8* getelementptr inbounds ([8 x i8]* @.str1, i64 0, i64 0), i32 1, i32 1, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !97 ; [debug line = 143:1]
  %mem_wreq.load = load volatile i1* @mem_wreq, align 1, !dbg !98 ; [#uses=1 type=i1] [debug line = 144:1]
  %tmp.8 = zext i1 %mem_wreq.load to i32, !dbg !98 ; [#uses=1 type=i32] [debug line = 144:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.8, i8* getelementptr inbounds ([8 x i8]* @.str1, i64 0, i64 0), i32 1, i32 1, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !98 ; [debug line = 144:1]
  %mem_wack.load = load volatile i1* @mem_wack, align 1, !dbg !99 ; [#uses=1 type=i1] [debug line = 145:1]
  %tmp.9 = zext i1 %mem_wack.load to i32, !dbg !99 ; [#uses=1 type=i32] [debug line = 145:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.9, i8* getelementptr inbounds ([8 x i8]* @.str1, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !99 ; [debug line = 145:1]
  %mem_rreq.load = load volatile i1* @mem_rreq, align 1, !dbg !100 ; [#uses=1 type=i1] [debug line = 146:1]
  %tmp.10 = zext i1 %mem_rreq.load to i32, !dbg !100 ; [#uses=1 type=i32] [debug line = 146:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.10, i8* getelementptr inbounds ([8 x i8]* @.str1, i64 0, i64 0), i32 1, i32 1, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !100 ; [debug line = 146:1]
  %mem_rack.load = load volatile i1* @mem_rack, align 1, !dbg !101 ; [#uses=1 type=i1] [debug line = 147:1]
  %tmp.11 = zext i1 %mem_rack.load to i32, !dbg !101 ; [#uses=1 type=i32] [debug line = 147:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.11, i8* getelementptr inbounds ([8 x i8]* @.str1, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0)), !dbg !101 ; [debug line = 147:1]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !102 ; [debug line = 158:2]
  store volatile i1 true, i1* @i2c_sda_out, align 1, !dbg !103 ; [debug line = 159:2]
  store volatile i1 false, i1* @i2c_sda_oe, align 1, !dbg !104 ; [debug line = 160:2]
  store volatile i8 0, i8* @mem_addr, align 1, !dbg !105 ; [debug line = 161:2]
  store volatile i8 0, i8* @mem_dout, align 1, !dbg !106 ; [debug line = 162:2]
  store volatile i1 false, i1* @mem_wreq, align 1, !dbg !107 ; [debug line = 163:2]
  store volatile i1 false, i1* @mem_rreq, align 1, !dbg !108 ; [debug line = 164:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !109 ; [debug line = 165:2]
  %rbegin123 = call i32 (...)* @_ssdm_op_SpecRegionBegin(i8* getelementptr inbounds ([12 x i8]* @.str2, i64 0, i64 0)) nounwind, !dbg !110 ; [#uses=0 type=i32] [debug line = 168:13]
  br label %.backedge, !dbg !110                  ; [debug line = 168:13]

.backedge.loopexit:                               ; preds = %._crit_edge101
  %re.7.lcssa = phi i8 [ %re.7, %._crit_edge101 ] ; [#uses=1 type=i8]
  %.lcssa10 = phi i8 [ %reg_data.10, %._crit_edge101 ] ; [#uses=1 type=i8]
  br label %.backedge.backedge

.backedge.backedge:                               ; preds = %.backedge.loopexit148, %.backedge.loopexit147, %.backedge.loopexit139, %.backedge.loopexit135, %.backedge.loopexit128, %.backedge.loopexit
  %reg_data.0.be = phi i8 [ %.lcssa10, %.backedge.loopexit ], [ %reg_data.4.lcssa, %.backedge.loopexit128 ], [ %.lcssa7, %.backedge.loopexit135 ], [ %reg_data.5, %.backedge.loopexit139 ], [ %reg_data, %.backedge.loopexit147 ], [ %reg_data, %.backedge.loopexit148 ] ; [#uses=1 type=i8]
  %re.0.be = phi i8 [ %re.7.lcssa, %.backedge.loopexit ], [ %re.7.lcssa1, %.backedge.loopexit128 ], [ %re.2.lcssa, %.backedge.loopexit135 ], [ %re.1.lcssa, %.backedge.loopexit139 ], [ %re, %.backedge.loopexit147 ], [ %re, %.backedge.loopexit148 ] ; [#uses=1 type=i8]
  %de.0.be = phi i7 [ %de.2.lcssa, %.backedge.loopexit ], [ %de.2.lcssa, %.backedge.loopexit128 ], [ %de.1.lcssa, %.backedge.loopexit135 ], [ %de.1.lcssa, %.backedge.loopexit139 ], [ %de, %.backedge.loopexit147 ], [ %de, %.backedge.loopexit148 ] ; [#uses=1 type=i7]
  br label %.backedge

.backedge.loopexit128:                            ; preds = %.preheader
  %re.7.lcssa1 = phi i8 [ %re.7, %.preheader ]    ; [#uses=1 type=i8]
  %reg_data.4.lcssa = phi i8 [ %reg_data.4, %.preheader ] ; [#uses=1 type=i8]
  br label %.backedge.backedge

.backedge.loopexit135:                            ; preds = %._crit_edge92
  %re.2.lcssa = phi i8 [ %re.2, %._crit_edge92 ]  ; [#uses=1 type=i8]
  %.lcssa7 = phi i8 [ %reg_data.6, %._crit_edge92 ] ; [#uses=1 type=i8]
  br label %.backedge.backedge

.backedge.loopexit139:                            ; preds = %16
  br label %.backedge.backedge

.backedge.loopexit147:                            ; preds = %.preheader50
  br label %.backedge.backedge

.backedge.loopexit148:                            ; preds = %.preheader52
  br label %.backedge.backedge

.backedge:                                        ; preds = %.backedge.backedge, %0
  %reg_data = phi i8 [ undef, %0 ], [ %reg_data.0.be, %.backedge.backedge ] ; [#uses=3 type=i8]
  %re = phi i8 [ undef, %0 ], [ %re.0.be, %.backedge.backedge ] ; [#uses=3 type=i8]
  %de = phi i7 [ undef, %0 ], [ %de.0.be, %.backedge.backedge ] ; [#uses=3 type=i7]
  %loop_begin = call i32 (...)* @_ssdm_op_SpecLoopBegin() nounwind ; [#uses=0 type=i32]
  store volatile i1 true, i1* @i2c_sda_out, align 1, !dbg !112 ; [debug line = 173:3]
  store volatile i1 false, i1* @i2c_sda_oe, align 1, !dbg !113 ; [debug line = 174:3]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !114 ; [debug line = 175:3]
  br label %.critedge, !dbg !115                  ; [debug line = 178:3]

.critedge:                                        ; preds = %.critedge.backedge, %.backedge
  %__Val2__ = load volatile i2* @i2c_in, align 1, !dbg !116 ; [#uses=3 type=i2] [debug line = 76:2@179:4]
  store i2 %__Val2__, i2* @i2c_val, align 1, !dbg !116 ; [debug line = 76:2@179:4]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__}, i64 0, metadata !120), !dbg !122 ; [debug line = 180:51] [debug variable = __Val2__]
  %__Result__ = call i2 @llvm.part.select.i2(i2 %__Val2__, i32 0, i32 0), !dbg !123 ; [#uses=1 type=i2] [debug line = 180:85]
  call void @llvm.dbg.value(metadata !{i2 %__Result__}, i64 0, metadata !124), !dbg !123 ; [debug line = 180:85] [debug variable = __Result__]
  %tmp.12 = icmp eq i2 %__Result__, 0, !dbg !123  ; [#uses=1 type=i1] [debug line = 180:85]
  br i1 %tmp.12, label %.critedge.backedge, label %1, !dbg !125 ; [debug line = 180:174]

; <label>:1                                       ; preds = %.critedge
  call void @llvm.dbg.value(metadata !{i2 %__Val2__}, i64 0, metadata !126), !dbg !128 ; [debug line = 180:224] [debug variable = __Val2__]
  %__Result__.1 = call i2 @llvm.part.select.i2(i2 %__Val2__, i32 1, i32 1), !dbg !129 ; [#uses=1 type=i2] [debug line = 180:0]
  call void @llvm.dbg.value(metadata !{i2 %__Result__.1}, i64 0, metadata !130), !dbg !129 ; [debug line = 180:0] [debug variable = __Result__]
  %tmp.13 = icmp eq i2 %__Result__.1, 0, !dbg !129 ; [#uses=1 type=i1] [debug line = 180:0]
  br i1 %tmp.13, label %.critedge.backedge, label %.preheader52.preheader, !dbg !129 ; [debug line = 180:0]

.preheader52.preheader:                           ; preds = %1
  br label %.preheader52, !dbg !131               ; [debug line = 76:2@184:4]

.critedge.backedge:                               ; preds = %1, %.critedge
  br label %.critedge

.preheader52:                                     ; preds = %2, %.preheader52.preheader
  %__Val2__.1 = load volatile i2* @i2c_in, align 1, !dbg !131 ; [#uses=3 type=i2] [debug line = 76:2@184:4]
  store i2 %__Val2__.1, i2* @i2c_val, align 1, !dbg !131 ; [debug line = 76:2@184:4]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.1}, i64 0, metadata !134), !dbg !136 ; [debug line = 185:47] [debug variable = __Val2__]
  %__Result__.2 = call i2 @llvm.part.select.i2(i2 %__Val2__.1, i32 0, i32 0), !dbg !137 ; [#uses=1 type=i2] [debug line = 185:81]
  call void @llvm.dbg.value(metadata !{i2 %__Result__.2}, i64 0, metadata !138), !dbg !137 ; [debug line = 185:81] [debug variable = __Result__]
  %tmp.14 = icmp eq i2 %__Result__.2, 0, !dbg !137 ; [#uses=1 type=i1] [debug line = 185:81]
  br i1 %tmp.14, label %.backedge.loopexit148, label %2, !dbg !139 ; [debug line = 185:170]

; <label>:2                                       ; preds = %.preheader52
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.1}, i64 0, metadata !140), !dbg !142 ; [debug line = 187:51] [debug variable = __Val2__]
  %__Result__.3 = call i2 @llvm.part.select.i2(i2 %__Val2__.1, i32 1, i32 1), !dbg !143 ; [#uses=1 type=i2] [debug line = 187:85]
  call void @llvm.dbg.value(metadata !{i2 %__Result__.3}, i64 0, metadata !144), !dbg !143 ; [debug line = 187:85] [debug variable = __Result__]
  %tmp.15 = icmp eq i2 %__Result__.3, 0, !dbg !143 ; [#uses=1 type=i1] [debug line = 187:85]
  br i1 %tmp.15, label %.preheader50.preheader, label %.preheader52, !dbg !145 ; [debug line = 187:174]

.preheader50.preheader:                           ; preds = %2
  br label %.preheader50, !dbg !146               ; [debug line = 76:2@191:4]

.preheader50:                                     ; preds = %3, %.preheader50.preheader
  %__Val2__.2 = load volatile i2* @i2c_in, align 1, !dbg !146 ; [#uses=3 type=i2] [debug line = 76:2@191:4]
  store i2 %__Val2__.2, i2* @i2c_val, align 1, !dbg !146 ; [debug line = 76:2@191:4]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.2}, i64 0, metadata !149), !dbg !151 ; [debug line = 192:47] [debug variable = __Val2__]
  %__Result__.4 = call i2 @llvm.part.select.i2(i2 %__Val2__.2, i32 1, i32 1), !dbg !152 ; [#uses=1 type=i2] [debug line = 192:81]
  call void @llvm.dbg.value(metadata !{i2 %__Result__.4}, i64 0, metadata !153), !dbg !152 ; [debug line = 192:81] [debug variable = __Result__]
  %tmp.16 = icmp eq i2 %__Result__.4, 0, !dbg !152 ; [#uses=1 type=i1] [debug line = 192:81]
  br i1 %tmp.16, label %3, label %.backedge.loopexit147, !dbg !154 ; [debug line = 192:170]

; <label>:3                                       ; preds = %.preheader50
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.2}, i64 0, metadata !155), !dbg !157 ; [debug line = 194:51] [debug variable = __Val2__]
  %__Result__.5 = call i2 @llvm.part.select.i2(i2 %__Val2__.2, i32 0, i32 0), !dbg !158 ; [#uses=1 type=i2] [debug line = 194:85]
  call void @llvm.dbg.value(metadata !{i2 %__Result__.5}, i64 0, metadata !159), !dbg !158 ; [debug line = 194:85] [debug variable = __Result__]
  %tmp.17 = icmp eq i2 %__Result__.5, 0, !dbg !158 ; [#uses=1 type=i1] [debug line = 194:85]
  br i1 %tmp.17, label %.preheader49.preheader, label %.preheader50, !dbg !160 ; [debug line = 194:174]

.preheader49.preheader:                           ; preds = %3
  br label %.preheader49, !dbg !161               ; [debug line = 199:8]

.preheader49:                                     ; preds = %5, %.preheader49.preheader
  %bit_cnt = phi i4 [ %bit_cnt.6, %5 ], [ 0, %.preheader49.preheader ] ; [#uses=2 type=i4]
  %de.1 = phi i7 [ %dev_addr, %5 ], [ %de, %.preheader49.preheader ] ; [#uses=2 type=i7]
  %exitcond1 = icmp eq i4 %bit_cnt, 7, !dbg !161  ; [#uses=1 type=i1] [debug line = 199:8]
  br i1 %exitcond1, label %6, label %.preheader48.preheader, !dbg !161 ; [debug line = 199:8]

.preheader48.preheader:                           ; preds = %.preheader49
  br label %.preheader48, !dbg !163               ; [debug line = 76:2@202:5]

.preheader48:                                     ; preds = %.preheader48, %.preheader48.preheader
  %__Val2__.4 = load volatile i2* @i2c_in, align 1, !dbg !163 ; [#uses=3 type=i2] [debug line = 76:2@202:5]
  store i2 %__Val2__.4, i2* @i2c_val, align 1, !dbg !163 ; [debug line = 76:2@202:5]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.4}, i64 0, metadata !167), !dbg !169 ; [debug line = 203:52] [debug variable = __Val2__]
  %__Result__12 = call i2 @llvm.part.select.i2(i2 %__Val2__.4, i32 0, i32 0), !dbg !170 ; [#uses=1 type=i2] [debug line = 203:86]
  call void @llvm.dbg.value(metadata !{i2 %__Result__12}, i64 0, metadata !171), !dbg !170 ; [debug line = 203:86] [debug variable = __Result__]
  %tmp.19 = icmp eq i2 %__Result__12, 0, !dbg !170 ; [#uses=1 type=i1] [debug line = 203:86]
  br i1 %tmp.19, label %.preheader48, label %4, !dbg !172 ; [debug line = 203:175]

; <label>:4                                       ; preds = %.preheader48
  %.lcssa2 = phi i2 [ %__Val2__.4, %.preheader48 ] ; [#uses=1 type=i2]
  %tmp.20 = shl i7 %de.1, 1, !dbg !173            ; [#uses=1 type=i7] [debug line = 205:195]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.4}, i64 0, metadata !175), !dbg !176 ; [debug line = 205:72] [debug variable = __Val2__]
  %__Result__14 = call i2 @llvm.part.select.i2(i2 %.lcssa2, i32 1, i32 1), !dbg !177 ; [#uses=1 type=i2] [debug line = 205:106]
  call void @llvm.dbg.value(metadata !{i2 %__Result__14}, i64 0, metadata !178), !dbg !177 ; [debug line = 205:106] [debug variable = __Result__]
  %tmp.21 = icmp ne i2 %__Result__14, 0, !dbg !177 ; [#uses=1 type=i1] [debug line = 205:106]
  %tmp.22 = zext i1 %tmp.21 to i7, !dbg !173      ; [#uses=1 type=i7] [debug line = 205:195]
  %dev_addr = or i7 %tmp.22, %tmp.20, !dbg !173   ; [#uses=1 type=i7] [debug line = 205:195]
  call void @llvm.dbg.value(metadata !{i7 %dev_addr}, i64 0, metadata !179), !dbg !173 ; [debug line = 205:195] [debug variable = dev_addr]
  br label %._crit_edge, !dbg !180                ; [debug line = 208:4]

._crit_edge:                                      ; preds = %._crit_edge, %4
  %__Val2__.6 = load volatile i2* @i2c_in, align 1, !dbg !181 ; [#uses=2 type=i2] [debug line = 76:2@209:5]
  store i2 %__Val2__.6, i2* @i2c_val, align 1, !dbg !181 ; [debug line = 76:2@209:5]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.6}, i64 0, metadata !184), !dbg !186 ; [debug line = 210:52] [debug variable = __Val2__]
  %__Result__16 = call i2 @llvm.part.select.i2(i2 %__Val2__.6, i32 0, i32 0), !dbg !187 ; [#uses=1 type=i2] [debug line = 210:86]
  call void @llvm.dbg.value(metadata !{i2 %__Result__16}, i64 0, metadata !188), !dbg !187 ; [debug line = 210:86] [debug variable = __Result__]
  %tmp.25 = icmp eq i2 %__Result__16, 0, !dbg !187 ; [#uses=1 type=i1] [debug line = 210:86]
  br i1 %tmp.25, label %5, label %._crit_edge, !dbg !189 ; [debug line = 210:175]

; <label>:5                                       ; preds = %._crit_edge
  %bit_cnt.6 = add i4 %bit_cnt, 1, !dbg !190      ; [#uses=1 type=i4] [debug line = 199:34]
  call void @llvm.dbg.value(metadata !{i4 %bit_cnt.6}, i64 0, metadata !191), !dbg !190 ; [debug line = 199:34] [debug variable = bit_cnt]
  br label %.preheader49, !dbg !190               ; [debug line = 199:34]

; <label>:6                                       ; preds = %.preheader49
  %de.1.lcssa = phi i7 [ %de.1, %.preheader49 ]   ; [#uses=4 type=i7]
  %dev_addr_in.load.1 = load volatile i7* @dev_addr_in, align 1, !dbg !194 ; [#uses=1 type=i7] [debug line = 214:3]
  %not. = icmp ne i7 %de.1.lcssa, %dev_addr_in.load.1, !dbg !194 ; [#uses=2 type=i1] [debug line = 214:3]
  br label %._crit_edge84, !dbg !195              ; [debug line = 220:3]

._crit_edge84:                                    ; preds = %._crit_edge84, %6
  %__Val2__.3 = load volatile i2* @i2c_in, align 1, !dbg !196 ; [#uses=3 type=i2] [debug line = 76:2@221:4]
  store i2 %__Val2__.3, i2* @i2c_val, align 1, !dbg !196 ; [debug line = 76:2@221:4]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.3}, i64 0, metadata !199), !dbg !201 ; [debug line = 222:51] [debug variable = __Val2__]
  %__Result__.6 = call i2 @llvm.part.select.i2(i2 %__Val2__.3, i32 0, i32 0), !dbg !202 ; [#uses=1 type=i2] [debug line = 222:85]
  call void @llvm.dbg.value(metadata !{i2 %__Result__.6}, i64 0, metadata !203), !dbg !202 ; [debug line = 222:85] [debug variable = __Result__]
  %tmp.18 = icmp eq i2 %__Result__.6, 0, !dbg !202 ; [#uses=1 type=i1] [debug line = 222:85]
  br i1 %tmp.18, label %._crit_edge84, label %7, !dbg !204 ; [debug line = 222:174]

; <label>:7                                       ; preds = %._crit_edge84
  %.lcssa1 = phi i2 [ %__Val2__.3, %._crit_edge84 ] ; [#uses=1 type=i2]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.3}, i64 0, metadata !205), !dbg !207 ; [debug line = 224:46] [debug variable = __Val2__]
  %__Result__.7 = call i2 @llvm.part.select.i2(i2 %.lcssa1, i32 1, i32 1), !dbg !208 ; [#uses=1 type=i2] [debug line = 224:80]
  call void @llvm.dbg.value(metadata !{i2 %__Result__.7}, i64 0, metadata !209), !dbg !208 ; [debug line = 224:80] [debug variable = __Result__]
  %not.1 = icmp ne i2 %__Result__.7, 0, !dbg !210 ; [#uses=2 type=i1] [debug line = 224:169]
  %ignore.0. = or i1 %not., %not.1, !dbg !210     ; [#uses=5 type=i1] [debug line = 224:169]
  br label %._crit_edge85, !dbg !211              ; [debug line = 227:3]

._crit_edge85:                                    ; preds = %._crit_edge85, %7
  %__Val2__.5 = load volatile i2* @i2c_in, align 1, !dbg !212 ; [#uses=2 type=i2] [debug line = 76:2@228:4]
  store i2 %__Val2__.5, i2* @i2c_val, align 1, !dbg !212 ; [debug line = 76:2@228:4]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.5}, i64 0, metadata !215), !dbg !217 ; [debug line = 229:51] [debug variable = __Val2__]
  %__Result__22 = call i2 @llvm.part.select.i2(i2 %__Val2__.5, i32 0, i32 0), !dbg !218 ; [#uses=1 type=i2] [debug line = 229:85]
  call void @llvm.dbg.value(metadata !{i2 %__Result__22}, i64 0, metadata !219), !dbg !218 ; [debug line = 229:85] [debug variable = __Result__]
  %tmp.24 = icmp eq i2 %__Result__22, 0, !dbg !218 ; [#uses=1 type=i1] [debug line = 229:85]
  br i1 %tmp.24, label %8, label %._crit_edge85, !dbg !220 ; [debug line = 229:174]

; <label>:8                                       ; preds = %._crit_edge85
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !221 ; [debug line = 232:3]
  store volatile i1 %ignore.0., i1* @i2c_sda_out, align 1, !dbg !222 ; [debug line = 233:3]
  %not.ignore.1.demorgan = or i1 %not., %not.1, !dbg !223 ; [#uses=1 type=i1] [debug line = 234:3]
  %not.ignore.1 = xor i1 %not.ignore.1.demorgan, true, !dbg !223 ; [#uses=3 type=i1] [debug line = 234:3]
  store volatile i1 %not.ignore.1, i1* @i2c_sda_oe, align 1, !dbg !223 ; [debug line = 234:3]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !224 ; [debug line = 235:3]
  br label %._crit_edge86, !dbg !225              ; [debug line = 237:3]

._crit_edge86:                                    ; preds = %._crit_edge86, %8
  %__Val2__.7 = load volatile i2* @i2c_in, align 1, !dbg !226 ; [#uses=2 type=i2] [debug line = 76:2@238:4]
  store i2 %__Val2__.7, i2* @i2c_val, align 1, !dbg !226 ; [debug line = 76:2@238:4]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.7}, i64 0, metadata !229), !dbg !231 ; [debug line = 239:51] [debug variable = __Val2__]
  %__Result__24 = call i2 @llvm.part.select.i2(i2 %__Val2__.7, i32 0, i32 0), !dbg !232 ; [#uses=1 type=i2] [debug line = 239:85]
  call void @llvm.dbg.value(metadata !{i2 %__Result__24}, i64 0, metadata !233), !dbg !232 ; [debug line = 239:85] [debug variable = __Result__]
  %tmp.27 = icmp eq i2 %__Result__24, 0, !dbg !232 ; [#uses=1 type=i1] [debug line = 239:85]
  br i1 %tmp.27, label %._crit_edge86, label %.preheader47.preheader, !dbg !234 ; [debug line = 239:174]

.preheader47.preheader:                           ; preds = %._crit_edge86
  br label %.preheader47, !dbg !235               ; [debug line = 76:2@242:4]

.preheader47:                                     ; preds = %.preheader47, %.preheader47.preheader
  %__Val2__.8 = load volatile i2* @i2c_in, align 1, !dbg !235 ; [#uses=2 type=i2] [debug line = 76:2@242:4]
  store i2 %__Val2__.8, i2* @i2c_val, align 1, !dbg !235 ; [debug line = 76:2@242:4]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.8}, i64 0, metadata !238), !dbg !240 ; [debug line = 243:51] [debug variable = __Val2__]
  %__Result__26 = call i2 @llvm.part.select.i2(i2 %__Val2__.8, i32 0, i32 0), !dbg !241 ; [#uses=1 type=i2] [debug line = 243:85]
  call void @llvm.dbg.value(metadata !{i2 %__Result__26}, i64 0, metadata !242), !dbg !241 ; [debug line = 243:85] [debug variable = __Result__]
  %tmp.28 = icmp eq i2 %__Result__26, 0, !dbg !241 ; [#uses=1 type=i1] [debug line = 243:85]
  br i1 %tmp.28, label %9, label %.preheader47, !dbg !243 ; [debug line = 243:174]

; <label>:9                                       ; preds = %.preheader47
  store volatile i1 false, i1* @i2c_sda_oe, align 1, !dbg !244 ; [debug line = 245:3]
  br label %10, !dbg !245                         ; [debug line = 250:8]

; <label>:10                                      ; preds = %12, %9
  %bit_cnt.1 = phi i4 [ 0, %9 ], [ %bit_cnt.7, %12 ] ; [#uses=2 type=i4]
  %re.1 = phi i8 [ %re, %9 ], [ %reg_addr, %12 ]  ; [#uses=2 type=i8]
  %exitcond2 = icmp eq i4 %bit_cnt.1, -8, !dbg !245 ; [#uses=1 type=i1] [debug line = 250:8]
  br i1 %exitcond2, label %13, label %.preheader46.preheader, !dbg !245 ; [debug line = 250:8]

.preheader46.preheader:                           ; preds = %10
  br label %.preheader46, !dbg !247               ; [debug line = 76:2@253:5]

.preheader46:                                     ; preds = %.preheader46, %.preheader46.preheader
  %__Val2__.10 = load volatile i2* @i2c_in, align 1, !dbg !247 ; [#uses=3 type=i2] [debug line = 76:2@253:5]
  store i2 %__Val2__.10, i2* @i2c_val, align 1, !dbg !247 ; [debug line = 76:2@253:5]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.10}, i64 0, metadata !251), !dbg !253 ; [debug line = 254:52] [debug variable = __Val2__]
  %__Result__.8 = call i2 @llvm.part.select.i2(i2 %__Val2__.10, i32 0, i32 0), !dbg !254 ; [#uses=1 type=i2] [debug line = 254:86]
  call void @llvm.dbg.value(metadata !{i2 %__Result__.8}, i64 0, metadata !255), !dbg !254 ; [debug line = 254:86] [debug variable = __Result__]
  %tmp.30 = icmp eq i2 %__Result__.8, 0, !dbg !254 ; [#uses=1 type=i1] [debug line = 254:86]
  br i1 %tmp.30, label %.preheader46, label %11, !dbg !256 ; [debug line = 254:175]

; <label>:11                                      ; preds = %.preheader46
  %.lcssa3 = phi i2 [ %__Val2__.10, %.preheader46 ] ; [#uses=1 type=i2]
  %tmp.31 = shl i8 %re.1, 1, !dbg !257            ; [#uses=1 type=i8] [debug line = 256:195]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.10}, i64 0, metadata !259), !dbg !260 ; [debug line = 256:72] [debug variable = __Val2__]
  %__Result__.9 = call i2 @llvm.part.select.i2(i2 %.lcssa3, i32 1, i32 1), !dbg !261 ; [#uses=1 type=i2] [debug line = 256:106]
  call void @llvm.dbg.value(metadata !{i2 %__Result__.9}, i64 0, metadata !262), !dbg !261 ; [debug line = 256:106] [debug variable = __Result__]
  %tmp.32 = icmp ne i2 %__Result__.9, 0, !dbg !261 ; [#uses=1 type=i1] [debug line = 256:106]
  %tmp.33 = zext i1 %tmp.32 to i8, !dbg !257      ; [#uses=1 type=i8] [debug line = 256:195]
  %reg_addr = or i8 %tmp.33, %tmp.31, !dbg !257   ; [#uses=1 type=i8] [debug line = 256:195]
  call void @llvm.dbg.value(metadata !{i8 %reg_addr}, i64 0, metadata !263), !dbg !257 ; [debug line = 256:195] [debug variable = reg_addr]
  br label %._crit_edge87, !dbg !264              ; [debug line = 259:4]

._crit_edge87:                                    ; preds = %._crit_edge87, %11
  %__Val2__.12 = load volatile i2* @i2c_in, align 1, !dbg !265 ; [#uses=2 type=i2] [debug line = 76:2@260:5]
  store i2 %__Val2__.12, i2* @i2c_val, align 1, !dbg !265 ; [debug line = 76:2@260:5]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.12}, i64 0, metadata !268), !dbg !270 ; [debug line = 261:52] [debug variable = __Val2__]
  %__Result__32 = call i2 @llvm.part.select.i2(i2 %__Val2__.12, i32 0, i32 0), !dbg !271 ; [#uses=1 type=i2] [debug line = 261:86]
  call void @llvm.dbg.value(metadata !{i2 %__Result__32}, i64 0, metadata !272), !dbg !271 ; [debug line = 261:86] [debug variable = __Result__]
  %tmp.36 = icmp eq i2 %__Result__32, 0, !dbg !271 ; [#uses=1 type=i1] [debug line = 261:86]
  br i1 %tmp.36, label %12, label %._crit_edge87, !dbg !273 ; [debug line = 261:175]

; <label>:12                                      ; preds = %._crit_edge87
  %bit_cnt.7 = add i4 %bit_cnt.1, 1, !dbg !274    ; [#uses=1 type=i4] [debug line = 250:34]
  call void @llvm.dbg.value(metadata !{i4 %bit_cnt.7}, i64 0, metadata !191), !dbg !274 ; [debug line = 250:34] [debug variable = bit_cnt]
  br label %10, !dbg !274                         ; [debug line = 250:34]

; <label>:13                                      ; preds = %10
  %re.1.lcssa = phi i8 [ %re.1, %10 ]             ; [#uses=6 type=i8]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !275 ; [debug line = 265:3]
  store volatile i1 %ignore.0., i1* @i2c_sda_out, align 1, !dbg !276 ; [debug line = 266:3]
  store volatile i1 %not.ignore.1, i1* @i2c_sda_oe, align 1, !dbg !277 ; [debug line = 267:3]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !278 ; [debug line = 268:3]
  br label %._crit_edge88, !dbg !279              ; [debug line = 270:3]

._crit_edge88:                                    ; preds = %._crit_edge88, %13
  %__Val2__.9 = load volatile i2* @i2c_in, align 1, !dbg !280 ; [#uses=2 type=i2] [debug line = 76:2@271:4]
  store i2 %__Val2__.9, i2* @i2c_val, align 1, !dbg !280 ; [debug line = 76:2@271:4]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.9}, i64 0, metadata !283), !dbg !285 ; [debug line = 272:51] [debug variable = __Val2__]
  %__Result__34 = call i2 @llvm.part.select.i2(i2 %__Val2__.9, i32 0, i32 0), !dbg !286 ; [#uses=1 type=i2] [debug line = 272:85]
  call void @llvm.dbg.value(metadata !{i2 %__Result__34}, i64 0, metadata !287), !dbg !286 ; [debug line = 272:85] [debug variable = __Result__]
  %tmp.29 = icmp eq i2 %__Result__34, 0, !dbg !286 ; [#uses=1 type=i1] [debug line = 272:85]
  br i1 %tmp.29, label %._crit_edge88, label %.preheader45.preheader, !dbg !288 ; [debug line = 272:174]

.preheader45.preheader:                           ; preds = %._crit_edge88
  br label %.preheader45, !dbg !289               ; [debug line = 76:2@275:4]

.preheader45:                                     ; preds = %.preheader45, %.preheader45.preheader
  %__Val2__.11 = load volatile i2* @i2c_in, align 1, !dbg !289 ; [#uses=2 type=i2] [debug line = 76:2@275:4]
  store i2 %__Val2__.11, i2* @i2c_val, align 1, !dbg !289 ; [debug line = 76:2@275:4]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.11}, i64 0, metadata !292), !dbg !294 ; [debug line = 276:51] [debug variable = __Val2__]
  %__Result__36 = call i2 @llvm.part.select.i2(i2 %__Val2__.11, i32 0, i32 0), !dbg !295 ; [#uses=1 type=i2] [debug line = 276:85]
  call void @llvm.dbg.value(metadata !{i2 %__Result__36}, i64 0, metadata !296), !dbg !295 ; [debug line = 276:85] [debug variable = __Result__]
  %tmp.35 = icmp eq i2 %__Result__36, 0, !dbg !295 ; [#uses=1 type=i1] [debug line = 276:85]
  br i1 %tmp.35, label %14, label %.preheader45, !dbg !297 ; [debug line = 276:174]

; <label>:14                                      ; preds = %.preheader45
  store volatile i1 false, i1* @i2c_sda_oe, align 1, !dbg !298 ; [debug line = 278:3]
  br label %._crit_edge89, !dbg !299              ; [debug line = 285:3]

._crit_edge89:                                    ; preds = %._crit_edge89, %14
  %__Val2__.13 = load volatile i2* @i2c_in, align 1, !dbg !300 ; [#uses=3 type=i2] [debug line = 76:2@286:4]
  store i2 %__Val2__.13, i2* @i2c_val, align 1, !dbg !300 ; [debug line = 76:2@286:4]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.13}, i64 0, metadata !303), !dbg !305 ; [debug line = 287:51] [debug variable = __Val2__]
  %__Result__38 = call i2 @llvm.part.select.i2(i2 %__Val2__.13, i32 0, i32 0), !dbg !306 ; [#uses=1 type=i2] [debug line = 287:85]
  call void @llvm.dbg.value(metadata !{i2 %__Result__38}, i64 0, metadata !307), !dbg !306 ; [debug line = 287:85] [debug variable = __Result__]
  %tmp.38 = icmp eq i2 %__Result__38, 0, !dbg !306 ; [#uses=1 type=i1] [debug line = 287:85]
  br i1 %tmp.38, label %._crit_edge89, label %15, !dbg !308 ; [debug line = 287:174]

; <label>:15                                      ; preds = %._crit_edge89
  %.lcssa4 = phi i2 [ %__Val2__.13, %._crit_edge89 ] ; [#uses=1 type=i2]
  %tmp.39 = shl i8 %reg_data, 1, !dbg !309        ; [#uses=1 type=i8] [debug line = 289:194]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.13}, i64 0, metadata !311), !dbg !312 ; [debug line = 289:71] [debug variable = __Val2__]
  %__Result__.10 = call i2 @llvm.part.select.i2(i2 %.lcssa4, i32 1, i32 1), !dbg !313 ; [#uses=1 type=i2] [debug line = 289:105]
  call void @llvm.dbg.value(metadata !{i2 %__Result__.10}, i64 0, metadata !314), !dbg !313 ; [debug line = 289:105] [debug variable = __Result__]
  %tmp.40 = icmp ne i2 %__Result__.10, 0, !dbg !313 ; [#uses=1 type=i1] [debug line = 289:105]
  %tmp.41 = zext i1 %tmp.40 to i8, !dbg !309      ; [#uses=1 type=i8] [debug line = 289:194]
  %reg_data.5 = or i8 %tmp.41, %tmp.39, !dbg !309 ; [#uses=2 type=i8] [debug line = 289:194]
  call void @llvm.dbg.value(metadata !{i8 %reg_data.5}, i64 0, metadata !315), !dbg !309 ; [debug line = 289:194] [debug variable = reg_data]
  br label %._crit_edge91, !dbg !316              ; [debug line = 291:3]

._crit_edge91:                                    ; preds = %17, %15
  %__Val2__.14 = load i2* @i2c_val, align 1, !dbg !317 ; [#uses=1 type=i2] [debug line = 292:61]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.14}, i64 0, metadata !320), !dbg !317 ; [debug line = 292:61] [debug variable = __Val2__]
  %__Result__42 = call i2 @llvm.part.select.i2(i2 %__Val2__.14, i32 1, i32 1), !dbg !321 ; [#uses=1 type=i2] [debug line = 292:95]
  call void @llvm.dbg.value(metadata !{i2 %__Result__42}, i64 0, metadata !322), !dbg !321 ; [debug line = 292:95] [debug variable = __Result__]
  %pre_i2c_sda_val = icmp ne i2 %__Result__42, 0, !dbg !321 ; [#uses=2 type=i1] [debug line = 292:95]
  call void @llvm.dbg.value(metadata !{i1 %pre_i2c_sda_val}, i64 0, metadata !323), !dbg !324 ; [debug line = 292:184] [debug variable = pre_i2c_sda_val]
  %__Val2__.15 = load volatile i2* @i2c_in, align 1, !dbg !325 ; [#uses=3 type=i2] [debug line = 76:2@293:4]
  store i2 %__Val2__.15, i2* @i2c_val, align 1, !dbg !325 ; [debug line = 76:2@293:4]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.15}, i64 0, metadata !327), !dbg !329 ; [debug line = 295:47] [debug variable = __Val2__]
  %__Result__44 = call i2 @llvm.part.select.i2(i2 %__Val2__.15, i32 0, i32 0), !dbg !330 ; [#uses=1 type=i2] [debug line = 295:81]
  call void @llvm.dbg.value(metadata !{i2 %__Result__44}, i64 0, metadata !331), !dbg !330 ; [debug line = 295:81] [debug variable = __Result__]
  %tmp.44 = icmp eq i2 %__Result__44, 0, !dbg !330 ; [#uses=1 type=i1] [debug line = 295:81]
  br i1 %tmp.44, label %.preheader40.preheader, label %16, !dbg !332 ; [debug line = 295:170]

.preheader40.preheader:                           ; preds = %._crit_edge91
  br label %.preheader40, !dbg !333               ; [debug line = 313:7]

; <label>:16                                      ; preds = %._crit_edge91
  br i1 %ignore.0., label %.backedge.loopexit139, label %17, !dbg !335 ; [debug line = 297:4]

; <label>:17                                      ; preds = %16
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.15}, i64 0, metadata !336), !dbg !338 ; [debug line = 297:62] [debug variable = __Val2__]
  %__Result__46 = call i2 @llvm.part.select.i2(i2 %__Val2__.15, i32 1, i32 1), !dbg !339 ; [#uses=1 type=i2] [debug line = 297:96]
  call void @llvm.dbg.value(metadata !{i2 %__Result__46}, i64 0, metadata !340), !dbg !339 ; [debug line = 297:96] [debug variable = __Result__]
  %.not = icmp ne i2 %__Result__46, 0, !dbg !341  ; [#uses=1 type=i1] [debug line = 297:185]
  %.not1 = xor i1 %pre_i2c_sda_val, true, !dbg !341 ; [#uses=1 type=i1] [debug line = 297:185]
  %brmerge = or i1 %.not, %.not1, !dbg !341       ; [#uses=1 type=i1] [debug line = 297:185]
  br i1 %brmerge, label %._crit_edge91, label %.preheader41.preheader, !dbg !341 ; [debug line = 297:185]

.preheader41.preheader:                           ; preds = %17
  %.lcssa5 = phi i1 [ %pre_i2c_sda_val, %17 ]     ; [#uses=1 type=i1]
  br label %.preheader41, !dbg !342               ; [debug line = 76:2@299:6]

.preheader41:                                     ; preds = %.preheader41, %.preheader41.preheader
  %__Val2__.16 = load volatile i2* @i2c_in, align 1, !dbg !342 ; [#uses=2 type=i2] [debug line = 76:2@299:6]
  store i2 %__Val2__.16, i2* @i2c_val, align 1, !dbg !342 ; [debug line = 76:2@299:6]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.16}, i64 0, metadata !346), !dbg !348 ; [debug line = 300:53] [debug variable = __Val2__]
  %__Result__48 = call i2 @llvm.part.select.i2(i2 %__Val2__.16, i32 0, i32 0), !dbg !349 ; [#uses=1 type=i2] [debug line = 300:87]
  call void @llvm.dbg.value(metadata !{i2 %__Result__48}, i64 0, metadata !350), !dbg !349 ; [debug line = 300:87] [debug variable = __Result__]
  %tmp.46 = icmp eq i2 %__Result__48, 0, !dbg !349 ; [#uses=1 type=i1] [debug line = 300:87]
  br i1 %tmp.46, label %.preheader34.preheader, label %.preheader41, !dbg !351 ; [debug line = 300:176]

.preheader34.preheader:                           ; preds = %.preheader41
  br label %.preheader34, !dbg !352               ; [debug line = 367:8]

.preheader40:                                     ; preds = %._crit_edge94, %.preheader40.preheader
  %bit_cnt.2 = phi i4 [ 0, %._crit_edge94 ], [ 1, %.preheader40.preheader ] ; [#uses=1 type=i4]
  %reg_data.1 = phi i8 [ %reg_data.2.lcssa, %._crit_edge94 ], [ %reg_data.5, %.preheader40.preheader ] ; [#uses=1 type=i8]
  %re.2 = phi i8 [ %re.4, %._crit_edge94 ], [ %re.1.lcssa, %.preheader40.preheader ] ; [#uses=5 type=i8]
  %rbegin1 = call i32 (...)* @_ssdm_op_SpecRegionBegin(i8* getelementptr inbounds ([12 x i8]* @.str3, i64 0, i64 0)) nounwind, !dbg !333 ; [#uses=1 type=i32] [debug line = 313:7]
  br label %18, !dbg !354                         ; [debug line = 314:4]

; <label>:18                                      ; preds = %21, %.preheader40
  %bit_cnt.3 = phi i4 [ %bit_cnt.2, %.preheader40 ], [ %bit_cnt.8, %21 ] ; [#uses=2 type=i4]
  %reg_data.2 = phi i8 [ %reg_data.1, %.preheader40 ], [ %reg_data.6, %21 ] ; [#uses=2 type=i8]
  %tmp.45 = icmp sgt i4 %bit_cnt.3, -1, !dbg !354 ; [#uses=1 type=i1] [debug line = 314:4]
  br i1 %tmp.45, label %.preheader37.preheader, label %22, !dbg !354 ; [debug line = 314:4]

.preheader37.preheader:                           ; preds = %18
  br label %.preheader37, !dbg !355               ; [debug line = 76:2@317:6]

.preheader37:                                     ; preds = %.preheader37, %.preheader37.preheader
  %__Val2__.17 = load volatile i2* @i2c_in, align 1, !dbg !355 ; [#uses=3 type=i2] [debug line = 76:2@317:6]
  store i2 %__Val2__.17, i2* @i2c_val, align 1, !dbg !355 ; [debug line = 76:2@317:6]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.17}, i64 0, metadata !359), !dbg !361 ; [debug line = 318:53] [debug variable = __Val2__]
  %__Result__.11 = call i2 @llvm.part.select.i2(i2 %__Val2__.17, i32 0, i32 0), !dbg !362 ; [#uses=1 type=i2] [debug line = 318:87]
  call void @llvm.dbg.value(metadata !{i2 %__Result__.11}, i64 0, metadata !363), !dbg !362 ; [debug line = 318:87] [debug variable = __Result__]
  %tmp.47 = icmp eq i2 %__Result__.11, 0, !dbg !362 ; [#uses=1 type=i1] [debug line = 318:87]
  br i1 %tmp.47, label %.preheader37, label %19, !dbg !364 ; [debug line = 318:176]

; <label>:19                                      ; preds = %.preheader37
  %.lcssa6 = phi i2 [ %__Val2__.17, %.preheader37 ] ; [#uses=1 type=i2]
  %tmp.49 = shl i8 %reg_data.2, 1, !dbg !365      ; [#uses=1 type=i8] [debug line = 320:196]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.17}, i64 0, metadata !367), !dbg !368 ; [debug line = 320:73] [debug variable = __Val2__]
  %__Result__52 = call i2 @llvm.part.select.i2(i2 %.lcssa6, i32 1, i32 1), !dbg !369 ; [#uses=1 type=i2] [debug line = 320:107]
  call void @llvm.dbg.value(metadata !{i2 %__Result__52}, i64 0, metadata !370), !dbg !369 ; [debug line = 320:107] [debug variable = __Result__]
  %tmp.50 = icmp ne i2 %__Result__52, 0, !dbg !369 ; [#uses=1 type=i1] [debug line = 320:107]
  %tmp.51 = zext i1 %tmp.50 to i8, !dbg !365      ; [#uses=1 type=i8] [debug line = 320:196]
  %reg_data.6 = or i8 %tmp.51, %tmp.49, !dbg !365 ; [#uses=2 type=i8] [debug line = 320:196]
  call void @llvm.dbg.value(metadata !{i8 %reg_data.6}, i64 0, metadata !315), !dbg !365 ; [debug line = 320:196] [debug variable = reg_data]
  br label %._crit_edge92, !dbg !371              ; [debug line = 323:5]

._crit_edge92:                                    ; preds = %20, %19
  %__Val2__.19 = load i2* @i2c_val, align 1, !dbg !372 ; [#uses=1 type=i2] [debug line = 324:63]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.19}, i64 0, metadata !375), !dbg !372 ; [debug line = 324:63] [debug variable = __Val2__]
  %__Result__54 = call i2 @llvm.part.select.i2(i2 %__Val2__.19, i32 1, i32 1), !dbg !376 ; [#uses=1 type=i2] [debug line = 324:97]
  call void @llvm.dbg.value(metadata !{i2 %__Result__54}, i64 0, metadata !377), !dbg !376 ; [debug line = 324:97] [debug variable = __Result__]
  %__Val2__.20 = load volatile i2* @i2c_in, align 1, !dbg !378 ; [#uses=3 type=i2] [debug line = 76:2@325:6]
  store i2 %__Val2__.20, i2* @i2c_val, align 1, !dbg !378 ; [debug line = 76:2@325:6]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.20}, i64 0, metadata !380), !dbg !382 ; [debug line = 326:49] [debug variable = __Val2__]
  %__Result__56 = call i2 @llvm.part.select.i2(i2 %__Val2__.20, i32 1, i32 1), !dbg !383 ; [#uses=1 type=i2] [debug line = 326:83]
  call void @llvm.dbg.value(metadata !{i2 %__Result__56}, i64 0, metadata !384), !dbg !383 ; [debug line = 326:83] [debug variable = __Result__]
  %tmp.53 = icmp eq i2 %__Result__56, 0, !dbg !383 ; [#uses=1 type=i1] [debug line = 326:83]
  %.not2 = icmp ne i2 %__Result__54, 0, !dbg !385 ; [#uses=1 type=i1] [debug line = 326:172]
  %brmerge1 = or i1 %tmp.53, %.not2, !dbg !385    ; [#uses=1 type=i1] [debug line = 326:172]
  br i1 %brmerge1, label %20, label %.backedge.loopexit135, !dbg !385 ; [debug line = 326:172]

; <label>:20                                      ; preds = %._crit_edge92
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.20}, i64 0, metadata !386), !dbg !388 ; [debug line = 328:53] [debug variable = __Val2__]
  %__Result__58 = call i2 @llvm.part.select.i2(i2 %__Val2__.20, i32 0, i32 0), !dbg !389 ; [#uses=1 type=i2] [debug line = 328:87]
  call void @llvm.dbg.value(metadata !{i2 %__Result__58}, i64 0, metadata !390), !dbg !389 ; [debug line = 328:87] [debug variable = __Result__]
  %tmp.55 = icmp eq i2 %__Result__58, 0, !dbg !389 ; [#uses=1 type=i1] [debug line = 328:87]
  br i1 %tmp.55, label %21, label %._crit_edge92, !dbg !391 ; [debug line = 328:176]

; <label>:21                                      ; preds = %20
  %bit_cnt.8 = add i4 %bit_cnt.3, 1, !dbg !392    ; [#uses=1 type=i4] [debug line = 330:5]
  call void @llvm.dbg.value(metadata !{i4 %bit_cnt.8}, i64 0, metadata !191), !dbg !392 ; [debug line = 330:5] [debug variable = bit_cnt]
  br label %18, !dbg !393                         ; [debug line = 331:4]

; <label>:22                                      ; preds = %18
  %reg_data.2.lcssa = phi i8 [ %reg_data.2, %18 ] ; [#uses=2 type=i8]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !394 ; [debug line = 334:4]
  store volatile i1 %ignore.0., i1* @i2c_sda_out, align 1, !dbg !395 ; [debug line = 335:4]
  store volatile i1 %not.ignore.1, i1* @i2c_sda_oe, align 1, !dbg !396 ; [debug line = 336:4]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !397 ; [debug line = 337:4]
  br label %._crit_edge93, !dbg !398              ; [debug line = 339:4]

._crit_edge93:                                    ; preds = %._crit_edge93, %22
  %__Val2__.18 = load volatile i2* @i2c_in, align 1, !dbg !399 ; [#uses=2 type=i2] [debug line = 76:2@340:5]
  store i2 %__Val2__.18, i2* @i2c_val, align 1, !dbg !399 ; [debug line = 76:2@340:5]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.18}, i64 0, metadata !402), !dbg !404 ; [debug line = 341:52] [debug variable = __Val2__]
  %__Result__.12 = call i2 @llvm.part.select.i2(i2 %__Val2__.18, i32 0, i32 0), !dbg !405 ; [#uses=1 type=i2] [debug line = 341:86]
  call void @llvm.dbg.value(metadata !{i2 %__Result__.12}, i64 0, metadata !406), !dbg !405 ; [debug line = 341:86] [debug variable = __Result__]
  %tmp.48 = icmp eq i2 %__Result__.12, 0, !dbg !405 ; [#uses=1 type=i1] [debug line = 341:86]
  br i1 %tmp.48, label %._crit_edge93, label %.preheader35.preheader, !dbg !407 ; [debug line = 341:175]

.preheader35.preheader:                           ; preds = %._crit_edge93
  br label %.preheader35, !dbg !408               ; [debug line = 76:2@344:5]

.preheader35:                                     ; preds = %.preheader35, %.preheader35.preheader
  %__Val2__.21 = load volatile i2* @i2c_in, align 1, !dbg !408 ; [#uses=2 type=i2] [debug line = 76:2@344:5]
  store i2 %__Val2__.21, i2* @i2c_val, align 1, !dbg !408 ; [debug line = 76:2@344:5]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.21}, i64 0, metadata !411), !dbg !413 ; [debug line = 345:52] [debug variable = __Val2__]
  %__Result__62 = call i2 @llvm.part.select.i2(i2 %__Val2__.21, i32 0, i32 0), !dbg !414 ; [#uses=1 type=i2] [debug line = 345:86]
  call void @llvm.dbg.value(metadata !{i2 %__Result__62}, i64 0, metadata !415), !dbg !414 ; [debug line = 345:86] [debug variable = __Result__]
  %tmp.54 = icmp eq i2 %__Result__62, 0, !dbg !414 ; [#uses=1 type=i1] [debug line = 345:86]
  br i1 %tmp.54, label %23, label %.preheader35, !dbg !416 ; [debug line = 345:175]

; <label>:23                                      ; preds = %.preheader35
  store volatile i1 false, i1* @i2c_sda_oe, align 1, !dbg !417 ; [debug line = 347:4]
  br i1 %ignore.0., label %._crit_edge94, label %24, !dbg !418 ; [debug line = 350:4]

; <label>:24                                      ; preds = %23
  %rbegin = call i32 (...)* @_ssdm_op_SpecRegionBegin(i8* getelementptr inbounds ([12 x i8]* @.str4, i64 0, i64 0)) nounwind, !dbg !419 ; [#uses=1 type=i32] [debug line = 350:22]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !421 ; [debug line = 351:5]
  call fastcc void @write_mem(i8 zeroext %re.2, i8 zeroext %reg_data.2.lcssa), !dbg !422 ; [debug line = 352:5]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !423 ; [debug line = 353:5]
  %auto_inc_regad_in.load.1 = load volatile i1* @auto_inc_regad_in, align 1, !dbg !424 ; [#uses=1 type=i1] [debug line = 354:5]
  %reg_addr.1 = add i8 %re.2, 1, !dbg !425        ; [#uses=1 type=i8] [debug line = 355:6]
  call void @llvm.dbg.value(metadata !{i8 %reg_addr.1}, i64 0, metadata !263), !dbg !425 ; [debug line = 355:6] [debug variable = reg_addr]
  %.re.2 = select i1 %auto_inc_regad_in.load.1, i8 %reg_addr.1, i8 %re.2, !dbg !424 ; [#uses=1 type=i8] [debug line = 354:5]
  %rend = call i32 (...)* @_ssdm_op_SpecRegionEnd(i8* getelementptr inbounds ([12 x i8]* @.str4, i64 0, i64 0), i32 %rbegin) nounwind, !dbg !426 ; [#uses=0 type=i32] [debug line = 356:4]
  br label %._crit_edge94, !dbg !426              ; [debug line = 356:4]

._crit_edge94:                                    ; preds = %24, %23
  %re.4 = phi i8 [ %.re.2, %24 ], [ %re.2, %23 ]  ; [#uses=1 type=i8]
  %rend122 = call i32 (...)* @_ssdm_op_SpecRegionEnd(i8* getelementptr inbounds ([12 x i8]* @.str3, i64 0, i64 0), i32 %rbegin1) nounwind, !dbg !427 ; [#uses=0 type=i32] [debug line = 359:3]
  br label %.preheader40, !dbg !427               ; [debug line = 359:3]

.preheader34:                                     ; preds = %27, %.preheader34.preheader
  %bit_cnt.4 = phi i4 [ %bit_cnt.9, %27 ], [ 0, %.preheader34.preheader ] ; [#uses=2 type=i4]
  %de.2 = phi i7 [ %dev_addr.1, %27 ], [ %de.1.lcssa, %.preheader34.preheader ] ; [#uses=2 type=i7]
  %exitcond = icmp eq i4 %bit_cnt.4, 7, !dbg !352 ; [#uses=1 type=i1] [debug line = 367:8]
  br i1 %exitcond, label %28, label %25, !dbg !352 ; [debug line = 367:8]

; <label>:25                                      ; preds = %.preheader34
  call void (...)* @_ssdm_op_SpecLoopName(i8* getelementptr inbounds ([17 x i8]* @.str5, i64 0, i64 0)), !dbg !428 ; [debug line = 367:46]
  %rbegin2 = call i32 (...)* @_ssdm_op_SpecRegionBegin(i8* getelementptr inbounds ([17 x i8]* @.str5, i64 0, i64 0)) nounwind, !dbg !428 ; [#uses=1 type=i32] [debug line = 367:46]
  br label %._crit_edge95, !dbg !430              ; [debug line = 369:4]

._crit_edge95:                                    ; preds = %._crit_edge95, %25
  %__Val2__.23 = load volatile i2* @i2c_in, align 1, !dbg !431 ; [#uses=3 type=i2] [debug line = 76:2@370:5]
  store i2 %__Val2__.23, i2* @i2c_val, align 1, !dbg !431 ; [debug line = 76:2@370:5]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.23}, i64 0, metadata !434), !dbg !436 ; [debug line = 371:52] [debug variable = __Val2__]
  %__Result__64 = call i2 @llvm.part.select.i2(i2 %__Val2__.23, i32 0, i32 0), !dbg !437 ; [#uses=1 type=i2] [debug line = 371:86]
  call void @llvm.dbg.value(metadata !{i2 %__Result__64}, i64 0, metadata !438), !dbg !437 ; [debug line = 371:86] [debug variable = __Result__]
  %tmp.57 = icmp eq i2 %__Result__64, 0, !dbg !437 ; [#uses=1 type=i1] [debug line = 371:86]
  br i1 %tmp.57, label %._crit_edge95, label %26, !dbg !439 ; [debug line = 371:175]

; <label>:26                                      ; preds = %._crit_edge95
  %.lcssa9 = phi i2 [ %__Val2__.23, %._crit_edge95 ] ; [#uses=1 type=i2]
  %tmp.61 = shl i7 %de.2, 1, !dbg !440            ; [#uses=1 type=i7] [debug line = 373:195]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.23}, i64 0, metadata !442), !dbg !443 ; [debug line = 373:72] [debug variable = __Val2__]
  %__Result__66 = call i2 @llvm.part.select.i2(i2 %.lcssa9, i32 1, i32 1), !dbg !444 ; [#uses=1 type=i2] [debug line = 373:106]
  call void @llvm.dbg.value(metadata !{i2 %__Result__66}, i64 0, metadata !445), !dbg !444 ; [debug line = 373:106] [debug variable = __Result__]
  %tmp.62 = icmp ne i2 %__Result__66, 0, !dbg !444 ; [#uses=1 type=i1] [debug line = 373:106]
  %tmp.63 = zext i1 %tmp.62 to i7, !dbg !440      ; [#uses=1 type=i7] [debug line = 373:195]
  %dev_addr.1 = or i7 %tmp.63, %tmp.61, !dbg !440 ; [#uses=1 type=i7] [debug line = 373:195]
  call void @llvm.dbg.value(metadata !{i7 %dev_addr.1}, i64 0, metadata !179), !dbg !440 ; [debug line = 373:195] [debug variable = dev_addr]
  br label %._crit_edge96, !dbg !446              ; [debug line = 376:4]

._crit_edge96:                                    ; preds = %._crit_edge96, %26
  %__Val2__.25 = load volatile i2* @i2c_in, align 1, !dbg !447 ; [#uses=2 type=i2] [debug line = 76:2@377:5]
  store i2 %__Val2__.25, i2* @i2c_val, align 1, !dbg !447 ; [debug line = 76:2@377:5]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.25}, i64 0, metadata !450), !dbg !452 ; [debug line = 378:52] [debug variable = __Val2__]
  %__Result__68 = call i2 @llvm.part.select.i2(i2 %__Val2__.25, i32 0, i32 0), !dbg !453 ; [#uses=1 type=i2] [debug line = 378:86]
  call void @llvm.dbg.value(metadata !{i2 %__Result__68}, i64 0, metadata !454), !dbg !453 ; [debug line = 378:86] [debug variable = __Result__]
  %tmp.66 = icmp eq i2 %__Result__68, 0, !dbg !453 ; [#uses=1 type=i1] [debug line = 378:86]
  br i1 %tmp.66, label %27, label %._crit_edge96, !dbg !455 ; [debug line = 378:175]

; <label>:27                                      ; preds = %._crit_edge96
  %rend127 = call i32 (...)* @_ssdm_op_SpecRegionEnd(i8* getelementptr inbounds ([17 x i8]* @.str5, i64 0, i64 0), i32 %rbegin2) nounwind, !dbg !456 ; [#uses=0 type=i32] [debug line = 379:3]
  %bit_cnt.9 = add i4 %bit_cnt.4, 1, !dbg !457    ; [#uses=1 type=i4] [debug line = 367:34]
  call void @llvm.dbg.value(metadata !{i4 %bit_cnt.9}, i64 0, metadata !191), !dbg !457 ; [debug line = 367:34] [debug variable = bit_cnt]
  br label %.preheader34, !dbg !457               ; [debug line = 367:34]

; <label>:28                                      ; preds = %.preheader34
  %de.2.lcssa = phi i7 [ %de.2, %.preheader34 ]   ; [#uses=3 type=i7]
  %dev_addr_in.load.2 = load volatile i7* @dev_addr_in, align 1, !dbg !458 ; [#uses=1 type=i7] [debug line = 382:3]
  %not.2 = icmp ne i7 %de.2.lcssa, %dev_addr_in.load.2, !dbg !458 ; [#uses=2 type=i1] [debug line = 382:3]
  br label %._crit_edge97, !dbg !459              ; [debug line = 388:3]

._crit_edge97:                                    ; preds = %._crit_edge97, %28
  %__Val2__.22 = load volatile i2* @i2c_in, align 1, !dbg !460 ; [#uses=3 type=i2] [debug line = 76:2@389:4]
  store i2 %__Val2__.22, i2* @i2c_val, align 1, !dbg !460 ; [debug line = 76:2@389:4]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.22}, i64 0, metadata !463), !dbg !465 ; [debug line = 390:51] [debug variable = __Val2__]
  %__Result__.13 = call i2 @llvm.part.select.i2(i2 %__Val2__.22, i32 0, i32 0), !dbg !466 ; [#uses=1 type=i2] [debug line = 390:85]
  call void @llvm.dbg.value(metadata !{i2 %__Result__.13}, i64 0, metadata !467), !dbg !466 ; [debug line = 390:85] [debug variable = __Result__]
  %tmp.56 = icmp eq i2 %__Result__.13, 0, !dbg !466 ; [#uses=1 type=i1] [debug line = 390:85]
  br i1 %tmp.56, label %._crit_edge97, label %29, !dbg !468 ; [debug line = 390:174]

; <label>:29                                      ; preds = %._crit_edge97
  %.lcssa8 = phi i2 [ %__Val2__.22, %._crit_edge97 ] ; [#uses=1 type=i2]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.22}, i64 0, metadata !469), !dbg !471 ; [debug line = 392:46] [debug variable = __Val2__]
  %__Result__72 = call i2 @llvm.part.select.i2(i2 %.lcssa8, i32 1, i32 1), !dbg !472 ; [#uses=1 type=i2] [debug line = 392:80]
  call void @llvm.dbg.value(metadata !{i2 %__Result__72}, i64 0, metadata !473), !dbg !472 ; [debug line = 392:80] [debug variable = __Result__]
  %tmp.60 = icmp eq i2 %__Result__72, 0, !dbg !472 ; [#uses=2 type=i1] [debug line = 392:80]
  %.ignore.2 = or i1 %tmp.60, %not.2, !dbg !474   ; [#uses=4 type=i1] [debug line = 392:169]
  br label %._crit_edge98, !dbg !475              ; [debug line = 395:3]

._crit_edge98:                                    ; preds = %._crit_edge98, %29
  %__Val2__.24 = load volatile i2* @i2c_in, align 1, !dbg !476 ; [#uses=2 type=i2] [debug line = 76:2@396:4]
  store i2 %__Val2__.24, i2* @i2c_val, align 1, !dbg !476 ; [debug line = 76:2@396:4]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.24}, i64 0, metadata !479), !dbg !481 ; [debug line = 397:51] [debug variable = __Val2__]
  %__Result__74 = call i2 @llvm.part.select.i2(i2 %__Val2__.24, i32 0, i32 0), !dbg !482 ; [#uses=1 type=i2] [debug line = 397:85]
  call void @llvm.dbg.value(metadata !{i2 %__Result__74}, i64 0, metadata !483), !dbg !482 ; [debug line = 397:85] [debug variable = __Result__]
  %tmp.65 = icmp eq i2 %__Result__74, 0, !dbg !482 ; [#uses=1 type=i1] [debug line = 397:85]
  br i1 %tmp.65, label %30, label %._crit_edge98, !dbg !484 ; [debug line = 397:174]

; <label>:30                                      ; preds = %._crit_edge98
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !485 ; [debug line = 400:3]
  store volatile i1 %.ignore.2, i1* @i2c_sda_out, align 1, !dbg !486 ; [debug line = 401:3]
  %not.ignore.3.demorgan = or i1 %tmp.60, %not.2, !dbg !487 ; [#uses=1 type=i1] [debug line = 402:3]
  %not.ignore.3 = xor i1 %not.ignore.3.demorgan, true, !dbg !487 ; [#uses=2 type=i1] [debug line = 402:3]
  store volatile i1 %not.ignore.3, i1* @i2c_sda_oe, align 1, !dbg !487 ; [debug line = 402:3]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !488 ; [debug line = 403:3]
  %reg_data.7 = call fastcc zeroext i8 @read_mem(i8 zeroext %re.1.lcssa), !dbg !489 ; [#uses=1 type=i8] [debug line = 404:14]
  call void @llvm.dbg.value(metadata !{i8 %reg_data.7}, i64 0, metadata !315), !dbg !489 ; [debug line = 404:14] [debug variable = reg_data]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !490 ; [debug line = 405:3]
  %auto_inc_regad_in.load.2 = load volatile i1* @auto_inc_regad_in, align 1, !dbg !491 ; [#uses=1 type=i1] [debug line = 406:3]
  br i1 %auto_inc_regad_in.load.2, label %31, label %._crit_edge99, !dbg !491 ; [debug line = 406:3]

; <label>:31                                      ; preds = %30
  %reg_addr.2 = add i8 %re.1.lcssa, 1, !dbg !492  ; [#uses=1 type=i8] [debug line = 407:4]
  call void @llvm.dbg.value(metadata !{i8 %reg_addr.2}, i64 0, metadata !263), !dbg !492 ; [debug line = 407:4] [debug variable = reg_addr]
  %re.1. = select i1 %.ignore.2, i8 %re.1.lcssa, i8 %reg_addr.2, !dbg !491 ; [#uses=1 type=i8] [debug line = 406:3]
  br label %._crit_edge99, !dbg !491              ; [debug line = 406:3]

._crit_edge99:                                    ; preds = %31, %30
  %re.6 = phi i8 [ %re.1., %31 ], [ %re.1.lcssa, %30 ] ; [#uses=1 type=i8]
  br label %._crit_edge100, !dbg !493             ; [debug line = 410:3]

._crit_edge100:                                   ; preds = %._crit_edge100, %._crit_edge99
  %__Val2__.26 = load volatile i2* @i2c_in, align 1, !dbg !494 ; [#uses=2 type=i2] [debug line = 76:2@411:4]
  store i2 %__Val2__.26, i2* @i2c_val, align 1, !dbg !494 ; [debug line = 76:2@411:4]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.26}, i64 0, metadata !497), !dbg !499 ; [debug line = 412:51] [debug variable = __Val2__]
  %__Result__76 = call i2 @llvm.part.select.i2(i2 %__Val2__.26, i32 0, i32 0), !dbg !500 ; [#uses=1 type=i2] [debug line = 412:85]
  call void @llvm.dbg.value(metadata !{i2 %__Result__76}, i64 0, metadata !501), !dbg !500 ; [debug line = 412:85] [debug variable = __Result__]
  %tmp.70 = icmp eq i2 %__Result__76, 0, !dbg !500 ; [#uses=1 type=i1] [debug line = 412:85]
  br i1 %tmp.70, label %._crit_edge100, label %.preheader33.preheader, !dbg !502 ; [debug line = 412:174]

.preheader33.preheader:                           ; preds = %._crit_edge100
  br label %.preheader33, !dbg !503               ; [debug line = 76:2@415:4]

.preheader33:                                     ; preds = %.preheader33, %.preheader33.preheader
  %__Val2__.27 = load volatile i2* @i2c_in, align 1, !dbg !503 ; [#uses=2 type=i2] [debug line = 76:2@415:4]
  store i2 %__Val2__.27, i2* @i2c_val, align 1, !dbg !503 ; [debug line = 76:2@415:4]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.27}, i64 0, metadata !506), !dbg !508 ; [debug line = 416:51] [debug variable = __Val2__]
  %__Result__78 = call i2 @llvm.part.select.i2(i2 %__Val2__.27, i32 0, i32 0), !dbg !509 ; [#uses=1 type=i2] [debug line = 416:85]
  call void @llvm.dbg.value(metadata !{i2 %__Result__78}, i64 0, metadata !510), !dbg !509 ; [debug line = 416:85] [debug variable = __Result__]
  %tmp.71 = icmp eq i2 %__Result__78, 0, !dbg !509 ; [#uses=1 type=i1] [debug line = 416:85]
  br i1 %tmp.71, label %.preheader31.preheader, label %.preheader33, !dbg !511 ; [debug line = 416:174]

.preheader31.preheader:                           ; preds = %.preheader33
  br label %.preheader31, !dbg !512               ; [debug line = 421:14]

.preheader31:                                     ; preds = %41, %.preheader31.preheader
  %terminate_read = phi i1 [ %terminate_read.1, %41 ], [ false, %.preheader31.preheader ] ; [#uses=1 type=i1]
  %__Val2__.28 = phi i8 [ %reg_data.8, %41 ], [ %reg_data.7, %.preheader31.preheader ] ; [#uses=2 type=i8]
  %re.7 = phi i8 [ %re.8, %41 ], [ %re.6, %.preheader31.preheader ] ; [#uses=6 type=i8]
  %rbegin3 = call i32 (...)* @_ssdm_op_SpecRegionBegin(i8* getelementptr inbounds ([12 x i8]* @.str6, i64 0, i64 0)) nounwind, !dbg !512 ; [#uses=1 type=i32] [debug line = 421:14]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !514 ; [debug line = 422:4]
  call void @llvm.dbg.value(metadata !{i8 %__Val2__.28}, i64 0, metadata !515), !dbg !517 ; [debug line = 423:59] [debug variable = __Val2__]
  %__Result__.14 = call i8 @llvm.part.select.i8(i8 %__Val2__.28, i32 7, i32 7), !dbg !518 ; [#uses=1 type=i8] [debug line = 423:94]
  call void @llvm.dbg.value(metadata !{i8 %__Result__.14}, i64 0, metadata !519), !dbg !518 ; [debug line = 423:94] [debug variable = __Result__]
  %tmp.72 = icmp ne i8 %__Result__.14, 0, !dbg !518 ; [#uses=1 type=i1] [debug line = 423:94]
  store volatile i1 %tmp.72, i1* @i2c_sda_out, align 1, !dbg !520 ; [debug line = 423:183]
  store volatile i1 %not.ignore.3, i1* @i2c_sda_oe, align 1, !dbg !521 ; [debug line = 424:4]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !522 ; [debug line = 425:4]
  %brmerge2 = or i1 %.ignore.2, %terminate_read, !dbg !523 ; [#uses=1 type=i1] [debug line = 431:6]
  br label %32, !dbg !527                         ; [debug line = 428:9]

; <label>:32                                      ; preds = %._crit_edge102, %.preheader31
  %bit_cnt.5 = phi i4 [ 0, %.preheader31 ], [ %bit_cnt.10, %._crit_edge102 ] ; [#uses=3 type=i4]
  %reg_data.4 = phi i8 [ %__Val2__.28, %.preheader31 ], [ %reg_data.10, %._crit_edge102 ] ; [#uses=2 type=i8]
  %tmp.73 = icmp sgt i4 %bit_cnt.5, -1, !dbg !527 ; [#uses=1 type=i1] [debug line = 428:9]
  br i1 %tmp.73, label %.preheader.preheader, label %38, !dbg !527 ; [debug line = 428:9]

.preheader.preheader:                             ; preds = %32
  br label %.preheader, !dbg !528                 ; [debug line = 76:2@430:6]

.preheader:                                       ; preds = %33, %.preheader.preheader
  %__Val2__.29 = load volatile i2* @i2c_in, align 1, !dbg !528 ; [#uses=2 type=i2] [debug line = 76:2@430:6]
  store i2 %__Val2__.29, i2* @i2c_val, align 1, !dbg !528 ; [debug line = 76:2@430:6]
  br i1 %brmerge2, label %.backedge.loopexit128, label %33, !dbg !523 ; [debug line = 431:6]

; <label>:33                                      ; preds = %.preheader
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.29}, i64 0, metadata !530), !dbg !532 ; [debug line = 433:53] [debug variable = __Val2__]
  %__Result__82 = call i2 @llvm.part.select.i2(i2 %__Val2__.29, i32 0, i32 0), !dbg !533 ; [#uses=1 type=i2] [debug line = 433:87]
  call void @llvm.dbg.value(metadata !{i2 %__Result__82}, i64 0, metadata !534), !dbg !533 ; [debug line = 433:87] [debug variable = __Result__]
  %tmp.76 = icmp eq i2 %__Result__82, 0, !dbg !533 ; [#uses=1 type=i1] [debug line = 433:87]
  br i1 %tmp.76, label %.preheader, label %34, !dbg !535 ; [debug line = 433:176]

; <label>:34                                      ; preds = %33
  %tmp.78 = shl i8 %reg_data.4, 1, !dbg !536      ; [#uses=1 type=i8] [debug line = 435:5]
  %reg_data.10 = or i8 %tmp.78, 1, !dbg !536      ; [#uses=3 type=i8] [debug line = 435:5]
  call void @llvm.dbg.value(metadata !{i8 %reg_data.10}, i64 0, metadata !315), !dbg !536 ; [debug line = 435:5] [debug variable = reg_data]
  br label %._crit_edge101, !dbg !537             ; [debug line = 437:5]

._crit_edge101:                                   ; preds = %35, %34
  %__Val2__.31 = load volatile i2* @i2c_in, align 1, !dbg !538 ; [#uses=3 type=i2] [debug line = 76:2@438:6]
  store i2 %__Val2__.31, i2* @i2c_val, align 1, !dbg !538 ; [debug line = 76:2@438:6]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.31}, i64 0, metadata !541), !dbg !543 ; [debug line = 439:49] [debug variable = __Val2__]
  %__Result__84 = call i2 @llvm.part.select.i2(i2 %__Val2__.31, i32 1, i32 1), !dbg !544 ; [#uses=1 type=i2] [debug line = 439:83]
  call void @llvm.dbg.value(metadata !{i2 %__Result__84}, i64 0, metadata !545), !dbg !544 ; [debug line = 439:83] [debug variable = __Result__]
  %tmp.81 = icmp eq i2 %__Result__84, 0, !dbg !544 ; [#uses=1 type=i1] [debug line = 439:83]
  %brmerge3 = or i1 %tmp.81, %.lcssa5, !dbg !546  ; [#uses=1 type=i1] [debug line = 439:172]
  br i1 %brmerge3, label %35, label %.backedge.loopexit, !dbg !546 ; [debug line = 439:172]

; <label>:35                                      ; preds = %._crit_edge101
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.31}, i64 0, metadata !547), !dbg !549 ; [debug line = 441:53] [debug variable = __Val2__]
  %__Result__86 = call i2 @llvm.part.select.i2(i2 %__Val2__.31, i32 0, i32 0), !dbg !550 ; [#uses=1 type=i2] [debug line = 441:87]
  call void @llvm.dbg.value(metadata !{i2 %__Result__86}, i64 0, metadata !551), !dbg !550 ; [debug line = 441:87] [debug variable = __Result__]
  %tmp.83 = icmp eq i2 %__Result__86, 0, !dbg !550 ; [#uses=1 type=i1] [debug line = 441:87]
  br i1 %tmp.83, label %36, label %._crit_edge101, !dbg !552 ; [debug line = 441:176]

; <label>:36                                      ; preds = %35
  %tmp.84 = icmp ult i4 %bit_cnt.5, 7, !dbg !553  ; [#uses=1 type=i1] [debug line = 443:5]
  br i1 %tmp.84, label %37, label %._crit_edge102, !dbg !553 ; [debug line = 443:5]

; <label>:37                                      ; preds = %36
  %rbegin4 = call i32 (...)* @_ssdm_op_SpecRegionBegin(i8* getelementptr inbounds ([12 x i8]* @.str7, i64 0, i64 0)) nounwind, !dbg !554 ; [#uses=1 type=i32] [debug line = 443:24]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !556 ; [debug line = 444:6]
  call void @llvm.dbg.value(metadata !{i8 %reg_data.10}, i64 0, metadata !557), !dbg !559 ; [debug line = 445:61] [debug variable = __Val2__]
  %__Result__88 = call i8 @llvm.part.select.i8(i8 %reg_data.10, i32 7, i32 7), !dbg !560 ; [#uses=1 type=i8] [debug line = 445:96]
  call void @llvm.dbg.value(metadata !{i8 %__Result__88}, i64 0, metadata !561), !dbg !560 ; [debug line = 445:96] [debug variable = __Result__]
  %tmp.85 = icmp ne i8 %__Result__88, 0, !dbg !560 ; [#uses=1 type=i1] [debug line = 445:96]
  store volatile i1 %tmp.85, i1* @i2c_sda_out, align 1, !dbg !562 ; [debug line = 445:185]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !563 ; [debug line = 446:6]
  %rend120 = call i32 (...)* @_ssdm_op_SpecRegionEnd(i8* getelementptr inbounds ([12 x i8]* @.str7, i64 0, i64 0), i32 %rbegin4) nounwind, !dbg !564 ; [#uses=0 type=i32] [debug line = 447:5]
  br label %._crit_edge102, !dbg !564             ; [debug line = 447:5]

._crit_edge102:                                   ; preds = %37, %36
  %bit_cnt.10 = add i4 %bit_cnt.5, 1, !dbg !565   ; [#uses=1 type=i4] [debug line = 428:35]
  call void @llvm.dbg.value(metadata !{i4 %bit_cnt.10}, i64 0, metadata !191), !dbg !565 ; [debug line = 428:35] [debug variable = bit_cnt]
  br label %32, !dbg !565                         ; [debug line = 428:35]

; <label>:38                                      ; preds = %32
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !566 ; [debug line = 450:4]
  store volatile i1 false, i1* @i2c_sda_oe, align 1, !dbg !567 ; [debug line = 451:4]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !568 ; [debug line = 452:4]
  %reg_data.8 = call fastcc zeroext i8 @read_mem(i8 zeroext %re.7), !dbg !569 ; [#uses=1 type=i8] [debug line = 453:15]
  call void @llvm.dbg.value(metadata !{i8 %reg_data.8}, i64 0, metadata !315), !dbg !569 ; [debug line = 453:15] [debug variable = reg_data]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !570 ; [debug line = 454:4]
  %auto_inc_regad_in.load.3 = load volatile i1* @auto_inc_regad_in, align 1, !dbg !571 ; [#uses=1 type=i1] [debug line = 455:4]
  br i1 %auto_inc_regad_in.load.3, label %39, label %._crit_edge103, !dbg !571 ; [debug line = 455:4]

; <label>:39                                      ; preds = %38
  %reg_addr.3 = add i8 %re.7, 1, !dbg !572        ; [#uses=1 type=i8] [debug line = 456:5]
  call void @llvm.dbg.value(metadata !{i8 %reg_addr.3}, i64 0, metadata !263), !dbg !572 ; [debug line = 456:5] [debug variable = reg_addr]
  %re.7. = select i1 %.ignore.2, i8 %re.7, i8 %reg_addr.3, !dbg !571 ; [#uses=1 type=i8] [debug line = 455:4]
  br label %._crit_edge103, !dbg !571             ; [debug line = 455:4]

._crit_edge103:                                   ; preds = %39, %38
  %re.8 = phi i8 [ %re.7., %39 ], [ %re.7, %38 ]  ; [#uses=1 type=i8]
  br label %._crit_edge104, !dbg !573             ; [debug line = 459:4]

._crit_edge104:                                   ; preds = %._crit_edge104, %._crit_edge103
  %__Val2__.30 = load volatile i2* @i2c_in, align 1, !dbg !574 ; [#uses=3 type=i2] [debug line = 76:2@460:5]
  store i2 %__Val2__.30, i2* @i2c_val, align 1, !dbg !574 ; [debug line = 76:2@460:5]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.30}, i64 0, metadata !577), !dbg !579 ; [debug line = 461:52] [debug variable = __Val2__]
  %__Result__.15 = call i2 @llvm.part.select.i2(i2 %__Val2__.30, i32 0, i32 0), !dbg !580 ; [#uses=1 type=i2] [debug line = 461:86]
  call void @llvm.dbg.value(metadata !{i2 %__Result__.15}, i64 0, metadata !581), !dbg !580 ; [debug line = 461:86] [debug variable = __Result__]
  %tmp.77 = icmp eq i2 %__Result__.15, 0, !dbg !580 ; [#uses=1 type=i1] [debug line = 461:86]
  br i1 %tmp.77, label %._crit_edge104, label %40, !dbg !582 ; [debug line = 461:175]

; <label>:40                                      ; preds = %._crit_edge104
  %.lcssa = phi i2 [ %__Val2__.30, %._crit_edge104 ] ; [#uses=1 type=i2]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.30}, i64 0, metadata !583), !dbg !585 ; [debug line = 463:60] [debug variable = __Val2__]
  %__Result__92 = call i2 @llvm.part.select.i2(i2 %.lcssa, i32 1, i32 1), !dbg !586 ; [#uses=1 type=i2] [debug line = 463:94]
  call void @llvm.dbg.value(metadata !{i2 %__Result__92}, i64 0, metadata !587), !dbg !586 ; [debug line = 463:94] [debug variable = __Result__]
  %terminate_read.1 = icmp ne i2 %__Result__92, 0, !dbg !586 ; [#uses=1 type=i1] [debug line = 463:94]
  call void @llvm.dbg.value(metadata !{i1 %terminate_read.1}, i64 0, metadata !588), !dbg !589 ; [debug line = 463:183] [debug variable = terminate_read]
  br label %._crit_edge105, !dbg !590             ; [debug line = 465:4]

._crit_edge105:                                   ; preds = %._crit_edge105, %40
  %__Val2__.32 = load volatile i2* @i2c_in, align 1, !dbg !591 ; [#uses=2 type=i2] [debug line = 76:2@466:5]
  store i2 %__Val2__.32, i2* @i2c_val, align 1, !dbg !591 ; [debug line = 76:2@466:5]
  call void @llvm.dbg.value(metadata !{i2 %__Val2__.32}, i64 0, metadata !594), !dbg !596 ; [debug line = 467:52] [debug variable = __Val2__]
  %__Result__94 = call i2 @llvm.part.select.i2(i2 %__Val2__.32, i32 0, i32 0), !dbg !597 ; [#uses=1 type=i2] [debug line = 467:86]
  call void @llvm.dbg.value(metadata !{i2 %__Result__94}, i64 0, metadata !598), !dbg !597 ; [debug line = 467:86] [debug variable = __Result__]
  %tmp.82 = icmp eq i2 %__Result__94, 0, !dbg !597 ; [#uses=1 type=i1] [debug line = 467:86]
  br i1 %tmp.82, label %41, label %._crit_edge105, !dbg !599 ; [debug line = 467:175]

; <label>:41                                      ; preds = %._crit_edge105
  %rend125 = call i32 (...)* @_ssdm_op_SpecRegionEnd(i8* getelementptr inbounds ([12 x i8]* @.str6, i64 0, i64 0), i32 %rbegin3) nounwind, !dbg !600 ; [#uses=0 type=i32] [debug line = 468:3]
  br label %.preheader31, !dbg !600               ; [debug line = 468:3]
}

; [#uses=29]
declare void @_ssdm_op_Wait(...) nounwind

; [#uses=1]
declare void @_ssdm_op_SpecTopModule(...) nounwind

; [#uses=5]
declare i32 @_ssdm_op_SpecRegionEnd(...)

; [#uses=6]
declare i32 @_ssdm_op_SpecRegionBegin(...)

; [#uses=1]
declare void @_ssdm_op_SpecLoopName(...) nounwind

; [#uses=1]
declare i32 @_ssdm_op_SpecLoopBegin(...)

; [#uses=12]
declare void @_ssdm_op_SpecInterface(...) nounwind

; [#uses=2]
declare void @_ssdm_InlineSelf(...) nounwind

!llvm.dbg.cu = !{!0}
!hls.encrypted.func = !{}

!0 = metadata !{i32 786449, i32 0, i32 1, metadata !"D:/21_streamer_car5_artix7/fpga_arty/i2c_slave_core/solution1/.autopilot/db/i2c_slave_core.pragma.2.c", metadata !"D:\5C21_streamer_car5_artix7\5Cfpga_arty", metadata !"clang version 3.1 ", i1 true, i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !20} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !5, metadata !11, metadata !16, metadata !19}
!5 = metadata !{i32 786478, i32 0, metadata !6, metadata !"read_i2c", metadata !"read_i2c", metadata !"", metadata !6, i32 73, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, metadata !9, i32 74} ; [ DW_TAG_subprogram ]
!6 = metadata !{i32 786473, metadata !"i2c_slave_core.c", metadata !"D:\5C21_streamer_car5_artix7\5Cfpga_arty", null} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!8 = metadata !{null}
!9 = metadata !{metadata !10}
!10 = metadata !{i32 786468}                      ; [ DW_TAG_base_type ]
!11 = metadata !{i32 786478, i32 0, metadata !6, metadata !"write_mem", metadata !"write_mem", metadata !"", metadata !6, i32 80, metadata !12, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (i8, i8)* @write_mem, null, null, metadata !9, i32 81} ; [ DW_TAG_subprogram ]
!12 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !13, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!13 = metadata !{null, metadata !14, metadata !14}
!14 = metadata !{i32 786454, null, metadata !"uint8", metadata !6, i32 10, i64 0, i64 0, i64 0, i32 0, metadata !15} ; [ DW_TAG_typedef ]
!15 = metadata !{i32 786468, null, metadata !"uint8", null, i32 0, i64 8, i64 8, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!16 = metadata !{i32 786478, i32 0, metadata !6, metadata !"read_mem", metadata !"read_mem", metadata !"", metadata !6, i32 103, metadata !17, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i8 (i8)* @read_mem, null, null, metadata !9, i32 104} ; [ DW_TAG_subprogram ]
!17 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !18, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!18 = metadata !{metadata !14, metadata !14}
!19 = metadata !{i32 786478, i32 0, metadata !6, metadata !"i2c_slave_core", metadata !"i2c_slave_core", metadata !"", metadata !6, i32 131, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @i2c_slave_core, null, null, metadata !9, i32 132} ; [ DW_TAG_subprogram ]
!20 = metadata !{metadata !21}
!21 = metadata !{metadata !22, metadata !26, metadata !27, metadata !29, metadata !30, metadata !31, metadata !32, metadata !36, metadata !37, metadata !38, metadata !39, metadata !43, metadata !44}
!22 = metadata !{i32 786484, i32 0, null, metadata !"i2c_sda_out", metadata !"i2c_sda_out", metadata !"", metadata !6, i32 48, metadata !23, i32 0, i32 1, i1* @i2c_sda_out} ; [ DW_TAG_variable ]
!23 = metadata !{i32 786485, null, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !24} ; [ DW_TAG_volatile_type ]
!24 = metadata !{i32 786454, null, metadata !"uint1", metadata !6, i32 3, i64 0, i64 0, i64 0, i32 0, metadata !25} ; [ DW_TAG_typedef ]
!25 = metadata !{i32 786468, null, metadata !"uint1", null, i32 0, i64 1, i64 1, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!26 = metadata !{i32 786484, i32 0, null, metadata !"i2c_sda_oe", metadata !"i2c_sda_oe", metadata !"", metadata !6, i32 49, metadata !23, i32 0, i32 1, i1* @i2c_sda_oe} ; [ DW_TAG_variable ]
!27 = metadata !{i32 786484, i32 0, null, metadata !"mem_addr", metadata !"mem_addr", metadata !"", metadata !6, i32 52, metadata !28, i32 0, i32 1, i8* @mem_addr} ; [ DW_TAG_variable ]
!28 = metadata !{i32 786485, null, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !14} ; [ DW_TAG_volatile_type ]
!29 = metadata !{i32 786484, i32 0, null, metadata !"mem_dout", metadata !"mem_dout", metadata !"", metadata !6, i32 54, metadata !28, i32 0, i32 1, i8* @mem_dout} ; [ DW_TAG_variable ]
!30 = metadata !{i32 786484, i32 0, null, metadata !"mem_wreq", metadata !"mem_wreq", metadata !"", metadata !6, i32 55, metadata !23, i32 0, i32 1, i1* @mem_wreq} ; [ DW_TAG_variable ]
!31 = metadata !{i32 786484, i32 0, null, metadata !"mem_rreq", metadata !"mem_rreq", metadata !"", metadata !6, i32 57, metadata !23, i32 0, i32 1, i1* @mem_rreq} ; [ DW_TAG_variable ]
!32 = metadata !{i32 786484, i32 0, null, metadata !"i2c_in", metadata !"i2c_in", metadata !"", metadata !6, i32 47, metadata !33, i32 0, i32 1, i2* @i2c_in} ; [ DW_TAG_variable ]
!33 = metadata !{i32 786485, null, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !34} ; [ DW_TAG_volatile_type ]
!34 = metadata !{i32 786454, null, metadata !"uint2", metadata !6, i32 4, i64 0, i64 0, i64 0, i32 0, metadata !35} ; [ DW_TAG_typedef ]
!35 = metadata !{i32 786468, null, metadata !"uint2", null, i32 0, i64 2, i64 2, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!36 = metadata !{i32 786484, i32 0, null, metadata !"mem_din", metadata !"mem_din", metadata !"", metadata !6, i32 53, metadata !28, i32 0, i32 1, i8* @mem_din} ; [ DW_TAG_variable ]
!37 = metadata !{i32 786484, i32 0, null, metadata !"mem_wack", metadata !"mem_wack", metadata !"", metadata !6, i32 56, metadata !23, i32 0, i32 1, i1* @mem_wack} ; [ DW_TAG_variable ]
!38 = metadata !{i32 786484, i32 0, null, metadata !"mem_rack", metadata !"mem_rack", metadata !"", metadata !6, i32 58, metadata !23, i32 0, i32 1, i1* @mem_rack} ; [ DW_TAG_variable ]
!39 = metadata !{i32 786484, i32 0, null, metadata !"dev_addr_in", metadata !"dev_addr_in", metadata !"", metadata !6, i32 60, metadata !40, i32 0, i32 1, i7* @dev_addr_in} ; [ DW_TAG_variable ]
!40 = metadata !{i32 786485, null, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !41} ; [ DW_TAG_volatile_type ]
!41 = metadata !{i32 786454, null, metadata !"uint7", metadata !6, i32 9, i64 0, i64 0, i64 0, i32 0, metadata !42} ; [ DW_TAG_typedef ]
!42 = metadata !{i32 786468, null, metadata !"uint7", null, i32 0, i64 7, i64 8, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!43 = metadata !{i32 786484, i32 0, null, metadata !"auto_inc_regad_in", metadata !"auto_inc_regad_in", metadata !"", metadata !6, i32 61, metadata !23, i32 0, i32 1, i1* @auto_inc_regad_in} ; [ DW_TAG_variable ]
!44 = metadata !{i32 786484, i32 0, null, metadata !"i2c_val", metadata !"i2c_val", metadata !"", metadata !6, i32 67, metadata !34, i32 0, i32 1, i2* @i2c_val} ; [ DW_TAG_variable ]
!45 = metadata !{i32 786689, metadata !11, metadata !"addr", metadata !6, i32 16777296, metadata !14, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!46 = metadata !{i32 80, i32 22, metadata !11, null}
!47 = metadata !{i32 786689, metadata !11, metadata !"data", metadata !6, i32 33554512, metadata !14, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!48 = metadata !{i32 80, i32 34, metadata !11, null}
!49 = metadata !{i32 82, i32 1, metadata !50, null}
!50 = metadata !{i32 786443, metadata !11, i32 81, i32 1, metadata !6, i32 1} ; [ DW_TAG_lexical_block ]
!51 = metadata !{i32 83, i32 2, metadata !50, null}
!52 = metadata !{i32 84, i32 2, metadata !50, null}
!53 = metadata !{i32 85, i32 2, metadata !50, null}
!54 = metadata !{i32 86, i32 2, metadata !50, null}
!55 = metadata !{i32 87, i32 2, metadata !50, null}
!56 = metadata !{i32 89, i32 2, metadata !50, null}
!57 = metadata !{i32 90, i32 3, metadata !58, null}
!58 = metadata !{i32 786443, metadata !50, i32 89, i32 5, metadata !6, i32 2} ; [ DW_TAG_lexical_block ]
!59 = metadata !{i32 91, i32 3, metadata !58, null}
!60 = metadata !{i32 92, i32 3, metadata !58, null}
!61 = metadata !{i32 93, i32 2, metadata !58, null}
!62 = metadata !{i32 94, i32 2, metadata !50, null}
!63 = metadata !{i32 96, i32 2, metadata !50, null}
!64 = metadata !{i32 97, i32 2, metadata !50, null}
!65 = metadata !{i32 98, i32 2, metadata !50, null}
!66 = metadata !{i32 99, i32 2, metadata !50, null}
!67 = metadata !{i32 100, i32 1, metadata !50, null}
!68 = metadata !{i32 786689, metadata !16, metadata !"addr", metadata !6, i32 16777319, metadata !14, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!69 = metadata !{i32 103, i32 22, metadata !16, null}
!70 = metadata !{i32 105, i32 1, metadata !71, null}
!71 = metadata !{i32 786443, metadata !16, i32 104, i32 1, metadata !6, i32 3} ; [ DW_TAG_lexical_block ]
!72 = metadata !{i32 108, i32 2, metadata !71, null}
!73 = metadata !{i32 109, i32 2, metadata !71, null}
!74 = metadata !{i32 110, i32 2, metadata !71, null}
!75 = metadata !{i32 111, i32 2, metadata !71, null}
!76 = metadata !{i32 113, i32 2, metadata !71, null}
!77 = metadata !{i32 114, i32 3, metadata !78, null}
!78 = metadata !{i32 786443, metadata !71, i32 113, i32 5, metadata !6, i32 4} ; [ DW_TAG_lexical_block ]
!79 = metadata !{i32 115, i32 3, metadata !78, null}
!80 = metadata !{i32 116, i32 3, metadata !78, null}
!81 = metadata !{i32 786688, metadata !71, metadata !"dt", metadata !6, i32 106, metadata !14, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!82 = metadata !{i32 117, i32 2, metadata !78, null}
!83 = metadata !{i32 118, i32 2, metadata !71, null}
!84 = metadata !{i32 120, i32 2, metadata !71, null}
!85 = metadata !{i32 121, i32 2, metadata !71, null}
!86 = metadata !{i32 122, i32 2, metadata !71, null}
!87 = metadata !{i32 124, i32 2, metadata !71, null}
!88 = metadata !{i32 133, i32 1, metadata !89, null}
!89 = metadata !{i32 786443, metadata !19, i32 132, i32 1, metadata !6, i32 5} ; [ DW_TAG_lexical_block ]
!90 = metadata !{i32 134, i32 1, metadata !89, null}
!91 = metadata !{i32 135, i32 1, metadata !89, null}
!92 = metadata !{i32 136, i32 1, metadata !89, null}
!93 = metadata !{i32 138, i32 1, metadata !89, null}
!94 = metadata !{i32 139, i32 1, metadata !89, null}
!95 = metadata !{i32 141, i32 1, metadata !89, null}
!96 = metadata !{i32 142, i32 1, metadata !89, null}
!97 = metadata !{i32 143, i32 1, metadata !89, null}
!98 = metadata !{i32 144, i32 1, metadata !89, null}
!99 = metadata !{i32 145, i32 1, metadata !89, null}
!100 = metadata !{i32 146, i32 1, metadata !89, null}
!101 = metadata !{i32 147, i32 1, metadata !89, null}
!102 = metadata !{i32 158, i32 2, metadata !89, null}
!103 = metadata !{i32 159, i32 2, metadata !89, null}
!104 = metadata !{i32 160, i32 2, metadata !89, null}
!105 = metadata !{i32 161, i32 2, metadata !89, null}
!106 = metadata !{i32 162, i32 2, metadata !89, null}
!107 = metadata !{i32 163, i32 2, metadata !89, null}
!108 = metadata !{i32 164, i32 2, metadata !89, null}
!109 = metadata !{i32 165, i32 2, metadata !89, null}
!110 = metadata !{i32 168, i32 13, metadata !111, null}
!111 = metadata !{i32 786443, metadata !89, i32 168, i32 12, metadata !6, i32 6} ; [ DW_TAG_lexical_block ]
!112 = metadata !{i32 173, i32 3, metadata !111, null}
!113 = metadata !{i32 174, i32 3, metadata !111, null}
!114 = metadata !{i32 175, i32 3, metadata !111, null}
!115 = metadata !{i32 178, i32 3, metadata !111, null}
!116 = metadata !{i32 76, i32 2, metadata !117, metadata !118}
!117 = metadata !{i32 786443, metadata !5, i32 74, i32 1, metadata !6, i32 0} ; [ DW_TAG_lexical_block ]
!118 = metadata !{i32 179, i32 4, metadata !119, null}
!119 = metadata !{i32 786443, metadata !111, i32 178, i32 6, metadata !6, i32 7} ; [ DW_TAG_lexical_block ]
!120 = metadata !{i32 786688, metadata !121, metadata !"__Val2__", metadata !6, i32 180, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!121 = metadata !{i32 786443, metadata !111, i32 180, i32 13, metadata !6, i32 8} ; [ DW_TAG_lexical_block ]
!122 = metadata !{i32 180, i32 51, metadata !121, null}
!123 = metadata !{i32 180, i32 85, metadata !121, null}
!124 = metadata !{i32 786688, metadata !121, metadata !"__Result__", metadata !6, i32 180, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!125 = metadata !{i32 180, i32 174, metadata !121, null}
!126 = metadata !{i32 786688, metadata !127, metadata !"__Val2__", metadata !6, i32 180, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!127 = metadata !{i32 786443, metadata !111, i32 180, i32 186, metadata !6, i32 9} ; [ DW_TAG_lexical_block ]
!128 = metadata !{i32 180, i32 224, metadata !127, null}
!129 = metadata !{i32 180, i32 0, metadata !127, null}
!130 = metadata !{i32 786688, metadata !127, metadata !"__Result__", metadata !6, i32 180, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!131 = metadata !{i32 76, i32 2, metadata !117, metadata !132}
!132 = metadata !{i32 184, i32 4, metadata !133, null}
!133 = metadata !{i32 786443, metadata !111, i32 183, i32 6, metadata !6, i32 10} ; [ DW_TAG_lexical_block ]
!134 = metadata !{i32 786688, metadata !135, metadata !"__Val2__", metadata !6, i32 185, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!135 = metadata !{i32 786443, metadata !133, i32 185, i32 9, metadata !6, i32 11} ; [ DW_TAG_lexical_block ]
!136 = metadata !{i32 185, i32 47, metadata !135, null}
!137 = metadata !{i32 185, i32 81, metadata !135, null}
!138 = metadata !{i32 786688, metadata !135, metadata !"__Result__", metadata !6, i32 185, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!139 = metadata !{i32 185, i32 170, metadata !135, null}
!140 = metadata !{i32 786688, metadata !141, metadata !"__Val2__", metadata !6, i32 187, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!141 = metadata !{i32 786443, metadata !111, i32 187, i32 13, metadata !6, i32 12} ; [ DW_TAG_lexical_block ]
!142 = metadata !{i32 187, i32 51, metadata !141, null}
!143 = metadata !{i32 187, i32 85, metadata !141, null}
!144 = metadata !{i32 786688, metadata !141, metadata !"__Result__", metadata !6, i32 187, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!145 = metadata !{i32 187, i32 174, metadata !141, null}
!146 = metadata !{i32 76, i32 2, metadata !117, metadata !147}
!147 = metadata !{i32 191, i32 4, metadata !148, null}
!148 = metadata !{i32 786443, metadata !111, i32 190, i32 6, metadata !6, i32 13} ; [ DW_TAG_lexical_block ]
!149 = metadata !{i32 786688, metadata !150, metadata !"__Val2__", metadata !6, i32 192, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!150 = metadata !{i32 786443, metadata !148, i32 192, i32 9, metadata !6, i32 14} ; [ DW_TAG_lexical_block ]
!151 = metadata !{i32 192, i32 47, metadata !150, null}
!152 = metadata !{i32 192, i32 81, metadata !150, null}
!153 = metadata !{i32 786688, metadata !150, metadata !"__Result__", metadata !6, i32 192, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!154 = metadata !{i32 192, i32 170, metadata !150, null}
!155 = metadata !{i32 786688, metadata !156, metadata !"__Val2__", metadata !6, i32 194, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!156 = metadata !{i32 786443, metadata !111, i32 194, i32 13, metadata !6, i32 15} ; [ DW_TAG_lexical_block ]
!157 = metadata !{i32 194, i32 51, metadata !156, null}
!158 = metadata !{i32 194, i32 85, metadata !156, null}
!159 = metadata !{i32 786688, metadata !156, metadata !"__Result__", metadata !6, i32 194, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!160 = metadata !{i32 194, i32 174, metadata !156, null}
!161 = metadata !{i32 199, i32 8, metadata !162, null}
!162 = metadata !{i32 786443, metadata !111, i32 199, i32 3, metadata !6, i32 16} ; [ DW_TAG_lexical_block ]
!163 = metadata !{i32 76, i32 2, metadata !117, metadata !164}
!164 = metadata !{i32 202, i32 5, metadata !165, null}
!165 = metadata !{i32 786443, metadata !166, i32 201, i32 7, metadata !6, i32 18} ; [ DW_TAG_lexical_block ]
!166 = metadata !{i32 786443, metadata !162, i32 199, i32 45, metadata !6, i32 17} ; [ DW_TAG_lexical_block ]
!167 = metadata !{i32 786688, metadata !168, metadata !"__Val2__", metadata !6, i32 203, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!168 = metadata !{i32 786443, metadata !166, i32 203, i32 14, metadata !6, i32 19} ; [ DW_TAG_lexical_block ]
!169 = metadata !{i32 203, i32 52, metadata !168, null}
!170 = metadata !{i32 203, i32 86, metadata !168, null}
!171 = metadata !{i32 786688, metadata !168, metadata !"__Result__", metadata !6, i32 203, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!172 = metadata !{i32 203, i32 175, metadata !168, null}
!173 = metadata !{i32 205, i32 195, metadata !174, null}
!174 = metadata !{i32 786443, metadata !166, i32 205, i32 34, metadata !6, i32 20} ; [ DW_TAG_lexical_block ]
!175 = metadata !{i32 786688, metadata !174, metadata !"__Val2__", metadata !6, i32 205, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!176 = metadata !{i32 205, i32 72, metadata !174, null}
!177 = metadata !{i32 205, i32 106, metadata !174, null}
!178 = metadata !{i32 786688, metadata !174, metadata !"__Result__", metadata !6, i32 205, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!179 = metadata !{i32 786688, metadata !89, metadata !"dev_addr", metadata !6, i32 150, metadata !41, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!180 = metadata !{i32 208, i32 4, metadata !166, null}
!181 = metadata !{i32 76, i32 2, metadata !117, metadata !182}
!182 = metadata !{i32 209, i32 5, metadata !183, null}
!183 = metadata !{i32 786443, metadata !166, i32 208, i32 7, metadata !6, i32 21} ; [ DW_TAG_lexical_block ]
!184 = metadata !{i32 786688, metadata !185, metadata !"__Val2__", metadata !6, i32 210, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!185 = metadata !{i32 786443, metadata !166, i32 210, i32 14, metadata !6, i32 22} ; [ DW_TAG_lexical_block ]
!186 = metadata !{i32 210, i32 52, metadata !185, null}
!187 = metadata !{i32 210, i32 86, metadata !185, null}
!188 = metadata !{i32 786688, metadata !185, metadata !"__Result__", metadata !6, i32 210, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!189 = metadata !{i32 210, i32 175, metadata !185, null}
!190 = metadata !{i32 199, i32 34, metadata !162, null}
!191 = metadata !{i32 786688, metadata !89, metadata !"bit_cnt", metadata !6, i32 153, metadata !192, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!192 = metadata !{i32 786454, null, metadata !"uint4", metadata !6, i32 6, i64 0, i64 0, i64 0, i32 0, metadata !193} ; [ DW_TAG_typedef ]
!193 = metadata !{i32 786468, null, metadata !"uint4", null, i32 0, i64 4, i64 4, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!194 = metadata !{i32 214, i32 3, metadata !111, null}
!195 = metadata !{i32 220, i32 3, metadata !111, null}
!196 = metadata !{i32 76, i32 2, metadata !117, metadata !197}
!197 = metadata !{i32 221, i32 4, metadata !198, null}
!198 = metadata !{i32 786443, metadata !111, i32 220, i32 6, metadata !6, i32 23} ; [ DW_TAG_lexical_block ]
!199 = metadata !{i32 786688, metadata !200, metadata !"__Val2__", metadata !6, i32 222, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!200 = metadata !{i32 786443, metadata !111, i32 222, i32 13, metadata !6, i32 24} ; [ DW_TAG_lexical_block ]
!201 = metadata !{i32 222, i32 51, metadata !200, null}
!202 = metadata !{i32 222, i32 85, metadata !200, null}
!203 = metadata !{i32 786688, metadata !200, metadata !"__Result__", metadata !6, i32 222, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!204 = metadata !{i32 222, i32 174, metadata !200, null}
!205 = metadata !{i32 786688, metadata !206, metadata !"__Val2__", metadata !6, i32 224, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!206 = metadata !{i32 786443, metadata !111, i32 224, i32 8, metadata !6, i32 25} ; [ DW_TAG_lexical_block ]
!207 = metadata !{i32 224, i32 46, metadata !206, null}
!208 = metadata !{i32 224, i32 80, metadata !206, null}
!209 = metadata !{i32 786688, metadata !206, metadata !"__Result__", metadata !6, i32 224, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!210 = metadata !{i32 224, i32 169, metadata !206, null}
!211 = metadata !{i32 227, i32 3, metadata !111, null}
!212 = metadata !{i32 76, i32 2, metadata !117, metadata !213}
!213 = metadata !{i32 228, i32 4, metadata !214, null}
!214 = metadata !{i32 786443, metadata !111, i32 227, i32 6, metadata !6, i32 26} ; [ DW_TAG_lexical_block ]
!215 = metadata !{i32 786688, metadata !216, metadata !"__Val2__", metadata !6, i32 229, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!216 = metadata !{i32 786443, metadata !111, i32 229, i32 13, metadata !6, i32 27} ; [ DW_TAG_lexical_block ]
!217 = metadata !{i32 229, i32 51, metadata !216, null}
!218 = metadata !{i32 229, i32 85, metadata !216, null}
!219 = metadata !{i32 786688, metadata !216, metadata !"__Result__", metadata !6, i32 229, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!220 = metadata !{i32 229, i32 174, metadata !216, null}
!221 = metadata !{i32 232, i32 3, metadata !111, null}
!222 = metadata !{i32 233, i32 3, metadata !111, null}
!223 = metadata !{i32 234, i32 3, metadata !111, null}
!224 = metadata !{i32 235, i32 3, metadata !111, null}
!225 = metadata !{i32 237, i32 3, metadata !111, null}
!226 = metadata !{i32 76, i32 2, metadata !117, metadata !227}
!227 = metadata !{i32 238, i32 4, metadata !228, null}
!228 = metadata !{i32 786443, metadata !111, i32 237, i32 6, metadata !6, i32 28} ; [ DW_TAG_lexical_block ]
!229 = metadata !{i32 786688, metadata !230, metadata !"__Val2__", metadata !6, i32 239, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!230 = metadata !{i32 786443, metadata !111, i32 239, i32 13, metadata !6, i32 29} ; [ DW_TAG_lexical_block ]
!231 = metadata !{i32 239, i32 51, metadata !230, null}
!232 = metadata !{i32 239, i32 85, metadata !230, null}
!233 = metadata !{i32 786688, metadata !230, metadata !"__Result__", metadata !6, i32 239, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!234 = metadata !{i32 239, i32 174, metadata !230, null}
!235 = metadata !{i32 76, i32 2, metadata !117, metadata !236}
!236 = metadata !{i32 242, i32 4, metadata !237, null}
!237 = metadata !{i32 786443, metadata !111, i32 241, i32 6, metadata !6, i32 30} ; [ DW_TAG_lexical_block ]
!238 = metadata !{i32 786688, metadata !239, metadata !"__Val2__", metadata !6, i32 243, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!239 = metadata !{i32 786443, metadata !111, i32 243, i32 13, metadata !6, i32 31} ; [ DW_TAG_lexical_block ]
!240 = metadata !{i32 243, i32 51, metadata !239, null}
!241 = metadata !{i32 243, i32 85, metadata !239, null}
!242 = metadata !{i32 786688, metadata !239, metadata !"__Result__", metadata !6, i32 243, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!243 = metadata !{i32 243, i32 174, metadata !239, null}
!244 = metadata !{i32 245, i32 3, metadata !111, null}
!245 = metadata !{i32 250, i32 8, metadata !246, null}
!246 = metadata !{i32 786443, metadata !111, i32 250, i32 3, metadata !6, i32 32} ; [ DW_TAG_lexical_block ]
!247 = metadata !{i32 76, i32 2, metadata !117, metadata !248}
!248 = metadata !{i32 253, i32 5, metadata !249, null}
!249 = metadata !{i32 786443, metadata !250, i32 252, i32 7, metadata !6, i32 34} ; [ DW_TAG_lexical_block ]
!250 = metadata !{i32 786443, metadata !246, i32 250, i32 45, metadata !6, i32 33} ; [ DW_TAG_lexical_block ]
!251 = metadata !{i32 786688, metadata !252, metadata !"__Val2__", metadata !6, i32 254, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!252 = metadata !{i32 786443, metadata !250, i32 254, i32 14, metadata !6, i32 35} ; [ DW_TAG_lexical_block ]
!253 = metadata !{i32 254, i32 52, metadata !252, null}
!254 = metadata !{i32 254, i32 86, metadata !252, null}
!255 = metadata !{i32 786688, metadata !252, metadata !"__Result__", metadata !6, i32 254, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!256 = metadata !{i32 254, i32 175, metadata !252, null}
!257 = metadata !{i32 256, i32 195, metadata !258, null}
!258 = metadata !{i32 786443, metadata !250, i32 256, i32 34, metadata !6, i32 36} ; [ DW_TAG_lexical_block ]
!259 = metadata !{i32 786688, metadata !258, metadata !"__Val2__", metadata !6, i32 256, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!260 = metadata !{i32 256, i32 72, metadata !258, null}
!261 = metadata !{i32 256, i32 106, metadata !258, null}
!262 = metadata !{i32 786688, metadata !258, metadata !"__Result__", metadata !6, i32 256, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!263 = metadata !{i32 786688, metadata !89, metadata !"reg_addr", metadata !6, i32 151, metadata !14, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!264 = metadata !{i32 259, i32 4, metadata !250, null}
!265 = metadata !{i32 76, i32 2, metadata !117, metadata !266}
!266 = metadata !{i32 260, i32 5, metadata !267, null}
!267 = metadata !{i32 786443, metadata !250, i32 259, i32 7, metadata !6, i32 37} ; [ DW_TAG_lexical_block ]
!268 = metadata !{i32 786688, metadata !269, metadata !"__Val2__", metadata !6, i32 261, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!269 = metadata !{i32 786443, metadata !250, i32 261, i32 14, metadata !6, i32 38} ; [ DW_TAG_lexical_block ]
!270 = metadata !{i32 261, i32 52, metadata !269, null}
!271 = metadata !{i32 261, i32 86, metadata !269, null}
!272 = metadata !{i32 786688, metadata !269, metadata !"__Result__", metadata !6, i32 261, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!273 = metadata !{i32 261, i32 175, metadata !269, null}
!274 = metadata !{i32 250, i32 34, metadata !246, null}
!275 = metadata !{i32 265, i32 3, metadata !111, null}
!276 = metadata !{i32 266, i32 3, metadata !111, null}
!277 = metadata !{i32 267, i32 3, metadata !111, null}
!278 = metadata !{i32 268, i32 3, metadata !111, null}
!279 = metadata !{i32 270, i32 3, metadata !111, null}
!280 = metadata !{i32 76, i32 2, metadata !117, metadata !281}
!281 = metadata !{i32 271, i32 4, metadata !282, null}
!282 = metadata !{i32 786443, metadata !111, i32 270, i32 6, metadata !6, i32 39} ; [ DW_TAG_lexical_block ]
!283 = metadata !{i32 786688, metadata !284, metadata !"__Val2__", metadata !6, i32 272, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!284 = metadata !{i32 786443, metadata !111, i32 272, i32 13, metadata !6, i32 40} ; [ DW_TAG_lexical_block ]
!285 = metadata !{i32 272, i32 51, metadata !284, null}
!286 = metadata !{i32 272, i32 85, metadata !284, null}
!287 = metadata !{i32 786688, metadata !284, metadata !"__Result__", metadata !6, i32 272, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!288 = metadata !{i32 272, i32 174, metadata !284, null}
!289 = metadata !{i32 76, i32 2, metadata !117, metadata !290}
!290 = metadata !{i32 275, i32 4, metadata !291, null}
!291 = metadata !{i32 786443, metadata !111, i32 274, i32 6, metadata !6, i32 41} ; [ DW_TAG_lexical_block ]
!292 = metadata !{i32 786688, metadata !293, metadata !"__Val2__", metadata !6, i32 276, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!293 = metadata !{i32 786443, metadata !111, i32 276, i32 13, metadata !6, i32 42} ; [ DW_TAG_lexical_block ]
!294 = metadata !{i32 276, i32 51, metadata !293, null}
!295 = metadata !{i32 276, i32 85, metadata !293, null}
!296 = metadata !{i32 786688, metadata !293, metadata !"__Result__", metadata !6, i32 276, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!297 = metadata !{i32 276, i32 174, metadata !293, null}
!298 = metadata !{i32 278, i32 3, metadata !111, null}
!299 = metadata !{i32 285, i32 3, metadata !111, null}
!300 = metadata !{i32 76, i32 2, metadata !117, metadata !301}
!301 = metadata !{i32 286, i32 4, metadata !302, null}
!302 = metadata !{i32 786443, metadata !111, i32 285, i32 6, metadata !6, i32 43} ; [ DW_TAG_lexical_block ]
!303 = metadata !{i32 786688, metadata !304, metadata !"__Val2__", metadata !6, i32 287, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!304 = metadata !{i32 786443, metadata !111, i32 287, i32 13, metadata !6, i32 44} ; [ DW_TAG_lexical_block ]
!305 = metadata !{i32 287, i32 51, metadata !304, null}
!306 = metadata !{i32 287, i32 85, metadata !304, null}
!307 = metadata !{i32 786688, metadata !304, metadata !"__Result__", metadata !6, i32 287, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!308 = metadata !{i32 287, i32 174, metadata !304, null}
!309 = metadata !{i32 289, i32 194, metadata !310, null}
!310 = metadata !{i32 786443, metadata !111, i32 289, i32 33, metadata !6, i32 45} ; [ DW_TAG_lexical_block ]
!311 = metadata !{i32 786688, metadata !310, metadata !"__Val2__", metadata !6, i32 289, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!312 = metadata !{i32 289, i32 71, metadata !310, null}
!313 = metadata !{i32 289, i32 105, metadata !310, null}
!314 = metadata !{i32 786688, metadata !310, metadata !"__Result__", metadata !6, i32 289, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!315 = metadata !{i32 786688, metadata !89, metadata !"reg_data", metadata !6, i32 152, metadata !14, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!316 = metadata !{i32 291, i32 3, metadata !111, null}
!317 = metadata !{i32 292, i32 61, metadata !318, null}
!318 = metadata !{i32 786443, metadata !319, i32 292, i32 23, metadata !6, i32 47} ; [ DW_TAG_lexical_block ]
!319 = metadata !{i32 786443, metadata !111, i32 291, i32 6, metadata !6, i32 46} ; [ DW_TAG_lexical_block ]
!320 = metadata !{i32 786688, metadata !318, metadata !"__Val2__", metadata !6, i32 292, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!321 = metadata !{i32 292, i32 95, metadata !318, null}
!322 = metadata !{i32 786688, metadata !318, metadata !"__Result__", metadata !6, i32 292, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!323 = metadata !{i32 786688, metadata !89, metadata !"pre_i2c_sda_val", metadata !6, i32 156, metadata !24, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!324 = metadata !{i32 292, i32 184, metadata !318, null}
!325 = metadata !{i32 76, i32 2, metadata !117, metadata !326}
!326 = metadata !{i32 293, i32 4, metadata !319, null}
!327 = metadata !{i32 786688, metadata !328, metadata !"__Val2__", metadata !6, i32 295, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!328 = metadata !{i32 786443, metadata !319, i32 295, i32 9, metadata !6, i32 48} ; [ DW_TAG_lexical_block ]
!329 = metadata !{i32 295, i32 47, metadata !328, null}
!330 = metadata !{i32 295, i32 81, metadata !328, null}
!331 = metadata !{i32 786688, metadata !328, metadata !"__Result__", metadata !6, i32 295, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!332 = metadata !{i32 295, i32 170, metadata !328, null}
!333 = metadata !{i32 313, i32 7, metadata !334, null}
!334 = metadata !{i32 786443, metadata !111, i32 313, i32 6, metadata !6, i32 53} ; [ DW_TAG_lexical_block ]
!335 = metadata !{i32 297, i32 4, metadata !319, null}
!336 = metadata !{i32 786688, metadata !337, metadata !"__Val2__", metadata !6, i32 297, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!337 = metadata !{i32 786443, metadata !319, i32 297, i32 24, metadata !6, i32 49} ; [ DW_TAG_lexical_block ]
!338 = metadata !{i32 297, i32 62, metadata !337, null}
!339 = metadata !{i32 297, i32 96, metadata !337, null}
!340 = metadata !{i32 786688, metadata !337, metadata !"__Result__", metadata !6, i32 297, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!341 = metadata !{i32 297, i32 185, metadata !337, null}
!342 = metadata !{i32 76, i32 2, metadata !117, metadata !343}
!343 = metadata !{i32 299, i32 6, metadata !344, null}
!344 = metadata !{i32 786443, metadata !345, i32 298, i32 8, metadata !6, i32 51} ; [ DW_TAG_lexical_block ]
!345 = metadata !{i32 786443, metadata !319, i32 297, i32 218, metadata !6, i32 50} ; [ DW_TAG_lexical_block ]
!346 = metadata !{i32 786688, metadata !347, metadata !"__Val2__", metadata !6, i32 300, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!347 = metadata !{i32 786443, metadata !345, i32 300, i32 15, metadata !6, i32 52} ; [ DW_TAG_lexical_block ]
!348 = metadata !{i32 300, i32 53, metadata !347, null}
!349 = metadata !{i32 300, i32 87, metadata !347, null}
!350 = metadata !{i32 786688, metadata !347, metadata !"__Result__", metadata !6, i32 300, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!351 = metadata !{i32 300, i32 176, metadata !347, null}
!352 = metadata !{i32 367, i32 8, metadata !353, null}
!353 = metadata !{i32 786443, metadata !111, i32 367, i32 3, metadata !6, i32 67} ; [ DW_TAG_lexical_block ]
!354 = metadata !{i32 314, i32 4, metadata !334, null}
!355 = metadata !{i32 76, i32 2, metadata !117, metadata !356}
!356 = metadata !{i32 317, i32 6, metadata !357, null}
!357 = metadata !{i32 786443, metadata !358, i32 316, i32 8, metadata !6, i32 55} ; [ DW_TAG_lexical_block ]
!358 = metadata !{i32 786443, metadata !334, i32 314, i32 24, metadata !6, i32 54} ; [ DW_TAG_lexical_block ]
!359 = metadata !{i32 786688, metadata !360, metadata !"__Val2__", metadata !6, i32 318, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!360 = metadata !{i32 786443, metadata !358, i32 318, i32 15, metadata !6, i32 56} ; [ DW_TAG_lexical_block ]
!361 = metadata !{i32 318, i32 53, metadata !360, null}
!362 = metadata !{i32 318, i32 87, metadata !360, null}
!363 = metadata !{i32 786688, metadata !360, metadata !"__Result__", metadata !6, i32 318, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!364 = metadata !{i32 318, i32 176, metadata !360, null}
!365 = metadata !{i32 320, i32 196, metadata !366, null}
!366 = metadata !{i32 786443, metadata !358, i32 320, i32 35, metadata !6, i32 57} ; [ DW_TAG_lexical_block ]
!367 = metadata !{i32 786688, metadata !366, metadata !"__Val2__", metadata !6, i32 320, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!368 = metadata !{i32 320, i32 73, metadata !366, null}
!369 = metadata !{i32 320, i32 107, metadata !366, null}
!370 = metadata !{i32 786688, metadata !366, metadata !"__Result__", metadata !6, i32 320, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!371 = metadata !{i32 323, i32 5, metadata !358, null}
!372 = metadata !{i32 324, i32 63, metadata !373, null}
!373 = metadata !{i32 786443, metadata !374, i32 324, i32 25, metadata !6, i32 59} ; [ DW_TAG_lexical_block ]
!374 = metadata !{i32 786443, metadata !358, i32 323, i32 8, metadata !6, i32 58} ; [ DW_TAG_lexical_block ]
!375 = metadata !{i32 786688, metadata !373, metadata !"__Val2__", metadata !6, i32 324, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!376 = metadata !{i32 324, i32 97, metadata !373, null}
!377 = metadata !{i32 786688, metadata !373, metadata !"__Result__", metadata !6, i32 324, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!378 = metadata !{i32 76, i32 2, metadata !117, metadata !379}
!379 = metadata !{i32 325, i32 6, metadata !374, null}
!380 = metadata !{i32 786688, metadata !381, metadata !"__Val2__", metadata !6, i32 326, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!381 = metadata !{i32 786443, metadata !374, i32 326, i32 11, metadata !6, i32 60} ; [ DW_TAG_lexical_block ]
!382 = metadata !{i32 326, i32 49, metadata !381, null}
!383 = metadata !{i32 326, i32 83, metadata !381, null}
!384 = metadata !{i32 786688, metadata !381, metadata !"__Result__", metadata !6, i32 326, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!385 = metadata !{i32 326, i32 172, metadata !381, null}
!386 = metadata !{i32 786688, metadata !387, metadata !"__Val2__", metadata !6, i32 328, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!387 = metadata !{i32 786443, metadata !358, i32 328, i32 15, metadata !6, i32 61} ; [ DW_TAG_lexical_block ]
!388 = metadata !{i32 328, i32 53, metadata !387, null}
!389 = metadata !{i32 328, i32 87, metadata !387, null}
!390 = metadata !{i32 786688, metadata !387, metadata !"__Result__", metadata !6, i32 328, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!391 = metadata !{i32 328, i32 176, metadata !387, null}
!392 = metadata !{i32 330, i32 5, metadata !358, null}
!393 = metadata !{i32 331, i32 4, metadata !358, null}
!394 = metadata !{i32 334, i32 4, metadata !334, null}
!395 = metadata !{i32 335, i32 4, metadata !334, null}
!396 = metadata !{i32 336, i32 4, metadata !334, null}
!397 = metadata !{i32 337, i32 4, metadata !334, null}
!398 = metadata !{i32 339, i32 4, metadata !334, null}
!399 = metadata !{i32 76, i32 2, metadata !117, metadata !400}
!400 = metadata !{i32 340, i32 5, metadata !401, null}
!401 = metadata !{i32 786443, metadata !334, i32 339, i32 7, metadata !6, i32 62} ; [ DW_TAG_lexical_block ]
!402 = metadata !{i32 786688, metadata !403, metadata !"__Val2__", metadata !6, i32 341, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!403 = metadata !{i32 786443, metadata !334, i32 341, i32 14, metadata !6, i32 63} ; [ DW_TAG_lexical_block ]
!404 = metadata !{i32 341, i32 52, metadata !403, null}
!405 = metadata !{i32 341, i32 86, metadata !403, null}
!406 = metadata !{i32 786688, metadata !403, metadata !"__Result__", metadata !6, i32 341, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!407 = metadata !{i32 341, i32 175, metadata !403, null}
!408 = metadata !{i32 76, i32 2, metadata !117, metadata !409}
!409 = metadata !{i32 344, i32 5, metadata !410, null}
!410 = metadata !{i32 786443, metadata !334, i32 343, i32 7, metadata !6, i32 64} ; [ DW_TAG_lexical_block ]
!411 = metadata !{i32 786688, metadata !412, metadata !"__Val2__", metadata !6, i32 345, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!412 = metadata !{i32 786443, metadata !334, i32 345, i32 14, metadata !6, i32 65} ; [ DW_TAG_lexical_block ]
!413 = metadata !{i32 345, i32 52, metadata !412, null}
!414 = metadata !{i32 345, i32 86, metadata !412, null}
!415 = metadata !{i32 786688, metadata !412, metadata !"__Result__", metadata !6, i32 345, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!416 = metadata !{i32 345, i32 175, metadata !412, null}
!417 = metadata !{i32 347, i32 4, metadata !334, null}
!418 = metadata !{i32 350, i32 4, metadata !334, null}
!419 = metadata !{i32 350, i32 22, metadata !420, null}
!420 = metadata !{i32 786443, metadata !334, i32 350, i32 21, metadata !6, i32 66} ; [ DW_TAG_lexical_block ]
!421 = metadata !{i32 351, i32 5, metadata !420, null}
!422 = metadata !{i32 352, i32 5, metadata !420, null}
!423 = metadata !{i32 353, i32 5, metadata !420, null}
!424 = metadata !{i32 354, i32 5, metadata !420, null}
!425 = metadata !{i32 355, i32 6, metadata !420, null}
!426 = metadata !{i32 356, i32 4, metadata !420, null}
!427 = metadata !{i32 359, i32 3, metadata !334, null}
!428 = metadata !{i32 367, i32 46, metadata !429, null}
!429 = metadata !{i32 786443, metadata !353, i32 367, i32 45, metadata !6, i32 68} ; [ DW_TAG_lexical_block ]
!430 = metadata !{i32 369, i32 4, metadata !429, null}
!431 = metadata !{i32 76, i32 2, metadata !117, metadata !432}
!432 = metadata !{i32 370, i32 5, metadata !433, null}
!433 = metadata !{i32 786443, metadata !429, i32 369, i32 7, metadata !6, i32 69} ; [ DW_TAG_lexical_block ]
!434 = metadata !{i32 786688, metadata !435, metadata !"__Val2__", metadata !6, i32 371, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!435 = metadata !{i32 786443, metadata !429, i32 371, i32 14, metadata !6, i32 70} ; [ DW_TAG_lexical_block ]
!436 = metadata !{i32 371, i32 52, metadata !435, null}
!437 = metadata !{i32 371, i32 86, metadata !435, null}
!438 = metadata !{i32 786688, metadata !435, metadata !"__Result__", metadata !6, i32 371, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!439 = metadata !{i32 371, i32 175, metadata !435, null}
!440 = metadata !{i32 373, i32 195, metadata !441, null}
!441 = metadata !{i32 786443, metadata !429, i32 373, i32 34, metadata !6, i32 71} ; [ DW_TAG_lexical_block ]
!442 = metadata !{i32 786688, metadata !441, metadata !"__Val2__", metadata !6, i32 373, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!443 = metadata !{i32 373, i32 72, metadata !441, null}
!444 = metadata !{i32 373, i32 106, metadata !441, null}
!445 = metadata !{i32 786688, metadata !441, metadata !"__Result__", metadata !6, i32 373, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!446 = metadata !{i32 376, i32 4, metadata !429, null}
!447 = metadata !{i32 76, i32 2, metadata !117, metadata !448}
!448 = metadata !{i32 377, i32 5, metadata !449, null}
!449 = metadata !{i32 786443, metadata !429, i32 376, i32 7, metadata !6, i32 72} ; [ DW_TAG_lexical_block ]
!450 = metadata !{i32 786688, metadata !451, metadata !"__Val2__", metadata !6, i32 378, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!451 = metadata !{i32 786443, metadata !429, i32 378, i32 14, metadata !6, i32 73} ; [ DW_TAG_lexical_block ]
!452 = metadata !{i32 378, i32 52, metadata !451, null}
!453 = metadata !{i32 378, i32 86, metadata !451, null}
!454 = metadata !{i32 786688, metadata !451, metadata !"__Result__", metadata !6, i32 378, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!455 = metadata !{i32 378, i32 175, metadata !451, null}
!456 = metadata !{i32 379, i32 3, metadata !429, null}
!457 = metadata !{i32 367, i32 34, metadata !353, null}
!458 = metadata !{i32 382, i32 3, metadata !111, null}
!459 = metadata !{i32 388, i32 3, metadata !111, null}
!460 = metadata !{i32 76, i32 2, metadata !117, metadata !461}
!461 = metadata !{i32 389, i32 4, metadata !462, null}
!462 = metadata !{i32 786443, metadata !111, i32 388, i32 6, metadata !6, i32 74} ; [ DW_TAG_lexical_block ]
!463 = metadata !{i32 786688, metadata !464, metadata !"__Val2__", metadata !6, i32 390, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!464 = metadata !{i32 786443, metadata !111, i32 390, i32 13, metadata !6, i32 75} ; [ DW_TAG_lexical_block ]
!465 = metadata !{i32 390, i32 51, metadata !464, null}
!466 = metadata !{i32 390, i32 85, metadata !464, null}
!467 = metadata !{i32 786688, metadata !464, metadata !"__Result__", metadata !6, i32 390, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!468 = metadata !{i32 390, i32 174, metadata !464, null}
!469 = metadata !{i32 786688, metadata !470, metadata !"__Val2__", metadata !6, i32 392, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!470 = metadata !{i32 786443, metadata !111, i32 392, i32 8, metadata !6, i32 76} ; [ DW_TAG_lexical_block ]
!471 = metadata !{i32 392, i32 46, metadata !470, null}
!472 = metadata !{i32 392, i32 80, metadata !470, null}
!473 = metadata !{i32 786688, metadata !470, metadata !"__Result__", metadata !6, i32 392, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!474 = metadata !{i32 392, i32 169, metadata !470, null}
!475 = metadata !{i32 395, i32 3, metadata !111, null}
!476 = metadata !{i32 76, i32 2, metadata !117, metadata !477}
!477 = metadata !{i32 396, i32 4, metadata !478, null}
!478 = metadata !{i32 786443, metadata !111, i32 395, i32 6, metadata !6, i32 77} ; [ DW_TAG_lexical_block ]
!479 = metadata !{i32 786688, metadata !480, metadata !"__Val2__", metadata !6, i32 397, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!480 = metadata !{i32 786443, metadata !111, i32 397, i32 13, metadata !6, i32 78} ; [ DW_TAG_lexical_block ]
!481 = metadata !{i32 397, i32 51, metadata !480, null}
!482 = metadata !{i32 397, i32 85, metadata !480, null}
!483 = metadata !{i32 786688, metadata !480, metadata !"__Result__", metadata !6, i32 397, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!484 = metadata !{i32 397, i32 174, metadata !480, null}
!485 = metadata !{i32 400, i32 3, metadata !111, null}
!486 = metadata !{i32 401, i32 3, metadata !111, null}
!487 = metadata !{i32 402, i32 3, metadata !111, null}
!488 = metadata !{i32 403, i32 3, metadata !111, null}
!489 = metadata !{i32 404, i32 14, metadata !111, null}
!490 = metadata !{i32 405, i32 3, metadata !111, null}
!491 = metadata !{i32 406, i32 3, metadata !111, null}
!492 = metadata !{i32 407, i32 4, metadata !111, null}
!493 = metadata !{i32 410, i32 3, metadata !111, null}
!494 = metadata !{i32 76, i32 2, metadata !117, metadata !495}
!495 = metadata !{i32 411, i32 4, metadata !496, null}
!496 = metadata !{i32 786443, metadata !111, i32 410, i32 6, metadata !6, i32 79} ; [ DW_TAG_lexical_block ]
!497 = metadata !{i32 786688, metadata !498, metadata !"__Val2__", metadata !6, i32 412, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!498 = metadata !{i32 786443, metadata !111, i32 412, i32 13, metadata !6, i32 80} ; [ DW_TAG_lexical_block ]
!499 = metadata !{i32 412, i32 51, metadata !498, null}
!500 = metadata !{i32 412, i32 85, metadata !498, null}
!501 = metadata !{i32 786688, metadata !498, metadata !"__Result__", metadata !6, i32 412, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!502 = metadata !{i32 412, i32 174, metadata !498, null}
!503 = metadata !{i32 76, i32 2, metadata !117, metadata !504}
!504 = metadata !{i32 415, i32 4, metadata !505, null}
!505 = metadata !{i32 786443, metadata !111, i32 414, i32 6, metadata !6, i32 81} ; [ DW_TAG_lexical_block ]
!506 = metadata !{i32 786688, metadata !507, metadata !"__Val2__", metadata !6, i32 416, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!507 = metadata !{i32 786443, metadata !111, i32 416, i32 13, metadata !6, i32 82} ; [ DW_TAG_lexical_block ]
!508 = metadata !{i32 416, i32 51, metadata !507, null}
!509 = metadata !{i32 416, i32 85, metadata !507, null}
!510 = metadata !{i32 786688, metadata !507, metadata !"__Result__", metadata !6, i32 416, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!511 = metadata !{i32 416, i32 174, metadata !507, null}
!512 = metadata !{i32 421, i32 14, metadata !513, null}
!513 = metadata !{i32 786443, metadata !111, i32 421, i32 13, metadata !6, i32 83} ; [ DW_TAG_lexical_block ]
!514 = metadata !{i32 422, i32 4, metadata !513, null}
!515 = metadata !{i32 786688, metadata !516, metadata !"__Val2__", metadata !6, i32 423, metadata !14, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!516 = metadata !{i32 786443, metadata !513, i32 423, i32 19, metadata !6, i32 84} ; [ DW_TAG_lexical_block ]
!517 = metadata !{i32 423, i32 59, metadata !516, null}
!518 = metadata !{i32 423, i32 94, metadata !516, null}
!519 = metadata !{i32 786688, metadata !516, metadata !"__Result__", metadata !6, i32 423, metadata !14, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!520 = metadata !{i32 423, i32 183, metadata !516, null}
!521 = metadata !{i32 424, i32 4, metadata !513, null}
!522 = metadata !{i32 425, i32 4, metadata !513, null}
!523 = metadata !{i32 431, i32 6, metadata !524, null}
!524 = metadata !{i32 786443, metadata !525, i32 429, i32 8, metadata !6, i32 87} ; [ DW_TAG_lexical_block ]
!525 = metadata !{i32 786443, metadata !526, i32 428, i32 46, metadata !6, i32 86} ; [ DW_TAG_lexical_block ]
!526 = metadata !{i32 786443, metadata !513, i32 428, i32 4, metadata !6, i32 85} ; [ DW_TAG_lexical_block ]
!527 = metadata !{i32 428, i32 9, metadata !526, null}
!528 = metadata !{i32 76, i32 2, metadata !117, metadata !529}
!529 = metadata !{i32 430, i32 6, metadata !524, null}
!530 = metadata !{i32 786688, metadata !531, metadata !"__Val2__", metadata !6, i32 433, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!531 = metadata !{i32 786443, metadata !525, i32 433, i32 15, metadata !6, i32 88} ; [ DW_TAG_lexical_block ]
!532 = metadata !{i32 433, i32 53, metadata !531, null}
!533 = metadata !{i32 433, i32 87, metadata !531, null}
!534 = metadata !{i32 786688, metadata !531, metadata !"__Result__", metadata !6, i32 433, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!535 = metadata !{i32 433, i32 176, metadata !531, null}
!536 = metadata !{i32 435, i32 5, metadata !525, null}
!537 = metadata !{i32 437, i32 5, metadata !525, null}
!538 = metadata !{i32 76, i32 2, metadata !117, metadata !539}
!539 = metadata !{i32 438, i32 6, metadata !540, null}
!540 = metadata !{i32 786443, metadata !525, i32 437, i32 8, metadata !6, i32 89} ; [ DW_TAG_lexical_block ]
!541 = metadata !{i32 786688, metadata !542, metadata !"__Val2__", metadata !6, i32 439, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!542 = metadata !{i32 786443, metadata !540, i32 439, i32 11, metadata !6, i32 90} ; [ DW_TAG_lexical_block ]
!543 = metadata !{i32 439, i32 49, metadata !542, null}
!544 = metadata !{i32 439, i32 83, metadata !542, null}
!545 = metadata !{i32 786688, metadata !542, metadata !"__Result__", metadata !6, i32 439, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!546 = metadata !{i32 439, i32 172, metadata !542, null}
!547 = metadata !{i32 786688, metadata !548, metadata !"__Val2__", metadata !6, i32 441, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!548 = metadata !{i32 786443, metadata !525, i32 441, i32 15, metadata !6, i32 91} ; [ DW_TAG_lexical_block ]
!549 = metadata !{i32 441, i32 53, metadata !548, null}
!550 = metadata !{i32 441, i32 87, metadata !548, null}
!551 = metadata !{i32 786688, metadata !548, metadata !"__Result__", metadata !6, i32 441, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!552 = metadata !{i32 441, i32 176, metadata !548, null}
!553 = metadata !{i32 443, i32 5, metadata !525, null}
!554 = metadata !{i32 443, i32 24, metadata !555, null}
!555 = metadata !{i32 786443, metadata !525, i32 443, i32 23, metadata !6, i32 92} ; [ DW_TAG_lexical_block ]
!556 = metadata !{i32 444, i32 6, metadata !555, null}
!557 = metadata !{i32 786688, metadata !558, metadata !"__Val2__", metadata !6, i32 445, metadata !14, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!558 = metadata !{i32 786443, metadata !555, i32 445, i32 21, metadata !6, i32 93} ; [ DW_TAG_lexical_block ]
!559 = metadata !{i32 445, i32 61, metadata !558, null}
!560 = metadata !{i32 445, i32 96, metadata !558, null}
!561 = metadata !{i32 786688, metadata !558, metadata !"__Result__", metadata !6, i32 445, metadata !14, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!562 = metadata !{i32 445, i32 185, metadata !558, null}
!563 = metadata !{i32 446, i32 6, metadata !555, null}
!564 = metadata !{i32 447, i32 5, metadata !555, null}
!565 = metadata !{i32 428, i32 35, metadata !526, null}
!566 = metadata !{i32 450, i32 4, metadata !513, null}
!567 = metadata !{i32 451, i32 4, metadata !513, null}
!568 = metadata !{i32 452, i32 4, metadata !513, null}
!569 = metadata !{i32 453, i32 15, metadata !513, null}
!570 = metadata !{i32 454, i32 4, metadata !513, null}
!571 = metadata !{i32 455, i32 4, metadata !513, null}
!572 = metadata !{i32 456, i32 5, metadata !513, null}
!573 = metadata !{i32 459, i32 4, metadata !513, null}
!574 = metadata !{i32 76, i32 2, metadata !117, metadata !575}
!575 = metadata !{i32 460, i32 5, metadata !576, null}
!576 = metadata !{i32 786443, metadata !513, i32 459, i32 7, metadata !6, i32 94} ; [ DW_TAG_lexical_block ]
!577 = metadata !{i32 786688, metadata !578, metadata !"__Val2__", metadata !6, i32 461, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!578 = metadata !{i32 786443, metadata !513, i32 461, i32 14, metadata !6, i32 95} ; [ DW_TAG_lexical_block ]
!579 = metadata !{i32 461, i32 52, metadata !578, null}
!580 = metadata !{i32 461, i32 86, metadata !578, null}
!581 = metadata !{i32 786688, metadata !578, metadata !"__Result__", metadata !6, i32 461, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!582 = metadata !{i32 461, i32 175, metadata !578, null}
!583 = metadata !{i32 786688, metadata !584, metadata !"__Val2__", metadata !6, i32 463, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!584 = metadata !{i32 786443, metadata !513, i32 463, i32 22, metadata !6, i32 96} ; [ DW_TAG_lexical_block ]
!585 = metadata !{i32 463, i32 60, metadata !584, null}
!586 = metadata !{i32 463, i32 94, metadata !584, null}
!587 = metadata !{i32 786688, metadata !584, metadata !"__Result__", metadata !6, i32 463, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!588 = metadata !{i32 786688, metadata !89, metadata !"terminate_read", metadata !6, i32 155, metadata !24, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!589 = metadata !{i32 463, i32 183, metadata !584, null}
!590 = metadata !{i32 465, i32 4, metadata !513, null}
!591 = metadata !{i32 76, i32 2, metadata !117, metadata !592}
!592 = metadata !{i32 466, i32 5, metadata !593, null}
!593 = metadata !{i32 786443, metadata !513, i32 465, i32 7, metadata !6, i32 97} ; [ DW_TAG_lexical_block ]
!594 = metadata !{i32 786688, metadata !595, metadata !"__Val2__", metadata !6, i32 467, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!595 = metadata !{i32 786443, metadata !513, i32 467, i32 14, metadata !6, i32 98} ; [ DW_TAG_lexical_block ]
!596 = metadata !{i32 467, i32 52, metadata !595, null}
!597 = metadata !{i32 467, i32 86, metadata !595, null}
!598 = metadata !{i32 786688, metadata !595, metadata !"__Result__", metadata !6, i32 467, metadata !34, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!599 = metadata !{i32 467, i32 175, metadata !595, null}
!600 = metadata !{i32 468, i32 3, metadata !513, null}
