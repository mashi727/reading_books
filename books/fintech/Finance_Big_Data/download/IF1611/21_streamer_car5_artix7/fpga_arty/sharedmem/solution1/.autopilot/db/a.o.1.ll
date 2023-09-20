; ModuleID = 'D:/21_streamer_car5_artix7/fpga_arty/sharedmem/solution1/.autopilot/db/a.g.1.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-w64-mingw32"

@w_req3 = common global i1 false, align 1         ; [#uses=4 type=i1*]
@w_req2 = common global i1 false, align 1         ; [#uses=4 type=i1*]
@w_req1 = common global i1 false, align 1         ; [#uses=4 type=i1*]
@w_req0 = common global i1 false, align 1         ; [#uses=4 type=i1*]
@w_ack3 = global i1 false, align 1                ; [#uses=5 type=i1*]
@w_ack2 = global i1 false, align 1                ; [#uses=5 type=i1*]
@w_ack1 = global i1 false, align 1                ; [#uses=5 type=i1*]
@w_ack0 = global i1 false, align 1                ; [#uses=5 type=i1*]
@r_req3 = common global i1 false, align 1         ; [#uses=4 type=i1*]
@r_req2 = common global i1 false, align 1         ; [#uses=4 type=i1*]
@r_req1 = common global i1 false, align 1         ; [#uses=4 type=i1*]
@r_req0 = common global i1 false, align 1         ; [#uses=4 type=i1*]
@r_ack3 = global i1 false, align 1                ; [#uses=5 type=i1*]
@r_ack2 = global i1 false, align 1                ; [#uses=5 type=i1*]
@r_ack1 = global i1 false, align 1                ; [#uses=5 type=i1*]
@r_ack0 = global i1 false, align 1                ; [#uses=5 type=i1*]
@dout3 = global i8 0, align 1                     ; [#uses=2 type=i8*]
@dout2 = global i8 0, align 1                     ; [#uses=2 type=i8*]
@dout1 = global i8 0, align 1                     ; [#uses=2 type=i8*]
@dout0 = global i8 0, align 1                     ; [#uses=2 type=i8*]
@din3 = common global i8 0, align 1               ; [#uses=2 type=i8*]
@din2 = common global i8 0, align 1               ; [#uses=2 type=i8*]
@din1 = common global i8 0, align 1               ; [#uses=2 type=i8*]
@din0 = common global i8 0, align 1               ; [#uses=2 type=i8*]
@addr3 = global i8 0, align 1                     ; [#uses=3 type=i8*]
@addr2 = global i8 0, align 1                     ; [#uses=3 type=i8*]
@addr1 = global i8 0, align 1                     ; [#uses=3 type=i8*]
@addr0 = global i8 0, align 1                     ; [#uses=3 type=i8*]
@.str9 = private unnamed_addr constant [12 x i8] c"hls_label_5\00", align 1 ; [#uses=1 type=[12 x i8]*]
@.str8 = private unnamed_addr constant [12 x i8] c"hls_label_8\00", align 1 ; [#uses=1 type=[12 x i8]*]
@.str7 = private unnamed_addr constant [12 x i8] c"hls_label_0\00", align 1 ; [#uses=1 type=[12 x i8]*]
@.str6 = private unnamed_addr constant [12 x i8] c"hls_label_4\00", align 1 ; [#uses=1 type=[12 x i8]*]
@.str5 = private unnamed_addr constant [12 x i8] c"hls_label_3\00", align 1 ; [#uses=1 type=[12 x i8]*]
@.str4 = private unnamed_addr constant [12 x i8] c"hls_label_6\00", align 1 ; [#uses=1 type=[12 x i8]*]
@.str3 = private unnamed_addr constant [12 x i8] c"hls_label_7\00", align 1 ; [#uses=1 type=[12 x i8]*]
@.str2 = private unnamed_addr constant [12 x i8] c"hls_label_1\00", align 1 ; [#uses=1 type=[12 x i8]*]
@.str1 = private unnamed_addr constant [1 x i8] zeroinitializer, align 1 ; [#uses=1 type=[1 x i8]*]
@.str = private unnamed_addr constant [8 x i8] c"ap_none\00", align 1 ; [#uses=1 type=[8 x i8]*]

; [#uses=0]
define void @sharedmem() noreturn nounwind uwtable {
  %mem = alloca [256 x i8], align 16              ; [#uses=9 type=[256 x i8]*]
  call void (...)* @_ssdm_op_SpecTopModule(), !dbg !47 ; [debug line = 88:1]
  %addr0.load = load volatile i8* @addr0, align 1, !dbg !49 ; [#uses=1 type=i8] [debug line = 90:1]
  %tmp = zext i8 %addr0.load to i32, !dbg !49     ; [#uses=1 type=i32] [debug line = 90:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp, i8* getelementptr inbounds ([8 x i8]* @.str, i64 0, i64 0), i32 1, i32 1, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0)), !dbg !49 ; [debug line = 90:1]
  %din0.load = load volatile i8* @din0, align 1, !dbg !50 ; [#uses=1 type=i8] [debug line = 91:1]
  %tmp.1 = zext i8 %din0.load to i32, !dbg !50    ; [#uses=1 type=i32] [debug line = 91:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.1, i8* getelementptr inbounds ([8 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0)), !dbg !50 ; [debug line = 91:1]
  %dout0.load = load volatile i8* @dout0, align 1, !dbg !51 ; [#uses=1 type=i8] [debug line = 92:1]
  %tmp.2 = zext i8 %dout0.load to i32, !dbg !51   ; [#uses=1 type=i32] [debug line = 92:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.2, i8* getelementptr inbounds ([8 x i8]* @.str, i64 0, i64 0), i32 1, i32 1, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0)), !dbg !51 ; [debug line = 92:1]
  %r_req0.load = load volatile i1* @r_req0, align 1, !dbg !52 ; [#uses=1 type=i1] [debug line = 93:1]
  %tmp.3 = zext i1 %r_req0.load to i32, !dbg !52  ; [#uses=1 type=i32] [debug line = 93:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.3, i8* getelementptr inbounds ([8 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0)), !dbg !52 ; [debug line = 93:1]
  %r_ack0.load = load volatile i1* @r_ack0, align 1, !dbg !53 ; [#uses=1 type=i1] [debug line = 94:1]
  %tmp.4 = zext i1 %r_ack0.load to i32, !dbg !53  ; [#uses=1 type=i32] [debug line = 94:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.4, i8* getelementptr inbounds ([8 x i8]* @.str, i64 0, i64 0), i32 1, i32 1, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0)), !dbg !53 ; [debug line = 94:1]
  %w_req0.load = load volatile i1* @w_req0, align 1, !dbg !54 ; [#uses=1 type=i1] [debug line = 95:1]
  %tmp.5 = zext i1 %w_req0.load to i32, !dbg !54  ; [#uses=1 type=i32] [debug line = 95:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.5, i8* getelementptr inbounds ([8 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0)), !dbg !54 ; [debug line = 95:1]
  %w_ack0.load = load volatile i1* @w_ack0, align 1, !dbg !55 ; [#uses=1 type=i1] [debug line = 96:1]
  %tmp.6 = zext i1 %w_ack0.load to i32, !dbg !55  ; [#uses=1 type=i32] [debug line = 96:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.6, i8* getelementptr inbounds ([8 x i8]* @.str, i64 0, i64 0), i32 1, i32 1, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0)), !dbg !55 ; [debug line = 96:1]
  %addr1.load = load volatile i8* @addr1, align 1, !dbg !56 ; [#uses=1 type=i8] [debug line = 98:1]
  %tmp.7 = zext i8 %addr1.load to i32, !dbg !56   ; [#uses=1 type=i32] [debug line = 98:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.7, i8* getelementptr inbounds ([8 x i8]* @.str, i64 0, i64 0), i32 1, i32 1, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0)), !dbg !56 ; [debug line = 98:1]
  %din1.load = load volatile i8* @din1, align 1, !dbg !57 ; [#uses=1 type=i8] [debug line = 99:1]
  %tmp.8 = zext i8 %din1.load to i32, !dbg !57    ; [#uses=1 type=i32] [debug line = 99:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.8, i8* getelementptr inbounds ([8 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0)), !dbg !57 ; [debug line = 99:1]
  %dout1.load = load volatile i8* @dout1, align 1, !dbg !58 ; [#uses=1 type=i8] [debug line = 100:1]
  %tmp.9 = zext i8 %dout1.load to i32, !dbg !58   ; [#uses=1 type=i32] [debug line = 100:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.9, i8* getelementptr inbounds ([8 x i8]* @.str, i64 0, i64 0), i32 1, i32 1, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0)), !dbg !58 ; [debug line = 100:1]
  %r_req1.load = load volatile i1* @r_req1, align 1, !dbg !59 ; [#uses=1 type=i1] [debug line = 101:1]
  %tmp.10 = zext i1 %r_req1.load to i32, !dbg !59 ; [#uses=1 type=i32] [debug line = 101:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.10, i8* getelementptr inbounds ([8 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0)), !dbg !59 ; [debug line = 101:1]
  %r_ack1.load = load volatile i1* @r_ack1, align 1, !dbg !60 ; [#uses=1 type=i1] [debug line = 102:1]
  %tmp.11 = zext i1 %r_ack1.load to i32, !dbg !60 ; [#uses=1 type=i32] [debug line = 102:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.11, i8* getelementptr inbounds ([8 x i8]* @.str, i64 0, i64 0), i32 1, i32 1, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0)), !dbg !60 ; [debug line = 102:1]
  %w_req1.load = load volatile i1* @w_req1, align 1, !dbg !61 ; [#uses=1 type=i1] [debug line = 103:1]
  %tmp.12 = zext i1 %w_req1.load to i32, !dbg !61 ; [#uses=1 type=i32] [debug line = 103:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.12, i8* getelementptr inbounds ([8 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0)), !dbg !61 ; [debug line = 103:1]
  %w_ack1.load = load volatile i1* @w_ack1, align 1, !dbg !62 ; [#uses=1 type=i1] [debug line = 104:1]
  %tmp.13 = zext i1 %w_ack1.load to i32, !dbg !62 ; [#uses=1 type=i32] [debug line = 104:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.13, i8* getelementptr inbounds ([8 x i8]* @.str, i64 0, i64 0), i32 1, i32 1, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0)), !dbg !62 ; [debug line = 104:1]
  %addr2.load = load volatile i8* @addr2, align 1, !dbg !63 ; [#uses=1 type=i8] [debug line = 106:1]
  %tmp.14 = zext i8 %addr2.load to i32, !dbg !63  ; [#uses=1 type=i32] [debug line = 106:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.14, i8* getelementptr inbounds ([8 x i8]* @.str, i64 0, i64 0), i32 1, i32 1, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0)), !dbg !63 ; [debug line = 106:1]
  %din2.load = load volatile i8* @din2, align 1, !dbg !64 ; [#uses=1 type=i8] [debug line = 107:1]
  %tmp.15 = zext i8 %din2.load to i32, !dbg !64   ; [#uses=1 type=i32] [debug line = 107:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.15, i8* getelementptr inbounds ([8 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0)), !dbg !64 ; [debug line = 107:1]
  %dout2.load = load volatile i8* @dout2, align 1, !dbg !65 ; [#uses=1 type=i8] [debug line = 108:1]
  %tmp.16 = zext i8 %dout2.load to i32, !dbg !65  ; [#uses=1 type=i32] [debug line = 108:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.16, i8* getelementptr inbounds ([8 x i8]* @.str, i64 0, i64 0), i32 1, i32 1, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0)), !dbg !65 ; [debug line = 108:1]
  %r_req2.load = load volatile i1* @r_req2, align 1, !dbg !66 ; [#uses=1 type=i1] [debug line = 109:1]
  %tmp.17 = zext i1 %r_req2.load to i32, !dbg !66 ; [#uses=1 type=i32] [debug line = 109:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.17, i8* getelementptr inbounds ([8 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0)), !dbg !66 ; [debug line = 109:1]
  %r_ack2.load = load volatile i1* @r_ack2, align 1, !dbg !67 ; [#uses=1 type=i1] [debug line = 110:1]
  %tmp.18 = zext i1 %r_ack2.load to i32, !dbg !67 ; [#uses=1 type=i32] [debug line = 110:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.18, i8* getelementptr inbounds ([8 x i8]* @.str, i64 0, i64 0), i32 1, i32 1, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0)), !dbg !67 ; [debug line = 110:1]
  %w_req2.load = load volatile i1* @w_req2, align 1, !dbg !68 ; [#uses=1 type=i1] [debug line = 111:1]
  %tmp.19 = zext i1 %w_req2.load to i32, !dbg !68 ; [#uses=1 type=i32] [debug line = 111:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.19, i8* getelementptr inbounds ([8 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0)), !dbg !68 ; [debug line = 111:1]
  %w_ack2.load = load volatile i1* @w_ack2, align 1, !dbg !69 ; [#uses=1 type=i1] [debug line = 112:1]
  %tmp.20 = zext i1 %w_ack2.load to i32, !dbg !69 ; [#uses=1 type=i32] [debug line = 112:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.20, i8* getelementptr inbounds ([8 x i8]* @.str, i64 0, i64 0), i32 1, i32 1, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0)), !dbg !69 ; [debug line = 112:1]
  %addr3.load = load volatile i8* @addr3, align 1, !dbg !70 ; [#uses=1 type=i8] [debug line = 114:1]
  %tmp.21 = zext i8 %addr3.load to i32, !dbg !70  ; [#uses=1 type=i32] [debug line = 114:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.21, i8* getelementptr inbounds ([8 x i8]* @.str, i64 0, i64 0), i32 1, i32 1, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0)), !dbg !70 ; [debug line = 114:1]
  %din3.load = load volatile i8* @din3, align 1, !dbg !71 ; [#uses=1 type=i8] [debug line = 115:1]
  %tmp.22 = zext i8 %din3.load to i32, !dbg !71   ; [#uses=1 type=i32] [debug line = 115:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.22, i8* getelementptr inbounds ([8 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0)), !dbg !71 ; [debug line = 115:1]
  %dout3.load = load volatile i8* @dout3, align 1, !dbg !72 ; [#uses=1 type=i8] [debug line = 116:1]
  %tmp.23 = zext i8 %dout3.load to i32, !dbg !72  ; [#uses=1 type=i32] [debug line = 116:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.23, i8* getelementptr inbounds ([8 x i8]* @.str, i64 0, i64 0), i32 1, i32 1, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0)), !dbg !72 ; [debug line = 116:1]
  %r_req3.load = load volatile i1* @r_req3, align 1, !dbg !73 ; [#uses=1 type=i1] [debug line = 117:1]
  %tmp.24 = zext i1 %r_req3.load to i32, !dbg !73 ; [#uses=1 type=i32] [debug line = 117:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.24, i8* getelementptr inbounds ([8 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0)), !dbg !73 ; [debug line = 117:1]
  %r_ack3.load = load volatile i1* @r_ack3, align 1, !dbg !74 ; [#uses=1 type=i1] [debug line = 118:1]
  %tmp.25 = zext i1 %r_ack3.load to i32, !dbg !74 ; [#uses=1 type=i32] [debug line = 118:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.25, i8* getelementptr inbounds ([8 x i8]* @.str, i64 0, i64 0), i32 1, i32 1, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0)), !dbg !74 ; [debug line = 118:1]
  %w_req3.load = load volatile i1* @w_req3, align 1, !dbg !75 ; [#uses=1 type=i1] [debug line = 119:1]
  %tmp.26 = zext i1 %w_req3.load to i32, !dbg !75 ; [#uses=1 type=i32] [debug line = 119:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.26, i8* getelementptr inbounds ([8 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0)), !dbg !75 ; [debug line = 119:1]
  %w_ack3.load = load volatile i1* @w_ack3, align 1, !dbg !76 ; [#uses=1 type=i1] [debug line = 120:1]
  %tmp.27 = zext i1 %w_ack3.load to i32, !dbg !76 ; [#uses=1 type=i32] [debug line = 120:1]
  call void (...)* @_ssdm_op_SpecInterface(i32 %tmp.27, i8* getelementptr inbounds ([8 x i8]* @.str, i64 0, i64 0), i32 1, i32 1, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0), i32 0, i32 0, i32 0, i32 0, i8* getelementptr inbounds ([1 x i8]* @.str1, i64 0, i64 0)), !dbg !76 ; [debug line = 120:1]
  call void @llvm.dbg.declare(metadata !{[256 x i8]* %mem}, metadata !77), !dbg !81 ; [debug line = 124:8] [debug variable = mem]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !82 ; [debug line = 127:2]
  store volatile i1 false, i1* @r_ack0, align 1, !dbg !83 ; [debug line = 128:2]
  store volatile i1 false, i1* @w_ack0, align 1, !dbg !84 ; [debug line = 129:2]
  store volatile i1 false, i1* @r_ack1, align 1, !dbg !85 ; [debug line = 130:2]
  store volatile i1 false, i1* @w_ack1, align 1, !dbg !86 ; [debug line = 131:2]
  store volatile i1 false, i1* @r_ack2, align 1, !dbg !87 ; [debug line = 132:2]
  store volatile i1 false, i1* @w_ack2, align 1, !dbg !88 ; [debug line = 133:2]
  store volatile i1 false, i1* @r_ack3, align 1, !dbg !89 ; [debug line = 134:2]
  store volatile i1 false, i1* @w_ack3, align 1, !dbg !90 ; [debug line = 135:2]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !91 ; [debug line = 136:2]
  br label %1, !dbg !92                           ; [debug line = 138:7]

; <label>:1                                       ; preds = %2, %0
  %addr = phi i9 [ 0, %0 ], [ %addr.1, %2 ]       ; [#uses=3 type=i9]
  %exitcond = icmp eq i9 %addr, -256, !dbg !92    ; [#uses=1 type=i1] [debug line = 138:7]
  br i1 %exitcond, label %.preheader.preheader, label %2, !dbg !92 ; [debug line = 138:7]

.preheader.preheader:                             ; preds = %1
  br label %.preheader

; <label>:2                                       ; preds = %1
  %tmp.28 = zext i9 %addr to i64, !dbg !94        ; [#uses=1 type=i64] [debug line = 139:3]
  %mem.addr = getelementptr inbounds [256 x i8]* %mem, i64 0, i64 %tmp.28, !dbg !94 ; [#uses=1 type=i8*] [debug line = 139:3]
  store i8 0, i8* %mem.addr, align 1, !dbg !94    ; [debug line = 139:3]
  %addr.1 = add i9 %addr, 1, !dbg !96             ; [#uses=1 type=i9] [debug line = 138:29]
  call void @llvm.dbg.value(metadata !{i9 %addr.1}, i64 0, metadata !97), !dbg !96 ; [debug line = 138:29] [debug variable = addr]
  br label %1, !dbg !96                           ; [debug line = 138:29]

.preheader:                                       ; preds = %.preheader.backedge, %.preheader.preheader
  %ch = phi i8 [ 0, %.preheader.preheader ], [ %ch.0.be, %.preheader.backedge ] ; [#uses=2 type=i8]
  %loop_begin = call i32 (...)* @_ssdm_op_SpecLoopBegin() nounwind ; [#uses=0 type=i32]
  %tmp.30 = zext i8 %ch to i32, !dbg !100         ; [#uses=1 type=i32] [debug line = 144:3]
  switch i32 %tmp.30, label %55 [
    i32 0, label %3
    i32 1, label %11
    i32 2, label %22
    i32 3, label %33
    i32 4, label %44
  ], !dbg !100                                    ; [debug line = 144:3]

; <label>:3                                       ; preds = %.preheader
  %r_req0.load.1 = load volatile i1* @r_req0, align 1, !dbg !102 ; [#uses=1 type=i1] [debug line = 146:4]
  br i1 %r_req0.load.1, label %._crit_edge, label %4, !dbg !102 ; [debug line = 146:4]

; <label>:4                                       ; preds = %3
  %w_req0.load.1 = load volatile i1* @w_req0, align 1, !dbg !102 ; [#uses=1 type=i1] [debug line = 146:4]
  br i1 %w_req0.load.1, label %._crit_edge, label %5, !dbg !102 ; [debug line = 146:4]

._crit_edge:                                      ; preds = %4, %3
  br label %.preheader.backedge, !dbg !104        ; [debug line = 147:5]

.preheader.backedge:                              ; preds = %55, %54, %43, %32, %21, %._crit_edge6, %10, %._crit_edge4, %._crit_edge2, %._crit_edge
  %ch.0.be = phi i8 [ 0, %55 ], [ 0, %54 ], [ 0, %43 ], [ 0, %32 ], [ 0, %21 ], [ 1, %._crit_edge ], [ 2, %._crit_edge2 ], [ 3, %._crit_edge4 ], [ 4, %._crit_edge6 ], [ %ch, %10 ] ; [#uses=1 type=i8]
  br label %.preheader

; <label>:5                                       ; preds = %4
  %r_req1.load.2 = load volatile i1* @r_req1, align 1, !dbg !105 ; [#uses=1 type=i1] [debug line = 148:9]
  br i1 %r_req1.load.2, label %._crit_edge2, label %6, !dbg !105 ; [debug line = 148:9]

; <label>:6                                       ; preds = %5
  %w_req1.load.2 = load volatile i1* @w_req1, align 1, !dbg !105 ; [#uses=1 type=i1] [debug line = 148:9]
  br i1 %w_req1.load.2, label %._crit_edge2, label %7, !dbg !105 ; [debug line = 148:9]

._crit_edge2:                                     ; preds = %6, %5
  br label %.preheader.backedge, !dbg !106        ; [debug line = 149:5]

; <label>:7                                       ; preds = %6
  %r_req2.load.3 = load volatile i1* @r_req2, align 1, !dbg !107 ; [#uses=1 type=i1] [debug line = 150:9]
  br i1 %r_req2.load.3, label %._crit_edge4, label %8, !dbg !107 ; [debug line = 150:9]

; <label>:8                                       ; preds = %7
  %w_req2.load.3 = load volatile i1* @w_req2, align 1, !dbg !107 ; [#uses=1 type=i1] [debug line = 150:9]
  br i1 %w_req2.load.3, label %._crit_edge4, label %9, !dbg !107 ; [debug line = 150:9]

._crit_edge4:                                     ; preds = %8, %7
  br label %.preheader.backedge, !dbg !108        ; [debug line = 151:5]

; <label>:9                                       ; preds = %8
  %r_req3.load.3 = load volatile i1* @r_req3, align 1, !dbg !109 ; [#uses=1 type=i1] [debug line = 152:9]
  br i1 %r_req3.load.3, label %._crit_edge6, label %10, !dbg !109 ; [debug line = 152:9]

; <label>:10                                      ; preds = %9
  %w_req3.load.3 = load volatile i1* @w_req3, align 1, !dbg !109 ; [#uses=1 type=i1] [debug line = 152:9]
  br i1 %w_req3.load.3, label %._crit_edge6, label %.preheader.backedge, !dbg !109 ; [debug line = 152:9]

._crit_edge6:                                     ; preds = %10, %9
  br label %.preheader.backedge, !dbg !110        ; [debug line = 153:5]

; <label>:11                                      ; preds = %.preheader
  %r_req0.load.2 = load volatile i1* @r_req0, align 1, !dbg !111 ; [#uses=1 type=i1] [debug line = 157:4]
  br i1 %r_req0.load.2, label %12, label %16, !dbg !111 ; [debug line = 157:4]

; <label>:12                                      ; preds = %11
  %rbegin1 = call i32 (...)* @_ssdm_op_SpecRegionBegin(i8* getelementptr inbounds ([12 x i8]* @.str2, i64 0, i64 0)) nounwind, !dbg !112 ; [#uses=1 type=i32] [debug line = 157:22]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !114 ; [debug line = 158:5]
  %addr0.load.1 = load volatile i8* @addr0, align 1, !dbg !115 ; [#uses=1 type=i8] [debug line = 159:5]
  %tmp.31 = zext i8 %addr0.load.1 to i64, !dbg !115 ; [#uses=1 type=i64] [debug line = 159:5]
  %mem.addr.1 = getelementptr inbounds [256 x i8]* %mem, i64 0, i64 %tmp.31, !dbg !115 ; [#uses=1 type=i8*] [debug line = 159:5]
  %mem.load = load i8* %mem.addr.1, align 1, !dbg !115 ; [#uses=2 type=i8] [debug line = 159:5]
  call void (...)* @_ssdm_SpecKeepArrayLoad(i8 %mem.load) nounwind
  store volatile i8 %mem.load, i8* @dout0, align 1, !dbg !115 ; [debug line = 159:5]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !116 ; [debug line = 160:5]
  br label %13, !dbg !117                         ; [debug line = 161:5]

; <label>:13                                      ; preds = %14, %12
  %r_req0.load.3 = load volatile i1* @r_req0, align 1, !dbg !117 ; [#uses=1 type=i1] [debug line = 161:5]
  br i1 %r_req0.load.3, label %14, label %15, !dbg !117 ; [debug line = 161:5]

; <label>:14                                      ; preds = %13
  store volatile i1 true, i1* @r_ack0, align 1, !dbg !118 ; [debug line = 162:6]
  br label %13, !dbg !120                         ; [debug line = 163:5]

; <label>:15                                      ; preds = %13
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !121 ; [debug line = 164:5]
  store volatile i1 false, i1* @r_ack0, align 1, !dbg !122 ; [debug line = 165:5]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !123 ; [debug line = 166:5]
  %rend13 = call i32 (...)* @_ssdm_op_SpecRegionEnd(i8* getelementptr inbounds ([12 x i8]* @.str2, i64 0, i64 0), i32 %rbegin1) nounwind, !dbg !124 ; [#uses=0 type=i32] [debug line = 167:4]
  br label %21, !dbg !124                         ; [debug line = 167:4]

; <label>:16                                      ; preds = %11
  %w_req0.load.2 = load volatile i1* @w_req0, align 1, !dbg !125 ; [#uses=1 type=i1] [debug line = 168:9]
  br i1 %w_req0.load.2, label %17, label %._crit_edge8, !dbg !125 ; [debug line = 168:9]

; <label>:17                                      ; preds = %16
  %rbegin5 = call i32 (...)* @_ssdm_op_SpecRegionBegin(i8* getelementptr inbounds ([12 x i8]* @.str3, i64 0, i64 0)) nounwind, !dbg !126 ; [#uses=1 type=i32] [debug line = 168:27]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !128 ; [debug line = 169:5]
  %din0.load.1 = load volatile i8* @din0, align 1, !dbg !129 ; [#uses=1 type=i8] [debug line = 170:5]
  %addr0.load.2 = load volatile i8* @addr0, align 1, !dbg !129 ; [#uses=1 type=i8] [debug line = 170:5]
  %tmp.35 = zext i8 %addr0.load.2 to i64, !dbg !129 ; [#uses=1 type=i64] [debug line = 170:5]
  %mem.addr.5 = getelementptr inbounds [256 x i8]* %mem, i64 0, i64 %tmp.35, !dbg !129 ; [#uses=1 type=i8*] [debug line = 170:5]
  store i8 %din0.load.1, i8* %mem.addr.5, align 1, !dbg !129 ; [debug line = 170:5]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !130 ; [debug line = 171:5]
  br label %18, !dbg !131                         ; [debug line = 172:5]

; <label>:18                                      ; preds = %19, %17
  %w_req0.load.3 = load volatile i1* @w_req0, align 1, !dbg !131 ; [#uses=1 type=i1] [debug line = 172:5]
  br i1 %w_req0.load.3, label %19, label %20, !dbg !131 ; [debug line = 172:5]

; <label>:19                                      ; preds = %18
  store volatile i1 true, i1* @w_ack0, align 1, !dbg !132 ; [debug line = 173:6]
  br label %18, !dbg !134                         ; [debug line = 174:5]

; <label>:20                                      ; preds = %18
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !135 ; [debug line = 175:5]
  store volatile i1 false, i1* @w_ack0, align 1, !dbg !136 ; [debug line = 176:5]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !137 ; [debug line = 177:5]
  %rend23 = call i32 (...)* @_ssdm_op_SpecRegionEnd(i8* getelementptr inbounds ([12 x i8]* @.str3, i64 0, i64 0), i32 %rbegin5) nounwind, !dbg !138 ; [#uses=0 type=i32] [debug line = 178:4]
  br label %._crit_edge8, !dbg !138               ; [debug line = 178:4]

._crit_edge8:                                     ; preds = %20, %16
  br label %21

; <label>:21                                      ; preds = %._crit_edge8, %15
  br label %.preheader.backedge, !dbg !139        ; [debug line = 180:4]

; <label>:22                                      ; preds = %.preheader
  %r_req1.load.1 = load volatile i1* @r_req1, align 1, !dbg !140 ; [#uses=1 type=i1] [debug line = 183:4]
  br i1 %r_req1.load.1, label %23, label %27, !dbg !140 ; [debug line = 183:4]

; <label>:23                                      ; preds = %22
  %rbegin2 = call i32 (...)* @_ssdm_op_SpecRegionBegin(i8* getelementptr inbounds ([12 x i8]* @.str4, i64 0, i64 0)) nounwind, !dbg !141 ; [#uses=1 type=i32] [debug line = 183:22]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !143 ; [debug line = 184:5]
  %addr1.load.1 = load volatile i8* @addr1, align 1, !dbg !144 ; [#uses=1 type=i8] [debug line = 185:5]
  %tmp.32 = zext i8 %addr1.load.1 to i64, !dbg !144 ; [#uses=1 type=i64] [debug line = 185:5]
  %mem.addr.2 = getelementptr inbounds [256 x i8]* %mem, i64 0, i64 %tmp.32, !dbg !144 ; [#uses=1 type=i8*] [debug line = 185:5]
  %mem.load.1 = load i8* %mem.addr.2, align 1, !dbg !144 ; [#uses=2 type=i8] [debug line = 185:5]
  call void (...)* @_ssdm_SpecKeepArrayLoad(i8 %mem.load.1) nounwind
  store volatile i8 %mem.load.1, i8* @dout1, align 1, !dbg !144 ; [debug line = 185:5]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !145 ; [debug line = 186:5]
  br label %24, !dbg !146                         ; [debug line = 187:5]

; <label>:24                                      ; preds = %25, %23
  %r_req1.load.3 = load volatile i1* @r_req1, align 1, !dbg !146 ; [#uses=1 type=i1] [debug line = 187:5]
  br i1 %r_req1.load.3, label %25, label %26, !dbg !146 ; [debug line = 187:5]

; <label>:25                                      ; preds = %24
  store volatile i1 true, i1* @r_ack1, align 1, !dbg !147 ; [debug line = 188:6]
  br label %24, !dbg !149                         ; [debug line = 189:5]

; <label>:26                                      ; preds = %24
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !150 ; [debug line = 190:5]
  store volatile i1 false, i1* @r_ack1, align 1, !dbg !151 ; [debug line = 191:5]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !152 ; [debug line = 192:5]
  %rend21 = call i32 (...)* @_ssdm_op_SpecRegionEnd(i8* getelementptr inbounds ([12 x i8]* @.str4, i64 0, i64 0), i32 %rbegin2) nounwind, !dbg !153 ; [#uses=0 type=i32] [debug line = 193:4]
  br label %32, !dbg !153                         ; [debug line = 193:4]

; <label>:27                                      ; preds = %22
  %w_req1.load.1 = load volatile i1* @w_req1, align 1, !dbg !154 ; [#uses=1 type=i1] [debug line = 194:9]
  br i1 %w_req1.load.1, label %28, label %._crit_edge9, !dbg !154 ; [debug line = 194:9]

; <label>:28                                      ; preds = %27
  %rbegin6 = call i32 (...)* @_ssdm_op_SpecRegionBegin(i8* getelementptr inbounds ([12 x i8]* @.str5, i64 0, i64 0)) nounwind, !dbg !155 ; [#uses=1 type=i32] [debug line = 194:27]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !157 ; [debug line = 195:5]
  %din1.load.1 = load volatile i8* @din1, align 1, !dbg !158 ; [#uses=1 type=i8] [debug line = 196:5]
  %addr1.load.2 = load volatile i8* @addr1, align 1, !dbg !158 ; [#uses=1 type=i8] [debug line = 196:5]
  %tmp.36 = zext i8 %addr1.load.2 to i64, !dbg !158 ; [#uses=1 type=i64] [debug line = 196:5]
  %mem.addr.6 = getelementptr inbounds [256 x i8]* %mem, i64 0, i64 %tmp.36, !dbg !158 ; [#uses=1 type=i8*] [debug line = 196:5]
  store i8 %din1.load.1, i8* %mem.addr.6, align 1, !dbg !158 ; [debug line = 196:5]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !159 ; [debug line = 197:5]
  br label %29, !dbg !160                         ; [debug line = 198:5]

; <label>:29                                      ; preds = %30, %28
  %w_req1.load.3 = load volatile i1* @w_req1, align 1, !dbg !160 ; [#uses=1 type=i1] [debug line = 198:5]
  br i1 %w_req1.load.3, label %30, label %31, !dbg !160 ; [debug line = 198:5]

; <label>:30                                      ; preds = %29
  store volatile i1 true, i1* @w_ack1, align 1, !dbg !161 ; [debug line = 199:6]
  br label %29, !dbg !163                         ; [debug line = 200:5]

; <label>:31                                      ; preds = %29
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !164 ; [debug line = 201:5]
  store volatile i1 false, i1* @w_ack1, align 1, !dbg !165 ; [debug line = 202:5]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !166 ; [debug line = 203:5]
  %rend15 = call i32 (...)* @_ssdm_op_SpecRegionEnd(i8* getelementptr inbounds ([12 x i8]* @.str5, i64 0, i64 0), i32 %rbegin6) nounwind, !dbg !167 ; [#uses=0 type=i32] [debug line = 204:4]
  br label %._crit_edge9, !dbg !167               ; [debug line = 204:4]

._crit_edge9:                                     ; preds = %31, %27
  br label %32

; <label>:32                                      ; preds = %._crit_edge9, %26
  br label %.preheader.backedge, !dbg !168        ; [debug line = 206:4]

; <label>:33                                      ; preds = %.preheader
  %r_req2.load.1 = load volatile i1* @r_req2, align 1, !dbg !169 ; [#uses=1 type=i1] [debug line = 209:4]
  br i1 %r_req2.load.1, label %34, label %38, !dbg !169 ; [debug line = 209:4]

; <label>:34                                      ; preds = %33
  %rbegin3 = call i32 (...)* @_ssdm_op_SpecRegionBegin(i8* getelementptr inbounds ([12 x i8]* @.str6, i64 0, i64 0)) nounwind, !dbg !170 ; [#uses=1 type=i32] [debug line = 209:22]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !172 ; [debug line = 210:5]
  %addr2.load.1 = load volatile i8* @addr2, align 1, !dbg !173 ; [#uses=1 type=i8] [debug line = 211:5]
  %tmp.33 = zext i8 %addr2.load.1 to i64, !dbg !173 ; [#uses=1 type=i64] [debug line = 211:5]
  %mem.addr.3 = getelementptr inbounds [256 x i8]* %mem, i64 0, i64 %tmp.33, !dbg !173 ; [#uses=1 type=i8*] [debug line = 211:5]
  %mem.load.2 = load i8* %mem.addr.3, align 1, !dbg !173 ; [#uses=2 type=i8] [debug line = 211:5]
  call void (...)* @_ssdm_SpecKeepArrayLoad(i8 %mem.load.2) nounwind
  store volatile i8 %mem.load.2, i8* @dout2, align 1, !dbg !173 ; [debug line = 211:5]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !174 ; [debug line = 212:5]
  br label %35, !dbg !175                         ; [debug line = 213:5]

; <label>:35                                      ; preds = %36, %34
  %r_req2.load.2 = load volatile i1* @r_req2, align 1, !dbg !175 ; [#uses=1 type=i1] [debug line = 213:5]
  br i1 %r_req2.load.2, label %36, label %37, !dbg !175 ; [debug line = 213:5]

; <label>:36                                      ; preds = %35
  store volatile i1 true, i1* @r_ack2, align 1, !dbg !176 ; [debug line = 214:6]
  br label %35, !dbg !178                         ; [debug line = 215:5]

; <label>:37                                      ; preds = %35
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !179 ; [debug line = 216:5]
  store volatile i1 false, i1* @r_ack2, align 1, !dbg !180 ; [debug line = 217:5]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !181 ; [debug line = 218:5]
  %rend17 = call i32 (...)* @_ssdm_op_SpecRegionEnd(i8* getelementptr inbounds ([12 x i8]* @.str6, i64 0, i64 0), i32 %rbegin3) nounwind, !dbg !182 ; [#uses=0 type=i32] [debug line = 219:4]
  br label %43, !dbg !182                         ; [debug line = 219:4]

; <label>:38                                      ; preds = %33
  %w_req2.load.1 = load volatile i1* @w_req2, align 1, !dbg !183 ; [#uses=1 type=i1] [debug line = 220:9]
  br i1 %w_req2.load.1, label %39, label %._crit_edge10, !dbg !183 ; [debug line = 220:9]

; <label>:39                                      ; preds = %38
  %rbegin = call i32 (...)* @_ssdm_op_SpecRegionBegin(i8* getelementptr inbounds ([12 x i8]* @.str7, i64 0, i64 0)) nounwind, !dbg !184 ; [#uses=1 type=i32] [debug line = 220:27]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !186 ; [debug line = 221:5]
  %din2.load.1 = load volatile i8* @din2, align 1, !dbg !187 ; [#uses=1 type=i8] [debug line = 222:5]
  %addr2.load.2 = load volatile i8* @addr2, align 1, !dbg !187 ; [#uses=1 type=i8] [debug line = 222:5]
  %tmp.37 = zext i8 %addr2.load.2 to i64, !dbg !187 ; [#uses=1 type=i64] [debug line = 222:5]
  %mem.addr.7 = getelementptr inbounds [256 x i8]* %mem, i64 0, i64 %tmp.37, !dbg !187 ; [#uses=1 type=i8*] [debug line = 222:5]
  store i8 %din2.load.1, i8* %mem.addr.7, align 1, !dbg !187 ; [debug line = 222:5]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !188 ; [debug line = 223:5]
  br label %40, !dbg !189                         ; [debug line = 224:5]

; <label>:40                                      ; preds = %41, %39
  %w_req2.load.2 = load volatile i1* @w_req2, align 1, !dbg !189 ; [#uses=1 type=i1] [debug line = 224:5]
  br i1 %w_req2.load.2, label %41, label %42, !dbg !189 ; [debug line = 224:5]

; <label>:41                                      ; preds = %40
  store volatile i1 true, i1* @w_ack2, align 1, !dbg !190 ; [debug line = 225:6]
  br label %40, !dbg !192                         ; [debug line = 226:5]

; <label>:42                                      ; preds = %40
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !193 ; [debug line = 227:5]
  store volatile i1 false, i1* @w_ack2, align 1, !dbg !194 ; [debug line = 228:5]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !195 ; [debug line = 229:5]
  %rend = call i32 (...)* @_ssdm_op_SpecRegionEnd(i8* getelementptr inbounds ([12 x i8]* @.str7, i64 0, i64 0), i32 %rbegin) nounwind, !dbg !196 ; [#uses=0 type=i32] [debug line = 230:4]
  br label %._crit_edge10, !dbg !196              ; [debug line = 230:4]

._crit_edge10:                                    ; preds = %42, %38
  br label %43

; <label>:43                                      ; preds = %._crit_edge10, %37
  br label %.preheader.backedge, !dbg !197        ; [debug line = 232:4]

; <label>:44                                      ; preds = %.preheader
  %r_req3.load.1 = load volatile i1* @r_req3, align 1, !dbg !198 ; [#uses=1 type=i1] [debug line = 235:4]
  br i1 %r_req3.load.1, label %45, label %49, !dbg !198 ; [debug line = 235:4]

; <label>:45                                      ; preds = %44
  %rbegin4 = call i32 (...)* @_ssdm_op_SpecRegionBegin(i8* getelementptr inbounds ([12 x i8]* @.str8, i64 0, i64 0)) nounwind, !dbg !199 ; [#uses=1 type=i32] [debug line = 235:22]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !201 ; [debug line = 236:5]
  %addr3.load.1 = load volatile i8* @addr3, align 1, !dbg !202 ; [#uses=1 type=i8] [debug line = 237:5]
  %tmp.34 = zext i8 %addr3.load.1 to i64, !dbg !202 ; [#uses=1 type=i64] [debug line = 237:5]
  %mem.addr.4 = getelementptr inbounds [256 x i8]* %mem, i64 0, i64 %tmp.34, !dbg !202 ; [#uses=1 type=i8*] [debug line = 237:5]
  %mem.load.3 = load i8* %mem.addr.4, align 1, !dbg !202 ; [#uses=2 type=i8] [debug line = 237:5]
  call void (...)* @_ssdm_SpecKeepArrayLoad(i8 %mem.load.3) nounwind
  store volatile i8 %mem.load.3, i8* @dout3, align 1, !dbg !202 ; [debug line = 237:5]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !203 ; [debug line = 238:5]
  br label %46, !dbg !204                         ; [debug line = 239:5]

; <label>:46                                      ; preds = %47, %45
  %r_req3.load.2 = load volatile i1* @r_req3, align 1, !dbg !204 ; [#uses=1 type=i1] [debug line = 239:5]
  br i1 %r_req3.load.2, label %47, label %48, !dbg !204 ; [debug line = 239:5]

; <label>:47                                      ; preds = %46
  store volatile i1 true, i1* @r_ack3, align 1, !dbg !205 ; [debug line = 240:6]
  br label %46, !dbg !207                         ; [debug line = 241:5]

; <label>:48                                      ; preds = %46
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !208 ; [debug line = 242:5]
  store volatile i1 false, i1* @r_ack3, align 1, !dbg !209 ; [debug line = 243:5]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !210 ; [debug line = 244:5]
  %rend25 = call i32 (...)* @_ssdm_op_SpecRegionEnd(i8* getelementptr inbounds ([12 x i8]* @.str8, i64 0, i64 0), i32 %rbegin4) nounwind, !dbg !211 ; [#uses=0 type=i32] [debug line = 245:4]
  br label %54, !dbg !211                         ; [debug line = 245:4]

; <label>:49                                      ; preds = %44
  %w_req3.load.1 = load volatile i1* @w_req3, align 1, !dbg !212 ; [#uses=1 type=i1] [debug line = 246:9]
  br i1 %w_req3.load.1, label %50, label %._crit_edge11, !dbg !212 ; [debug line = 246:9]

; <label>:50                                      ; preds = %49
  %rbegin7 = call i32 (...)* @_ssdm_op_SpecRegionBegin(i8* getelementptr inbounds ([12 x i8]* @.str9, i64 0, i64 0)) nounwind, !dbg !213 ; [#uses=1 type=i32] [debug line = 246:27]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !215 ; [debug line = 247:5]
  %din3.load.1 = load volatile i8* @din3, align 1, !dbg !216 ; [#uses=1 type=i8] [debug line = 248:5]
  %addr3.load.2 = load volatile i8* @addr3, align 1, !dbg !216 ; [#uses=1 type=i8] [debug line = 248:5]
  %tmp.38 = zext i8 %addr3.load.2 to i64, !dbg !216 ; [#uses=1 type=i64] [debug line = 248:5]
  %mem.addr.8 = getelementptr inbounds [256 x i8]* %mem, i64 0, i64 %tmp.38, !dbg !216 ; [#uses=1 type=i8*] [debug line = 248:5]
  store i8 %din3.load.1, i8* %mem.addr.8, align 1, !dbg !216 ; [debug line = 248:5]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !217 ; [debug line = 249:5]
  br label %51, !dbg !218                         ; [debug line = 250:5]

; <label>:51                                      ; preds = %52, %50
  %w_req3.load.2 = load volatile i1* @w_req3, align 1, !dbg !218 ; [#uses=1 type=i1] [debug line = 250:5]
  br i1 %w_req3.load.2, label %52, label %53, !dbg !218 ; [debug line = 250:5]

; <label>:52                                      ; preds = %51
  store volatile i1 true, i1* @w_ack3, align 1, !dbg !219 ; [debug line = 251:6]
  br label %51, !dbg !221                         ; [debug line = 252:5]

; <label>:53                                      ; preds = %51
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !222 ; [debug line = 253:5]
  store volatile i1 false, i1* @w_ack3, align 1, !dbg !223 ; [debug line = 254:5]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !224 ; [debug line = 255:5]
  %rend19 = call i32 (...)* @_ssdm_op_SpecRegionEnd(i8* getelementptr inbounds ([12 x i8]* @.str9, i64 0, i64 0), i32 %rbegin7) nounwind, !dbg !225 ; [#uses=0 type=i32] [debug line = 256:4]
  br label %._crit_edge11, !dbg !225              ; [debug line = 256:4]

._crit_edge11:                                    ; preds = %53, %49
  br label %54

; <label>:54                                      ; preds = %._crit_edge11, %48
  br label %.preheader.backedge, !dbg !226        ; [debug line = 258:4]

; <label>:55                                      ; preds = %.preheader
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !227 ; [debug line = 261:4]
  store volatile i1 false, i1* @r_ack0, align 1, !dbg !228 ; [debug line = 262:4]
  store volatile i1 false, i1* @w_ack0, align 1, !dbg !229 ; [debug line = 263:4]
  store volatile i1 false, i1* @r_ack1, align 1, !dbg !230 ; [debug line = 264:4]
  store volatile i1 false, i1* @w_ack1, align 1, !dbg !231 ; [debug line = 265:4]
  store volatile i1 false, i1* @r_ack2, align 1, !dbg !232 ; [debug line = 266:4]
  store volatile i1 false, i1* @w_ack2, align 1, !dbg !233 ; [debug line = 267:4]
  store volatile i1 false, i1* @r_ack3, align 1, !dbg !234 ; [debug line = 268:4]
  store volatile i1 false, i1* @w_ack3, align 1, !dbg !235 ; [debug line = 269:4]
  call void (...)* @_ssdm_op_Wait(i32 1), !dbg !236 ; [debug line = 270:4]
  br label %.preheader.backedge, !dbg !237        ; [debug line = 273:4]
}

; [#uses=1]
declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

; [#uses=1]
declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

; [#uses=36]
declare void @_ssdm_op_Wait(...) nounwind

; [#uses=1]
declare void @_ssdm_op_SpecTopModule(...) nounwind

; [#uses=8]
declare i32 @_ssdm_op_SpecRegionEnd(...)

; [#uses=8]
declare i32 @_ssdm_op_SpecRegionBegin(...)

; [#uses=1]
declare i32 @_ssdm_op_SpecLoopBegin(...)

; [#uses=28]
declare void @_ssdm_op_SpecInterface(...) nounwind

; [#uses=4]
declare void @_ssdm_SpecKeepArrayLoad(...)

!llvm.dbg.cu = !{!0}
!hls.encrypted.func = !{}

!0 = metadata !{i32 786449, i32 0, i32 1, metadata !"D:/21_streamer_car5_artix7/fpga_arty/sharedmem/solution1/.autopilot/db/sharedmem.pragma.2.c", metadata !"D:\5C21_streamer_car5_artix7\5Cfpga_arty", metadata !"clang version 3.1 ", i1 true, i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !11} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !5}
!5 = metadata !{i32 786478, i32 0, metadata !6, metadata !"sharedmem", metadata !"sharedmem", metadata !"", metadata !6, i32 86, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 false, void ()* @sharedmem, null, null, metadata !9, i32 87} ; [ DW_TAG_subprogram ]
!6 = metadata !{i32 786473, metadata !"sharedmem.c", metadata !"D:\5C21_streamer_car5_artix7\5Cfpga_arty", null} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!8 = metadata !{null}
!9 = metadata !{metadata !10}
!10 = metadata !{i32 786468}                      ; [ DW_TAG_base_type ]
!11 = metadata !{metadata !12}
!12 = metadata !{metadata !13, metadata !17, metadata !18, metadata !22, metadata !23, metadata !24, metadata !25, metadata !26, metadata !27, metadata !28, metadata !29, metadata !30, metadata !31, metadata !32, metadata !33, metadata !34, metadata !35, metadata !36, metadata !37, metadata !38, metadata !39, metadata !40, metadata !41, metadata !42, metadata !43, metadata !44, metadata !45, metadata !46}
!13 = metadata !{i32 786484, i32 0, null, metadata !"addr0", metadata !"addr0", metadata !"", metadata !6, i32 47, metadata !14, i32 0, i32 1, i8* @addr0} ; [ DW_TAG_variable ]
!14 = metadata !{i32 786485, null, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !15} ; [ DW_TAG_volatile_type ]
!15 = metadata !{i32 786454, null, metadata !"uint8", metadata !6, i32 10, i64 0, i64 0, i64 0, i32 0, metadata !16} ; [ DW_TAG_typedef ]
!16 = metadata !{i32 786468, null, metadata !"uint8", null, i32 0, i64 8, i64 8, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!17 = metadata !{i32 786484, i32 0, null, metadata !"dout0", metadata !"dout0", metadata !"", metadata !6, i32 49, metadata !14, i32 0, i32 1, i8* @dout0} ; [ DW_TAG_variable ]
!18 = metadata !{i32 786484, i32 0, null, metadata !"r_ack0", metadata !"r_ack0", metadata !"", metadata !6, i32 51, metadata !19, i32 0, i32 1, i1* @r_ack0} ; [ DW_TAG_variable ]
!19 = metadata !{i32 786485, null, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !20} ; [ DW_TAG_volatile_type ]
!20 = metadata !{i32 786454, null, metadata !"uint1", metadata !6, i32 3, i64 0, i64 0, i64 0, i32 0, metadata !21} ; [ DW_TAG_typedef ]
!21 = metadata !{i32 786468, null, metadata !"uint1", null, i32 0, i64 1, i64 1, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!22 = metadata !{i32 786484, i32 0, null, metadata !"w_ack0", metadata !"w_ack0", metadata !"", metadata !6, i32 53, metadata !19, i32 0, i32 1, i1* @w_ack0} ; [ DW_TAG_variable ]
!23 = metadata !{i32 786484, i32 0, null, metadata !"addr1", metadata !"addr1", metadata !"", metadata !6, i32 56, metadata !14, i32 0, i32 1, i8* @addr1} ; [ DW_TAG_variable ]
!24 = metadata !{i32 786484, i32 0, null, metadata !"dout1", metadata !"dout1", metadata !"", metadata !6, i32 58, metadata !14, i32 0, i32 1, i8* @dout1} ; [ DW_TAG_variable ]
!25 = metadata !{i32 786484, i32 0, null, metadata !"r_ack1", metadata !"r_ack1", metadata !"", metadata !6, i32 60, metadata !19, i32 0, i32 1, i1* @r_ack1} ; [ DW_TAG_variable ]
!26 = metadata !{i32 786484, i32 0, null, metadata !"w_ack1", metadata !"w_ack1", metadata !"", metadata !6, i32 62, metadata !19, i32 0, i32 1, i1* @w_ack1} ; [ DW_TAG_variable ]
!27 = metadata !{i32 786484, i32 0, null, metadata !"addr2", metadata !"addr2", metadata !"", metadata !6, i32 65, metadata !14, i32 0, i32 1, i8* @addr2} ; [ DW_TAG_variable ]
!28 = metadata !{i32 786484, i32 0, null, metadata !"dout2", metadata !"dout2", metadata !"", metadata !6, i32 67, metadata !14, i32 0, i32 1, i8* @dout2} ; [ DW_TAG_variable ]
!29 = metadata !{i32 786484, i32 0, null, metadata !"r_ack2", metadata !"r_ack2", metadata !"", metadata !6, i32 69, metadata !19, i32 0, i32 1, i1* @r_ack2} ; [ DW_TAG_variable ]
!30 = metadata !{i32 786484, i32 0, null, metadata !"w_ack2", metadata !"w_ack2", metadata !"", metadata !6, i32 71, metadata !19, i32 0, i32 1, i1* @w_ack2} ; [ DW_TAG_variable ]
!31 = metadata !{i32 786484, i32 0, null, metadata !"addr3", metadata !"addr3", metadata !"", metadata !6, i32 74, metadata !14, i32 0, i32 1, i8* @addr3} ; [ DW_TAG_variable ]
!32 = metadata !{i32 786484, i32 0, null, metadata !"dout3", metadata !"dout3", metadata !"", metadata !6, i32 76, metadata !14, i32 0, i32 1, i8* @dout3} ; [ DW_TAG_variable ]
!33 = metadata !{i32 786484, i32 0, null, metadata !"r_ack3", metadata !"r_ack3", metadata !"", metadata !6, i32 78, metadata !19, i32 0, i32 1, i1* @r_ack3} ; [ DW_TAG_variable ]
!34 = metadata !{i32 786484, i32 0, null, metadata !"w_ack3", metadata !"w_ack3", metadata !"", metadata !6, i32 80, metadata !19, i32 0, i32 1, i1* @w_ack3} ; [ DW_TAG_variable ]
!35 = metadata !{i32 786484, i32 0, null, metadata !"din0", metadata !"din0", metadata !"", metadata !6, i32 48, metadata !14, i32 0, i32 1, i8* @din0} ; [ DW_TAG_variable ]
!36 = metadata !{i32 786484, i32 0, null, metadata !"r_req0", metadata !"r_req0", metadata !"", metadata !6, i32 50, metadata !19, i32 0, i32 1, i1* @r_req0} ; [ DW_TAG_variable ]
!37 = metadata !{i32 786484, i32 0, null, metadata !"w_req0", metadata !"w_req0", metadata !"", metadata !6, i32 52, metadata !19, i32 0, i32 1, i1* @w_req0} ; [ DW_TAG_variable ]
!38 = metadata !{i32 786484, i32 0, null, metadata !"din1", metadata !"din1", metadata !"", metadata !6, i32 57, metadata !14, i32 0, i32 1, i8* @din1} ; [ DW_TAG_variable ]
!39 = metadata !{i32 786484, i32 0, null, metadata !"r_req1", metadata !"r_req1", metadata !"", metadata !6, i32 59, metadata !19, i32 0, i32 1, i1* @r_req1} ; [ DW_TAG_variable ]
!40 = metadata !{i32 786484, i32 0, null, metadata !"w_req1", metadata !"w_req1", metadata !"", metadata !6, i32 61, metadata !19, i32 0, i32 1, i1* @w_req1} ; [ DW_TAG_variable ]
!41 = metadata !{i32 786484, i32 0, null, metadata !"din2", metadata !"din2", metadata !"", metadata !6, i32 66, metadata !14, i32 0, i32 1, i8* @din2} ; [ DW_TAG_variable ]
!42 = metadata !{i32 786484, i32 0, null, metadata !"r_req2", metadata !"r_req2", metadata !"", metadata !6, i32 68, metadata !19, i32 0, i32 1, i1* @r_req2} ; [ DW_TAG_variable ]
!43 = metadata !{i32 786484, i32 0, null, metadata !"w_req2", metadata !"w_req2", metadata !"", metadata !6, i32 70, metadata !19, i32 0, i32 1, i1* @w_req2} ; [ DW_TAG_variable ]
!44 = metadata !{i32 786484, i32 0, null, metadata !"din3", metadata !"din3", metadata !"", metadata !6, i32 75, metadata !14, i32 0, i32 1, i8* @din3} ; [ DW_TAG_variable ]
!45 = metadata !{i32 786484, i32 0, null, metadata !"r_req3", metadata !"r_req3", metadata !"", metadata !6, i32 77, metadata !19, i32 0, i32 1, i1* @r_req3} ; [ DW_TAG_variable ]
!46 = metadata !{i32 786484, i32 0, null, metadata !"w_req3", metadata !"w_req3", metadata !"", metadata !6, i32 79, metadata !19, i32 0, i32 1, i1* @w_req3} ; [ DW_TAG_variable ]
!47 = metadata !{i32 88, i32 1, metadata !48, null}
!48 = metadata !{i32 786443, metadata !5, i32 87, i32 1, metadata !6, i32 0} ; [ DW_TAG_lexical_block ]
!49 = metadata !{i32 90, i32 1, metadata !48, null}
!50 = metadata !{i32 91, i32 1, metadata !48, null}
!51 = metadata !{i32 92, i32 1, metadata !48, null}
!52 = metadata !{i32 93, i32 1, metadata !48, null}
!53 = metadata !{i32 94, i32 1, metadata !48, null}
!54 = metadata !{i32 95, i32 1, metadata !48, null}
!55 = metadata !{i32 96, i32 1, metadata !48, null}
!56 = metadata !{i32 98, i32 1, metadata !48, null}
!57 = metadata !{i32 99, i32 1, metadata !48, null}
!58 = metadata !{i32 100, i32 1, metadata !48, null}
!59 = metadata !{i32 101, i32 1, metadata !48, null}
!60 = metadata !{i32 102, i32 1, metadata !48, null}
!61 = metadata !{i32 103, i32 1, metadata !48, null}
!62 = metadata !{i32 104, i32 1, metadata !48, null}
!63 = metadata !{i32 106, i32 1, metadata !48, null}
!64 = metadata !{i32 107, i32 1, metadata !48, null}
!65 = metadata !{i32 108, i32 1, metadata !48, null}
!66 = metadata !{i32 109, i32 1, metadata !48, null}
!67 = metadata !{i32 110, i32 1, metadata !48, null}
!68 = metadata !{i32 111, i32 1, metadata !48, null}
!69 = metadata !{i32 112, i32 1, metadata !48, null}
!70 = metadata !{i32 114, i32 1, metadata !48, null}
!71 = metadata !{i32 115, i32 1, metadata !48, null}
!72 = metadata !{i32 116, i32 1, metadata !48, null}
!73 = metadata !{i32 117, i32 1, metadata !48, null}
!74 = metadata !{i32 118, i32 1, metadata !48, null}
!75 = metadata !{i32 119, i32 1, metadata !48, null}
!76 = metadata !{i32 120, i32 1, metadata !48, null}
!77 = metadata !{i32 786688, metadata !48, metadata !"mem", metadata !6, i32 124, metadata !78, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!78 = metadata !{i32 786433, null, metadata !"", null, i32 0, i64 2048, i64 8, i32 0, i32 0, metadata !15, metadata !79, i32 0, i32 0} ; [ DW_TAG_array_type ]
!79 = metadata !{metadata !80}
!80 = metadata !{i32 786465, i64 0, i64 255}      ; [ DW_TAG_subrange_type ]
!81 = metadata !{i32 124, i32 8, metadata !48, null}
!82 = metadata !{i32 127, i32 2, metadata !48, null}
!83 = metadata !{i32 128, i32 2, metadata !48, null}
!84 = metadata !{i32 129, i32 2, metadata !48, null}
!85 = metadata !{i32 130, i32 2, metadata !48, null}
!86 = metadata !{i32 131, i32 2, metadata !48, null}
!87 = metadata !{i32 132, i32 2, metadata !48, null}
!88 = metadata !{i32 133, i32 2, metadata !48, null}
!89 = metadata !{i32 134, i32 2, metadata !48, null}
!90 = metadata !{i32 135, i32 2, metadata !48, null}
!91 = metadata !{i32 136, i32 2, metadata !48, null}
!92 = metadata !{i32 138, i32 7, metadata !93, null}
!93 = metadata !{i32 786443, metadata !48, i32 138, i32 2, metadata !6, i32 1} ; [ DW_TAG_lexical_block ]
!94 = metadata !{i32 139, i32 3, metadata !95, null}
!95 = metadata !{i32 786443, metadata !93, i32 138, i32 37, metadata !6, i32 2} ; [ DW_TAG_lexical_block ]
!96 = metadata !{i32 138, i32 29, metadata !93, null}
!97 = metadata !{i32 786688, metadata !48, metadata !"addr", metadata !6, i32 125, metadata !98, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!98 = metadata !{i32 786454, null, metadata !"uint9", metadata !6, i32 11, i64 0, i64 0, i64 0, i32 0, metadata !99} ; [ DW_TAG_typedef ]
!99 = metadata !{i32 786468, null, metadata !"uint9", null, i32 0, i64 9, i64 16, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!100 = metadata !{i32 144, i32 3, metadata !101, null}
!101 = metadata !{i32 786443, metadata !48, i32 143, i32 12, metadata !6, i32 3} ; [ DW_TAG_lexical_block ]
!102 = metadata !{i32 146, i32 4, metadata !103, null}
!103 = metadata !{i32 786443, metadata !101, i32 144, i32 15, metadata !6, i32 4} ; [ DW_TAG_lexical_block ]
!104 = metadata !{i32 147, i32 5, metadata !103, null}
!105 = metadata !{i32 148, i32 9, metadata !103, null}
!106 = metadata !{i32 149, i32 5, metadata !103, null}
!107 = metadata !{i32 150, i32 9, metadata !103, null}
!108 = metadata !{i32 151, i32 5, metadata !103, null}
!109 = metadata !{i32 152, i32 9, metadata !103, null}
!110 = metadata !{i32 153, i32 5, metadata !103, null}
!111 = metadata !{i32 157, i32 4, metadata !103, null}
!112 = metadata !{i32 157, i32 22, metadata !113, null}
!113 = metadata !{i32 786443, metadata !103, i32 157, i32 21, metadata !6, i32 5} ; [ DW_TAG_lexical_block ]
!114 = metadata !{i32 158, i32 5, metadata !113, null}
!115 = metadata !{i32 159, i32 5, metadata !113, null}
!116 = metadata !{i32 160, i32 5, metadata !113, null}
!117 = metadata !{i32 161, i32 5, metadata !113, null}
!118 = metadata !{i32 162, i32 6, metadata !119, null}
!119 = metadata !{i32 786443, metadata !113, i32 161, i32 25, metadata !6, i32 6} ; [ DW_TAG_lexical_block ]
!120 = metadata !{i32 163, i32 5, metadata !119, null}
!121 = metadata !{i32 164, i32 5, metadata !113, null}
!122 = metadata !{i32 165, i32 5, metadata !113, null}
!123 = metadata !{i32 166, i32 5, metadata !113, null}
!124 = metadata !{i32 167, i32 4, metadata !113, null}
!125 = metadata !{i32 168, i32 9, metadata !103, null}
!126 = metadata !{i32 168, i32 27, metadata !127, null}
!127 = metadata !{i32 786443, metadata !103, i32 168, i32 26, metadata !6, i32 7} ; [ DW_TAG_lexical_block ]
!128 = metadata !{i32 169, i32 5, metadata !127, null}
!129 = metadata !{i32 170, i32 5, metadata !127, null}
!130 = metadata !{i32 171, i32 5, metadata !127, null}
!131 = metadata !{i32 172, i32 5, metadata !127, null}
!132 = metadata !{i32 173, i32 6, metadata !133, null}
!133 = metadata !{i32 786443, metadata !127, i32 172, i32 25, metadata !6, i32 8} ; [ DW_TAG_lexical_block ]
!134 = metadata !{i32 174, i32 5, metadata !133, null}
!135 = metadata !{i32 175, i32 5, metadata !127, null}
!136 = metadata !{i32 176, i32 5, metadata !127, null}
!137 = metadata !{i32 177, i32 5, metadata !127, null}
!138 = metadata !{i32 178, i32 4, metadata !127, null}
!139 = metadata !{i32 180, i32 4, metadata !103, null}
!140 = metadata !{i32 183, i32 4, metadata !103, null}
!141 = metadata !{i32 183, i32 22, metadata !142, null}
!142 = metadata !{i32 786443, metadata !103, i32 183, i32 21, metadata !6, i32 9} ; [ DW_TAG_lexical_block ]
!143 = metadata !{i32 184, i32 5, metadata !142, null}
!144 = metadata !{i32 185, i32 5, metadata !142, null}
!145 = metadata !{i32 186, i32 5, metadata !142, null}
!146 = metadata !{i32 187, i32 5, metadata !142, null}
!147 = metadata !{i32 188, i32 6, metadata !148, null}
!148 = metadata !{i32 786443, metadata !142, i32 187, i32 25, metadata !6, i32 10} ; [ DW_TAG_lexical_block ]
!149 = metadata !{i32 189, i32 5, metadata !148, null}
!150 = metadata !{i32 190, i32 5, metadata !142, null}
!151 = metadata !{i32 191, i32 5, metadata !142, null}
!152 = metadata !{i32 192, i32 5, metadata !142, null}
!153 = metadata !{i32 193, i32 4, metadata !142, null}
!154 = metadata !{i32 194, i32 9, metadata !103, null}
!155 = metadata !{i32 194, i32 27, metadata !156, null}
!156 = metadata !{i32 786443, metadata !103, i32 194, i32 26, metadata !6, i32 11} ; [ DW_TAG_lexical_block ]
!157 = metadata !{i32 195, i32 5, metadata !156, null}
!158 = metadata !{i32 196, i32 5, metadata !156, null}
!159 = metadata !{i32 197, i32 5, metadata !156, null}
!160 = metadata !{i32 198, i32 5, metadata !156, null}
!161 = metadata !{i32 199, i32 6, metadata !162, null}
!162 = metadata !{i32 786443, metadata !156, i32 198, i32 25, metadata !6, i32 12} ; [ DW_TAG_lexical_block ]
!163 = metadata !{i32 200, i32 5, metadata !162, null}
!164 = metadata !{i32 201, i32 5, metadata !156, null}
!165 = metadata !{i32 202, i32 5, metadata !156, null}
!166 = metadata !{i32 203, i32 5, metadata !156, null}
!167 = metadata !{i32 204, i32 4, metadata !156, null}
!168 = metadata !{i32 206, i32 4, metadata !103, null}
!169 = metadata !{i32 209, i32 4, metadata !103, null}
!170 = metadata !{i32 209, i32 22, metadata !171, null}
!171 = metadata !{i32 786443, metadata !103, i32 209, i32 21, metadata !6, i32 13} ; [ DW_TAG_lexical_block ]
!172 = metadata !{i32 210, i32 5, metadata !171, null}
!173 = metadata !{i32 211, i32 5, metadata !171, null}
!174 = metadata !{i32 212, i32 5, metadata !171, null}
!175 = metadata !{i32 213, i32 5, metadata !171, null}
!176 = metadata !{i32 214, i32 6, metadata !177, null}
!177 = metadata !{i32 786443, metadata !171, i32 213, i32 25, metadata !6, i32 14} ; [ DW_TAG_lexical_block ]
!178 = metadata !{i32 215, i32 5, metadata !177, null}
!179 = metadata !{i32 216, i32 5, metadata !171, null}
!180 = metadata !{i32 217, i32 5, metadata !171, null}
!181 = metadata !{i32 218, i32 5, metadata !171, null}
!182 = metadata !{i32 219, i32 4, metadata !171, null}
!183 = metadata !{i32 220, i32 9, metadata !103, null}
!184 = metadata !{i32 220, i32 27, metadata !185, null}
!185 = metadata !{i32 786443, metadata !103, i32 220, i32 26, metadata !6, i32 15} ; [ DW_TAG_lexical_block ]
!186 = metadata !{i32 221, i32 5, metadata !185, null}
!187 = metadata !{i32 222, i32 5, metadata !185, null}
!188 = metadata !{i32 223, i32 5, metadata !185, null}
!189 = metadata !{i32 224, i32 5, metadata !185, null}
!190 = metadata !{i32 225, i32 6, metadata !191, null}
!191 = metadata !{i32 786443, metadata !185, i32 224, i32 25, metadata !6, i32 16} ; [ DW_TAG_lexical_block ]
!192 = metadata !{i32 226, i32 5, metadata !191, null}
!193 = metadata !{i32 227, i32 5, metadata !185, null}
!194 = metadata !{i32 228, i32 5, metadata !185, null}
!195 = metadata !{i32 229, i32 5, metadata !185, null}
!196 = metadata !{i32 230, i32 4, metadata !185, null}
!197 = metadata !{i32 232, i32 4, metadata !103, null}
!198 = metadata !{i32 235, i32 4, metadata !103, null}
!199 = metadata !{i32 235, i32 22, metadata !200, null}
!200 = metadata !{i32 786443, metadata !103, i32 235, i32 21, metadata !6, i32 17} ; [ DW_TAG_lexical_block ]
!201 = metadata !{i32 236, i32 5, metadata !200, null}
!202 = metadata !{i32 237, i32 5, metadata !200, null}
!203 = metadata !{i32 238, i32 5, metadata !200, null}
!204 = metadata !{i32 239, i32 5, metadata !200, null}
!205 = metadata !{i32 240, i32 6, metadata !206, null}
!206 = metadata !{i32 786443, metadata !200, i32 239, i32 25, metadata !6, i32 18} ; [ DW_TAG_lexical_block ]
!207 = metadata !{i32 241, i32 5, metadata !206, null}
!208 = metadata !{i32 242, i32 5, metadata !200, null}
!209 = metadata !{i32 243, i32 5, metadata !200, null}
!210 = metadata !{i32 244, i32 5, metadata !200, null}
!211 = metadata !{i32 245, i32 4, metadata !200, null}
!212 = metadata !{i32 246, i32 9, metadata !103, null}
!213 = metadata !{i32 246, i32 27, metadata !214, null}
!214 = metadata !{i32 786443, metadata !103, i32 246, i32 26, metadata !6, i32 19} ; [ DW_TAG_lexical_block ]
!215 = metadata !{i32 247, i32 5, metadata !214, null}
!216 = metadata !{i32 248, i32 5, metadata !214, null}
!217 = metadata !{i32 249, i32 5, metadata !214, null}
!218 = metadata !{i32 250, i32 5, metadata !214, null}
!219 = metadata !{i32 251, i32 6, metadata !220, null}
!220 = metadata !{i32 786443, metadata !214, i32 250, i32 25, metadata !6, i32 20} ; [ DW_TAG_lexical_block ]
!221 = metadata !{i32 252, i32 5, metadata !220, null}
!222 = metadata !{i32 253, i32 5, metadata !214, null}
!223 = metadata !{i32 254, i32 5, metadata !214, null}
!224 = metadata !{i32 255, i32 5, metadata !214, null}
!225 = metadata !{i32 256, i32 4, metadata !214, null}
!226 = metadata !{i32 258, i32 4, metadata !103, null}
!227 = metadata !{i32 261, i32 4, metadata !103, null}
!228 = metadata !{i32 262, i32 4, metadata !103, null}
!229 = metadata !{i32 263, i32 4, metadata !103, null}
!230 = metadata !{i32 264, i32 4, metadata !103, null}
!231 = metadata !{i32 265, i32 4, metadata !103, null}
!232 = metadata !{i32 266, i32 4, metadata !103, null}
!233 = metadata !{i32 267, i32 4, metadata !103, null}
!234 = metadata !{i32 268, i32 4, metadata !103, null}
!235 = metadata !{i32 269, i32 4, metadata !103, null}
!236 = metadata !{i32 270, i32 4, metadata !103, null}
!237 = metadata !{i32 273, i32 4, metadata !103, null}
