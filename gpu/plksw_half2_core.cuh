// =============================================================================
// plksw_half2_core.cuh  —  half2 2-way 核心递推(参考实现,需集成 + 对拍)
// -----------------------------------------------------------------------------
// 用途:把 plksw_kernel.cuh 前向 batch 循环里"一个 lane 算 1 个 int8 cell"
//       改成"一个 lane 用 half2 同时算 2 个相邻 cell(2k, 2k+1)"。
//
// 设计原则(保真优先):
//   * 值路径(6 个 delta)全程 half2 packed —— 性能红利在此,寄存器不翻倍。
//   * 方向字节 d 仍用你原来的标量逻辑,对 low/high 两个 half 各跑一遍 ——
//     构造上逐位等于原 int8 代码,不碰高风险的 packed 谓词抽取。
//   * 全局存储仍是 int8(load 2 个 int8 → 寄存器里 pack → 算 → 拆回 2 个 int8 存)。
//     => plmem.cu / plalign.cu / map.c 一律不动,只动 kernel。
//   * Phase B(H 重建 + argmax + Z-drop)读的还是 int8 v_arr,完全不变。
//
// ⚠ 必须对拍验证(我无法在此编译/运行):
//   1. left_align(!right_align)与 right_align 两种模式都要和 CPU ksw2_extd2_sse
//      逐位对齐(尤其 d 的 tie-break)。
//   2. 确认你的罚分集下没有任何中间量越过 int8 的 ±127(FP16 在 [-2048,2048]
//      精确表示整数,越界才会和 int8 分叉)。正常 DNA 罚分不会越界。
//   3. sc_mch / gap 常量广播成 half2 时按真值放(都是小整数,FP16 精确)。
// =============================================================================

#include <cuda_fp16.h>

// ---- int8 值转 half(真值,精确)----
__device__ __forceinline__ __half i8_to_h(int8_t v) { return __short2half_rn((short)v); }
// ---- half 转回 int8(round-to-nearest;值都是小整数,精确)----
__device__ __forceinline__ int8_t h_to_i8(__half v) { return (int8_t)__half2short_rn(v); }

// =============================================================================
// 方向字节:逐位复刻 plksw_kernel.cuh 原标量逻辑(366-470 行)。
// 输入是某一个 cell 的标量值(从 half2 抽出来的那一半)。
// score, a0,b0,a20,b20 = 选 max 之前的候选(int8 真值)。
// 注意:continuation 位用的是 *减法之后* 的 a,b,a2,b2(= 原始候选 - (z-q))。
// =============================================================================
__device__ __forceinline__ uint8_t dp_dir_byte_scalar(
        int z, int a0, int b0, int a20, int b20,           // max 前候选(score 已在 z 内)
        int q, int q2, bool right_align)
{
    uint8_t d = 0;

    // ---- bits 0..2:谁赢(顺序优先级 + tie-break)----
    if (!right_align) {                  // 严格 > ,最左赢
        if (a0  > z) { z = a0;  d = 1; }
        if (b0  > z) { z = b0;  d = 2; }
        if (a20 > z) { z = a20; d = 3; }
        if (b20 > z) { z = b20; d = 4; }
    } else {                             // !(z>x) 即 x>=z,平局给后者(右对齐)
        if (!(z > a0))  { z = a0;  d = 1; }
        if (!(z > b0))  { z = b0;  d = 2; }
        if (!(z > a20)) { z = a20; d = 3; }
        if (!(z > b20)) { z = b20; d = 4; }
    }
    // z 现在 == 选中的最大值(clamp 不影响 d 的判定,与原码一致:
    // 原码先选 d 再 clamp,clamp 只改值不改 d)

    // ---- bits 3..6:gap 续延位(用减法后的候选符号)----
    int aa  = a0  - (z - q);
    int bb  = b0  - (z - q);
    int aa2 = a20 - (z - q2);
    int bb2 = b20 - (z - q2);
    if (!right_align) {
        if (aa  > 0) d |= 0x08;
        if (bb  > 0) d |= 0x10;
        if (aa2 > 0) d |= 0x20;
        if (bb2 > 0) d |= 0x40;
    } else {
        if (!(0 > aa))  d |= 0x08;   // aa >= 0
        if (!(0 > bb))  d |= 0x10;
        if (!(0 > aa2)) d |= 0x20;
        if (!(0 > bb2)) d |= 0x40;
    }
    return d;
}

// =============================================================================
// 核心:一个 lane 用 half2 同时算 2 个 cell 的 6 个 delta(值路径打包)。
// 入参 half2 里:.x(low)= cell 2k,.y(high)= cell 2k+1。
//   xv1 = {x1, x1'}  vv1 = {v1, v1'}  x2v1 = {x21, x21'}
//   uup = {u_prev,*} yyp = {y_prev,*} y2yp = {y2_prev,*}
//   ssc = {score, score'}
// 常量 half2(本地广播一次):gO={q,q} gOL={q2,q2} gOE={qe,qe} gOEL={qe2,qe2}
//   negOE={-qe,-qe} negOEL={-qe2,-qe2} scMch={sc_mch,sc_mch}
// 出参:6 个 delta 的 half2(每个 .x/.y 是 2 个 cell),以及 2 个方向字节 d0/d1。
// =============================================================================
__device__ __forceinline__ void dp_cell_pair_half2(
        half2 xv1, half2 vv1, half2 x2v1,
        half2 uup, half2 yyp, half2 y2yp,
        half2 ssc,
        half2 gO, half2 gOL, half2 gOE, half2 gOEL,
        half2 negOE, half2 negOEL, half2 scMch,
        int q, int q2, bool with_cigar, bool right_align,
        // outputs:
        half2 *new_u, half2 *new_v, half2 *new_x, half2 *new_y,
        half2 *new_x2, half2 *new_y2,
        uint8_t *d0, uint8_t *d1)
{
    // ---- 候选(选 max 之前)----
    half2 z  = ssc;
    half2 a  = __hadd2(xv1,  vv1);    // 短 E
    half2 b  = __hadd2(yyp,  uup);    // 短 F
    half2 a2 = __hadd2(x2v1, vv1);    // 长 E
    half2 b2 = __hadd2(y2yp, uup);    // 长 F

    // ---- 方向字节:在打包前抽出标量,用原逻辑各算一遍(保真)----
    if (with_cigar) {
        *d0 = dp_dir_byte_scalar(
                (int)__half2short_rn(__low2half(ssc)),
                (int)__half2short_rn(__low2half(a)),
                (int)__half2short_rn(__low2half(b)),
                (int)__half2short_rn(__low2half(a2)),
                (int)__half2short_rn(__low2half(b2)),
                q, q2, right_align);
        *d1 = dp_dir_byte_scalar(
                (int)__half2short_rn(__high2half(ssc)),
                (int)__half2short_rn(__high2half(a)),
                (int)__half2short_rn(__high2half(b)),
                (int)__half2short_rn(__high2half(a2)),
                (int)__half2short_rn(__high2half(b2)),
                q, q2, right_align);
    } else {
        *d0 = *d1 = 0;
    }

    // ---- 选 max(值路径,packed)----
    z = __hmax2(z, a);
    z = __hmax2(z, b);
    z = __hmax2(z, a2);
    z = __hmax2(z, b2);
    z = __hmin2(z, scMch);             // clamp 到 sc_mch

    // ---- 两个 H 分差 ----
    *new_u = __hsub2(z, vv1);
    *new_v = __hsub2(z, uup);

    // ---- gap delta:a -= (z-q) 等,再 new_x = max(a-qe, -qe) ----
    half2 zq  = __hsub2(z, gO);
    half2 zq2 = __hsub2(z, gOL);
    a  = __hsub2(a,  zq);
    b  = __hsub2(b,  zq);
    a2 = __hsub2(a2, zq2);
    b2 = __hsub2(b2, zq2);

    // new_x=(a>0)?a-qe:-qe  ==  max(a-qe,-qe)  (两种 align 模式值相同)
    *new_x  = __hmax2(__hsub2(a,  gOE),  negOE);
    *new_y  = __hmax2(__hsub2(b,  gOE),  negOE);
    *new_x2 = __hmax2(__hsub2(a2, gOEL), negOEL);
    *new_y2 = __hmax2(__hsub2(b2, gOEL), negOEL);
}

// =============================================================================
// 集成清单(改 plksw_kernel.cuh 的前向 batch 循环,~349-470 行):
//
// [1] 常量广播(进 r 循环前一次):
//     half2 gO=__half2half2(i8_to_h(q)), gOL=..., gOE=..., gOEL=...,
//           negOE=__half2half2(i8_to_h(-qe)), negOEL=..., scMch=__half2half2(i8_to_h(sc_mch));
//
// [2] batch 循环 stride: WARP_SIZE -> WARP_SIZE*2(一次推进 64 个 cell)。
//     每 lane 负责 cell idx0=batch_start+2*lane_id, idx1=idx0+1。
//     active0 = idx0<band_size; active1 = idx1<band_size;  // tail 处理两个
//
// [3] 读 6 个 int8 邻居,pack 成 half2:
//     low  = cell idx0 的 (x1,v1,x21,u_prev,y_prev,y2_prev)
//     high = cell idx1 的 同上
//     cell idx0 左邻居 = idx0-1:lane_id==0 用 batch_x1_boundary,否则 x_arr[t0-1];
//     cell idx1 左邻居 = idx0(同 lane 的 low cell)的 *旧* x/v/x2 ——
//        ⚠ 因为"读全在写前",直接读 x_arr[t0] 拿到的就是旧值,正确。
//     xv1 = __halves2half2(i8_to_h(x1_lo), i8_to_h(x1_hi)); 其余同理。
//     score: ssc = __halves2half2(i8_to_h(score_lo), i8_to_h(score_hi));
//       (score 仍用 dp_compute_score 各算一次,或查表;它不在打包关键路径)
//
// [4] 调 dp_cell_pair_half2(...) 拿 6 个 new_* half2 + d0,d1。
//
// [5] 拆回 int8 存(保持 int8 存储,Phase B 不变):
//     if(active0){ u_arr[t0]=h_to_i8(__low2half(*new_u)); ... ; pr[t0-st0]=d0; }
//     if(active1){ u_arr[t1]=h_to_i8(__high2half(*new_u)); ...; pr[t1-st0]=d1; }
//
// [6] 边界 shuffle:把本 64-batch 最后一个有效 cell 的 *旧* x/v/x2(int8)
//     shuffle 给下一 batch 的 lane0。逻辑同原码,只是 last_lane 现在对应
//     idx=batch_start+63 那个 cell(注意它是某 lane 的 high half)。
//     建议:仍 shuffle int8(不是 half2),保持和原码一致最稳。
//
// [7] Phase B(484 行后 H 重建+argmax+Z-drop):**一行都不用改**
//     ——它读 int8 v_arr/u_arr,而你存储仍是 int8。
//
// 验证步骤:
//   a) 先关 cigar 跑(只对分数 + Z-drop),确认值路径对。
//   b) 开 cigar,left_align 对拍 CIGAR;再 right_align 对拍。
//   c) 用 ncu 比 1-cell vs half2 的 寄存器数 / achieved occupancy / runtime。
//      half2 预期:寄存器 ~不变甚至降、occupancy 不掉、段从 32→64。
// =============================================================================
