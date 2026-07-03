// =============================================================================
// plksw_dpx_core.cuh — Hopper DPX 2-cell-per-lane DP core.
//
// Uses the SM_90 Dynamic Programming X (DPX) intrinsics — most notably
//   __viaddmax_s16x2(a,b,c) = max(a+b, c) per int16 lane, one instruction
// which replaces the HALF2 path's separate "packed add" + "packed max" pair
// in the Suzuki–Kasahara recurrence.
//
// Values are packed as int16x2 in `unsigned int` (low 16 bits = cell 2k,
// high 16 bits = cell 2k+1) — the DPX intrinsic ABI.  Global memory stays
// int8; convert only in registers.
//
// Requires:  GPUARCH=sm_90+  (H100),  CUDA >= 12.0
// =============================================================================
#ifndef __PLKSW_DPX_CORE_CUH__
#define __PLKSW_DPX_CORE_CUH__

#include <cuda_runtime.h>

// ---- int16x2 pack / unpack (DPX ABI: lo = bits 0..15, hi = bits 16..31) ----
__device__ __forceinline__ unsigned dpx_pack(short lo, short hi) {
    return (((unsigned)(uint16_t)hi) << 16) | (unsigned)(uint16_t)lo;
}
__device__ __forceinline__ short dpx_lo(unsigned p) { return (short)(p & 0xffff); }
__device__ __forceinline__ short dpx_hi(unsigned p) { return (short)((p >> 16) & 0xffff); }
__device__ __forceinline__ unsigned dpx_bcast(short v) { return dpx_pack(v, v); }
__device__ __forceinline__ unsigned dpx_from_i8x2(int8_t lo, int8_t hi) {
    return dpx_pack((short)lo, (short)hi);
}
__device__ __forceinline__ int8_t dpx_to_i8_lo(unsigned p) { return (int8_t)dpx_lo(p); }
__device__ __forceinline__ int8_t dpx_to_i8_hi(unsigned p) { return (int8_t)dpx_hi(p); }

// ---- Direction byte (scalar) — same logic as HALF2 / int8 paths ----
__device__ __forceinline__ uint8_t dpx_dir_byte_scalar(
        int z, int a0, int b0, int a20, int b20,
        int q, int q2, bool right_align)
{
    uint8_t d = 0;
    if (!right_align) {
        if (a0  > z) { z = a0;  d = 1; }
        if (b0  > z) { z = b0;  d = 2; }
        if (a20 > z) { z = a20; d = 3; }
        if (b20 > z) { z = b20; d = 4; }
    } else {
        if (!(z > a0))  { z = a0;  d = 1; }
        if (!(z > b0))  { z = b0;  d = 2; }
        if (!(z > a20)) { z = a20; d = 3; }
        if (!(z > b20)) { z = b20; d = 4; }
    }
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
        if (!(0 > aa))  d |= 0x08;
        if (!(0 > bb))  d |= 0x10;
        if (!(0 > aa2)) d |= 0x20;
        if (!(0 > bb2)) d |= 0x40;
    }
    return d;
}

// =============================================================================
// One lane, 2 cells — value path uses DPX fused add+max (__viaddmax_s16x2).
// Compared to HALF2 (packed add + packed max = 2 instructions),
// DPX collapses to 1 fused instruction per candidate → 4 candidates halve.
// =============================================================================
__device__ __forceinline__ void dp_cell_pair_dpx(
        unsigned xv1,   unsigned vv1,   unsigned x2v1,
        unsigned uup,   unsigned yyp,   unsigned y2yp,
        unsigned ssc,
        unsigned gO,    unsigned gOL,
        unsigned negOE, unsigned negOEL,
        unsigned scMch,
        int q, int q2, bool with_cigar, bool right_align,
        unsigned *new_u, unsigned *new_v,
        unsigned *new_x, unsigned *new_y,
        unsigned *new_x2, unsigned *new_y2,
        uint8_t *d0, uint8_t *d1)
{
    // ---- Direction byte first (needs pre-max scalar candidates) ----
    if (with_cigar) {
        int z0  = dpx_lo(ssc),  z1  = dpx_hi(ssc);
        int a0  = dpx_lo(xv1)  + dpx_lo(vv1);
        int a1  = dpx_hi(xv1)  + dpx_hi(vv1);
        int b0  = dpx_lo(yyp)  + dpx_lo(uup);
        int b1  = dpx_hi(yyp)  + dpx_hi(uup);
        int a20 = dpx_lo(x2v1) + dpx_lo(vv1);
        int a21 = dpx_hi(x2v1) + dpx_hi(vv1);
        int b20 = dpx_lo(y2yp) + dpx_lo(uup);
        int b21 = dpx_hi(y2yp) + dpx_hi(uup);
        *d0 = dpx_dir_byte_scalar(z0, a0, b0, a20, b20, q, q2, right_align);
        *d1 = dpx_dir_byte_scalar(z1, a1, b1, a21, b21, q, q2, right_align);
    } else {
        *d0 = *d1 = 0;
    }

    // ---- Value path: DPX fused add+max (this is where the win is) ----
    unsigned z = ssc;
    z = __viaddmax_s16x2(xv1,  vv1, z);   // z = max(x1  + v1,       z)
    z = __viaddmax_s16x2(yyp,  uup, z);   // z = max(y_p + u_p,      z)
    z = __viaddmax_s16x2(x2v1, vv1, z);   // z = max(x21 + v1,       z)
    z = __viaddmax_s16x2(y2yp, uup, z);   // z = max(y2_p+ u_p,      z)
    z = __vmins2(z, scMch);                // clamp to sc_mch (packed signed min)

    // ---- H deltas (packed 16-bit lane sub — no cross-lane carry) ----
    *new_u = __vsub2(z, vv1);
    *new_v = __vsub2(z, uup);

    // ---- Recompute a,b,a2,b2 for gap-delta update ----
    unsigned a  = __vadd2(xv1,  vv1);
    unsigned b  = __vadd2(yyp,  uup);
    unsigned a2 = __vadd2(x2v1, vv1);
    unsigned b2 = __vadd2(y2yp, uup);

    // ---- a -= (z - q);  a2,b2 -= (z - q2) ----
    unsigned zq  = __vsub2(z, gO);
    unsigned zq2 = __vsub2(z, gOL);
    a  = __vsub2(a,  zq);
    b  = __vsub2(b,  zq);
    a2 = __vsub2(a2, zq2);
    b2 = __vsub2(b2, zq2);

    // ---- new_x = max(a - qe, -qe) = max(a + (-qe), -qe) — DPX 1 inst ----
    *new_x  = __viaddmax_s16x2(a,  negOE,  negOE);
    *new_y  = __viaddmax_s16x2(b,  negOE,  negOE);
    *new_x2 = __viaddmax_s16x2(a2, negOEL, negOEL);
    *new_y2 = __viaddmax_s16x2(b2, negOEL, negOEL);
}

#endif // __PLKSW_DPX_CORE_CUH__
