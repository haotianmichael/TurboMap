// =============================================================================
// plksw_half2_core.cuh — half2 2-cells-per-lane DP core (Suzuki-Kasahara recurrence).
// Global storage stays int8; values are packed into half2 only in registers.
// =============================================================================

#include <cuda_fp16.h>

__device__ __forceinline__ __half i8_to_h(int8_t v) { return __short2half_rn((short)v); }
__device__ __forceinline__ int8_t h_to_i8(__half v) { return (int8_t)__half2short_rn(v); }

__device__ __forceinline__ uint8_t dp_dir_byte_scalar(
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
        if (!(0 > aa))  d |= 0x08;   // aa >= 0
        if (!(0 > bb))  d |= 0x10;
        if (!(0 > aa2)) d |= 0x20;
        if (!(0 > bb2)) d |= 0x40;
    }
    return d;
}

// =============================================================================
//   xv1 = {x1, x1'}  vv1 = {v1, v1'}  x2v1 = {x21, x21'}
//   uup = {u_prev,*} yyp = {y_prev,*} y2yp = {y2_prev,*}
//   ssc = {score, score'}
//   negOE={-qe,-qe} negOEL={-qe2,-qe2} scMch={sc_mch,sc_mch}
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
    half2 z  = ssc;
    half2 a  = __hadd2(xv1,  vv1);
    half2 b  = __hadd2(yyp,  uup);
    half2 a2 = __hadd2(x2v1, vv1);
    half2 b2 = __hadd2(y2yp, uup);

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

    z = __hmax2(z, a);
    z = __hmax2(z, b);
    z = __hmax2(z, a2);
    z = __hmax2(z, b2);
    z = __hmin2(z, scMch);

    *new_u = __hsub2(z, vv1);
    *new_v = __hsub2(z, uup);

    half2 zq  = __hsub2(z, gO);
    half2 zq2 = __hsub2(z, gOL);
    a  = __hsub2(a,  zq);
    b  = __hsub2(b,  zq);
    a2 = __hsub2(a2, zq2);
    b2 = __hsub2(b2, zq2);

    *new_x  = __hmax2(__hsub2(a,  gOE),  negOE);
    *new_y  = __hmax2(__hsub2(b,  gOE),  negOE);
    *new_x2 = __hmax2(__hsub2(a2, gOEL), negOEL);
    *new_y2 = __hmax2(__hsub2(b2, gOEL), negOEL);
}
