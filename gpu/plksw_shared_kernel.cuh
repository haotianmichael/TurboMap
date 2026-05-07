/*
 * plksw_shared_kernel.cuh — Shared-memory variant of the long-task DP kernel.
 *
 * ============================================================================
 * MOTIVATION
 * ============================================================================
 *
 * For super-long tasks (tlen ~ 22 kbp), the legacy ksw_fused_persistent_kernel
 * is bound by global-memory latency on the per-cell delta arrays
 * (u/v/x/y/x2/y2 — int8 each).  Each cell does ~9 reads + 7 writes; with
 * only 1 warp per SM (bt_p budget caps slots at ~15), there is no other warp
 * to hide the ~200-cycle L2 latency, so each delta access stalls the warp.
 *
 * Profiled per-task time on the user's data: 5.1 s for one super-long task.
 *
 * ============================================================================
 * APPROACH
 * ============================================================================
 *
 * Move the 6 int8 delta arrays into per-block shared memory:
 *
 *   shared_buf  →  [u][v][x][y][x2][y2]   each of size  (max_tlen + 1) bytes
 *
 * Shared-memory access is ~30 cycles (vs ~200 for L2).  Expected speedup on
 * the inner DP loop: 3-5×.
 *
 * What stays in global / cache:
 *   - H[]                  (int32 abs scores, only used by lane 0 once per
 *                          antidiag for max-tracking — not hot per cell)
 *   - qr[], target[]       (sequence bytes, read every cell — but read-only,
 *                          so __ldg() routes through L1 read-only cache; on
 *                          A100 these stay hot in L1 across cells)
 *   - bt_p[]               (per-cell direction byte, ~1 GB per slot for
 *                          super-long; too big for shared, stays global —
 *                          stream-write during forward, stream-read during
 *                          backtrack; L2 doesn't help)
 *   - off[], off_end[]     (per-antidiag, tiny, stays global)
 *
 * ============================================================================
 * SIZING & DISPATCH
 * ============================================================================
 *
 * A100 (CC 8.0): max dynamic shared per block = 163 KB (after
 * cudaFuncSetAttribute(MaxDynamicSharedMemorySize, 163 KB)).
 *
 *   tlen | 6 × (tlen+1)   | fits 163 KB?
 *   -----|----------------|------------
 *   16k  |  96 KB         | ✓ comfortable
 *   22k  | 132 KB         | ✓ ok
 *   27k  | 162 KB         | ✓ at limit
 *   28k+ | 168 KB+        | ✗ exceeds
 *
 * Dispatch rule (in plalign.cu):
 *   if (max_tlen_in_batch <= SHARED_KERNEL_TLEN_LIMIT  &&
 *       legacy_phase_concurrent_slots < 100)
 *       use ksw_long_shared_kernel;
 *   else
 *       use ksw_fused_persistent_kernel;
 *
 * The second condition avoids switching for "regular long" batches where the
 * legacy kernel already runs many slots (1024+) — the shared kernel would
 * cap concurrency at ~108 (one block per A100 SM) and slow them down.
 *
 * ============================================================================
 * IMPORTANT: KEEP IN SYNC WITH plksw_kernel.cuh
 * ============================================================================
 *
 * The DP recurrence and CIGAR-walk logic must be byte-identical to
 * ksw_fused_persistent_kernel — any change there must be mirrored here, or
 * the two kernels will produce different CIGARs for the same input.
 */

#ifndef __PLKSW_SHARED_KERNEL_CUH__
#define __PLKSW_SHARED_KERNEL_CUH__
#include "gasal_kernels.h"

/* ──────────────────────────────────────────────────────────────────────── */
/* Compile-time configuration.                                              */
/* ──────────────────────────────────────────────────────────────────────── */

/* Hard ceiling on tlen for which 6×(tlen+1) bytes still fits in 163 KB.    */
#ifndef SHARED_KERNEL_TLEN_LIMIT
#define SHARED_KERNEL_TLEN_LIMIT 27000
#endif

/* Max dynamic shared per block to request from the driver (A100 = 163 KB). */
#ifndef SHARED_KERNEL_MAX_BYTES
#define SHARED_KERNEL_MAX_BYTES (163 * 1024)
#endif

/* ──────────────────────────────────────────────────────────────────────── */
/* Same flags as legacy kernel (reproduced here for header independence).   */
/* ──────────────────────────────────────────────────────────────────────── */
#ifndef KSW_EZ_SCORE_ONLY
#define KSW_EZ_SCORE_ONLY  0x01
#define KSW_EZ_RIGHT       0x02
#define KSW_EZ_GENERIC_SC  0x04
#define KSW_EZ_APPROX_MAX  0x08
#define KSW_EZ_APPROX_DROP 0x10
#define KSW_EZ_EXTZ_ONLY   0x40
#define KSW_EZ_REV_CIGAR   0x80
#endif

#ifndef KSW_CIGAR_MATCH
#define KSW_CIGAR_MATCH  0
#define KSW_CIGAR_INS    1
#define KSW_CIGAR_DEL    2
#define KSW_CIGAR_N_SKIP 3
#endif

#ifndef KSW_NEG_INF
#define KSW_NEG_INF -0x40000000
#endif

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

/* ──────────────────────────────────────────────────────────────────────── */
/* CIGAR push helper (identical to legacy).                                  */
/* ──────────────────────────────────────────────────────────────────────── */
__device__ static inline uint32_t* sh_push_cigar(
    int *n_cigar, int max_cigar_len,
    uint32_t *cigar, uint32_t op, int len)
{
    if (op > 3) op = 0;
    if (len <= 0) return cigar;
    if (*n_cigar == 0 || op != (cigar[(*n_cigar) - 1] & 0xf)) {
        if (*n_cigar < max_cigar_len)
            cigar[(*n_cigar)++] = (len << 4) | op;
    } else {
        cigar[(*n_cigar) - 1] += len << 4;
    }
    return cigar;
}

__device__ __forceinline__ int8_t sh_dp_score(
    uint8_t a, uint8_t b, const int8_t *mat, int m)
{
    return (a < m && b < m) ? mat[a * m + b] : 0;
}

/* ============================================================================
 *
 *                  SHARED-MEMORY LONG-TASK KERNEL
 *
 * Layout per block:
 *   blockDim = 32 (one warp), gridDim = grid_slots
 *   shared memory = 6 × (max_tlen_per_block + 1) bytes
 *
 * The launch-time `max_tlen_per_block` parameter is the largest tlen across
 * tasks in this batch (each task's own tlen <= this, so its delta sub-arrays
 * fit in [0..tlen]).  Unused tail of each sub-array is wasted but never read.
 *
 * ============================================================================ */
__global__ void ksw_long_shared_kernel(
    int      *d_task_counter,
    uint32_t *packed_query_batch,
    uint32_t *packed_ref_batch,
    uint32_t *query_batch_lens,
    uint32_t *target_batch_lens,
    uint32_t *query_batch_offsets,
    uint32_t *target_batch_offsets,
    gasal_res_t *device_res,
    const int8_t *device_mat,
    uint8_t  *backtrack_p,
    int      *backtrack_off,
    int      *backtrack_off_end,
    int       max_backtrack_size,
    int       max_antidiag,
    void     *d_temp_buffer,        /* H[] + qr[] + target[] only          */
    int      *d_flag,
    int32_t  *d_bw,
    size_t    temp_per_task,
    int       n_tasks,
    int8_t    m,
    int32_t   zdrop,
    int       end_bonus,
    uint32_t *cigar_buffer,
    int      *cigar_lengths,
    int       max_cigar_len,
    int       max_tlen_per_block)   /* shared-mem stride per delta array   */
{
    extern __shared__ int8_t shared_buf[];

    const int slot_id = blockIdx.x;
    const int lane_id = threadIdx.x;

    /* Per-block shared layout: 6 contiguous arrays of size                 */
    /* (max_tlen_per_block + 1) bytes each.                                 */
    const int stride = max_tlen_per_block + 1;
    int8_t *u_arr  = shared_buf + 0 * stride;
    int8_t *v_arr  = shared_buf + 1 * stride;
    int8_t *x_arr  = shared_buf + 2 * stride;
    int8_t *y_arr  = shared_buf + 3 * stride;
    int8_t *x2_arr = shared_buf + 4 * stride;
    int8_t *y2_arr = shared_buf + 5 * stride;

    while (true) {
        int task_id;
        if (lane_id == 0) task_id = atomicAdd(d_task_counter, 1);
        task_id = __shfl_sync(0xffffffff, task_id, 0);
        if (task_id >= n_tasks) return;

        int qlen = query_batch_lens[task_id];
        int tlen = target_batch_lens[task_id];
        int flag = d_flag[task_id];

        int32_t ez_max = 0;
        int32_t ez_max_q = -1, ez_max_t = -1;
        int32_t ez_mqe = KSW_NEG_INF, ez_mqe_t = -1;
        int32_t ez_mte = KSW_NEG_INF, ez_mte_q = -1;
        int32_t ez_score = KSW_NEG_INF;
        int ez_zdropped = 0, ez_reach_end = 0;

        if (qlen <= 0 || tlen <= 0) {
            if (lane_id == 0) {
                device_res->aln_score[task_id]        = 0;
                device_res->query_batch_end[task_id]  = -1;
                device_res->target_batch_end[task_id] = -1;
                device_res->mqe[task_id]   = KSW_NEG_INF;
                device_res->mqe_t[task_id] = -1;
                device_res->mte[task_id]   = KSW_NEG_INF;
                device_res->mte_q[task_id] = -1;
                device_res->zdropped[task_id] = 0;
                if (cigar_buffer) cigar_lengths[task_id] = 0;
            }
            __syncwarp();
            continue;
        }

        /* ───── Gap parameters (mirror legacy) ───── */
        int8_t q  = _cudaGapO;
        int8_t e  = _cudaGapExtend;
        int8_t q2 = _cudaGapOL;
        int8_t e2 = _cudaGapExtendL;
        int32_t w = d_bw[task_id];

        if (q2 + e2 < q + e) {
            int8_t tmp = q; q = q2; q2 = tmp;
            tmp = e; e = e2; e2 = tmp;
        }
        int8_t qe  = q  + e;
        int8_t qe2 = q2 + e2;

        int wl = (w < 0) ? max(qlen, tlen) : w;
        int wr = (w < 0) ? max(qlen, tlen) : w;

        int long_thres = (e != e2) ? (q2 - q) / (e - e2) - 1 : 0;
        if (q2 + e2 + long_thres * e2 > q + e + long_thres * e) ++long_thres;
        int32_t long_diff = long_thres * (e - e2) - (q2 - q) - e2;

        int8_t sc_mch = device_mat[0];

        /* ───── Per-slot global temp buffer.  We share the SAME allocation as
         * the legacy ksw_fused_persistent_kernel for compatibility, so the
         * layout matches its plksw_kernel.cuh layout:
         *       [ H[tlen] ][ unused 6×(tlen+1) deltas ][ qr[qlen] ][ target[tlen] ]
         * The 6×(tlen+1) delta region is unused here (deltas live in shared);
         * we just skip past it to find qr and target.                            */
        char *task_buf = (char*)d_temp_buffer + (size_t)slot_id * temp_per_task;
        int32_t *H = (int32_t*)task_buf;
        size_t qr_offset = (size_t)tlen * sizeof(int32_t)
                         + (size_t)6 * (size_t)(tlen + 1);  /* skip deltas */
        uint8_t *qr     = (uint8_t*)(task_buf + qr_offset);
        uint8_t *target = qr + qlen;

        /* ───── Initialise shared deltas (lane-strided across [0..tlen]). ───── */
        int8_t neg_qe  = (int8_t)(-(int)(q + e));
        int8_t neg_qe2 = (int8_t)(-(int)(q2 + e2));
        for (int i = lane_id; i < tlen; i += WARP_SIZE) {
            H[i]      = KSW_NEG_INF;
            u_arr[i]  = neg_qe;  v_arr[i]  = neg_qe;
            x_arr[i]  = neg_qe;  y_arr[i]  = neg_qe;
            x2_arr[i] = neg_qe2; y2_arr[i] = neg_qe2;
        }
        if (lane_id == 0) {
            u_arr[tlen]  = neg_qe;  v_arr[tlen]  = neg_qe;
            x_arr[tlen]  = neg_qe;  y_arr[tlen]  = neg_qe;
            x2_arr[tlen] = neg_qe2; y2_arr[tlen] = neg_qe2;
        }

        /* ───── Unpack sequences into per-task scratch (read via __ldg later). ───── */
        int packed_query_offset  = query_batch_offsets[task_id]  >> 3;
        int packed_target_offset = target_batch_offsets[task_id] >> 3;

        for (int i = lane_id; i < qlen; i += WARP_SIZE) {
            int packed_idx = i / 8;
            int bit_offset = (7 - (i % 8)) * 4;
            uint32_t pv = packed_query_batch[packed_query_offset + packed_idx];
            qr[qlen - 1 - i] = (pv >> bit_offset) & 0xF;
        }
        for (int i = lane_id; i < tlen; i += WARP_SIZE) {
            int packed_idx = i / 8;
            int bit_offset = (7 - (i % 8)) * 4;
            uint32_t pv = packed_ref_batch[packed_target_offset + packed_idx];
            target[i] = (pv >> bit_offset) & 0xF;
        }
        __syncwarp();

        /* ───── DP configuration ───── */
        int last_H0_t = 0;
        int32_t H0 = 0;

        int n_col = (qlen < tlen) ? qlen : tlen;
        n_col = (n_col < w + 1) ? n_col : (w + 1);

        int with_cigar  = !(flag & KSW_EZ_SCORE_ONLY);
        int approx_max  = !!(flag & KSW_EZ_APPROX_MAX);
        int right_align = !!(flag & KSW_EZ_RIGHT);

        uint8_t *p_bt   = backtrack_p   + (size_t)slot_id * max_backtrack_size;
        int     *off    = backtrack_off     + (size_t)slot_id * max_antidiag;
        int     *off_end = backtrack_off_end + (size_t)slot_id * max_antidiag;

        /* ───── Main DP loop (anti-diagonal, identical recurrence to legacy) ───── */
        int last_st = -1, last_en = -1;
        for (int r = 0; r < qlen + tlen - 1; r++) {
            if (r >= max_antidiag) {
                if (lane_id == 0) ez_zdropped = 1;
                break;
            }

            int st = 0, en = tlen - 1;
            if (st < r - qlen + 1) st = r - qlen + 1;
            if (en > r) en = r;
            if (st < (r - wr + 1) >> 1) st = (r - wr + 1) >> 1;
            if (en > (r + wl) >> 1) en = (r + wl) >> 1;

            if (st > en) {
                if (lane_id == 0) ez_zdropped = 1;
                break;
            }

            int st0 = st, en0 = en;

            /* ───── Boundary setup (lane 0) ───── */
            int8_t x1_boundary, v1_boundary, x21_boundary;
            if (lane_id == 0) {
                if (st0 > 0) {
                    if (st0 - 1 >= last_st && st0 - 1 <= last_en) {
                        x1_boundary  = x_arr[st0 - 1];
                        x21_boundary = x2_arr[st0 - 1];
                        v1_boundary  = v_arr[st0 - 1];
                    } else {
                        x1_boundary  = -q - e;
                        x21_boundary = -q2 - e2;
                        v1_boundary  = -q - e;
                    }
                } else {
                    x1_boundary  = -q - e;
                    x21_boundary = -q2 - e2;
                    if (r == 0)              v1_boundary = -q - e;
                    else if (r < long_thres) v1_boundary = -e;
                    else if (r == long_thres) v1_boundary = (int8_t)long_diff;
                    else                     v1_boundary = -e2;
                }

                if (en0 >= r && r < tlen) {
                    y_arr[r]  = -q - e;
                    y2_arr[r] = -q2 - e2;
                    if (r == 0)              u_arr[r] = -q - e;
                    else if (r < long_thres) u_arr[r] = -e;
                    else if (r == long_thres) u_arr[r] = (int8_t)long_diff;
                    else                     u_arr[r] = -e2;
                }

                if (with_cigar) {
                    off[r] = st0;
                    off_end[r] = en0;
                }
            }
            __syncwarp();

            x1_boundary  = __shfl_sync(0xffffffff, x1_boundary, 0);
            v1_boundary  = __shfl_sync(0xffffffff, v1_boundary, 0);
            x21_boundary = __shfl_sync(0xffffffff, x21_boundary, 0);

            uint8_t *pr = (with_cigar && cigar_buffer) ? (p_bt + (size_t)r * n_col) : NULL;

            /* ───── Inner batched parallel DP (same recurrence as legacy) ───── */
            int band_size = en0 - st0 + 1;
            int8_t batch_x1_boundary  = x1_boundary;
            int8_t batch_v1_boundary  = v1_boundary;
            int8_t batch_x21_boundary = x21_boundary;

            for (int batch_start = 0; batch_start < band_size; batch_start += WARP_SIZE) {
                int idx = batch_start + lane_id;
                bool active = (idx < band_size);

                int8_t my_x1, my_v1, my_x21;
                int8_t my_u_prev, my_y_prev, my_y2_prev;
                int8_t my_score;
                int8_t old_x_at_t = 0, old_v_at_t = 0, old_x2_at_t = 0;

                if (active) {
                    int t = st0 + idx;
                    int qi = r - t;
                    int qi_rev = qlen - 1 - qi;

                    /* Pre-load OLD values at position t before any writes. */
                    old_x_at_t  = x_arr[t];
                    old_v_at_t  = v_arr[t];
                    old_x2_at_t = x2_arr[t];

                    if (lane_id == 0) {
                        my_x1  = batch_x1_boundary;
                        my_v1  = batch_v1_boundary;
                        my_x21 = batch_x21_boundary;
                    } else {
                        my_x1  = x_arr[t - 1];
                        my_v1  = v_arr[t - 1];
                        my_x21 = x2_arr[t - 1];
                    }

                    my_u_prev  = u_arr[t];
                    my_y_prev  = y_arr[t];
                    my_y2_prev = y2_arr[t];

                    /* qr/target via __ldg for read-only L1 caching.        */
                    if (qi >= 0 && qi < qlen && t >= 0 && t < tlen) {
                        uint8_t qb = __ldg(&qr[qi_rev]);
                        uint8_t tb = __ldg(&target[t]);
                        my_score = sh_dp_score(qb, tb, device_mat, m);
                    } else {
                        my_score = 0;
                    }
                } else {
                    my_x1 = my_v1 = my_x21 = 0;
                    my_u_prev = my_y_prev = my_y2_prev = 0;
                    my_score = 0;
                }

                int8_t new_x = 0, new_v_out = 0, new_x2 = 0;

                if (active) {
                    int t = st0 + idx;

                    int8_t z = my_score;
                    int8_t a  = my_x1 + my_v1;
                    int8_t b  = my_y_prev + my_u_prev;
                    int8_t a2 = my_x21 + my_v1;
                    int8_t b2 = my_y2_prev + my_u_prev;

                    int8_t ut = my_u_prev;
                    uint8_t d = 0;

                    if (!with_cigar) {
                        if (a > z) z = a;
                        if (b > z) z = b;
                        if (a2 > z) z = a2;
                        if (b2 > z) z = b2;
                    } else if (!right_align) {
                        if (a > z) { z = a; d = 1; }
                        if (b > z) { z = b; d = 2; }
                        if (a2 > z) { z = a2; d = 3; }
                        if (b2 > z) { z = b2; d = 4; }
                    } else {
                        if (!(z > a)) { z = a; d = 1; }
                        if (!(z > b)) { z = b; d = 2; }
                        if (!(z > a2)) { z = a2; d = 3; }
                        if (!(z > b2)) { z = b2; d = 4; }
                    }

                    if (z > sc_mch) z = sc_mch;

                    int8_t new_u = z - my_v1;
                    new_v_out = z - ut;

                    int tmp_val = z - q;
                    a -= tmp_val;
                    b -= tmp_val;
                    tmp_val = z - q2;
                    a2 -= tmp_val;
                    b2 -= tmp_val;

                    int8_t new_y, new_y2;
                    if (!with_cigar || !right_align) {
                        new_x  = (a  > 0) ? (a  - qe)  : (-qe);
                        new_y  = (b  > 0) ? (b  - qe)  : (-qe);
                        new_x2 = (a2 > 0) ? (a2 - qe2) : (-qe2);
                        new_y2 = (b2 > 0) ? (b2 - qe2) : (-qe2);
                        if (with_cigar) {
                            if (a  > 0) d |= 0x08;
                            if (b  > 0) d |= 0x10;
                            if (a2 > 0) d |= 0x20;
                            if (b2 > 0) d |= 0x40;
                        }
                    } else {
                        new_x  = (!(0 > a))  ? (a  - qe)  : (-qe);
                        new_y  = (!(0 > b))  ? (b  - qe)  : (-qe);
                        new_x2 = (!(0 > a2)) ? (a2 - qe2) : (-qe2);
                        new_y2 = (!(0 > b2)) ? (b2 - qe2) : (-qe2);
                        if (!(0 > a))  d |= 0x08;
                        if (!(0 > b))  d |= 0x10;
                        if (!(0 > a2)) d |= 0x20;
                        if (!(0 > b2)) d |= 0x40;
                    }

                    /* Writes go to shared memory — the whole point of this kernel. */
                    u_arr[t]  = new_u;
                    v_arr[t]  = new_v_out;
                    x_arr[t]  = new_x;
                    y_arr[t]  = new_y;
                    x2_arr[t] = new_x2;
                    y2_arr[t] = new_y2;

                    if (pr != NULL) {
                        pr[t - st0] = d;
                    }
                }

                int last_lane = min(WARP_SIZE - 1, band_size - batch_start - 1);
                if (batch_start + WARP_SIZE < band_size) {
                    batch_x1_boundary  = __shfl_sync(0xffffffff, old_x_at_t,  last_lane);
                    batch_v1_boundary  = __shfl_sync(0xffffffff, old_v_at_t,  last_lane);
                    batch_x21_boundary = __shfl_sync(0xffffffff, old_x2_at_t, last_lane);
                }
                __syncwarp();
            }

            /* ───── Max tracking + z-drop (lane 0) — identical to legacy. ───── */
            if (lane_id == 0) {
                if (!approx_max) {
                    int32_t max_H, max_t_pos;
                    if (r == 0) {
                        H[0] = (int32_t)v_arr[0] - qe;
                        max_H = H[0];
                        max_t_pos = 0;
                    } else {
                        int32_t H_en0_old = (en0 > 0) ? H[en0 - 1] : H[en0];
                        if (en0 > 0) H[en0] = H_en0_old + (int32_t)u_arr[en0];
                        else         H[en0] += (int32_t)v_arr[en0];
                        max_H = H[en0];
                        max_t_pos = en0;
                        for (int t = st0; t < en0; ++t) {
                            H[t] += (int32_t)v_arr[t];
                            if (H[t] > max_H) { max_H = H[t]; max_t_pos = t; }
                        }
                    }
                    int j = max_t_pos;
                    int i = r - j;
                    if (max_H > ez_max) { ez_max = max_H; ez_max_t = j; ez_max_q = i; }
                    if (j >= ez_max_t && i >= ez_max_q) {
                        int tl = j - ez_max_t, ql = i - ez_max_q;
                        int l = (tl > ql) ? (tl - ql) : (ql - tl);
                        if (zdrop >= 0 && ez_max - max_H > zdrop + l * e2) ez_zdropped = 1;
                    }
                    if (en0 == tlen - 1 && H[en0] > ez_mte) { ez_mte = H[en0]; ez_mte_q = r - en0; }
                    if (r - st0 == qlen - 1 && st0 >= 0 && st0 < tlen && H[st0] > ez_mqe) {
                        ez_mqe = H[st0]; ez_mqe_t = st0;
                    }
                    if (r == qlen + tlen - 2 && en0 == tlen - 1) ez_score = H[tlen - 1];
                } else {
                    if (r > 0) {
                        if (last_H0_t >= st0 && last_H0_t <= en0 &&
                            last_H0_t + 1 >= st0 && last_H0_t + 1 <= en0) {
                            int32_t d0 = v_arr[last_H0_t];
                            int32_t d1 = u_arr[last_H0_t + 1];
                            if (d0 > d1) H0 += d0;
                            else { H0 += d1; ++last_H0_t; }
                        } else if (last_H0_t >= st0 && last_H0_t <= en0) {
                            H0 += v_arr[last_H0_t];
                        } else {
                            ++last_H0_t;
                            H0 += u_arr[last_H0_t];
                        }
                    } else {
                        H0 = v_arr[0] - qe;
                        last_H0_t = 0;
                    }
                    if (r == qlen + tlen - 2 && en0 == tlen - 1) ez_score = H0;
                }
                last_st = st0;
                last_en = en0;
            }

            int zdropped_flag = __shfl_sync(0xffffffff, ez_zdropped, 0);
            if (zdropped_flag) break;
            last_st = __shfl_sync(0xffffffff, last_st, 0);
            last_en = __shfl_sync(0xffffffff, last_en, 0);
        }

        /* ───── Result write ───── */
        int backtrack_q = -1, backtrack_t = -1;
        if (lane_id == 0) {
            if (!ez_zdropped && !(flag & KSW_EZ_EXTZ_ONLY)) {
                backtrack_q = qlen - 1;
                backtrack_t = tlen - 1;
            } else if (!ez_zdropped && (flag & KSW_EZ_EXTZ_ONLY) &&
                       ez_mqe + end_bonus > ez_max) {
                backtrack_q = qlen - 1;
                backtrack_t = ez_mqe_t;
                ez_reach_end = 1;
            } else if (ez_max_t >= 0 && ez_max_q >= 0) {
                backtrack_q = ez_max_q;
                backtrack_t = ez_max_t;
            }

            int32_t out_score;
            if (flag & KSW_EZ_EXTZ_ONLY) out_score = ez_max;
            else if (ez_zdropped)         out_score = ez_max;
            else                          out_score = ez_score;
            device_res->aln_score[task_id]        = out_score;
            device_res->query_batch_end[task_id]  = backtrack_q;
            device_res->target_batch_end[task_id] = backtrack_t;
            device_res->mqe[task_id]              = ez_mqe;
            device_res->mqe_t[task_id]            = ez_mqe_t;
            device_res->mte[task_id]              = ez_mte;
            device_res->mte_q[task_id]            = ez_mte_q;
            device_res->zdropped[task_id]         = ez_zdropped;
        }
        backtrack_q = __shfl_sync(0xffffffff, backtrack_q, 0);
        backtrack_t = __shfl_sync(0xffffffff, backtrack_t, 0);

        /* ───── Fused backtrack + CIGAR (lane 0).  bt_p still in global. ───── */
        if (cigar_buffer && with_cigar && lane_id == 0 &&
            backtrack_q >= 0 && backtrack_t >= 0) {

            int i = backtrack_t, j = backtrack_q;
            int state = 0;
            int n_cigar_ops = 0;
            int is_rev = !!(flag & KSW_EZ_REV_CIGAR);
            uint32_t *cigar = cigar_buffer + (size_t)task_id * max_cigar_len;

            while (i >= 0 && j >= 0) {
                int force_state = -1;
                int diag_r = i + j;
                if (i < off[diag_r])     force_state = 2;
                if (i > off_end[diag_r]) force_state = 1;

                uint8_t tmp_bt = 0;
                if (force_state < 0) {
                    size_t p_idx = (size_t)diag_r * n_col + i - off[diag_r];
                    tmp_bt = p_bt[p_idx];
                }

                if (state == 0)                                state = tmp_bt & 7;
                else if (!(tmp_bt >> (state + 2) & 1))         state = 0;
                if (state == 0)                                state = tmp_bt & 7;
                if (force_state >= 0)                          state = force_state;

                if (state == 0) {
                    sh_push_cigar(&n_cigar_ops, max_cigar_len, cigar, KSW_CIGAR_MATCH, 1);
                    --i; --j;
                } else if (state == 1 || state == 3) {
                    sh_push_cigar(&n_cigar_ops, max_cigar_len, cigar, KSW_CIGAR_DEL, 1);
                    --i;
                } else {
                    sh_push_cigar(&n_cigar_ops, max_cigar_len, cigar, KSW_CIGAR_INS, 1);
                    --j;
                }
                if (n_cigar_ops >= max_cigar_len - 2) break;
            }

            if (i >= 0)
                sh_push_cigar(&n_cigar_ops, max_cigar_len, cigar, KSW_CIGAR_DEL, i + 1);
            if (j >= 0)
                sh_push_cigar(&n_cigar_ops, max_cigar_len, cigar, KSW_CIGAR_INS, j + 1);

            if (!is_rev) {
                for (int k = 0; k < n_cigar_ops / 2; ++k) {
                    uint32_t tmp_c = cigar[k];
                    cigar[k] = cigar[n_cigar_ops - 1 - k];
                    cigar[n_cigar_ops - 1 - k] = tmp_c;
                }
            }
            cigar_lengths[task_id] = n_cigar_ops;
        } else if (cigar_buffer && lane_id == 0) {
            cigar_lengths[task_id] = 0;
        }
        __syncwarp();
    } /* persistent loop */
}

#endif /* __PLKSW_SHARED_KERNEL_CUH__ */
