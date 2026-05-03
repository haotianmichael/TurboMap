/*
 * plgrid_kernel.cuh — Gridded Traceback Forward + Backtrack Kernels
 *
 * See plgrid_config.h for the high-level design.  This file contains:
 *   1. ksw_gridded_forward_kernel  — same Suzuki-Kasahara DP as the legacy
 *      ksw_fused_persistent_kernel, but writes G-spaced delta-state
 *      checkpoints instead of per-cell direction bytes.  No fused backtrack.
 *
 *   2. ksw_gridded_backtrack_kernel — separate kernel that walks each task's
 *      path backwards.  When it crosses a G-block boundary it loads the
 *      checkpoint, replays the forward DP for one G-block of antidiagonals
 *      into a per-slot scratch buffer (writing direction bytes), then walks
 *      that scratch to extract CIGAR.
 *
 * Both kernels are guarded by USE_GRIDDED_BT — when 0 they compile to nothing
 * so the legacy code path is completely undisturbed.
 *
 * Implementation note: the inner DP recurrence (the "batch loop" that
 * processes WARP_SIZE cells at a time) is duplicated here from
 * plksw_kernel.cuh because both forward and replay need it.  Any change to
 * the DP recurrence MUST be mirrored in all three places (legacy forward,
 * gridded forward, gridded replay).  Keep them in sync.
 */

#ifndef __PLGRID_KERNEL_CUH__
#define __PLGRID_KERNEL_CUH__

#include "plgrid_config.h"

#if USE_GRIDDED_BT

#include "gasal_kernels.h"

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
/* CIGAR push helper (same as legacy kernel).                                */
/* ──────────────────────────────────────────────────────────────────────── */
__device__ static inline void grid_push_cigar(
    int *n_cigar, int max_cigar_len,
    uint32_t *cigar, uint32_t op, int len)
{
    if (op > 3) op = 0;
    if (len <= 0) return;
    if (*n_cigar == 0 || op != (cigar[(*n_cigar) - 1] & 0xf)) {
        if (*n_cigar < max_cigar_len)
            cigar[(*n_cigar)++] = (len << 4) | op;
    } else {
        cigar[(*n_cigar) - 1] += len << 4;
    }
}

__device__ static inline int8_t grid_dp_score(uint8_t a, uint8_t b, int8_t *mat, int m) {
    return (a < m && b < m) ? mat[a * m + b] : 0;
}

/* ──────────────────────────────────────────────────────────────────────── */
/* Snapshot the antidiag delta arrays into the per-task checkpoint slot.    */
/*                                                                          */
/* Called by lane 0..len after the antidiag's DP completes.  Writes to      */
/*   ckpt_base[0..6*max_n_col-1]                                            */
/* in the layout [u][v][x][y][x2][y2] where each sub-array is max_n_col     */
/* bytes wide.  Cells outside [st, en] are NOT written (they are never read */
/* during replay because replay also restricts itself to [st, en]).         */
/*                                                                          */
/* Note: max_n_col is the per-task n_col (worst-case band width for that    */
/* task).  Each sub-array is sized to max_n_col so we can index by          */
/* (t - st0) directly.                                                      */
/* ──────────────────────────────────────────────────────────────────────── */
__device__ static inline void grid_save_checkpoint(
    int8_t *ckpt_base,    /* points at the 6*max_n_col-byte slot for this checkpoint */
    int max_n_col,
    int st, int en,
    const int8_t *u_arr, const int8_t *v_arr,
    const int8_t *x_arr, const int8_t *y_arr,
    const int8_t *x2_arr, const int8_t *y2_arr,
    int lane_id)
{
    int len = en - st + 1;
    int8_t *u_save  = ckpt_base + 0 * max_n_col;
    int8_t *v_save  = ckpt_base + 1 * max_n_col;
    int8_t *x_save  = ckpt_base + 2 * max_n_col;
    int8_t *y_save  = ckpt_base + 3 * max_n_col;
    int8_t *x2_save = ckpt_base + 4 * max_n_col;
    int8_t *y2_save = ckpt_base + 5 * max_n_col;
    for (int i = lane_id; i < len; i += WARP_SIZE) {
        u_save[i]  = u_arr[st + i];
        v_save[i]  = v_arr[st + i];
        x_save[i]  = x_arr[st + i];
        y_save[i]  = y_arr[st + i];
        x2_save[i] = x2_arr[st + i];
        y2_save[i] = y2_arr[st + i];
    }
}

/* ──────────────────────────────────────────────────────────────────────── */
/* Restore the antidiag delta arrays from a checkpoint into the slot's      */
/* working u/v/x/y/x2/y2 arrays.  Inverse of grid_save_checkpoint.          */
/* ──────────────────────────────────────────────────────────────────────── */
__device__ static inline void grid_load_checkpoint(
    const int8_t *ckpt_base,
    int max_n_col,
    int st, int en,
    int8_t *u_arr, int8_t *v_arr,
    int8_t *x_arr, int8_t *y_arr,
    int8_t *x2_arr, int8_t *y2_arr,
    int lane_id)
{
    int len = en - st + 1;
    const int8_t *u_save  = ckpt_base + 0 * max_n_col;
    const int8_t *v_save  = ckpt_base + 1 * max_n_col;
    const int8_t *x_save  = ckpt_base + 2 * max_n_col;
    const int8_t *y_save  = ckpt_base + 3 * max_n_col;
    const int8_t *x2_save = ckpt_base + 4 * max_n_col;
    const int8_t *y2_save = ckpt_base + 5 * max_n_col;
    for (int i = lane_id; i < len; i += WARP_SIZE) {
        u_arr[st + i]  = u_save[i];
        v_arr[st + i]  = v_save[i];
        x_arr[st + i]  = x_save[i];
        y_arr[st + i]  = y_save[i];
        x2_arr[st + i] = x2_save[i];
        y2_arr[st + i] = y2_save[i];
    }
}

/* ──────────────────────────────────────────────────────────────────────── */
/* The inner antidiag DP body — same as the inner "batch loop" in           */
/* plksw_kernel.cuh:ksw_fused_persistent_kernel.  Both the gridded forward  */
/* kernel and the replay path inside the gridded backtrack call this.       */
/*                                                                          */
/* If pr != NULL writes per-cell direction byte to pr[t - st0] (replay).    */
/* If pr == NULL skips that write (forward).                                */
/*                                                                          */
/* Pre-conditions (set up by caller via grid_setup_boundaries):             */
/*   - left-boundary deltas (x1_boundary, v1_boundary, x21_boundary)        */
/*     placed BEFORE call (broadcast to all lanes via __shfl_sync)          */
/*   - right-boundary u/y/y2 placed in u_arr/y_arr/y2_arr (lane 0)          */
/*                                                                          */
/* MUST stay bit-identical to plksw_kernel.cuh's batch loop for replay      */
/* correctness.  Any change there must be mirrored here.                    */
/* ──────────────────────────────────────────────────────────────────────── */
__device__ static inline void grid_dp_antidiag_full(
    int r, int st0, int en0,
    int qlen, int tlen,
    int8_t q,  int8_t e,
    int8_t q2, int8_t e2,
    int8_t qe, int8_t qe2,
    int8_t sc_mch, int8_t m,
    const uint8_t *qr, const uint8_t *target, const int8_t *device_mat,
    int8_t *u_arr, int8_t *v_arr,
    int8_t *x_arr, int8_t *y_arr,
    int8_t *x2_arr, int8_t *y2_arr,
    int8_t x1_boundary_in, int8_t v1_boundary_in, int8_t x21_boundary_in,
    int with_cigar, int right_align,
    uint8_t *pr,                /* if non-NULL, write d byte to pr[t-st0]   */
    int lane_id)
{
    int band_size = en0 - st0 + 1;
    int8_t batch_x1_boundary  = x1_boundary_in;
    int8_t batch_v1_boundary  = v1_boundary_in;
    int8_t batch_x21_boundary = x21_boundary_in;

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

            if (qi >= 0 && qi < qlen && t >= 0 && t < tlen) {
                my_score = grid_dp_score(qr[qi_rev], target[t], (int8_t*)device_mat, m);
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
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Helper: set up the per-antidiag boundary deltas (left-boundary x1/v1/x21  */
/* and right-boundary u/y/y2).  Mirrors the legacy kernel's lane-0 setup      */
/* block.  Must be called by lane 0 before grid_dp_antidiag_full.             */
/* ────────────────────────────────────────────────────────────────────────── */
__device__ static inline void grid_setup_boundaries(
    int r, int st0, int en0,
    int qlen, int tlen,
    int last_st, int last_en,
    int8_t q, int8_t e, int8_t q2, int8_t e2,
    int long_thres, int32_t long_diff,
    int8_t *u_arr, int8_t *v_arr,
    int8_t *x_arr, int8_t *y_arr,
    int8_t *x2_arr, int8_t *y2_arr,
    int8_t *out_x1, int8_t *out_v1, int8_t *out_x21,
    int lane_id)
{
    if (lane_id != 0) return;

    int8_t x1_boundary, v1_boundary, x21_boundary;
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
        if (r == 0) {
            v1_boundary = -q - e;
        } else if (r < long_thres) {
            v1_boundary = -e;
        } else if (r == long_thres) {
            v1_boundary = (int8_t)long_diff;
        } else {
            v1_boundary = -e2;
        }
    }
    if (en0 >= r && r < tlen) {
        y_arr[r]  = -q - e;
        y2_arr[r] = -q2 - e2;
        if (r == 0) {
            u_arr[r] = -q - e;
        } else if (r < long_thres) {
            u_arr[r] = -e;
        } else if (r == long_thres) {
            u_arr[r] = (int8_t)long_diff;
        } else {
            u_arr[r] = -e2;
        }
    }
    *out_x1  = x1_boundary;
    *out_v1  = v1_boundary;
    *out_x21 = x21_boundary;
}

/* ============================================================================
 *
 *                    GRIDDED FORWARD KERNEL
 *
 * Identical to ksw_fused_persistent_kernel except:
 *   - No per-cell direction byte written.
 *   - Every G antidiagonals, snapshot u/v/x/y/x2/y2 to the per-task dblock.
 *   - off[r] / off_end[r] are still written (they are tiny and reused by
 *     the backtrack kernel to know the band geometry for replay).
 *   - No fused backtrack at the end — that is now a separate kernel.
 *
 * ============================================================================
 */
__global__ void ksw_gridded_forward_kernel(
    int *d_task_counter,
    uint32_t *packed_query_batch,
    uint32_t *packed_ref_batch,
    uint32_t *query_batch_lens,
    uint32_t *target_batch_lens,
    uint32_t *query_batch_offsets,
    uint32_t *target_batch_offsets,
    gasal_res_t *device_res,
    int8_t *device_mat,
    int8_t *dblock_buf,           /* per-slot stride = max_dblock_per_slot     */
    size_t dblock_per_slot,        /* bytes                                      */
    int *backtrack_off,
    int *backtrack_off_end,
    int max_antidiag,
    int max_n_col_per_slot,        /* width of each delta sub-array per ckpt    */
    void *d_temp_buffer,
    int *d_flag,
    int32_t *d_bw,
    size_t temp_per_task,
    int n_tasks,
    int8_t m,
    int32_t zdrop,
    int end_bonus,
    int *cigar_lengths             /* set to 0 only — actual CIGAR done later   */
)
{
    const int slot_id = blockIdx.x;
    const int lane_id = threadIdx.x;

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
                device_res->aln_score[task_id] = 0;
                device_res->query_batch_end[task_id]  = -1;
                device_res->target_batch_end[task_id] = -1;
                device_res->mqe[task_id]   = KSW_NEG_INF;
                device_res->mqe_t[task_id] = -1;
                device_res->mte[task_id]   = KSW_NEG_INF;
                device_res->mte_q[task_id] = -1;
                device_res->zdropped[task_id] = 0;
                if (cigar_lengths) cigar_lengths[task_id] = 0;
            }
            __syncwarp();
            continue;
        }

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

        char *task_buf = (char*)d_temp_buffer + (size_t)slot_id * temp_per_task;
        size_t buf_offset = 0;
        int32_t *H = (int32_t*)(task_buf + buf_offset); buf_offset += tlen * sizeof(int32_t);
        int8_t *u_arr = (int8_t*)(task_buf + buf_offset); buf_offset += (tlen + 1);
        int8_t *v_arr = (int8_t*)(task_buf + buf_offset); buf_offset += (tlen + 1);
        int8_t *x_arr = (int8_t*)(task_buf + buf_offset); buf_offset += (tlen + 1);
        int8_t *y_arr = (int8_t*)(task_buf + buf_offset); buf_offset += (tlen + 1);
        int8_t *x2_arr = (int8_t*)(task_buf + buf_offset); buf_offset += (tlen + 1);
        int8_t *y2_arr = (int8_t*)(task_buf + buf_offset); buf_offset += (tlen + 1);
        uint8_t *qr = (uint8_t*)(task_buf + buf_offset); buf_offset += qlen;
        uint8_t *target = (uint8_t*)(task_buf + buf_offset);

        int8_t neg_qe  = (int8_t)(-(int)(q + e));
        int8_t neg_qe2 = (int8_t)(-(int)(q2 + e2));
        for (int i = lane_id; i < tlen; i += WARP_SIZE) {
            H[i] = KSW_NEG_INF;
            u_arr[i] = neg_qe;  v_arr[i] = neg_qe;
            x_arr[i] = neg_qe;  y_arr[i] = neg_qe;
            x2_arr[i] = neg_qe2; y2_arr[i] = neg_qe2;
        }
        if (lane_id == 0) {
            u_arr[tlen] = neg_qe;  v_arr[tlen] = neg_qe;
            x_arr[tlen] = neg_qe;  y_arr[tlen] = neg_qe;
            x2_arr[tlen] = neg_qe2; y2_arr[tlen] = neg_qe2;
        }

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

        int last_H0_t = 0;
        int32_t H0 = 0;

        int n_col = (qlen < tlen) ? qlen : tlen;
        n_col = (n_col < w + 1) ? n_col : (w + 1);

        int with_cigar  = !(flag & KSW_EZ_SCORE_ONLY);
        int approx_max  = !!(flag & KSW_EZ_APPROX_MAX);
        int right_align = !!(flag & KSW_EZ_RIGHT);

        /* Per-task slot pointers into the global dblock buffer.            */
        int8_t *slot_dblock = dblock_buf + (size_t)slot_id * dblock_per_slot;
        int    *off     = backtrack_off     + (size_t)slot_id * max_antidiag;
        int    *off_end = backtrack_off_end + (size_t)slot_id * max_antidiag;

        int last_st = -1, last_en = -1;
        int total_diags = qlen + tlen - 1;

        for (int r = 0; r < total_diags; r++) {
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

            int8_t x1_boundary = 0, v1_boundary = 0, x21_boundary = 0;
            grid_setup_boundaries(r, st0, en0, qlen, tlen, last_st, last_en,
                                  q, e, q2, e2, long_thres, long_diff,
                                  u_arr, v_arr, x_arr, y_arr, x2_arr, y2_arr,
                                  &x1_boundary, &v1_boundary, &x21_boundary, lane_id);
            if (lane_id == 0 && with_cigar) {
                off[r] = st0;
                off_end[r] = en0;
            }
            __syncwarp();
            x1_boundary  = __shfl_sync(0xffffffff, x1_boundary,  0);
            v1_boundary  = __shfl_sync(0xffffffff, v1_boundary,  0);
            x21_boundary = __shfl_sync(0xffffffff, x21_boundary, 0);

            /* Run the antidiag DP.  Pass NULL for pr — gridded forward never */
            /* writes per-cell direction bytes during forward pass.           */
            grid_dp_antidiag_full(r, st0, en0, qlen, tlen,
                                  q, e, q2, e2, qe, qe2, sc_mch, m,
                                  qr, target, device_mat,
                                  u_arr, v_arr, x_arr, y_arr, x2_arr, y2_arr,
                                  x1_boundary, v1_boundary, x21_boundary,
                                  with_cigar, right_align,
                                  /*pr=*/NULL, lane_id);

            /* Max-tracking + z-drop (lane 0).  Identical to legacy kernel.   */
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

            /* ───────── checkpoint write ───────── */
            /* Save the post-state of antidiag r (i.e. the input to r+1)   */
            /* whenever (r+1) % G == 0, AND r+1 is not the very last       */
            /* antidiag (no need to save right before termination).        */
            if (with_cigar && ((r + 1) % GRID_BLOCK_SIZE) == 0 && (r + 1) < total_diags) {
                int ckpt_idx = (r + 1) / GRID_BLOCK_SIZE - 1;  /* 0-indexed */
                int8_t *ckpt_base = slot_dblock +
                    (size_t)ckpt_idx * GRID_CKPT_STRIDE_BYTES(max_n_col_per_slot);
                grid_save_checkpoint(ckpt_base, max_n_col_per_slot,
                                     st0, en0,
                                     u_arr, v_arr, x_arr, y_arr, x2_arr, y2_arr,
                                     lane_id);
                __syncwarp();
            }
        } /* antidiag loop */

        /* ───────── Write per-task results ───────── */
        int backtrack_q = -1, backtrack_t = -1;
        if (lane_id == 0) {
            if (!ez_zdropped && !(flag & KSW_EZ_EXTZ_ONLY)) {
                backtrack_q = qlen - 1;  backtrack_t = tlen - 1;
            } else if (!ez_zdropped && (flag & KSW_EZ_EXTZ_ONLY) &&
                       ez_mqe + end_bonus > ez_max) {
                backtrack_q = qlen - 1;  backtrack_t = ez_mqe_t;
                ez_reach_end = 1;
            } else if (ez_max_t >= 0 && ez_max_q >= 0) {
                backtrack_q = ez_max_q;  backtrack_t = ez_max_t;
            }

            int32_t out_score;
            if (flag & KSW_EZ_EXTZ_ONLY) out_score = ez_max;
            else if (ez_zdropped)        out_score = ez_max;
            else                         out_score = ez_score;
            device_res->aln_score[task_id]        = out_score;
            device_res->query_batch_end[task_id]  = backtrack_q;
            device_res->target_batch_end[task_id] = backtrack_t;
            device_res->mqe[task_id]              = ez_mqe;
            device_res->mqe_t[task_id]            = ez_mqe_t;
            device_res->mte[task_id]              = ez_mte;
            device_res->mte_q[task_id]            = ez_mte_q;
            device_res->zdropped[task_id]         = ez_zdropped;
            /* Initialise CIGAR length to 0 — the backtrack kernel fills it. */
            if (cigar_lengths) cigar_lengths[task_id] = 0;
        }
        __syncwarp();
    }
}


/* ============================================================================
 *
 *                    GRIDDED BACKTRACK KERNEL
 *
 * One warp per slot.  Persistent task loop matching the forward kernel.
 * For each task:
 *   1. Load the saved (i, j) endpoint from device_res.
 *   2. While (i, j) within bounds:
 *        a. Compute current G-block from r = i+j.
 *        b. If we have not yet replayed this block, do so:
 *             - Re-init slot state arrays (or load from previous checkpoint)
 *             - Replay G antidiagonals via grid_dp_antidiag_full(),
 *               this time WITH pr non-NULL so direction bytes land in scratch.
 *        c. Read direction byte at (r, t=j) from scratch and walk one step.
 *
 * To keep memory bounded, the slot-resident u/v/x/y/x2/y2 arrays are reused.
 * Re-initialisation for block 0 is the same as forward init.  For block k>=1
 * we load the checkpoint at index k-1.
 *
 * ============================================================================
 */
__global__ void ksw_gridded_backtrack_kernel(
    int *d_task_counter,
    uint32_t *packed_query_batch,
    uint32_t *packed_ref_batch,
    uint32_t *query_batch_lens,
    uint32_t *target_batch_lens,
    uint32_t *query_batch_offsets,
    uint32_t *target_batch_offsets,
    gasal_res_t *device_res,
    int8_t *device_mat,
    int8_t *dblock_buf,           /* per-slot stride = dblock_per_slot          */
    size_t  dblock_per_slot,
    uint8_t *scratch_buf,         /* per-slot stride = G * max_n_col_per_slot   */
    size_t  scratch_per_slot,
    int *backtrack_off,
    int *backtrack_off_end,
    int max_antidiag,
    int max_n_col_per_slot,
    void *d_temp_buffer,
    int *d_flag,
    int32_t *d_bw,
    size_t temp_per_task,
    int n_tasks,
    int8_t m,
    uint32_t *cigar_buffer,
    int *cigar_lengths,
    int max_cigar_len)
{
    const int slot_id = blockIdx.x;
    const int lane_id = threadIdx.x;

    while (true) {
        int task_id;
        if (lane_id == 0) task_id = atomicAdd(d_task_counter, 1);
        task_id = __shfl_sync(0xffffffff, task_id, 0);
        if (task_id >= n_tasks) return;

        int qlen = query_batch_lens[task_id];
        int tlen = target_batch_lens[task_id];
        int flag = d_flag[task_id];

        int with_cigar = !(flag & KSW_EZ_SCORE_ONLY);
        if (!with_cigar || qlen <= 0 || tlen <= 0 ||
            cigar_buffer == NULL) {
            if (lane_id == 0 && cigar_lengths) cigar_lengths[task_id] = 0;
            __syncwarp();
            continue;
        }

        /* Load saved endpoints from forward kernel.                        */
        int backtrack_q = -1, backtrack_t = -1;
        if (lane_id == 0) {
            backtrack_q = device_res->query_batch_end[task_id];
            backtrack_t = device_res->target_batch_end[task_id];
        }
        backtrack_q = __shfl_sync(0xffffffff, backtrack_q, 0);
        backtrack_t = __shfl_sync(0xffffffff, backtrack_t, 0);

        if (backtrack_q < 0 || backtrack_t < 0) {
            if (lane_id == 0) cigar_lengths[task_id] = 0;
            __syncwarp();
            continue;
        }

        /* Gap params (mirror forward setup).                                */
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
        int right_align = !!(flag & KSW_EZ_RIGHT);

        int n_col = (qlen < tlen) ? qlen : tlen;
        n_col = (n_col < w + 1) ? n_col : (w + 1);

        /* Slot-resident temp buffers.                                       */
        char *task_buf = (char*)d_temp_buffer + (size_t)slot_id * temp_per_task;
        size_t buf_offset = 0;
        int32_t *H = (int32_t*)(task_buf + buf_offset); buf_offset += tlen * sizeof(int32_t);
        int8_t *u_arr = (int8_t*)(task_buf + buf_offset); buf_offset += (tlen + 1);
        int8_t *v_arr = (int8_t*)(task_buf + buf_offset); buf_offset += (tlen + 1);
        int8_t *x_arr = (int8_t*)(task_buf + buf_offset); buf_offset += (tlen + 1);
        int8_t *y_arr = (int8_t*)(task_buf + buf_offset); buf_offset += (tlen + 1);
        int8_t *x2_arr = (int8_t*)(task_buf + buf_offset); buf_offset += (tlen + 1);
        int8_t *y2_arr = (int8_t*)(task_buf + buf_offset); buf_offset += (tlen + 1);
        uint8_t *qr = (uint8_t*)(task_buf + buf_offset); buf_offset += qlen;
        uint8_t *target = (uint8_t*)(task_buf + buf_offset);
        (void)H;

        /* Sequence unpack — same as forward.                                */
        int packed_query_offset  = query_batch_offsets[task_id]  >> 3;
        int packed_target_offset = target_batch_offsets[task_id] >> 3;
        for (int i = lane_id; i < qlen; i += WARP_SIZE) {
            int pidx = i / 8;
            int boff = (7 - (i % 8)) * 4;
            uint32_t pv = packed_query_batch[packed_query_offset + pidx];
            qr[qlen - 1 - i] = (pv >> boff) & 0xF;
        }
        for (int i = lane_id; i < tlen; i += WARP_SIZE) {
            int pidx = i / 8;
            int boff = (7 - (i % 8)) * 4;
            uint32_t pv = packed_ref_batch[packed_target_offset + pidx];
            target[i] = (pv >> boff) & 0xF;
        }
        __syncwarp();

        /* Per-task pointers into global dblock + per-slot scratch.          */
        int8_t  *slot_dblock  = dblock_buf  + (size_t)slot_id * dblock_per_slot;
        uint8_t *slot_scratch = scratch_buf + (size_t)slot_id * scratch_per_slot;
        int *off     = backtrack_off     + (size_t)slot_id * max_antidiag;
        int *off_end = backtrack_off_end + (size_t)slot_id * max_antidiag;

        /* Replay state — which G-block currently in scratch (-1 = none).    */
        int cached_block = -1;

        int i = backtrack_t;
        int j = backtrack_q;
        int state = 0;
        int n_cigar_ops = 0;
        int is_rev = !!(flag & KSW_EZ_REV_CIGAR);
        uint32_t *cigar = cigar_buffer + (size_t)task_id * max_cigar_len;

        /* Walk back.  Lane 0 drives, other lanes participate during replay. */
        while (i >= 0 && j >= 0) {
            int r = i + j;
            int desired_block = r / GRID_BLOCK_SIZE;

            if (desired_block != cached_block) {
                /* Need to replay block `desired_block`.                     */
                int block_first_r = desired_block * GRID_BLOCK_SIZE;
                int block_last_r  = block_first_r + GRID_BLOCK_SIZE - 1;
                if (block_last_r > qlen + tlen - 2) block_last_r = qlen + tlen - 2;

                /* Re-init or load checkpoint to seed the working delta arrays. */
                int8_t neg_qe  = (int8_t)(-(int)(q + e));
                int8_t neg_qe2 = (int8_t)(-(int)(q2 + e2));

                if (desired_block == 0) {
                    /* Block 0: same init as forward.                         */
                    for (int t = lane_id; t < tlen; t += WARP_SIZE) {
                        u_arr[t] = neg_qe;  v_arr[t] = neg_qe;
                        x_arr[t] = neg_qe;  y_arr[t] = neg_qe;
                        x2_arr[t] = neg_qe2; y2_arr[t] = neg_qe2;
                    }
                    if (lane_id == 0) {
                        u_arr[tlen] = neg_qe;  v_arr[tlen] = neg_qe;
                        x_arr[tlen] = neg_qe;  y_arr[tlen] = neg_qe;
                        x2_arr[tlen] = neg_qe2; y2_arr[tlen] = neg_qe2;
                    }
                } else {
                    /* Load checkpoint for the START of this block.           */
                    /* Checkpoint k saves state AFTER antidiag k*G + (G-1),   */
                    /* which is exactly the input to antidiag (k+1)*G.        */
                    /* So for block `desired_block`, we want checkpoint       */
                    /* index (desired_block - 1) — saved after antidiag       */
                    /* desired_block*G - 1.                                   */
                    int ckpt_idx = desired_block - 1;
                    int prev_r = block_first_r - 1;
                    int prev_st = (with_cigar) ? off[prev_r]     : 0;
                    int prev_en = (with_cigar) ? off_end[prev_r] : tlen - 1;

                    /* Reset arrays to defaults first (cells outside band must */
                    /* hold default delta values for boundary lookups in       */
                    /* grid_dp_antidiag_full when band expands).              */
                    for (int t = lane_id; t < tlen; t += WARP_SIZE) {
                        u_arr[t] = neg_qe;  v_arr[t] = neg_qe;
                        x_arr[t] = neg_qe;  y_arr[t] = neg_qe;
                        x2_arr[t] = neg_qe2; y2_arr[t] = neg_qe2;
                    }
                    if (lane_id == 0) {
                        u_arr[tlen] = neg_qe;  v_arr[tlen] = neg_qe;
                        x_arr[tlen] = neg_qe;  y_arr[tlen] = neg_qe;
                        x2_arr[tlen] = neg_qe2; y2_arr[tlen] = neg_qe2;
                    }
                    __syncwarp();

                    int8_t *ckpt_base = slot_dblock +
                        (size_t)ckpt_idx * GRID_CKPT_STRIDE_BYTES(max_n_col_per_slot);
                    grid_load_checkpoint(ckpt_base, max_n_col_per_slot,
                                         prev_st, prev_en,
                                         u_arr, v_arr, x_arr, y_arr, x2_arr, y2_arr,
                                         lane_id);
                    __syncwarp();
                }

                /* Replay antidiagonals in [block_first_r .. block_last_r]    */
                /* writing direction bytes into scratch[(r-block_first_r)*    */
                /*                                       max_n_col + (t-st0)] */
                int last_st_replay = (desired_block == 0) ? -1 : off[block_first_r - 1];
                int last_en_replay = (desired_block == 0) ? -1 : off_end[block_first_r - 1];

                for (int rr = block_first_r; rr <= block_last_r; rr++) {
                    int st0 = off[rr];
                    int en0 = off_end[rr];
                    int8_t x1b = 0, v1b = 0, x21b = 0;
                    grid_setup_boundaries(rr, st0, en0, qlen, tlen,
                                          last_st_replay, last_en_replay,
                                          q, e, q2, e2, long_thres, long_diff,
                                          u_arr, v_arr, x_arr, y_arr, x2_arr, y2_arr,
                                          &x1b, &v1b, &x21b, lane_id);
                    __syncwarp();
                    x1b  = __shfl_sync(0xffffffff, x1b,  0);
                    v1b  = __shfl_sync(0xffffffff, v1b,  0);
                    x21b = __shfl_sync(0xffffffff, x21b, 0);

                    int local_r = rr - block_first_r;
                    uint8_t *pr_scratch = slot_scratch + (size_t)local_r * max_n_col_per_slot;

                    grid_dp_antidiag_full(rr, st0, en0, qlen, tlen,
                                          q, e, q2, e2, qe, qe2, sc_mch, m,
                                          qr, target, device_mat,
                                          u_arr, v_arr, x_arr, y_arr, x2_arr, y2_arr,
                                          x1b, v1b, x21b,
                                          /*with_cigar=*/1, right_align,
                                          /*pr=*/pr_scratch, lane_id);
                    last_st_replay = st0;
                    last_en_replay = en0;
                    __syncwarp();
                }

                cached_block = desired_block;
            }

            /* ── Walk one step using scratch direction byte ── */
            if (lane_id == 0) {
                int block_first_r = cached_block * GRID_BLOCK_SIZE;
                int local_r = r - block_first_r;
                int st0 = off[r];
                int en0 = off_end[r];

                int force_state = -1;
                if (i < st0)     force_state = 2;
                if (i > en0)     force_state = 1;

                uint8_t tmp_bt = 0;
                if (force_state < 0) {
                    /* In the inner DP we wrote pr[t - st0] = d where t is the */
                    /* target position.  Outer walk var `i` IS the target      */
                    /* position (= backtrack_t), so look up at index i - st0.  */
                    uint8_t *pr_scratch = slot_scratch +
                        (size_t)local_r * max_n_col_per_slot;
                    int idx_in_band = i - st0;
                    if (idx_in_band >= 0 && idx_in_band < (en0 - st0 + 1)) {
                        tmp_bt = pr_scratch[idx_in_band];
                    }
                }

                if (state == 0) {
                    state = tmp_bt & 7;
                } else {
                    if (!(tmp_bt >> (state + 2) & 1)) state = 0;
                }
                if (state == 0) state = tmp_bt & 7;
                if (force_state >= 0) state = force_state;

                if (state == 0) {
                    grid_push_cigar(&n_cigar_ops, max_cigar_len, cigar,
                                    KSW_CIGAR_MATCH, 1);
                    --i; --j;
                } else if (state == 1 || state == 3) {
                    grid_push_cigar(&n_cigar_ops, max_cigar_len, cigar,
                                    KSW_CIGAR_DEL, 1);
                    --i;
                } else {
                    grid_push_cigar(&n_cigar_ops, max_cigar_len, cigar,
                                    KSW_CIGAR_INS, 1);
                    --j;
                }
            }
            /* Broadcast i, j, n_cigar_ops, state so all lanes can decide on  */
            /* block-switch / loop continue.                                  */
            i             = __shfl_sync(0xffffffff, i, 0);
            j             = __shfl_sync(0xffffffff, j, 0);
            state         = __shfl_sync(0xffffffff, state, 0);
            n_cigar_ops   = __shfl_sync(0xffffffff, n_cigar_ops, 0);

            if (n_cigar_ops >= max_cigar_len - 2) break;
        }

        if (lane_id == 0) {
            if (i >= 0) {
                grid_push_cigar(&n_cigar_ops, max_cigar_len, cigar,
                                KSW_CIGAR_DEL, i + 1);
            }
            if (j >= 0) {
                grid_push_cigar(&n_cigar_ops, max_cigar_len, cigar,
                                KSW_CIGAR_INS, j + 1);
            }
            if (!is_rev) {
                for (int k = 0; k < n_cigar_ops / 2; ++k) {
                    uint32_t tmp_c = cigar[k];
                    cigar[k] = cigar[n_cigar_ops - 1 - k];
                    cigar[n_cigar_ops - 1 - k] = tmp_c;
                }
            }
            cigar_lengths[task_id] = n_cigar_ops;
        }
        __syncwarp();
    }
}

#endif /* USE_GRIDDED_BT */

#endif /* __PLGRID_KERNEL_CUH__ */
