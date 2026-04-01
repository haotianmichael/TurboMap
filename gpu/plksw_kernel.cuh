#ifndef __PLKSW2_KERNEL_CUH__
#define __PLKSW2_KERNEL_CUH__
#include "gasal_kernels.h"

/*
 * plksw_kernel.cuh — Anti-diagonal KSW2 kernel with direct int32 H/E/F/E2/F2
 *
 * Replaces the Suzuki-Kasahara int8 difference encoding with direct int32
 * values to avoid overflow on long sequences. Uses triple-buffered H arrays
 * (H from r-2 needed for diagonal, H from r-1 for gap open).
 *
 * Dual-affine gap: short gap (q+e) and long gap (q2+e2).
 * Anti-diagonal wavefront: all cells on diagonal r are independent.
 * Persistent kernel with atomic work-stealing.
 * Fused backtracking + CIGAR generation.
 */

#define KSW_EZ_SCORE_ONLY  0x01
#define KSW_EZ_RIGHT       0x02
#define KSW_EZ_GENERIC_SC  0x04
#define KSW_EZ_APPROX_MAX  0x08
#define KSW_EZ_APPROX_DROP 0x10
#define KSW_EZ_EXTZ_ONLY   0x40
#define KSW_EZ_REV_CIGAR   0x80
#define KSW_EZ_SPLICE_FOR  0x100
#define KSW_EZ_SPLICE_REV  0x200
#define KSW_EZ_SPLICE_FLANK 0x400

#define KSW_CIGAR_MATCH  0
#define KSW_CIGAR_INS    1
#define KSW_CIGAR_DEL    2
#define KSW_CIGAR_N_SKIP 3
#define KSW_NEG_INF     -0x40000000

#define WARP_SIZE 32

__device__ static inline uint32_t* ksw_push_cigar_device(
    int *n_cigar,
    int max_cigar_len,
    uint32_t *cigar,
    uint32_t op,
    int len)
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

__device__ __forceinline__ int8_t dp_compute_score(uint8_t a, uint8_t b, int8_t *mat, int m) {
    return (a < m && b < m) ? mat[a * m + b] : 0;
}

/*
 * Fused Persistent KSW Kernel — Direct int32 dual-affine
 *
 * Per-slot temp buffer layout:
 *   H_buf0[(tlen+1) * int32]   — triple-buffered H, buffer 0
 *   H_buf1[(tlen+1) * int32]   — triple-buffered H, buffer 1
 *   H_buf2[(tlen+1) * int32]   — triple-buffered H, buffer 2
 *   E[(tlen+1) * int32]        — short gap in query (vertical/deletion)
 *   F[(tlen+1) * int32]        — short gap in target (horizontal/insertion)
 *   E2[(tlen+1) * int32]       — long gap in query
 *   F2[(tlen+1) * int32]       — long gap in target
 *   qr[qlen * uint8]           — reversed query sequence
 *   target[tlen * uint8]       — target sequence
 *
 * Backtrack direction byte (same format as before):
 *   bits 0-2: state — 0=diag, 1=E, 2=F, 3=E2, 4=F2
 *   bit 3: E continuation (gap extend won over gap open)
 *   bit 4: F continuation
 *   bit 5: E2 continuation
 *   bit 6: F2 continuation
 */
__global__ void ksw_fused_persistent_kernel(
    int *d_task_counter,
    uint32_t *packed_query_batch,
    uint32_t *packed_ref_batch,
    uint32_t *query_batch_lens,
    uint32_t *target_batch_lens,
    uint32_t *query_batch_offsets,
    uint32_t *target_batch_offsets,
    gasal_res_t *device_res,
    int8_t *device_mat,
    uint8_t *backtrack_p,
    int *backtrack_off,
    int *backtrack_off_end,
    int max_backtrack_size,
    int max_antidiag,
    void *d_temp_buffer,
    int *d_flag,
    size_t temp_per_task,
    int n_tasks,
    int8_t m,
    int32_t zdrop,
    int end_bonus,
    uint32_t *cigar_buffer,
    int *cigar_lengths,
    int max_cigar_len
)
{
    const int slot_id = blockIdx.x;
    const int lane_id = threadIdx.x;

    while (true) {
        int task_id;
        if (lane_id == 0) {
            task_id = atomicAdd(d_task_counter, 1);
        }
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
                if (cigar_buffer) cigar_lengths[task_id] = 0;
            }
            __syncwarp();
            continue;
        }

        // ========== Gap / Band Parameters ==========
        int8_t q  = _cudaGapO;
        int8_t e  = _cudaGapExtend;
        int8_t q2 = _cudaGapOL;
        int8_t e2 = _cudaGapExtendL;
        int32_t w = _cudaBandWidth;

        // Ensure q+e <= q2+e2 (short gap <= long gap penalty)
        if (q2 + e2 < q + e) {
            int8_t tmp = q; q = q2; q2 = tmp;
            tmp = e; e = e2; e2 = tmp;
        }
        int32_t qe  = q  + e;
        int32_t qe2 = q2 + e2;

        int wl = (w < 0) ? max(qlen, tlen) : w;
        int wr = (w < 0) ? max(qlen, tlen) : w;

        // ========== Slot-indexed Temp Buffer ==========
        char *task_buf = (char*)d_temp_buffer + (size_t)slot_id * temp_per_task;

        size_t arr_len = (size_t)(tlen + 1);
        size_t offset = 0;
        int32_t *H_bufs[3];
        H_bufs[0] = (int32_t*)(task_buf + offset); offset += arr_len * sizeof(int32_t);
        H_bufs[1] = (int32_t*)(task_buf + offset); offset += arr_len * sizeof(int32_t);
        H_bufs[2] = (int32_t*)(task_buf + offset); offset += arr_len * sizeof(int32_t);
        int32_t *E_arr  = (int32_t*)(task_buf + offset); offset += arr_len * sizeof(int32_t);
        int32_t *F_arr  = (int32_t*)(task_buf + offset); offset += arr_len * sizeof(int32_t);
        int32_t *E2_arr = (int32_t*)(task_buf + offset); offset += arr_len * sizeof(int32_t);
        int32_t *F2_arr = (int32_t*)(task_buf + offset); offset += arr_len * sizeof(int32_t);
        uint8_t *qr     = (uint8_t*)(task_buf + offset); offset += qlen * sizeof(uint8_t);
        uint8_t *target = (uint8_t*)(task_buf + offset);

        // ========== Initialization ==========
        // H boundary: H[-1][j] = 0, H[i][-1] = 0
        // Initialize all H buffers to 0 (boundary value)
        for (int i = lane_id; i < (int)arr_len; i += WARP_SIZE) {
            H_bufs[0][i] = 0;
            H_bufs[1][i] = 0;
            H_bufs[2][i] = 0;
            E_arr[i]  = KSW_NEG_INF;
            F_arr[i]  = KSW_NEG_INF;
            E2_arr[i] = KSW_NEG_INF;
            F2_arr[i] = KSW_NEG_INF;
        }

        // ========== Sequence Unpacking ==========
        int packed_query_offset  = query_batch_offsets[task_id] >> 3;
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

        // ========== DP Configuration ==========
        int last_H0_t = 0;
        int32_t H0 = 0;

        int n_col = (qlen < tlen) ? qlen : tlen;
        n_col = (n_col < w + 1) ? n_col : (w + 1);
        n_col = ((n_col + 15) / 16 + 1) * 16;

        int with_cigar = !(flag & KSW_EZ_SCORE_ONLY);
        int approx_max = !!(flag & KSW_EZ_APPROX_MAX);
        int right_align = !!(flag & KSW_EZ_RIGHT);

        // Slot-indexed backtrack buffers
        uint8_t *p_bt   = backtrack_p   + (size_t)slot_id * max_backtrack_size;
        int     *off     = backtrack_off     + (size_t)slot_id * max_antidiag;
        int     *off_end = backtrack_off_end + (size_t)slot_id * max_antidiag;

        // ========== Main DP Loop: Anti-diagonal Traversal ==========
        int last_st = -1, last_en = -1;  // previous anti-diagonal band boundaries
        for (int r = 0; r < qlen + tlen - 1; r++) {
            // Band boundaries
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

            // Record backtrack range
            if (lane_id == 0 && with_cigar) {
                off[r] = st0;
                off_end[r] = en0;
            }

            // Triple-buffer rotation
            int32_t *H_cur   = H_bufs[r % 3];       // current anti-diagonal's H
            int32_t *H_prev  = H_bufs[(r + 2) % 3]; // anti-diagonal r-1
            int32_t *H_pprev = H_bufs[(r + 1) % 3]; // anti-diagonal r-2

            uint8_t *pr = (with_cigar && cigar_buffer) ? (p_bt + (size_t)r * n_col) : NULL;

            // ========== Parallel DP: all cells on anti-diagonal r ==========
            for (int idx = lane_id; idx <= (en0 - st0); idx += WARP_SIZE) {
                int t = st0 + idx;             // target position j
                int qi = r - t;                // query position i (in original coords)
                int qi_rev = qlen - 1 - qi;    // reversed query index

                // Substitution score
                int8_t score;
                if (qi >= 0 && qi < qlen && t >= 0 && t < tlen) {
                    score = dp_compute_score(qr[qi_rev], target[t], device_mat, m);
                } else {
                    score = 0;
                }

                // Diagonal: H[i-1][j-1] from anti-diagonal r-2
                int32_t H_diag;
                if (qi == 0 || t == 0) {
                    H_diag = 0;  // boundary: H[-1][j] = H[i][-1] = 0
                } else {
                    H_diag = H_pprev[t - 1];
                }

                // Gap in query (E): vertical move (i-1,j) -> (i,j)
                // E[i][j] = max(H[i-1][j] - qe, E[i-1][j] - e)
                int32_t H_up;
                if (qi == 0) {
                    H_up = 0;  // H[-1][j] = 0
                } else {
                    H_up = H_prev[t];
                }
                int32_t e_open  = H_up - qe;
                // E_arr[t] is valid only if t was in band at r-1
                int32_t e_ext;
                if (t >= last_st && t <= last_en) {
                    e_ext = E_arr[t] - e;
                } else {
                    e_ext = KSW_NEG_INF;  // out-of-band: no valid E from previous
                }
                int32_t E_val   = max(e_open, e_ext);

                // Gap in target (F): horizontal move (i,j-1) -> (i,j)
                // F[i][j] = max(H[i][j-1] - qe, F[i][j-1] - e)
                int32_t H_left;
                if (t == 0) {
                    H_left = 0;  // H[i][-1] = 0
                } else {
                    H_left = H_prev[t - 1];
                }
                int32_t f_open  = H_left - qe;
                // F_arr[t-1] is valid only if t-1 was in band at r-1
                int32_t f_ext;
                int32_t F_val;
                if (t == 0) {
                    F_val = f_open;  // boundary: no F extension
                } else if (t - 1 >= last_st && t - 1 <= last_en) {
                    f_ext = F_arr[t - 1] - e;
                    F_val = max(f_open, f_ext);
                } else {
                    f_ext = KSW_NEG_INF;  // out-of-band: no valid F from previous
                    F_val = f_open;
                }

                // Long gap in query (E2)
                int32_t e2_open = H_up - qe2;
                int32_t e2_ext;
                if (t >= last_st && t <= last_en) {
                    e2_ext = E2_arr[t] - e2;
                } else {
                    e2_ext = KSW_NEG_INF;
                }
                int32_t E2_val  = max(e2_open, e2_ext);

                // Long gap in target (F2)
                int32_t f2_open = H_left - qe2;
                int32_t f2_ext;
                int32_t F2_val;
                if (t == 0) {
                    F2_val = f2_open;
                } else if (t - 1 >= last_st && t - 1 <= last_en) {
                    f2_ext = F2_arr[t - 1] - e2;
                    F2_val = max(f2_open, f2_ext);
                } else {
                    f2_ext = KSW_NEG_INF;
                    F2_val = f2_open;
                }

                // H[i][j] = max(H_diag + score, E, F, E2, F2)
                int32_t H_val = H_diag + (int32_t)score;
                uint8_t d = 0;

                if (!with_cigar) {
                    H_val = max(H_val, E_val);
                    H_val = max(H_val, F_val);
                    H_val = max(H_val, E2_val);
                    H_val = max(H_val, F2_val);
                } else if (!right_align) {
                    // State mapping must match ksw2 backtrack convention:
                    // state 1 = horizontal gap (F, target gap) = DEL → --i
                    // state 2 = vertical gap (E, query gap)   = INS → --j
                    // state 3 = long horizontal (F2)          = DEL
                    // state 4 = long vertical (E2)            = INS
                    if (F_val  > H_val) { H_val = F_val;  d = 1; }
                    if (E_val  > H_val) { H_val = E_val;  d = 2; }
                    if (F2_val > H_val) { H_val = F2_val; d = 3; }
                    if (E2_val > H_val) { H_val = E2_val; d = 4; }
                } else {
                    if (!(H_val > F_val))  { H_val = F_val;  d = 1; }
                    if (!(H_val > E_val))  { H_val = E_val;  d = 2; }
                    if (!(H_val > F2_val)) { H_val = F2_val; d = 3; }
                    if (!(H_val > E2_val)) { H_val = E2_val; d = 4; }
                }

                // Backtrack continuation bits — must match state numbering:
                // bit 3 (0x08) = state 1 continuation = F (horizontal gap extension)
                // bit 4 (0x10) = state 2 continuation = E (vertical gap extension)
                // bit 5 (0x20) = state 3 continuation = F2 (long horizontal)
                // bit 6 (0x40) = state 4 continuation = E2 (long vertical)
                if (with_cigar) {
                    if (!right_align) {
                        if (f_ext > f_open && t > 0)       d |= 0x08;
                        if (e_ext > e_open)                d |= 0x10;
                        if (f2_ext > f2_open && t > 0)     d |= 0x20;
                        if (e2_ext > e2_open)              d |= 0x40;
                    } else {
                        if (f_ext >= f_open && t > 0)      d |= 0x08;
                        if (e_ext >= e_open)               d |= 0x10;
                        if (f2_ext >= f2_open && t > 0)    d |= 0x20;
                        if (e2_ext >= e2_open)             d |= 0x40;
                    }
                }

                // Store results
                H_cur[t]   = H_val;
                E_arr[t]   = E_val;
                F_arr[t]   = F_val;
                E2_arr[t]  = E2_val;
                F2_arr[t]  = F2_val;

                if (pr != NULL) {
                    pr[t - st0] = d;
                }
            }
            __syncwarp();

            // ========== Track Maximum Score (lane 0 only) ==========
            if (lane_id == 0) {
                if (!approx_max) {
                    int32_t max_H = KSW_NEG_INF;
                    int max_t_pos = st0;

                    for (int t = st0; t <= en0; ++t) {
                        if (H_cur[t] > max_H) {
                            max_H = H_cur[t];
                            max_t_pos = t;
                        }
                    }

                    int j = max_t_pos;
                    int i = r - j;
                    if (max_H > ez_max) {
                        ez_max = max_H;
                        ez_max_t = j;
                        ez_max_q = i;
                    }

                    // Z-drop check
                    if (j >= ez_max_t && i >= ez_max_q) {
                        int tl = j - ez_max_t, ql = i - ez_max_q;
                        int l = (tl > ql) ? (tl - ql) : (ql - tl);
                        if (zdrop >= 0 && ez_max - max_H > zdrop + l * e2) {
                            ez_zdropped = 1;
                        }
                    }

                    // Check boundary conditions
                    if (en0 == tlen - 1 && H_cur[en0] > ez_mte) {
                        ez_mte = H_cur[en0];
                        ez_mte_q = r - en0;
                    }
                    if (r - st0 == qlen - 1 && st0 >= 0 && st0 < tlen && H_cur[st0] > ez_mqe) {
                        ez_mqe = H_cur[st0];
                        ez_mqe_t = st0;
                    }
                    if (r == qlen + tlen - 2 && en0 == tlen - 1) {
                        ez_score = H_cur[tlen - 1];
                    }
                } else {
                    // Approximate max tracking
                    if (r > 0) {
                        if (last_H0_t >= st0 && last_H0_t <= en0 &&
                            last_H0_t + 1 >= st0 && last_H0_t + 1 <= en0) {
                            int32_t h0 = H_cur[last_H0_t];
                            int32_t h1 = H_cur[last_H0_t + 1];
                            if (h0 >= h1) {
                                H0 = h0;
                            } else {
                                H0 = h1;
                                ++last_H0_t;
                            }
                        } else if (last_H0_t >= st0 && last_H0_t <= en0) {
                            H0 = H_cur[last_H0_t];
                        } else {
                            ++last_H0_t;
                            H0 = H_cur[last_H0_t];
                        }
                    } else {
                        H0 = H_cur[0];
                        last_H0_t = 0;
                    }
                    if (r == qlen + tlen - 2 && en0 == tlen - 1) {
                        ez_score = H0;
                    }
                }
            }

            // Update band boundaries for next anti-diagonal
            last_st = st0;
            last_en = en0;

            int zdropped_flag = __shfl_sync(0xffffffff, ez_zdropped, 0);
            if (zdropped_flag) break;
        } // End anti-diagonal loop

        // ========== Write Results ==========
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

            device_res->aln_score[task_id]        = ez_zdropped ? ez_max : ez_score;
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

        // ========== Fused Backtrack + CIGAR ==========
        if (cigar_buffer && with_cigar && lane_id == 0 &&
            backtrack_q >= 0 && backtrack_t >= 0) {

            int i0 = backtrack_t;
            int j0 = backtrack_q;
            int i = i0, j = j0;
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

                if (state == 0) {
                    state = tmp_bt & 7;
                } else {
                    if (!(tmp_bt >> (state + 2) & 1)) state = 0;
                }
                if (state == 0) state = tmp_bt & 7;
                if (force_state >= 0) state = force_state;

                if (state == 0) {
                    ksw_push_cigar_device(&n_cigar_ops, max_cigar_len, cigar,
                                          KSW_CIGAR_MATCH, 1);
                    --i; --j;
                } else if (state == 1 || state == 3) {
                    ksw_push_cigar_device(&n_cigar_ops, max_cigar_len, cigar,
                                          KSW_CIGAR_DEL, 1);
                    --i;
                } else {
                    ksw_push_cigar_device(&n_cigar_ops, max_cigar_len, cigar,
                                          KSW_CIGAR_INS, 1);
                    --j;
                }
                if (n_cigar_ops >= max_cigar_len - 2) break;
            }

            if (i >= 0) {
                ksw_push_cigar_device(&n_cigar_ops, max_cigar_len, cigar,
                                      KSW_CIGAR_DEL, i + 1);
            }
            if (j >= 0) {
                ksw_push_cigar_device(&n_cigar_ops, max_cigar_len, cigar,
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
        } else if (cigar_buffer && lane_id == 0) {
            cigar_lengths[task_id] = 0;
        }

        __syncwarp();
    } // persistent loop
}

#endif
