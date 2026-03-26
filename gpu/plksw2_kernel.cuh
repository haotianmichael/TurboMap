/*
 * plksw2_kernel.cuh — CUDASW4-style Column-Parallel KSW2 Kernel
 *
 * Design:
 *   - Column-parallel DP: each thread holds NUM_REGS consecutive target
 *     positions in registers.  With GROUP_SIZE=32 × NUM_REGS=32 the kernel
 *     covers up to 1024 bp targets in a single pass.
 *   - Dual affine gap (q+e / q2+e2) matching minimap2's ksw_extd.
 *   - Standard DP formulation (int32_t H/E/F/E2/F2) — no Suzuki-Kasahara.
 *   - Row-major backtrack: bt[qi * tlen + tj] — simpler than anti-diagonal.
 *   - Fused backtrack / CIGAR generation (lane 0 serial).
 *   - Persistent kernel with atomic work-stealing.
 *   - No Z-drop, no banding.
 *
 * Backtrack direction byte (same format as original ksw kernel):
 *   bits 0-2: state — 0=diag, 1=E, 2=F, 3=E2, 4=F2
 *   bit 3:    E continuation (gap extension won over gap open)
 *   bit 4:    F continuation
 *   bit 5:    E2 continuation
 *   bit 6:    F2 continuation
 */

#ifndef __PLKSW2_COL_KERNEL_CUH__
#define __PLKSW2_COL_KERNEL_CUH__

#include "gasal_kernels.h"

/* ---- KSW flag macros (duplicated from plksw_kernel.cuh for independence) ---- */
#ifndef KSW_EZ_SCORE_ONLY
#define KSW_EZ_SCORE_ONLY   0x01
#define KSW_EZ_RIGHT        0x02
#define KSW_EZ_APPROX_MAX   0x08
#define KSW_EZ_EXTZ_ONLY    0x40
#define KSW_EZ_REV_CIGAR    0x80
#endif

#ifndef KSW_CIGAR_MATCH
#define KSW_CIGAR_MATCH  0
#define KSW_CIGAR_INS    1
#define KSW_CIGAR_DEL    2
#define KSW_CIGAR_N_SKIP 3
#endif

#ifndef KSW_NEG_INF
#define KSW_NEG_INF (-0x40000000)
#endif

/* ---- Column-parallel constants ---- */
#define KSW2_NUM_REGS   32
#define KSW2_GROUP_SIZE 32
#define KSW2_MAX_TARGET (KSW2_GROUP_SIZE * KSW2_NUM_REGS)  /* 1024 */
#define KSW2_WARPS_PER_BLOCK 4   /* multiple warps per block for latency hiding */

/* Negative infinity for int32 DP (headroom for gap penalties) */
#define KSW2_NEG_INF32  (-0x10000000)

/* ------------------------------------------------------------------ */
/* Inline CIGAR push (same as original kernel)                        */
/* ------------------------------------------------------------------ */
__device__ static inline void ksw2_push_cigar(
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

/* ------------------------------------------------------------------ */
/* Warp-level argmax reduction: returns (max_val, max_pos)            */
/* All lanes receive the result after the call.                       */
/* ------------------------------------------------------------------ */
__device__ static inline void warp_argmax_broadcast(
    int32_t &val, int32_t &pos)
{
    for (int offset = 16; offset > 0; offset >>= 1) {
        int32_t v2 = __shfl_down_sync(0xffffffff, val, offset);
        int32_t p2 = __shfl_down_sync(0xffffffff, pos, offset);
        if (v2 > val) { val = v2; pos = p2; }
    }
    val = __shfl_sync(0xffffffff, val, 0);
    pos = __shfl_sync(0xffffffff, pos, 0);
}

/* ------------------------------------------------------------------ */
/* Main persistent kernel                                             */
/* ------------------------------------------------------------------ */
__global__ void ksw2_col_persistent_kernel(
    int          *d_task_counter,
    uint32_t     *packed_query_batch,
    uint32_t     *packed_ref_batch,
    uint32_t     *query_batch_lens,
    uint32_t     *target_batch_lens,
    uint32_t     *query_batch_offsets,
    uint32_t     *target_batch_offsets,
    gasal_res_t  *device_res,
    int8_t       *device_mat,          /* 5×5 scoring matrix             */
    uint8_t      *backtrack_p,         /* slot-indexed [slots×bt_size]   */
    int           max_bt_size,         /* max qlen × max tlen per slot   */
    void         *d_temp_buffer,       /* slot-indexed [slots×temp_pp]   */
    int32_t      *d_flag,
    size_t        temp_per_task,
    int           n_tasks,
    int           max_slots,           /* total available buffer slots    */
    int8_t        m,                   /* alphabet size (5=ACGTN)        */
    int           end_bonus,
    uint32_t     *cigar_buffer,        /* task-indexed (NULL → no CIGAR) */
    int          *cigar_lengths,       /* task-indexed                   */
    int           max_cigar_len
)
{
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int warps_per_block = blockDim.x / 32;
    const int slot_id = blockIdx.x * warps_per_block + warp_id;

    /* Guard: last block may have excess warps beyond available slots */
    if (slot_id >= max_slots) return;

    /* ================================================================ */
    /* Persistent task loop                                             */
    /* ================================================================ */
    while (true) {
        /* --- Claim next task --- */
        int task_id;
        if (lane_id == 0)
            task_id = atomicAdd(d_task_counter, 1);
        task_id = __shfl_sync(0xffffffff, task_id, 0);
        if (task_id >= n_tasks) return;

        /* ---- Task parameters ---- */
        int qlen = query_batch_lens[task_id];
        int tlen = target_batch_lens[task_id];
        int flag = d_flag[task_id];

        /* Handle empty tasks */
        if (qlen <= 0 || tlen <= 0) {
            if (lane_id == 0) {
                device_res->aln_score[task_id]        = 0;
                device_res->query_batch_end[task_id]  = -1;
                device_res->target_batch_end[task_id] = -1;
                device_res->mqe[task_id]   = KSW_NEG_INF;
                device_res->mqe_t[task_id] = -1;
                device_res->mte[task_id]   = KSW_NEG_INF;
                device_res->mte_q[task_id] = -1;
                if (cigar_buffer) cigar_lengths[task_id] = 0;
            }
            __syncwarp();
            continue;
        }

        /* ---- Gap penalties (from constant memory) ---- */
        int32_t q_pen  = _cudaGapO;
        int32_t e_pen  = _cudaGapExtend;
        int32_t q2_pen = _cudaGapOL;
        int32_t e2_pen = _cudaGapExtendL;

        /* Ensure q+e <= q2+e2 (short gap model has smaller opening cost) */
        if (q2_pen + e2_pen < q_pen + e_pen) {
            int32_t tmp;
            tmp = q_pen;  q_pen  = q2_pen; q2_pen = tmp;
            tmp = e_pen;  e_pen  = e2_pen; e2_pen = tmp;
        }
        int32_t qe  = q_pen  + e_pen;
        int32_t qe2 = q2_pen + e2_pen;

        int with_cigar  = !(flag & KSW_EZ_SCORE_ONLY);
        int right_align = !!(flag & KSW_EZ_RIGHT);

        /* ---- Slot-indexed temp buffer: [query | target] ---- */
        char    *task_buf = (char *)d_temp_buffer + (size_t)slot_id * temp_per_task;
        uint8_t *qr       = (uint8_t *)task_buf;
        uint8_t *target   = qr + qlen;

        /* ---- Unpack sequences (warp-parallel) ---- */
        int packed_query_off  = query_batch_offsets[task_id]  >> 3;
        int packed_target_off = target_batch_offsets[task_id] >> 3;

        for (int i = lane_id; i < qlen; i += 32) {
            int pidx = i / 8;
            int boff = (7 - (i % 8)) * 4;
            uint32_t pv = packed_query_batch[packed_query_off + pidx];
            qr[i] = (pv >> boff) & 0xF;   /* forward query (no reversal needed for row-major DP) */
        }
        for (int i = lane_id; i < tlen; i += 32) {
            int pidx = i / 8;
            int boff = (7 - (i % 8)) * 4;
            uint32_t pv = packed_ref_batch[packed_target_off + pidx];
            target[i] = (pv >> boff) & 0xF;
        }
        __syncwarp();

        /* ---- Slot-indexed backtrack buffer ---- */
        uint8_t *bt = backtrack_p + (size_t)slot_id * max_bt_size;

        /* ============================================================ */
        /* Load target bases into registers                             */
        /* ============================================================ */
        int32_t tb[KSW2_NUM_REGS];   /* target base at each register */
        #pragma unroll
        for (int r = 0; r < KSW2_NUM_REGS; r++) {
            int tj = lane_id * KSW2_NUM_REGS + r;
            tb[r] = (tj < tlen) ? (int32_t)target[tj] : (int32_t)(m - 1);
        }

        /* ============================================================ */
        /* DP register arrays — per thread, per target position         */
        /* H_arr[r] = H[prev_row][tj] after initialization             */
        /* F_arr[r] = F value for the CURRENT row (pre-computed)        */
        /* ============================================================ */
        int32_t H_arr[KSW2_NUM_REGS];
        int32_t F_arr[KSW2_NUM_REGS];
        int32_t F2_arr[KSW2_NUM_REGS];

        /* F continuation bitmasks: bit r = 1 if F[qi][tj] at register r
         * was computed by extending F[qi-1][tj] rather than opening from H[qi-1][tj].
         * Tracked per pre-computation to provide correct backtrack direction. */
        uint32_t F_cont_bits  = 0;
        uint32_t F2_cont_bits = 0;

        /* gap_cost(k) = min(q + k*e, q2 + k*e2) */
        #pragma unroll
        for (int r = 0; r < KSW2_NUM_REGS; r++) {
            int tj = lane_id * KSW2_NUM_REGS + r;
            if (tj < tlen) {
                int k = tj + 1;
                int32_t gc1 = q_pen + k * e_pen;
                int32_t gc2 = q2_pen + k * e2_pen;
                H_arr[r] = -(gc1 < gc2 ? gc1 : gc2);
            } else {
                H_arr[r] = KSW2_NEG_INF32;
            }
            /* F[0][tj] = H[-1][tj] - qe  (F[-1][tj] = -inf, so always open) */
            F_arr[r]  = H_arr[r] - qe;
            F2_arr[r] = H_arr[r] - qe2;
            /* F[0] is always an opening (not extension), so bits stay 0 */
        }

        /* ---- Score tracking ---- */
        int32_t ez_max   = 0,  ez_max_q = -1, ez_max_t = -1;
        int32_t ez_mqe   = KSW_NEG_INF, ez_mqe_t = -1;
        int32_t ez_mte   = KSW_NEG_INF, ez_mte_q = -1;
        int32_t ez_score = KSW_NEG_INF;

        /* H_last: the last register's H from the previous row,
         * shuffled to the next lane to provide the diagonal for register 0. */
        int32_t H_last_prev = H_arr[KSW2_NUM_REGS - 1];

        /* E_out / E2_out: E value at the END of the register loop
         * (position lane*NR + NR-1) from the previous row.
         * Shuffled to the next lane at the start of each row.
         * (CUDASW4 approximate pattern: E from row qi-1 of lane k
         * is used as starting E for row qi of lane k+1.) */
        int32_t E_out_prev  = KSW2_NEG_INF32;
        int32_t E2_out_prev = KSW2_NEG_INF32;
        int32_t E_cont_out_prev  = 0;
        int32_t E2_cont_out_prev = 0;

        /* ============================================================ */
        /* Main DP loop — iterate over query rows                       */
        /* ============================================================ */
        for (int qi = 0; qi < qlen; qi++) {
            /* Broadcast query base to all lanes */
            int32_t qi_base;
            if (lane_id == 0) qi_base = qr[qi];
            qi_base = __shfl_sync(0xffffffff, qi_base, 0);

            /* Boundary values for lane 0 */
            int32_t k_qi = qi + 1;
            int32_t gc1_qi = q_pen + k_qi * e_pen;
            int32_t gc2_qi = q2_pen + k_qi * e2_pen;
            int32_t H_boundary_qi = -(gc1_qi < gc2_qi ? gc1_qi : gc2_qi);
            /* H[-1][-1] = 0 for qi=0, else H[qi-1][-1] = boundary of previous row */
            int32_t H_diag_boundary;
            if (qi == 0) {
                H_diag_boundary = 0;  /* H[-1][-1] */
            } else {
                int32_t k_prev = qi;  /* (qi-1)+1 */
                int32_t g1 = q_pen + k_prev * e_pen;
                int32_t g2 = q2_pen + k_prev * e2_pen;
                H_diag_boundary = -(g1 < g2 ? g1 : g2);
            }

            /* ---- Shuffle H_last from previous row for diagonal ---- */
            int32_t H_diag = __shfl_up_sync(0xffffffff, H_last_prev, 1);
            if (lane_id == 0) H_diag = H_diag_boundary;

            /* ---- Shuffle E, E2 from previous lane (CUDASW4 pattern) ---- */
            /* Lane k+1 receives E from lane k's PREVIOUS row (approximate).
             * Lane 0 gets boundary E from H[qi][-1]. */
            int32_t E  = __shfl_up_sync(0xffffffff, E_out_prev, 1);
            int32_t E2 = __shfl_up_sync(0xffffffff, E2_out_prev, 1);
            int32_t E_cont  = __shfl_up_sync(0xffffffff, E_cont_out_prev, 1);
            int32_t E2_cont = __shfl_up_sync(0xffffffff, E2_cont_out_prev, 1);
            if (lane_id == 0) {
                E       = H_boundary_qi - qe;
                E2      = H_boundary_qi - qe2;
                E_cont  = 0;
                E2_cont = 0;
            }

            /* ---- Inner loop over registers ---- */
            int32_t local_max_val = KSW2_NEG_INF32;
            int32_t local_max_pos = -1;

            /* Track mqe (last query row) per thread */
            int32_t local_mqe_val = KSW2_NEG_INF32;
            int32_t local_mqe_pos = -1;

            #pragma unroll
            for (int r = 0; r < KSW2_NUM_REGS; r++) {
                int tj = lane_id * KSW2_NUM_REGS + r;

                /* Substitution score */
                int32_t score = (qi_base < m && tb[r] < (m - 1) && tj < tlen)
                    ? (int32_t)device_mat[qi_base * m + tb[r]]
                    : (int32_t)device_mat[(m - 1) * m + (m - 1)]; /* N vs N */

                /* Previous-row H at this position (will become diagonal
                 * for next register) */
                int32_t H_prev = H_arr[r];

                /* ---- Compute H[qi][tj] ---- */
                int32_t H_diag_score = H_diag + score;
                int32_t H_new = H_diag_score;
                uint8_t d = 0;  /* direction byte */

                if (tj < tlen) {
                    if (!right_align) {
                        if (E  > H_new) { H_new = E;  d = 1; }
                        if (F_arr[r]  > H_new) { H_new = F_arr[r];  d = 2; }
                        if (E2 > H_new) { H_new = E2; d = 3; }
                        if (F2_arr[r] > H_new) { H_new = F2_arr[r]; d = 4; }
                    } else {
                        if (!(H_new > E )) { H_new = E;  d = 1; }
                        if (!(H_new > F_arr[r] )) { H_new = F_arr[r];  d = 2; }
                        if (!(H_new > E2)) { H_new = E2; d = 3; }
                        if (!(H_new > F2_arr[r])) { H_new = F2_arr[r]; d = 4; }
                    }

                    /* ---- Continuation bits ---- */
                    if (with_cigar) {
                        /* E continuation: tracked as propagating scalar */
                        if (!right_align) {
                            if (E_cont)  d |= 0x08;
                            if ((F_cont_bits >> r) & 1)  d |= 0x10;
                            if (E2_cont) d |= 0x20;
                            if ((F2_cont_bits >> r) & 1) d |= 0x40;
                        } else {
                            if (E_cont)  d |= 0x08;
                            if ((F_cont_bits >> r) & 1)  d |= 0x10;
                            if (E2_cont) d |= 0x20;
                            if ((F2_cont_bits >> r) & 1) d |= 0x40;
                        }
                    }

                    /* ---- Store direction byte (row-major) ---- */
                    if (with_cigar && cigar_buffer)
                        bt[(size_t)qi * tlen + tj] = d;

                    /* ---- Update H ---- */
                    H_arr[r] = H_new;

                    /* ---- Update E for next register ---- */
                    int32_t E_from_open = H_new - qe;
                    int32_t E_from_ext  = E - e_pen;
                    if (!right_align) {
                        E_cont = (E_from_ext > E_from_open) ? 1 : 0;
                    } else {
                        E_cont = (!(E_from_open > E_from_ext)) ? 1 : 0;
                    }
                    E = (E_from_ext > E_from_open) ? E_from_ext : E_from_open;

                    int32_t E2_from_open = H_new - qe2;
                    int32_t E2_from_ext  = E2 - e2_pen;
                    if (!right_align) {
                        E2_cont = (E2_from_ext > E2_from_open) ? 1 : 0;
                    } else {
                        E2_cont = (!(E2_from_open > E2_from_ext)) ? 1 : 0;
                    }
                    E2 = (E2_from_ext > E2_from_open) ? E2_from_ext : E2_from_open;

                    /* ---- Pre-compute F for next row ---- */
                    int32_t F_from_open  = H_new - qe;
                    int32_t F_from_ext   = F_arr[r] - e_pen;
                    F_arr[r] = (F_from_ext > F_from_open) ? F_from_ext : F_from_open;
                    if (!right_align) {
                        if (F_from_ext > F_from_open) F_cont_bits |= (1u << r); else F_cont_bits &= ~(1u << r);
                    } else {
                        if (!(F_from_open > F_from_ext)) F_cont_bits |= (1u << r); else F_cont_bits &= ~(1u << r);
                    }

                    int32_t F2_from_open = H_new - qe2;
                    int32_t F2_from_ext  = F2_arr[r] - e2_pen;
                    F2_arr[r] = (F2_from_ext > F2_from_open) ? F2_from_ext : F2_from_open;
                    if (!right_align) {
                        if (F2_from_ext > F2_from_open) F2_cont_bits |= (1u << r); else F2_cont_bits &= ~(1u << r);
                    } else {
                        if (!(F2_from_open > F2_from_ext)) F2_cont_bits |= (1u << r); else F2_cont_bits &= ~(1u << r);
                    }

                    /* ---- Score tracking ---- */
                    if (H_new > local_max_val) {
                        local_max_val = H_new;
                        local_max_pos = tj;
                    }

                    /* mqe: score at the last query row */
                    if (qi == qlen - 1 && H_new > local_mqe_val) {
                        local_mqe_val = H_new;
                        local_mqe_pos = tj;
                    }

                    /* mte: score at the last target column */
                    if (tj == tlen - 1 && H_new > ez_mte) {
                        ez_mte   = H_new;
                        ez_mte_q = qi;
                    }

                    /* ez_score: H[qlen-1][tlen-1] */
                    if (qi == qlen - 1 && tj == tlen - 1) {
                        ez_score = H_new;
                    }
                } else {
                    /* Beyond target — keep -inf */
                    H_arr[r] = KSW2_NEG_INF32;
                }

                /* Diagonal for next register = previous row's H at this position */
                H_diag = H_prev;
            } /* end register loop */

            /* Save E_out for shuffle in next row (CUDASW4 pattern) */
            E_out_prev       = E;
            E2_out_prev      = E2;
            E_cont_out_prev  = E_cont;
            E2_cont_out_prev = E2_cont;

            /* Save H_last for shuffle in next row */
            H_last_prev = H_arr[KSW2_NUM_REGS - 1];

            /* ---- Warp-level max reduction for this row ---- */
            int32_t row_max_val = local_max_val;
            int32_t row_max_pos = local_max_pos;
            /* Encode (qi, tj) for the global max: qi is the same for all
             * lanes in this row, so we only need to reduce over tj. */
            warp_argmax_broadcast(row_max_val, row_max_pos);
            if (row_max_val > ez_max) {
                ez_max   = row_max_val;
                ez_max_q = qi;
                ez_max_t = row_max_pos;
            }

            /* mqe warp reduction (only at last query row) */
            if (qi == qlen - 1) {
                int32_t mqe_val = local_mqe_val;
                int32_t mqe_pos = local_mqe_pos;
                warp_argmax_broadcast(mqe_val, mqe_pos);
                ez_mqe   = mqe_val;
                ez_mqe_t = mqe_pos;
            }
        } /* end query row loop */

        /* ============================================================ */
        /* Warp-reduce ez_score, ez_mte, ez_mte_q                      */
        /* These may have been set by a lane other than lane 0          */
        /* ============================================================ */
        {
            /* ez_score: only the lane holding tj==tlen-1 has the value;
             * all other lanes have KSW_NEG_INF.  Max-reduce to lane 0. */
            int32_t sc = ez_score;
            for (int off = 16; off > 0; off >>= 1) {
                int32_t v = __shfl_down_sync(0xffffffff, sc, off);
                if (v > sc) sc = v;
            }
            ez_score = __shfl_sync(0xffffffff, sc, 0);

            /* ez_mte / ez_mte_q: argmax reduce (same as mqe) */
            int32_t mte_v = ez_mte;
            int32_t mte_q = ez_mte_q;
            warp_argmax_broadcast(mte_v, mte_q);
            ez_mte   = mte_v;
            ez_mte_q = mte_q;
        }

        /* ============================================================ */
        /* Determine backtrack endpoint                                 */
        /* ============================================================ */
        int backtrack_q = -1, backtrack_t = -1;
        if (lane_id == 0) {
            if (!(flag & KSW_EZ_EXTZ_ONLY)) {
                /* Global/semi-global: backtrack from corner */
                backtrack_q = qlen - 1;
                backtrack_t = tlen - 1;
            } else if ((flag & KSW_EZ_EXTZ_ONLY) &&
                       ez_mqe + end_bonus > ez_max) {
                backtrack_q = qlen - 1;
                backtrack_t = ez_mqe_t;
            } else if (ez_max_t >= 0 && ez_max_q >= 0) {
                backtrack_q = ez_max_q;
                backtrack_t = ez_max_t;
            }

            device_res->aln_score[task_id]        = ez_score;
            device_res->query_batch_end[task_id]  = backtrack_q;
            device_res->target_batch_end[task_id] = backtrack_t;
            device_res->mqe[task_id]              = ez_mqe;
            device_res->mqe_t[task_id]            = ez_mqe_t;
            device_res->mte[task_id]              = ez_mte;
            device_res->mte_q[task_id]            = ez_mte_q;
        }
        backtrack_q = __shfl_sync(0xffffffff, backtrack_q, 0);
        backtrack_t = __shfl_sync(0xffffffff, backtrack_t, 0);

        /* ============================================================ */
        /* Fused backtrack — lane 0 serial                              */
        /* Row-major direction bytes: bt[qi * tlen + tj]                */
        /* ============================================================ */
        if (cigar_buffer && with_cigar && lane_id == 0 &&
            backtrack_q >= 0 && backtrack_t >= 0) {

            int i = backtrack_t;   /* target position */
            int j = backtrack_q;   /* query position  */
            int state = 0;
            int n_cigar = 0;
            int is_rev = !!(flag & KSW_EZ_REV_CIGAR);

            uint32_t *cigar = cigar_buffer + (size_t)task_id * max_cigar_len;

            while (i >= 0 && j >= 0) {
                uint8_t tmp_bt = bt[(size_t)j * tlen + i];

                if (state == 0) {
                    state = tmp_bt & 7;
                } else {
                    if (!(tmp_bt >> (state + 2) & 1)) state = 0;
                }
                if (state == 0) state = tmp_bt & 7;

                if (state == 0) {
                    ksw2_push_cigar(&n_cigar, max_cigar_len, cigar,
                                    KSW_CIGAR_MATCH, 1);
                    --i; --j;
                } else if (state == 1 || state == 3) {
                    ksw2_push_cigar(&n_cigar, max_cigar_len, cigar,
                                    KSW_CIGAR_DEL, 1);
                    --i;
                } else {
                    ksw2_push_cigar(&n_cigar, max_cigar_len, cigar,
                                    KSW_CIGAR_INS, 1);
                    --j;
                }
                if (n_cigar >= max_cigar_len - 2) break;
            }

            if (i >= 0)
                ksw2_push_cigar(&n_cigar, max_cigar_len, cigar,
                                KSW_CIGAR_DEL, i + 1);
            if (j >= 0)
                ksw2_push_cigar(&n_cigar, max_cigar_len, cigar,
                                KSW_CIGAR_INS, j + 1);

            /* Reverse CIGAR if not REV_CIGAR flag */
            if (!is_rev) {
                for (int k = 0; k < n_cigar / 2; ++k) {
                    uint32_t tmp_c    = cigar[k];
                    cigar[k]          = cigar[n_cigar - 1 - k];
                    cigar[n_cigar - 1 - k] = tmp_c;
                }
            }

            cigar_lengths[task_id] = n_cigar;
        } else if (cigar_buffer && lane_id == 0) {
            cigar_lengths[task_id] = 0;
        }

        __syncwarp();
        /* Loop back to claim next task */
    }
}

#endif /* __PLKSW2_COL_KERNEL_CUH__ */
