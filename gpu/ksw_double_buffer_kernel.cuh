/*
 * ksw_double_buffer_kernel v3 — Always-async cp.async via st0 alignment.
 *
 * Previous issues:
 *   v1: cg::memcpy_async with runtime size → synchronous fallback, no cp.async
 *   v2: aligned_size_t<4>(W) + alignment check → only ~25% anti-diagonals use cp.async
 *
 * v3 fix: align t_base DOWN to 4-byte boundary by loading `prefix = st0 & 3`
 *   extra ghost cells before st0.  The SMEM pointer is then offset by `prefix`
 *   so the computation loop sees idx=0 → t=st0 exactly as before.
 *   cp.async fires on EVERY window of EVERY anti-diagonal.
 *
 * Ghost cells [t_aligned, st0):
 *   - Loaded from GMEM (contain correct prev anti-diagonal values)
 *   - active=false → not computed
 *   - NOT written back (write-back loop starts from su[0] = SMEM[prefix])
 *   - No precision impact
 *
 * prefix is constant within one anti-diagonal (win_off is always a multiple of W,
 * and W is a multiple of 4, so (st0 + win_off) & 3 == st0 & 3 for all windows).
 *
 * SMEM layout:
 *   2 buffers × 6 arrays × (W + 4) bytes = 2 * 6 * (W+4) bytes per block
 *   WP = W + 4 to accommodate up to 3 ghost cells + 1 alignment pad
 *
 * A100 occupancy (W=512, WP=516):
 *   SMEM = 2 × 6 × 516 = 6192 B/block → 30 blocks/SM (vs 31 for v1/v2)
 *
 * Launch: <<<n_slots, 32, 2*6*(WINDOW_SIZE+4), stream>>>
 *
 * plmem.cu change: same as v2 (delta_stride padded to 4 bytes, 3 places),
 * gated under -DSHARED.
 *
 * Selected at build time with `make SHARED=1` (see gpu/gpu.mk).  Included by
 * plalign.cu only when SHARED is defined; the default build uses the legacy
 * ksw_fused_persistent_kernel instead.
 */

#ifndef __PLKSW_DOUBLE_BUFFER_KERNEL_V3_CUH__
#define __PLKSW_DOUBLE_BUFFER_KERNEL_V3_CUH__

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>

namespace cg = cooperative_groups;

/* ──────────────────────────────────────────────────────────────────────────
 * Device helpers (relocated from the now-deleted plksw_shared_kernel.cuh, the
 * only other consumer of which was this kernel).  KSW_CIGAR_* / KSW_EZ_* macros
 * come from plksw_kernel.cuh, which plalign.cu includes before this file.
 * ────────────────────────────────────────────────────────────────────────── */
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
    uint8_t a, uint8_t b, const int8_t *mat, int m, int8_t sc_N)
{
    // Mirror CPU ksw2_extd2_sse: an ambiguous base (value m-1, i.e. 'N') scores sc_N
    // (= mat[m*m-1]==0 ? -e2 : mat[m*m-1]); every other pair uses the matrix.
    return (a >= (uint8_t)(m - 1) || b >= (uint8_t)(m - 1)) ? sc_N : mat[a * m + b];
}

template<int WINDOW_SIZE = 512>
__global__
__launch_bounds__(32)
void ksw_double_buffer_kernel(
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
    void     *d_temp_buffer,
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
    int       task_id_base
)
{
    /* WP = padded window size: W + 4 bytes per array to hold up to 3 ghost cells */
    constexpr int W  = WINDOW_SIZE;
    constexpr int WP = W + 4;

    /* SMEM: 2 buffers × 6 arrays × WP bytes
     * buf[0] = smem_double[0 .. 6*WP-1]
     * buf[1] = smem_double[6*WP .. 12*WP-1]
     * Within each buf: [u | v | x | y | x2 | y2], each WP bytes           */
    extern __shared__ int8_t smem_double[];
    int8_t *smem_buf[2] = { smem_double, smem_double + 6 * WP };

    const int slot_id = blockIdx.x;
    const int lane_id = threadIdx.x;
    auto tile32 = cg::tiled_partition<32>(cg::this_thread_block());

    while (true) {
        int task_id;
        if (lane_id == 0) task_id = atomicAdd(d_task_counter, 1);
        task_id = __shfl_sync(0xffffffff, task_id, 0);
        if (task_id >= n_tasks) return;

        int qlen = query_batch_lens[task_id];
        int tlen = target_batch_lens[task_id];
        int flag = d_flag[task_id];

        int32_t ez_max = 0, ez_max_q = -1, ez_max_t = -1;
        int32_t ez_mqe = KSW_NEG_INF, ez_mqe_t = -1;
        int32_t ez_mte = KSW_NEG_INF, ez_mte_q = -1;
        int32_t ez_score = KSW_NEG_INF;
        int     ez_zdropped = 0;

        if (qlen <= 0 || tlen <= 0) {
            if (lane_id == 0) {
                int gid = task_id + task_id_base;
                device_res->aln_score[gid]        = 0;
                device_res->query_batch_end[gid]  = -1;
                device_res->target_batch_end[gid] = -1;
                device_res->mqe[gid]   = KSW_NEG_INF; device_res->mqe_t[gid] = -1;
                device_res->mte[gid]   = KSW_NEG_INF; device_res->mte_q[gid] = -1;
                device_res->zdropped[gid] = 0;
                if (cigar_buffer) cigar_lengths[gid] = 0;
            }
            __syncwarp();
            continue;
        }

        int8_t q  = _cudaGapO,  e  = _cudaGapExtend;
        int8_t q2 = _cudaGapOL, e2 = _cudaGapExtendL;
        int32_t w = d_bw[task_id];
        if (q2 + e2 < q + e) {
            int8_t tmp = q; q = q2; q2 = tmp; tmp = e; e = e2; e2 = tmp;
        }
        int8_t qe  = q  + e,  qe2 = q2 + e2;
        int8_t sc_N = (device_mat[m*m-1] == 0) ? (int8_t)(-e2) : device_mat[m*m-1];
        int wl = (w < 0) ? max(qlen, tlen) : w;
        int wr = (w < 0) ? max(qlen, tlen) : w;
        int long_thres = (e != e2) ? (q2-q)/(e-e2) - 1 : 0;
        if (q2+e2+long_thres*e2 > q+e+long_thres*e) ++long_thres;
        int32_t long_diff = long_thres*(e-e2) - (q2-q) - e2;
        int8_t sc_mch = device_mat[0];
        int8_t neg_qe  = (int8_t)(-(int)(q +e));
        int8_t neg_qe2 = (int8_t)(-(int)(q2+e2));

        /* GMEM temp buffer layout (plmem.cu must use padded delta_stride):
         *   H[tlen * int32]
         *   u/v/x/y/x2/y2 [delta_stride * int8 each]  ← delta_stride=((tlen+1)+3)&~3
         *   qr[qlen * uint8]  target[tlen * uint8]                           */
        int delta_stride = ((tlen + 1) + 3) & ~3;
        char   *task_buf = (char*)d_temp_buffer + (size_t)slot_id * temp_per_task;
        int32_t *H       = (int32_t*)task_buf;
        int8_t *u_arr    = (int8_t*)(H + tlen);
        int8_t *v_arr    = u_arr  + delta_stride;
        int8_t *x_arr    = v_arr  + delta_stride;
        int8_t *y_arr    = x_arr  + delta_stride;
        int8_t *x2_arr   = y_arr  + delta_stride;
        int8_t *y2_arr   = x2_arr + delta_stride;
        uint8_t *qr      = (uint8_t*)(y2_arr + delta_stride);
        uint8_t *target  = qr + qlen;

        for (int i = lane_id; i < delta_stride; i += WARP_SIZE) {
            if (i < tlen) H[i] = KSW_NEG_INF;
            u_arr[i]=neg_qe;  v_arr[i]=neg_qe;  x_arr[i]=neg_qe;
            y_arr[i]=neg_qe;  x2_arr[i]=neg_qe2; y2_arr[i]=neg_qe2;
        }
        int pq_base = query_batch_offsets[task_id]  >> 3;
        int pt_base = target_batch_offsets[task_id] >> 3;
        for (int i = lane_id; i < qlen; i += WARP_SIZE) {
            uint32_t pv = packed_query_batch[pq_base + i/8];
            qr[qlen-1-i] = (pv >> ((7-i%8)*4)) & 0xF;
        }
        for (int i = lane_id; i < tlen; i += WARP_SIZE) {
            uint32_t pv = packed_ref_batch[pt_base + i/8];
            target[i] = (pv >> ((7-i%8)*4)) & 0xF;
        }
        __syncwarp();

        int n_col = (qlen < tlen) ? qlen : tlen;
        if (w >= 0 && w+1 < n_col) n_col = w+1;
        int with_cigar  = !(flag & KSW_EZ_SCORE_ONLY);
        int approx_max  = !!(flag & KSW_EZ_APPROX_MAX);
        int right_align = !!(flag & KSW_EZ_RIGHT);
        uint8_t *p_bt    = backtrack_p   + (size_t)slot_id * max_backtrack_size;
        int     *off     = backtrack_off     + (size_t)slot_id * max_antidiag;
        int     *off_end = backtrack_off_end + (size_t)slot_id * max_antidiag;
        int last_H0_t = 0; int32_t H0 = 0;
        int last_st = -1, last_en = -1;

        /* ════════════════════════ MAIN DP LOOP ═════════════════════════════ */
        for (int r = 0; r < qlen + tlen - 1; r++) {

            if (r >= max_antidiag) { __trap(); break; }

            int st = 0, en = tlen-1;
            if (st < r - qlen + 1)       st = r - qlen + 1;
            if (en > r)                  en = r;
            if (st < (r - wr + 1) >> 1)  st = (r - wr + 1) >> 1;
            if (en > (r + wl) >> 1)      en = (r + wl) >> 1;
            if (st > en) { if (lane_id == 0) ez_zdropped = 1; break; }
            int st0 = st, en0 = en;

            int8_t x1_bnd, v1_bnd, x21_bnd;
            if (lane_id == 0) {
                if (st0>0 && st0-1>=last_st && st0-1<=last_en) {
                    x1_bnd=x_arr[st0-1]; x21_bnd=x2_arr[st0-1]; v1_bnd=v_arr[st0-1];
                } else if (st0>0) {
                    x1_bnd=-(q+e); x21_bnd=-(q2+e2); v1_bnd=-(q+e);
                } else {
                    x1_bnd=-(q+e); x21_bnd=-(q2+e2);
                    if      (r==0)          v1_bnd=-(q+e);
                    else if (r<long_thres)  v1_bnd=-e;
                    else if (r==long_thres) v1_bnd=(int8_t)long_diff;
                    else                    v1_bnd=-e2;
                }
                if (en0>=r && r<tlen) {
                    y_arr[r]=-(q+e); y2_arr[r]=-(q2+e2);
                    if      (r==0)          u_arr[r]=-(q+e);
                    else if (r<long_thres)  u_arr[r]=-e;
                    else if (r==long_thres) u_arr[r]=(int8_t)long_diff;
                    else                    u_arr[r]=-e2;
                }
                if (with_cigar) { off[r]=st0; off_end[r]=en0; }
            }
            __syncwarp();

            x1_bnd  = __shfl_sync(0xffffffff, x1_bnd,  0);
            v1_bnd  = __shfl_sync(0xffffffff, v1_bnd,  0);
            x21_bnd = __shfl_sync(0xffffffff, x21_bnd, 0);

            int band_size = en0 - st0 + 1;
            uint8_t *pr = (with_cigar && cigar_buffer) ? (p_bt + (size_t)r*n_col) : NULL;

            /* ── prefix: ghost cells to prepend for 4-byte alignment ────────
             *
             * prefix = st0 & 3  (0, 1, 2, or 3)
             * t_aligned = st0 & ~3  (4-byte aligned GMEM source for all windows)
             *
             * Since win_off is always a multiple of W (=512, multiple of 4):
             *   (st0 + win_off) & 3 == st0 & 3 == prefix  ← constant!
             * So all windows in this anti-diagonal use the same prefix.      */
            int prefix   = st0 & 3;
            int t_aligned_base = st0 & ~3;   /* = st0 - prefix */

            /* ── do_async_load ───────────────────────────────────────────────
             * Always uses cp.async (t_aligned is 4-byte aligned by construction).
             * Loads W bytes from t_aligned into smem_buf[b].
             * SMEM[0..prefix-1] = ghost cells before st0 (not computed).
             * SMEM[prefix..W-1] = real cells starting at st0.                */
            auto do_async_load = [&](int b, int win_off) {
                int t_aligned = t_aligned_base + win_off;   /* always 4B aligned */
                int8_t *d = smem_buf[b];
                cg::memcpy_async(tile32, d+0*WP, u_arr +t_aligned, cuda::aligned_size_t<4>(W));
                cg::memcpy_async(tile32, d+1*WP, v_arr +t_aligned, cuda::aligned_size_t<4>(W));
                cg::memcpy_async(tile32, d+2*WP, x_arr +t_aligned, cuda::aligned_size_t<4>(W));
                cg::memcpy_async(tile32, d+3*WP, y_arr +t_aligned, cuda::aligned_size_t<4>(W));
                cg::memcpy_async(tile32, d+4*WP, x2_arr+t_aligned, cuda::aligned_size_t<4>(W));
                cg::memcpy_async(tile32, d+5*WP, y2_arr+t_aligned, cuda::aligned_size_t<4>(W));
                /* 6 cp.async in flight; caller must wait before reading smem_buf[b] */
            };

            /* ── pipeline ───────────────────────────────────────────────────
             * All loads are async → wait_prior<6> keeps nxt's 6 in flight.
             * No bool needed; no fallback.                                   */
            int cur = 0, nxt = 1;
            do_async_load(nxt, 0);   /* prologue: load window 0 into buf[1] */

            int8_t win_x1 = x1_bnd, win_v1 = v1_bnd, win_x21 = x21_bnd;

            for (int win_off = 0; win_off < band_size; win_off += W) {

                cur = nxt; nxt = 1 - cur;
                int win_sz = (win_off + W <= band_size) ? W : (band_size - win_off);
                int t_base = st0 + win_off;          /* real start (not aligned) */
                int next_off = win_off + W;
                bool has_next = (next_off < band_size);

                /* Issue prefetch for next window before waiting for cur.     */
                if (has_next) {
                    do_async_load(nxt, next_off);
                    /* cur's 6 + nxt's 6 = 12 outstanding                    */
                    cg::wait_prior<6>(tile32);  /* cur's 6 done, nxt's 6 in flight */
                } else {
                    cg::wait(tile32);           /* last window: wait for cur's 6  */
                }
                /* buf[cur] is now ready */

                /* SMEM array pointers offset by prefix:
                 * su[0] = SMEM[prefix]      ↔  t = st0
                 * su[1] = SMEM[prefix+1]    ↔  t = st0+1
                 * …
                 * Computation loop is UNCHANGED (idx → t = t_base + idx).  */
                int8_t *su  = smem_buf[cur] + prefix + 0*WP;
                int8_t *sv  = smem_buf[cur] + prefix + 1*WP;
                int8_t *sx  = smem_buf[cur] + prefix + 2*WP;
                int8_t *sy  = smem_buf[cur] + prefix + 3*WP;
                int8_t *sx2 = smem_buf[cur] + prefix + 4*WP;
                int8_t *sy2 = smem_buf[cur] + prefix + 5*WP;

                /* ── inner WARP_SIZE-cell batches ─────────────────────────── */
                int8_t bat_x1=win_x1, bat_v1=win_v1, bat_x21=win_x21;
                int8_t nxt_x1=0, nxt_v1=0, nxt_x21=0;

                for (int bs = 0; bs < win_sz; bs += WARP_SIZE) {
                    int  idx    = bs + lane_id;
                    bool active = (idx < win_sz);
                    int  t      = t_base + idx;
                    int  qi     = r - t, qi_rev = qlen - 1 - qi;

                    int8_t old_x=0, old_v=0, old_x2=0;
                    int8_t my_u=0,  my_y=0,  my_y2=0, my_sc=0;

                    if (active) {
                        old_x  = sx[idx]; old_v  = sv[idx]; old_x2 = sx2[idx];
                        my_u   = su[idx]; my_y   = sy[idx]; my_y2  = sy2[idx];
                        if (qi>=0 && qi<qlen && t<tlen)
                            my_sc = sh_dp_score(__ldg(&qr[qi_rev]),
                                                __ldg(&target[t]),
                                                device_mat, m, sc_N);
                    }

                    int8_t shfl_x  = (int8_t)__shfl_up_sync(0xffffffff,(int32_t)old_x, 1);
                    int8_t shfl_v  = (int8_t)__shfl_up_sync(0xffffffff,(int32_t)old_v, 1);
                    int8_t shfl_x2 = (int8_t)__shfl_up_sync(0xffffffff,(int32_t)old_x2,1);

                    int8_t my_x1, my_v1, my_x21;
                    if (active) {
                        my_x1  = (lane_id==0) ? bat_x1  : shfl_x;
                        my_v1  = (lane_id==0) ? bat_v1  : shfl_v;
                        my_x21 = (lane_id==0) ? bat_x21 : shfl_x2;
                    } else { my_x1=my_v1=my_x21=0; }

                    /* ── CORE DP RECURRENCE (identical to plksw_kernel.cuh) ── */
                    int8_t new_x=0, new_v=0, new_x2=0;
                    if (active) {
                        int8_t z  = my_sc;
                        int8_t a  = my_x1+my_v1,  b  = my_y+my_u;
                        int8_t a2 = my_x21+my_v1, b2 = my_y2+my_u;
                        int8_t ut = my_u;
                        uint8_t d = 0;

                        if (!with_cigar) {
                            if(a>z)z=a; if(b>z)z=b; if(a2>z)z=a2; if(b2>z)z=b2;
                        } else if (!right_align) {
                            if(a >z){z=a; d=1;} if(b >z){z=b; d=2;}
                            if(a2>z){z=a2;d=3;} if(b2>z){z=b2;d=4;}
                        } else {
                            if(!(z>a)) {z=a; d=1;} if(!(z>b)) {z=b; d=2;}
                            if(!(z>a2)){z=a2;d=3;} if(!(z>b2)){z=b2;d=4;}
                        }
                        if (z > sc_mch) z = sc_mch;

                        int8_t new_u = z - my_v1;
                        new_v = z - ut;
                        int tv=z-q;  a-=tv; b-=tv; tv=z-q2; a2-=tv; b2-=tv;

                        int8_t new_y, new_y2;
                        if (!with_cigar || !right_align) {
                            new_x =(a >0)?(a -qe) :(-qe);
                            new_y =(b >0)?(b -qe) :(-qe);
                            new_x2=(a2>0)?(a2-qe2):(-qe2);
                            new_y2=(b2>0)?(b2-qe2):(-qe2);
                            if (with_cigar) {
                                if(a >0)d|=0x08; if(b >0)d|=0x10;
                                if(a2>0)d|=0x20; if(b2>0)d|=0x40;
                            }
                        } else {
                            new_x =(!(0>a)) ?(a -qe) :(-qe);
                            new_y =(!(0>b)) ?(b -qe) :(-qe);
                            new_x2=(!(0>a2))?(a2-qe2):(-qe2);
                            new_y2=(!(0>b2))?(b2-qe2):(-qe2);
                            if(!(0>a)) d|=0x08; if(!(0>b)) d|=0x10;
                            if(!(0>a2))d|=0x20; if(!(0>b2))d|=0x40;
                        }

                        /* write to SMEM (through prefix-offset pointer) */
                        su[idx]=new_u; sv[idx]=new_v;
                        sx[idx]=new_x; sy[idx]=new_y;
                        sx2[idx]=new_x2; sy2[idx]=new_y2;

                        /* bt_p: win_off+idx = t - st0, same as original kernel */
                        if (pr) pr[win_off + idx] = d;
                    }

                    int  last_lane  = min(WARP_SIZE-1, win_sz-bs-1);
                    bool last_batch = (bs + WARP_SIZE >= win_sz);
                    if (!last_batch) {
                        bat_x1  = __shfl_sync(0xffffffff, old_x,  last_lane);
                        bat_v1  = __shfl_sync(0xffffffff, old_v,  last_lane);
                        bat_x21 = __shfl_sync(0xffffffff, old_x2, last_lane);
                    } else {
                        nxt_x1  = __shfl_sync(0xffffffff, old_x,  last_lane);
                        nxt_v1  = __shfl_sync(0xffffffff, old_v,  last_lane);
                        nxt_x21 = __shfl_sync(0xffffffff, old_x2, last_lane);
                    }
                    __syncwarp();
                }

                win_x1=nxt_x1; win_v1=nxt_v1; win_x21=nxt_x21;

                /* write-back: su[0..win_sz-1] → u_arr[t_base..t_base+win_sz-1]
                 * Ghost cells (SMEM[0..prefix-1]) are below the su pointer,
                 * so they are NOT written back. GMEM unchanged for t<st0. ✓  */
                for (int i = lane_id; i < win_sz; i += WARP_SIZE) {
                    u_arr[t_base+i] =su[i]; v_arr[t_base+i] =sv[i];
                    x_arr[t_base+i] =sx[i]; y_arr[t_base+i] =sy[i];
                    x2_arr[t_base+i]=sx2[i]; y2_arr[t_base+i]=sy2[i];
                }
                __syncwarp();

            } /* end window loop */

            /* ── H update + z-drop ─────────────────────────────────────── */
            if (!approx_max) {
                int32_t max_H, max_t_pos;
                if (r == 0) {
                    if (lane_id==0) H[0]=(int32_t)v_arr[0]-qe;
                    __syncwarp();
                    max_H=H[0]; max_t_pos=0;
                } else {
                    if (lane_id==0) {
                        H[en0]=(en0>0)?H[en0-1]+(int32_t)u_arr[en0]
                                      :H[en0]  +(int32_t)v_arr[en0];
                    }
                    __syncwarp();
                    int32_t lH=KSW_NEG_INF; int lt=-1;
                    for (int t=st0+lane_id; t<en0; t+=WARP_SIZE) {
                        H[t]+=(int32_t)v_arr[t];
                        if(H[t]>lH){lH=H[t];lt=t;}
                    }
                    for (int d=WARP_SIZE/2; d>0; d>>=1) {
                        int32_t oh=__shfl_down_sync(0xffffffff,lH,d);
                        int     ot=__shfl_down_sync(0xffffffff,lt,d);
                        if(oh>lH||(oh==lH&&ot<lt)){lH=oh;lt=ot;}
                    }
                    if (lane_id==0 && H[en0]>=lH){lH=H[en0];lt=en0;}
                    max_H    =__shfl_sync(0xffffffff,lH,0);
                    max_t_pos=__shfl_sync(0xffffffff,lt,0);
                }
                if (lane_id==0) {
                    int j=max_t_pos, i=r-j;
                    if(max_H>ez_max){ez_max=max_H;ez_max_t=j;ez_max_q=i;}
                    if(j>=ez_max_t&&i>=ez_max_q){
                        int tl=j-ez_max_t,ql=i-ez_max_q;
                        int l=(tl>ql)?(tl-ql):(ql-tl);
                        if(zdrop>=0&&ez_max-max_H>zdrop+l*e2) ez_zdropped=1;
                    }
                    if(en0==tlen-1&&H[en0]>ez_mte){ez_mte=H[en0];ez_mte_q=r-en0;}
                    if(r-st0==qlen-1&&st0<tlen&&H[st0]>ez_mqe){ez_mqe=H[st0];ez_mqe_t=st0;}
                    if(r==qlen+tlen-2&&en0==tlen-1) ez_score=H[tlen-1];
                    last_st=st0; last_en=en0;
                }
            } else {
                if (lane_id==0) {
                    if (r>0) {
                        bool in0=(last_H0_t  >=st0&&last_H0_t  <=en0);
                        bool in1=(last_H0_t+1>=st0&&last_H0_t+1<=en0);
                        if(in0&&in1){
                            int32_t d0=v_arr[last_H0_t],d1=u_arr[last_H0_t+1];
                            if(d0>d1) H0+=d0; else{H0+=d1;++last_H0_t;}
                        } else if(in0){ H0+=v_arr[last_H0_t];
                        } else { ++last_H0_t; H0+=u_arr[last_H0_t]; }
                    } else { H0=v_arr[0]-qe; last_H0_t=0; }
                    if(r==qlen+tlen-2&&en0==tlen-1) ez_score=H0;
                    last_st=st0; last_en=en0;
                }
            }

            int zdropped_flag=__shfl_sync(0xffffffff,ez_zdropped,0);
            if(zdropped_flag) break;
            last_st=__shfl_sync(0xffffffff,last_st,0);
            last_en=__shfl_sync(0xffffffff,last_en,0);

        } /* end anti-diagonal loop */

        /* ── write results ─────────────────────────────────────────────── */
        int bq=-1, bt_res=-1;
        if (lane_id==0) {
            if(!ez_zdropped&&!(flag&KSW_EZ_EXTZ_ONLY)){
                bq=qlen-1; bt_res=tlen-1;
            } else if(!ez_zdropped&&(flag&KSW_EZ_EXTZ_ONLY)&&ez_mqe+end_bonus>ez_max){
                bq=qlen-1; bt_res=ez_mqe_t;
            } else if(ez_max_t>=0){ bq=ez_max_q; bt_res=ez_max_t; }
            int32_t out_sc=(flag&KSW_EZ_EXTZ_ONLY)?ez_max:(ez_zdropped?ez_max:ez_score);
            int gid=task_id+task_id_base;
            device_res->aln_score[gid]       =out_sc;
            device_res->query_batch_end[gid] =bq;
            device_res->target_batch_end[gid]=bt_res;
            device_res->mqe[gid]=ez_mqe;   device_res->mqe_t[gid]=ez_mqe_t;
            device_res->mte[gid]=ez_mte;   device_res->mte_q[gid]=ez_mte_q;
            device_res->zdropped[gid]=ez_zdropped;
        }
        bq    =__shfl_sync(0xffffffff,bq,    0);
        bt_res=__shfl_sync(0xffffffff,bt_res,0);

        /* ── fused backtrack + CIGAR ───────────────────────────────────── */
        if (cigar_buffer&&with_cigar&&lane_id==0&&bq>=0&&bt_res>=0) {
            int i=bt_res,j=bq,state=0,nc=0;
            int is_rev=!!(flag&KSW_EZ_REV_CIGAR);
            uint32_t *cigar=cigar_buffer+(size_t)(task_id+task_id_base)*max_cigar_len;
            while(i>=0&&j>=0){
                int diag_r=i+j,fs=-1;
                if(i<off[diag_r])     fs=2;
                if(i>off_end[diag_r]) fs=1;
                uint8_t db=0;
                if(fs<0){size_t pi=(size_t)diag_r*n_col+i-off[diag_r];db=p_bt[pi];}
                if(state==0)                  state=db&7;
                else if(!(db>>(state+2)&1))   state=0;
                if(state==0) state=db&7;
                if(fs>=0)    state=fs;
                uint32_t op=(state==0)           ?KSW_CIGAR_MATCH
                           :(state==1||state==3) ?KSW_CIGAR_DEL:KSW_CIGAR_INS;
                sh_push_cigar(&nc,max_cigar_len,cigar,op,1);
                if(state==0)            {--i;--j;}
                else if(op==KSW_CIGAR_DEL) --i;
                else                       --j;
                if(nc>=max_cigar_len-2) break;
            }
            if(i>=0) sh_push_cigar(&nc,max_cigar_len,cigar,KSW_CIGAR_DEL,i+1);
            if(j>=0) sh_push_cigar(&nc,max_cigar_len,cigar,KSW_CIGAR_INS,j+1);
            if(!is_rev)
                for(int k=0;k<nc/2;++k){
                    uint32_t tmp=cigar[k]; cigar[k]=cigar[nc-1-k]; cigar[nc-1-k]=tmp;
                }
            cigar_lengths[task_id+task_id_base]=nc;
        } else if(cigar_buffer&&lane_id==0) {
            cigar_lengths[task_id+task_id_base]=0;
        }
        __syncwarp();
    }
}

/* ── Instantiations ──────────────────────────────────────────────────────────
 *
 * Launch SMEM = 2 * 6 * (WINDOW_SIZE + 4) bytes:
 *   W=512:  2*6*516 = 6192 B → ~30 blocks/SM on A100
 *   W=1024: 2*6*1028 = 12336 B → ~15 blocks/SM
 *
 * In plalign.cu:
 *   constexpr int W = 512;
 *   ksw_double_buffer_kernel<W><<<sX_eff, 32, 2*6*(W+4), stream>>>(...);
 * ─────────────────────────────────────────────────────────────────────────── */

#define KSW_DB_ARGS \
    int*,uint32_t*,uint32_t*,uint32_t*,uint32_t*,uint32_t*,uint32_t*, \
    gasal_res_t*,const int8_t*, \
    uint8_t*,int*,int*,int,int, \
    void*,int*,int32_t*,size_t,int,int8_t,int32_t,int, \
    uint32_t*,int*,int,int

template __global__ void ksw_double_buffer_kernel<512> (KSW_DB_ARGS);
template __global__ void ksw_double_buffer_kernel<1024>(KSW_DB_ARGS);

#undef KSW_DB_ARGS

#endif /* __PLKSW_DOUBLE_BUFFER_KERNEL_V3_CUH__ */
