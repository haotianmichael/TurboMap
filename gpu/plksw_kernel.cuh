#ifndef __PLKSW2_KERNEL_CUH__
#define __PLKSW2_KERNEL_CUH__
#include "gasal_kernels.h"

#define KSW_EZ_SCORE_ONLY  0x01 // don't record alignment path/cigar
#define KSW_EZ_RIGHT       0x02 // right-align gaps
#define KSW_EZ_GENERIC_SC  0x04 // without this flag: match/mismatch only; last symbol is a wildcard
#define KSW_EZ_APPROX_MAX  0x08 // approximate max; this is faster with sse
#define KSW_EZ_APPROX_DROP 0x10 // approximate Z-drop; faster with sse
#define KSW_EZ_EXTZ_ONLY   0x40 // only perform extension
#define KSW_EZ_REV_CIGAR   0x80 // reverse CIGAR in the output
#define KSW_EZ_SPLICE_FOR  0x100
#define KSW_EZ_SPLICE_REV  0x200
#define KSW_EZ_SPLICE_FLANK 0x400

// CIGAR operations
#define KSW_CIGAR_MATCH  0
#define KSW_CIGAR_INS    1
#define KSW_CIGAR_DEL    2
#define KSW_CIGAR_N_SKIP 3
#define KSW_NEG_INF     -0x40000000

__global__ void ksw_sort(uint32_t *packed_query_batch, uint32_t *packed_ref_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, int n_tasks, uint32_t max_query_len, short2 *global_buffer_top)
{

    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;//thread ID

	uint32_t query_len, ref_len, packed_query_len, packed_ref_len;

	short2* global_ub_idx = (short2*)(global_buffer_top+max_query_len*(blockDim.x/8)*gridDim.x*3);

	if (tid < n_tasks) {

		query_len = query_batch_lens[tid];
		ref_len = target_batch_lens[tid];
		packed_query_len = (query_len >> 3) + (query_len & 7 ? 1 : 0);//number of 32-bit words holding query_batch sequence
		packed_ref_len = (ref_len >> 3) + (ref_len & 7 ? 1 : 0);

		global_ub_idx[tid] = make_short2((packed_ref_len + packed_query_len-1), tid);
	}
	return;
}


__device__ static inline uint32_t* ksw_push_cigar_device(
    int *n_cigar, 
    int max_cigar_len,
    uint32_t *cigar, 
    uint32_t op, 
    int len)
{
    // Safety check: ensure op is in valid range (0-3 for M/I/D/N)
    if (op > 3) {
        // Invalid op, this should never happen - indicates a bug
        // Set to MATCH (0) to avoid corruption
        op = 0;
    }

    // Safety check: ensure length is positive
    if (len <= 0) {
        return cigar;  // Skip invalid entries
    }
    if (*n_cigar == 0 || op != (cigar[(*n_cigar) - 1] & 0xf)) {
        if (*n_cigar < max_cigar_len) {
            cigar[(*n_cigar)++] = (len << 4) | op;
        }
    } else {
        cigar[(*n_cigar) - 1] += len << 4;
    }
    return cigar;
}

__global__ void ksw_backtrack_kernel(
    uint8_t *backtrack_p,           
    int *backtrack_off,             
    int *backtrack_off_end,         
    int *backtrack_n_col,           
    uint32_t *query_batch_lens,     
    uint32_t *target_batch_lens,    
    gasal_res_t *device_res,        
    uint32_t *cigar_buffer,         
    int *cigar_lengths,             
    int max_cigar_len,              
    int max_backtrack_size,         
    int max_antidiag,               
    int32_t *d_flag,                
    int n_tasks                    
)
{
    const int warp_size = 32;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = tid / warp_size;
    const int lane_id = tid % warp_size;
    
    // 每个warp处理一个task，只使用lane 0
    if (warp_id >= n_tasks) return;
    if (lane_id != 0) return;

    int task_id = warp_id;
    
    // ========== 获取任务信息 ==========
    int qlen = query_batch_lens[task_id];
    int tlen = target_batch_lens[task_id];
    int n_col = backtrack_n_col[task_id];
    int flag = d_flag[task_id];
    
    // 获取回溯起点（对齐终点）
    int j0 = device_res->query_batch_end[task_id];      // query终点
    int i0 = device_res->target_batch_end[task_id];     // target终点
    
    // 检查有效性
    if (i0 < 0 || j0 < 0 || qlen <= 0 || tlen <= 0) {
        cigar_lengths[task_id] = 0;
        return;
    }
    
    // ========== 获取缓冲区指针 ==========
    uint8_t *p = backtrack_p + (size_t)task_id * max_backtrack_size;
    // CRITICAL: Use max_antidiag stride, NOT (qlen+tlen), to match memory allocation
    int *off = backtrack_off + (size_t)task_id * max_antidiag;
    int *off_end = backtrack_off_end + (size_t)task_id * max_antidiag;
    uint32_t *cigar = cigar_buffer + (size_t)task_id * max_cigar_len;
    
    // ========== 回溯参数 ==========
    int is_rot = 1;                                     // 使用反对角线遍历
    int is_rev = !!(flag & KSW_EZ_REV_CIGAR);                        // 是否反转CIGAR（默认反转）
    int min_intron_len = 0;                             // 不处理splicing
    
    // ========== 回溯主循环 ==========
    int n_cigar = 0;
    int i = i0, j = j0;     // 当前位置（query坐标，target坐标）
    int state = 0;          // 当前状态：0=H, 1=E, 2=F, 3=long E, 4=long F
    int r;                  // 反对角线编号
    uint8_t tmp;
    
    // 从终点回溯到起点
    while (i >= 0 && j >= 0) {
        int force_state = -1;
        
        // ========== 读取方向信息 ==========
        // is_rot=1: 使用反对角线坐标系
        r = i + j;  // 反对角线编号
        
        // 检查是否在band范围内
        if (i < off[r]) {
            force_state = 2;  // 强制为F状态（水平移动）
        }
        if (i > off_end[r]) {
            force_state = 1;  // 强制为E状态（垂直移动）
        } 
        // 读取回溯方向
        if (force_state < 0) {
            // 计算在回溯数组中的位置
            size_t p_idx = (size_t)r * n_col + i - off[r];
            tmp = p[p_idx];
        } else {
            tmp = 0;
        }
        
        // ========== 状态转换 ==========
        if (state == 0) {
            // H状态：查找哪个状态产生了最大值
            state = tmp & 7;  // 低3位：0=H, 1=E, 2=F, 3=long E, 4=long F
        } else {
            // 其他状态：检查是否继续当前状态
            // 如果对应的continuation bit为1，保持当前状态；否则回到H
            if (!(tmp >> (state + 2) & 1)) {
                state = 0;
            }
        }
        
        // 如果回到H状态，重新查找最优前驱
        if (state == 0) {
            state = tmp & 7;
        }
        
        // 强制状态优先
        if (force_state >= 0) {
            state = force_state;
        }
        
        // ========== 根据状态生成CIGAR并移动坐标 ==========
        if (state == 0) {
            // H状态：匹配/错配（对角线移动）
            ksw_push_cigar_device(&n_cigar, max_cigar_len, cigar, 
                                  KSW_CIGAR_MATCH, 1);
            --i;
            --j;
        } 
        else if (state == 1 || (state == 3 && min_intron_len <= 0)) {
            // E状态：deletion（垂直移动，query消耗）
            ksw_push_cigar_device(&n_cigar, max_cigar_len, cigar, 
                                  KSW_CIGAR_DEL, 1);
            --i;
        } 
        else if (state == 3 && min_intron_len > 0) {
            // 长deletion（splicing，N操作）
            ksw_push_cigar_device(&n_cigar, max_cigar_len, cigar, 
                                  KSW_CIGAR_N_SKIP, 1);
            --i;
        } 
        else {
            // F状态：insertion（水平移动，target消耗）
            ksw_push_cigar_device(&n_cigar, max_cigar_len, cigar, 
                                  KSW_CIGAR_INS, 1);
            --j;
        }
        
        // 防止无限循环
        if (n_cigar >= max_cigar_len - 2) {
            break;
        }
    }
    
    // ========== 处理剩余的query部分 ==========
    if (i >= 0) {
        // query还有剩余：添加deletion
        int op = (min_intron_len > 0 && i >= min_intron_len) ? 
                 KSW_CIGAR_N_SKIP : KSW_CIGAR_DEL;
        ksw_push_cigar_device(&n_cigar, max_cigar_len, cigar, op, i + 1);
    }
    
    // ========== 处理剩余的target部分 ==========
    if (j >= 0) {
        // target还有剩余：添加insertion
        ksw_push_cigar_device(&n_cigar, max_cigar_len, cigar, 
                              KSW_CIGAR_INS, j + 1);
    }
    
    // ========== 反转CIGAR（如果需要）==========
    // 回溯是从终点到起点，所以CIGAR是反向的
    // 通常需要反转使其从起点到终点
    if (!is_rev) {
        for (int k = 0; k < n_cigar / 2; ++k) {
            uint32_t temp = cigar[k];
            cigar[k] = cigar[n_cigar - 1 - k];
            cigar[n_cigar - 1 - k] = temp;
        }
    }
    
    // ========== 输出CIGAR长度 ==========
    cigar_lengths[task_id] = n_cigar;
}


/*
 * KSW2 Anti-diagonal Parallel Kernel 
 *
 * Per cell storage in shared memory:
 * - Query base: 1 byte (uint8_t)
 * - Target base: 1 byte (uint8_t)
 * - DP state variables (int8_t): u, v, x, y, x2, y2 = 6 bytes
 * - Match score: 1 byte (int8_t)
 * Total per cell: 9 bytes
 *
 * For Y=128 cells per buffer:
 * - Query bases: 128 bytes
 * - Target bases: 128 bytes
 * - DP states (u, v, x, y, x2, y2): 6 × 128 = 768 bytes
 * - Scores: 128 bytes
 * Total per buffer: 1152 bytes (padded to 1536 for alignment)
 * Double buffer: 2 × 1536 = 3072 bytes
 *
 * V100s occupancy:
 * - Shared memory per SM: 96KB = 98,304 bytes
 * - Max blocks per SM (hardware): 32
 * - Actual blocks per SM: min(32, 98,304 ÷ 3,072) = min(32, 32) = 32 blocks
 * - Total V100s (80 SMs): 80 × 32 = 2,560 concurrent blocks
 */

#define WARP_SIZE 32
#define MAX_CELLS_PER_SEGMENT 128  

__device__ __forceinline__ int8_t dp_compute_score(uint8_t a, uint8_t b, int8_t *mat, int m) {
    return (a < m && b < m) ? mat[a * m + b] : 0;
}

__global__ void ksw_semi_global_kernel(
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
    int *backtrack_n_col,
    int max_backtrack_size,
    int max_antidiag,
    ksw_extz_t *ez_array,
    void *d_temp_buffer,
    int *d_flag,
    size_t temp_per_task,
    int n_tasks,
    int8_t m,
    int32_t zdrop,
    int end_bonus
)
{
    // ========== Shared Memory Layout (Double Buffering) ==========
    /*
     * Layout per buffer (1536 bytes):
     * - query_bases[128]:  128 bytes (uint8_t)
     * - target_bases[128]: 128 bytes (uint8_t)
     * - u[128]:            128 bytes (int8_t)
     * - v[128]:            128 bytes (int8_t)
     * - x[128]:            128 bytes (int8_t)
     * - y[128]:            128 bytes (int8_t)
     * - x2[128]:           128 bytes (int8_t)
     * - y2[128]:           128 bytes (int8_t)
     * - score[128]:        128 bytes (int8_t)
     * Total per buffer:   1152 bytes
     * Padding to 1536:     384 bytes
     *
     * Double buffer: 2 × 1536 = 3072 bytes
     */
    extern __shared__ int8_t smem[];

    // Previous anti-diagonal buffer (1536 bytes)
    uint8_t *s_query_prev = (uint8_t*)smem;                              // [128]
    uint8_t *s_target_prev = s_query_prev + MAX_CELLS_PER_SEGMENT;      // [128]
    int8_t *s_u_prev = (int8_t*)(s_target_prev + MAX_CELLS_PER_SEGMENT); // [128]
    int8_t *s_v_prev = s_u_prev + MAX_CELLS_PER_SEGMENT;                 // [128]
    int8_t *s_x_prev = s_v_prev + MAX_CELLS_PER_SEGMENT;                 // [128]
    int8_t *s_y_prev = s_x_prev + MAX_CELLS_PER_SEGMENT;                 // [128]
    int8_t *s_x2_prev = s_y_prev + MAX_CELLS_PER_SEGMENT;                // [128]
    int8_t *s_y2_prev = s_x2_prev + MAX_CELLS_PER_SEGMENT;               // [128]
    int8_t *s_score_prev = s_y2_prev + MAX_CELLS_PER_SEGMENT;            // [128]
    // Padding to 1536 bytes

    // Current anti-diagonal buffer (1536 bytes)
    uint8_t *s_query_curr = (uint8_t*)smem + 1536;                       // [128]
    uint8_t *s_target_curr = s_query_curr + MAX_CELLS_PER_SEGMENT;       // [128]
    int8_t *s_u_curr = (int8_t*)(s_target_curr + MAX_CELLS_PER_SEGMENT); // [128]
    int8_t *s_v_curr = s_u_curr + MAX_CELLS_PER_SEGMENT;                 // [128]
    int8_t *s_x_curr = s_v_curr + MAX_CELLS_PER_SEGMENT;                 // [128]
    int8_t *s_y_curr = s_x_curr + MAX_CELLS_PER_SEGMENT;                 // [128]
    int8_t *s_x2_curr = s_y_curr + MAX_CELLS_PER_SEGMENT;                // [128]
    int8_t *s_y2_curr = s_x2_curr + MAX_CELLS_PER_SEGMENT;               // [128]
    int8_t *s_score_curr = s_y2_curr + MAX_CELLS_PER_SEGMENT;            // [128]

    // ========== Thread and Task Configuration ==========
    const int lane_id = threadIdx.x;
    const int task_id = blockIdx.x;
    if (task_id >= n_tasks) return;

    // ========== Load Task Parameters ==========
    int qlen = query_batch_lens[task_id];
    int tlen = target_batch_lens[task_id];
    ksw_extz_t* ez = &ez_array[task_id];
    int flag = d_flag[task_id];

    if (qlen <= 0 || tlen <= 0) {
        if (lane_id == 0) {
            ez->max = KSW_NEG_INF;
            ez->max_q = ez->max_t = -1;
            ez->score = 0;
            ez->zdropped = 0;
            device_res->aln_score[task_id] = 0;
            device_res->query_batch_end[task_id] = -1;
            device_res->target_batch_end[task_id] = -1;
        }
        return;
    }

    if (lane_id == 0) {
        ez->max_q = ez->max_t = ez->mqe_t = ez->mte_q = -1;
        ez->max = 0;
        ez->score = ez->mqe = ez->mte = KSW_NEG_INF;
        ez->n_cigar = 0;
        ez->zdropped = 0;
        ez->reach_end = 0;
    }

    int8_t q = _cudaGapO;
    int8_t e = _cudaGapExtend;
    int8_t q2 = _cudaGapOL;
    int8_t e2 = _cudaGapExtendL;
    int32_t w = _cudaBandWidth;

    if (q2 + e2 < q + e) {
        int8_t tmp = q; q = q2; q2 = tmp;
        tmp = e; e = e2; e2 = tmp;
    }
    int qe = q + e;
    int qe2 = q2 + e2;

    int wl = (w < 0) ? max(qlen, tlen) : w;
    int wr = (w < 0) ? max(qlen, tlen) : w;

    int long_thres = (e != e2) ? (q2 - q) / (e - e2) - 1 : 0;
    if (q2 + e2 + long_thres * e2 > q + e + long_thres * e) {
        ++long_thres;
    }
    int32_t long_diff = long_thres * (e - e2) - (q2 - q) - e2;

    // ========== Allocate Global Memory Buffers ==========
    char *task_buf = (char*)d_temp_buffer + task_id * temp_per_task;

    size_t offset = 0;
    int32_t *H = (int32_t*)(task_buf + offset);
    offset += tlen * sizeof(int32_t);

    int8_t *u = (int8_t*)(task_buf + offset);
    offset += (tlen + 1) * sizeof(int8_t);
    int8_t *v = (int8_t*)(task_buf + offset);
    offset += (tlen + 1) * sizeof(int8_t);
    int8_t *x = (int8_t*)(task_buf + offset);
    offset += (tlen + 1) * sizeof(int8_t);
    int8_t *y = (int8_t*)(task_buf + offset);
    offset += (tlen + 1) * sizeof(int8_t);
    int8_t *x2 = (int8_t*)(task_buf + offset);
    offset += (tlen + 1) * sizeof(int8_t);
    int8_t *y2 = (int8_t*)(task_buf + offset);
    offset += (tlen + 1) * sizeof(int8_t);

    uint8_t *qr = (uint8_t*)(task_buf + offset);
    offset += qlen * sizeof(uint8_t);
    uint8_t *target = (uint8_t*)(task_buf + offset);

    // ========== Parallel Initialization ==========
    for (int i = lane_id; i < tlen; i += WARP_SIZE) {
        H[i] = KSW_NEG_INF;
        u[i] = -q - e;
        v[i] = -q - e;
        x[i] = -q - e;
        y[i] = -q - e;
        x2[i] = -q2 - e2;
        y2[i] = -q2 - e2;
    }
    if (lane_id == 0) {
        u[tlen] = -q - e;
        v[tlen] = -q - e;
        x[tlen] = -q - e;
        y[tlen] = -q - e;
        x2[tlen] = -q2 - e2;
        y2[tlen] = -q2 - e2;
    }

    // ========== Parallel Sequence Unpacking ==========
    int packed_query_offset = query_batch_offsets[task_id] >> 3;
    int packed_target_offset = target_batch_offsets[task_id] >> 3;

    for (int i = lane_id; i < qlen; i += WARP_SIZE) {
        int packed_idx = i / 8;
        int bit_offset = (7 - (i % 8)) * 4;
        uint32_t packed_val = packed_query_batch[packed_query_offset + packed_idx];
        qr[qlen - 1 - i] = (packed_val >> bit_offset) & 0xF;
    }

    for (int i = lane_id; i < tlen; i += WARP_SIZE) {
        int packed_idx = i / 8;
        int bit_offset = (7 - (i % 8)) * 4;
        uint32_t packed_val = packed_ref_batch[packed_target_offset + packed_idx];
        target[i] = (packed_val >> bit_offset) & 0xF;
    }

    // ========== DP Configuration ==========
    int last_H0_t = 0;
    int32_t H0 = 0;
    int last_st = -1, last_en = -1;
    int n_col = (qlen < tlen) ? qlen : tlen;
    n_col = (n_col < w + 1) ? n_col : (w + 1);
    n_col = ((n_col + 15) / 16 + 1) * 16;

    int with_cigar = !(flag & KSW_EZ_SCORE_ONLY);
    int approx_max = !!(flag & KSW_EZ_APPROX_MAX);
    int right_align = !!(flag & KSW_EZ_RIGHT);

    uint8_t *p = backtrack_p + (size_t)task_id * max_backtrack_size;
    int *off = backtrack_off + (size_t)task_id * max_antidiag;
    int *off_end = backtrack_off_end + (size_t)task_id * max_antidiag;

    int8_t sc_mch = device_mat[0];

    // ========== Main DP Loop: Anti-diagonal Traversal ==========
    for (int r = 0; r < qlen + tlen - 1; r++) {
        // ========== Calculate Band Boundaries ==========
        int st = 0, en = tlen - 1;
        if (st < r - qlen + 1) st = r - qlen + 1;
        if (en > r) en = r;
        if (st < (r - wr + 1) >> 1) st = (r - wr + 1) >> 1;
        if (en > (r + wl) >> 1) en = (r + wl) >> 1;

        if (st > en) {
            if (lane_id == 0) ez->zdropped = 1;
            break;
        }

        int st0 = st;
        int en0 = en;

        // ========== Segmented Processing ==========
        int8_t next_seg_x1_boundary, next_seg_v1_boundary, next_seg_x21_boundary;
        bool has_next_seg_boundary = false;

        for (int seg_start = st0; seg_start <= en0; seg_start += MAX_CELLS_PER_SEGMENT) {
            int seg_end = min(seg_start + MAX_CELLS_PER_SEGMENT - 1, en0);
            int seg_size = seg_end - seg_start + 1;

            // ========== Set Boundary Conditions FIRST ==========
            int8_t x1_boundary, v1_boundary, x21_boundary;
            if (lane_id == 0) {
                // Set left boundary values (x1, v1, x21)
                if (seg_start > st0) {
                    // For subsequent segments: use saved boundary from PREVIOUS anti-diagonal
                    if (has_next_seg_boundary) {
                        x1_boundary = next_seg_x1_boundary;
                        x21_boundary = next_seg_x21_boundary;
                        v1_boundary = next_seg_v1_boundary;
                    } else {
                        // Fallback (should not happen)
                        x1_boundary = -q - e;
                        x21_boundary = -q2 - e2;
                        v1_boundary = -q - e;
                    }
                } else if (st0 > 0) {
                    // For first segment: read from previous anti-diagonal (if exists)
                    if (st0 - 1 >= last_st && st0 - 1 <= last_en) {
                        x1_boundary = x[st0 - 1];
                        x21_boundary = x2[st0 - 1];
                        v1_boundary = v[st0 - 1];
                    } else {
                        // Previous anti-diagonal doesn't include st0-1
                        x1_boundary = -q - e;
                        x21_boundary = -q2 - e2;
                        v1_boundary = -q - e;
                    }
                } else {
                    // First segment and st0 == 0: use initial boundary
                    x1_boundary = -q - e;
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

                // Set right boundary values (u[r], y[r]) 
                if (en >= r && r < tlen) {
                    y[r] = -q - e;
                    y2[r] = -q2 - e2;
                    if (r == 0) {
                        u[r] = -q - e;
                    } else if (r < long_thres) {
                        u[r] = -e;
                    } else if (r == long_thres) {
                        u[r] = (int8_t)long_diff;
                    } else {
                        u[r] = -e2;
                    }
                }

                // Record backtrack range
                if (with_cigar && seg_start == st0) {
                    off[r] = st0;
                    off_end[r] = en0;
                }
            }
            __syncwarp();

            x1_boundary = __shfl_sync(0xffffffff, x1_boundary, 0);
            v1_boundary = __shfl_sync(0xffffffff, v1_boundary, 0);
            x21_boundary = __shfl_sync(0xffffffff, x21_boundary, 0);

            // ========== Load Segment Data to Shared Memory ==========
            for (int idx = lane_id; idx < seg_size; idx += WARP_SIZE) {
                int t = seg_start + idx;
                int qi = qlen - 1 - r + t;

                // Cache query and target bases
                s_query_curr[idx] = (qi >= 0 && qi < qlen) ? qr[qi] : 0;
                s_target_curr[idx] = (t >= 0 && t < tlen) ? target[t] : 0;

                // Pre-compute and cache match/mismatch scores
                s_score_curr[idx] = dp_compute_score(s_query_curr[idx],
                                                     s_target_curr[idx],
                                                     device_mat, m);

                // Load previous anti-diagonal DP states (int8_t)
                // If t==r, this loads the boundary values set above
                s_u_prev[idx] = u[t];
                s_v_prev[idx] = v[t];
                s_x_prev[idx] = x[t];
                s_y_prev[idx] = y[t];
                s_x2_prev[idx] = x2[t];
                s_y2_prev[idx] = y2[t];
            }
            __syncwarp();

            uint8_t *pr = with_cigar ? (p + (size_t)r * n_col) : NULL;

            // ========== Parallel DP Recurrence ==========
            for (int idx = lane_id; idx < seg_size; idx += WARP_SIZE) {
                int t = seg_start + idx;

                // Get left boundary values
                int8_t x1 = (idx == 0) ? x1_boundary : s_x_prev[idx - 1];
                int8_t v1 = (idx == 0) ? v1_boundary : s_v_prev[idx - 1];
                int8_t x21 = (idx == 0) ? x21_boundary : s_x2_prev[idx - 1];

                // DP recurrence (all from shared memory, int8_t arithmetic)
                int8_t z = s_score_curr[idx];
                int8_t a = x1 + v1;
                int8_t b = s_y_prev[idx] + s_u_prev[idx];
                int8_t a2 = x21 + v1;
                int8_t b2 = s_y2_prev[idx] + s_u_prev[idx];

                uint8_t d = 0;

                // Save old values
                int8_t ut = s_u_prev[idx];
                int8_t vt = s_v_prev[idx];

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

                int8_t new_u = z - v1;
                int8_t new_v = z - ut;

                int tmp = z - q;
                a -= tmp;
                b -= tmp;
                tmp = z - q2;
                a2 -= tmp;
                b2 -= tmp;

                int8_t new_x, new_y, new_x2, new_y2;

                if (!with_cigar || !right_align) {
                    new_x = (a > 0) ? (a - qe) : (-qe);
                    new_y = (b > 0) ? (b - qe) : (-qe);
                    new_x2 = (a2 > 0) ? (a2 - qe2) : (-qe2);
                    new_y2 = (b2 > 0) ? (b2 - qe2) : (-qe2);

                    if (with_cigar) {
                        if (a > 0) d |= 0x08;
                        if (b > 0) d |= 0x10;
                        if (a2 > 0) d |= 0x20;
                        if (b2 > 0) d |= 0x40;
                    }
                } else {
                    new_x = (!(0 > a)) ? (a - qe) : (-qe);
                    new_y = (!(0 > b)) ? (b - qe) : (-qe);
                    new_x2 = (!(0 > a2)) ? (a2 - qe2) : (-qe2);
                    new_y2 = (!(0 > b2)) ? (b2 - qe2) : (-qe2);

                    if (!(0 > a)) d |= 0x08;
                    if (!(0 > b)) d |= 0x10;
                    if (!(0 > a2)) d |= 0x20;
                    if (!(0 > b2)) d |= 0x40;
                }

                // Store to shared memory (current anti-diagonal)
                s_u_curr[idx] = new_u;
                s_v_curr[idx] = new_v;
                s_x_curr[idx] = new_x;
                s_y_curr[idx] = new_y;
                s_x2_curr[idx] = new_x2;
                s_y2_curr[idx] = new_y2;

                if (pr != NULL) {
                    pr[t - st0] = d;
                }
            }
            __syncwarp();

            // ========== Write Back to Global Memory ==========
            for (int idx = lane_id; idx < seg_size; idx += WARP_SIZE) {
                int t = seg_start + idx;
                u[t] = s_u_curr[idx];
                v[t] = s_v_curr[idx];
                x[t] = s_x_curr[idx];
                y[t] = s_y_curr[idx];
                x2[t] = s_x2_curr[idx];
                y2[t] = s_y2_curr[idx];
            }
            __syncwarp();

            // ========== Save Boundary for Next Segment ==========
            if (lane_id == 0 && seg_end < en0) {
                int last_idx = seg_size - 1;
                next_seg_x1_boundary = s_x_prev[last_idx];
                next_seg_v1_boundary = s_v_prev[last_idx];
                next_seg_x21_boundary = s_x2_prev[last_idx];
                has_next_seg_boundary = true;
            }

        } // End segment loop

        // ========== Track Maximum Score ==========
        if (lane_id == 0) {
            if (!approx_max) {
                int32_t max_H, max_t;
                if (r == 0) {
                    H[0] = (int32_t)v[0] - qe;
                    max_H = H[0];
                    max_t = 0;
                } else {
                    int32_t H_en0_old = (en0 > 0) ? H[en0 - 1] : H[en0];
                    max_H = KSW_NEG_INF;
                    max_t = st0;

                    for (int t = st0; t < en0; ++t) {
                        H[t] += (int32_t)v[t];
                        if (H[t] > max_H) {
                            max_H = H[t];
                            max_t = t;
                        }
                    }

                    if (en0 > 0) {
                        H[en0] = H_en0_old + (int32_t)u[en0];
                    } else {
                        H[en0] += (int32_t)v[en0];
                    }

                    if (H[en0] > max_H) {
                        max_H = H[en0];
                        max_t = en0;
                    }
                }

                int j = max_t;
                int i = r - j;
                if (max_H > ez->max) {
                    ez->max = max_H;
                    ez->max_t = j;
                    ez->max_q = i;
                }

                if (j >= ez->max_t && i >= ez->max_q) {
                    int tl = j - ez->max_t;
                    int ql = i - ez->max_q;
                    int l = (tl > ql) ? (tl - ql) : (ql - tl);
                    if (zdrop >= 0 && ez->max - max_H > zdrop + l * e2) {
                        ez->zdropped = 1;
                    }
                }

                if (en0 == tlen - 1 && H[en0] > ez->mte) {
                    ez->mte = H[en0];
                    ez->mte_q = r - en0;
                }
                if (r - st0 == qlen - 1 && st0 >= 0 && st0 < tlen && H[st0] > ez->mqe) {
                    ez->mqe = H[st0];
                    ez->mqe_t = st0;
                }
                if (r == qlen + tlen - 2 && en0 == tlen - 1) {
                    ez->score = H[tlen - 1];
                }
            } else {
                if (r > 0) {
                    if (last_H0_t >= st0 && last_H0_t <= en0 &&
                        last_H0_t + 1 >= st0 && last_H0_t + 1 <= en0) {
                        int32_t d0 = v[last_H0_t];
                        int32_t d1 = u[last_H0_t + 1];
                        if (d0 > d1) {
                            H0 += d0;
                        } else {
                            H0 += d1;
                            ++last_H0_t;
                        }
                    } else if (last_H0_t >= st0 && last_H0_t <= en0) {
                        H0 += v[last_H0_t];
                    } else {
                        ++last_H0_t;
                        H0 += u[last_H0_t];
                    }
                } else {
                    H0 = v[0] - qe;
                    last_H0_t = 0;
                }

                if (r == qlen + tlen - 2 && en0 == tlen - 1) {
                    ez->score = H0;
                }
            }

            last_st = st;
            last_en = en;
        }

        int zdropped_flag = __shfl_sync(0xffffffff, ez->zdropped, 0);
        if (zdropped_flag) break;

    } // End anti-diagonal loop

    // ========== Determine Backtrack Endpoint ==========
    if (lane_id == 0) {
        int backtrack_q = -1, backtrack_t = -1;

        if (!ez->zdropped && !(flag & KSW_EZ_EXTZ_ONLY)) {
            backtrack_q = qlen - 1;
            backtrack_t = tlen - 1;
        } else if (!ez->zdropped && (flag & KSW_EZ_EXTZ_ONLY) &&
                   ez->mqe + end_bonus > ez->max) {
            backtrack_q = qlen - 1;
            backtrack_t = ez->mqe_t;
            ez->reach_end = 1;
        } else if (ez->max_t >= 0 && ez->max_q >= 0) {
            backtrack_q = ez->max_q;
            backtrack_t = ez->max_t;
        }

        device_res->aln_score[task_id] = ez->zdropped ? ez->max : ez->score;
        device_res->query_batch_end[task_id] = backtrack_q;
        device_res->target_batch_end[task_id] = backtrack_t;
        device_res->mqe[task_id] = ez->mqe;
        device_res->mqe_t[task_id] = ez->mqe_t;
        device_res->mte[task_id] = ez->mte;
        device_res->mte_q[task_id] = ez->mte_q;
        device_res->zdropped[task_id] = ez->zdropped;

        if (with_cigar) {
            backtrack_n_col[task_id] = n_col;
        }
    }
}


/*
 * Fused Persistent KSW Kernel
 *
 * Combines alignment and backtrack into a single persistent kernel using
 * atomic work stealing. Each block (slot) processes tasks serially, eliminating:
 *   1. Separate backtrack kernel launch (was: 2 kernels/batch, now: 1)
 *   2. Load imbalance (fast blocks immediately claim next task)
 *
 * Memory layout:
 *   backtrack_p/off/off_end indexed by slot_id (blockIdx.x)  -- reused per task
 *   cigar_buffer/results    indexed by task_id (atomic claim) -- one slot per task
 */
__global__ void ksw_fused_persistent_kernel(
    int *d_task_counter,            // atomic work queue counter
    uint32_t *packed_query_batch,
    uint32_t *packed_ref_batch,
    uint32_t *query_batch_lens,
    uint32_t *target_batch_lens,
    uint32_t *query_batch_offsets,
    uint32_t *target_batch_offsets,
    gasal_res_t *device_res,
    int8_t *device_mat,
    uint8_t *backtrack_p,           // slot-indexed: [N_SLOTS * max_backtrack_size]
    int *backtrack_off,             // slot-indexed: [N_SLOTS * max_antidiag]
    int *backtrack_off_end,         // slot-indexed: [N_SLOTS * max_antidiag]
    int max_backtrack_size,
    int max_antidiag,
    void *d_temp_buffer,            // slot-indexed: [N_SLOTS * temp_per_task]
    int *d_flag,
    size_t temp_per_task,
    int n_tasks,
    int8_t m,
    int32_t zdrop,
    int end_bonus,
    uint32_t *cigar_buffer,         // task-indexed (can be NULL)
    int *cigar_lengths,             // task-indexed (can be NULL)
    int max_cigar_len
)
{
    // ========== Shared Memory Layout (same as ksw_semi_global_kernel) ==========
    extern __shared__ int8_t smem[];

    uint8_t *s_query_prev  = (uint8_t*)smem;
    uint8_t *s_target_prev = s_query_prev  + MAX_CELLS_PER_SEGMENT;
    int8_t  *s_u_prev      = (int8_t*)(s_target_prev + MAX_CELLS_PER_SEGMENT);
    int8_t  *s_v_prev      = s_u_prev  + MAX_CELLS_PER_SEGMENT;
    int8_t  *s_x_prev      = s_v_prev  + MAX_CELLS_PER_SEGMENT;
    int8_t  *s_y_prev      = s_x_prev  + MAX_CELLS_PER_SEGMENT;
    int8_t  *s_x2_prev     = s_y_prev  + MAX_CELLS_PER_SEGMENT;
    int8_t  *s_y2_prev     = s_x2_prev + MAX_CELLS_PER_SEGMENT;
    int8_t  *s_score_prev  = s_y2_prev + MAX_CELLS_PER_SEGMENT;

    uint8_t *s_query_curr  = (uint8_t*)smem + 1536;
    uint8_t *s_target_curr = s_query_curr  + MAX_CELLS_PER_SEGMENT;
    int8_t  *s_u_curr      = (int8_t*)(s_target_curr + MAX_CELLS_PER_SEGMENT);
    int8_t  *s_v_curr      = s_u_curr  + MAX_CELLS_PER_SEGMENT;
    int8_t  *s_x_curr      = s_v_curr  + MAX_CELLS_PER_SEGMENT;
    int8_t  *s_y_curr      = s_x_curr  + MAX_CELLS_PER_SEGMENT;
    int8_t  *s_x2_curr     = s_y_curr  + MAX_CELLS_PER_SEGMENT;
    int8_t  *s_y2_curr     = s_x2_curr + MAX_CELLS_PER_SEGMENT;
    int8_t  *s_score_curr  = s_y2_curr + MAX_CELLS_PER_SEGMENT;

    const int slot_id = blockIdx.x;   // this block's persistent slot
    const int lane_id = threadIdx.x;

    // ========== Persistent Task Loop ==========
    while (true) {
        // --- Atomically claim next task ---
        int task_id;
        if (lane_id == 0) {
            task_id = atomicAdd(d_task_counter, 1);
        }
        task_id = __shfl_sync(0xffffffff, task_id, 0);
        if (task_id >= n_tasks) return;

        // ========== Task Parameters ==========
        int qlen = query_batch_lens[task_id];
        int tlen = target_batch_lens[task_id];
        int flag = d_flag[task_id];

        // Local ez state (only lane 0 updates; others read via __shfl_sync)
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

        if (q2 + e2 < q + e) {
            int8_t tmp = q; q = q2; q2 = tmp;
            tmp = e; e = e2; e2 = tmp;
        }
        int qe  = q  + e;
        int qe2 = q2 + e2;

        int wl = (w < 0) ? max(qlen, tlen) : w;
        int wr = (w < 0) ? max(qlen, tlen) : w;

        int long_thres = (e != e2) ? (q2 - q) / (e - e2) - 1 : 0;
        if (q2 + e2 + long_thres * e2 > q + e + long_thres * e) ++long_thres;
        int32_t long_diff = long_thres * (e - e2) - (q2 - q) - e2;

        // ========== Slot-indexed DP Temp Buffer ==========
        char *task_buf = (char*)d_temp_buffer + (size_t)slot_id * temp_per_task;

        size_t offset = 0;
        int32_t *H  = (int32_t*)(task_buf + offset); offset += tlen * sizeof(int32_t);
        int8_t  *u  = (int8_t*) (task_buf + offset); offset += (tlen + 1) * sizeof(int8_t);
        int8_t  *v  = (int8_t*) (task_buf + offset); offset += (tlen + 1) * sizeof(int8_t);
        int8_t  *x  = (int8_t*) (task_buf + offset); offset += (tlen + 1) * sizeof(int8_t);
        int8_t  *y  = (int8_t*) (task_buf + offset); offset += (tlen + 1) * sizeof(int8_t);
        int8_t  *x2 = (int8_t*) (task_buf + offset); offset += (tlen + 1) * sizeof(int8_t);
        int8_t  *y2 = (int8_t*) (task_buf + offset); offset += (tlen + 1) * sizeof(int8_t);
        uint8_t *qr     = (uint8_t*)(task_buf + offset); offset += qlen * sizeof(uint8_t);
        uint8_t *target = (uint8_t*)(task_buf + offset);

        // ========== Initialization ==========
        for (int i = lane_id; i < tlen; i += WARP_SIZE) {
            H[i]  = KSW_NEG_INF;
            u[i]  = -q - e;
            v[i]  = -q - e;
            x[i]  = -q - e;
            y[i]  = -q - e;
            x2[i] = -q2 - e2;
            y2[i] = -q2 - e2;
        }
        if (lane_id == 0) {
            u[tlen] = -q - e; v[tlen] = -q - e;
            x[tlen] = -q - e; y[tlen] = -q - e;
            x2[tlen] = -q2 - e2; y2[tlen] = -q2 - e2;
        }

        // ========== Sequence Unpacking ==========
        int packed_query_offset  = query_batch_offsets[task_id]  >> 3;
        int packed_target_offset = target_batch_offsets[task_id] >> 3;

        for (int i = lane_id; i < qlen; i += WARP_SIZE) {
            int packed_idx  = i / 8;
            int bit_offset  = (7 - (i % 8)) * 4;
            uint32_t pv = packed_query_batch[packed_query_offset + packed_idx];
            qr[qlen - 1 - i] = (pv >> bit_offset) & 0xF;
        }
        for (int i = lane_id; i < tlen; i += WARP_SIZE) {
            int packed_idx = i / 8;
            int bit_offset = (7 - (i % 8)) * 4;
            uint32_t pv = packed_ref_batch[packed_target_offset + packed_idx];
            target[i] = (pv >> bit_offset) & 0xF;
        }

        // ========== DP Configuration ==========
        int last_H0_t = 0;
        int32_t H0 = 0;
        int last_st = -1, last_en = -1;
        int n_col = (qlen < tlen) ? qlen : tlen;
        n_col = (n_col < w + 1) ? n_col : (w + 1);
        n_col = ((n_col + 15) / 16 + 1) * 16;

        int with_cigar = !(flag & KSW_EZ_SCORE_ONLY);
        int approx_max = !!(flag & KSW_EZ_APPROX_MAX);
        int right_align = !!(flag & KSW_EZ_RIGHT);

        // Slot-indexed backtrack buffers
        uint8_t *p   = backtrack_p   + (size_t)slot_id * max_backtrack_size;
        int     *off     = backtrack_off     + (size_t)slot_id * max_antidiag;
        int     *off_end = backtrack_off_end + (size_t)slot_id * max_antidiag;

        int8_t sc_mch = device_mat[0];

        // ========== Main DP Loop ==========
        for (int r = 0; r < qlen + tlen - 1; r++) {
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
            int8_t next_seg_x1_boundary, next_seg_v1_boundary, next_seg_x21_boundary;
            bool has_next_seg_boundary = false;

            for (int seg_start = st0; seg_start <= en0; seg_start += MAX_CELLS_PER_SEGMENT) {
                int seg_end  = min(seg_start + MAX_CELLS_PER_SEGMENT - 1, en0);
                int seg_size = seg_end - seg_start + 1;

                int8_t x1_boundary, v1_boundary, x21_boundary;
                if (lane_id == 0) {
                    if (seg_start > st0) {
                        if (has_next_seg_boundary) {
                            x1_boundary  = next_seg_x1_boundary;
                            x21_boundary = next_seg_x21_boundary;
                            v1_boundary  = next_seg_v1_boundary;
                        } else {
                            x1_boundary = x21_boundary = -q - e;
                            v1_boundary = -q - e;
                        }
                    } else if (st0 > 0) {
                        if (st0 - 1 >= last_st && st0 - 1 <= last_en) {
                            x1_boundary  = x[st0 - 1];
                            x21_boundary = x2[st0 - 1];
                            v1_boundary  = v[st0 - 1];
                        } else {
                            x1_boundary = x21_boundary = -q - e;
                            v1_boundary = -q - e;
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

                    if (en >= r && r < tlen) {
                        y[r]  = -q  - e;
                        y2[r] = -q2 - e2;
                        if (r == 0) {
                            u[r] = -q - e;
                        } else if (r < long_thres) {
                            u[r] = -e;
                        } else if (r == long_thres) {
                            u[r] = (int8_t)long_diff;
                        } else {
                            u[r] = -e2;
                        }
                    }

                    if (with_cigar && seg_start == st0) {
                        off[r]     = st0;
                        off_end[r] = en0;
                    }
                }
                __syncwarp();

                x1_boundary  = __shfl_sync(0xffffffff, x1_boundary,  0);
                v1_boundary  = __shfl_sync(0xffffffff, v1_boundary,  0);
                x21_boundary = __shfl_sync(0xffffffff, x21_boundary, 0);

                for (int idx = lane_id; idx < seg_size; idx += WARP_SIZE) {
                    int t  = seg_start + idx;
                    int qi = qlen - 1 - r + t;
                    s_query_curr[idx]  = (qi >= 0 && qi < qlen) ? qr[qi]     : 0;
                    s_target_curr[idx] = (t  >= 0 && t  < tlen) ? target[t]  : 0;
                    s_score_curr[idx]  = dp_compute_score(s_query_curr[idx], s_target_curr[idx],
                                                          device_mat, m);
                    s_u_prev[idx]  = u[t];
                    s_v_prev[idx]  = v[t];
                    s_x_prev[idx]  = x[t];
                    s_y_prev[idx]  = y[t];
                    s_x2_prev[idx] = x2[t];
                    s_y2_prev[idx] = y2[t];
                }
                __syncwarp();

                uint8_t *pr = (with_cigar && cigar_buffer) ? (p + (size_t)r * n_col) : NULL;

                for (int idx = lane_id; idx < seg_size; idx += WARP_SIZE) {
                    int t = seg_start + idx;
                    int8_t x1  = (idx == 0) ? x1_boundary  : s_x_prev[idx - 1];
                    int8_t v1  = (idx == 0) ? v1_boundary  : s_v_prev[idx - 1];
                    int8_t x21 = (idx == 0) ? x21_boundary : s_x2_prev[idx - 1];

                    int8_t z = s_score_curr[idx];
                    int8_t a  = x1  + v1;
                    int8_t b  = s_y_prev[idx] + s_u_prev[idx];
                    int8_t a2 = x21 + v1;
                    int8_t b2 = s_y2_prev[idx] + s_u_prev[idx];

                    uint8_t d = 0;
                    int8_t ut = s_u_prev[idx];
                    int8_t vt = s_v_prev[idx];

                    if (!with_cigar) {
                        if (a  > z) z = a;
                        if (b  > z) z = b;
                        if (a2 > z) z = a2;
                        if (b2 > z) z = b2;
                    } else if (!right_align) {
                        if (a  > z) { z = a;  d = 1; }
                        if (b  > z) { z = b;  d = 2; }
                        if (a2 > z) { z = a2; d = 3; }
                        if (b2 > z) { z = b2; d = 4; }
                    } else {
                        if (!(z > a )) { z = a;  d = 1; }
                        if (!(z > b )) { z = b;  d = 2; }
                        if (!(z > a2)) { z = a2; d = 3; }
                        if (!(z > b2)) { z = b2; d = 4; }
                    }

                    if (z > sc_mch) z = sc_mch;

                    int8_t new_u = z - v1;
                    int8_t new_v = z - ut;

                    int tmp = z - q;
                    a  -= tmp; b  -= tmp;
                    tmp = z - q2;
                    a2 -= tmp; b2 -= tmp;

                    int8_t new_x, new_y, new_x2, new_y2;

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
                        new_x  = (!(0 > a )) ? (a  - qe)  : (-qe);
                        new_y  = (!(0 > b )) ? (b  - qe)  : (-qe);
                        new_x2 = (!(0 > a2)) ? (a2 - qe2) : (-qe2);
                        new_y2 = (!(0 > b2)) ? (b2 - qe2) : (-qe2);
                        if (!(0 > a )) d |= 0x08;
                        if (!(0 > b )) d |= 0x10;
                        if (!(0 > a2)) d |= 0x20;
                        if (!(0 > b2)) d |= 0x40;
                    }

                    s_u_curr[idx]  = new_u;
                    s_v_curr[idx]  = new_v;
                    s_x_curr[idx]  = new_x;
                    s_y_curr[idx]  = new_y;
                    s_x2_curr[idx] = new_x2;
                    s_y2_curr[idx] = new_y2;

                    if (pr != NULL) pr[t - st0] = d;
                }
                __syncwarp();

                for (int idx = lane_id; idx < seg_size; idx += WARP_SIZE) {
                    int t = seg_start + idx;
                    u[t]  = s_u_curr[idx];
                    v[t]  = s_v_curr[idx];
                    x[t]  = s_x_curr[idx];
                    y[t]  = s_y_curr[idx];
                    x2[t] = s_x2_curr[idx];
                    y2[t] = s_y2_curr[idx];
                }
                __syncwarp();

                if (lane_id == 0 && seg_end < en0) {
                    int last_idx = seg_size - 1;
                    next_seg_x1_boundary  = s_x_prev[last_idx];
                    next_seg_v1_boundary  = s_v_prev[last_idx];
                    next_seg_x21_boundary = s_x2_prev[last_idx];
                    has_next_seg_boundary = true;
                }
            } // End segment loop

            // ========== Track Maximum Score ==========
            if (lane_id == 0) {
                if (!approx_max) {
                    int32_t max_H, max_t;
                    if (r == 0) {
                        H[0] = (int32_t)v[0] - qe;
                        max_H = H[0]; max_t = 0;
                    } else {
                        int32_t H_en0_old = (en0 > 0) ? H[en0 - 1] : H[en0];
                        max_H = KSW_NEG_INF; max_t = st0;
                        for (int t = st0; t < en0; ++t) {
                            H[t] += (int32_t)v[t];
                            if (H[t] > max_H) { max_H = H[t]; max_t = t; }
                        }
                        if (en0 > 0) {
                            H[en0] = H_en0_old + (int32_t)u[en0];
                        } else {
                            H[en0] += (int32_t)v[en0];
                        }
                        if (H[en0] > max_H) { max_H = H[en0]; max_t = en0; }
                    }

                    int j = max_t, i = r - j;
                    if (max_H > ez_max) { ez_max = max_H; ez_max_t = j; ez_max_q = i; }

                    if (j >= ez_max_t && i >= ez_max_q) {
                        int tl = j - ez_max_t, ql = i - ez_max_q;
                        int l = (tl > ql) ? (tl - ql) : (ql - tl);
                        if (zdrop >= 0 && ez_max - max_H > zdrop + l * e2) ez_zdropped = 1;
                    }

                    if (en0 == tlen - 1 && H[en0] > ez_mte) {
                        ez_mte = H[en0]; ez_mte_q = r - en0;
                    }
                    if (r - st0 == qlen - 1 && st0 >= 0 && st0 < tlen && H[st0] > ez_mqe) {
                        ez_mqe = H[st0]; ez_mqe_t = st0;
                    }
                    if (r == qlen + tlen - 2 && en0 == tlen - 1) ez_score = H[tlen - 1];
                } else {
                    if (r > 0) {
                        if (last_H0_t >= st0 && last_H0_t <= en0 &&
                            last_H0_t + 1 >= st0 && last_H0_t + 1 <= en0) {
                            int32_t d0 = v[last_H0_t], d1 = u[last_H0_t + 1];
                            if (d0 > d1) { H0 += d0; }
                            else         { H0 += d1; ++last_H0_t; }
                        } else if (last_H0_t >= st0 && last_H0_t <= en0) {
                            H0 += v[last_H0_t];
                        } else {
                            ++last_H0_t; H0 += u[last_H0_t];
                        }
                    } else {
                        H0 = v[0] - qe; last_H0_t = 0;
                    }
                    if (r == qlen + tlen - 2 && en0 == tlen - 1) ez_score = H0;
                }
                last_st = st; last_en = en;
            }

            int zdropped_flag = __shfl_sync(0xffffffff, ez_zdropped, 0);
            if (zdropped_flag) break;
        } // End anti-diagonal loop

        // ========== Determine Backtrack Endpoint & Write Results ==========
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
        // Broadcast backtrack endpoints to all lanes (needed for CIGAR phase check)
        backtrack_q = __shfl_sync(0xffffffff, backtrack_q, 0);
        backtrack_t = __shfl_sync(0xffffffff, backtrack_t, 0);

        // ========== Immediate Backtrack Phase (fused in same block) ==========
        // Only lane 0 performs the serial backtrack; others are idle.
        if (cigar_buffer && with_cigar && lane_id == 0 &&
            backtrack_q >= 0 && backtrack_t >= 0) {

            int i0 = backtrack_t;  // target endpoint
            int j0 = backtrack_q;  // query endpoint
            int i  = i0, j = j0;
            int state = 0;
            int n_cigar = 0;
            int is_rev  = !!(flag & KSW_EZ_REV_CIGAR);
            int min_intron_len = 0;

            uint32_t *cigar = cigar_buffer + (size_t)task_id * max_cigar_len;

            while (i >= 0 && j >= 0) {
                int force_state = -1;
                int r = i + j;
                if (i < off[r])     force_state = 2;
                if (i > off_end[r]) force_state = 1;

                uint8_t tmp_bt = 0;
                if (force_state < 0) {
                    size_t p_idx = (size_t)r * n_col + i - off[r];
                    tmp_bt = p[p_idx];
                }

                if (state == 0) {
                    state = tmp_bt & 7;
                } else {
                    if (!(tmp_bt >> (state + 2) & 1)) state = 0;
                }
                if (state == 0) state = tmp_bt & 7;
                if (force_state >= 0) state = force_state;

                if (state == 0) {
                    ksw_push_cigar_device(&n_cigar, max_cigar_len, cigar, KSW_CIGAR_MATCH, 1);
                    --i; --j;
                } else if (state == 1 || (state == 3 && min_intron_len <= 0)) {
                    ksw_push_cigar_device(&n_cigar, max_cigar_len, cigar, KSW_CIGAR_DEL, 1);
                    --i;
                } else if (state == 3 && min_intron_len > 0) {
                    ksw_push_cigar_device(&n_cigar, max_cigar_len, cigar, KSW_CIGAR_N_SKIP, 1);
                    --i;
                } else {
                    ksw_push_cigar_device(&n_cigar, max_cigar_len, cigar, KSW_CIGAR_INS, 1);
                    --j;
                }
                if (n_cigar >= max_cigar_len - 2) break;
            }

            if (i >= 0) {
                int op = (min_intron_len > 0 && i >= min_intron_len) ?
                         KSW_CIGAR_N_SKIP : KSW_CIGAR_DEL;
                ksw_push_cigar_device(&n_cigar, max_cigar_len, cigar, op, i + 1);
            }
            if (j >= 0) {
                ksw_push_cigar_device(&n_cigar, max_cigar_len, cigar, KSW_CIGAR_INS, j + 1);
            }

            if (!is_rev) {
                for (int k = 0; k < n_cigar / 2; ++k) {
                    uint32_t tmp_c = cigar[k];
                    cigar[k] = cigar[n_cigar - 1 - k];
                    cigar[n_cigar - 1 - k] = tmp_c;
                }
            }

            cigar_lengths[task_id] = n_cigar;
        } else if (cigar_buffer && lane_id == 0) {
            cigar_lengths[task_id] = 0;
        }

        __syncwarp();
        // Loop back to claim next task
    }
}

#endif