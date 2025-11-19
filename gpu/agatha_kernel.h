#ifndef __AGATHA_KERNEL__
#define __AGATHA_KERNEL__
// This old core provides the same result as the currently LOCAL core, but lacks some optimization. Left for historical / comparative purposes.
// Deprecated code from GASAL2 (left as reference)
#define CORE_LOCAL_DEPRECATED_COMPUTE() \
		uint32_t rbase = (packed_ref_literal >> l) & 15;/*get a base from target_batch sequence */ \
		DEV_GET_SUB_SCORE_LOCAL(temp_score, qbase, rbase);/* check equality of qbase and rbase */ \
		f[m] = max(h[m]- _cudaGapOE, f[m] - _cudaGapExtend);/* whether to introduce or extend a gap in query_batch sequence */ \
		h[m] = p[m] + temp_score; /*score if qbase is aligned to rbase*/ \
		h[m] = max(h[m], f[m]); \
		h[m] = max(h[m], 0); \
		e = max(h[m - 1] - _cudaGapOE, e - _cudaGapExtend);/*whether to introduce or extend a gap in target_batch sequence */\
		h[m] = max(h[m], e); \
		max_ref_idx = (max_score < h[m]) ? ref_idx + (m-1) : max_ref_idx; \
		max_score = (max_score < h[m]) ? h[m] : max_score; \
		p[m] = h[m-1];

#define CORE_COMPUTE() \
		uint32_t rbase = (packed_ref_literal >> l) & 15;\
		DEV_GET_SUB_SCORE_GLOBAL(temp_score, qbase, rbase) \
		temp_score += p[m]; \
		h[m] = max(temp_score, f[m]); \
		h[m] = max(h[m], e); \
		f[m] = max(temp_score- _cudaGapOE, f[m] - _cudaGapExtend); \
		e = max(temp_score- _cudaGapOE, e - _cudaGapExtend); \
		p[m] = h[m-1]; \
		diag_idx = ((ref_idx + m-1+query_idx)&(total_shm-1))<<5;\
		antidiag_max[real_warp_id+diag_idx] = max(antidiag_max[real_warp_id+diag_idx], (h[m]<<16) +ref_idx+ m-1);\

#define CORE_COMPUTE_BOUNDARY() \
		if (query_idx + _cudaBandWidth < ref_idx + m-1 || query_idx - _cudaBandWidth > ref_idx + m-1) { \
			p[m] = h[m-1]; \
		} else { \
			uint32_t rbase = (packed_ref_literal >> l) & 15;\
			DEV_GET_SUB_SCORE_GLOBAL(temp_score, qbase, rbase) \
			temp_score += p[m]; \
			h[m] = max(temp_score, f[m]); \
			h[m] = max(h[m], e); \
			f[m] = max(temp_score- _cudaGapOE, f[m] - _cudaGapExtend); \
			e = max(temp_score- _cudaGapOE, e - _cudaGapExtend); \
			p[m] = h[m-1]; \
			diag_idx = ((ref_idx + m-1+query_idx)&(total_shm-1))<<5;\
			antidiag_max[real_warp_id+diag_idx] = max(antidiag_max[real_warp_id+diag_idx], (h[m]<<16) +ref_idx+ m-1);\
		}
		

__global__ void agatha_kernel(uint32_t *packed_query_batch, uint32_t *packed_ref_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, gasal_res_t *device_res, gasal_res_t *device_res_second, uint4 *packed_tb_matrices, int n_tasks, uint32_t max_query_len, short2 *global_buffer_top)
{
    /*Initial kernel setup*/

	// Initializing variables 
	int32_t i, k, m, l, y, e;
	int32_t ub_idx, job_idx, ref_idx, query_idx;
	short2 HD;
	int32_t temp_score;
	int slice_start, slice_end, finished_blocks, chunk_start, chunk_end;
	int packed_ref_idx, packed_query_idx;
	int total_anti_diags;
	register uint32_t packed_ref_literal, packed_query_literal; 
	bool active, terminated;
	int32_t packed_ref_batch_idx, packed_query_batch_idx, query_len, ref_len, packed_query_len, packed_ref_len;
	int diag_idx, temp, last_diag;

	// Initializing max score and its idx
    int32_t max_score = 0; 
	int32_t max_ref_idx = 0; 
    int32_t prev_max_score = 0;
    int32_t max_query_idx = 0;

	// Setting constant values
	const short2 initHD = make_short2(MINUS_INF2, MINUS_INF2); //used to initialize short2
	const int32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x; //thread ID within the entire kernel
	const int packed_len = 8; //number of bps (literals) packed into a single int32
	const int const_warp_len = 8; //number of threads per subwarp (before subwarp rejoining occurs)
	const int real_warp_id = threadIdx.x % 32; //thread ID within a single (full 32-thread) warp
	const int warp_per_kernel = (gridDim.x * blockDim.x) / const_warp_len; // number of subwarps. assume number of threads % const_warp_len == 0
	const int job_per_warp = n_tasks % warp_per_kernel ? (n_tasks / warp_per_kernel + 1) : n_tasks / warp_per_kernel; //number of jobs (alignments/tasks) needed to be done by a single subwarp
	const int job_per_query = max_query_len % const_warp_len ? (max_query_len / const_warp_len + 1) : max_query_len / const_warp_len; //number of a literal's initial score to fill per thread
	const int job_start_idx = (tid / const_warp_len)*job_per_warp; // the boundary of jobs of a subwarp 
	const int job_end_idx = (job_start_idx + job_per_warp) < n_tasks ? (job_start_idx + job_per_warp) : n_tasks; // the boundary of jobs of a subwarp
	const int total_shm = packed_len*(_cudaSliceWidth+1); // amount of shared memory a single thread uses
	
	// Arrays for saving intermediate values
	int32_t h[9];
	int32_t f[9];
	int32_t p[9];

	// Global memory setup
	short2* global_buffer_left = (short2*)(global_buffer_top+max_query_len*(blockDim.x/8)*gridDim.x);
	int32_t* global_buffer_topleft= (int32_t*)(global_buffer_left+max_query_len*(blockDim.x/8)*gridDim.x);
	short2* global_ub_idx = (short2*)(global_buffer_top+max_query_len*(blockDim.x/8)*gridDim.x*3);

	// Shared memory setup
	extern __shared__ int32_t shared_maxHH[];
	int32_t* antidiag_max = (int32_t*)(shared_maxHH+(threadIdx.x/32)*total_shm*32);
	int32_t* shared_job = shared_maxHH+(blockDim.x/32)*total_shm*32+(threadIdx.x/32)*28;

	/* Setup values that will change after Subwarp Rejoining */
	int warp_len = const_warp_len;
	int warp_id = threadIdx.x % warp_len; // id of a thread in a subwarp 
	int warp_num = tid / warp_len;
	// mask that is true for threads in the same subwarp
	unsigned same_threads = __match_any_sync(0xffffffff, warp_num);
	if (warp_id==0) shared_job[(warp_num&3)] = -1;

	/* Iterating over jobs/alignments */
	for (job_idx = job_start_idx; job_idx < job_end_idx; job_idx++) {
		
		/*Uneven Bucketing*/
		// the first subwarp fetches a long sequence's idx, while the remaining subwarps fetch short sequences' idx
		ub_idx = ((job_idx&3)==0)? global_ub_idx[n_tasks-(job_idx>>2)-1].y: global_ub_idx[job_idx-(job_idx>>2)-1].y;
				
		// get target and query sequence information
		packed_ref_batch_idx = target_batch_offsets[ub_idx] >> 3; //starting index of the target_batch sequence
		packed_query_batch_idx = query_batch_offsets[ub_idx] >> 3;//starting index of the query_batch sequence
		query_len = query_batch_lens[ub_idx]; // query sequence length
		ref_len = target_batch_lens[ub_idx]; // reference sequence length 
		packed_query_len = (query_len >> 3) + (query_len & 7 ? 1 : 0);//number of 32-bit words holding query_batch sequence
		packed_ref_len = (ref_len >> 3) + (ref_len & 7 ? 1 : 0);//number of 32-bit words holding target_batch sequence

		/*Buffer Initialization*/
		// fill global buffer with initial value
		// global_buffer_top: used to store intermediate scores H and E in the horizontal strip (scores from the top)
		for (i = 0; i < job_per_query; i++) {
			l = i*warp_len + warp_id;
			if ((l) < max_query_len) {
				k = -(_cudaGapOE + (_cudaGapExtend*(l)));
				global_buffer_top[warp_num*max_query_len + l] =  l <= _cudaBandWidth? make_short2(k, k-_cudaGapOE):initHD;	
			}
		}
		// global_buffer_left: used to store intermediate scores H and F in the vertical strip (scores from the left)
		for (i = 0; i < job_per_query; i++) {
			l = i*warp_len + warp_id;
			if ((l) < max_query_len) {
				k = -(_cudaGapOE + (_cudaGapExtend*(l)));
				global_buffer_left[warp_num*max_query_len + l] =  l <= _cudaBandWidth? make_short2(k, k-_cudaGapOE):initHD;	
			}
		}
		// global_buffer_topleft: used to store intermediate scores H in the diagonal strip (scores from the top-left)
		for (i = 0; i < job_per_query; i++) {
			l = i*warp_len + warp_id;
			if (l < max_query_len) {
				k = -(_cudaGapOE+(_cudaGapExtend*(l*packed_len-1)));
				global_buffer_topleft[warp_num*max_query_len + l] = l==0? 0: (l*packed_len-1) <= _cudaBandWidth? k: MINUS_INF2; 	
			}
		}
		
		// fill shared memory with initial value
		for (m = 0; m < total_shm; m++) {
			antidiag_max[real_warp_id + m*32] = INT_MIN;
		}

		__syncwarp();

		// Initialize variables
		max_score = 0; 
		prev_max_score = 0;
		max_ref_idx = 0; 
    	max_query_idx = 0;
		terminated = false;

		i = 0; //chunk
		total_anti_diags = packed_ref_len + packed_query_len-1; //chunk

		/*Subwarp Rejoining*/
		//set shared memory that is used to maintain values for subwarp rejoining
		if (warp_id==0) shared_job[(warp_num&3)] = total_anti_diags;
		else if (warp_id==1) shared_job[4+(warp_num&3)] = packed_ref_batch_idx;
		else if (warp_id==2) shared_job[8+(warp_num&3)] = packed_query_batch_idx;
		else if (warp_id==3) shared_job[12+(warp_num&3)] = (ref_len<<16)+query_len;
		else if (warp_id==4) shared_job[16+(warp_num&3)] = ub_idx;

		same_threads = __match_any_sync(__activemask(), warp_num);

		__syncwarp();

		/*Main Alignment Loop*/
		while (i < total_anti_diags) {
			
			// set boundaries for current slice
			slice_start = max(0, (i-packed_query_len+1));
			slice_start = max(slice_start, (i*packed_len + packed_len-1+1 - _cudaBandWidth)/2/packed_len);
			slice_end = min(packed_ref_len-1, i+_cudaSliceWidth-1);
			slice_end = min(slice_end, ((i+_cudaSliceWidth-1)*packed_len + packed_len-1 + _cudaBandWidth)/2/packed_len);
			finished_blocks = slice_start;
			
			if (slice_start > slice_end) {
				terminated = true;
			}

			while (!terminated && finished_blocks <= slice_end) {
				// while the entire chunk diag is not finished
				packed_ref_idx = finished_blocks + warp_id;
				packed_query_idx = i - packed_ref_idx;
				active = (packed_ref_idx <= slice_end);	//whether the current thread has cells to fill or not
				
				if (active) {
					ref_idx = packed_ref_idx << 3;
					query_idx = packed_query_idx << 3;

					// load intermediate values from global buffers
					p[1] = global_buffer_topleft[warp_num*max_query_len + packed_ref_idx];

					for (m = 1; m < 9; m++) {
						if ( (ref_idx + m-1) < ref_len) {
							HD = global_buffer_left[warp_num*max_query_len + ref_idx + m-1];
							h[m] = HD.x;
							f[m] = HD.y;
						} else {
							// if index out of bound of the score table 
							h[m] = MINUS_INF2;
							f[m] = MINUS_INF2;
						}
						
					}

					for (m=2;m<9;m++) {
						p[m] = h[m-1];
					}

					// Set boundaries for the current chunk
					chunk_start = (max(0, (packed_ref_idx*packed_len - _cudaBandWidth)))/packed_len;
					chunk_end = min( packed_query_len-1, ( (packed_ref_idx*packed_len + packed_len -1 + _cudaBandWidth )) /packed_len );
					packed_ref_literal = packed_ref_batch[packed_ref_batch_idx + packed_ref_idx];
				}
					
				// Compute the current chunk
				for (y = 0; y < _cudaSliceWidth; y++) {
					if (active && chunk_start <= packed_query_idx && packed_query_idx <= chunk_end) {
						
						packed_query_literal = packed_query_batch[packed_query_batch_idx + packed_query_idx]; 
						query_idx = packed_query_idx << 3;
						
						for (k = 28; k >= 0 && query_idx < query_len; k -= 4) {
							uint32_t qbase = (packed_query_literal >> k) & 15;	//get a base from query_batch sequence
							// load intermediate values from global buffers
							HD = global_buffer_top[warp_num*max_query_len + query_idx];
							h[0] = HD.x;
							e = HD.y;

							if (packed_query_idx == chunk_start || packed_query_idx == chunk_end) {
								#pragma unroll 8
								for (l = 28, m = 1; m < 9; l -= 4, m++) {
									CORE_COMPUTE_BOUNDARY();
								}
							} else {
								#pragma unroll 8
								for (l = 28, m = 1; m < 9; l -= 4, m++) {
									CORE_COMPUTE();
								}
							}
							
							// write intermediate values to global buffers
							HD.x = h[m-1];
							HD.y = e;
							global_buffer_top[warp_num*max_query_len + query_idx] = HD;

							query_idx++;

						}

					}
					

					packed_query_idx++;
					
				}
				
				// write intermediate values to global buffers
				if (active) {	
					for (m = 1; m < 9; m++) {
						if ( ref_idx + m-1 < ref_len) {
							HD.x = h[m];
							HD.y = f[m];
							global_buffer_left[warp_num*max_query_len + ref_idx + m-1] = HD;
						}
					}
					global_buffer_topleft[warp_num*max_query_len + packed_ref_idx] = p[1];
				}
				
				finished_blocks+=warp_len;
			}

			__syncwarp();

			last_diag = (i+_cudaSliceWidth)<<3;
			prev_max_score = query_len+ref_len-1;

			/* Termination Condition & Score Update */
			if (!terminated) {
				for (diag_idx = i<<3; diag_idx < last_diag; diag_idx++) {
					if (diag_idx <prev_max_score) {
						m = diag_idx&(total_shm-1);
						temp = __reduce_max_sync(same_threads, antidiag_max[(m<<5)+real_warp_id]);
						if ((temp>>16) > max_score) {				
							max_score = temp>>16;
							max_ref_idx = (temp&65535);
							max_query_idx = diag_idx-max_ref_idx; 
						} else if ( (temp&65535) >= max_ref_idx && (diag_idx-(temp&65535)) >= max_query_idx) {
							int tl =  (temp&65535) - max_ref_idx, ql = (diag_idx-(temp&65535)) - max_query_idx, l;
							l = tl > ql? tl - ql : ql - tl;
							if (_cudaZThreshold >= 0 && max_score - (temp>>16) > _cudaZThreshold + l*_cudaGapExtend) {
								// Termination condition is met
								terminated = true;
								break;
							}
						}
						// reset shared memory buffer for next slice
						antidiag_max[(m<<5)+real_warp_id]=INT_MIN;
					}
				}
			}
			
			__syncwarp();

			// If job is finished
			if (terminated) {
				total_anti_diags = i; // set the total amount of diagonals as the current diagonal (to indicate that the job has finished)	
				if (warp_id==0) shared_job[(warp_num&3)] = total_anti_diags; //update this to shared memory as well (this will be used in Subwarp Rejoining as an indicator that the subwarp's job is done)
			}
			
			// Update the max score and its index to shared memory (used in Subwarp Rejoining)
			if (warp_id==1) shared_job[20+(warp_num&3)] = max_score;
			else if (warp_id==2) shared_job[24+(warp_num&3)] = (max_ref_idx<<16) + max_query_idx;
 
			__syncwarp();

			i += _cudaSliceWidth;

			/*Job wrap-up*/
			// If the job is done (either due to (1) meeting the termination condition (2) all the diagonals have been computed)
			if (i >= total_anti_diags) {
				
				// In the case of (2), check the termination condition & score update for the last diagonal block
				if (!terminated) {
					diag_idx = (i*packed_len)&(total_shm-1);
					for (k = i*packed_len, m = diag_idx; m < diag_idx+packed_len; m++, k++) {
						temp = __reduce_max_sync(same_threads, antidiag_max[(m<<5)+real_warp_id]);
						if ((temp>>16) > max_score) {				
							max_score = temp>>16;
							max_ref_idx = (temp&65535);
							max_query_idx = k-max_ref_idx; 
						} else if ( (temp&65535) >= max_ref_idx && (k-(temp&65535)) >= max_query_idx) {
							int tl =  (temp&65535) - max_ref_idx, ql = (k-(temp&65535)) - max_query_idx, l;
							l = tl > ql? tl - ql : ql - tl;
							if (_cudaZThreshold >= 0 && max_score - (temp>>16) > _cudaZThreshold + l*_cudaGapExtend) {
								// Termination condition is met
								terminated = true;
								break;
							}
						}
						antidiag_max[(m<<5)+real_warp_id]=INT_MIN;
					}
				}
				
				// Spill the results to GPU memory to be later moved to the CPU
				if (warp_id==0) {
					device_res->aln_score[ub_idx] = max_score;//copy the max score to the output array in the GPU mem
					device_res->query_batch_end[ub_idx] = max_query_idx;//copy the end position on query_batch sequence to the output array in the GPU mem
					device_res->target_batch_end[ub_idx] = max_ref_idx;//copy the end position on target_batch sequence to the output array in the GPU mem
				}

				/*Subwarp Rejoining*/
				// The subwarp that has no job looks for new jobs by iterating over other subwarp's job
				for (m = 0; m < (32/const_warp_len); m++) {
					// if the selected job still has remainig diagonals
					if (shared_job[m] > i) { // possible because all subwarps sync after each diagonal block is finished
						// read the selected job's info
						total_anti_diags = shared_job[m];
						warp_num = ((warp_num>>2)<<2)+m;
						ub_idx = shared_job[16+m];

						packed_ref_batch_idx = shared_job[4+m];
						packed_query_batch_idx = shared_job[8+m];
						ref_len = shared_job[12+m];
						query_len = ref_len&65535;
						ref_len = ref_len>>16;
						packed_query_len = (query_len >> 3) + (query_len & 7 ? 1 : 0);
						packed_ref_len = (ref_len >> 3) + (ref_len & 7 ? 1 : 0);
						
						max_score = shared_job[20+m];
						max_ref_idx = shared_job[24+m];
						max_query_idx = max_ref_idx&65535;
						max_ref_idx = max_ref_idx>>16;
						
						// reset the flag
						terminated = false;

						// reset shared memory buffer
						for (m = 0; m < total_shm; m++) {
							antidiag_max[(m<<5)+real_warp_id]=INT_MIN;
						}
						
						break;
					}
				}

			}

			__syncwarp();
			
			/*Subwarp Rejoining*/
			//Set the mask, warp length and thread id within the warp 
			same_threads = __match_any_sync(__activemask(), warp_num);
			warp_len = __popc(same_threads);
			warp_id = __popc((((0xffffffff) << (threadIdx.x % 32))&same_threads))-1;
			
			__syncwarp();

		}

		__syncwarp();
		/*Subwarp Rejoining*/
		//Reset subwarp and job related values for the next iteration
		warp_len = const_warp_len;
		warp_num = tid / warp_len;
		warp_id = tid % const_warp_len;
		ub_idx = shared_job[16+(warp_num&3)];

		__syncwarp();



	}
	
	return;


}


__global__ void agatha_sort(uint32_t *packed_query_batch, uint32_t *packed_ref_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, int n_tasks, uint32_t max_query_len, short2 *global_buffer_top)
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
#define KSW_NEG_INF     -0x40000000


// ========== CIGAR操作辅助函数 ==========
__device__ static inline uint32_t* ksw_push_cigar_device(
    int *n_cigar, 
    int max_cigar_len,
    uint32_t *cigar, 
    uint32_t op, 
    int len)
{
    // 如果CIGAR为空或者操作类型不同，添加新元素
    if (*n_cigar == 0 || op != (cigar[(*n_cigar) - 1] & 0xf)) {
        if (*n_cigar < max_cigar_len) {
            cigar[(*n_cigar)++] = (len << 4) | op;
        }
    } else {
        // 相同操作类型，累加长度
        cigar[(*n_cigar) - 1] += len << 4;
    }
    return cigar;
}

// ========== CUDA回溯Kernel ==========
__global__ void ksw_backtrack_kernel(
    uint8_t *backtrack_p,           // 回溯方向数组
    int *backtrack_off,             // 每个反对角线的起始位置
    int *backtrack_n_col,           // 每个task的n_col值
    uint32_t *query_batch_lens,     // query长度数组
    uint32_t *target_batch_lens,    // target长度数组
    gasal_res_t *device_res,        // 对齐结果（包含终点坐标）
    uint32_t *cigar_buffer,         // CIGAR输出缓冲区
    int *cigar_lengths,             // CIGAR长度输出
    int max_cigar_len,              // 每个task的最大CIGAR长度
    int max_backtrack_size,         // 每个task的回溯缓冲区大小
    int32_t *d_flag,                // 标志位
    int n_tasks                    // 任务数量
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
    int *off = backtrack_off + (size_t)task_id * (qlen + tlen);
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

__global__ void ksw_semi_global_cuda_kernel(
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
    // ========== Gap penalty parameters ==========
    int8_t q = _cudaGapO;           // 短gap开放罚分
    int8_t e = _cudaGapExtend;      // 短gap延伸罚分
    int8_t q2 = _cudaGapOL;         // 长gap开放罚分
    int8_t e2 = _cudaGapExtendL;    // 长gap延伸罚分
    int32_t w = _cudaBandWidth;     // 带宽限制

    // ========== Thread configuration ==========
    const int warp_size = 32;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = tid / warp_size;
    const int lane_id = tid % warp_size;
    
    // 每个warp处理一个task，只使用lane 0
    if (warp_id >= n_tasks) return;
    if (lane_id != 0) return;

    // ========== Task initialization ==========
    int task_id = warp_id;
    int qlen = query_batch_lens[task_id];
    int tlen = target_batch_lens[task_id];
    ksw_extz_t* ez = &ez_array[task_id];
    int flag = d_flag[task_id];

    // 空序列检查
    if (qlen <= 0 || tlen <= 0) {
        ez->max = KSW_NEG_INF;
        ez->max_q = ez->max_t = -1;
        ez->score = 0;
        ez->zdropped = 0;
        device_res->aln_score[task_id] = 0;
        device_res->query_batch_end[task_id] = -1;
        device_res->target_batch_end[task_id] = -1;
        return;
    }

    // 初始化ez结构体
    ez->max_q = ez->max_t = ez->mqe_t = ez->mte_q = -1;
    ez->max = 0;
    ez->score = ez->mqe = ez->mte = KSW_NEG_INF;
    ez->n_cigar = 0;
    ez->zdropped = 0;
    ez->reach_end = 0;

    int packed_query_offset = query_batch_offsets[task_id] >> 3;
    int packed_target_offset = target_batch_offsets[task_id] >> 3;

    // ========== Gap penalty normalization ==========
    // 确保q+e不大于q2+e2（短gap罚分不大于长gap罚分）
    if(q2 + e2 < q + e) {
        int8_t tmp = q; q = q2; q2 = tmp;
        tmp = e; e = e2; e2 = tmp;
    }
    int qe = q + e;      // 短gap总罚分
    int qe2 = q2 + e2;   // 长gap总罚分

    // ========== Band width setup ==========
    int wl = (w < 0) ? max(qlen, tlen) : w;  // 左带宽
    int wr = (w < 0) ? max(qlen, tlen) : w;  // 右带宽

    // ========== Long gap threshold ==========
    // long_thres: 从短gap切换到长gap的阈值位置
    // 当gap长度超过long_thres时，使用长gap罚分更优
    int long_thres = (e != e2) ? (q2 - q) / (e - e2) - 1 : 0;
    if(q2 + e2 + long_thres * e2 > q + e + long_thres * e) {
        ++long_thres;
    }
    // long_diff: 在阈值位置的罚分差异
    int32_t long_diff = long_thres * (e - e2) - (q2 - q) - e2;

    // ========== Allocate task-local buffers ==========
    char *task_buf = (char*)d_temp_buffer + task_id * temp_per_task;
    
    size_t offset = 0;
    // H[t]: 到位置t的累积得分
    int32_t *H = (int32_t*)(task_buf + offset);
    offset += tlen * sizeof(int32_t);
    
    // u[t], v[t]: 短gap状态（u垂直，v水平）
    int8_t *u = (int8_t*)(task_buf + offset);
    offset += (tlen + 1) * sizeof(int8_t);
    int8_t *v = (int8_t*)(task_buf + offset);
    offset += (tlen + 1) * sizeof(int8_t);
    
    // x[t], y[t]: 短gap扩展状态
    int8_t *x = (int8_t*)(task_buf + offset);
    offset += (tlen + 1) * sizeof(int8_t);
    int8_t *y = (int8_t*)(task_buf + offset);
    offset += (tlen + 1) * sizeof(int8_t);
    
    // x2[t], y2[t]: 长gap扩展状态
    int8_t *x2 = (int8_t*)(task_buf + offset);
    offset += (tlen + 1) * sizeof(int8_t);
    int8_t *y2 = (int8_t*)(task_buf + offset);
    offset += (tlen + 1) * sizeof(int8_t);
    
    // s[t]: 当前对角线的匹配/错配得分
    int8_t *s = (int8_t*)(task_buf + offset);
    offset += tlen * sizeof(int8_t);
    
    // qr[]: 反转的query序列
    uint8_t *qr = (uint8_t*)(task_buf + offset);
    offset += qlen * sizeof(uint8_t);
    // target[]: target序列
    uint8_t *target = (uint8_t*)(task_buf + offset);

    // ========== Initialize DP arrays ==========
    // H数组初始化为负无穷
    for (int i = 0; i < tlen; ++i) {
        H[i] = KSW_NEG_INF;
    }

    // 初始化gap状态数组
    for (int i = 0; i <= tlen; i++) {
        u[i] = -q - e;
        v[i] = -q - e;
        x[i] = -q - e;
        y[i] = -q - e;
        x2[i] = -q2 - e2;
        y2[i] = -q2 - e2;
    }

    // ========== Unpack sequences ==========
    // 解包query序列（4-bit packed）并反转
    for (int i = 0; i < qlen; i++) {
        int packed_idx = i / 8;
        int bit_offset = (7 - (i % 8)) * 4;
        uint32_t packed_val = packed_query_batch[packed_query_offset + packed_idx];
        qr[qlen - 1 - i] = (packed_val >> bit_offset) & 0xF;
    }

    // 解包target序列
    for (int i = 0; i < tlen; i++) {
        int packed_idx = i / 8;
        int bit_offset = (7 - (i % 8)) * 4;
        uint32_t packed_val = packed_ref_batch[packed_target_offset + packed_idx];
        target[i] = (packed_val >> bit_offset) & 0xF;
    }

    // ========== DP configuration ==========
    int last_H0_t = 0;
    int32_t H0 = 0;
    int last_st = -1, last_en = -1;
    int n_col = (qlen < tlen) ? qlen : tlen;
    n_col = (n_col < w + 1) ? n_col : (w + 1);
    
    // 根据flag设置计算模式
    int with_cigar = !(flag & KSW_EZ_SCORE_ONLY);      // 是否需要回溯信息
    int approx_max = !!(flag & KSW_EZ_APPROX_MAX);     // 是否使用近似最大值
    int right_align = !!(flag & KSW_EZ_RIGHT);    // gap右对齐 vs 左对齐

    uint8_t *p = backtrack_p + (size_t)task_id * max_backtrack_size;
    int *off = backtrack_off + (size_t)task_id * (qlen + tlen);
    int *off_end = backtrack_off_end + (size_t)task_id * (qlen + tlen);

    // ========== Main DP loop (anti-diagonal traversal) ==========
    // r是反对角线编号: r=0时(q=qlen-1,t=0), r=qlen+tlen-2时(q=0,t=tlen-1)
    int r = 0;
    for (r = 0; r < qlen + tlen - 1; r++) {
        int st = 0, en = tlen - 1;  // 当前反对角线的起止位置
        int st0, en0;

        // ========== Calculate band boundaries ==========
        // 根据反对角线编号和带宽限制计算有效范围
        if (st < r - qlen + 1) st = r - qlen + 1;
        if (en > r) en = r;
        if (st < (r - wr + 1) >> 1) st = (r - wr + 1) >> 1;
        if (en > (r + wl) >> 1) en = (r + wl) >> 1;
        
        if (st > en) {
            ez->zdropped = 1;
            break;
        }

        st0 = st;
        en0 = en;

        // ========== Initialize boundary conditions ==========
        int8_t x1, v1, x21;
        if (st > 0) {
            // 从上一轮获取边界值
            if(st - 1 >= last_st && st - 1 <= last_en) {
                x1 = x[st - 1];
                x21 = x2[st - 1];
                v1 = v[st - 1];
            } else {
                x1 = -q - e;
                x21 = -q2 - e2;
                v1 = -q - e;
            }
        } else {
            x1 = -q - e;
            x21 = -q2 - e2;
            // 根据位置设置v1（处理长gap阈值）
            if(r == 0) {
                v1 = -q - e;
            } else if(r < long_thres) {
                v1 = -e;
            } else if(r == long_thres) {
                v1 = (int8_t)long_diff;
            } else {
                v1 = -e2;
            }
        }
        
        // 设置对角线末端的边界条件
        if (en >= r && r < tlen) {
            y[r] = -q - e;
            y2[r] = -q2 - e2;
            if(r == 0) {
                u[r] = -q - e;
            } else if(r < long_thres) {
                u[r] = -e;	
            } else if(r == long_thres) {
                u[r] = (int8_t)long_diff;
            } else {
                u[r] = -e2;
            }
        }
        
        // ========== Compute match/mismatch scores ==========
        for (int t = st0; t <= en0; t++) {
            int qi = qlen - 1 - r + t;  // query索引
            if (qi >= 0 && qi < qlen && t >= 0 && t < tlen) {
                uint8_t tbase = target[t];
                uint8_t qbase = qr[qi];
                if (tbase < m && qbase < m) {
                    s[t] = device_mat[tbase * m + qbase];
                } else {
                    s[t] = 0;
                }
            } else {
                s[t] = 0;
            }
        }

        // 记录回溯范围
        if(with_cigar) {
            off[r] = st;
            off_end[r] = en;
        }
        
        uint8_t *pr = with_cigar ? (p + (size_t)r * n_col) : NULL;
        
        // ========== DP recurrence (three modes) ==========
        
        if (!with_cigar) {
            // ==================== Mode 1: Score only ====================
            // 只计算得分，不记录回溯信息（最快）
            for (int t = st0; t <= en0; t++) {
                int8_t z = s[t];
                
                // 计算四个可能的前驱得分
                int8_t a = x1 + v1;      // 从左上方来（匹配/错配）
                int8_t b = y[t] + u[t];  // 从上方来（垂直gap）
                int8_t a2 = x21 + v1;    // 从左上方来（长gap）
                int8_t b2 = y2[t] + u[t];// 从上方来（长gap）

                // 保存旧值（用于下次迭代）
                int8_t ut = u[t];
                int8_t vt = v[t];
                int8_t xt = x[t];
                int8_t x2t = x2[t];
                
                // 找最大得分
                if (a > z) z = a;
                if (b > z) z = b;
                if (a2 > z) z = a2;
                if (b2 > z) z = b2;

                // 限制最大得分
                int8_t sc_mch = device_mat[0];
                if (z > sc_mch) z = sc_mch;
                
                // 更新u和v（使用最终的z）
                u[t] = z - v1;
                v[t] = z - ut;

                // 更新gap扩展状态
                int tmp = z - q;
                a -= tmp; 
                b -= tmp;
                tmp = z - q2;
                a2 -= tmp; 
                b2 -= tmp;
                
                // 更新x, y, x2, y2
                x[t] = (a > 0) ? (a - qe) : (-qe);
                y[t] = (b > 0) ? (b - qe) : (-qe);
                x2[t] = (a2 > 0) ? (a2 - qe2) : (-qe2);
                y2[t] = (b2 > 0) ? (b2 - qe2) : (-qe2);
                
                // 为下一次迭代准备
                v1 = vt;
                x1 = xt;
                x21 = x2t;
            }
            
        } else if (!right_align) {
            // ==================== Mode 2: Gap left-alignment ====================
            // 当多个前驱得分相同时，优先选择左边的（先遇到的）
            // 使用严格大于(>)来更新方向
            for (int t = st0; t <= en0; t++) {
                int8_t z = s[t];
                
                int8_t a = x1 + v1;
                int8_t b = y[t] + u[t];
                int8_t a2 = x21 + v1;
                int8_t b2 = y2[t] + u[t];

                uint8_t d = 0;  // 方向标记：0=s, 1=a, 2=b, 3=a2, 4=b2
                
                int8_t ut = u[t];
                int8_t vt = v[t];
                int8_t xt = x[t];
                int8_t x2t = x2[t];
                
                // Left-alignment: 只有严格大于时才更新方向和得分
                if (a > z) { z = a; d = 1; }
                if (b > z) { z = b; d = 2; }
                if (a2 > z) { z = a2; d = 3; }
                if (b2 > z) { z = b2; d = 4; }

                int8_t sc_mch = device_mat[0];
                if (z > sc_mch) z = sc_mch;
                
                u[t] = z - v1;
                v[t] = z - ut;

                int tmp = z - q;
                a -= tmp; 
                b -= tmp;
                tmp = z - q2;
                a2 -= tmp; 
                b2 -= tmp;
                
                // 更新gap扩展状态，并记录扩展方向
                if (a > 0) { 
                    x[t] = a - qe; 
                    d |= 0x08;  // 第3位：x可扩展
                } else { 
                    x[t] = -qe; 
                }
                
                if (b > 0) { 
                    y[t] = b - qe; 
                    d |= 0x10;  // 第4位：y可扩展
                } else { 
                    y[t] = -qe; 
                }
                
                if (a2 > 0) { 
                    x2[t] = a2 - qe2; 
                    d |= 0x20;  // 第5位：x2可扩展
                } else { 
                    x2[t] = -qe2; 
                }
                
                if (b2 > 0) { 
                    y2[t] = b2 - qe2; 
                    d |= 0x40;  // 第6位：y2可扩展
                } else { 
                    y2[t] = -qe2; 
                }
                
                // 保存回溯方向
                if(pr != NULL) {
                    pr[t - st] = d;
                }
                
                v1 = vt;
                x1 = xt;
                x21 = x2t;
            }
            
        } else {
            // ==================== Mode 3: Gap right-alignment ====================
            // 当多个前驱得分相同时，优先选择右边的（后遇到的）
            // 使用大于等于(>=)来更新方向，即"不严格小于"
            for (int t = st0; t <= en0; t++) {
                int8_t z = s[t];
                
                int8_t a = x1 + v1;
                int8_t b = y[t] + u[t];
                int8_t a2 = x21 + v1;
                int8_t b2 = y2[t] + u[t];

                uint8_t d = 0;
                
                int8_t ut = u[t];
                int8_t vt = v[t];
                int8_t xt = x[t];
                int8_t x2t = x2[t];
                
                // Right-alignment: z不大于新值时，更新方向和得分
                // 相当于：如果a>=z，则选择a
                if (!(z > a)) { z = a; d = 1; }
                if (!(z > b)) { z = b; d = 2; }
                if (!(z > a2)) { z = a2; d = 3; }
                if (!(z > b2)) { z = b2; d = 4; }

                int8_t sc_mch = device_mat[0];
                if (z > sc_mch) z = sc_mch;
                
                u[t] = z - v1;
                v[t] = z - ut;

                int tmp = z - q;
                a -= tmp; 
                b -= tmp;
                tmp = z - q2;
                a2 -= tmp; 
                b2 -= tmp;
                
                // Right-alignment的gap扩展判断也要调整
                if (!(0 > a)) {  // a >= 0
                    x[t] = a - qe; 
                    d |= 0x08;
                } else { 
                    x[t] = -qe; 
                }
                
                if (!(0 > b)) {  // b >= 0
                    y[t] = b - qe; 
                    d |= 0x10;
                } else { 
                    y[t] = -qe; 
                }
                
                if (!(0 > a2)) {  // a2 >= 0
                    x2[t] = a2 - qe2; 
                    d |= 0x20;
                } else { 
                    x2[t] = -qe2; 
                }
                
                if (!(0 > b2)) {  // b2 >= 0
                    y2[t] = b2 - qe2; 
                    d |= 0x40;
                } else { 
                    y2[t] = -qe2; 
                }
                
                if(pr != NULL) {
                    pr[t - st] = d;
                }
                
                v1 = vt;
                x1 = xt;
                x21 = x2t;
            }
        }
        
        // ========== Track maximum score and update H array ==========
        if(!approx_max) {
            // 精确跟踪最大值（需要维护完整的H数组）
            int32_t max_H, max_t;
            
            if (r == 0) {
                H[0] = (int32_t)v[0] - qe;
                max_H = H[0];
                max_t = 0;
            } else {
                // 更新H数组：H[r][t] = H[r-1][t-1] + v[r][t] 或 H[r-1][t] + u[r][t]
                // 先保存H[en0-1]用于更新H[en0]
                int32_t H_en0_old = en0 > 0 ? H[en0 - 1] : H[en0];
                
                max_H = KSW_NEG_INF;
                max_t = st0;
                
                // 更新H[st0] ~ H[en0-1]（水平方向传递）
                for (int t = st0; t < en0; ++t) {
                    H[t] += (int32_t)v[t];
                    if (H[t] > max_H) {
                        max_H = H[t];
                        max_t = t;
                    }
                }
                
                // 更新H[en0]（垂直方向传递）
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

            // 更新全局最大值
            int j = max_t;
            int i = r - j;
            if (max_H > ez->max) {
                ez->max = max_H;
                ez->max_t = j;
                ez->max_q = i;
            }
            
            // ========== Z-drop check ==========
            // 如果得分下降超过zdrop阈值，提前终止
            if (j >= ez->max_t && i >= ez->max_q) {
                int tl = j - ez->max_t;
                int ql = i - ez->max_q;
                int l = (tl > ql) ? (tl - ql) : (ql - tl);
                if (zdrop >= 0 && ez->max - max_H > zdrop + l * e2) {
                    ez->zdropped = 1;
                    break;
                }
            }

            // 记录query末端和target末端的最佳得分
            if (en0 == tlen - 1 && H[en0] > ez->mte) {
                ez->mte = H[en0];
                ez->mte_q = r - en0;
            }
            if (r - st0 == qlen - 1 && st0 >= 0 && st0 < tlen && H[st0] > ez->mqe) {
                ez->mqe = H[st0];
                ez->mqe_t = st0;
            }
            
            // 记录最终得分
            if (r == qlen + tlen - 2 && en0 == tlen - 1) {
                ez->score = H[tlen - 1];
            }
        } else {
            // 近似最大值（只跟踪H0，不维护完整H数组）
            if (r > 0) {
                // 根据上一轮的H0位置更新
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
    
    // ========== Write results ==========
    device_res->aln_score[task_id] = ez->score;
    device_res->query_batch_end[task_id] = ez->reach_end ? qlen - 1 : ez->max_q;
    device_res->target_batch_end[task_id] = ez->reach_end ? tlen - 1 : ez->max_t;
    device_res->mqe[task_id] = ez->mqe;
    device_res->mqe_t[task_id] = ez->mqe_t;
    device_res->mte[task_id] = ez->mte;
    device_res->mte_q[task_id] = ez->mte_q;

    if (with_cigar) {
        backtrack_n_col[task_id] = n_col;
    }
}

#endif
