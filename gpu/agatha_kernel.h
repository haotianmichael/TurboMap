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

// CIGAR operations
#define KSW_CIGAR_MATCH  0
#define KSW_CIGAR_INS    1
#define KSW_CIGAR_DEL    2
#define KSW_NEG_INF     -0x40000000

// Backtrack states encoding (same as KSW scalar)
// bit 0-2: state type (0=H, 1=E, 2=F)
// bit 3: E continuation
// bit 4: F continuation

// Backtrack buffer structure
typedef struct {
    uint8_t *p;      // Backtrack matrix
    int *off;        // Offset array for each anti-diagonal
    int n_col;       // Number of columns in backtrack matrix
    int qlen;        // Query length
    int tlen;        // Target length
} ksw_backtrack_buf_t;

/**
 * Separate backtracking kernel - runs after alignment kernel
 * Each thread/warp processes one task's backtracking
 */
__global__ void ksw_backtrack_kernel(
    uint8_t *backtrack_p,
    int *backtrack_off,
    int *backtrack_n_col,
    uint32_t *query_batch_lens,
    uint32_t *target_batch_lens,
    gasal_res_t *device_res,
    uint32_t *cigar_buffer,      // Pre-allocated CIGAR buffer
    int *cigar_offsets,          // Offset for each task's CIGAR
    int max_cigar_len,           // Max CIGAR length per task
    int max_backtrack_size,
    int n_tasks,
    int8_t gapo
)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= n_tasks) return;
    
    int task_id = tid;
    int qlen = query_batch_lens[task_id];
    int tlen = target_batch_lens[task_id];
    int n_col = backtrack_n_col[task_id];
    
    // Get backtrack arrays for this task
    uint8_t *p = backtrack_p + (size_t)task_id * max_backtrack_size;
    int *off = backtrack_off + (size_t)task_id * (qlen + tlen);
    
    // Get CIGAR buffer for this task
    uint32_t *cigar = cigar_buffer + (size_t)task_id * max_cigar_len;
    int n_cigar = 0;
    
    // Backtracking
    int i = tlen - 1, j = qlen - 1;
    int state = 0;
    
    while (i >= 0 && j >= 0) {
        int r = i + j;
        int force_state = -1;
        
        if (i < off[r]) force_state = 2;
        uint8_t tmp = (force_state < 0) ? p[(size_t)r * n_col + i - off[r]] : 0;
        
        if (state == 0) state = tmp & 7;
        else if (!(tmp >> (state + 2) & 1)) state = 0;
        if (state == 0) state = tmp & 7;
        if (force_state >= 0) state = force_state;
        
        // Determine CIGAR operation
        uint32_t op;
        if (state == 0) {
            op = KSW_CIGAR_MATCH;
            --i; --j;
        } else if (state == 1) {
            op = KSW_CIGAR_DEL;
            --i;
        } else {
            op = KSW_CIGAR_INS;
            --j;
        }
        
        // Extend or add new CIGAR op
        if (n_cigar == 0 || op != (cigar[n_cigar - 1] & 0xf)) {
            if (n_cigar < max_cigar_len) {
                cigar[n_cigar++] = (1 << 4) | op;
            }
        } else {
            cigar[n_cigar - 1] += (1 << 4);
        }
    }
    
    // Handle remaining insertions/deletions
    if (i >= 0) {
        if (n_cigar == 0 || KSW_CIGAR_DEL != (cigar[n_cigar - 1] & 0xf)) {
            if (n_cigar < max_cigar_len) {
                cigar[n_cigar++] = ((i + 1) << 4) | KSW_CIGAR_DEL;
            }
        } else {
            cigar[n_cigar - 1] += ((i + 1) << 4);
        }
    }
    
    if (j >= 0) {
        if (n_cigar == 0 || KSW_CIGAR_INS != (cigar[n_cigar - 1] & 0xf)) {
            if (n_cigar < max_cigar_len) {
                cigar[n_cigar++] = ((j + 1) << 4) | KSW_CIGAR_INS;
            }
        } else {
            cigar[n_cigar - 1] += ((j + 1) << 4);
        }
    }
    
    // Reverse CIGAR (computed backwards)
    for (int k = 0; k < n_cigar / 2; k++) {
        uint32_t tmp = cigar[k];
        cigar[k] = cigar[n_cigar - 1 - k];
        cigar[n_cigar - 1 - k] = tmp;
    }
    
    // Store CIGAR info
    cigar_offsets[task_id] = n_cigar;
}


__global__ void ksw_global_cuda_kernel(
    uint32_t *packed_query_batch,
    uint32_t *packed_ref_batch,
    uint32_t *query_batch_lens,
    uint32_t *target_batch_lens,
    uint32_t *query_batch_offsets,
    uint32_t *target_batch_offsets,
    gasal_res_t *device_res,
    int8_t *device_mat,
    uint8_t *backtrack_p,        // Pre-allocated backtrack matrix buffer
    int *backtrack_off,          // Pre-allocated offset buffer
    int *backtrack_n_col,        // Store n_col for each task
    int max_backtrack_size,      // Max size for backtrack matrix per task
    int n_tasks,
    int8_t m,
    int8_t gapo,
    int8_t gape,
    int w  // band width, <0 to disable
)
{
    const int warp_size = 32;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = tid / warp_size;
    const int lane_id = tid % warp_size;
    
    if (warp_id >= n_tasks) return;
    
    // Get sequence info for this task
    int task_id = warp_id;
    int qlen = query_batch_lens[task_id];
    int tlen = target_batch_lens[task_id];
    int packed_query_offset = query_batch_offsets[task_id] >> 3;
    int packed_target_offset = target_batch_offsets[task_id] >> 3;
    
    // Gap penalties
    int qe = gapo + gape;
    int qe2 = qe + qe;
    
    // Band width
    int bandwidth = (w < 0) ? max(qlen, tlen) : w;
    int n_col = min(bandwidth + 1, tlen);
    
    // Allocate arrays in global memory (simplified version)
    // In production, use shared memory or pre-allocated buffers
    int8_t *u = NULL;  // H[i,j] - H[i-1,j]
    int8_t *v = NULL;  // H[i,j] - H[i,j-1]
    int8_t *x = NULL;  // E[i+1,j] - H[i,j]
    int8_t *y = NULL;  // F[i,j+1] - H[i,j]
    int8_t *s = NULL;  // match/mismatch scores
    
    // Backtrack arrays stored in global memory
    uint8_t *p = backtrack_p + (size_t)task_id * max_backtrack_size;
    int *off = backtrack_off + (size_t)task_id * (qlen + tlen);
    
    // Only lane 0 allocates (in real implementation, pre-allocate or use shared mem)
    if (lane_id == 0) {
        u = (int8_t*)malloc((tlen + 1) * sizeof(int8_t));
        v = (int8_t*)malloc((tlen + 1) * sizeof(int8_t));
        x = (int8_t*)malloc((tlen + 1) * sizeof(int8_t));
        y = (int8_t*)malloc((tlen + 1) * sizeof(int8_t));
        s = (int8_t*)malloc(tlen * sizeof(int8_t));
        
        // Initialize arrays
        for (int i = 0; i <= tlen; i++) {
            u[i] = v[i] = x[i] = y[i] = 0;
        }
    }
    
    __syncwarp();
    
    if (lane_id != 0 || u == NULL) return;  // Only lane 0 computes
    
    // Unpack query to reverse query array
    uint8_t *qr = (uint8_t*)malloc(qlen * sizeof(uint8_t));
    for (int i = 0; i < qlen; i++) {
        int packed_idx = i / 8;
        int bit_offset = (7 - (i % 8)) * 4;
        uint32_t packed_val = packed_query_batch[packed_query_offset + packed_idx];
        qr[qlen - 1 - i] = (packed_val >> bit_offset) & 0xF;
    }
    
    // Unpack target sequence
    uint8_t *target = (uint8_t*)malloc(tlen * sizeof(uint8_t));
    for (int i = 0; i < tlen; i++) {
        int packed_idx = i / 8;
        int bit_offset = (7 - (i % 8)) * 4;
        uint32_t packed_val = packed_ref_batch[packed_target_offset + packed_idx];
        target[i] = (packed_val >> bit_offset) & 0xF;
    }
    
    int last_H0_t = 0;
    int H0 = 0;
    
    // Main KSW loop: iterate over anti-diagonals (r = i + j)
    for (int r = 0; r < qlen + tlen - 1; r++) {
        int st = 0, en = tlen - 1;
        
        // Determine band boundaries
        if (st < r - qlen + 1) st = r - qlen + 1;
        if (en > r) en = r;
        if (st < (r - bandwidth + 1) >> 1) st = (r - bandwidth + 1) >> 1;
        if (en > (r + bandwidth) >> 1) en = (r + bandwidth) >> 1;
        
        // Initialize boundary conditions
        int8_t x1, v1;
        if (st != 0) {
            if (r > st + st + bandwidth - 1) {
                x1 = 0; v1 = 0;
            } else {
                x1 = s[st - 1];
                v1 = v[st - 1];
            }
        } else {
            x1 = 0;
            v1 = r ? gapo : 0;
        }
        
        if (en != r) {
            if (r < en + en - bandwidth - 1) {
                y[en] = 0;
                u[en] = 0;
            }
        } else {
            y[r] = 0;
            u[r] = r ? gapo : 0;
        }
        
        // Compute match/mismatch scores for this anti-diagonal
        for (int t = st; t <= en; t++) {
            int qi = t + qlen - 1 - r;
            if (qi >= 0 && qi < qlen && t < tlen) {
                s[t] = device_mat[target[t] * m + qr[qi]];
            }
        }
        
        // Store offset for backtracking
        off[r] = st;
        uint8_t *pr = p + (size_t)r * n_col;
        
        // KSW differential recurrence along the anti-diagonal
        for (int t = st; t <= en; t++) {
            uint8_t d;
            int8_t u1;
            
            // z = max(s[t] + 2qe, x1 + v1, y[t] + u[t])
            int8_t z = s[t] + qe2;
            int8_t a = x1 + v1;
            int8_t b = y[t] + u[t];
            
            // Track which achieves max
            d = (a > z) ? 1 : 0;
            z = (a > z) ? a : z;
            d = (b > z) ? 2 : d;
            z = (b > z) ? b : z;
            
            // Update differences
            u1 = u[t];
            u[t] = z - v1;
            v1 = v[t];
            v[t] = z - u1;
            
            z -= gapo;
            a -= z;
            b -= z;
            
            x1 = x[t];
            d |= (a > 0) ? 0x08 : 0;
            x[t] = (a > 0) ? a : 0;
            d |= (b > 0) ? 0x10 : 0;
            y[t] = (b > 0) ? b : 0;
            
            // Store backtrack info
            pr[t - st] = d;
        }
        
        // Update H0 (score at [0,0] -> [qlen-1, tlen-1])
        if (r > 0) {
            if (last_H0_t >= st && last_H0_t <= en) {
                H0 += v[last_H0_t] - qe;
            } else {
                ++last_H0_t;
                H0 += u[last_H0_t] - qe;
            }
        } else {
            H0 = v[0] - qe - qe;
            last_H0_t = 0;
        }
    }
    
    // Store results and backtrack info
    device_res->aln_score[task_id] = H0;
    device_res->query_batch_end[task_id] = qlen - 1;
    device_res->target_batch_end[task_id] = tlen - 1;
    backtrack_n_col[task_id] = n_col;
    
    // Cleanup temporary arrays only (p and off are in pre-allocated global memory)
    free(u); free(v); free(x); free(y); free(s);
    free(qr); free(target);
}


#endif
