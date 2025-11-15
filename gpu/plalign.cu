#include "plalign.cuh"
#include "gasal_kernels.h"
#include "plmem.cuh"  // For deviceMemPtr
#include <algorithm>


#define CHECKCUDAERROR(error) \
		do{\
			err = error;\
			if (cudaSuccess != err ) { \
				fprintf(stderr, "[GASAL CUDA ERROR:] %s(CUDA error no.=%d). Line no. %d in file %s\n", cudaGetErrorString(err), err,  __LINE__, __FILE__); \
				exit(EXIT_FAILURE);\
			}\
		}while(0)\

static align_config_t g_config = {
    .blocks = 28,
    .threads = 256,
    .slice_width = 3,
    .z_threshold = 400,
    .band_width = 751,
    .match_score = 2,
    .mismatch_score = 4,
    .gap_open = 4,
    .gap_extend = 2,
    .gap_open_long = 24,
    .gap_extend_long = 1
};

// Global pointer to current stream's device memory
// Set by gpu_align_set_device_mem() before calling gpu_align_batch_execute()
deviceMemPtr *g_current_dev_mem = NULL;
static bool g_subst_scores_uploaded = false;

static void ksw_gen_simple_mat(int m, int8_t *mat, int8_t a, int8_t b, int8_t sc_ambi)
{
	int i, j;
	a = a < 0? -a : a;
	b = b > 0? -b : b;
	sc_ambi = sc_ambi > 0? -sc_ambi : sc_ambi;
	for (i = 0; i < m - 1; ++i) {
		for (j = 0; j < m - 1; ++j)
			mat[i * m + j] = i == j? a : b;
		mat[i * m + m - 1] = sc_ambi;
	}
	for (j = 0; j < m; ++j)
		mat[(m - 1) * m + j] = sc_ambi;
}

void gasal_copy_subst_scores(gasal_subst_scores *subst){

	cudaError_t err;
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaGapO, &(subst->gap_open), sizeof(int8_t), 0, cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaGapExtend, &(subst->gap_extend), sizeof(int8_t), 0, cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaGapOL, &(subst->gap_open_long), sizeof(int8_t), 0, cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaGapExtendL, &(subst->gap_extend_long), sizeof(int8_t), 0, cudaMemcpyHostToDevice));
	int32_t gapoe = (subst->gap_open + subst->gap_extend);
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaGapOE, &(gapoe), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaMatchScore, &(subst->match), sizeof(int8_t), 0, cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaMismatchScore, &(subst->mismatch), sizeof(int8_t), 0, cudaMemcpyHostToDevice));
	// For AGAThA
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaSliceWidth, &(subst->slice_width), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaZThreshold, &(subst->z_threshold), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaBandWidth, &(subst->band_width), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	return;
}

// Set the device memory pointer for alignment operations
void gpu_align_copy_param() {

    // Upload substitution scores on first call
    if (!g_subst_scores_uploaded) {
        gasal_subst_scores subst;
        subst.match = g_config.match_score;
        subst.mismatch = g_config.mismatch_score;
        subst.gap_open = g_config.gap_open;
        subst.gap_extend = g_config.gap_extend;
        subst.gap_open_long = g_config.gap_open_long;
        subst.gap_extend_long = g_config.gap_extend_long;
        subst.slice_width = g_config.slice_width;
        subst.z_threshold = g_config.z_threshold;
        subst.band_width = g_config.band_width;
        gasal_copy_subst_scores(&subst);
        g_subst_scores_uploaded = true;
    }
}

extern "C" void gpu_align_batch_execute(const mm_mapopt_t *opt, gpu_align_task_t *tasks, int n_tasks,
                            uint8_t *seq_buffer, uint32_t *cigar_buffer);
void gpu_align_batch_execute(const mm_mapopt_t *opt, gpu_align_task_t *tasks, int n_tasks,
                            uint8_t *seq_buffer, uint32_t *cigar_buffer) {
    if (n_tasks <= 0) return;
    gpu_align_copy_param();
    // Check that device memory has been set
    if (!g_current_dev_mem) {
        fprintf(stderr, "[ERROR] Device memory not set. Call gpu_align_set_device_mem() first.\n");
        return;
    }

    deviceMemPtr *dev_mem = g_current_dev_mem;

    // Create local variables for easier code migration
    int kernel_blocks = 28;
    const int max_concurrent_tasks = 60;  // Must match backtrack buffer allocation
    if (n_tasks > max_concurrent_tasks) {
        fprintf(stderr, "[WARNING] n_tasks=%d exceeds max_concurrent_tasks=%d, clamping to max\n",
                n_tasks, max_concurrent_tasks);
        n_tasks = max_concurrent_tasks;
    }
    int kernel_threads = 256;
    uint8_t *d_unpacked_query = dev_mem->d_align_unpacked_query;
    uint8_t *d_unpacked_target = dev_mem->d_align_unpacked_target;
    uint32_t *d_packed_query = dev_mem->d_align_packed_query;
    uint32_t *d_packed_target = dev_mem->d_align_packed_target;
    uint32_t *d_query_offsets = dev_mem->d_align_query_offsets;
    uint32_t *d_target_offsets = dev_mem->d_align_target_offsets;
    uint32_t *d_query_lens = dev_mem->d_align_query_lens;
    uint32_t *d_target_lens = dev_mem->d_align_target_lens;
    int32_t *d_flag = dev_mem->d_align_flag;
    void *d_global_buffer = dev_mem->d_align_global_buffer;
    void *d_ksw_temp_buffer = dev_mem->d_align_ksw_temp_buffer;
    size_t ksw_temp_per_task = dev_mem->align_ksw_temp_per_task;
    uint8_t *d_backtrack_p = dev_mem->d_align_backtrack_p;
    int *d_backtrack_off = dev_mem->d_align_backtrack_off;
    int *d_backtrack_off_end = dev_mem->d_align_backtrack_off_end;
    int *d_backtrack_n_col = dev_mem->d_align_backtrack_n_col;
    uint32_t *d_cigar_buffer = dev_mem->d_align_cigar_buffer;
    int *d_cigar_lengths = dev_mem->d_align_cigar_lengths;
    size_t max_backtrack_size = dev_mem->max_align_backtrack_size;
    size_t max_cigar_len = dev_mem->max_align_cigar_len;
    size_t max_query_len_limit = dev_mem->max_align_query_len;
    int8_t *d_mat = dev_mem->d_align_mat;
    void *device_res = dev_mem->d_align_device_res;
    void *d_ez_array = dev_mem->d_align_ez_array;
    int32_t *d_scores = dev_mem->d_align_scores;
    int32_t *d_query_ends = dev_mem->d_align_query_ends;
    int32_t *d_target_ends = dev_mem->d_align_target_ends;

    // Host buffers for CIGAR and results
    uint32_t *h_cigar_buffer = (uint32_t*)calloc(max_concurrent_tasks * max_cigar_len, sizeof(uint32_t));
    int *h_cigar_lengths = (int*)calloc(max_concurrent_tasks, sizeof(int));
    int32_t *h_scores = (int32_t*)calloc(max_concurrent_tasks, sizeof(int32_t));
    int32_t *h_query_ends = (int32_t*)calloc(max_concurrent_tasks, sizeof(int32_t));
    int32_t *h_target_ends = (int32_t*)calloc(max_concurrent_tasks, sizeof(int32_t));
    short2 *h_sort_buffer = (short2*)calloc(n_tasks, sizeof(short2));

    // Calculate memory requirements
    size_t total_query_bytes = 0, total_target_bytes = 0;
    uint32_t max_query_len = 0;

    // Host arrays for batch preparation
    uint32_t *h_query_offsets = (uint32_t*)calloc(n_tasks, sizeof(uint32_t));
    uint32_t *h_target_offsets = (uint32_t*)calloc(n_tasks, sizeof(uint32_t));
    uint32_t *h_query_lens = (uint32_t*)calloc(n_tasks, sizeof(uint32_t));
    uint32_t *h_target_lens = (uint32_t*)calloc(n_tasks, sizeof(uint32_t));
    int32_t *h_flag = (int32_t*)calloc(n_tasks, sizeof(int32_t));
    int32_t *h_task_to_align_id = (int32_t*)calloc(n_tasks, sizeof(int32_t));

    // Calculate offsets and prepare sequences
    for (int i = 0; i < n_tasks; i++) {

         // Validate sequence lengths against buffer limits
        if (tasks[i].qlen > max_query_len_limit) {
            fprintf(stderr, "[WARNING] Task %d: qlen=%d exceeds max_query_len=%zu, clamping\n",
                    i, tasks[i].qlen, max_query_len_limit);
            tasks[i].qlen = max_query_len_limit;
        }
        if (tasks[i].tlen > max_query_len_limit) {
            fprintf(stderr, "[WARNING] Task %d: tlen=%d exceeds max_query_len=%zu, clamping\n",
                    i, tasks[i].tlen, max_query_len_limit);
            tasks[i].tlen = max_query_len_limit;
        }
        // Align to 8-byte boundary for AGATHA
        size_t qlen_aligned = ((tasks[i].qlen + 7) / 8) * 8;
        size_t tlen_aligned = ((tasks[i].tlen + 7) / 8) * 8;

        h_query_offsets[i] = total_query_bytes;
        h_target_offsets[i] = total_target_bytes;
        h_query_lens[i] = tasks[i].qlen;
        h_target_lens[i] = tasks[i].tlen;
        h_flag[i] = tasks[i].flag;

        total_query_bytes += qlen_aligned;
        total_target_bytes += tlen_aligned;

        if (tasks[i].qlen > max_query_len) {
            max_query_len = tasks[i].qlen;
        }

        // Store task-to-alignment mapping
        h_task_to_align_id[i] = i;
    }

    // Prepare unpacked sequences
    uint8_t *h_unpacked_query = (uint8_t*)calloc(total_query_bytes, 1);
    uint8_t *h_unpacked_target = (uint8_t*)calloc(total_target_bytes, 1);

    const uint8_t N_BASE = 4; 
    for (int i = 0; i < n_tasks; i++) {
        // Copy sequences
        memcpy(h_unpacked_query + h_query_offsets[i], 
               seq_buffer + tasks[i].qseq_offset, tasks[i].qlen);
        memcpy(h_unpacked_target + h_target_offsets[i], 
               seq_buffer + tasks[i].tseq_offset, tasks[i].tlen);
        
        // Pad with N (0x0F)
        size_t qlen_aligned = ((tasks[i].qlen + 7) / 8) * 8;
        size_t tlen_aligned = ((tasks[i].tlen + 7) / 8) * 8;
        for (int j = tasks[i].qlen; j < qlen_aligned; j++) {
            h_unpacked_query[h_query_offsets[i] + j] = N_BASE;
        }
        for (int j = tasks[i].tlen; j < tlen_aligned; j++) {
            h_unpacked_target[h_target_offsets[i] + j] = N_BASE;
        }
    }

    // Create stream for async operations (we'll use the default stream for now)
    cudaStream_t stream = 0;  // Default stream

    // Copy to GPU
    cudaMemcpy(d_unpacked_query, h_unpacked_query,
                    total_query_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_unpacked_target, h_unpacked_target,
                    total_target_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_query_offsets, h_query_offsets,
                    n_tasks * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target_offsets, h_target_offsets,
                    n_tasks * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query_lens, h_query_lens,
                    n_tasks * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target_lens, h_target_lens,
                    n_tasks * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flag, h_flag,
                    n_tasks * sizeof(int32_t), cudaMemcpyHostToDevice);


    // Launch packing kernel
    int query_tasks_per_thread = (int)ceil((double)total_query_bytes /
                                          (8 * kernel_threads * kernel_blocks));
    int target_tasks_per_thread = (int)ceil((double)total_target_bytes /
                                           (8 * kernel_threads * kernel_blocks));

    gasal_pack_kernel<<<kernel_blocks, kernel_threads>>>(
        (uint32_t*)d_unpacked_query,
        (uint32_t*)d_unpacked_target,
        d_packed_query,
        d_packed_target,
        query_tasks_per_thread,
        target_tasks_per_thread,
        total_query_bytes / 4,
        total_target_bytes / 4
    );
    
    // Launch sorting kernel
    agatha_sort<<<kernel_blocks, kernel_threads>>>(
        d_packed_query,
        d_packed_target,
        d_query_lens,
        d_target_lens,
        d_query_offsets,
        d_target_offsets,
        n_tasks,
        max_query_len_limit,
        (short2*)d_global_buffer
    );

    // Sort on CPU (hybrid approach)
    size_t sort_offset = kernel_blocks * (kernel_threads / 8) *
                        max_query_len_limit * 3;
    cudaMemcpy(h_sort_buffer,
                    (short2*)d_global_buffer + sort_offset,
                    n_tasks * sizeof(short2),
                    cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    std::sort(h_sort_buffer, h_sort_buffer + n_tasks,
              [](short2 a, short2 b) { return a.x < b.x; });

    cudaMemcpy((short2*)d_global_buffer + sort_offset,
                    h_sort_buffer,
                    n_tasks * sizeof(short2),
                    cudaMemcpyHostToDevice);

    // Configure and launch AGATHA kernel
    size_t shared_mem = (kernel_threads / 32) *
                       ((32 * (8 * (g_config.slice_width + 1))) + 28) * sizeof(int32_t);
    cudaFuncSetAttribute(agatha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem);

    int8_t h_scoring_matrix[25];
    ksw_gen_simple_mat(5, h_scoring_matrix, opt->a, opt->b, opt->sc_ambi);
    cudaMemcpy(d_mat, h_scoring_matrix, 25 * sizeof(int8_t),
                    cudaMemcpyHostToDevice);
    int32_t extra_flag = 0;

    // Prepare device result structure
    gasal_res_t h_res_ptrs;
    h_res_ptrs.aln_score = d_scores;
    h_res_ptrs.query_batch_end = d_query_ends;
    h_res_ptrs.target_batch_end = d_target_ends;
    cudaMemcpy(device_res, &h_res_ptrs, sizeof(gasal_res_t), cudaMemcpyHostToDevice);

    fprintf(stderr, "[Info::%s] gpu initialized for ksw with %d tasks\n", __func__, n_tasks);

    // ===== KSW Alignment Kernel (Phase 1: Compute scores and save backtrack) =====
    ksw_semi_global_cuda_kernel<<<kernel_blocks, kernel_threads,
                    shared_mem>>>(
        d_packed_query,
        d_packed_target,
        d_query_lens,
        d_target_lens,
        d_query_offsets,
        d_target_offsets,
        (gasal_res_t*)device_res,
        d_mat,
        d_backtrack_p,
        d_backtrack_off,
        d_backtrack_off_end,
        d_backtrack_n_col,
        max_backtrack_size,
        (ksw_extz_t*)d_ez_array,
        (uint8_t*)d_ksw_temp_buffer,
        d_flag,
        ksw_temp_per_task,
        n_tasks,
        5,  // m = alphabet size(ACGTN)
        opt->zdrop,
        opt->end_bonus
    );

    // ===== KSW Backtracking Kernel (Phase 2: Generate CIGAR) =====
    if (cigar_buffer) {
        ksw_backtrack_kernel<<<kernel_blocks, kernel_threads,
                        shared_mem>>>(
            d_backtrack_p,
            d_backtrack_off,
            d_backtrack_n_col,
            d_query_lens,
            d_target_lens,
            (gasal_res_t*)device_res,
            d_cigar_buffer,
            d_cigar_lengths,
            max_cigar_len,
            max_backtrack_size,
            d_flag,
            n_tasks
        );

        // Copy CIGAR results back to host
        cudaMemcpy(h_cigar_buffer,
                        d_cigar_buffer,
                        n_tasks * max_cigar_len * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost);
        cudaMemcpy(h_cigar_lengths,
                        d_cigar_lengths,
                        n_tasks * sizeof(int),
                        cudaMemcpyDeviceToHost);
    }

    // Copy results back
    cudaMemcpy(h_scores,
                    d_scores,
                    n_tasks * sizeof(int32_t),
                    cudaMemcpyDeviceToHost);
    cudaMemcpy(h_query_ends,
                    d_query_ends,
                    n_tasks * sizeof(int32_t),
                    cudaMemcpyDeviceToHost);
    cudaMemcpy(h_target_ends,
                    d_target_ends,
                    n_tasks * sizeof(int32_t),
                    cudaMemcpyDeviceToHost);

    // Wait for completion
    cudaDeviceSynchronize();
    cudaCheck();

    // Map results back to tasks
    for (int i = 0; i < n_tasks; i++) {
        int align_id = h_task_to_align_id[i];

        tasks[i].score = h_scores[align_id];
        tasks[i].max_q = h_query_ends[align_id];
        tasks[i].max_t = h_target_ends[align_id];

        // Copy CIGAR to output buffer
        if (cigar_buffer) {
            int n_cigar = h_cigar_lengths[align_id];
            tasks[i].n_cigar = n_cigar;

            // Copy CIGAR operations
            if (n_cigar > 0 && tasks[i].cigar_offset + n_cigar <= tasks[i].max_cigar) {
                memcpy(cigar_buffer + tasks[i].cigar_offset,
                       h_cigar_buffer + align_id * max_cigar_len,
                       n_cigar * sizeof(uint32_t));
            }
        } else {
            tasks[i].n_cigar = 0;
        }
        
        // Set completion flags
        tasks[i].reach_end = (tasks[i].max_q == tasks[i].qlen) && 
                            (tasks[i].max_t == tasks[i].tlen);
        tasks[i].zdropped = 0;
    }
    
    // Cleanup host buffers
    free(h_unpacked_query);
    free(h_unpacked_target);
    free(h_query_offsets);
    free(h_target_offsets);
    free(h_query_lens);
    free(h_target_lens);
    free(h_flag);
    free(h_task_to_align_id);
    free(h_cigar_buffer);
    free(h_cigar_lengths);
    free(h_scores);
    free(h_query_ends);
    free(h_target_ends);
    free(h_sort_buffer);
}
