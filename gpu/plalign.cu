#include "plalign.h"
#include "gasal_kernels.h"
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

gpu_align_storage_t *g_storage = NULL;
bool g_initialized = false;

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

static int init_gpu_storage(size_t initial_tasks, size_t initial_seq_bytes) {
    if (g_initialized) return 0;
    
    g_storage = (gpu_align_storage_t*)calloc(1, sizeof(gpu_align_storage_t));
    if (!g_storage) return -1;
    
    // Set configuration
    g_storage->kernel_blocks = g_config.blocks;
    g_storage->kernel_threads = g_config.threads;
    g_storage->max_tasks = initial_tasks;
    g_storage->max_query_bytes = initial_seq_bytes;
    g_storage->max_target_bytes = initial_seq_bytes;
    g_storage->max_query_len = 100000; 

    // Copy substitution scores to device constants
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

    // Create CUDA stream
    cudaStreamCreate(&g_storage->stream);
    cudaMalloc(&g_storage->mat, 25 * sizeof(int8_t));

    // Allocate device memory for sequences
    cudaMalloc(&g_storage->d_unpacked_query, g_storage->max_query_bytes);
    cudaMalloc(&g_storage->d_unpacked_target, g_storage->max_target_bytes);
    cudaMalloc(&g_storage->d_packed_query, (g_storage->max_query_bytes / 8) * sizeof(uint32_t));
    cudaMalloc(&g_storage->d_packed_target, (g_storage->max_target_bytes / 8) * sizeof(uint32_t));
    
    // Allocate metadata arrays
    cudaMalloc(&g_storage->d_query_offsets, g_storage->max_tasks * sizeof(uint32_t));
    cudaMalloc(&g_storage->d_target_offsets, g_storage->max_tasks * sizeof(uint32_t));
    cudaMalloc(&g_storage->d_query_lens, g_storage->max_tasks * sizeof(uint32_t));
    cudaMalloc(&g_storage->d_target_lens, g_storage->max_tasks * sizeof(uint32_t));
    cudaMalloc(&g_storage->d_flag, g_storage->max_tasks * sizeof(int32_t));


     // Allocate AGATHA global buffer
    size_t global_buffer_size = g_storage->kernel_blocks * (g_storage->kernel_threads / 8) * 
                                g_storage->max_query_len * 4;
    cudaMalloc(&g_storage->d_global_buffer, global_buffer_size * sizeof(short2));
    
    // Allocate host sorting buffer
    g_storage->h_sort_buffer = (short2*)calloc(g_storage->max_tasks, sizeof(short2));

    size_t max_len = g_storage->max_query_len;
    size_t H_size = max_len * sizeof(int32_t);
    size_t u8_arrays_size = (max_len + 1) * 7 * sizeof(int8_t); // u,v,x,y,x2,y2,s
    size_t seq_size = max_len * 2 * sizeof(uint8_t);           // qr, target
    size_t raw_size = H_size + u8_arrays_size + seq_size;
    g_storage->ksw_temp_per_task = (raw_size + 7) & ~7ULL;  // 8-byte alignment
    cudaError_t err = cudaMalloc(&g_storage->d_ksw_temp_buffer, 
                             225 * g_storage->ksw_temp_per_task);
    if (err != cudaSuccess) {
        fprintf(stderr, "[ERROR] Failed to allocate d_ksw_temp_buffer: %s (requested %.2f GB)\n",
                cudaGetErrorString(err),
                (g_storage->max_tasks * (double)g_storage->ksw_temp_per_task) / (1024*1024*1024));
        return -1;
    }
    
    // ==================== Backtrack Buffers ====================
    // Purpose: Store information for CIGAR generation (alignment path reconstruction)
    // 
    // For each alignment with qlen and tlen:
    // - Total antidiagonals: qlen + tlen - 1
    // - Cells per antidiagonal: n_col = min(bandwidth+1, min(qlen, tlen))
    //
    // backtrack_p[]: Direction bits for each DP cell
    //   - Size per task: (qlen + tlen - 1) × n_col bytes
    //   - Each byte stores 4 bits for direction (which cell we came from)
    //                     + 4 bits for state flags
    //
    // backtrack_off[], backtrack_off_end[]: Valid range for each antidiagonal
    //   - Size per task: (qlen + tlen - 1) × 2 integers
    //   - Tells us which cells in each antidiagonal are within the band
    //
    // backtrack_n_col[]: The n_col value for each task
    //   - Size per task: 1 integer
    
    // Calculate sizes
    size_t max_antidiag = 2 * g_storage->max_query_len - 1;  // Worst case: qlen = tlen = max_query_len
    
    // n_col = min(bandwidth+1, min(qlen, tlen))
    size_t max_n_col;
    if (g_config.band_width < 0) {
        // No banding, worst case n_col = max_query_len
        max_n_col = g_storage->max_query_len;
    } else {
        // With banding, n_col limited by bandwidth
        max_n_col = g_config.band_width + 1;
    }
    
    // Per-task backtrack sizes
    g_storage->max_backtrack_size = max_antidiag * max_n_col;  // bytes per task
    g_storage->max_cigar_len = 2 * g_storage->max_query_len;   // uint32_t per task (worst case: all indels)
    
    // Memory estimate for full allocation
    size_t backtrack_mem_per_task = g_storage->max_backtrack_size +              // backtrack_p
                                     max_antidiag * 2 * sizeof(int) +            // off + off_end
                                     sizeof(int) +                               // n_col
                                     g_storage->max_cigar_len * sizeof(uint32_t); // cigar
    
    double total_backtrack_gb = (224 * backtrack_mem_per_task) / (1024.0 * 1024.0 * 1024.0);
    
    fprintf(stderr, "[INFO] Backtrack memory estimate: %.2f GB for %zu tasks\n", 
            total_backtrack_gb, 224);
        
    //FIXME: only for testing
    size_t alloc_tasks = 20; 
    
    // Allocate KSW backtracking buffers
    cudaMalloc(&g_storage->d_backtrack_p, alloc_tasks * g_storage->max_backtrack_size);
    cudaMalloc(&g_storage->d_backtrack_off, alloc_tasks * max_antidiag * sizeof(int));
    cudaMalloc(&g_storage->d_backtrack_off_end, alloc_tasks * max_antidiag * sizeof(int));
    cudaMalloc(&g_storage->d_backtrack_n_col, alloc_tasks * sizeof(int));
    cudaMalloc(&g_storage->d_cigar_buffer, alloc_tasks * g_storage->max_cigar_len * sizeof(uint32_t));
    cudaMalloc(&g_storage->d_cigar_lengths, alloc_tasks * sizeof(int));
    
    // Allocate host CIGAR buffers
    g_storage->h_cigar_buffer = (uint32_t*)calloc(alloc_tasks * g_storage->max_cigar_len, sizeof(uint32_t));
    g_storage->h_cigar_lengths = (int*)calloc(alloc_tasks, sizeof(int));
    
    fprintf(stderr, "[INFO] Allocated backtrack buffers for %zu tasks (%.2f MB total)\n",
            alloc_tasks, 
            (alloc_tasks * backtrack_mem_per_task) / (1024.0 * 1024.0));

    // Allocate task mapping
    cudaMalloc(&g_storage->d_task_to_align_id, g_storage->max_tasks * sizeof(int32_t));
    g_storage->h_task_to_align_id = (int32_t*)calloc(g_storage->max_tasks, sizeof(int32_t));
    
    // Allocate result structures
    g_storage->host_res = (gasal_res_t*)calloc(1, sizeof(gasal_res_t));
    g_storage->host_res->aln_score = (int32_t*)calloc(g_storage->max_tasks, sizeof(int32_t));
    g_storage->host_res->query_batch_end = (int32_t*)calloc(g_storage->max_tasks, sizeof(int32_t));
    g_storage->host_res->target_batch_end = (int32_t*)calloc(g_storage->max_tasks, sizeof(int32_t));
    
    // Allocate device result structure
    cudaMalloc(&g_storage->device_res, sizeof(gasal_res_t));
    g_storage->device_res_ptrs = (gasal_res_t*)calloc(1, sizeof(gasal_res_t));
    cudaMalloc(&g_storage->d_ez_array, sizeof(ksw_extz_t) * 224);  // FIXED: use concurrent tasks
    
    // Allocate device result arrays
    int32_t *d_scores, *d_query_ends, *d_target_ends;
    cudaMalloc(&d_scores, g_storage->max_tasks * sizeof(int32_t));
    cudaMalloc(&d_query_ends, g_storage->max_tasks * sizeof(int32_t));
    cudaMalloc(&d_target_ends, g_storage->max_tasks * sizeof(int32_t));
    
    // Store device pointers
    g_storage->device_res_ptrs->aln_score = d_scores;
    g_storage->device_res_ptrs->query_batch_end = d_query_ends;
    g_storage->device_res_ptrs->target_batch_end = d_target_ends;
    
    // Copy device pointers to device struct
    cudaMemcpy(g_storage->device_res, g_storage->device_res_ptrs, 
               sizeof(gasal_res_t), cudaMemcpyHostToDevice);
    
    g_initialized = true;
    
    // Print summary
    fprintf(stderr, "\n========== GPU Storage Initialized ==========\n");
    fprintf(stderr, "Max tasks: %zu (concurrent: %zu)\n", g_storage->max_tasks, 224);
    fprintf(stderr, "Max sequence length: %zu\n", g_storage->max_query_len);
    fprintf(stderr, "Max backtrack size per task: %zu bytes\n", g_storage->max_backtrack_size);
    fprintf(stderr, "Bandwidth: %d\n", g_config.band_width);
    fprintf(stderr, "============================================\n\n");
    
    return 0;
}

static int realloc_gpu_storage(size_t new_tasks, size_t new_seq_bytes) {
    if (!g_initialized) return -1;
    
    bool need_realloc = false;
    
    // Check if we need more task capacity
    if (new_tasks > g_storage->max_tasks) {
        g_storage->max_tasks = new_tasks * 2; // Allocate 2x for growth
        need_realloc = true;
    }
    
    // Check if we need more sequence capacity
    if (new_seq_bytes > g_storage->max_query_bytes || 
        new_seq_bytes > g_storage->max_target_bytes) {
        g_storage->max_query_bytes = new_seq_bytes * 2;
        g_storage->max_target_bytes = new_seq_bytes * 2;
        need_realloc = true;
    }
    
    if (!need_realloc) return 0;
    
    // Recalculate backtracking sizes
    size_t max_antidiag = 2 * g_storage->max_query_len;
    size_t max_n_col = (g_config.band_width < 0) ? 
                       g_storage->max_query_len : (g_config.band_width + 1);
    g_storage->max_backtrack_size = max_antidiag * max_n_col;
    g_storage->max_cigar_len = 2 * g_storage->max_query_len;
    
    // Reallocate sequence buffers
    cudaFree(g_storage->d_unpacked_query);
    cudaFree(g_storage->d_unpacked_target);
    cudaFree(g_storage->d_packed_query);
    cudaFree(g_storage->d_packed_target);
    
    cudaMalloc(&g_storage->d_unpacked_query, g_storage->max_query_bytes);
    cudaMalloc(&g_storage->d_unpacked_target, g_storage->max_target_bytes);
    cudaMalloc(&g_storage->d_packed_query, (g_storage->max_query_bytes / 8) * sizeof(uint32_t));
    cudaMalloc(&g_storage->d_packed_target, (g_storage->max_target_bytes / 8) * sizeof(uint32_t));
    
    // Reallocate metadata arrays
    cudaFree(g_storage->d_query_offsets);
    cudaFree(g_storage->d_target_offsets);
    cudaFree(g_storage->d_query_lens);
    cudaFree(g_storage->d_target_lens);
    
    cudaMalloc(&g_storage->d_query_offsets, g_storage->max_tasks * sizeof(uint32_t));
    cudaMalloc(&g_storage->d_target_offsets, g_storage->max_tasks * sizeof(uint32_t));
    cudaMalloc(&g_storage->d_query_lens, g_storage->max_tasks * sizeof(uint32_t));
    cudaMalloc(&g_storage->d_target_lens, g_storage->max_tasks * sizeof(uint32_t));
    
    // Reallocate backtracking buffers
    cudaFree(g_storage->d_backtrack_p);
    cudaFree(g_storage->d_backtrack_off);
    cudaFree(g_storage->d_backtrack_n_col);
    cudaFree(g_storage->d_cigar_buffer);
    cudaFree(g_storage->d_cigar_lengths);
    
    cudaMalloc(&g_storage->d_backtrack_p, 
               g_storage->max_tasks * g_storage->max_backtrack_size);
    cudaMalloc(&g_storage->d_backtrack_off, 
               g_storage->max_tasks * max_antidiag * sizeof(int));
    cudaMalloc(&g_storage->d_backtrack_n_col, 
               g_storage->max_tasks * sizeof(int));
    cudaMalloc(&g_storage->d_cigar_buffer, 
               g_storage->max_tasks * g_storage->max_cigar_len * sizeof(uint32_t));
    cudaMalloc(&g_storage->d_cigar_lengths, 
               g_storage->max_tasks * sizeof(int));
    
    // Reallocate host CIGAR buffers
    free(g_storage->h_cigar_buffer);
    free(g_storage->h_cigar_lengths);
    g_storage->h_cigar_buffer = (uint32_t*)calloc(
        g_storage->max_tasks * g_storage->max_cigar_len, sizeof(uint32_t));
    g_storage->h_cigar_lengths = (int*)calloc(g_storage->max_tasks, sizeof(int));
    
    // Reallocate other task-dependent buffers
    cudaFree(g_storage->d_task_to_align_id);
    free(g_storage->h_task_to_align_id);
    free(g_storage->h_sort_buffer);
    
    cudaMalloc(&g_storage->d_task_to_align_id, g_storage->max_tasks * sizeof(int32_t));
    g_storage->h_task_to_align_id = (int32_t*)calloc(g_storage->max_tasks, sizeof(int32_t));
    g_storage->h_sort_buffer = (short2*)calloc(g_storage->max_tasks, sizeof(short2));
    
    // Reallocate result arrays
    free(g_storage->host_res->aln_score);
    free(g_storage->host_res->query_batch_end);
    free(g_storage->host_res->target_batch_end);
    
    g_storage->host_res->aln_score = (int32_t*)calloc(g_storage->max_tasks, sizeof(int32_t));
    g_storage->host_res->query_batch_end = (int32_t*)calloc(g_storage->max_tasks, sizeof(int32_t));
    g_storage->host_res->target_batch_end = (int32_t*)calloc(g_storage->max_tasks, sizeof(int32_t));
    
    cudaFree(g_storage->device_res_ptrs->aln_score);
    cudaFree(g_storage->device_res_ptrs->query_batch_end);
    cudaFree(g_storage->device_res_ptrs->target_batch_end);
    
    cudaMalloc(&g_storage->device_res_ptrs->aln_score, g_storage->max_tasks * sizeof(int32_t));
    cudaMalloc(&g_storage->device_res_ptrs->query_batch_end, g_storage->max_tasks * sizeof(int32_t));
    cudaMalloc(&g_storage->device_res_ptrs->target_batch_end, g_storage->max_tasks * sizeof(int32_t));
    
    cudaMemcpy(g_storage->device_res, g_storage->device_res_ptrs, 
               sizeof(gasal_res_t), cudaMemcpyHostToDevice);
    
    return 0;
}

extern "C" void gpu_align_batch_execute(const mm_mapopt_t *opt, gpu_align_task_t *tasks, int n_tasks, 
                            uint8_t *seq_buffer, uint32_t *cigar_buffer);
void gpu_align_batch_execute(const mm_mapopt_t *opt, gpu_align_task_t *tasks, int n_tasks, 
                            uint8_t *seq_buffer, uint32_t *cigar_buffer) {
    if (n_tasks <= 0) return;
    
    // Initialize storage if needed
    if (!g_initialized) {
        if (init_gpu_storage(20000, 100*1024*1024) != 0) {
            fprintf(stderr, "[ERROR] Failed to initialize GPU storage\n");
            return;
        }
    }
    n_tasks = 20;  
    // Calculate memory requirements
    size_t total_query_bytes = 0, total_target_bytes = 0;
    uint32_t max_query_len = 0;
    
    // Host arrays for batch preparation
    uint32_t *h_query_offsets = (uint32_t*)calloc(n_tasks, sizeof(uint32_t));
    uint32_t *h_target_offsets = (uint32_t*)calloc(n_tasks, sizeof(uint32_t));
    uint32_t *h_query_lens = (uint32_t*)calloc(n_tasks, sizeof(uint32_t));
    uint32_t *h_target_lens = (uint32_t*)calloc(n_tasks, sizeof(uint32_t));
    int32_t *h_flag = (int32_t*)calloc(n_tasks, sizeof(int32_t));
    
    // Calculate offsets and prepare sequences
    for (int i = 0; i < n_tasks; i++) {
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
        g_storage->h_task_to_align_id[i] = i;
    }
    
    // Update max query length if needed
    if (max_query_len > g_storage->max_query_len) {
        g_storage->max_query_len = max_query_len;
    }
    
    // Reallocate if needed
    size_t max_bytes = (total_query_bytes > total_target_bytes) ? 
                       total_query_bytes : total_target_bytes;
    // Prepare unpacked sequences
    uint8_t *h_unpacked_query = (uint8_t*)calloc(total_query_bytes, 1);
    uint8_t *h_unpacked_target = (uint8_t*)calloc(total_target_bytes, 1);
  
    if (realloc_gpu_storage(n_tasks, max_bytes) != 0) {
        fprintf(stderr, "[ERROR] Failed to reallocate GPU storage\n");
        free(h_unpacked_query);
        free(h_unpacked_target);
        free(h_query_offsets);
        free(h_target_offsets);
        free(h_query_lens);
        free(h_target_lens);
        return;
    }
    
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

    // Copy to GPU
    cudaMemcpyAsync(g_storage->d_unpacked_query, h_unpacked_query, 
                    total_query_bytes, cudaMemcpyHostToDevice, g_storage->stream);
    cudaMemcpyAsync(g_storage->d_unpacked_target, h_unpacked_target, 
                    total_target_bytes, cudaMemcpyHostToDevice, g_storage->stream);
    cudaMemcpyAsync(g_storage->d_query_offsets, h_query_offsets, 
                    n_tasks * sizeof(uint32_t), cudaMemcpyHostToDevice, g_storage->stream);
    cudaMemcpyAsync(g_storage->d_target_offsets, h_target_offsets, 
                    n_tasks * sizeof(uint32_t), cudaMemcpyHostToDevice, g_storage->stream);
    cudaMemcpyAsync(g_storage->d_query_lens, h_query_lens, 
                    n_tasks * sizeof(uint32_t), cudaMemcpyHostToDevice, g_storage->stream);
    cudaMemcpyAsync(g_storage->d_target_lens, h_target_lens, 
                    n_tasks * sizeof(uint32_t), cudaMemcpyHostToDevice, g_storage->stream);
    cudaMemcpyAsync(g_storage->d_flag, h_flag, 
                    n_tasks * sizeof(int32_t), cudaMemcpyHostToDevice, g_storage->stream);
    

    // Launch packing kernel
    int query_tasks_per_thread = (int)ceil((double)total_query_bytes / 
                                          (8 * g_storage->kernel_threads * g_storage->kernel_blocks));
    int target_tasks_per_thread = (int)ceil((double)total_target_bytes / 
                                           (8 * g_storage->kernel_threads * g_storage->kernel_blocks));
    
    gasal_pack_kernel<<<g_storage->kernel_blocks, g_storage->kernel_threads, 0, g_storage->stream>>>(
        (uint32_t*)g_storage->d_unpacked_query,
        (uint32_t*)g_storage->d_unpacked_target,
        g_storage->d_packed_query,
        g_storage->d_packed_target,
        query_tasks_per_thread,
        target_tasks_per_thread,
        total_query_bytes / 4,
        total_target_bytes / 4
    );
    
    // Launch sorting kernel
    agatha_sort<<<g_storage->kernel_blocks, g_storage->kernel_threads, 0, g_storage->stream>>>(
        g_storage->d_packed_query,
        g_storage->d_packed_target,
        g_storage->d_query_lens,
        g_storage->d_target_lens,
        g_storage->d_query_offsets,
        g_storage->d_target_offsets,
        n_tasks,
        g_storage->max_query_len,
        g_storage->d_global_buffer
    );
    
    // Sort on CPU (hybrid approach)
    size_t sort_offset = g_storage->kernel_blocks * (g_storage->kernel_threads / 8) * 
                        g_storage->max_query_len * 3;
    cudaMemcpyAsync(g_storage->h_sort_buffer, 
                    g_storage->d_global_buffer + sort_offset,
                    n_tasks * sizeof(short2), 
                    cudaMemcpyDeviceToHost, g_storage->stream);
    cudaStreamSynchronize(g_storage->stream);
    
    std::sort(g_storage->h_sort_buffer, g_storage->h_sort_buffer + n_tasks, 
              [](short2 a, short2 b) { return a.x < b.x; });
    
    cudaMemcpyAsync(g_storage->d_global_buffer + sort_offset,
                    g_storage->h_sort_buffer,
                    n_tasks * sizeof(short2), 
                    cudaMemcpyHostToDevice, g_storage->stream);
    
    // Configure and launch AGATHA kernel
    size_t shared_mem = (g_storage->kernel_threads / 32) * 
                       ((32 * (8 * (g_config.slice_width + 1))) + 28) * sizeof(int32_t);
    cudaFuncSetAttribute(agatha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem);
    
    int8_t h_scoring_matrix[25];
    ksw_gen_simple_mat(5, h_scoring_matrix, opt->a, opt->b, opt->sc_ambi);
    cudaMemcpyAsync(g_storage->mat, h_scoring_matrix, 25 * sizeof(int8_t),
                    cudaMemcpyHostToDevice, g_storage->stream);
    int32_t extra_flag = 0;

    // ===== KSW Alignment Kernel (Phase 1: Compute scores and save backtrack) =====
    ksw_semi_global_cuda_kernel<<<g_storage->kernel_blocks, g_storage->kernel_threads, 
                    shared_mem, g_storage->stream>>>(
        g_storage->d_packed_query,
        g_storage->d_packed_target,
        g_storage->d_query_lens,
        g_storage->d_target_lens,
        g_storage->d_query_offsets,
        g_storage->d_target_offsets,
        g_storage->device_res,
        g_storage->mat, 
        g_storage->d_backtrack_p,
        g_storage->d_backtrack_off,
        g_storage->d_backtrack_off_end,
        g_storage->d_backtrack_n_col,
        g_storage->max_backtrack_size,
        g_storage->d_ez_array,
        g_storage->d_ksw_temp_buffer,
        g_storage->d_flag,
        g_storage->ksw_temp_per_task,
        n_tasks,
        5,  // m = alphabet size(ACGTN)
        opt->zdrop,
        opt->end_bonus
    );

    // ===== KSW Backtracking Kernel (Phase 2: Generate CIGAR) =====
    if (cigar_buffer) {
        ksw_backtrack_kernel<<<g_storage->kernel_blocks, g_storage->kernel_threads,
                        shared_mem, g_storage->stream>>>(
            g_storage->d_backtrack_p,
            g_storage->d_backtrack_off,
            g_storage->d_backtrack_n_col,
            g_storage->d_query_lens,
            g_storage->d_target_lens,
            g_storage->device_res,
            g_storage->d_cigar_buffer,
            g_storage->d_cigar_lengths,
            g_storage->max_cigar_len,
            g_storage->max_backtrack_size,
            g_storage->d_flag,
            n_tasks
        );

        // Copy CIGAR results back to host
        cudaMemcpyAsync(g_storage->h_cigar_buffer,
                        g_storage->d_cigar_buffer,
                        n_tasks * g_storage->max_cigar_len * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost, g_storage->stream);
        cudaMemcpyAsync(g_storage->h_cigar_lengths,
                        g_storage->d_cigar_lengths,
                        n_tasks * sizeof(int),
                        cudaMemcpyDeviceToHost, g_storage->stream);
    }

    // Copy results back
    cudaMemcpyAsync(g_storage->host_res->aln_score, 
                    g_storage->device_res_ptrs->aln_score,
                    n_tasks * sizeof(int32_t), 
                    cudaMemcpyDeviceToHost, g_storage->stream);
    cudaMemcpyAsync(g_storage->host_res->query_batch_end, 
                    g_storage->device_res_ptrs->query_batch_end,
                    n_tasks * sizeof(int32_t), 
                    cudaMemcpyDeviceToHost, g_storage->stream);
    cudaMemcpyAsync(g_storage->host_res->target_batch_end, 
                    g_storage->device_res_ptrs->target_batch_end,
                    n_tasks * sizeof(int32_t), 
                    cudaMemcpyDeviceToHost, g_storage->stream);
    
    // Wait for completion
    cudaStreamSynchronize(g_storage->stream);
    
    // Map results back to tasks
    for (int i = 0; i < n_tasks; i++) {
        int align_id = g_storage->h_task_to_align_id[i];
        
        tasks[i].score = g_storage->host_res->aln_score[align_id];
        tasks[i].max_q = g_storage->host_res->query_batch_end[align_id];
        tasks[i].max_t = g_storage->host_res->target_batch_end[align_id];
        
        // Copy CIGAR to output buffer
        if (cigar_buffer) {
            int n_cigar = g_storage->h_cigar_lengths[align_id];
            tasks[i].n_cigar = n_cigar;
            
            // Copy CIGAR operations
            if (n_cigar > 0 && tasks[i].cigar_offset + n_cigar <= tasks[i].max_cigar) {
                memcpy(cigar_buffer + tasks[i].cigar_offset,
                       g_storage->h_cigar_buffer + align_id * g_storage->max_cigar_len,
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
}

void gpu_align_cleanup() {
    if (!g_initialized) return;
    
    // Free device memory
    cudaFree(g_storage->d_unpacked_query);
    cudaFree(g_storage->d_unpacked_target);
    cudaFree(g_storage->d_packed_query);
    cudaFree(g_storage->d_packed_target);
    cudaFree(g_storage->d_query_offsets);
    cudaFree(g_storage->d_target_offsets);
    cudaFree(g_storage->d_query_lens);
    cudaFree(g_storage->d_target_lens);
    cudaFree(g_storage->d_task_to_align_id);
    cudaFree(g_storage->d_global_buffer);
    cudaFree(g_storage->device_res);
    cudaFree(g_storage->device_res_ptrs->aln_score);
    cudaFree(g_storage->device_res_ptrs->query_batch_end);
    cudaFree(g_storage->device_res_ptrs->target_batch_end);
    
    // Free host memory
    free(g_storage->h_task_to_align_id);
    free(g_storage->host_res->aln_score);
    free(g_storage->host_res->query_batch_end);
    free(g_storage->host_res->target_batch_end);
    free(g_storage->host_res);
    free(g_storage->device_res_ptrs);
    free(g_storage->h_sort_buffer);
    
    // Destroy CUDA resources
    cudaStreamDestroy(g_storage->stream);
    
    free(g_storage);
    g_storage = NULL;
    g_initialized = false;
}