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
    .slice_width = 32,
    .z_threshold = 100,
    .band_width = 500,
    .match_score = 2,
    .mismatch_score = -4,
    .gap_open = 4,
    .gap_extend = 2
};

gpu_align_storage_t *g_storage = NULL;
bool g_initialized = false;

void gasal_copy_subst_scores(gasal_subst_scores *subst){

	cudaError_t err;
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaGapO, &(subst->gap_open), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaGapExtend, &(subst->gap_extend), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	int32_t gapoe = (subst->gap_open + subst->gap_extend);
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaGapOE, &(gapoe), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaMatchScore, &(subst->match), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaMismatchScore, &(subst->mismatch), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
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
    g_storage->slice_width = g_config.slice_width;
    g_storage->z_threshold = g_config.z_threshold;
    g_storage->band_width = g_config.band_width;
    
    // Set initial capacities
    g_storage->max_tasks = initial_tasks;
    g_storage->max_query_bytes = initial_seq_bytes;
    g_storage->max_target_bytes = initial_seq_bytes;
    g_storage->max_query_len = 100000; // Default max query length
    
    // Create CUDA stream
    cudaStreamCreate(&g_storage->stream);
    
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
    
    // Allocate AGATHA global buffer
    size_t global_buffer_size = g_storage->kernel_blocks * (g_storage->kernel_threads / 8) * 
                                g_storage->max_query_len * 4;
    cudaMalloc(&g_storage->d_global_buffer, global_buffer_size * sizeof(short2));
    
    // Allocate host sorting buffer
    g_storage->h_sort_buffer = (short2*)calloc(g_storage->max_tasks, sizeof(short2));
    
    // Copy substitution scores to device constants
    gasal_subst_scores subst;
    subst.match = g_config.match_score;
    subst.mismatch = g_config.mismatch_score;
    subst.gap_open = g_config.gap_open;
    subst.gap_extend = g_config.gap_extend;
    subst.slice_width = g_config.slice_width;
    subst.z_threshold = g_config.z_threshold;
    subst.band_width = g_config.band_width;
    gasal_copy_subst_scores(&subst);
    
    g_initialized = true;
    return 0;
}

static int realloc_gpu_storage(size_t needed_tasks, size_t needed_seq_bytes) {
    if (!g_initialized) return -1;
    
    bool needs_realloc = false;
    size_t new_max_tasks = g_storage->max_tasks;
    size_t new_max_bytes = g_storage->max_query_bytes;
    
    // Check if reallocation needed
    if (needed_tasks > g_storage->max_tasks) {
        new_max_tasks = needed_tasks * 2;
        needs_realloc = true;
    }
    
    if (needed_seq_bytes > g_storage->max_query_bytes) {
        new_max_bytes = needed_seq_bytes * 2;
        needs_realloc = true;
    }
    
    if (!needs_realloc) return 0;
    
    // Reallocate sequences
    if (new_max_bytes > g_storage->max_query_bytes) {
        cudaFree(g_storage->d_unpacked_query);
        cudaFree(g_storage->d_unpacked_target);
        cudaFree(g_storage->d_packed_query);
        cudaFree(g_storage->d_packed_target);
        
        cudaMalloc(&g_storage->d_unpacked_query, new_max_bytes);
        cudaMalloc(&g_storage->d_unpacked_target, new_max_bytes);
        cudaMalloc(&g_storage->d_packed_query, (new_max_bytes / 8) * sizeof(uint32_t));
        cudaMalloc(&g_storage->d_packed_target, (new_max_bytes / 8) * sizeof(uint32_t));
        
        g_storage->max_query_bytes = new_max_bytes;
        g_storage->max_target_bytes = new_max_bytes;
    }
    
    // Reallocate task-related arrays
    if (new_max_tasks > g_storage->max_tasks) {
        cudaFree(g_storage->d_query_offsets);
        cudaFree(g_storage->d_target_offsets);
        cudaFree(g_storage->d_query_lens);
        cudaFree(g_storage->d_target_lens);
        cudaFree(g_storage->d_task_to_align_id);
        
        cudaMalloc(&g_storage->d_query_offsets, new_max_tasks * sizeof(uint32_t));
        cudaMalloc(&g_storage->d_target_offsets, new_max_tasks * sizeof(uint32_t));
        cudaMalloc(&g_storage->d_query_lens, new_max_tasks * sizeof(uint32_t));
        cudaMalloc(&g_storage->d_target_lens, new_max_tasks * sizeof(uint32_t));
        cudaMalloc(&g_storage->d_task_to_align_id, new_max_tasks * sizeof(int32_t));
        
        // Reallocate host arrays
        free(g_storage->h_task_to_align_id);
        free(g_storage->host_res->aln_score);
        free(g_storage->host_res->query_batch_end);
        free(g_storage->host_res->target_batch_end);
        free(g_storage->h_sort_buffer);
        
        g_storage->h_task_to_align_id = (int32_t*)calloc(new_max_tasks, sizeof(int32_t));
        g_storage->host_res->aln_score = (int32_t*)calloc(new_max_tasks, sizeof(int32_t));
        g_storage->host_res->query_batch_end = (int32_t*)calloc(new_max_tasks, sizeof(int32_t));
        g_storage->host_res->target_batch_end = (int32_t*)calloc(new_max_tasks, sizeof(int32_t));
        g_storage->h_sort_buffer = (short2*)calloc(new_max_tasks, sizeof(short2));
        
        // Reallocate device results
        cudaFree(g_storage->device_res_ptrs->aln_score);
        cudaFree(g_storage->device_res_ptrs->query_batch_end);
        cudaFree(g_storage->device_res_ptrs->target_batch_end);
        
        int32_t *d_scores, *d_query_ends, *d_target_ends;
        cudaMalloc(&d_scores, new_max_tasks * sizeof(int32_t));
        cudaMalloc(&d_query_ends, new_max_tasks * sizeof(int32_t));
        cudaMalloc(&d_target_ends, new_max_tasks * sizeof(int32_t));
        
        g_storage->device_res_ptrs->aln_score = d_scores;
        g_storage->device_res_ptrs->query_batch_end = d_query_ends;
        g_storage->device_res_ptrs->target_batch_end = d_target_ends;
        
        cudaMemcpy(g_storage->device_res, g_storage->device_res_ptrs, 
                   sizeof(gasal_res_t), cudaMemcpyHostToDevice);
        
        g_storage->max_tasks = new_max_tasks;
    }
    
    return 0;
}

extern "C" void gpu_align_batch_execute(gpu_align_task_t *tasks, int n_tasks, 
                            uint8_t *seq_buffer, uint32_t *cigar_buffer);
void gpu_align_batch_execute(gpu_align_task_t *tasks, int n_tasks, 
                            uint8_t *seq_buffer, uint32_t *cigar_buffer) {
    if (n_tasks <= 0) return;
    
    // Initialize storage if needed
    if (!g_initialized) {
        if (init_gpu_storage(1024, 1024*1024) != 0) {
            fprintf(stderr, "[ERROR] Failed to initialize GPU storage\n");
            return;
        }
    }
    
    // Calculate memory requirements
    size_t total_query_bytes = 0, total_target_bytes = 0;
    uint32_t max_query_len = 0;
    
    // Host arrays for batch preparation
    uint32_t *h_query_offsets = (uint32_t*)calloc(n_tasks, sizeof(uint32_t));
    uint32_t *h_target_offsets = (uint32_t*)calloc(n_tasks, sizeof(uint32_t));
    uint32_t *h_query_lens = (uint32_t*)calloc(n_tasks, sizeof(uint32_t));
    uint32_t *h_target_lens = (uint32_t*)calloc(n_tasks, sizeof(uint32_t));
    
    // Calculate offsets and prepare sequences
    for (int i = 0; i < n_tasks; i++) {
        // Align to 8-byte boundary for AGATHA
        size_t qlen_aligned = ((tasks[i].qlen + 7) / 8) * 8;
        size_t tlen_aligned = ((tasks[i].tlen + 7) / 8) * 8;
        
        h_query_offsets[i] = total_query_bytes;
        h_target_offsets[i] = total_target_bytes;
        h_query_lens[i] = tasks[i].qlen;
        h_target_lens[i] = tasks[i].tlen;
        
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
            h_unpacked_query[h_query_offsets[i] + j] = 0x0F;
        }
        for (int j = tasks[i].tlen; j < tlen_aligned; j++) {
            h_unpacked_target[h_target_offsets[i] + j] = 0x0F;
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
                       ((32 * (8 * (g_storage->slice_width + 1))) + 28) * sizeof(int32_t);
    cudaFuncSetAttribute(agatha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem);
    
    agatha_kernel<<<g_storage->kernel_blocks, g_storage->kernel_threads, 
                    shared_mem, g_storage->stream>>>(
        g_storage->d_packed_query,
        g_storage->d_packed_target,
        g_storage->d_query_lens,
        g_storage->d_target_lens,
        g_storage->d_query_offsets,
        g_storage->d_target_offsets,
        g_storage->device_res,
        NULL, // device_res_second not used
        g_storage->d_packed_tb_matrices,
        n_tasks,
        g_storage->max_query_len,
        g_storage->d_global_buffer
    );
    
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
        
        // Generate basic CIGAR if buffer provided
        if (cigar_buffer && tasks[i].cigar_offset < tasks[i].max_cigar) {
            // For now, generate simple match CIGAR
            // TODO: Implement proper CIGAR generation or traceback
            int match_len = (tasks[i].max_q < tasks[i].max_t) ? 
                           tasks[i].max_q : tasks[i].max_t;
            if (match_len > 0) {
                cigar_buffer[tasks[i].cigar_offset] = (match_len << 4) | 0; // M operation
                tasks[i].n_cigar = 1;
            } else {
                tasks[i].n_cigar = 0;
            }
        }
        
        // Set completion flags
        tasks[i].reach_end = (tasks[i].max_q == tasks[i].qlen) && 
                            (tasks[i].max_t == tasks[i].tlen);
        tasks[i].zdropped = 0; // AGATHA doesn't provide zdrop info directly
    }
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