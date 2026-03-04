#include "plalign.cuh"
#include "gasal_kernels.h"
#include "plmem.cuh"  // For deviceMemPtr
#include "plksw_kernel.cuh"


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

// Kernel to initialize gasal_res_t structure on device
// This avoids cudaMemcpy host-to-device structure alignment issues
__global__ void init_gasal_res(gasal_res_t *res,
                                int32_t *aln_score, int32_t *query_batch_end, int32_t *target_batch_end,
                                int32_t *mqe, int32_t *mqe_t, int32_t *mte, int32_t *mte_q) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        res->aln_score = aln_score;
        res->query_batch_end = query_batch_end;
        res->target_batch_end = target_batch_end;
        res->query_batch_start = NULL;
        res->target_batch_start = NULL;
        res->mqe = mqe;
        res->mqe_t = mqe_t;
        res->mte = mte;
        res->mte_q = mte_q;
        res->cigar = NULL;
        res->n_cigar_ops = NULL;
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

    // ========== Two-Tier Batched Processing Setup ==========
    // Strategy: Process short tasks (max_len <= 1000bp) first with large batches,
    //           then process long tasks (max_len > 1000bp) with smaller batches
    //
    // This optimizes memory usage because:
    // - Short tasks (90% of workload): 10,000 tasks/batch × 1.5MB = ~15GB backtrack
    // - Long tasks (10% of workload): 200 tasks/batch × 75MB = ~14.3GB backtrack

    int kernel_blocks = 28;
    size_t short_task_max_len = dev_mem->short_task_max_len;      // 1000bp
    size_t short_batch_size = dev_mem->short_task_batch_size;      // 10,000
    size_t long_batch_size = dev_mem->long_task_batch_size;        // 200

    // Phase 1: Classify tasks by length
    // CRITICAL: Use max(qlen, tlen) NOT (qlen+tlen) to prevent buffer overflow
    int *task_indices_short = (int*)malloc(n_tasks * sizeof(int));
    int *task_indices_long = (int*)malloc(n_tasks * sizeof(int));
    int n_short_tasks = 0;
    int n_long_tasks = 0;

    for (int i = 0; i < n_tasks; i++) {
        size_t max_seq_len = (tasks[i].qlen > tasks[i].tlen) ? tasks[i].qlen : tasks[i].tlen;
        if (max_seq_len <= short_task_max_len) {
            task_indices_short[n_short_tasks++] = i;
        } else {
            task_indices_long[n_long_tasks++] = i;
        }
    }

    int total_batches_short = (n_short_tasks + short_batch_size - 1) / short_batch_size;
    int total_batches_long = (n_long_tasks + long_batch_size - 1) / long_batch_size;
    int total_batches = total_batches_short + total_batches_long;

    fprintf(stderr, "[Info::%s] Two-tier processing: %d short tasks (max_len≤%zubp) + %d long tasks (max_len>%zubp)\n",
            __func__, n_short_tasks, short_task_max_len, n_long_tasks, short_task_max_len);
    fprintf(stderr, "[Info::%s]   Short: %d batches × %zu tasks/batch\n",
            __func__, total_batches_short, short_batch_size);
    fprintf(stderr, "[Info::%s]   Long:  %d batches × %zu tasks/batch\n",
            __func__, total_batches_long, long_batch_size);

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
    void *d_ksw_temp_buffer = dev_mem->d_align_ksw_temp_buffer;
    size_t ksw_temp_per_task = dev_mem->align_ksw_temp_per_task;
    uint8_t *d_backtrack_p = dev_mem->d_align_backtrack_p;
    int *d_backtrack_off = dev_mem->d_align_backtrack_off;
    int *d_backtrack_off_end = dev_mem->d_align_backtrack_off_end;
    uint32_t *d_cigar_buffer = dev_mem->d_align_cigar_buffer;
    int *d_cigar_lengths = dev_mem->d_align_cigar_lengths;
    size_t max_backtrack_size = dev_mem->max_align_backtrack_size;
    size_t max_cigar_len = dev_mem->max_align_cigar_len;
    size_t max_query_len_limit = dev_mem->max_align_query_len;
    int8_t *d_mat = dev_mem->d_align_mat;
    void *device_res = dev_mem->d_align_device_res;
    int32_t *d_scores = dev_mem->d_align_scores;
    int32_t *d_query_ends = dev_mem->d_align_query_ends;
    int32_t *d_target_ends = dev_mem->d_align_target_ends;
    int32_t *d_mqe = dev_mem->d_align_mqe;
    int32_t *d_mqe_t = dev_mem->d_align_mqe_t;
    int32_t *d_mte = dev_mem->d_align_mte;
    int32_t *d_mte_q = dev_mem->d_align_mte_q;
    int  *d_task_counter = dev_mem->d_align_task_counter;
    int   n_concurrent_blocks = dev_mem->n_align_concurrent_blocks;

    // Host buffers for CIGAR and results
    // Note: Device buffer is sized for 10000 short tasks OR 200 long tasks (same total size)
    // Persistent kernel batch sizes: limited by CIGAR buffer (960MB = max_align_tasks × max_cigar_len).
    // Short tasks: one batch = max_align_tasks (120000) → ceil(n_short/120000) launches.
    // Long  tasks: one batch = 960MB / (long_cigar_len × 4) = 2400 → ceil(n_long/2400) launches.
    // (Previously: short=7000→23 launches, long=128→31 launches)
    size_t long_cigar_len = 2 * max_query_len_limit;  // 100000 for long tasks (50000bp)
    size_t short_batch_persistent = (size_t)dev_mem->max_align_tasks;  // 120000
    size_t cigar_buf_total_tasks   = (size_t)dev_mem->max_align_tasks;  // device allocation
    size_t long_batch_persistent   = cigar_buf_total_tasks * max_cigar_len / long_cigar_len;  // 2400
    size_t max_batch_size = (short_batch_persistent > long_batch_persistent) ?
                             short_batch_persistent : long_batch_persistent;  // 120000
    // CIGAR host buffer covers the largest phase (short: 120000×2000×4=960MB)
    size_t cigar_buffer_size = max_batch_size * max_cigar_len;  // 240M uint32_t entries = 960MB
    uint32_t *h_cigar_buffer = (uint32_t*)calloc(cigar_buffer_size, sizeof(uint32_t));
    int *h_cigar_lengths = (int*)calloc(max_batch_size, sizeof(int));
    int32_t *h_scores = (int32_t*)calloc(max_batch_size, sizeof(int32_t));
    int32_t *h_query_ends = (int32_t*)calloc(max_batch_size, sizeof(int32_t));
    int32_t *h_target_ends = (int32_t*)calloc(max_batch_size, sizeof(int32_t));
    int32_t *h_mqe = (int32_t*)calloc(max_batch_size, sizeof(int32_t));
    int32_t *h_mqe_t = (int32_t*)calloc(max_batch_size, sizeof(int32_t));
    int32_t *h_mte = (int32_t*)calloc(max_batch_size, sizeof(int32_t));
    int32_t *h_mte_q = (int32_t*)calloc(max_batch_size, sizeof(int32_t));

    // Host arrays for batch preparation (sized for LARGEST batch)
    uint32_t *h_query_offsets = (uint32_t*)calloc(max_batch_size, sizeof(uint32_t));
    uint32_t *h_target_offsets = (uint32_t*)calloc(max_batch_size, sizeof(uint32_t));
    uint32_t *h_query_lens = (uint32_t*)calloc(max_batch_size, sizeof(uint32_t));
    uint32_t *h_target_lens = (uint32_t*)calloc(max_batch_size, sizeof(uint32_t));
    int32_t *h_flag = (int32_t*)calloc(max_batch_size, sizeof(int32_t));
    int32_t *h_task_to_align_id = (int32_t*)calloc(max_batch_size, sizeof(int32_t));

    int8_t h_scoring_matrix[25];
    ksw_gen_simple_mat(5, h_scoring_matrix, opt->a, opt->b, opt->sc_ambi);
    cudaError_t err;
    CHECKCUDAERROR(cudaMemcpy(d_mat, h_scoring_matrix, 25 * sizeof(int8_t), cudaMemcpyHostToDevice));

    // Initialize device result structure directly on device
    // Using a kernel avoids host-device structure alignment issues with cudaMemcpy
    // Performance impact: ~5-10 microseconds (negligible compared to alignment kernel runtime)
    init_gasal_res<<<1, 1>>>((gasal_res_t*)device_res, d_scores, d_query_ends, d_target_ends,
                              d_mqe, d_mqe_t, d_mte, d_mte_q);
    CHECKCUDAERROR(cudaGetLastError());

    // ========== TWO-TIER BATCHED PROCESSING LOOP ==========
    // We'll process both phases (short and long) using a unified loop
    // Phase selector: 0 = short tasks, 1 = long tasks

    int batch_num = 0;
    int total_tasks_processed = 0;

    for (int phase = 0; phase < 2; phase++) {
        // Select phase-specific parameters
        int *current_task_indices = (phase == 0) ? task_indices_short : task_indices_long;
        int n_tasks_in_phase = (phase == 0) ? n_short_tasks : n_long_tasks;
        // Use persistent-kernel batch sizes (CIGAR-buffer limited, not backtrack-limited)
        size_t current_batch_size = (phase == 0) ? short_batch_persistent : long_batch_persistent;
        const char *phase_name = (phase == 0) ? "Short Tasks" : "Long Tasks";

        // Dynamic backtrack buffer sizing based on phase
        size_t current_max_antidiag = (phase == 0) ? (2 * short_task_max_len) : (2 * dev_mem->max_align_query_len);
        size_t current_max_backtrack_size = current_max_antidiag * 752;  // 752 = bandwidth + 1
        size_t current_max_cigar_len = (phase == 0) ? (2 * short_task_max_len) : (2 * dev_mem->max_align_query_len);

        if (n_tasks_in_phase == 0) continue;  // Skip empty phase

        fprintf(stderr, "[Info::%s] === Processing %s (phase %d/2) ===\n", __func__, phase_name, phase + 1);

        // Problem: result buffers are 120,000 elements but we only clear phase_batch_size
        // This causes long phase to read stale data from short phase!
        // Solution: Clear ENTIRE physical buffer allocation at phase start

        // Clear ksw_temp_buffer (allocated for short_batch_size tasks)
        size_t ksw_temp_buffer_size = short_batch_size * ksw_temp_per_task;
        //cudaMemset(d_ksw_temp_buffer, 0, ksw_temp_buffer_size);

        // Both short (10000×2000) and long (200×100000) phases use same 15GB buffer with different strides
        // Must clear entire 15GB to prevent cross-phase contamination
        size_t full_backtrack_p_size = short_batch_size * (2 * short_task_max_len) * 752;  // Always 15GB
        size_t full_backtrack_off_size = short_batch_size * (2 * short_task_max_len) * sizeof(int);
        /*cudaMemset(d_backtrack_p, 0, full_backtrack_p_size);
        cudaMemset(d_backtrack_off, 0, full_backtrack_off_size);
        cudaMemset(d_backtrack_off_end, 0, full_backtrack_off_size);
        cudaMemset(d_backtrack_n_col, 0, short_batch_size * sizeof(int));
        cudaMemset(d_cigar_buffer, 0, short_batch_size * (2 * short_task_max_len) * sizeof(uint32_t));
        cudaMemset(d_cigar_lengths, 0, short_batch_size * sizeof(int));
        cudaMemset(d_ez_array, 0, short_batch_size * sizeof(ksw_extz_t));*.

        // These are shared across all phases and must be completely cleared
        size_t max_align_tasks = 120000;  // From plmem.cu
        /*cudaMemset(d_scores, 0, max_align_tasks * sizeof(int32_t));
        cudaMemset(d_query_ends, 0, max_align_tasks * sizeof(int32_t));
        cudaMemset(d_target_ends, 0, max_align_tasks * sizeof(int32_t));
        cudaMemset(d_mqe, 0, max_align_tasks * sizeof(int32_t));
        cudaMemset(d_mqe_t, 0, max_align_tasks * sizeof(int32_t));
        cudaMemset(d_mte, 0, max_align_tasks * sizeof(int32_t));
        cudaMemset(d_mte_q, 0, max_align_tasks * sizeof(int32_t));*/

        int tasks_processed_in_phase = 0;
        while (tasks_processed_in_phase < n_tasks_in_phase) {
            int batch_start = tasks_processed_in_phase;
            int batch_size = (tasks_processed_in_phase + current_batch_size <= n_tasks_in_phase) ?
                             current_batch_size : (n_tasks_in_phase - tasks_processed_in_phase);
            batch_num++;

            // Clear ONLY result buffers before each batch to prevent reading stale data
            // If a task fails (zdropped etc), kernel may not write to result buffers
            // Without clearing, we'd read previous batch's stale results
            // Note: backtrack buffers don't need clearing as they're fully written by kernel
            /*cudaMemset(d_scores, 0, batch_size * sizeof(int32_t));
            cudaMemset(d_query_ends, 0, batch_size * sizeof(int32_t));
            cudaMemset(d_target_ends, 0, batch_size * sizeof(int32_t));
            cudaMemset(d_mqe, 0, batch_size * sizeof(int32_t));
            cudaMemset(d_mqe_t, 0, batch_size * sizeof(int32_t));
            cudaMemset(d_mte, 0, batch_size * sizeof(int32_t));
            cudaMemset(d_mte_q, 0, batch_size * sizeof(int32_t));
            cudaMemset(d_cigar_lengths, 0, batch_size * sizeof(int));*/

            // Calculate memory requirements for this batch
            size_t total_query_bytes = 0, total_target_bytes = 0;
            uint32_t max_query_len = 0;

            // Calculate offsets and prepare sequences for this batch
            for (int i = 0; i < batch_size; i++) {
                int task_idx = current_task_indices[batch_start + i];

                // Validate sequence lengths against buffer limits
                if (tasks[task_idx].qlen > max_query_len_limit) {
                    fprintf(stderr, "[WARNING] Task %d: qlen=%d exceeds max_query_len=%zu, clamping\n",
                            task_idx, tasks[task_idx].qlen, max_query_len_limit);
                    tasks[task_idx].qlen = max_query_len_limit;
                }
                if (tasks[task_idx].tlen > max_query_len_limit) {
                    fprintf(stderr, "[WARNING] Task %d: tlen=%d exceeds max_query_len=%zu, clamping\n",
                            task_idx, tasks[task_idx].tlen, max_query_len_limit);
                    tasks[task_idx].tlen = max_query_len_limit;
                }

                // Align to 8-byte boundary for AGATHA
                size_t qlen_aligned = ((tasks[task_idx].qlen + 7) / 8) * 8;
                size_t tlen_aligned = ((tasks[task_idx].tlen + 7) / 8) * 8;

                h_query_offsets[i] = total_query_bytes;
                h_target_offsets[i] = total_target_bytes;
                h_query_lens[i] = tasks[task_idx].qlen;
                h_target_lens[i] = tasks[task_idx].tlen;
                h_flag[i] = tasks[task_idx].flag;

                total_query_bytes += qlen_aligned;
                total_target_bytes += tlen_aligned;

                if (tasks[task_idx].qlen > max_query_len) {
                    max_query_len = tasks[task_idx].qlen;
                }

                // Store task-to-alignment mapping
                h_task_to_align_id[i] = i;
            }

            // Prepare unpacked sequences for this batch
            uint8_t *h_unpacked_query = (uint8_t*)calloc(total_query_bytes, 1);
            uint8_t *h_unpacked_target = (uint8_t*)calloc(total_target_bytes, 1);

            const uint8_t N_BASE = 4;
            for (int i = 0; i < batch_size; i++) {
                int task_idx = current_task_indices[batch_start + i];
                // Copy sequences
                memcpy(h_unpacked_query + h_query_offsets[i],
                       seq_buffer + tasks[task_idx].qseq_offset, tasks[task_idx].qlen);
                memcpy(h_unpacked_target + h_target_offsets[i],
                       seq_buffer + tasks[task_idx].tseq_offset, tasks[task_idx].tlen);

                // Pad with N (0x0F)
                size_t qlen_aligned = ((tasks[task_idx].qlen + 7) / 8) * 8;
                size_t tlen_aligned = ((tasks[task_idx].tlen + 7) / 8) * 8;
                for (int j = tasks[task_idx].qlen; j < qlen_aligned; j++) {
                    h_unpacked_query[h_query_offsets[i] + j] = N_BASE;
                }
                for (int j = tasks[task_idx].tlen; j < tlen_aligned; j++) {
                    h_unpacked_target[h_target_offsets[i] + j] = N_BASE;
                }
            }

            // Copy batch data to GPU
            cudaMemcpy(d_unpacked_query, h_unpacked_query,
                       total_query_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_unpacked_target, h_unpacked_target,
                       total_target_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_query_offsets, h_query_offsets,
                       batch_size * sizeof(uint32_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_target_offsets, h_target_offsets,
                       batch_size * sizeof(uint32_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_query_lens, h_query_lens,
                       batch_size * sizeof(uint32_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_target_lens, h_target_lens,
                       batch_size * sizeof(uint32_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_flag, h_flag,
                       batch_size * sizeof(int32_t), cudaMemcpyHostToDevice);


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

            // ===== Fused Persistent KSW Kernel (align + backtrack in one launch) =====
            // Clamp concurrent blocks per phase: backtrack_p is allocated as
            //   short_task_batch_size × max_align_backtrack_size (short per-slot size).
            // For long tasks current_max_backtrack_size >> short per-slot size, so far
            // fewer slots fit. Cap to avoid out-of-bounds access.
            int parallel_threads = 32;   // one warp per block
            size_t parallel_smem = 3072; // 3072 bytes smem → 32 blocks/SM on V100

            size_t bt_p_total_bytes = (size_t)dev_mem->short_task_batch_size *
                                      dev_mem->max_align_backtrack_size;
            size_t max_slots_this_phase = bt_p_total_bytes /
                                          (size_t)current_max_backtrack_size;
            int phase_concurrent_blocks = n_concurrent_blocks;
            if ((size_t)phase_concurrent_blocks > max_slots_this_phase)
                phase_concurrent_blocks = (int)max_slots_this_phase;
            if (phase_concurrent_blocks > batch_size)
                phase_concurrent_blocks = batch_size;
            if (phase_concurrent_blocks < 1) phase_concurrent_blocks = 1;

            // Also clamp temp_buffer: allocated for short_task_batch_size slots
            // (safe for both phases since 7000 >> 2560 >> 128)

            // Reset atomic task counter to 0 before this batch
            int zero = 0;
            cudaMemcpy(d_task_counter, &zero, sizeof(int), cudaMemcpyHostToDevice);

            ksw_fused_persistent_kernel<<<phase_concurrent_blocks, parallel_threads,
                                          parallel_smem>>>(
                d_task_counter,
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
                (int)current_max_backtrack_size,
                (int)current_max_antidiag,
                d_ksw_temp_buffer,
                d_flag,
                ksw_temp_per_task,
                batch_size,
                5,              // m = alphabet size (ACGTN)
                opt->zdrop,
                opt->end_bonus,
                cigar_buffer ? d_cigar_buffer  : NULL,
                cigar_buffer ? d_cigar_lengths : NULL,
                (int)current_max_cigar_len
            );

            cudaError_t kernel_err = cudaGetLastError();
            if (kernel_err != cudaSuccess) {
                fprintf(stderr, "[ERROR] KSW fused persistent kernel launch failed: %s\n",
                        cudaGetErrorString(kernel_err));
            }
            cudaDeviceSynchronize();
            kernel_err = cudaGetLastError();
            if (kernel_err != cudaSuccess) {
                fprintf(stderr, "[ERROR] KSW fused persistent kernel execution failed: %s\n",
                        cudaGetErrorString(kernel_err));
            }

            if (cigar_buffer) {
                cudaMemcpy(h_cigar_buffer,
                           d_cigar_buffer,
                           batch_size * current_max_cigar_len * sizeof(uint32_t),
                           cudaMemcpyDeviceToHost);
                cudaMemcpy(h_cigar_lengths,
                           d_cigar_lengths,
                           batch_size * sizeof(int),
                           cudaMemcpyDeviceToHost);
            }

            // Copy results back
            cudaMemcpy(h_scores,
                       d_scores,
                       batch_size * sizeof(int32_t),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(h_query_ends,
                       d_query_ends,
                       batch_size * sizeof(int32_t),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(h_target_ends,
                       d_target_ends,
                       batch_size * sizeof(int32_t),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(h_mqe,
                       d_mqe,
                       batch_size * sizeof(int32_t),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(h_mqe_t,
                       d_mqe_t,
                       batch_size * sizeof(int32_t),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(h_mte,
                       d_mte,
                       batch_size * sizeof(int32_t),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(h_mte_q,
                       d_mte_q,
                       batch_size * sizeof(int32_t),
                       cudaMemcpyDeviceToHost);

            // Wait for completion
            cudaDeviceSynchronize();
            cudaCheck();

            // Map results back to tasks
            for (int i = 0; i < batch_size; i++) {
                int task_idx = current_task_indices[batch_start + i];  // Use task index from current phase
                int align_id = h_task_to_align_id[i];

                tasks[task_idx].score = h_scores[align_id];
                // In approx_max mode (KSW_EZ_APPROX_MAX), max_q/max_t are not tracked and should be -1
                // h_query_ends/h_target_ends contain backtrack endpoints, not max score positions
                if (tasks[task_idx].flag & KSW_EZ_APPROX_MAX) {
                    tasks[task_idx].max_q = -1;
                    tasks[task_idx].max_t = -1;
                } else {
                    tasks[task_idx].max_q = h_query_ends[align_id];
                    tasks[task_idx].max_t = h_target_ends[align_id];
                }
                tasks[task_idx].mqe = h_mqe[align_id];
                tasks[task_idx].mqe_t = h_mqe_t[align_id];
                tasks[task_idx].mte = h_mte[align_id];
                tasks[task_idx].mte_q = h_mte_q[align_id];

                // Copy CIGAR to output buffer
                if (cigar_buffer) {
                    int n_cigar = h_cigar_lengths[align_id];
                    tasks[task_idx].n_cigar = n_cigar;

                    // Copy CIGAR operations (check n_cigar doesn't exceed allocated capacity)
                    if (n_cigar > 0 && n_cigar <= tasks[task_idx].max_cigar) {
                        memcpy(cigar_buffer + tasks[task_idx].cigar_offset,
                               h_cigar_buffer + align_id * current_max_cigar_len,
                               n_cigar * sizeof(uint32_t));
                    } else if (n_cigar > tasks[task_idx].max_cigar) {
                        tasks[task_idx].n_cigar = 0;  // Reset to avoid corruption
                    }
                } else {
                    tasks[task_idx].n_cigar = 0;
                }

                // Set completion flags
                // In approx_max mode, use backtrack endpoints (h_query_ends/h_target_ends) to check reach_end
                // Otherwise use max_q/max_t (which are 0-based indices, compare with len-1)
                if (tasks[task_idx].flag & KSW_EZ_APPROX_MAX) {
                    tasks[task_idx].reach_end = (h_query_ends[align_id] == tasks[task_idx].qlen - 1) &&
                                                (h_target_ends[align_id] == tasks[task_idx].tlen - 1);
                } else {
                    tasks[task_idx].reach_end = (tasks[task_idx].max_q == tasks[task_idx].qlen - 1) &&
                                                (tasks[task_idx].max_t == tasks[task_idx].tlen - 1);
                }
                tasks[task_idx].zdropped = 0;
            }

            // Cleanup batch buffers
            free(h_unpacked_query);
            free(h_unpacked_target);

            // Update progress
            tasks_processed_in_phase += batch_size;
            total_tasks_processed += batch_size;

            // Progress bar (show phase-specific progress)
            int percent = (tasks_processed_in_phase * 100) / n_tasks_in_phase;
            int bar_width = 30;
            int filled = (percent * bar_width) / 100;
            fprintf(stderr, "\r[%s] [", phase_name);
            for (int i = 0; i < bar_width; i++) {
                if (i < filled) fprintf(stderr, "=");
                else if (i == filled) fprintf(stderr, ">");
                else fprintf(stderr, " ");
            }
            fprintf(stderr, "] %3d%% (%d/%d in phase, %d/%d total, batch %d/%d)",
                    percent, tasks_processed_in_phase, n_tasks_in_phase,
                    total_tasks_processed, n_tasks, batch_num, total_batches);
            fflush(stderr);
        }  // End of batch loop within phase

        // Phase completion message
        fprintf(stderr, "\n[Info::%s] === %s completed: %d tasks processed ===\n",
                __func__, phase_name, n_tasks_in_phase);
    }  // End of two-phase loop

    // Final progress update
    fprintf(stderr, "[Align Progress] [");
    for (int i = 0; i < 30; i++) fprintf(stderr, "=");
    fprintf(stderr, "] 100%% (%d/%d tasks, %d batches) - DONE\n",
            n_tasks, n_tasks, batch_num);

    // Cleanup phase-specific arrays
    free(task_indices_short);
    free(task_indices_long);

    // Cleanup host buffers (allocated once, used across all batches)
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
    free(h_mqe);
    free(h_mqe_t);
    free(h_mte);
    free(h_mte_q);
}