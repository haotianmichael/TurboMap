#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>


#include "mmpriv.h"
#include "plmem.cuh"
#include "plrange.cuh"
#include "plscore.cuh"
#include "plbacktrack.cuh"
#include "plchain.h"
#include <utility>
#include <algorithm>

#define CUDA_DEVICE 0
// utils functions
struct
{
    bool operator()(std::pair<size_t,unsigned> a, std::pair<size_t,unsigned> b) const {
        return a.first > b.first;
    }
}
comp;

void pairsort(seg_t *sdata, unsigned *map, unsigned N){
    std::pair<size_t,unsigned> elements[N];
    for (unsigned i = 0; i < N; ++i){
      elements[i].second = map[i];
      elements[i].first = sdata[i].end_idx - sdata[i].start_idx;
    }

    std::sort(elements, elements+N, comp); // descent order

    for (unsigned i = 0; i < N; ++i){
      map[i] = elements[i].second;
    }
}


/**
 * translate relative predecessor index to abs index 
 * Input
 *  rel[]   relative predecessor index
 * Output
 *  p[]     absolute predecessor index (of each read)
 */
void p_rel2idx(const uint16_t* rel, int64_t* p, size_t n) {
    for (int i = 0; i < n; ++i) {
        if (rel[i] == 0)
            p[i] = -1;
        else
            p[i] = i - rel[i];
    }
}

//////////////////////////////////////////////////////////////////////////
///////////         Backtracking    //////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

/**
 * @brief start from end index of the chain, find the location of min score on
 * the chain until anchor has no predecessor OR anchor is in another chain
 *
 * @param max_drop
 * @param z [in] {sc, anchor idx}, sorted by sc
 * @param f [in] score
 * @param p [in] predecessor
 * @param k [in] chain end index
 * @param t [update] 0 for unchained anchor, 1 for chained anchor
 * @return min_i minmute score location in the chain
 */

static int64_t mg_chain_bk_end(int32_t max_drop, const mm128_t *z,
                               const int32_t *f, const int64_t *p, int32_t *t,
                               int64_t k) {
    int64_t i = z[k].y, end_i = -1, max_i = i;
    int32_t max_s = 0;
    if (i < 0 || t[i] != 0) return i;
    do {
        int32_t s;
        t[i] = 2;
        end_i = i = p[i];
        s = i < 0 ? z[k].x : (int32_t)z[k].x - f[i];
        if (s > max_s)
            max_s = s, max_i = i;
        else if (max_s - s > max_drop)
            break;
    } while (i >= 0 && t[i] == 0);
    for (i = z[k].y; i >= 0 && i != end_i; i = p[i])  // reset modified t[]
        t[i] = 0;
    return max_i;
}

void plchain_backtracking(hostMemPtr *host_mem, deviceMemPtr *dev_mem, chain_read_t *reads, Misc misc, void* km){
    int max_drop = misc.bw;
    if (misc.max_dist_x < misc.bw) misc.max_dist_x = misc.bw;
    if (misc.max_dist_y < misc.bw && !misc.is_cdna) misc.max_dist_y = misc.bw;
    if (misc.is_cdna) max_drop = INT32_MAX;

    size_t n_read = host_mem->size;

    uint16_t* p_hostmem = host_mem->p;
    int32_t* f = host_mem->f;
    for (int i = 0; i < n_read; i++) {
        int64_t* p;
        KMALLOC(km, p, reads[i].n);
        p_rel2idx(p_hostmem, p, reads[i].n);
        /* Backtracking*/
        uint64_t* u;
        int32_t *v, *t;
        KMALLOC(km, v, reads[i].n);
        KCALLOC(km, t, reads[i].n);
        int32_t n_u, n_v;
        u = mg_chain_backtrack(km, reads[i].n, f, p, v, t, misc.min_cnt, misc.min_score, max_drop, &n_u, &n_v);
        reads[i].u = u;
        reads[i].n_u = n_u;
        kfree(km, p);
        // here f is not managed by km memory pool
        kfree(km, t);
        if (n_u == 0) {
            kfree(km, reads[i].a);
            kfree(km, v);
            reads[i].a = 0;

            f += reads[i].n;
            p_hostmem += reads[i].n;
            continue;
        }

        mm128_t* new_a = compact_a(km, n_u, u, n_v, v, reads[i].a);
        reads[i].a = new_a;

        f += reads[i].n;
        p_hostmem += reads[i].n;
    }

}


//////////////////////////////////////////////////////////////////////////
///////////         Stream Management    /////////////////////////////////
//////////////////////////////////////////////////////////////////////////

/**
 * Wait and find a free stream
 *  stream_setup:    [in]
 *  batchid:         [in]
 *  stream_id:      [out] stream_id to schedule to
 * RETURN
 *  true if need to cleanup current stream
*/
int plchain_schedule_stream(const streamSetup_t stream_setup, const int batchid){
    /* Haven't fill all the streams*/
    if (batchid < stream_setup.num_stream) {
        return batchid;
    }
    
    // wait until one stream is free
    int streamid = -1;
    while(streamid == -1){
        for (int t = 0; t < stream_setup.num_stream; t++){
            if (!cudaEventQuery(stream_setup.streams[t].stopevent)) {
                streamid = t;
                // FIXME: unnecessary recreate?
                cudaEventDestroy(stream_setup.streams[t].stopevent);
                cudaEventCreate(&stream_setup.streams[t].stopevent);
                cudaCheck();
                break;
            }
            // cudaCheck();
        }
    }
    return streamid;
}


/*
 * ════════════════════════════════════════════════════════════════════════
 *  Single-stream pipeline: ALL ops (chain → backtrack → align) run on
 *  one cudastream (stream 0).  Main thread accumulates next batch while
 *  drain worker processes current batch on GPU.
 *
 *    stream[0]: [chain_0][bt_0][align_0][chain_1][bt_1][align_1] ...
 *
 *  Per cycle in map.c gpu_batch_consumer:
 *    1. Accumulate reads from seeded queue
 *    2. Wait for previous drain to finish (if busy)
 *    3. Launch chain async, signal drain worker
 *    4. Drain worker: sync chain → backtrack → align
 * ════════════════════════════════════════════════════════════════════════
 */

/**
 * launch_chain_gpu: Launch forward DP chain async on cudastream.
 *   Writes results to host_mems[0..micro_batch-1].
 *   Returns number of reads that could NOT fit into micro-batches (overflow).
 */
static int launch_chain_impl(chain_read_t *reads, int n_read,
                              stream_ptr_t *sp) {
    deviceMemPtr *dev_mem = &sp->dev_mem;
    cudaStream_t cudastream = sp->cudastream;

    cudaEventRecord(sp->startevent, cudastream);

    // Reset long seg counters
    cudaMemsetAsync(dev_mem->d_long_seg_count, 0, sizeof(unsigned int), cudastream);
    cudaMemsetAsync(dev_mem->d_total_n_long, 0, sizeof(size_t), cudastream);
    cudaCheck();
    sp->long_mem.total_long_segs_num[0] = 0;
    sp->long_mem.total_long_segs_n[0] = 0;
    for (int uid = 0; uid < score_kernel_config.micro_batch; uid++) {
        sp->host_mems[uid].long_segs_num[0] = 0;
        sp->host_mems[uid].index = uid;
        sp->host_mems[uid].griddim = 0;
        sp->host_mems[uid].size = 0;
        sp->host_mems[uid].total_n = 0;
        sp->host_mems[uid].cut_num = 0;
    }

    sp->reads = reads;
    sp->n_read = n_read;
    int read_start = 0;

    for (int uid = 0; uid < score_kernel_config.micro_batch; uid++) {
        if (read_start == n_read) continue;
        size_t batch_n = 0;
        int read_end = 0;
        size_t cut_num = 0;
        int griddim = 0;
        for (read_end = read_start; read_end < n_read; read_end++) {
            /* Debug: log input anchor count for target reads */
            {
                static const char *_dbgn[] = {
                    "08c178a9-9054-40c4-87fa-0c636d52df41",
                    "299e4b51-a23d-444d-a888-f08804a03cf4",
                    "07d825c2-74a5-45e1-b8b6-e4e60ee7c6f1",
                    "3262e09f-9576-411f-b49c-4343f2822652", NULL };
                for (int _di = 0; _dbgn[_di]; _di++)
                    if (strcmp(reads[read_end].seq.name, _dbgn[_di]) == 0) {
                        FILE *_f = fopen("/tmp/chain_debug.txt","a");
                        if (!_f) _f = stderr;
                        fprintf(_f, "[GPU_INPUT] %s  n_anchors=%ld  uid=%d  batch_n_so_far=%zu  max=%zu\n",
                                reads[read_end].seq.name, (long)reads[read_end].n,
                                uid, batch_n, stream_setup.max_anchors_stream);
                        fflush(_f); if (_f != stderr) fclose(_f);
                        break;
                    }
            }
            if (batch_n + reads[read_end].n > stream_setup.max_anchors_stream) break;
            batch_n += reads[read_end].n;
            int an_p_block = range_kernel_config.anchor_per_block;
            int an_p_cut = range_kernel_config.blockdim;
            griddim += (reads[read_end].n - 1) / an_p_block + 1;
            cut_num += (reads[read_end].n - 1) / an_p_cut + 1;
        }

        if (read_end == read_start) {
            // No reads fit in this micro-batch (single read exceeds buffer)
            // Skip kernel launches, will be reported as overflow below
            break;
        }

        assert(stream_setup.max_anchors_stream >= batch_n);
        assert(stream_setup.max_range_grid >= griddim);
        assert(stream_setup.max_num_cut >= cut_num);

        plmem_reorg_input_arr(reads + read_start, read_end - read_start,
                              &sp->host_mems[uid], range_kernel_config);
        plmem_async_h2d_short_memcpy(sp, uid);
        plrange_async_range_selection(&sp->dev_mem, &sp->cudastream);
        plscore_async_short_mid_forward_dp(&sp->dev_mem, &sp->cudastream);
        plmem_async_d2h_short_memcpy(sp, uid);
        read_start = read_end;
    }

    int overflow = n_read - read_start;
    if (overflow > 0) {
        fprintf(stderr, "[WARNING] Unable to fit reads %d - %d into a microbatch. Fall back to cpu chaining\n",
                read_start, n_read - 1);
    }
    sp->busy = true;
    return overflow;
}

/**
 * sync_chain_gpu: Synchronize cudastream, then process long segments.
 *   Must be called before start_backtrack_gpu for the same batch.
 *   After return, host_mems[] has final f[]/p[] with long-seg merged.
 */
static void sync_chain_impl(stream_ptr_t *sp) {
    deviceMemPtr *dev_mem = &sp->dev_mem;
    cudaStream_t cudastream = sp->cudastream;

    // Sync short+mid micro-batches
    cudaStreamSynchronize(cudastream);

    // Long-seg sort on CPU
    unsigned int num_long_seg;
    cudaMemcpyAsync(&num_long_seg, dev_mem->d_long_seg_count, sizeof(unsigned int),
                    cudaMemcpyDeviceToHost, cudastream);
    cudaStreamSynchronize(cudastream);

    seg_t *long_segs_og = (seg_t *)malloc(sizeof(seg_t) * num_long_seg);
    cudaMemcpyAsync(long_segs_og, dev_mem->d_long_seg_og, sizeof(seg_t) * num_long_seg,
                    cudaMemcpyDeviceToHost, cudastream);
    cudaStreamSynchronize(cudastream);

    unsigned *map = (unsigned *)malloc(sizeof(unsigned) * num_long_seg);
    for (unsigned i = 0; i < num_long_seg; i++)
        map[i] = i;
    pairsort(long_segs_og, map, num_long_seg);
    free(long_segs_og);

    // d_map is pre-allocated in plmem.cu with capacity = max_long_segs
    if (num_long_seg > 0) {
        cudaMemcpyAsync(dev_mem->d_map, map, sizeof(unsigned) * num_long_seg,
                        cudaMemcpyHostToDevice, cudastream);
    }
    free(map);

    // Launch long-seg kernel + D2H
    plscore_async_long_forward_dp(&sp->dev_mem, &sp->cudastream);
    plmem_async_d2h_long_memcpy(sp);
    cudaStreamSynchronize(cudastream);

    // Merge long segment results into host f[]/p[].
    // Use long_segs_buf_idx[k].start_idx as the actual offset into f_long/p_long
    // because the two atomicAdds in score_generation_short (total_n_long and
    // long_seg_count) race between concurrent blocks, so segment index k may not
    // correspond to a contiguous sequential range in f_long.
    seg_t *long_segs_og_h  = sp->long_mem.long_segs_og_idx;
    seg_t *long_segs_buf_h = sp->long_mem.long_segs_buf_idx;
    size_t long_seg_idx = 0;
    for (int uid = 0; uid < score_kernel_config.micro_batch; uid++) {
        if (sp->host_mems[uid].size == 0) continue;
        unsigned int long_segs_num = sp->host_mems[uid].long_segs_num[0];
        for (; long_seg_idx < long_segs_num; long_seg_idx++) {
            size_t orig_start = long_segs_og_h[long_seg_idx].start_idx;
            size_t orig_end   = long_segs_og_h[long_seg_idx].end_idx;
            size_t buf_start  = long_segs_buf_h[long_seg_idx].start_idx;
            for (size_t j = 0; j < orig_end - orig_start; j++) {
                sp->host_mems[uid].f[orig_start + j] = sp->long_mem.f_long[buf_start + j];
                sp->host_mems[uid].p[orig_start + j] = sp->long_mem.p_long[buf_start + j];
            }
        }
    }
}

/**
 * start_backtrack_gpu: Queue H2D from host_mems[] to d_bt_*_in,
 *   then launch backtrack kernels — all on cudastream (unified stream).
 *   Returns (combined_n_reads, combined_total_n) through out-params.
 */
static void start_backtrack_impl(stream_ptr_t *sp,
                                  chain_read_t *reads,
                                  Misc misc, void *km,
                                  int *out_n_reads, size_t *out_total_n) {
    deviceMemPtr *dev_mem = &sp->dev_mem;
    cudaStream_t bt_stream = sp->cudastream;  // unified: all ops on one stream
    size_t combined_total_n = 0;
    int combined_n_reads = 0;

    // H2D: host_mems → d_bt_*_in on cudastream
    for (int uid = 0; uid < score_kernel_config.micro_batch; uid++) {
        hostMemPtr *hm = &sp->host_mems[uid];
        if (hm->size == 0) continue;
        size_t n = hm->total_n;
        cudaMemcpyAsync(dev_mem->d_bt_ax_in   + combined_total_n, hm->ax,
                        sizeof(int32_t) * n, cudaMemcpyHostToDevice, bt_stream);
        cudaMemcpyAsync(dev_mem->d_bt_ay_in   + combined_total_n, hm->ay,
                        sizeof(int32_t) * n, cudaMemcpyHostToDevice, bt_stream);
        cudaMemcpyAsync(dev_mem->d_bt_xrev_in + combined_total_n, hm->xrev,
                        sizeof(int32_t) * n, cudaMemcpyHostToDevice, bt_stream);
        cudaMemcpyAsync(dev_mem->d_bt_yrev_in + combined_total_n, hm->yrev,
                        sizeof(int32_t) * n, cudaMemcpyHostToDevice, bt_stream);
        cudaMemcpyAsync(dev_mem->d_bt_f_in    + combined_total_n, hm->f,
                        sizeof(int32_t) * n, cudaMemcpyHostToDevice, bt_stream);
        cudaMemcpyAsync(dev_mem->d_bt_p_in    + combined_total_n, hm->p,
                        sizeof(uint16_t) * n, cudaMemcpyHostToDevice, bt_stream);
        combined_total_n += n;
        combined_n_reads += hm->size;
    }

    // Backtrack kernels on bt_stream (H2D completes first by stream ordering)
    plbacktrack_gpu(combined_n_reads, combined_total_n, dev_mem,
                    reads, misc, km, bt_stream);

    *out_n_reads = combined_n_reads;
    *out_total_n = combined_total_n;
}

/**
 * finish_backtrack_impl: D2H anchor data + post-chaining for all reads.
 *
 * Previously this split reads into "needs RMQ rescue" (GPU voting path)
 * vs "non-rechain" (D2H path). The voting heuristic diverged from the
 * CPU chain algorithm and was a major source of alignment mismatches,
 * so it has been removed. Every read now takes a uniform D2H path, and
 * the long-read rescue is performed on the host by post_chaining_helper
 * via a second mg_lchain_dp call with bw_long (see map.c).
 */
static void finish_backtrack_impl(const mm_idx_t *mi, const mm_mapopt_t *opt,
                                   stream_ptr_t *sp,
                                   chain_read_t *reads, int n_read,
                                   Misc misc, void *km) {
    deviceMemPtr *dev_mem = &sp->dev_mem;
    cudaStream_t stream   = sp->cudastream;

    /* Bulk D2H: copy all 4 compacted anchor arrays in one shot (4 transfers + 1 sync).
     * Replaces n_read separate per-read D2H calls, each with its own non-pinned malloc
     * and cudaStreamSynchronize (O(n_read) GPU stalls → O(1)).
     *
     * Buffers are allocated per-batch with cudaMallocHost (pinned → truly async D2H).
     * Pre-allocating at d_bt_max_total_n would lock ~600 MB per stream (38M anchors ×
     * 4 arrays × 4 bytes) — grossly wasteful.  Per-batch cost: 8 mlock/unlock total. */
    {
        int *h_offset = (int*)dev_mem->bt_h_offset;
        int *h_n_u    = (int*)dev_mem->bt_h_n_u;

        // Compute actual compacted span (last valid read's end).
        size_t compacted_total = 0;
        for (int i = 0; i < n_read; i++) {
            if (h_n_u[i] > 0) {
                size_t end = (size_t)(h_offset[i] + reads[i].n);
                if (end > compacted_total) compacted_total = end;
            }
        }

        if (compacted_total > 0) {
            int32_t *h_ax, *h_ay, *h_xrev, *h_yrev;
            cudaMallocHost(&h_ax,   compacted_total * sizeof(int32_t));
            cudaMallocHost(&h_ay,   compacted_total * sizeof(int32_t));
            cudaMallocHost(&h_xrev, compacted_total * sizeof(int32_t));
            cudaMallocHost(&h_yrev, compacted_total * sizeof(int32_t));

            cudaMemcpyAsync(h_ax,   dev_mem->d_bt_ax_out,   compacted_total * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(h_ay,   dev_mem->d_bt_ay_out,   compacted_total * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(h_xrev, dev_mem->d_bt_xrev_out, compacted_total * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(h_yrev, dev_mem->d_bt_yrev_out, compacted_total * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);  // ONE sync for all reads (was n_read syncs before)

            // CPU-only: unpack per-read anchor data from the bulk host buffer.
            for (int i = 0; i < n_read; i++) {
                if (h_n_u[i] <= 0) continue;
                int ofs   = h_offset[i];
                int new_n = reads[i].n;
                mm128_t *new_a;
                KMALLOC(km, new_a, new_n);
                for (int j = 0; j < new_n; j++) {
                    new_a[j].x = ((uint64_t)h_xrev[ofs + j] << 32) | (uint32_t)h_ax[ofs + j];
                    new_a[j].y = ((uint64_t)h_yrev[ofs + j] << 32) | (uint32_t)h_ay[ofs + j];
                }
                kfree(km, reads[i].a);
                reads[i].a = new_a;
            }

            cudaFreeHost(h_ax);
            cudaFreeHost(h_ay);
            cudaFreeHost(h_xrev);
            cudaFreeHost(h_yrev);
        }
    }
    plbacktrack_d2h_finish(dev_mem);

    /* Post-chaining on host (long-read rescue via mg_lchain_dp(bw_long)
     * happens inside post_chaining_helper). */
    for (int i = 0; i < n_read; i++)
        post_chaining_helper(mi, opt, &reads[i], misc, km);

    sp->busy = false;
}


#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

deviceMemPtr* gpu_get_dev_mem(int stream_id) {
    return &stream_setup.streams[stream_id].dev_mem;
}

cudaStream_t gpu_get_cudastream(int stream_id) {
    return stream_setup.streams[stream_id].cudastream;
}

void init_stream_gpu(size_t *total_n, int *max_reads, int *min_n, char gpu_config_file[], Misc misc) {
    plmem_stream_initialize(total_n, max_reads, min_n, gpu_config_file);
    plrange_upload_misc(misc);
    plscore_upload_misc(misc);
}

/**
 * launch_chain_gpu: Launch forward DP chain for a batch [async on cudastream].
 *   Returns overflow count (reads that didn't fit).
 */
int launch_chain_gpu(chain_read_t *reads, int n_read, int stream_id) {
    cudaSetDevice(CUDA_DEVICE);
    return launch_chain_impl(reads, n_read,
                             &stream_setup.streams[stream_id]);
}

/**
 * sync_chain_gpu: Wait for chain to finish + process long segments.
 */
void sync_chain_gpu(int stream_id) {
    cudaSetDevice(CUDA_DEVICE);
    sync_chain_impl(&stream_setup.streams[stream_id]);
}

/**
 * start_backtrack_gpu: H2D from host_mems + backtrack kernels on cudastream.
 *   NOTE: plbacktrack_gpu has internal cudaStreamSynchronize calls, so this is
 *   effectively BLOCKING — stream is fully synced when this returns.
 *   Returns n_reads actually processed through out-param.
 */
void start_backtrack_gpu(const mm_idx_t *mi, const mm_mapopt_t *opt,
                         chain_read_t *reads, int stream_id,
                         void *km, int *out_n_reads) {
    cudaSetDevice(CUDA_DEVICE);
    assert(opt->max_frag_len <= 0);
    Misc misc = build_misc(mi, opt, 0, 1);
    size_t total_n;
    start_backtrack_impl(&stream_setup.streams[stream_id],
                         reads, misc, km, out_n_reads, &total_n);
}

/**
 * finish_backtrack_gpu: Voting + post_chaining (CPU work).
 *   After return, reads[] have u/n_u/rep_len/frag_gap set.
 */
void finish_backtrack_gpu(const mm_idx_t *mi, const mm_mapopt_t *opt,
                          chain_read_t *reads, int n_read,
                          int stream_id, void *km) {
    cudaSetDevice(CUDA_DEVICE);
    assert(opt->max_frag_len <= 0);
    Misc misc = build_misc(mi, opt, 0, 1);
    finish_backtrack_impl(mi, opt, &stream_setup.streams[stream_id],
                          reads, n_read, misc, km);
}

int gpu_get_num_streams(void) {
    return stream_setup.num_stream;
}

/**
 * chain_stream_gpu: Legacy wrapper — launches chain and returns previous batch.
 *   Single-buffered host_mems: backtrack must complete before chain launch
 *   to avoid host_mems conflict.
 *   Kept for fallback_batch processing in map.c.
 */
void chain_stream_gpu(const mm_idx_t *mi, const mm_mapopt_t *opt, chain_read_t **in_arr_, int *n_read_,
                      int thread_id, void *km) {
    assert(opt->max_frag_len <= 0);
    cudaSetDevice(CUDA_DEVICE);
    stream_ptr_t *sp = &stream_setup.streams[thread_id];
    Misc misc = build_misc(mi, opt, 0, 1);

    // Save new input before overwriting the pointers
    chain_read_t *new_reads = *in_arr_;
    int new_n_read = *n_read_;
    *in_arr_ = NULL;
    *n_read_ = 0;

    if (sp->busy) {
        // Sync + long-seg for previous chain
        sync_chain_impl(sp);
        chain_read_t *prev_reads = sp->reads;

        // Backtrack BEFORE launching new chain (single-buffered host_mems)
        int bt_n_reads;
        size_t bt_total_n;
        start_backtrack_impl(sp, prev_reads, misc, km,
                             &bt_n_reads, &bt_total_n);
        finish_backtrack_impl(mi, opt, sp, prev_reads, bt_n_reads, misc, km);
        *in_arr_ = prev_reads;
        *n_read_ = bt_n_reads;
    }

    // Now safe to launch new chain (host_mems free after backtrack H2D)
    launch_chain_impl(new_reads, new_n_read, sp);
}

/**
 * finish_stream_gpu: Drain the last chain batch — sync + backtrack + voting.
 */
void finish_stream_gpu(const mm_idx_t *mi, const mm_mapopt_t *opt, chain_read_t **reads_,
                       int *n_read_, int t, void *km) {
    assert(opt->max_frag_len <= 0);
    cudaSetDevice(CUDA_DEVICE);
    stream_ptr_t *sp = &stream_setup.streams[t];

    if (!sp->busy) {
        *reads_ = NULL;
        *n_read_ = 0;
        return;
    }

    sync_chain_impl(sp);

    int bt_n_reads;
    size_t bt_total_n;
    Misc misc = build_misc(mi, opt, 0, 1);
    chain_read_t *reads = sp->reads;
    start_backtrack_impl(sp, reads, misc, km, &bt_n_reads, &bt_total_n);
    finish_backtrack_impl(mi, opt, sp, reads, bt_n_reads, misc, km);

    *reads_ = reads;
    *n_read_ = bt_n_reads;
}

void free_stream_gpu(int n_threads) {
    plmem_stream_cleanup();
}

/* Long-read rescue is now performed on the host via a second
 * mg_lchain_dp(bw_long) call in post_chaining_helper (map.c). The
 * previous GPU voting-based path diverged from the CPU chain algorithm
 * and has been removed. */

#ifdef __cplusplus
} // extern "C"
#endif  // __cplusplus
