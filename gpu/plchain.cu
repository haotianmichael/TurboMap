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
#include "plvoting.cuh"
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
 * Accepts a stream that has already been synced, and finished processing a batch
 * Finish and cleanup the stream, save primary chain results to unpinned CPU memory.  
 * RETURN: number of reads in last batch 
*/
int plchain_post_gpu_helper(streamSetup_t stream_setup, int stream_id,
                            Misc misc, void* km)
{
    deviceMemPtr *dev_mem = &stream_setup.streams[stream_id].dev_mem;
    cudaStream_t bt_stream = dev_mem->backtrack_stream;
    seg_t* long_segs = stream_setup.streams[stream_id].long_mem.long_segs_og_idx;
    size_t long_seg_idx = 0;
    size_t long_i = 0;

    // Phase 1: Merge long segment results into host f[]/p[] (CPU)
    for (int uid = 0; uid < score_kernel_config.micro_batch; uid++) {
        if (stream_setup.streams[stream_id].host_mems[uid].size == 0) continue;
        unsigned int long_segs_num =
            stream_setup.streams[stream_id].host_mems[uid].long_segs_num[0];
        for (; long_seg_idx < long_segs_num; long_seg_idx++) {
            for (size_t i = long_segs[long_seg_idx].start_idx;
                 i < long_segs[long_seg_idx].end_idx; i++, long_i++) {
                stream_setup.streams[stream_id].host_mems[uid].f[i] =
                    stream_setup.streams[stream_id].long_mem.f_long[long_i];
                stream_setup.streams[stream_id].host_mems[uid].p[i] =
                    stream_setup.streams[stream_id].long_mem.p_long[long_i];
            }
        }
    }

    // Phase 2: Concatenate all micro-batches' data to d_bt_*_in at offsets
    // Uses backtrack_stream — does NOT touch d_ax (chain can use it concurrently)
    size_t combined_total_n = 0;
    int combined_n_reads = 0;
    for (int uid = 0; uid < score_kernel_config.micro_batch; uid++) {
        hostMemPtr *hm = &stream_setup.streams[stream_id].host_mems[uid];
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

    // Phase 3: Single merged backtrack on backtrack_stream
    // Stream ordering: H2D on bt_stream completes before kernels on bt_stream
    // chain(B_N) on cudastream runs concurrently — no memory conflict
    plbacktrack_gpu(combined_n_reads, combined_total_n, dev_mem,
                    stream_setup.streams[stream_id].reads, misc, km, bt_stream);

    return combined_n_reads;
}

/*
 * Two-stage pipelined chaining:
 *   1. Sync previous chain → queue H2D for backtrack → sync H2D
 *   2. Launch NEW chain micro-batches [async on cudastream]
 *   3. Run backtrack of PREVIOUS batch [on backtrack_stream]
 *      GPU overlap: chain(B_N) on cudastream || backtrack(B_{N-1}) on backtrack_stream
 *   4. Sync new chain for long-seg processing
 */
void plchain_cal_score_async(const mm_idx_t *mi, const mm_mapopt_t *opt, chain_read_t **reads_, int *n_read_, Misc misc, streamSetup_t stream_setup, int thread_id, void* km){
	cudaSetDevice(CUDA_DEVICE);
    chain_read_t* reads = *reads_;
    *reads_ = NULL;
    int n_read = *n_read_;
    *n_read_ = 0;

    int stream_id = thread_id;
    stream_ptr_t *sp = &stream_setup.streams[stream_id];
    deviceMemPtr *dev_mem = &sp->dev_mem;
    cudaStream_t cudastream = sp->cudastream;
    cudaStream_t bt_stream = dev_mem->backtrack_stream;

    // ── Pipeline Stage A: prepare previous batch's backtrack data ────────
    // We must copy B_{N-1}'s host data to d_bt_*_in BEFORE launching B_N
    // (which overwrites host_mems via plmem_reorg_input_arr).
    bool had_prev = sp->busy;
    size_t prev_combined_total_n = 0;
    int prev_combined_n_reads = 0;
    chain_read_t *prev_reads = NULL;

    if (had_prev) {
        // Sync cudastream: wait for B_{N-1} long-seg kernel + D2H
        cudaStreamSynchronize(cudastream);
        cudaCheck();

        prev_reads = sp->reads;

        // Merge long segment results into host f[]/p[] (CPU)
        {
            seg_t* long_segs = sp->long_mem.long_segs_og_idx;
            size_t long_seg_idx = 0, long_i = 0;
            for (int uid = 0; uid < score_kernel_config.micro_batch; uid++) {
                if (sp->host_mems[uid].size == 0) continue;
                unsigned int long_segs_num = sp->host_mems[uid].long_segs_num[0];
                for (; long_seg_idx < long_segs_num; long_seg_idx++) {
                    for (size_t i = long_segs[long_seg_idx].start_idx;
                         i < long_segs[long_seg_idx].end_idx; i++, long_i++) {
                        sp->host_mems[uid].f[i] = sp->long_mem.f_long[long_i];
                        sp->host_mems[uid].p[i] = sp->long_mem.p_long[long_i];
                    }
                }
            }
        }

        // Queue H2D: copy B_{N-1} data from host_mems → d_bt_*_in (on backtrack_stream)
        for (int uid = 0; uid < score_kernel_config.micro_batch; uid++) {
            hostMemPtr *hm = &sp->host_mems[uid];
            if (hm->size == 0) continue;
            size_t n = hm->total_n;
            cudaMemcpyAsync(dev_mem->d_bt_ax_in   + prev_combined_total_n, hm->ax,
                            sizeof(int32_t) * n, cudaMemcpyHostToDevice, bt_stream);
            cudaMemcpyAsync(dev_mem->d_bt_ay_in   + prev_combined_total_n, hm->ay,
                            sizeof(int32_t) * n, cudaMemcpyHostToDevice, bt_stream);
            cudaMemcpyAsync(dev_mem->d_bt_xrev_in + prev_combined_total_n, hm->xrev,
                            sizeof(int32_t) * n, cudaMemcpyHostToDevice, bt_stream);
            cudaMemcpyAsync(dev_mem->d_bt_yrev_in + prev_combined_total_n, hm->yrev,
                            sizeof(int32_t) * n, cudaMemcpyHostToDevice, bt_stream);
            cudaMemcpyAsync(dev_mem->d_bt_f_in    + prev_combined_total_n, hm->f,
                            sizeof(int32_t) * n, cudaMemcpyHostToDevice, bt_stream);
            cudaMemcpyAsync(dev_mem->d_bt_p_in    + prev_combined_total_n, hm->p,
                            sizeof(uint16_t) * n, cudaMemcpyHostToDevice, bt_stream);
            prev_combined_total_n += n;
            prev_combined_n_reads += hm->size;
        }
        // Sync backtrack_stream: ensure H2D done before reorg overwrites host_mems
        cudaStreamSynchronize(bt_stream);
    }

    // ── Pipeline Stage B: launch new chain B_N [async on cudastream] ─────
    cudaEventRecord(sp->startevent, cudastream);
    size_t total_n = 0;
    for (int i = 0; i < n_read; i++)
        total_n += reads[i].n;

    // reset long seg counters
    cudaMemsetAsync(dev_mem->d_long_seg_count, 0, sizeof(unsigned int), cudastream);
    cudaMemsetAsync(dev_mem->d_total_n_long, 0, sizeof(size_t), cudastream);
    cudaCheck();
    sp->long_mem.total_long_segs_num[0] = 0;
    sp->long_mem.total_long_segs_n[0] = 0;
    for(int uid = 0; uid < score_kernel_config.micro_batch; uid++) {
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
            if (batch_n + reads[read_end].n > stream_setup.max_anchors_stream) break;
            batch_n += reads[read_end].n;
            int an_p_block = range_kernel_config.anchor_per_block;
            int an_p_cut = range_kernel_config.blockdim;
            griddim += (reads[read_end].n - 1) / an_p_block + 1;
            cut_num += (reads[read_end].n - 1) / an_p_cut + 1;
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

    if (read_start < n_read) {
        fprintf(stderr, "[WARNING] Unable to fit reads %d - %d into a microbatch. Fall back to cpu chaining\n", read_start, n_read-1);
    }

    // ── Pipeline Stage C: backtrack B_{N-1} [on backtrack_stream] ────────
    // GPU overlap: chain(B_N) micro-batches on cudastream ||
    //              backtrack(B_{N-1}) kernels on backtrack_stream
    if (had_prev) {
        // Backtrack kernels read from d_bt_*_in (H2D already done in Stage A)
        plbacktrack_gpu(prev_combined_n_reads, prev_combined_total_n, dev_mem,
                        prev_reads, misc, km, bt_stream);

        *reads_ = prev_reads;
        *n_read_ = prev_combined_n_reads;
        sp->busy = false;

        // Voting + post-chaining for previous batch
        if (*reads_) {
            chain_read_t* out_arr = *reads_;
            int n_out = *n_read_;

            int *rechain_indices = (int*)malloc(sizeof(int) * n_out);
            int n_rechain = 0;
            for (int i = 0; i < n_out; i++) {
                if (needs_rmq_rechain(opt, &out_arr[i]))
                    rechain_indices[n_rechain++] = i;
            }
            if (n_rechain > 0)
                // TODO: plvoting uses null stream (cudaMalloc/cudaMemcpy inside).
                // This serializes with cudastream, briefly stalling chain(B_N) overlap.
                // Fix: pre-allocate voting buffers + use explicit backtrack_stream.
                plvoting_rechain_batch(mi, opt, out_arr, rechain_indices, n_rechain, misc, km);
            free(rechain_indices);

            for (int i = 0; i < n_out; i++)
                post_chaining_helper(mi, opt, &out_arr[i], misc, km);
        }
    }

    // ── Pipeline Stage D: long-seg processing for B_N ────────────────────
    cudaStreamSynchronize(cudastream);
    cudaCheck();
    unsigned int num_long_seg;
    cudaMemcpyAsync(&num_long_seg, dev_mem->d_long_seg_count, sizeof(unsigned int),
                    cudaMemcpyDeviceToHost, cudastream);
    cudaStreamSynchronize(cudastream);

    seg_t* long_segs_og = (seg_t*)malloc(sizeof(seg_t) * num_long_seg);
    cudaMemcpyAsync(long_segs_og, dev_mem->d_long_seg_og, sizeof(seg_t) * num_long_seg,
                    cudaMemcpyDeviceToHost, cudastream);
    cudaStreamSynchronize(cudastream);

    unsigned *map = new unsigned[num_long_seg];
    for (unsigned i = 0; i < num_long_seg; i++)
        map[i] = i;
    pairsort(long_segs_og, map, num_long_seg);
    free(long_segs_og);

    if (dev_mem->d_map)
        cudaFree(dev_mem->d_map);
    cudaMalloc(&dev_mem->d_map, sizeof(unsigned) * num_long_seg);
    cudaMemcpyAsync(dev_mem->d_map, map, sizeof(unsigned) * num_long_seg,
                    cudaMemcpyHostToDevice, cudastream);
    free(map);

    plscore_async_long_forward_dp(&sp->dev_mem, &sp->cudastream);
    cudaEventRecord(sp->stopevent, cudastream);
    plmem_async_d2h_long_memcpy(sp);
    sp->busy = true;
    cudaCheck();
}

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

void init_stream_gpu(size_t* total_n, int* max_reads, int *min_n, char gpu_config_file[], Misc misc) {
    plmem_stream_initialize(total_n, max_reads, min_n, gpu_config_file);
    plrange_upload_misc(misc);
    plscore_upload_misc(misc);
}

/**
 * worker for launching forward chaining on gpu (streaming)
 * use KMALLOC and kfree for cpu memory management
 * [in/out] in_arr_: ptr to array of reads, updated to a batch launched in previous run 
 *                  (NULL if no finishing batch)
 * [in/out] n_read_: ptr to num of reads in array, updated to a batch launched in previous run
 *                  (NULL if no finishing batch)
*/
void chain_stream_gpu(const mm_idx_t *mi, const mm_mapopt_t *opt, chain_read_t **in_arr_, int *n_read_,
                      int thread_id, void* km) {
    // assume only one seg. and qlen_sum desn't matter
    assert(opt->max_frag_len <= 0);
    Misc misc = build_misc(mi, opt, 0, 1);
    plchain_cal_score_async(mi, opt, in_arr_, n_read_, misc, stream_setup, thread_id, km);
}


/**
 * worker for finish all forward chaining kernenls on gpu
 * use KMALLOC and kfree for cpu memory management
 * [out] batches:   array of batches
 * [out] num_reads: array of number of reads in each batch
 */
void finish_stream_gpu(const mm_idx_t *mi, const mm_mapopt_t *opt, chain_read_t** reads_,
                       int* n_read_, int t, void* km) {
    // assume only one seg. and qlen_sum desn't matter
    assert(opt->max_frag_len <= 0);
    Misc misc = build_misc(mi, opt, 0, 1);
    /* Sync all the pending batches + backtracking */
    if (!stream_setup.streams[t].busy) {
        *reads_ = NULL;
        *n_read_ = 0;
        return;
    }

    chain_read_t* reads;
    int n_read = 0;
    cudaStreamSynchronize(stream_setup.streams[t].cudastream);
    cudaCheck();

    n_read = plchain_post_gpu_helper(stream_setup, t, misc, km);
    reads = stream_setup.streams[t].reads;
    stream_setup.streams[t].busy = false;

    // Collect reads that need GPU re-chaining
    int *rechain_indices = (int*)malloc(sizeof(int) * n_read);
    int n_rechain = 0;
    for (int i = 0; i < n_read; i++) {
        if (needs_rmq_rechain(opt, &reads[i])) {
            rechain_indices[n_rechain++] = i;
        }
    }

    // GPU voting-based re-chain if any reads need it
    if (n_rechain > 0) {
        plvoting_rechain_batch(mi, opt, reads, rechain_indices, n_rechain, misc, km);
    }
    free(rechain_indices);

    // Call post_chaining_helper for all reads (sets frag_gap, handles other cases)
    for (int i = 0; i < n_read; i++) {
        post_chaining_helper(mi, opt, &reads[i], misc, km);
    }

    *reads_ = reads;
    *n_read_ = n_read;

}


void free_stream_gpu(int n_threads){
    plmem_stream_cleanup();
}

/* gpu_rechain_batch has been superseded by plvoting_rechain_batch (plvoting.cu).
 * The voting-based approach replaces the RMQ-tree with a GPU histogram that
 * identifies high-coverage reference regions, then runs mg_lchain_dp with
 * bw_long on the filtered anchor set.  See gpu/plvoting.cu for details. */

#ifdef __cplusplus
} // extern "C"
#endif  // __cplusplus
