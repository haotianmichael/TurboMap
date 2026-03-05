#ifndef _PLCHAIN_H_
#define _PLCHAIN_H_

/* Range Kernel configuaration */
typedef struct range_kernel_config_t {
    int blockdim;           // number of threads in each block
    int cut_check_anchors;  // number of anchors to check around each cut
    int anchor_per_block;   // number of anchors assgined to one block = max_it * blockdim
} range_kernel_config_t;

/* Score Generation Kernel configuration */
typedef struct score_kernel_config_t{
    int micro_batch;
    int short_blockdim;
    int long_blockdim;
    int mid_blockdim;
    int short_griddim;
    int long_griddim;
    int mid_griddim;
    int cut_unit;
    int long_seg_cutoff;
    int mid_seg_cutoff;
} score_kernel_config_t;

/* -----------------------------------------------------------------------
 * Pipeline-friendly split of chain_stream_gpu:
 *
 *   chain_stream_launch  - non-blocking: submit a batch to slot_id and return
 *   chain_stream_collect - blocking: sync slot_id, run backtrack+voting+post,
 *                          return the chain_read_t array + count
 *
 * Prerequisite: chain_stream_launch requires the slot to be IDLE (not busy).
 * chain_stream_collect returns NULL if the slot was never launched.
 * ----------------------------------------------------------------------- */
#ifdef __cplusplus
extern "C" {
#endif

struct mm_idx_t;
struct mm_mapopt_t;
typedef struct chain_read_t chain_read_t;

void chain_stream_launch(const struct mm_idx_t *mi, const struct mm_mapopt_t *opt,
                         chain_read_t *reads, int n_reads,
                         int slot_id, void *km);

chain_read_t *chain_stream_collect(const struct mm_idx_t *mi, const struct mm_mapopt_t *opt,
                                   int slot_id, int *n_reads_out, void *km);

#ifdef __cplusplus
}
#endif

#endif // _PLCHAIN_H_