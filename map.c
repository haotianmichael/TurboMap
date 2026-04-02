#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include "kthread.h"
#include "kvec.h"
#include "kalloc.h"
#include "sdust.h"
#include "mmpriv.h"
#include "bseq.h"
#include "khash.h"
#include "gpu/plalign.cuh"
#include "ksw2.h"
#include <nvToolsExt.h>

#define __AMD_SPLIT_KERNELS__ 1
struct mm_tbuf_s {
	void *km;
	int rep_len, frag_gap; // updated per read. 
	double timers[MM_N_THR_TIMERS];
}; // per thread

#if defined(__AMD_SPLIT_KERNELS__)

#include "plutils.h"

#define N_ACCUM 64

typedef struct{
    int batchid;
    void *km;           // memory pool for each batch
    int count;			// number of reads in the batch
    size_t total_n;		// total number of anchors in the batch
    chain_read_t *reads;
} mm_batch_trbuf_t;

// local variables required for each read processed by a CPU thread
typedef struct {
    mm_batch_trbuf_t acc_batch;
    int is_full;

    mm_batch_trbuf_t launched_batch;
    int has_launched;

    mm_batch_trbuf_t pending_batch;
    int is_pending;

} mm_trbuf_t;  // per thread

#endif

mm_tbuf_t *mm_tbuf_init(void)
{
	mm_tbuf_t *b;
	b = (mm_tbuf_t*)calloc(1, sizeof(mm_tbuf_t));
	if (!(mm_dbg_flag & 1)) b->km = km_init();
	return b;
}

void mm_tbuf_destroy(mm_tbuf_t *b)
{
	if (b == 0) return;
	km_destroy(b->km);
	free(b);
}

void *mm_tbuf_get_km(mm_tbuf_t *b)
{
	return b->km;
}

#if defined(__AMD_SPLIT_KERNELS__)
void mm_trbuf_batch_init(mm_batch_trbuf_t *batch_, int batch_max_reads) {
    batch_->count = 0;
    batch_->total_n = 0;
    batch_->reads = (chain_read_t *)malloc(sizeof(chain_read_t) * batch_max_reads);
    memset(batch_->reads, 0, sizeof(chain_read_t) * batch_max_reads);
    batch_->batchid = -1;
    batch_->km = km_init();
}

void mm_trbuf_batch_reset(mm_batch_trbuf_t *batch_, int batch_max_reads, const mm_mapopt_t *opt) {
	// free all the reads in the batch
    for (int i = 0; i < batch_->count; i++) {
        free_read(&batch_->reads[i], batch_->km);
    }



	/* reset memory pool km */
    km_stat_t kmst;
    if (batch_->km) {
        chain_read_t *last_read = batch_->reads + batch_->count;
        km_stat(batch_->km, &kmst);
		if (mm_dbg_flag & MM_DBG_PRINT_QNAME)
			fprintf(stderr, "QM\t%s\t%d\tBid=%d\tcap=%ld,avail=%ld,nCore=%ld,largest=%ld\n", 
			last_read->seq.name, last_read->seq.qlen_sum, batch_->batchid, kmst.capacity, kmst.available, kmst.n_cores, kmst.largest);
		assert(kmst.n_blocks == kmst.n_cores); // otherwise, there is a memory leak
        assert(kmst.capacity == kmst.meta_size + kmst.available);
        if (kmst.largest > 1U<<28 || (opt->cap_kalloc > 0 && kmst.capacity > opt->cap_kalloc)) {
			if (mm_dbg_flag & MM_DBG_PRINT_QNAME)
				fprintf(stderr, "[W::%s] reset thread-local memory after read %s\n", __func__, last_read->seq.name);
			km_destroy(batch_->km);
            batch_->km = km_init();
        }
    }

    batch_->count = 0;
    batch_->total_n = 0;
    batch_->batchid = -1;
}

void mm_trbuf_batch_destroy(mm_batch_trbuf_t *batch_){
	// free reads in the batch
    for (int i = 0; i < batch_->count; i++){
        free_read(&batch_->reads[i], batch_->km);
    }
    batch_->batchid = -1;
    batch_->total_n = 0;
    batch_->count = 0;
	/* clean memory pool */
    km_stat_t kmst;
    if (batch_->km) {
        km_stat(batch_->km, &kmst);
		if (mm_dbg_flag & MM_DBG_PRINT_QNAME)
			fprintf(stderr, "Destroy memory pool cap=%ld,avail=%ld,nCore=%ld,largest=%ld\n", 
			kmst.capacity, kmst.available, kmst.n_cores, kmst.largest);
		assert(kmst.n_blocks == kmst.n_cores); // otherwise, there is a memory leak
        assert(kmst.capacity == kmst.meta_size + kmst.available);
        km_destroy(batch_->km);
        batch_->km = 0;
    }
    free(batch_->reads);
    batch_->reads = 0;
}


mm_trbuf_t *mm_trbuf_init(const int batch_max_reads, const mm_mapopt_t *opt)
{
    mm_trbuf_t *tr;
    tr = (mm_trbuf_t *)calloc(1, sizeof(mm_trbuf_t));
	tr->is_full = 0;
    tr->is_pending = 0;
    tr->has_launched = 0;
    mm_trbuf_batch_init(&tr->acc_batch, batch_max_reads);
    tr->acc_batch.batchid = 0;
    mm_trbuf_batch_init(&tr->pending_batch, batch_max_reads);
    tr->pending_batch.batchid = 1;
    mm_trbuf_batch_init(&tr->launched_batch, batch_max_reads);
    tr->launched_batch.batchid = 2;
    return tr;
}

void mm_trbuf_destroy(mm_trbuf_t *tr)
{
    if (tr == 0) return;
    mm_trbuf_batch_destroy(&tr->acc_batch);
    mm_trbuf_batch_destroy(&tr->pending_batch);
    mm_trbuf_batch_destroy(&tr->launched_batch);
    free(tr);
}
#endif

static int mm_dust_minier(void *km, int n, mm128_t *a, int l_seq, const char *seq, int sdust_thres)
{
	int n_dreg, j, k, u = 0;
	const uint64_t *dreg;
	sdust_buf_t *sdb;
	if (sdust_thres <= 0) return n;
	sdb = sdust_buf_init(km);
	dreg = sdust_core((const uint8_t*)seq, l_seq, sdust_thres, 64, &n_dreg, sdb);
	for (j = k = 0; j < n; ++j) { // squeeze out minimizers that significantly overlap with LCRs
		int32_t qpos = (uint32_t)a[j].y>>1, span = a[j].x&0xff;
		int32_t s = qpos - (span - 1), e = s + span;
		while (u < n_dreg && (int32_t)dreg[u] <= s) ++u;
		if (u < n_dreg && (int32_t)(dreg[u]>>32) < e) {
			int v, l = 0;
			for (v = u; v < n_dreg && (int32_t)(dreg[v]>>32) < e; ++v) { // iterate over LCRs overlapping this minimizer
				int ss = s > (int32_t)(dreg[v]>>32)? s : dreg[v]>>32;
				int ee = e < (int32_t)dreg[v]? e : (uint32_t)dreg[v];
				l += ee - ss;
			}
			if (l <= span>>1) a[k++] = a[j]; // keep the minimizer if less than half of it falls in masked region
		} else a[k++] = a[j];
	}
	sdust_buf_destroy(sdb);
	return k; // the new size
}

static void collect_minimizers(void *km, const mm_mapopt_t *opt, const mm_idx_t *mi, int n_segs, const int *qlens, const char **seqs, mm128_v *mv)
{
	int i, n, sum = 0;
	mv->n = 0;
	for (i = n = 0; i < n_segs; ++i) {
		size_t j;
		mm_sketch(km, seqs[i], qlens[i], mi->w, mi->k, i, mi->flag&MM_I_HPC, mv);
		for (j = n; j < mv->n; ++j)
			mv->a[j].y += sum << 1;
		if (opt->sdust_thres > 0) // mask low-complexity minimizers
			mv->n = n + mm_dust_minier(km, mv->n - n, mv->a + n, qlens[i], seqs[i], opt->sdust_thres);
		sum += qlens[i], n = mv->n;
	}
}

#include "ksort.h"
#define heap_lt(a, b) ((a).x > (b).x)
KSORT_INIT(heap, mm128_t, heap_lt)

static inline int skip_seed(int flag, uint64_t r, const mm_seed_t *q, const char *qname, int qlen, const mm_idx_t *mi, int *is_self)
{
	*is_self = 0;
	if (qname && (flag & (MM_F_NO_DIAG|MM_F_NO_DUAL))) {
		const mm_idx_seq_t *s = &mi->seq[r>>32];
		int cmp;
		cmp = strcmp(qname, s->name);
		if ((flag&MM_F_NO_DIAG) && cmp == 0 && (int)s->len == qlen) {
			if ((uint32_t)r>>1 == (q->q_pos>>1)) return 1; // avoid the diagnonal anchors
			if ((r&1) == (q->q_pos&1)) *is_self = 1; // this flag is used to avoid spurious extension on self chain
		}
		if ((flag&MM_F_NO_DUAL) && cmp > 0) // all-vs-all mode: map once
			return 1;
	}
	if (flag & (MM_F_FOR_ONLY|MM_F_REV_ONLY)) {
		if ((r&1) == (q->q_pos&1)) { // forward strand
			if (flag & MM_F_REV_ONLY) return 1;
		} else {
			if (flag & MM_F_FOR_ONLY) return 1;
		}
	}
	return 0;
}

static mm128_t *collect_seed_hits_heap(void *km, const mm_mapopt_t *opt, int max_occ, const mm_idx_t *mi, const char *qname, const mm128_v *mv, int qlen, int64_t *n_a, int *rep_len,
								  int *n_mini_pos, uint64_t **mini_pos)
{
	int i, n_m, heap_size = 0;
	int64_t j, n_for = 0, n_rev = 0;
	mm_seed_t *m;
	mm128_t *a, *heap;

	m = mm_collect_matches(km, &n_m, qlen, max_occ, opt->max_max_occ, opt->occ_dist, mi, mv, n_a, rep_len, n_mini_pos, mini_pos);

	heap = (mm128_t*)kmalloc(km, n_m * sizeof(mm128_t));
	a = (mm128_t*)kmalloc(km, *n_a * sizeof(mm128_t));

	for (i = 0, heap_size = 0; i < n_m; ++i) {
		if (m[i].n > 0) {
			heap[heap_size].x = m[i].cr[0];
			heap[heap_size].y = (uint64_t)i<<32;
			++heap_size;
		}
	}
	ks_heapmake_heap(heap_size, heap);
	while (heap_size > 0) {
		mm_seed_t *q = &m[heap->y>>32];
		mm128_t *p;
		uint64_t r = heap->x;
		int32_t is_self, rpos = (uint32_t)r >> 1;
		if (!skip_seed(opt->flag, r, q, qname, qlen, mi, &is_self)) {
			if ((r&1) == (q->q_pos&1)) { // forward strand
				p = &a[n_for++];
				p->x = (r&0xffffffff00000000ULL) | rpos;
				p->y = (uint64_t)q->q_span << 32 | q->q_pos >> 1;
			} else { // reverse strand
				p = &a[(*n_a) - (++n_rev)];
				p->x = 1ULL<<63 | (r&0xffffffff00000000ULL) | rpos;
				p->y = (uint64_t)q->q_span << 32 | (qlen - ((q->q_pos>>1) + 1 - q->q_span) - 1);
			}
			p->y |= (uint64_t)q->seg_id << MM_SEED_SEG_SHIFT;
			if (q->is_tandem) p->y |= MM_SEED_TANDEM;
			if (is_self) p->y |= MM_SEED_SELF;
		}
		// update the heap
		if ((uint32_t)heap->y < q->n - 1) {
			++heap[0].y;
			heap[0].x = m[heap[0].y>>32].cr[(uint32_t)heap[0].y];
		} else {
			heap[0] = heap[heap_size - 1];
			--heap_size;
		}
		ks_heapdown_heap(0, heap_size, heap);
	}
	kfree(km, m);
	kfree(km, heap);

	// reverse anchors on the reverse strand, as they are in the descending order
	for (j = 0; j < n_rev>>1; ++j) {
		mm128_t t = a[(*n_a) - 1 - j];
		a[(*n_a) - 1 - j] = a[(*n_a) - (n_rev - j)];
		a[(*n_a) - (n_rev - j)] = t;
	}
	if (*n_a > n_for + n_rev) {
		memmove(a + n_for, a + (*n_a) - n_rev, n_rev * sizeof(mm128_t));
		*n_a = n_for + n_rev;
	}
	return a;
}

static mm128_t *collect_seed_hits(void *km, const mm_mapopt_t *opt, int max_occ, const mm_idx_t *mi, const char *qname, const mm128_v *mv, int qlen, int64_t *n_a, int *rep_len,
								  int *n_mini_pos, uint64_t **mini_pos)
{
	int i, n_m;
	mm_seed_t *m;
	mm128_t *a;
	m = mm_collect_matches(km, &n_m, qlen, max_occ, opt->max_max_occ, opt->occ_dist, mi, mv, n_a, rep_len, n_mini_pos, mini_pos);
	a = (mm128_t*)kmalloc(km, *n_a * sizeof(mm128_t));
	for (i = 0, *n_a = 0; i < n_m; ++i) {
		mm_seed_t *q = &m[i];
		const uint64_t *r = q->cr;
		uint32_t k;
		for (k = 0; k < q->n; ++k) {
			int32_t is_self, rpos = (uint32_t)r[k] >> 1;
			mm128_t *p;
			if (skip_seed(opt->flag, r[k], q, qname, qlen, mi, &is_self)) continue;
			p = &a[(*n_a)++];
			if ((r[k]&1) == (q->q_pos&1)) { // forward strand
				p->x = (r[k]&0xffffffff00000000ULL) | rpos;
				p->y = (uint64_t)q->q_span << 32 | q->q_pos >> 1;
			} else if (!(opt->flag & MM_F_QSTRAND)) { // reverse strand and not in the query-strand mode
				p->x = 1ULL<<63 | (r[k]&0xffffffff00000000ULL) | rpos;
				p->y = (uint64_t)q->q_span << 32 | (qlen - ((q->q_pos>>1) + 1 - q->q_span) - 1);
			} else { // reverse strand; query-strand
				int32_t len = mi->seq[r[k]>>32].len;
				p->x = 1ULL<<63 | (r[k]&0xffffffff00000000ULL) | (len - (rpos + 1 - q->q_span) - 1); // coordinate only accurate for non-HPC seeds
				p->y = (uint64_t)q->q_span << 32 | q->q_pos >> 1;
			}
			p->y |= (uint64_t)q->seg_id << MM_SEED_SEG_SHIFT;
			if (q->is_tandem) p->y |= MM_SEED_TANDEM;
			if (is_self) p->y |= MM_SEED_SELF;
		}
	}
	kfree(km, m);
	radix_sort_128x(a, a + (*n_a));
	return a;
}

static void chain_post(const mm_mapopt_t *opt, int max_chain_gap_ref, const mm_idx_t *mi, void *km, int qlen, int n_segs, const int *qlens, int *n_regs, mm_reg1_t *regs, mm128_t *a)
{
	if (!(opt->flag & MM_F_ALL_CHAINS)) { // don't choose primary mapping(s)
		mm_set_parent(km, opt->mask_level, opt->mask_len, *n_regs, regs, opt->a * 2 + opt->b, opt->flag&MM_F_HARD_MLEVEL, opt->alt_drop);
		if (n_segs <= 1) mm_select_sub(km, opt->pri_ratio, mi->k*2, opt->best_n, 1, opt->max_gap * 0.8, n_regs, regs);
		else mm_select_sub_multi(km, opt->pri_ratio, 0.2f, 0.7f, max_chain_gap_ref, mi->k*2, opt->best_n, n_segs, qlens, n_regs, regs);
	}
}

static mm_reg1_t *align_regs(const mm_mapopt_t *opt, const mm_idx_t *mi, void *km, int qlen, const char *seq, int *n_regs, mm_reg1_t *regs, mm128_t *a)
{
	if (!(opt->flag & MM_F_CIGAR)) return regs;
	regs = mm_align_skeleton(km, opt, mi, qlen, seq, n_regs, regs, a); // this calls mm_filter_regs()
	if (!(opt->flag & MM_F_ALL_CHAINS)) { // don't choose primary mapping(s)
		mm_set_parent(km, opt->mask_level, opt->mask_len, *n_regs, regs, opt->a * 2 + opt->b, opt->flag&MM_F_HARD_MLEVEL, opt->alt_drop);
		mm_select_sub(km, opt->pri_ratio, mi->k*2, opt->best_n, 0, opt->max_gap * 0.8, n_regs, regs);
		mm_set_sam_pri(*n_regs, regs);
	}
	return regs;
}

#if defined(__AMD_SPLIT_KERNELS__)
void mm_map_seed(const mm_idx_t *mi, const mm_mapopt_t *opt,
                 chain_read_t *read_, mm_tbuf_t *b, void *km) {
    int n_segs = read_->n_seg;
    const int *qlens = read_->qlens;
	const char **seqs = read_->qseqs;
    const char *qname = read_->seq.name;
    int *rep_len = &read_->rep_len;
    int *qlen_sum = &read_->seq.qlen_sum;
    int *n_mini_pos = &read_->n_mini_pos;
    uint64_t **mini_pos = &read_->mini_pos;
    int64_t *n_a = &read_->n;
    mm128_t **a = &read_->a;

    int i;
    mm128_v mv = {0,0,0};
	double *timers = b->timers;
	double t1 = realtime();

    for (i = 0, *qlen_sum = 0; i < n_segs; ++i) *qlen_sum += qlens[i];

    if (*qlen_sum == 0 || n_segs <= 0 || n_segs > MM_MAX_SEG) return;
	if (opt->max_qlen > 0 && *qlen_sum > opt->max_qlen) return;

	collect_minimizers(km, opt, mi, n_segs, qlens, seqs, &mv);
	if (opt->q_occ_frac > 0.0f) mm_seed_mz_flt(km, &mv, opt->mid_occ, opt->q_occ_frac);
	if (opt->flag & MM_F_HEAP_SORT) *a = collect_seed_hits_heap(km, opt, opt->mid_occ, mi, qname, &mv, *qlen_sum, n_a, rep_len, n_mini_pos, mini_pos);
	else *a = collect_seed_hits(km, opt, opt->mid_occ, mi, qname, &mv, *qlen_sum, n_a, rep_len, n_mini_pos, mini_pos);

	if (mm_dbg_flag & MM_DBG_PRINT_SEED) {
		fprintf(stderr, "RS\t%d\n", *rep_len);
		for (i = 0; i < *n_a; ++i)
			fprintf(stderr, "SD\t%s\t%d\t%c\t%d\t%d\t%d\n", mi->seq[(*a)[i].x<<1>>33].name, (int32_t)(*a)[i].x, "+-"[(*a)[i].x>>63], (int32_t)(*a)[i].y, (int32_t)((*a)[i].y>>32&0xff),
					i == 0? 0 : ((int32_t)(*a)[i].y - (int32_t)(*a)[i-1].y) - ((int32_t)(*a)[i].x - (int32_t)(*a)[i-1].x));
	}
	kfree(km, mv.a);
	timers[MM_TIME_SEED] += realtime() - t1;
}

Misc build_misc(const mm_idx_t *mi, const mm_mapopt_t *opt, const int64_t qlen_sum, const int n_seg) {
    int max_chain_gap_qry, max_chain_gap_ref, is_splice = !!(opt->flag & MM_F_SPLICE), is_sr = !!(opt->flag & MM_F_SR);
	float chn_pen_gap, chn_pen_skip;

	// set max chaining gap on the query and the reference sequence
	if (is_sr)
		max_chain_gap_qry = qlen_sum > opt->max_gap? qlen_sum : opt->max_gap;
	else max_chain_gap_qry = opt->max_gap;
	if (opt->max_gap_ref > 0) {
		max_chain_gap_ref = opt->max_gap_ref; // always honor mm_mapopt_t::max_gap_ref if set
	} else if (opt->max_frag_len > 0) {
		max_chain_gap_ref = opt->max_frag_len - qlen_sum;
		if (max_chain_gap_ref < opt->max_gap) max_chain_gap_ref = opt->max_gap;
	} else max_chain_gap_ref = opt->max_gap;

	chn_pen_gap  = opt->chain_gap_scale * 0.01 * mi->k;
	chn_pen_skip = opt->chain_skip_scale * 0.01 * mi->k;

    Misc misc;
    misc.max_iter = opt->max_chain_iter; // always set up MAX_UINT
    misc.max_dist_y = max_chain_gap_qry;
    misc.max_dist_x = max_chain_gap_ref;
    misc.max_skip = opt->max_chain_skip;
    misc.bw = opt->bw;
    misc.min_cnt = opt->min_cnt;
    misc.min_score = opt->min_chain_score;
    misc.is_cdna = is_splice;
    misc.n_seg = n_seg;

    misc.chn_pen_gap = chn_pen_gap;
    misc.chn_pen_skip = chn_pen_skip;

    return misc;
}

// Check if a read needs RMQ-style re-chaining (for GPU batch processing)
int needs_rmq_rechain(const mm_mapopt_t *opt, chain_read_t* read) {
    int n_segs = read->n_seg;
    int n_regs0 = read->n_u;
    int qlen_sum = read->seq.qlen_sum;
    uint64_t *u = read->u;
    mm128_t *a = read->a;

    if (opt->bw_long > opt->bw &&
        (opt->flag & (MM_F_SPLICE | MM_F_SR | MM_F_NO_LJOIN)) == 0 &&
        n_segs == 1 && n_regs0 > 1 && u != NULL && a != NULL) {
        if ((int32_t)u[0] <= 0) {
            return 0;  // Invalid chain, skip
        }
        int32_t st = (int32_t)a[0].y;
        int32_t en = (int32_t)a[(int32_t)u[0] - 1].y;
        if (qlen_sum - (en - st) > opt->rmq_rescue_size ||
            en - st > qlen_sum * opt->rmq_rescue_ratio) {
            return 1;  // Needs re-chaining
        }
    }
    return 0;
}

// Prepare a read's anchors for re-chaining (consolidate and sort)
void prepare_rechain_anchors(chain_read_t* read, void *km) {
    int n_regs0 = read->n_u;
    uint64_t *u = read->u;
    mm128_t *a = read->a;

    // Calculate total anchor count from all chains
    int64_t total_n = 0;
    for (int i = 0; i < n_regs0; i++) {
        total_n += (int32_t)u[i];
    }
    read->n = total_n;

    // Free old chain array
    kfree(km, u);
    read->u = NULL;
    read->n_u = 0;

    // Sort anchors for re-chaining
    radix_sort_128x(a, a + total_n);
}

void post_chaining_helper(const mm_idx_t *mi, const mm_mapopt_t *opt, chain_read_t* read, Misc misc, void *km) {
    int n_segs = read->n_seg;
    const char *qname = read->seq.name;
	int *rep_len = &read->rep_len;
    int *frag_gap = &read->frag_gap;
    int *qlen_sum = &read->seq.qlen_sum;
    int *n_regs0 = &read->n_u;
	int *n_mini_pos = &read->n_mini_pos;
    uint64_t **mini_pos = &read->mini_pos;
    int64_t *n_a = &read->n;
    uint64_t **u = &read->u;
    mm128_t **a = &read->a;

    int i;
    mm128_v mv = {0, 0, 0};

    // RMQ re-chaining is handled by GPU batch processing (gpu_batch_rechain)
    // Skip this branch - reads needing re-chain are identified by needs_rmq_rechain()
    if (opt->bw_long > opt->bw &&
        (opt->flag & (MM_F_SPLICE | MM_F_SR | MM_F_NO_LJOIN)) == 0 &&
        n_segs == 1 && *n_regs0 >= 1 && *u != NULL && *a != NULL) {
        // GPU batch re-chaining handles this case (covers 1 or more chains)
        // Do nothing here - just fall through to set frag_gap
    }
    else if (opt->max_occ > opt->mid_occ && *rep_len > 0 &&
             !(opt->flag & MM_F_RMQ)) {  // re-chain, mostly for short reads
        int rechain = 0;
		if (*n_regs0 > 0) { // test if the best chain has all the segments
			int n_chained_segs = 1, max = 0, max_i = -1, max_off = -1, off = 0;
			for (i = 0; i < *n_regs0; ++i) { // find the best chain
				if (max < (int)((*u)[i]>>32)) max = (*u)[i]>>32, max_i = i, max_off = off;
				off += (uint32_t)(*u)[i];
			}
			for (i = 1; i < (int32_t)(*u)[max_i]; ++i) // count the number of segments in the best chain
				if (((*a)[max_off+i].y&MM_SEED_SEG_MASK) != ((*a)[max_off+i-1].y&MM_SEED_SEG_MASK))
					++n_chained_segs;
			if (n_chained_segs < n_segs)
				rechain = 1;
		} else rechain = 1;
		if (rechain) { // redo chaining with a higher max_occ threshold
			kfree(km, *a);
			kfree(km, *u);
			kfree(km, *mini_pos);
			if (opt->flag & MM_F_HEAP_SORT) *a = collect_seed_hits_heap(km, opt, opt->max_occ, mi, qname, &mv, *qlen_sum, n_a, rep_len, n_mini_pos, mini_pos);
			else *a = collect_seed_hits(km, opt, opt->max_occ, mi, qname, &mv, *qlen_sum, n_a, rep_len, n_mini_pos, mini_pos);
			*a = mg_lchain_dp(misc.max_dist_x, misc.max_dist_y, opt->bw, opt->max_chain_skip, opt->max_chain_iter, opt->min_cnt, opt->min_chain_score,
							 misc.chn_pen_gap, misc.chn_pen_skip, misc.is_cdna, n_segs, *n_a, *a, n_regs0, u, km);
			kfree(km, mv.a);
		}
    }
    *frag_gap = misc.max_dist_x;
}

void mm_map_chain(const mm_idx_t *mi, const mm_mapopt_t *opt,
                  chain_read_t *read_, mm_tbuf_t *b, void *km) {
    int n_segs = read_->n_seg;
    const char *qname = read_->seq.name;
    int *rep_len = &read_->rep_len;
    int *frag_gap = &read_->frag_gap;
    int *qlen_sum = &read_->seq.qlen_sum;
    int *n_regs0 = &read_->n_u;
    int *n_mini_pos = &read_->n_mini_pos;
    uint64_t **mini_pos = &read_->mini_pos;
    int64_t *n_a = &read_->n;
    uint64_t **u = &read_->u;
    mm128_t **a = &read_->a;

    int i;
	int max_chain_gap_qry, max_chain_gap_ref, is_splice = !!(opt->flag & MM_F_SPLICE), is_sr = !!(opt->flag & MM_F_SR);
	mm128_v mv = {0,0,0};
	float chn_pen_gap, chn_pen_skip;
	double *timers = b->timers;

	// set max chaining gap on the query and the reference sequence
	if (is_sr)
		max_chain_gap_qry = *qlen_sum > opt->max_gap? *qlen_sum : opt->max_gap;
	else max_chain_gap_qry = opt->max_gap;
	if (opt->max_gap_ref > 0) {
		max_chain_gap_ref = opt->max_gap_ref; // always honor mm_mapopt_t::max_gap_ref if set
	} else if (opt->max_frag_len > 0) {
		max_chain_gap_ref = opt->max_frag_len - *qlen_sum;
		if (max_chain_gap_ref < opt->max_gap) max_chain_gap_ref = opt->max_gap;
	} else max_chain_gap_ref = opt->max_gap;

	chn_pen_gap  = opt->chain_gap_scale * 0.01 * mi->k;
	chn_pen_skip = opt->chain_skip_scale * 0.01 * mi->k;
	if (opt->flag & MM_F_RMQ) {
		*a = mg_lchain_rmq(opt->max_gap, opt->rmq_inner_dist, opt->bw, opt->max_chain_skip, opt->rmq_size_cap, opt->min_cnt, opt->min_chain_score,
						  chn_pen_gap, chn_pen_skip, *n_a, *a, n_regs0, u, km);
	} else {
		*a = mg_lchain_dp(max_chain_gap_ref, max_chain_gap_qry, opt->bw, opt->max_chain_skip, opt->max_chain_iter, opt->min_cnt, opt->min_chain_score,
						 chn_pen_gap, chn_pen_skip, is_splice, n_segs, *n_a, *a, n_regs0, u, km);
	}

	if (opt->bw_long > opt->bw && (opt->flag & (MM_F_SPLICE|MM_F_SR|MM_F_NO_LJOIN)) == 0 && n_segs == 1 && *n_regs0 > 1) { // re-chain/long-join for long sequences
		int32_t st = (int32_t)(*a)[0].y, en = (int32_t)(*a)[(int32_t)(*u)[0] - 1].y;
		if (*qlen_sum - (en - st) > opt->rmq_rescue_size || en - st > *qlen_sum * opt->rmq_rescue_ratio) {
			int32_t i;
			for (i = 0, *n_a = 0; i < *n_regs0; ++i) *n_a += (int32_t)(*u)[i];
			kfree(km, *u);
			radix_sort_128x(*a, (*a) + *n_a);
			*a = mg_lchain_rmq(opt->max_gap, opt->rmq_inner_dist, opt->bw_long, opt->max_chain_skip, opt->rmq_size_cap, opt->min_cnt, opt->min_chain_score,
							  chn_pen_gap, chn_pen_skip, *n_a, *a, n_regs0, u, km);
		}
	} else if (opt->max_occ > opt->mid_occ && *rep_len > 0 && !(opt->flag & MM_F_RMQ)) { // re-chain, mostly for short reads
		int rechain = 0;
		if (*n_regs0 > 0) { // test if the best chain has all the segments
			int n_chained_segs = 1, max = 0, max_i = -1, max_off = -1, off = 0;
			for (i = 0; i < *n_regs0; ++i) { // find the best chain
				if (max < (int)((*u)[i]>>32)) max = (*u)[i]>>32, max_i = i, max_off = off;
				off += (uint32_t)(*u)[i];
			}
			for (i = 1; i < (int32_t)(*u)[max_i]; ++i) // count the number of segments in the best chain
				if (((*a)[max_off+i].y&MM_SEED_SEG_MASK) != ((*a)[max_off+i-1].y&MM_SEED_SEG_MASK))
					++n_chained_segs;
			if (n_chained_segs < n_segs)
				rechain = 1;
		} else rechain = 1;
		if (rechain) { // redo chaining with a higher max_occ threshold
			kfree(km, *a);
			kfree(km, *u);
			kfree(km, *mini_pos);
			if (opt->flag & MM_F_HEAP_SORT) *a = collect_seed_hits_heap(km, opt, opt->max_occ, mi, qname, &mv, *qlen_sum, n_a, rep_len, n_mini_pos, mini_pos);
			else *a = collect_seed_hits(km, opt, opt->max_occ, mi, qname, &mv, *qlen_sum, n_a, rep_len, n_mini_pos, mini_pos);
			*a = mg_lchain_dp(max_chain_gap_ref, max_chain_gap_qry, opt->bw, opt->max_chain_skip, opt->max_chain_iter, opt->min_cnt, opt->min_chain_score,
							 chn_pen_gap, chn_pen_skip, is_splice, n_segs, *n_a, *a, n_regs0, u, km);
			kfree(km, mv.a);
		}
	}
	*frag_gap = max_chain_gap_ref;
}

static inline mm_reg1_t *mm_insert_reg(const mm_reg1_t *r, int i, int *n_regs, mm_reg1_t *regs)
{
	regs = (mm_reg1_t*)realloc(regs, (*n_regs + 1) * sizeof(mm_reg1_t));
	if (i + 1 != *n_regs)
		memmove(&regs[i + 2], &regs[i + 1], sizeof(mm_reg1_t) * (*n_regs - i - 1));
	regs[i + 1] = *r;
	++*n_regs;
	return regs;
}

void mm_map_align(const mm_idx_t *mi, const mm_mapopt_t *opt,
                  chain_read_t *read_, mm_reg1_t **regs, int *n_regs, mm_tbuf_t *b, void *km) {
    int n_segs = read_->n_seg;
    uint64_t **mini_pos = &read_->mini_pos;
    uint64_t *u = read_->u;
    mm128_t *a = read_->a;


	kfree(km, a);
	kfree(km, u);
	kfree(km, *mini_pos);
}

#endif

void mm_map_frag(const mm_idx_t *mi, int n_segs, const int *qlens, const char **seqs, int *n_regs, mm_reg1_t **regs, mm_tbuf_t *b, const mm_mapopt_t *opt, const char *qname)
{
	int i, j, rep_len, qlen_sum, n_regs0, n_mini_pos;
	int max_chain_gap_qry, max_chain_gap_ref, is_splice = !!(opt->flag & MM_F_SPLICE), is_sr = !!(opt->flag & MM_F_SR);
	uint32_t hash;
	int64_t n_a;
	uint64_t *u, *mini_pos;
	mm128_t *a;
	mm128_v mv = {0,0,0};
	mm_reg1_t *regs0;
	km_stat_t kmst;
	float chn_pen_gap, chn_pen_skip;
	double *timers = b->timers;
	double t1 = realtime();

	for (i = 0, qlen_sum = 0; i < n_segs; ++i)
		qlen_sum += qlens[i], n_regs[i] = 0, regs[i] = 0;

	if (qlen_sum == 0 || n_segs <= 0 || n_segs > MM_MAX_SEG) return;
	if (opt->max_qlen > 0 && qlen_sum > opt->max_qlen) return;

	hash  = qname && !(opt->flag & MM_F_NO_HASH_NAME)? __ac_X31_hash_string(qname) : 0;
	hash ^= __ac_Wang_hash(qlen_sum) + __ac_Wang_hash(opt->seed);
	hash  = __ac_Wang_hash(hash);

	collect_minimizers(b->km, opt, mi, n_segs, qlens, seqs, &mv);
	if (opt->q_occ_frac > 0.0f) mm_seed_mz_flt(b->km, &mv, opt->mid_occ, opt->q_occ_frac);
	if (opt->flag & MM_F_HEAP_SORT) a = collect_seed_hits_heap(b->km, opt, opt->mid_occ, mi, qname, &mv, qlen_sum, &n_a, &rep_len, &n_mini_pos, &mini_pos);
	else a = collect_seed_hits(b->km, opt, opt->mid_occ, mi, qname, &mv, qlen_sum, &n_a, &rep_len, &n_mini_pos, &mini_pos);

	if (mm_dbg_flag & MM_DBG_PRINT_SEED) {
		fprintf(stderr, "RS\t%d\n", rep_len);
		for (i = 0; i < n_a; ++i)
			fprintf(stderr, "SD\t%s\t%d\t%c\t%d\t%d\t%d\n", mi->seq[a[i].x<<1>>33].name, (int32_t)a[i].x, "+-"[a[i].x>>63], (int32_t)a[i].y, (int32_t)(a[i].y>>32&0xff),
					i == 0? 0 : ((int32_t)a[i].y - (int32_t)a[i-1].y) - ((int32_t)a[i].x - (int32_t)a[i-1].x));
	}
	timers[MM_TIME_SEED] += realtime() - t1;

	t1 = realtime();
	// set max chaining gap on the query and the reference sequence
	if (is_sr)
		max_chain_gap_qry = qlen_sum > opt->max_gap? qlen_sum : opt->max_gap;
	else max_chain_gap_qry = opt->max_gap;
	if (opt->max_gap_ref > 0) {
		max_chain_gap_ref = opt->max_gap_ref; // always honor mm_mapopt_t::max_gap_ref if set
	} else if (opt->max_frag_len > 0) {
		max_chain_gap_ref = opt->max_frag_len - qlen_sum;
		if (max_chain_gap_ref < opt->max_gap) max_chain_gap_ref = opt->max_gap;
	} else max_chain_gap_ref = opt->max_gap;

	chn_pen_gap  = opt->chain_gap_scale * 0.01 * mi->k;
	chn_pen_skip = opt->chain_skip_scale * 0.01 * mi->k;
	if (opt->flag & MM_F_RMQ) {
		a = mg_lchain_rmq(opt->max_gap, opt->rmq_inner_dist, opt->bw, opt->max_chain_skip, opt->rmq_size_cap, opt->min_cnt, opt->min_chain_score,
						  chn_pen_gap, chn_pen_skip, n_a, a, &n_regs0, &u, b->km);
	} else {
		a = mg_lchain_dp(max_chain_gap_ref, max_chain_gap_qry, opt->bw, opt->max_chain_skip, opt->max_chain_iter, opt->min_cnt, opt->min_chain_score,
						 chn_pen_gap, chn_pen_skip, is_splice, n_segs, n_a, a, &n_regs0, &u, b->km);
	}

	if (opt->bw_long > opt->bw && (opt->flag & (MM_F_SPLICE|MM_F_SR|MM_F_NO_LJOIN)) == 0 && n_segs == 1 && n_regs0 > 1) { // re-chain/long-join for long sequences
		int32_t st = (int32_t)a[0].y, en = (int32_t)a[(int32_t)u[0] - 1].y;
		if (qlen_sum - (en - st) > opt->rmq_rescue_size || en - st > qlen_sum * opt->rmq_rescue_ratio) {
			int32_t i;
			for (i = 0, n_a = 0; i < n_regs0; ++i) n_a += (int32_t)u[i];
			kfree(b->km, u);
			radix_sort_128x(a, a + n_a);
			a = mg_lchain_rmq(opt->max_gap, opt->rmq_inner_dist, opt->bw_long, opt->max_chain_skip, opt->rmq_size_cap, opt->min_cnt, opt->min_chain_score,
							  chn_pen_gap, chn_pen_skip, n_a, a, &n_regs0, &u, b->km);
		}
	} else if (opt->max_occ > opt->mid_occ && rep_len > 0 && !(opt->flag & MM_F_RMQ)) { // re-chain, mostly for short reads
		int rechain = 0;
		if (n_regs0 > 0) { // test if the best chain has all the segments
			int n_chained_segs = 1, max = 0, max_i = -1, max_off = -1, off = 0;
			for (i = 0; i < n_regs0; ++i) { // find the best chain
				if (max < (int)(u[i]>>32)) max = u[i]>>32, max_i = i, max_off = off;
				off += (uint32_t)u[i];
			}
			for (i = 1; i < (int32_t)u[max_i]; ++i) // count the number of segments in the best chain
				if ((a[max_off+i].y&MM_SEED_SEG_MASK) != (a[max_off+i-1].y&MM_SEED_SEG_MASK))
					++n_chained_segs;
			if (n_chained_segs < n_segs)
				rechain = 1;
		} else rechain = 1;
		if (rechain) { // redo chaining with a higher max_occ threshold
			kfree(b->km, a);
			kfree(b->km, u);
			kfree(b->km, mini_pos);
			if (opt->flag & MM_F_HEAP_SORT) a = collect_seed_hits_heap(b->km, opt, opt->max_occ, mi, qname, &mv, qlen_sum, &n_a, &rep_len, &n_mini_pos, &mini_pos);
			else a = collect_seed_hits(b->km, opt, opt->max_occ, mi, qname, &mv, qlen_sum, &n_a, &rep_len, &n_mini_pos, &mini_pos);
			a = mg_lchain_dp(max_chain_gap_ref, max_chain_gap_qry, opt->bw, opt->max_chain_skip, opt->max_chain_iter, opt->min_cnt, opt->min_chain_score,
							 chn_pen_gap, chn_pen_skip, is_splice, n_segs, n_a, a, &n_regs0, &u, b->km);
		}
	}
	b->frag_gap = max_chain_gap_ref;
	b->rep_len = rep_len;
	timers[MM_TIME_CHAIN] += realtime() - t1;

	t1 = realtime();
	regs0 = mm_gen_regs(b->km, hash, qlen_sum, n_regs0, u, a, !!(opt->flag&MM_F_QSTRAND));
	if (mi->n_alt) {
		mm_mark_alt(mi, n_regs0, regs0);
		mm_hit_sort(b->km, &n_regs0, regs0, opt->alt_drop); // this step can be merged into mm_gen_regs(); will do if this shows up in profile
	}

	if (mm_dbg_flag & (MM_DBG_PRINT_SEED|MM_DBG_PRINT_CHAIN))
		for (j = 0; j < n_regs0; ++j)
			for (i = regs0[j].as; i < regs0[j].as + regs0[j].cnt; ++i)
				fprintf(stderr, "CN\t%d\t%s\t%d\t%c\t%d\t%d\t%d\n", j, mi->seq[a[i].x<<1>>33].name, (int32_t)a[i].x, "+-"[a[i].x>>63], (int32_t)a[i].y, (int32_t)(a[i].y>>32&0xff),
						i == regs0[j].as? 0 : ((int32_t)a[i].y - (int32_t)a[i-1].y) - ((int32_t)a[i].x - (int32_t)a[i-1].x));

	chain_post(opt, max_chain_gap_ref, mi, b->km, qlen_sum, n_segs, qlens, &n_regs0, regs0, a);
	if (!is_sr && !(opt->flag&MM_F_QSTRAND)) {
		mm_est_err(mi, qlen_sum, n_regs0, regs0, a, n_mini_pos, mini_pos);
		n_regs0 = mm_filter_strand_retained(n_regs0, regs0);
	}

	if (n_segs == 1) { // uni-segment
		regs0 = align_regs(opt, mi, b->km, qlens[0], seqs[0], &n_regs0, regs0, a);
		regs0 = (mm_reg1_t*)realloc(regs0, sizeof(*regs0) * n_regs0);
		mm_set_mapq(b->km, n_regs0, regs0, opt->min_chain_score, opt->a, rep_len, is_sr);
		n_regs[0] = n_regs0, regs[0] = regs0;
	} else { // multi-segment
		mm_seg_t *seg;
		seg = mm_seg_gen(b->km, hash, n_segs, qlens, n_regs0, regs0, n_regs, regs, a); // split fragment chain to separate segment chains
		free(regs0);
		for (i = 0; i < n_segs; ++i) {
			mm_set_parent(b->km, opt->mask_level, opt->mask_len, n_regs[i], regs[i], opt->a * 2 + opt->b, opt->flag&MM_F_HARD_MLEVEL, opt->alt_drop); // update mm_reg1_t::parent
			regs[i] = align_regs(opt, mi, b->km, qlens[i], seqs[i], &n_regs[i], regs[i], seg[i].a);
			mm_set_mapq(b->km, n_regs[i], regs[i], opt->min_chain_score, opt->a, rep_len, is_sr);
		}
		mm_seg_free(b->km, n_segs, seg);
		if (n_segs == 2 && opt->pe_ori >= 0 && (opt->flag&MM_F_CIGAR))
			mm_pair(b->km, max_chain_gap_ref, opt->pe_bonus, opt->a * 2 + opt->b, opt->a, qlens, n_regs, regs); // pairing
	}
	timers[MM_TIME_ALIGN] += realtime() - t1;

	kfree(b->km, mv.a);
	kfree(b->km, a);
	kfree(b->km, u);
	kfree(b->km, mini_pos);

	if (b->km) {
		km_stat(b->km, &kmst);
		if (mm_dbg_flag & MM_DBG_PRINT_QNAME)
			fprintf(stderr, "QM\t%s\t%d\tcap=%ld,nCore=%ld,largest=%ld\n", qname, qlen_sum, kmst.capacity, kmst.n_cores, kmst.largest);
		assert(kmst.n_blocks == kmst.n_cores); // otherwise, there is a memory leak
		if (kmst.largest > 1U<<28 || (opt->cap_kalloc > 0 && kmst.capacity > opt->cap_kalloc)) {
			if (mm_dbg_flag & MM_DBG_PRINT_QNAME)
				fprintf(stderr, "[W::%s] reset thread-local memory after read %s\n", __func__, qname);
			km_destroy(b->km);
			b->km = km_init();
		}
	}
}

mm_reg1_t *mm_map(const mm_idx_t *mi, int qlen, const char *seq, int *n_regs, mm_tbuf_t *b, const mm_mapopt_t *opt, const char *qname)
{
	mm_reg1_t *regs;
	mm_map_frag(mi, 1, &qlen, &seq, n_regs, &regs, b, opt, qname);
	return regs;
}

/**************************
 * Multi-threaded mapping *
 **************************/

typedef struct {
	int n_processed, n_threads, n_fp;
	int64_t mini_batch_size;
	const mm_mapopt_t *opt;
	mm_bseq_file_t **fp;
	const mm_idx_t *mi;
	kstring_t str;

	int n_parts;
	uint32_t *rid_shift;
	FILE *fp_split, **fp_parts;
} pipeline_t;

typedef struct {
	const pipeline_t *p;
    int n_seq, n_frag;
	mm_bseq1_t *seq;
	int *n_reg, *seg_off, *n_seg, *rep_len, *frag_gap;
	mm_reg1_t **reg;
	mm_tbuf_t **buf;
#if defined(__AMD_SPLIT_KERNELS__)
    mm_trbuf_t **trbuf;
	int batch_max_reads;
    size_t batch_max_anchors;
    int gpu_min_n;
#endif
} step_t;

#define MIN(a, b) ((a)<(b)?(a):(b))
#define MAX(a, b) ((a)>(b)?(a):(b))

// consolidate timers from worker threads
void mm_consolidate_timers(step_t *s, pipeline_t *p)
{
	// TODO: disabled for sysbio submission
	return;
	mm_time_seed_min = 0;
    mm_time_chain_min = 0;
    mm_time_align_min = 0;
    mm_time_seed_max = 0;
    mm_time_chain_max = 0;
    mm_time_align_max = 0;
    mm_time_seed_avg = 0;
    mm_time_chain_avg = 0;
    mm_time_align_avg = 0;
    for (int i = 0; i < p->n_threads; ++i) {
        mm_time_seed_min = MIN(mm_time_seed_min, s->buf[i]->timers[MM_TIME_SEED]);
		mm_time_chain_min = MIN(mm_time_chain_min, s->buf[i]->timers[MM_TIME_CHAIN]);
		mm_time_align_min = MIN(mm_time_align_min, s->buf[i]->timers[MM_TIME_ALIGN]);
		mm_time_seed_max = MAX(mm_time_seed_max, s->buf[i]->timers[MM_TIME_SEED]);
		mm_time_chain_max = MAX(mm_time_chain_max, s->buf[i]->timers[MM_TIME_CHAIN]);
		mm_time_align_max = MAX(mm_time_align_max, s->buf[i]->timers[MM_TIME_ALIGN]);
        mm_time_seed_avg += s->buf[i]->timers[MM_TIME_SEED];
        mm_time_chain_avg += s->buf[i]->timers[MM_TIME_CHAIN];
        mm_time_align_avg += s->buf[i]->timers[MM_TIME_ALIGN];
        mm_time_seed_sum += s->buf[i]->timers[MM_TIME_SEED];
		mm_time_chain_sum += s->buf[i]->timers[MM_TIME_CHAIN];
		mm_time_align_sum += s->buf[i]->timers[MM_TIME_ALIGN];
    }
    mm_time_seed_avg /= p->n_threads;
    mm_time_chain_avg /= p->n_threads;
    mm_time_align_avg /= p->n_threads;

    fprintf(stderr, "----------------------------------------------------\n");
	fprintf(stderr, "              Min (sec)  Max (sec)  Avg (sec)  \n");
	fprintf(stderr, "----------------------------------------------------\n");
	fprintf(stderr, "Seed    = %11.3f %11.3f %11.3f\n",
			mm_time_seed_min, mm_time_seed_max, mm_time_seed_avg);
	fprintf(stderr, "Chain   = %11.3f %11.3f %11.3f\n",
			mm_time_chain_min, mm_time_chain_max, mm_time_chain_avg);
	fprintf(stderr, "Align   = %11.3f %11.3f %11.3f\n",
			mm_time_align_min, mm_time_align_max, mm_time_align_avg);
	fprintf(stderr, "----------------------------------------------------\n");
	fprintf(stderr, "Avg (seed + chain + align) per thread = %.3f secs\n", (mm_time_seed_avg + mm_time_chain_avg + mm_time_align_avg));
	fprintf(stderr, "Total (seed + chain + align) (all batches) for %d thread(s) = %.3f secs\n", p->n_threads, (mm_time_seed_sum + mm_time_chain_sum + mm_time_align_sum));

}


#if defined(__AMD_SPLIT_KERNELS__)

void mm_trbuf_is_full(mm_trbuf_t* tr, step_t *s){
    while (tr->acc_batch.total_n > s->batch_max_anchors) { // if the batch is full
        tr->is_full = 1;
        tr->is_pending = 1;
		// move last read from acc_batch to pending batch (another memory poll)
        chain_read_t *read_ptr_acc_batch = &tr->acc_batch.reads[tr->acc_batch.count - 1];
        chain_read_t *read_ptr_pending_batch = &tr->pending_batch.reads[tr->pending_batch.count];
		/* deep copy, with memory pool transaction*/
        *read_ptr_pending_batch = *read_ptr_acc_batch;
        if (s->p->opt->flag & MM_F_INDEPEND_SEG) {
            read_ptr_pending_batch->qlens = (int *)kmalloc(tr->pending_batch.km, sizeof(int));
            read_ptr_pending_batch->qseqs = (const char **)kmalloc(tr->pending_batch.km, sizeof(const char*));
            read_ptr_pending_batch->qlens[0] = read_ptr_acc_batch->qlens[0];
            read_ptr_pending_batch->qseqs[0] = read_ptr_acc_batch->qseqs[0];
        } else {
            read_ptr_pending_batch->qlens = (int *)kmalloc(tr->pending_batch.km, sizeof(int)*read_ptr_acc_batch->n_seg);
            read_ptr_pending_batch->qseqs = (const char **)kmalloc(tr->pending_batch.km, sizeof(const char*)*read_ptr_acc_batch->n_seg);
            memcpy(read_ptr_pending_batch->qlens,  read_ptr_acc_batch->qlens, sizeof(int)*read_ptr_acc_batch->n_seg);
            memcpy(read_ptr_pending_batch->qseqs, read_ptr_acc_batch->qseqs, sizeof(const char*)*read_ptr_acc_batch->n_seg);
        }
        read_ptr_pending_batch->mini_pos = (uint64_t*)kmalloc(tr->pending_batch.km, read_ptr_acc_batch->n_mini_pos * sizeof(uint64_t));
		read_ptr_pending_batch->a = (mm128_t*)kmalloc(tr->pending_batch.km, read_ptr_acc_batch->n * sizeof(mm128_t));
        memcpy(read_ptr_pending_batch->mini_pos, read_ptr_acc_batch->mini_pos, read_ptr_acc_batch->n_mini_pos * sizeof(uint64_t));
        memcpy(read_ptr_pending_batch->a, read_ptr_acc_batch->a, read_ptr_acc_batch->n * sizeof(mm128_t));
        strcpy(read_ptr_pending_batch->seq.name, read_ptr_acc_batch->seq.name);
        tr->pending_batch.count++;
        tr->pending_batch.total_n += read_ptr_acc_batch->n;

        // remove read from acc_batch
        tr->acc_batch.count--;
        tr->acc_batch.total_n -= read_ptr_acc_batch->n;
        kfree(tr->acc_batch.km, read_ptr_acc_batch->mini_pos);
        kfree(tr->acc_batch.km, read_ptr_acc_batch->a);
        kfree(tr->acc_batch.km, read_ptr_acc_batch->qlens);
        kfree(tr->acc_batch.km, read_ptr_acc_batch->qseqs);
    }
}
#endif

#ifndef __AMD_SPLIT_KERNELS__
static void worker_for(void *_data, long i, int tid) // kt_for() callback
{
    step_t *s = (step_t*)_data;
	int qlens[MM_MAX_SEG], j, off = s->seg_off[i], pe_ori = s->p->opt->pe_ori;
	const char *qseqs[MM_MAX_SEG];
	double t = 0.0;
	mm_tbuf_t *b = s->buf[tid];
	assert(s->n_seg[i] <= MM_MAX_SEG);
	if (mm_dbg_flag & MM_DBG_PRINT_QNAME) {
		fprintf(stderr, "QR\t%s\t%d\t%d\n", s->seq[off].name, tid, s->seq[off].l_seq);
		t = realtime();
	}
	for (j = 0; j < s->n_seg[i]; ++j) {
		if (s->n_seg[i] == 2 && ((j == 0 && (pe_ori>>1&1)) || (j == 1 && (pe_ori&1))))
			mm_revcomp_bseq(&s->seq[off + j]);
		qlens[j] = s->seq[off + j].l_seq;
		qseqs[j] = s->seq[off + j].seq;
	}
	if (s->p->opt->flag & MM_F_INDEPEND_SEG) {
		for (j = 0; j < s->n_seg[i]; ++j) {
			mm_map_frag(s->p->mi, 1, &qlens[j], &qseqs[j], &s->n_reg[off+j], &s->reg[off+j], b, s->p->opt, s->seq[off+j].name);
			s->rep_len[off + j] = b->rep_len;
			s->frag_gap[off + j] = b->frag_gap;
		}
	} else {
		mm_map_frag(s->p->mi, s->n_seg[i], qlens, qseqs, &s->n_reg[off], &s->reg[off], b, s->p->opt, s->seq[off].name);
		for (j = 0; j < s->n_seg[i]; ++j) {
			s->rep_len[off + j] = b->rep_len;
			s->frag_gap[off + j] = b->frag_gap;
		}
	}
	for (j = 0; j < s->n_seg[i]; ++j) // flip the query strand and coordinate to the original read strand
		if (s->n_seg[i] == 2 && ((j == 0 && (pe_ori>>1&1)) || (j == 1 && (pe_ori&1)))) {
			int k, t;
			mm_revcomp_bseq(&s->seq[off + j]);
			for (k = 0; k < s->n_reg[off + j]; ++k) {
				mm_reg1_t *r = &s->reg[off + j][k];
				t = r->qs;
				r->qs = qlens[j] - r->qe;
				r->qe = qlens[j] - t;
				r->rev = !r->rev;
			}
		}
	if (mm_dbg_flag & MM_DBG_PRINT_QNAME)
		fprintf(stderr, "QT\t%s\t%d\t%.6f\n", s->seq[off].name, tid, realtime() - t);
}
#endif

static void merge_hits(step_t *s)
{
	int f, i, k0, k, max_seg = 0, *n_reg_part, *rep_len_part, *frag_gap_part, *qlens;
	void *km;
	FILE **fp = s->p->fp_parts;
	const mm_mapopt_t *opt = s->p->opt;

	km = km_init();
	for (f = 0; f < s->n_frag; ++f)
		max_seg = max_seg > s->n_seg[f]? max_seg : s->n_seg[f];
	qlens = CALLOC(int, max_seg + s->p->n_parts * 3);
	n_reg_part = qlens + max_seg;
	rep_len_part = n_reg_part + s->p->n_parts;
	frag_gap_part = rep_len_part + s->p->n_parts;
	for (f = 0, k = k0 = 0; f < s->n_frag; ++f) {
		k0 = k;
		for (i = 0; i < s->n_seg[f]; ++i, ++k) {
			int j, l, t, rep_len = 0;
			qlens[i] = s->seq[k].l_seq;
			for (j = 0, s->n_reg[k] = 0; j < s->p->n_parts; ++j) {
				mm_err_fread(&n_reg_part[j],    sizeof(int), 1, fp[j]);
				mm_err_fread(&rep_len_part[j],  sizeof(int), 1, fp[j]);
				mm_err_fread(&frag_gap_part[j], sizeof(int), 1, fp[j]);
				s->n_reg[k] += n_reg_part[j];
				if (rep_len < rep_len_part[j])
					rep_len = rep_len_part[j];
			}
			s->reg[k] = CALLOC(mm_reg1_t, s->n_reg[k]);
			for (j = 0, l = 0; j < s->p->n_parts; ++j) {
				for (t = 0; t < n_reg_part[j]; ++t, ++l) {
					mm_reg1_t *r = &s->reg[k][l];
					uint32_t capacity;
					mm_err_fread(r, sizeof(mm_reg1_t), 1, fp[j]);
					r->rid += s->p->rid_shift[j];
					if (opt->flag & MM_F_CIGAR) {
						mm_err_fread(&capacity, 4, 1, fp[j]);
						r->p = (mm_extra_t*)calloc(capacity, 4);
						r->p->capacity = capacity;
						mm_err_fread(r->p, r->p->capacity, 4, fp[j]);
					}
				}
			}
			if (!(opt->flag&MM_F_SR) && s->seq[k].l_seq >= opt->rank_min_len)
				mm_update_dp_max(s->seq[k].l_seq, s->n_reg[k], s->reg[k], opt->rank_frac, opt->a, opt->b);
			for (j = 0; j < s->n_reg[k]; ++j) {
				mm_reg1_t *r = &s->reg[k][j];
				if (r->p) r->p->dp_max2 = 0; // reset ->dp_max2 as mm_set_parent() doesn't clear it; necessary with mm_update_dp_max()
				r->subsc = 0; // this may not be necessary
				r->n_sub = 0; // n_sub will be an underestimate as we don't see all the chains now, but it can't be accurate anyway
			}
			mm_hit_sort(km, &s->n_reg[k], s->reg[k], opt->alt_drop);
			mm_set_parent(km, opt->mask_level, opt->mask_len, s->n_reg[k], s->reg[k], opt->a * 2 + opt->b, opt->flag&MM_F_HARD_MLEVEL, opt->alt_drop);
			if (!(opt->flag & MM_F_ALL_CHAINS)) {
				mm_select_sub(km, opt->pri_ratio, s->p->mi->k*2, opt->best_n, 0, opt->max_gap * 0.8, &s->n_reg[k], s->reg[k]);
				mm_set_sam_pri(s->n_reg[k], s->reg[k]);
			}
			mm_set_mapq(km, s->n_reg[k], s->reg[k], opt->min_chain_score, opt->a, rep_len, !!(opt->flag & MM_F_SR));
		}
		if (s->n_seg[f] == 2 && opt->pe_ori >= 0 && (opt->flag&MM_F_CIGAR))
			mm_pair(km, frag_gap_part[0], opt->pe_bonus, opt->a * 2 + opt->b, opt->a, qlens, &s->n_reg[k0], &s->reg[k0]);
	}
	free(qlens);
	km_destroy(km);
}

static void* kt_worker_manager(void *shared, void *in);
static void *worker_pipeline(void *shared, int step, void *in)
{
	int i, j, k;
    pipeline_t *p = (pipeline_t*)shared;
    if (step == 0) { // step 0: read sequences
		int with_qual = (!!(p->opt->flag & MM_F_OUT_SAM) && !(p->opt->flag & MM_F_NO_QUAL));
		int with_comment = !!(p->opt->flag & MM_F_COPY_COMMENT);
		int frag_mode = (p->n_fp > 1 || !!(p->opt->flag & MM_F_FRAG_MODE));
        step_t *s;
        s = (step_t*)calloc(1, sizeof(step_t));
		if (p->n_fp > 1) s->seq = mm_bseq_read_frag2(p->n_fp, p->fp, p->mini_batch_size, with_qual, with_comment, &s->n_seq);
		else s->seq = mm_bseq_read3(p->fp[0], p->mini_batch_size, with_qual, with_comment, frag_mode, &s->n_seq);
		if (s->seq) {
			s->p = p;
			for (i = 0; i < s->n_seq; ++i)
				s->seq[i].rid = p->n_processed++;
			s->buf = (mm_tbuf_t**)calloc(p->n_threads, sizeof(mm_tbuf_t*));
			for (i = 0; i < p->n_threads; ++i)
				s->buf[i] = mm_tbuf_init();
#if defined(__AMD_SPLIT_KERNELS__)
			s->trbuf = (mm_trbuf_t**)calloc(p->n_threads, sizeof(mm_trbuf_t*));
#endif

			s->n_reg = (int*)calloc(5 * s->n_seq, sizeof(int));
			s->seg_off = s->n_reg + s->n_seq; // seg_off, n_seg, rep_len and frag_gap are allocated together with n_reg
			s->n_seg = s->seg_off + s->n_seq;
			s->rep_len = s->n_seg + s->n_seq;
			s->frag_gap = s->rep_len + s->n_seq;
			s->reg = (mm_reg1_t**)calloc(s->n_seq, sizeof(mm_reg1_t*));
			for (i = 1, j = 0; i <= s->n_seq; ++i)
				if (i == s->n_seq || !frag_mode || !mm_qname_same(s->seq[i-1].name, s->seq[i].name)) {
					s->n_seg[s->n_frag] = i - j;
					s->seg_off[s->n_frag++] = j;
					j = i;
				}
			return s;
		} else free(s);
    } else if (step == 1) { // step 1: map
#if defined(__AMD_SPLIT_KERNELS__)
		return kt_worker_manager(shared, in);
#endif
    } else if (step == 2) { // step 2: output
		void *km = 0;
		step_t *s = (step_t*)in;
		const mm_idx_t *mi = p->mi;
		// consolidate timers from threads
		mm_consolidate_timers (s, p);
		for (i = 0; i < p->n_threads; ++i) mm_tbuf_destroy(s->buf[i]);
		free(s->buf);

		if ((p->opt->flag & MM_F_OUT_CS) && !(mm_dbg_flag & MM_DBG_NO_KALLOC)) km = km_init();
		for (k = 0; k < s->n_frag; ++k) {
			int seg_st = s->seg_off[k], seg_en = s->seg_off[k] + s->n_seg[k];
			for (i = seg_st; i < seg_en; ++i) {
				mm_bseq1_t *t = &s->seq[i];
				if (p->opt->split_prefix && p->n_parts == 0) { // then write to temporary files
					mm_err_fwrite(&s->n_reg[i],    sizeof(int), 1, p->fp_split);
					mm_err_fwrite(&s->rep_len[i],  sizeof(int), 1, p->fp_split);
					mm_err_fwrite(&s->frag_gap[i], sizeof(int), 1, p->fp_split);
					for (j = 0; j < s->n_reg[i]; ++j) {
						mm_reg1_t *r = &s->reg[i][j];
						mm_err_fwrite(r, sizeof(mm_reg1_t), 1, p->fp_split);
						if (p->opt->flag & MM_F_CIGAR) {
							mm_err_fwrite(&r->p->capacity, 4, 1, p->fp_split);
							mm_err_fwrite(r->p, r->p->capacity, 4, p->fp_split);
						}
					}
				} else if (s->n_reg[i] > 0) { // the query has at least one hit
					for (j = 0; j < s->n_reg[i]; ++j) {
						mm_reg1_t *r = &s->reg[i][j];
						assert(!r->sam_pri || r->id == r->parent);
						if ((p->opt->flag & MM_F_NO_PRINT_2ND) && r->id != r->parent)
							continue;
						if (p->opt->flag & MM_F_OUT_SAM)
							mm_write_sam3(&p->str, mi, t, i - seg_st, j, s->n_seg[k], &s->n_reg[seg_st], (const mm_reg1_t*const*)&s->reg[seg_st], km, p->opt->flag, s->rep_len[i]);
						else
							mm_write_paf3(&p->str, mi, t, r, km, p->opt->flag, s->rep_len[i]);
						mm_err_puts(p->str.s);
					}
				} else if ((p->opt->flag & MM_F_PAF_NO_HIT) || ((p->opt->flag & MM_F_OUT_SAM) && !(p->opt->flag & MM_F_SAM_HIT_ONLY))) { // output an empty hit, if requested
					if (p->opt->flag & MM_F_OUT_SAM)
						mm_write_sam3(&p->str, mi, t, i - seg_st, -1, s->n_seg[k], &s->n_reg[seg_st], (const mm_reg1_t*const*)&s->reg[seg_st], km, p->opt->flag, s->rep_len[i]);
					else
						mm_write_paf3(&p->str, mi, t, 0, 0, p->opt->flag, s->rep_len[i]);
					mm_err_puts(p->str.s);
				}
			}
			for (i = seg_st; i < seg_en; ++i) {
				for (j = 0; j < s->n_reg[i]; ++j) free(s->reg[i][j].p);
				free(s->reg[i]);
				free(s->seq[i].seq); free(s->seq[i].name);
				if (s->seq[i].qual) free(s->seq[i].qual);
				if (s->seq[i].comment) free(s->seq[i].comment);
			}
		}
		free(s->reg); free(s->n_reg); free(s->seq); // seg_off, n_seg, rep_len and frag_gap were allocated with reg; no memory leak here
		km_destroy(km);
		if (mm_verbose >= 3)
			fprintf(stderr, "[M::%s::%.3f*%.2f] mapped %d sequences\n", __func__, realtime() - mm_realtime0, cputime() / (realtime() - mm_realtime0), s->n_seq);
		free(s);
	}
    return 0;
}

static mm_bseq_file_t **open_bseqs(int n, const char **fn)
{
	mm_bseq_file_t **fp;
	int i, j;
	fp = (mm_bseq_file_t**)calloc(n, sizeof(mm_bseq_file_t*));
	for (i = 0; i < n; ++i) {
		if ((fp[i] = mm_bseq_open(fn[i])) == 0) {
			if (mm_verbose >= 1)
				fprintf(stderr, "ERROR: failed to open file '%s': %s\n", fn[i], strerror(errno));
			for (j = 0; j < i; ++j)
				mm_bseq_close(fp[j]);
			free(fp);
			return 0;
		}
	}
	return fp;
}

int mm_map_file_frag(const mm_idx_t *idx, int n_segs, const char **fn, const mm_mapopt_t *opt, int n_threads)
{
	int i, pl_threads;
	pipeline_t pl;
	if (n_segs < 1) return -1;
	memset(&pl, 0, sizeof(pipeline_t));
	pl.n_fp = n_segs;
	pl.fp = open_bseqs(pl.n_fp, fn);
	if (pl.fp == 0) return -1;
	pl.opt = opt, pl.mi = idx;
	pl.n_threads = n_threads > 1? n_threads : 1;
	pl.mini_batch_size = opt->mini_batch_size;
	if (opt->split_prefix)
		pl.fp_split = mm_split_init(opt->split_prefix, idx);
	pl_threads = n_threads == 1? 1 : (opt->flag&MM_F_2_IO_THREADS)? 3 : 2;
	kt_pipeline(pl_threads, worker_pipeline, &pl, 3);

	free(pl.str.s);
	if (pl.fp_split) fclose(pl.fp_split);
	for (i = 0; i < pl.n_fp; ++i)
		mm_bseq_close(pl.fp[i]);
	free(pl.fp);
	return 0;
}

int mm_map_file(const mm_idx_t *idx, const char *fn, const mm_mapopt_t *opt, int n_threads)
{
	return mm_map_file_frag(idx, 1, &fn, opt, n_threads);
}

int mm_split_merge(int n_segs, const char **fn, const mm_mapopt_t *opt, int n_split_idx)
{
	int i;
	pipeline_t pl;
	mm_idx_t *mi;
	if (n_segs < 1 || n_split_idx < 1) return -1;
	memset(&pl, 0, sizeof(pipeline_t));
	pl.n_fp = n_segs;
	pl.fp = open_bseqs(pl.n_fp, fn);
	if (pl.fp == 0) return -1;
	pl.opt = opt;
	pl.mini_batch_size = opt->mini_batch_size;

	pl.n_parts = n_split_idx;
	pl.fp_parts  = CALLOC(FILE*, pl.n_parts);
	pl.rid_shift = CALLOC(uint32_t, pl.n_parts);
	pl.mi = mi = mm_split_merge_prep(opt->split_prefix, n_split_idx, pl.fp_parts, pl.rid_shift);
	if (pl.mi == 0) {
		free(pl.fp_parts);
		free(pl.rid_shift);
		return -1;
	}
	for (i = n_split_idx - 1; i > 0; --i)
		pl.rid_shift[i] = pl.rid_shift[i - 1];
	for (pl.rid_shift[0] = 0, i = 1; i < n_split_idx; ++i)
		pl.rid_shift[i] += pl.rid_shift[i - 1];
	if (opt->flag & MM_F_OUT_SAM)
		for (i = 0; i < (int32_t)pl.mi->n_seq; ++i)
			printf("@SQ\tSN:%s\tLN:%d\n", pl.mi->seq[i].name, pl.mi->seq[i].len);

	kt_pipeline(2, worker_pipeline, &pl, 3);

	free(pl.str.s);
	mm_idx_destroy(mi);
	free(pl.rid_shift);
	for (i = 0; i < n_split_idx; ++i)
		fclose(pl.fp_parts[i]);
	free(pl.fp_parts);
	for (i = 0; i < pl.n_fp; ++i)
		mm_bseq_close(pl.fp[i]);
	free(pl.fp);
	mm_split_rm_tmp(opt->split_prefix, n_split_idx);
	return 0;
}






/*********************************GPU Wrapper Func**********************/
#if defined(__AMD_SPLIT_KERNELS__)

void gpu_align_batch_execute(const mm_mapopt_t *opt, gpu_align_task_t *tasks, int n_tasks,
                                   uint8_t *seq_buffer, uint32_t *cigar_buffer, int stream_id);							   
extern void mm_align1(void *km, const mm_mapopt_t *opt, const mm_idx_t *mi,
                      int qlen, uint8_t *qseq0[2], mm_reg1_t *r, mm_reg1_t *r2,
                      int n_a, mm128_t *a, ksw_extz_t *ez, int splice_flag);
extern int mm_align1_inv(void *km, const mm_mapopt_t *opt, const mm_idx_t *mi,
                         int qlen, uint8_t *qseq0[2], const mm_reg1_t *r1,
                         const mm_reg1_t *r2, mm_reg1_t *r_inv, ksw_extz_t *ez);
extern void mm_align1_batched(gpu_align_batch_t *gpu_batch, void *km,
                             const mm_mapopt_t *opt, const mm_idx_t *mi, 
                             int qlen, uint8_t *qseq0[2], mm_reg1_t *r, mm_reg1_t *r2,
                             int n_a, mm128_t *a, int read_idx, int reg_idx);
extern void mm_append_cigar(mm_reg1_t *r, uint32_t n_cigar, uint32_t *cigar);
extern void mm_update_extra(mm_reg1_t *r, const uint8_t *qseq, const uint8_t *tseq, const int8_t *mat, int8_t q, int8_t e, int is_eqx, int log_gap);
extern void ksw_gen_simple_mat(int m, int8_t *mat, int8_t a, int8_t b, int8_t sc_ambi);

static gpu_align_batch_t* gpu_align_batch_init(int n_reads, void *km)
{
    gpu_align_batch_t *gpu_batch = (gpu_align_batch_t*)kcalloc(km, 1, sizeof(gpu_align_batch_t));
    
    // Conservative estimates for task and buffer requirements
    int estimated_tasks = n_reads * 5000; // ~200 tasks per read on average
	size_t estimated_seq_size = (size_t)n_reads * (4ULL * 1024ULL * 1024ULL); // ~4MB sequences per read
	size_t estimated_cigar_bytes = (size_t)n_reads * (4ULL * 1024ULL * 1024ULL); // ~4MB CIGAR per read (in bytes)
    size_t estimated_cigar_size = estimated_cigar_bytes / sizeof(uint32_t); // Convert to uint32_t count

    gpu_batch->max_tasks = estimated_tasks;
    gpu_batch->tasks = (gpu_align_task_t*)kcalloc(km, estimated_tasks, sizeof(gpu_align_task_t));
    
    gpu_batch->seq_buffer_size = estimated_seq_size;
    gpu_batch->seq_buffer = (uint8_t*)kmalloc(km, estimated_seq_size);
    
    gpu_batch->cigar_buffer_size = estimated_cigar_size;  // Now in uint32_t count, not bytes!
    gpu_batch->cigar_buffer = (uint32_t*)kmalloc(km, estimated_cigar_bytes); 
	
    gpu_batch->n_reads = n_reads;
    gpu_batch->read_ctxs = (read_align_ctx_t*)kcalloc(km, n_reads, sizeof(read_align_ctx_t));

	gpu_batch->n_tasks = 0;
    gpu_batch->seq_buffer_used = 0;
    gpu_batch->cigar_buffer_used = 0;
    
    return gpu_batch;
}

// sorting by read_idx、reg_idx、task_type、task_sub_idx
static int task_compare(const void *a, const void *b) {
    const gpu_align_task_t *ta = (const gpu_align_task_t*)a;
    const gpu_align_task_t *tb = (const gpu_align_task_t*)b;
    
    if (ta->read_idx != tb->read_idx) 
        return ta->read_idx - tb->read_idx;
    if (ta->reg_idx != tb->reg_idx) 
        return ta->reg_idx - tb->reg_idx;
    if (ta->task_type != tb->task_type) 
        return ta->task_type - tb->task_type;
    return ta->task_sub_idx - tb->task_sub_idx;
}

static void gpu_batch_process_results(gpu_align_batch_t *gpu_batch,
                                       const mm_mapopt_t *opt,
                                       const mm_idx_t *mi, void *km)
{
    nvtxRangePushA("gpu_batch_process_results");
    if (gpu_batch->n_tasks == 0) { nvtxRangePop(); return; }
   
    qsort(gpu_batch->tasks, gpu_batch->n_tasks, 
          sizeof(gpu_align_task_t), task_compare);
    
    int current_read = -1, current_reg = -1;
    read_align_ctx_t *ctx = NULL;
    mm_reg1_t *r = NULL;
    int32_t rs1 = 0, qs1 = 0, re1 = 0, qe1 = 0;
    int qlen = 0;
    int dropped = 0;
    int8_t mat[25];
    int initialized = 0;
    
    ksw_gen_simple_mat(5, mat, opt->a, opt->b, opt->sc_ambi);
    
    for (int i = 0; i < gpu_batch->n_tasks; i++) {
        gpu_align_task_t *task = &gpu_batch->tasks[i];

		//int has_valid_alignment = (task->n_cigar > 0 && task->max_q >= 0 && task->max_t >= 0);
		int has_valid_alignment = (task->n_cigar > 0);
        
        // 切换到新region
        if (task->read_idx != current_read || task->reg_idx != current_reg) {
            current_read = task->read_idx;
            current_reg = task->reg_idx;
            if (current_read < 0 || current_read >= gpu_batch->n_reads) {
                fprintf(stderr, "[BUG] task[%d]: read_idx=%d OOB (n_reads=%d), skipping\n",
                        i, current_read, gpu_batch->n_reads);
                dropped = 1; continue;
            }
            ctx = &gpu_batch->read_ctxs[current_read];
            if (current_reg < 0 || current_reg >= ctx->n_regs) {
                fprintf(stderr, "[BUG] task[%d]: reg_idx=%d OOB (n_regs=%d, read=%d), skipping\n",
                        i, current_reg, ctx->n_regs, current_read);
                dropped = 1; continue;
            }
            r = &ctx->regs0[current_reg];
            
            // 获取query长度
             qlen = ctx->qlen; 
            
            dropped = 0;
            initialized = 0;
            
            // 确保r->p已分配
            if (!r->p) {
                uint32_t capacity = sizeof(mm_extra_t)/4 + 100;
                kroundup32(capacity);
                r->p = (mm_extra_t*)calloc(capacity, 4);
                r->p->capacity = capacity;
                r->p->n_cigar = 0;
            }
        }
        
        if (dropped) continue;
        
        // 第一次遇到这个region的任务，初始化边界
        if (!initialized) {
            // 从task_ctx获取seed的起点位置
            if (task->task_type == GPU_TASK_LEFT_EXT) {
                // 左扩展任务的ref_rs/ref_qs是seed的起点
                rs1 = task->task_ctx.ref_rs;
                qs1 = task->task_ctx.ref_qs;
                re1 = rs1;
                qe1 = qs1;
            } else if (task->task_type == GPU_TASK_GAP_FILL && task->task_sub_idx == 1) {
                // 第一个gap填充任务的ref_rs/ref_qs是seed的起点
                rs1 = task->task_ctx.ref_rs;
                qs1 = task->task_ctx.ref_qs;
                re1 = rs1;
                qe1 = qs1;
            } else {
                // 如果没有左扩展也没有gap填充，可能是只有右扩展
                rs1 = task->task_ctx.ref_rs;
                qs1 = task->task_ctx.ref_qs;
                re1 = rs1;
                qe1 = qs1;
            }
            initialized = 1;
        }
        
        // 处理CIGAR（左扩展需要反向）
        if (has_valid_alignment && task->n_cigar > 0) {
            uint32_t *cigar = gpu_batch->cigar_buffer + task->cigar_offset;

            // KSW_EZ_REV_CIGAR flag (used by LEFT_EXT) makes the kernel
            // skip its internal CIGAR reversal, so the CIGAR is already
            // in the correct appending order.  Do NOT reverse it again.
            mm_append_cigar(r, task->n_cigar, cigar);
			// For GAP_FILL tasks: if alignment terminated early, add CIGAR ops to cover unaligned region
            if (task->task_type == GPU_TASK_GAP_FILL && has_valid_alignment) {
                int aligned_qlen = task->max_q + 1;
                int aligned_tlen = task->max_t + 1;
                int expected_qlen = task->task_ctx.ref_qe - task->task_ctx.ref_qs;
                int expected_tlen = task->task_ctx.ref_re - task->task_ctx.ref_rs;

                if (aligned_tlen < expected_tlen || aligned_qlen < expected_qlen) {
                    // Alignment terminated early - need to fill the gap
                    int remaining_qlen = expected_qlen - aligned_qlen;
                    int remaining_tlen = expected_tlen - aligned_tlen;

                    // Add CIGAR operations to cover the remaining region
                    if (remaining_qlen > 0 && remaining_tlen > 0) {
                        // Both query and target have remaining bases
                        int min_len = remaining_qlen < remaining_tlen ? remaining_qlen : remaining_tlen;
                        uint32_t match_op = min_len << 4 | MM_CIGAR_MATCH;
                        mm_append_cigar(r, 1, &match_op);

                        if (remaining_tlen > remaining_qlen) {
                            uint32_t del_op = (remaining_tlen - remaining_qlen) << 4 | MM_CIGAR_DEL;
                            mm_append_cigar(r, 1, &del_op);
                        } else if (remaining_qlen > remaining_tlen) {
                            uint32_t ins_op = (remaining_qlen - remaining_tlen) << 4 | MM_CIGAR_INS;
                            mm_append_cigar(r, 1, &ins_op);
                        }
                    } else if (remaining_tlen > 0) {
                        // Only target has remaining bases - add deletion
                        uint32_t del_op = remaining_tlen << 4 | MM_CIGAR_DEL;
                        mm_append_cigar(r, 1, &del_op);
                    } else if (remaining_qlen > 0) {
                        // Only query has remaining bases - add insertion
                        uint32_t ins_op = remaining_qlen << 4 | MM_CIGAR_INS;
                        mm_append_cigar(r, 1, &ins_op);
                    }
                }
            }
        }
        
        // 累积score
        if (has_valid_alignment && r->p && task->score > 0) {
			int old_score = r->p->dp_score;
            r->p->dp_score += task->score;
        } 
        
        // 根据任务类型更新边界
        switch (task->task_type) {
            case GPU_TASK_LEFT_EXT:
                // 左扩展：向前（向起点方向）扩展，更新rs1/qs1
               // 只有在有有效对齐结果时才更新坐标
                if (has_valid_alignment) {
                    if (task->reach_end) {
                        // 扩展到了对齐区域的起点
                        rs1 = task->task_ctx.rs0;
                        qs1 = task->task_ctx.qs0;
                    } else {
                        // 部分扩展
                        rs1 = task->task_ctx.ref_rs - (task->max_t + 1);
                        qs1 = task->task_ctx.ref_qs - (task->max_q + 1);
                        if (rs1 < 0 || qs1 < 0) {
                            fprintf(stderr, "[BUG] LEFT_EXT underflow: rs1=%d qs1=%d (ref_rs=%d max_t=%d ref_qs=%d max_q=%d) task[%d] read=%d reg=%d\n",
                                    rs1, qs1, task->task_ctx.ref_rs, task->max_t, task->task_ctx.ref_qs, task->max_q, i, current_read, current_reg);
                        }
                    }
                }
                // re1/qe1保持为seed的起点位置（不变）
                break;
                
            case GPU_TASK_GAP_FILL:
                // Gap填充：更新re1/qe1到gap的终点
                if (has_valid_alignment && task->zdropped) {
                    // Z-drop truncation: update coordinates to drop point
                    re1 = task->task_ctx.ref_rs + (task->max_t + 1);
                    qe1 = task->task_ctx.ref_qs + (task->max_q + 1);
                    dropped = 1;

                    // Split remaining anchors into r2 (matches CPU mm_align1 logic).
                    // Note: the GPU always aligns with opt->zdrop, skipping the CPU
                    // two-pass approach (APPROX_MAX + mm_test_zdrop) that distinguishes
                    // normal z-drop (zdrop_code=1) from inversion z-drop (zdrop_code=2).
                    // Therefore r2->split_inv is never set here; inversion detection
                    // will happen when the CPU fallback re-aligns r2 via mm_align1.
                    {
                        int as1 = task->task_ctx.as1;
                        int cnt1 = task->task_ctx.cnt1;
                        int gap_i = task->task_sub_idx;  // gap fill anchor index within as1..as1+cnt1
                        int rs_gap = task->task_ctx.ref_rs;
                        int j;
                        // Find last anchor before the z-drop point
                        for (j = gap_i - 1; j >= 0; --j)
                            if ((int32_t)ctx->a[as1 + j].x <= rs_gap + task->max_t)
                                break;
                        if (j < 0) j = 0;
                        if (cnt1 - (j + 1) >= opt->min_cnt) {
                            mm_reg1_t r2;
                            memset(&r2, 0, sizeof(mm_reg1_t));
                            mm_split_reg(r, &r2, as1 + j + 1 - r->as, qlen, ctx->a, !!(opt->flag & MM_F_QSTRAND));
                            if (r2.cnt > 0) {
                                ctx->regs0 = mm_insert_reg(&r2, current_reg, &ctx->n_regs, ctx->regs0);
                                // Update r pointer since realloc may have moved the array
                                r = &ctx->regs0[current_reg];
                            }
                        }
                    }
				} else if (!has_valid_alignment) {
                    // 任务失败：需要在CIGAR中添加操作来表示整个gap
                    // 计算gap大小
                    int gap_qlen = task->task_ctx.ref_qe - task->task_ctx.ref_qs;
                    int gap_tlen = task->task_ctx.ref_re - task->task_ctx.ref_rs;

                    // 添加CIGAR操作来表示gap
                    // 需要同时消耗query和reference碱基
                    if (gap_qlen > 0 && gap_tlen > 0) {
                        // 都大于0：先添加min个M，然后添加差值的I或D
                        int min_len = gap_qlen < gap_tlen ? gap_qlen : gap_tlen;
                        uint32_t match_op = min_len << 4 | 0; // M操作
                        mm_append_cigar(r, 1, &match_op);

                        if (gap_tlen > gap_qlen) {
                            uint32_t del_op = (gap_tlen - gap_qlen) << 4 | 2; // D操作
                            mm_append_cigar(r, 1, &del_op);
                        } else if (gap_qlen > gap_tlen) {
                            uint32_t ins_op = (gap_qlen - gap_tlen) << 4 | 1; // I操作
                            mm_append_cigar(r, 1, &ins_op);
                        }
                    } else if (gap_qlen > 0) {
                        // 只有query碱基：添加insertion
                        uint32_t ins_op = gap_qlen << 4 | 1;
                        mm_append_cigar(r, 1, &ins_op);
                    } else if (gap_tlen > 0) {
                        // 只有reference碱基：添加deletion
                        uint32_t del_op = gap_tlen << 4 | 2;
                        mm_append_cigar(r, 1, &del_op);
                    }

                    // 更新坐标到gap的终点
                    re1 = task->task_ctx.ref_re;
                    qe1 = task->task_ctx.ref_qe;
                } else {
					// Normal GAP_FILL: update to end of gap
                    // Note: We add filler CIGAR ops for early termination above,
                    // so we can always use ref_re/ref_qe here
                    re1 = task->task_ctx.ref_re;
                    qe1 = task->task_ctx.ref_qe;	
                }
                // rs1/qs1保持不变（已经由左扩展或初始化确定）
                break;
                
            case GPU_TASK_RIGHT_EXT:
                // 右扩展：向后（向终点方向）扩展，更新re1/qe1
				 // 只有在有有效对齐结果时才更新坐标
                if (has_valid_alignment) {
                    if (task->reach_end) {
                        // 扩展到了对齐区域的终点
                        re1 = task->task_ctx.re0;
                        qe1 = task->task_ctx.qe0;
                    } else {
                        // 部分扩展：从ref_rs/ref_qs（起始位置）对齐了max_t/max_q个字符
                        re1 = task->task_ctx.ref_rs + (task->max_t + 1);
                        qe1 = task->task_ctx.ref_qs + (task->max_q + 1);
                    }
                }
                // rs1/qs1保持不变
                break;
        }
        
        // 检查是否是当前region的最后一个任务
        int is_last_task = (i == gpu_batch->n_tasks - 1) ||
                          (gpu_batch->tasks[i+1].read_idx != current_read) ||
                          (gpu_batch->tasks[i+1].reg_idx != current_reg);

        if (is_last_task) {
            // Set final boundaries (even for dropped regions — they still need
            // valid coordinates for the truncated alignment, matching CPU logic)
            r->rs = rs1;
            r->re = re1;

            int rev = task->task_ctx.rev;
            if (!rev || (opt->flag & MM_F_QSTRAND)) {
                r->qs = qs1;
                r->qe = qe1;
            } else {
                r->qs = qlen - qe1;
                r->qe = qlen - qs1;
            }

            if (!dropped) {
                // Update alignment statistics
                // P2/P3 optimisation: GPU already ran fix_cigar+stats for most tasks.
                // Use GPU results unless leading I/D remains (pass 3b needed) or EQX mode.
                if (r->p && r->p->n_cigar > 0) {
                    int use_gpu_stats = task->gpu_stats_valid &&
                                        !(opt->flag & MM_F_EQX);
                    if (use_gpu_stats) {
                        // GPU fix_cigar (passes 1/2/3a) already applied to CIGAR in-place.
                        // Precision note: dp_max may differ +/-1 vs CPU (integer vs float log2).
                        r->blen       = task->blen;
                        r->mlen       = task->mlen;
                        r->p->n_ambi  = task->n_ambi;
                        r->p->dp_max  = task->dp_max;
                        if (rev && r->p->trans_strand) r->p->trans_strand ^= 3;
                    } else {
                        // CPU fallback: leading-I/D coordinate fix (pass 3b) or EQX mode.
                        if (qs1 < 0 || qs1 >= qlen || rs1 < 0 || re1 <= rs1) {
                            fprintf(stderr, "[BUG] mm_update_extra bounds: qs1=%d qe1=%d qlen=%d rs1=%d re1=%d read=%d reg=%d task[%d] type=%d\n",
                                    qs1, qe1, qlen, rs1, re1, current_read, current_reg, i, task->task_type);
                        } else {
                        // DEBUG: validate CIGAR vs sequence lengths before mm_update_extra
                        {
                            static int cigar_mismatch_count = 0;
                            int cigar_qlen = 0, cigar_tlen = 0;
                            int has_zero_len = 0;
                            for (int ci = 0; ci < (int)r->p->n_cigar; ci++) {
                                uint32_t op = r->p->cigar[ci] & 0xf;
                                int len = r->p->cigar[ci] >> 4;
                                if (len == 0) has_zero_len = 1;
                                if (op == 0) { cigar_qlen += len; cigar_tlen += len; }
                                else if (op == 1) { cigar_qlen += len; }
                                else if (op == 2) { cigar_tlen += len; }
                                else if (op == 3) { cigar_tlen += len; }
                            }
                            int exp_qlen = qe1 - qs1;
                            int exp_tlen = re1 - rs1;
                            if (cigar_qlen != exp_qlen || cigar_tlen != exp_tlen) {
                                cigar_mismatch_count++;
                                if (cigar_mismatch_count <= 3) {
                                    fprintf(stderr, "[DEBUG] CIGAR mismatch #%d: cq=%d ct=%d eq=%d et=%d "
                                            "n_cigar=%d task[%d] type=%d score=%d zlen=%d\n",
                                            cigar_mismatch_count, cigar_qlen, cigar_tlen, exp_qlen, exp_tlen,
                                            r->p->n_cigar, i, task->task_type, task->score, has_zero_len);
                                    for (int ci = 0; ci < (int)r->p->n_cigar && ci < 6; ci++)
                                        fprintf(stderr, "  [%d]=%d%c", ci, r->p->cigar[ci]>>4, "MIDNSHP=X"[r->p->cigar[ci]&0xf]);
                                    fprintf(stderr, "\n");
                                } else if (cigar_mismatch_count == 4) {
                                    fprintf(stderr, "[DEBUG] suppressing further CIGAR mismatch details (count=%d)\n", cigar_mismatch_count);
                                }
                                goto skip_update_extra;
                            }
                        }
                        uint8_t *qseq;
                        if (!rev || (opt->flag & MM_F_QSTRAND)) {
                            qseq = ctx->qseq0[0] + qs1;
                        } else {
                            qseq = ctx->qseq0[1] + qs1;
                        }
                        uint8_t *tseq = (uint8_t*)kmalloc(km, re1 - rs1);
                        mm_idx_getseq(mi, task->task_ctx.rid, rs1, re1, tseq);
                        mm_update_extra(r, qseq, tseq, mat, opt->q, opt->e,
                                       opt->flag & MM_F_EQX, !(opt->flag & MM_F_SR));
                        if (rev && r->p->trans_strand) r->p->trans_strand ^= 3;
                        kfree(km, tseq);
                        skip_update_extra:;
                        }  // end bounds-check else
                    }
                }
            } else {
                // Dropped region (z-drop truncated): run mm_update_extra on
                // the truncated CIGAR so blen/mlen/dp_max are set correctly.
                if (r->p && r->p->n_cigar > 0 && re1 > rs1 && qs1 >= 0 && qs1 < qlen) {
                    // DEBUG: validate CIGAR for dropped region
                    {
                        static int drop_mismatch_count = 0;
                        int cigar_qlen = 0, cigar_tlen = 0;
                        for (int ci = 0; ci < (int)r->p->n_cigar; ci++) {
                            uint32_t op = r->p->cigar[ci] & 0xf;
                            int len = r->p->cigar[ci] >> 4;
                            if (op == 0) { cigar_qlen += len; cigar_tlen += len; }
                            else if (op == 1) { cigar_qlen += len; }
                            else if (op == 2 || op == 3) { cigar_tlen += len; }
                        }
                        int exp_qlen = qe1 - qs1;
                        int exp_tlen = re1 - rs1;
                        if (cigar_qlen != exp_qlen || cigar_tlen != exp_tlen) {
                            if (++drop_mismatch_count <= 3)
                                fprintf(stderr, "[DEBUG] DROPPED mismatch #%d: cq=%d ct=%d eq=%d et=%d task[%d]\n",
                                        drop_mismatch_count, cigar_qlen, cigar_tlen, exp_qlen, exp_tlen, i);
                            goto skip_dropped_update;
                        }
                    }
                    uint8_t *qseq;
                    if (!rev || (opt->flag & MM_F_QSTRAND))
                        qseq = ctx->qseq0[0] + qs1;
                    else
                        qseq = ctx->qseq0[1] + qs1;
                    uint8_t *tseq = (uint8_t*)kmalloc(km, re1 - rs1);
                    mm_idx_getseq(mi, task->task_ctx.rid, rs1, re1, tseq);
                    mm_update_extra(r, qseq, tseq, mat, opt->q, opt->e,
                                   opt->flag & MM_F_EQX, !(opt->flag & MM_F_SR));
                    if (rev && r->p->trans_strand) r->p->trans_strand ^= 3;
                    kfree(km, tseq);
                    skip_dropped_update:;
                }
            }
        }
    }
    nvtxRangePop();
}

/* Forward declarations for align.c functions that lack header declarations */
void ksw_gen_simple_mat(int m, int8_t *mat, int8_t a, int8_t b, int8_t sc_ambi);
void mm_append_cigar(mm_reg1_t *r, uint32_t n_cigar, uint32_t *cigar);



static void gpu_batch_submit_and_process(const mm_mapopt_t *opt, gpu_align_batch_t *gpu_batch, const mm_idx_t *mi, void *km, int stream_id)
{
    if (gpu_batch->n_tasks == 0) return;

    // Submit to GPU kernel
    gpu_align_batch_execute(opt, gpu_batch->tasks, gpu_batch->n_tasks,
                           gpu_batch->seq_buffer, gpu_batch->cigar_buffer, stream_id);
    
    // Process results back to mm_reg1_t structures
    gpu_batch_process_results(gpu_batch, opt, mi, km);
}

static void pre_align_helper_gpu(const mm_idx_t *mi, const mm_mapopt_t *opt,
                                chain_read_t *read_, void *km, gpu_align_batch_t *gpu_batch, int read_idx)
{
    extern unsigned char seq_nt4_table[256];

    /* Voting reads now output b[]/u[] directly into rd->a/rd->u and flow
     * through the normal path below (mm_gen_regs → mm_align1_batched). */
    int n_segs = read_->n_seg;
    const int *qlens = read_->qlens;
    const char **seqs = read_->qseqs;
    const char *qname = read_->seq.name;
    int rep_len = read_->rep_len;
    int frag_gap = read_->frag_gap;
    int qlen_sum = read_->seq.qlen_sum;
    int *n_regs0 = &read_->n_u;
    int n_mini_pos = read_->n_mini_pos;
    uint64_t **mini_pos = &read_->mini_pos;
    uint64_t *u = read_->u;
    mm128_t *a = read_->a;

    int i, j;
    int max_chain_gap_ref = frag_gap;
    int is_sr = !!(opt->flag & MM_F_SR);
    uint32_t hash;
    mm_reg1_t *regs0;

    hash  = qname && !(opt->flag & MM_F_NO_HASH_NAME)? __ac_X31_hash_string(qname) : 0;
    hash ^= __ac_Wang_hash(qlen_sum) + __ac_Wang_hash(opt->seed);
    hash  = __ac_Wang_hash(hash);

    regs0 = mm_gen_regs(km, hash, qlen_sum, *n_regs0, u, a, !!(opt->flag&MM_F_QSTRAND));

    if (mi->n_alt) {
        mm_mark_alt(mi, *n_regs0, regs0);
        mm_hit_sort(km, n_regs0, regs0, opt->alt_drop);
    }

    if (mm_dbg_flag & (MM_DBG_PRINT_SEED|MM_DBG_PRINT_CHAIN))
        for (j = 0; j < *n_regs0; ++j)
            for (i = regs0[j].as; i < regs0[j].as + regs0[j].cnt; ++i)
                fprintf(stderr, "CN\t%d\t%s\t%d\t%c\t%d\t%d\t%d\n", j, mi->seq[a[i].x<<1>>33].name, (int32_t)a[i].x, "+-"[a[i].x>>63], (int32_t)a[i].y, (int32_t)(a[i].y>>32&0xff),
                        i == regs0[j].as? 0 : ((int32_t)a[i].y - (int32_t)a[i-1].y) - ((int32_t)a[i].x - (int32_t)a[i-1].x));

    chain_post(opt, max_chain_gap_ref, mi, km, qlen_sum, n_segs, qlens, n_regs0, regs0, a);
    if (!is_sr && !(opt->flag&MM_F_QSTRAND)) {
        mm_est_err(mi, qlen_sum, *n_regs0, regs0, a, n_mini_pos, *mini_pos);
        *n_regs0 = mm_filter_strand_retained(*n_regs0, regs0);
    }

    assert(n_segs == 1); 
    assert((opt->flag & MM_F_CIGAR));

    /*******************************
     *START: GPU replacement for mm_align_skeleton logic
     **************************************/

    int32_t skele_n_regs = *n_regs0, n_a;
    uint8_t *qseq0[2];

    qseq0[0] = (uint8_t*)kmalloc(km, qlens[0] * 2);
    qseq0[1] = qseq0[0] + qlens[0];
    for (i = 0; i < qlens[0]; ++i) {
        qseq0[0][i] = seq_nt4_table[(uint8_t)seqs[0][i]];
        qseq0[1][qlens[0] - 1 - i] = qseq0[0][i] < 4? 3 - qseq0[0][i] : 4;
    }

    n_a = mm_squeeze_a(km, skele_n_regs, regs0, a);

    // Set up read context for GPU processing
    read_align_ctx_t *ctx = &gpu_batch->read_ctxs[read_idx];
    ctx->regs0 = regs0;
    ctx->n_regs = skele_n_regs;
    ctx->qseq0[0] = qseq0[0];
    ctx->qseq0[1] = qseq0[1];
    ctx->n_a = n_a;
    ctx->a = a;
	ctx->qlen = qlens[0];  // Store the actual query length
    
    for (i = 0; i < skele_n_regs; ++i) {
        mm_reg1_t r2;
        memset(&r2, 0, sizeof(mm_reg1_t));

        // This replaces the mm_align1 call with task collection
		assert(!((opt->flag&MM_F_SPLICE) && (opt->flag&MM_F_SPLICE_FOR) && (opt->flag&MM_F_SPLICE_REV)));
        mm_align1_batched(gpu_batch, km, opt, mi, qlens[0], qseq0, &regs0[i], &r2, n_a, a, read_idx, i);
        // Note: r2.cnt is always 0 here — mm_align1_batched only collects tasks,
        // it never calls mm_split_reg.  Z-drop splits are handled during GPU
        // result processing in gpu_batch_process_results().
    }

    *n_regs0 = skele_n_regs;
    
    // Note: qseq0 cleanup and final processing will happen after GPU batch processing
}

static void post_align_helper_gpu(const mm_idx_t *mi, const mm_mapopt_t *opt,
                  chain_read_t *read_, gpu_align_batch_t *gpu_batch, void *km, int read_idx) {

	const int *qlens = read_->qlens;
	mm128_t *a = read_->a;
	uint64_t *u = read_->u;
	uint64_t *mini_pos = read_->mini_pos;
	int rep_len = read_->rep_len;
	int is_sr = !!(opt->flag & MM_F_SR);
	read_align_ctx_t *ctx = &gpu_batch->read_ctxs[read_idx];
	int *n_regs_after_align = &ctx->n_regs;
	mm_reg1_t *regs_after_align = ctx->regs0;
	if(0 == *n_regs_after_align) {
		kfree(km, a);
		kfree(km, u);
		kfree(km, mini_pos);
		return;
	}

	mm_filter_regs(opt, qlens[0], n_regs_after_align, regs_after_align);
	if (!(opt->flag&MM_F_SR) && !opt->split_prefix && qlens[0] >= opt->rank_min_len) {
	    mm_update_dp_max(qlens[0], *n_regs_after_align, regs_after_align, opt->rank_frac, opt->a, opt->b);
	    mm_filter_regs(opt, qlens[0], n_regs_after_align, regs_after_align);
	}

	mm_hit_sort(km, n_regs_after_align, regs_after_align, opt->alt_drop);

	if (!(opt->flag & MM_F_ALL_CHAINS)) { // don't choose primary mapping(s)
		mm_set_parent(km, opt->mask_level, opt->mask_len, *n_regs_after_align, regs_after_align, opt->a * 2 + opt->b, opt->flag&MM_F_HARD_MLEVEL, opt->alt_drop);
		mm_select_sub(km, opt->pri_ratio, mi->k*2, opt->best_n, 0, opt->max_gap * 0.8, n_regs_after_align, regs_after_align);
		mm_set_sam_pri(*n_regs_after_align, regs_after_align);
	}
	/*******************************
	 *END: put part of mm_align_skeletion here for simplicity.
	 * **************************************/
	regs_after_align = (mm_reg1_t*)realloc(regs_after_align, sizeof(*regs_after_align) * *n_regs_after_align);
	ctx->regs0 = regs_after_align;
	mm_set_mapq(km, *n_regs_after_align, regs_after_align, opt->min_chain_score, opt->a, rep_len, is_sr);
	kfree(km, a);
	kfree(km, u);
	kfree(km, mini_pos);

	read_->a = NULL;
	read_->u = NULL;
	read_->mini_pos = NULL;
}

static void prepare_align_batch_gpu(mm_batch_trbuf_t *batch, mm_tbuf_t *b, step_t *s, int stream_id)
{
    nvtxRangePushA("prepare_align_batch_gpu");
    gpu_align_batch_t *gpu_batch = gpu_align_batch_init(batch->count, batch->km);

    // Process each read and collect alignment tasks
    nvtxRangePushA("pre_align_helper_gpu_loop");
    for (int iread = 0; iread < batch->count; iread++) {
        pre_align_helper_gpu(s->p->mi, s->p->opt, &batch->reads[iread],
                            	batch->km, gpu_batch, iread);
    }
    nvtxRangePop();

    // Submit all tasks to GPU and process results
    nvtxRangePushA("gpu_submit_and_process");
    gpu_batch_submit_and_process(s->p->opt, gpu_batch, s->p->mi, batch->km, stream_id);
    nvtxRangePop();

    // CPU fallback: align any r2 regions created by z-drop splits during GPU
    // result processing.  These regions were inserted into ctx->regs0 by
    // mm_split_reg+mm_insert_reg but have no CIGAR yet (p==NULL).
    nvtxRangePushA("zdrop_split_cpu_fallback");
    for (int iread = 0; iread < batch->count; iread++) {
        read_align_ctx_t *ctx = &gpu_batch->read_ctxs[iread];
        for (int ireg = 0; ireg < ctx->n_regs; ireg++) {
            mm_reg1_t *reg = &ctx->regs0[ireg];
            if (reg->cnt > 0 && reg->p == NULL) {
                // Unaligned region (from z-drop split) — run CPU mm_align1
                ksw_extz_t ez;
                memset(&ez, 0, sizeof(ksw_extz_t));
                mm_reg1_t r2_cpu;
                memset(&r2_cpu, 0, sizeof(mm_reg1_t));
                mm_align1(batch->km, s->p->opt, s->p->mi, ctx->qlen,
                          ctx->qseq0, reg, &r2_cpu, ctx->n_a, ctx->a,
                          &ez, s->p->opt->flag);
                kfree(batch->km, ez.cigar);
                if (r2_cpu.cnt > 0) {
                    ctx->regs0 = mm_insert_reg(&r2_cpu, ireg, &ctx->n_regs, ctx->regs0);
                    reg = &ctx->regs0[ireg]; // realloc may have moved the array
                }
                // Handle inversion z-drop: mm_align1 sets split_inv when
                // mm_test_zdrop detects zdrop_code==2 (matches CPU mm_align_skeleton)
                if (ireg > 0 && reg->split_inv && !(s->p->opt->flag & MM_F_NO_INV)) {
                    mm_reg1_t r2_inv;
                    memset(&r2_inv, 0, sizeof(mm_reg1_t));
                    ksw_extz_t ez_inv;
                    memset(&ez_inv, 0, sizeof(ksw_extz_t));
                    if (mm_align1_inv(batch->km, s->p->opt, s->p->mi, ctx->qlen,
                                     ctx->qseq0, &ctx->regs0[ireg-1], reg, &r2_inv, &ez_inv)) {
                        ctx->regs0 = mm_insert_reg(&r2_inv, ireg, &ctx->n_regs, ctx->regs0);
                        ++ireg; // skip the inserted INV alignment
                    }
                    kfree(batch->km, ez_inv.cigar);
                }
            }
        }
    }
    nvtxRangePop();

    nvtxRangePushA("post_align_helper_gpu_loop");
	for (int iread = 0; iread < batch->count; iread++) {
        post_align_helper_gpu(s->p->mi, s->p->opt, &batch->reads[iread],
                         		gpu_batch, batch->km, iread);
    }
    nvtxRangePop();

	// After Align
    nvtxRangePushA("result_copyback");
	int pe_ori = s->p->opt->pe_ori;
	for (int iread = 0; iread < batch->count; iread++) {
		int i = batch->reads[iread].seq.i;
		int off = s->seg_off[i];
		int j = batch->reads[iread].seq.seg_id;
	    read_align_ctx_t *ctx = &gpu_batch->read_ctxs[iread];
		s->reg[off + j] =  ctx->regs0;
		s->n_reg[off + j] = ctx->n_regs;
		if (s->p->opt->flag & MM_F_INDEPEND_SEG) {
			if (s->n_seg[i] == 2 && ((j == 0 && (pe_ori >> 1 & 1)) ||
										(j == 1 && (pe_ori & 1)))) {
				int k, t;
				mm_revcomp_bseq(&s->seq[off + j]);
				for (k = 0; k < s->n_reg[off + j]; ++k) {
					mm_reg1_t *r = &s->reg[off + j][k];
					t = r->qs;
                       r->qs = batch->reads[iread].qlens[j] - r->qe;
                       r->qe = batch->reads[iread].qlens[j] - t;
                       r->rev = !r->rev;
				}
			}
		} else {
            for (j = 0; j < batch->reads[iread].n_seg; ++j) { 
				 // flip the query strand and coordinate to the
                 // original read strand
                if (s->n_seg[i] == 2 &&
					((j == 0 && (pe_ori >> 1 & 1)) ||
					(j == 1 && (pe_ori & 1)))) {
					int k, t;
					mm_revcomp_bseq(&s->seq[off + j]);
					for (k = 0; k < s->n_reg[off + j]; ++k) {
						mm_reg1_t *r = &s->reg[off + j][k];
						t = r->qs;
                           r->qs = batch->reads[iread].qlens[j] - r->qe;
                           r->qe = batch->reads[iread].qlens[j] - t;
                           r->rev = !r->rev;
					}
				}
            }
        }
		//if (mm_dbg_flag & MM_DBG_PRINT_QNAME)
			//fprintf(stderr, "QT\t%s\t%d\t%.6f\n", s->seq[off].name, tid, realtime() - t);
    }

    nvtxRangePop(); // result_copyback

	// Final cleanup for each read
    for (int iread = 0; iread < batch->count; iread++) {
        read_align_ctx_t *ctx = &gpu_batch->read_ctxs[iread];
    	kfree(batch->km, ctx->qseq0[0]);
    }
    kfree(batch->km, gpu_batch->tasks);
    kfree(batch->km, gpu_batch->seq_buffer);
    kfree(batch->km, gpu_batch->cigar_buffer);
    kfree(batch->km, gpu_batch->read_ctxs);
    kfree(batch->km, gpu_batch);
    nvtxRangePop(); // prepare_align_batch_gpu
}

static seeded_queue_t *g_seeded_queue = NULL;
static void worker_for(void *_data, long i_in, int tid) {
  	step_t *s = (step_t *)_data;
    long i = i_in;
    int j, iread, off, pe_ori = s->p->opt->pe_ori;
	mm_tbuf_t *b = s->buf[tid];
	void *km = km_init();

	if(i == -1) {
		mark_worker_finished(g_seeded_queue);
		km_destroy(km);
		return;
	}

	off  = s->seg_off[i];
	assert(s->n_seg[i] <= MM_MAX_SEG);
	if (mm_dbg_flag & MM_DBG_PRINT_QNAME) {
        fprintf(stderr, "QR\t%s\t%d\t%d\n", s->seq[off].name, tid, s->seq[off].l_seq);
    }
    
    int n_indep_reads = (s->p->opt->flag & MM_F_INDEPEND_SEG) ? s->n_seg[i] : 1;

	// Process each independent read
    for (int read_idx = 0; read_idx < n_indep_reads; read_idx++) {
        chain_read_t read;
        memset(&read, 0, sizeof(chain_read_t));
        
        if (s->p->opt->flag & MM_F_INDEPEND_SEG) {
            // Handle independent segments
            j = read_idx;
            read.qlens = (int *)kmalloc(km, sizeof(int));
            read.qseqs = (const char **)kmalloc(km, sizeof(const char *));
            
            if (s->n_seg[i] == 2 && ((j == 0 && (pe_ori>>1&1)) || (j == 1 && (pe_ori&1))))
                mm_revcomp_bseq(&s->seq[off + j]);
                
            read.qlens[0] = s->seq[off + j].l_seq;
            read.qseqs[0] = s->seq[off + j].seq;
            read.n_seg = 1;
            
            read.seq.i = i;
            read.seq.seg_id = j;
            strcpy(read.seq.name, s->seq[off + j].name);
            read.seq.n_alt = s->p->mi->n_alt;
            read.seq.is_alt = 0;
        } else {
            // Handle all segments together
            read.qlens = (int *)kmalloc(km, s->n_seg[i] * sizeof(int));
            read.qseqs = (const char **)kmalloc(km, s->n_seg[i] * sizeof(const char *));
            read.n_seg = s->n_seg[i];
            
            for (j = 0; j < s->n_seg[i]; ++j) {
                if (s->n_seg[i] == 2 && ((j == 0 && (pe_ori>>1&1)) || (j == 1 && (pe_ori&1))))
                    mm_revcomp_bseq(&s->seq[off + j]);
                read.qlens[j] = s->seq[off + j].l_seq;
                read.qseqs[j] = s->seq[off + j].seq;
            }
            
            read.seq.i = i;
            read.seq.seg_id = 0;
            strcpy(read.seq.name, s->seq[off].name);
            read.seq.n_alt = s->p->mi->n_alt;
            read.seq.is_alt = 0;
        }
        
        // Perform seeding
        mm_map_seed(s->p->mi, s->p->opt, &read, b, km);
        
        if (mm_dbg_flag & MM_DBG_PRINT_QNAME) {
            fprintf(stderr, "SEED\t%s\t%d\t%d_anchors\n", read.seq.name, tid, read.n);
        }
        
        // Store thread ID for later use
        read.thread_id = tid;
        
        // Push to global seeded queue
        push_seeded_read(g_seeded_queue, &read);
    }
	km_destroy(km);
}

// Deep-copy a chain_read_t (with its anchor data) into a batch using the batch's km.
// Used to defer CPU-fallback reads for later GPU processing.
static void deep_copy_read_to_batch(mm_batch_trbuf_t *dst, const chain_read_t *src,
                                    const mm_mapopt_t *opt)
{
    chain_read_t *r = &dst->reads[dst->count];
    *r = *src;  // shallow copy of all scalars and pointer values

    // Deep-copy arrays into dst->km so their lifetime is tied to the batch
    if (opt->flag & MM_F_INDEPEND_SEG) {
        r->qlens = (int*)kmalloc(dst->km, sizeof(int));
        r->qseqs = (const char**)kmalloc(dst->km, sizeof(const char*));
        r->qlens[0] = src->qlens[0];
        r->qseqs[0] = src->qseqs[0];
    } else {
        r->qlens = (int*)kmalloc(dst->km, sizeof(int) * src->n_seg);
        r->qseqs = (const char**)kmalloc(dst->km, sizeof(const char*) * src->n_seg);
        memcpy(r->qlens, src->qlens, sizeof(int) * src->n_seg);
        memcpy(r->qseqs, src->qseqs, sizeof(const char*) * src->n_seg);
    }
    r->mini_pos = (uint64_t*)kmalloc(dst->km, src->n_mini_pos * sizeof(uint64_t));
    memcpy(r->mini_pos, src->mini_pos, src->n_mini_pos * sizeof(uint64_t));
    r->a = (mm128_t*)kmalloc(dst->km, src->n * sizeof(mm128_t));
    memcpy(r->a, src->a, src->n * sizeof(mm128_t));

    // u/n_u are chaining outputs – not yet available on these fallback reads
    r->u    = NULL;
    r->n_u  = 0;

    dst->count++;
    dst->total_n += src->n;
}

/*
 * Multi-stream task-parallel GPU batch consumer.
 *
 * Each CUDA stream runs the full pipeline independently:
 *   chain → sync → backtrack → voting → alignment
 *
 * Parallelism: while the CPU processes stream N's results (backtrack/voting/
 * alignment), stream N+1's chain kernels run concurrently on the GPU.
 *
 * Timeline (2 streams, steady state):
 *
 *   GPU stream 0:  ████ chain(A) ████████████  bt(A)  align(A)  ████ chain(C) ████
 *   GPU stream 1:       ████ chain(B) ████████████  bt(B)  align(B)  ████ chain(D) ███
 *   Drain thread 0:                        sync(A) bt(A) vote(A) align(A)
 *   Drain thread 1:                                      sync(B) bt(B) vote(B) align(B)
 *   Main thread:    launch(A) accum launch(B) accum ... (drain threads run in parallel)
 */

// ══════════════════════════════════════════════════════════════════════
//  Dual-thread drain architecture (方案A):
//  Each CUDA stream gets its own CPU drain thread so that drain(stream0)
//  and drain(stream1) can run truly in parallel on different CPU cores,
//  each driving its own GPU stream.
// ══════════════════════════════════════════════════════════════════════

// Per-stream worker thread state
typedef struct {
    mm_batch_trbuf_t batch;
    int busy;               // 1 = chain launched, not yet collected

    // Worker thread synchronization
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t cond_work;   // main → worker: "you have work"
    pthread_cond_t cond_done;   // worker → main: "I'm done"
    int work_ready;             // 1 = batch needs draining
    int drain_done;             // 1 = drain complete
    int shutdown;               // 1 = worker should exit

    // Per-worker batch sizing (set once during init)
    int batch_max_reads;
} gpu_stream_slot_t;

// Shared context passed to each drain worker
typedef struct {
    step_t *s;
    gpu_stream_slot_t *slot;
    int stream_id;
    mm_tbuf_t *wb;  // dedicated tbuf (avoids conflict with seeding threads)
} drain_worker_ctx_t;

// Helper: copy rep_len/frag_gap from batch reads back to step arrays
// Thread-safe: each read writes to its own s->rep_len[off]/s->frag_gap[off]
// slot determined by read's seq.i — no two streams process the same read.
static void copy_rep_frag(step_t *s, mm_batch_trbuf_t *batch) {
    for (int iread = 0; iread < batch->count; iread++) {
        int i_ = batch->reads[iread].seq.i;
        int j_ = batch->reads[iread].seq.seg_id;
        int off_ = s->seg_off[i_] + j_;
        for (int k_ = 0; k_ < batch->reads[iread].n_seg; k_++) {
            s->rep_len[off_ + k_] = batch->reads[iread].rep_len;
            s->frag_gap[off_ + k_] = batch->reads[iread].frag_gap;
        }
    }
}



// Drain worker thread function.
// Each worker owns one CUDA stream and processes batches independently.
// No shared mutable state between workers.
// s->reg[], s->rep_len[] etc. are indexed per-read — no conflicts.
static void* drain_worker_fn(void *arg) {
    drain_worker_ctx_t *ctx = (drain_worker_ctx_t*)arg;
    step_t *s = ctx->s;
    gpu_stream_slot_t *slot = ctx->slot;
    int sid = ctx->stream_id;
    mm_tbuf_t *wb = ctx->wb;  // dedicated tbuf (not shared with seeding threads)

    while (1) {
        // Wait for work signal from main thread
        pthread_mutex_lock(&slot->mutex);
        while (!slot->work_ready && !slot->shutdown)
            pthread_cond_wait(&slot->cond_work, &slot->mutex);

        if (slot->shutdown) {
            pthread_mutex_unlock(&slot->mutex);
            break;
        }
        slot->work_ready = 0;
        pthread_mutex_unlock(&slot->mutex);

        // ── Drain this stream's batch ────────────────────────────────
        if (mm_dbg_flag & MM_DBG_PRINT_QNAME)
            fprintf(stderr, "DRAIN_WORKER(%d): count=%d\n", sid, slot->batch.count);

        // ── Process main batch (chain already launched by consumer) ──
        nvtxRangePushA("sync_chain_gpu");
        sync_chain_gpu(sid);
        nvtxRangePop();

        nvtxRangePushA("start_backtrack_gpu");
        int bt_n = 0;
        start_backtrack_gpu(s->p->mi, s->p->opt, slot->batch.reads,
                            sid, slot->batch.km, &bt_n);
        nvtxRangePop();
        // bt_n == batch.count: backtrack processes exactly what chain fitted.
        // Chain overflow was put back into acc_batch by the consumer.

        nvtxRangePushA("finish_backtrack_gpu");
        finish_backtrack_gpu(s->p->mi, s->p->opt, slot->batch.reads,
                             bt_n, sid, slot->batch.km);
        nvtxRangePop();

        slot->batch.count = bt_n;
        copy_rep_frag(s, &slot->batch);
        prepare_align_batch_gpu(&slot->batch, wb, s, sid);
        mm_trbuf_batch_reset(&slot->batch, slot->batch_max_reads, s->p->opt);

        // Overflow reads are handled by the consumer — put back into
        // acc_batch and included in the next dispatch.

        // Signal main thread: drain complete
        pthread_mutex_lock(&slot->mutex);
        slot->drain_done = 1;
        slot->busy = 0;
        pthread_cond_signal(&slot->cond_done);
        pthread_mutex_unlock(&slot->mutex);
    }
    return NULL;
}

static void* gpu_batch_consumer(void *data) {
    step_t *s = (step_t*)data;

    #define NUM_GPU_STREAMS 2

    #define INIT_BATCH(b_, id_) do { \
        (b_).km = km_init(); \
        (b_).count = 0; \
        (b_).total_n = 0; \
        (b_).reads = (chain_read_t*)malloc(s->batch_max_reads * sizeof(chain_read_t)); \
        memset((b_).reads, 0, s->batch_max_reads * sizeof(chain_read_t)); \
        (b_).batchid = (id_); \
    } while (0)

    // Two stream slots for pipeline overlap
    gpu_stream_slot_t slots[NUM_GPU_STREAMS];
    drain_worker_ctx_t worker_ctxs[NUM_GPU_STREAMS];
    for (int i = 0; i < NUM_GPU_STREAMS; i++) {
        memset(&slots[i], 0, sizeof(gpu_stream_slot_t));
        INIT_BATCH(slots[i].batch, i);
        slots[i].batch_max_reads = s->batch_max_reads;
        pthread_mutex_init(&slots[i].mutex, NULL);
        pthread_cond_init(&slots[i].cond_work, NULL);
        pthread_cond_init(&slots[i].cond_done, NULL);
        worker_ctxs[i].s = s;
        worker_ctxs[i].slot = &slots[i];
        worker_ctxs[i].stream_id = i;
        worker_ctxs[i].wb = mm_tbuf_init();  // dedicated tbuf per drain worker
    }

    mm_batch_trbuf_t acc_batch;
    INIT_BATCH(acc_batch, -1);
    #undef INIT_BATCH

    // Start drain worker threads (one per stream)
    for (int i = 0; i < NUM_GPU_STREAMS; i++)
        pthread_create(&slots[i].thread, NULL, drain_worker_fn, &worker_ctxs[i]);

    #define SIGNAL_DRAIN(sl_) do { \
        pthread_mutex_lock(&(sl_)->mutex); \
        (sl_)->work_ready = 1; \
        (sl_)->drain_done = 0; \
        pthread_cond_signal(&(sl_)->cond_work); \
        pthread_mutex_unlock(&(sl_)->mutex); \
    } while (0)

    #define WAIT_DRAIN(sl_) do { \
        pthread_mutex_lock(&(sl_)->mutex); \
        while (!(sl_)->drain_done) \
            pthread_cond_wait(&(sl_)->cond_done, &(sl_)->mutex); \
        (sl_)->drain_done = 0; \
        pthread_mutex_unlock(&(sl_)->mutex); \
    } while (0)

    int queue_finished = 0;
    int is_full = 0;
    int cur_stream = 0;  // round-robin stream index
    chain_read_t read;

    // ── Main loop: accumulate → dispatch to stream[cur] → rotate → repeat ──
    while (1) {
        // Step 1: Pop reads and accumulate into acc_batch
        if (!queue_finished) {
            int got_read = pop_seeded_read(g_seeded_queue, &read);
            if (got_read) {
                chain_read_t *batch_read = &acc_batch.reads[acc_batch.count];
                *batch_read = read;
                if (s->p->opt->flag & MM_F_INDEPEND_SEG) {
                    batch_read->qlens = (int*)kmalloc(acc_batch.km, sizeof(int));
                    batch_read->qseqs = (const char**)kmalloc(acc_batch.km, sizeof(const char*));
                    batch_read->qlens[0] = read.qlens[0];
                    batch_read->qseqs[0] = read.qseqs[0];
                } else {
                    batch_read->qlens = (int*)kmalloc(acc_batch.km, sizeof(int) * read.n_seg);
                    batch_read->qseqs = (const char**)kmalloc(acc_batch.km, sizeof(const char*) * read.n_seg);
                    memcpy(batch_read->qlens, read.qlens, sizeof(int) * read.n_seg);
                    memcpy(batch_read->qseqs, read.qseqs, sizeof(const char*) * read.n_seg);
                }
                batch_read->mini_pos = (uint64_t*)kmalloc(acc_batch.km, read.n_mini_pos * sizeof(uint64_t));
                batch_read->a = (mm128_t*)kmalloc(acc_batch.km, read.n * sizeof(mm128_t));
                memcpy(batch_read->mini_pos, read.mini_pos, read.n_mini_pos * sizeof(uint64_t));
                memcpy(batch_read->a, read.a, read.n * sizeof(mm128_t));
                acc_batch.count++;
                acc_batch.total_n += read.n;
                free_queue_read(&read);
                if (acc_batch.total_n >= s->batch_max_anchors) {
                    is_full = 1;
                    if (mm_dbg_flag & MM_DBG_PRINT_QNAME)
                        fprintf(stderr, "ACC_FULL: count=%d, total_n=%zu\n",
                                acc_batch.count, acc_batch.total_n);
                }
            } else {
                queue_finished = 1;
                is_full = (acc_batch.count > 0) ? 1 : 0;
                if (mm_dbg_flag & MM_DBG_PRINT_QNAME)
                    fprintf(stderr, "QUEUE_FINISHED: acc_count=%d\n", acc_batch.count);
            }
        }

        // Flush acc_batch if queue is done and we have leftover reads (e.g. overflow)
        if (queue_finished && acc_batch.count > 0)
            is_full = 1;

        // Step 2: When batch is full, dispatch to stream[cur_stream]
        if (is_full && acc_batch.count > 0) {
            gpu_stream_slot_t *sl = &slots[cur_stream];

            // If this stream is still busy, wait for its drain to finish
            if (sl->busy) {
                WAIT_DRAIN(sl);
            }

            if (mm_dbg_flag & MM_DBG_PRINT_QNAME)
                fprintf(stderr, "LAUNCH_CHAIN(stream=%d): count=%d, total_n=%zu\n",
                        cur_stream, acc_batch.count, acc_batch.total_n);

            // Launch chain on stream[cur_stream] (async)
            nvtxRangePushA("launch_chain_gpu");
            int overflow = launch_chain_gpu(acc_batch.reads, acc_batch.count, cur_stream);
            nvtxRangePop();

            int fit = acc_batch.count - overflow;

            // Safety: if nothing fits at all, these reads exceed GPU capacity
            if (fit == 0) {
                fprintf(stderr, "[WARNING] %d reads exceed GPU chain capacity (stream=%d), "
                        "skipping\n", acc_batch.count, cur_stream);
                mm_trbuf_batch_reset(&acc_batch, s->batch_max_reads, s->p->opt);
                is_full = 0;
                continue;
            }

            // Swap fitted reads into slot for draining
            mm_batch_trbuf_t tmp = sl->batch;
            sl->batch = acc_batch;
            sl->batch.count = fit;  // only fitted reads go to drain
            acc_batch = tmp;

            // Put overflow reads back into acc_batch for next round
            acc_batch.count = 0;
            acc_batch.total_n = 0;
            if (overflow > 0) {
                for (int r_ = fit; r_ < fit + overflow; r_++) {
                    deep_copy_read_to_batch(&acc_batch,
                                            &sl->batch.reads[r_], s->p->opt);
                }
                if (mm_dbg_flag & MM_DBG_PRINT_QNAME)
                    fprintf(stderr, "[INFO] %d overflow reads back to acc_batch\n",
                            overflow);
            }

            sl->busy = 1;

            // Signal this stream's drain worker (non-blocking)
            SIGNAL_DRAIN(sl);
            is_full = 0;

            // Rotate to next stream
            cur_stream = (cur_stream + 1) % NUM_GPU_STREAMS;
        }

        // Step 3: Queue done and no pending reads, drain all busy streams
        if (queue_finished && acc_batch.count == 0) {
            for (int i = 0; i < NUM_GPU_STREAMS; i++) {
                if (slots[i].busy) {
                    WAIT_DRAIN(&slots[i]);
                }
            }
            break;
        }
    }

    #undef SIGNAL_DRAIN
    #undef WAIT_DRAIN

    // Shutdown all drain workers
    for (int i = 0; i < NUM_GPU_STREAMS; i++) {
        pthread_mutex_lock(&slots[i].mutex);
        slots[i].shutdown = 1;
        pthread_cond_signal(&slots[i].cond_work);
        pthread_mutex_unlock(&slots[i].mutex);
        pthread_join(slots[i].thread, NULL);
    }

    // Cleanup
    mm_trbuf_batch_reset(&acc_batch, s->batch_max_reads, s->p->opt);
    free(acc_batch.reads);
    km_destroy(acc_batch.km);

    for (int i = 0; i < NUM_GPU_STREAMS; i++) {
        mm_trbuf_batch_reset(&slots[i].batch, s->batch_max_reads, s->p->opt);
        free(slots[i].batch.reads);
        km_destroy(slots[i].batch.km);
        pthread_mutex_destroy(&slots[i].mutex);
        pthread_cond_destroy(&slots[i].cond_work);
        pthread_cond_destroy(&slots[i].cond_done);
        mm_tbuf_destroy(worker_ctxs[i].wb);
    }

    #undef NUM_GPU_STREAMS
    return NULL;
}
static void* kt_worker_manager(void *shared, void *in) {
	pipeline_t *p = (pipeline_t*)shared;

	step_t *s = (step_t *)in;
    
    // Initialize global seeded queue if not exists
    if (g_seeded_queue == NULL) {
        int queue_capacity = 11000;
        g_seeded_queue = init_seeded_queue(queue_capacity, p->n_threads);
    }
    
    // Set batch parameters
    if (p->opt->flag & MM_F_GPU_CHAIN) {
        s->batch_max_anchors = p->opt->gpu_chain_max_anchors;  // 200M
        s->batch_max_reads = p->opt->gpu_chain_max_reads; // 200K
        s->gpu_min_n = p->opt->gpu_chain_min_n; // 512
    } else {
        s->batch_max_anchors = SIZE_MAX;
        s->batch_max_reads = N_ACCUM;
    }
    
    //Reset finished workers counter
    pthread_mutex_lock(&g_seeded_queue->mutex);
    g_seeded_queue->finished_workers = 0;
    pthread_mutex_unlock(&g_seeded_queue->mutex);
	
	for(int i = 0; i < p->n_threads; i ++) 
		s->buf[i] = mm_tbuf_init();

	if (p->n_parts > 0) merge_hits((step_t*)in);
	else kt_for_async(p->n_threads, worker_for, in, s->n_frag, 
                    gpu_batch_consumer, s);
	//else kt_for(p->n_threads, old_worker_for, in, s->n_frag);

	return in;
}

#endif