/*
 * plgrid_config.h — Gridded Traceback Configuration
 *
 * ============================================================================
 * DESIGN OVERVIEW
 * ============================================================================
 *
 * Problem: super-long alignment tasks (50kbp × 50kbp) have backtrack-pointer
 * (bt_p) stride = max_antidiag × n_col ≈ 800 MB per slot.  With a 9 GB bt_p
 * pool only ~10 concurrent slots fit, so the GPU runs at ~3% occupancy.
 *
 * Solution (gridded traceback, inspired by G3SA):
 *   - Forward DP no longer writes a per-cell direction byte.
 *   - Instead, every G antidiagonals we save a "checkpoint" snapshot of the
 *     Suzuki-Kasahara delta arrays (u, v, x, y, x2, y2) for cells [st..en].
 *   - During backtrack we walk the path antidiag-by-antidiag.  When we cross
 *     a G-block boundary we load the checkpoint, replay the forward DP for
 *     G antidiagonals, and write the per-cell direction bytes into a small
 *     per-slot scratch buffer.  Path-walk reads from this scratch.
 *
 * Memory comparison per long-task slot, 50kbp × 50kbp (n_col=20000):
 *   Original bt_p stride       :  ~800 MB    (1 byte × max_antidiag × n_col)
 *   Gridded checkpoint storage :  ~38  MB    (6 bytes × n_col × n_checkpoints)
 *   Gridded scratch (per slot) :  ~2.5 MB    (G × n_col)
 *   ─────────────────────────────────────────
 *   Reduction                  :  ~20×   →   ~120 concurrent slots possible.
 *
 * Compute cost: backtrack now requires re-running G antidiagonals per block
 * crossed.  Total replay work across the whole path equals roughly one extra
 * forward pass — i.e. backtrack-using tasks take ~2× as long, but with 20×
 * more slots running, throughput rises by ~10×.
 *
 * Precision: NO truncation.  Checkpoints store exact int8 delta state used
 * by the forward DP, so replay is bit-identical to the forward pass.  CIGAR
 * output should match the legacy path exactly (modulo tie-breaking, which
 * is also identical because we replay with the same code).
 *
 * ============================================================================
 * COMPILE-TIME SWITCH
 * ============================================================================
 *
 *   USE_GRIDDED_BT == 0  (default)   Legacy per-cell bt_p path.  No new
 *                                    buffers allocated.  Code is unchanged
 *                                    at runtime — gridded paths are #ifdef'd
 *                                    out entirely.
 *
 *   USE_GRIDDED_BT == 1              Gridded traceback path.  Forward kernel
 *                                    writes checkpoints; new backtrack kernel
 *                                    replays + walks.  bt_p is still allocated
 *                                    for fallback but unused.
 *
 * To enable: add  -DUSE_GRIDDED_BT=1  to NVCCFLAGS and recompile.
 *
 * ============================================================================
 * CHECKPOINT BUFFER LAYOUT
 * ============================================================================
 *
 * Per task:
 *   n_checkpoints  = ⌈(qlen + tlen - 1) / G⌉
 *   ckpt_stride    = 6 * n_col bytes              // u/v/x/y/x2/y2 packed
 *
 * Per slot (worst-case sizing):
 *   slot_dblock_bytes  = max_n_checkpoints × ckpt_stride
 *                      = ⌈max_antidiag / G⌉ × 6 × max_n_col
 *
 * Per slot (scratch for backtrack replay):
 *   slot_scratch_bytes = G × max_n_col
 *
 * Total VRAM:
 *   n_slots × (slot_dblock_bytes + slot_scratch_bytes)
 *
 * The host arena allocates a single contiguous block per slot containing both
 * dblock and scratch.  Slot index → slot_base via stride multiplication.
 *
 * Indexing for checkpoint write at antidiag r0 (where r0 % G == G-1):
 *   ckpt_idx       = r0 / G                         // 0-indexed block
 *   ckpt_ptr       = slot_dblock + ckpt_idx × ckpt_stride
 *   u_save[i]      = u_arr[st0 + i]                 // i ∈ [0, en0-st0]
 *   v_save[i]      = v_arr[st0 + i]   ... etc
 *
 * (off[r] and off_end[r] are still written every antidiag — they're tiny.)
 *
 * Indexing for scratch write at replayed antidiag r (within block b):
 *   local_r        = r - b × G
 *   scratch_ptr    = slot_scratch + local_r × max_n_col
 *   scratch[i]     = direction byte for cell (r, st0(r) + i)
 */

#ifndef __PLGRID_CONFIG_H__
#define __PLGRID_CONFIG_H__

/* ──────────────────────────────────────────────────────────────────────── */
/* Master compile-time switch.                                              */
/* ──────────────────────────────────────────────────────────────────────── */
#ifndef USE_GRIDDED_BT
#define USE_GRIDDED_BT 0
#endif

/* ──────────────────────────────────────────────────────────────────────── */
/* Grid block size G — number of antidiagonals between checkpoints.         */
/* Larger G  ⇒  less checkpoint memory, more replay work per backtrack.     */
/* Smaller G ⇒  more checkpoint memory, less replay.                        */
/* G=128 is the recommended starting point (10× memory reduction, ~1.3%     */
/* extra compute on backtrack-using tasks).  Must be a positive integer;    */
/* powers of two are preferred so divisions optimise to shifts.             */
/* ──────────────────────────────────────────────────────────────────────── */
#ifndef GRID_BLOCK_SIZE
#define GRID_BLOCK_SIZE 128
#endif

/* Number of int8 delta arrays saved per checkpoint cell:                   */
/*   u, v, x, y, x2, y2  →  6                                              */
#define GRID_NUM_DELTA_ARRAYS 6

/* Layout of one checkpoint in the dblock buffer (bytes per checkpoint):    */
/*   [u[len]][v[len]][x[len]][y[len]][x2[len]][y2[len]]                    */
/* where len = max_n_col (we always allocate worst-case width and pad       */
/* unused tail; saves dynamic indexing during replay).                      */
#define GRID_CKPT_STRIDE_BYTES(max_n_col) \
    ((size_t)GRID_NUM_DELTA_ARRAYS * (size_t)(max_n_col))

/* dblock bytes per slot for a task with the given antidiag count.          */
#define GRID_SLOT_DBLOCK_BYTES(max_antidiag, max_n_col) \
    (((size_t)(max_antidiag) + GRID_BLOCK_SIZE - 1) / GRID_BLOCK_SIZE \
     * GRID_CKPT_STRIDE_BYTES(max_n_col))

/* Scratch bytes per slot (one G-block of direction bytes during backtrack).*/
#define GRID_SLOT_SCRATCH_BYTES(max_n_col) \
    ((size_t)GRID_BLOCK_SIZE * (size_t)(max_n_col))

/* Total per-slot allocation for gridded traceback.                         */
#define GRID_SLOT_TOTAL_BYTES(max_antidiag, max_n_col) \
    (GRID_SLOT_DBLOCK_BYTES(max_antidiag, max_n_col) + \
     GRID_SLOT_SCRATCH_BYTES(max_n_col))

#endif  /* __PLGRID_CONFIG_H__ */
