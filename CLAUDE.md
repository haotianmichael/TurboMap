# TurboMap GPU Extension — Developer Reference

Auto-loaded at session start. Covers the Extension (chain+backtrack) GPU memory
architecture and the chain→align pipeline. Update it when the formulas or layout change.

---

## ABSOLUTE PROHIBITION — GPU EXTENSION PHASE

**禁止Extension阶段任何提到CPU FALL BACK的尝试。**
**禁止Extension阶段任何提到CPU FALL BACK的尝试。**
**禁止Extension阶段任何提到CPU FALL BACK的尝试。**

THE GPU EXTENSION MUST REPLICATE THE CPU EXTENSION LOGIC ON GPU.
DO NOT ADD CPU FALLBACK FOR THE MAIN EXTENSION DP.

### Current CPU/GPU split
- **GPU**: core DP (left/gap/right extension + backtrack + CIGAR, replicating
  `ksw_extd2_sse`), the two-pass approx→exact z-drop retry, and re-alignment of
  z-drop split remainders (fed back through `mm_align1_batched` → the same kernel,
  iterated to a fixpoint).
- **CPU (deliberate exceptions — different DP)**: `mm_test_zdrop` between the two
  GPU passes (incl. the `ksw_ll_i16` inversion probe) and inversion `mm_align1_inv`.
  A `mm_align1` net in `post_align_helper_gpu` is a backstop (no-op when GPU
  re-align succeeds).
- Splice (`ksw_exts2_sse`) is **not** implemented on GPU.

---

## Working Branch

`a100-extension-trubomap-lastShot`

---

## Arena Allocation Model

All GPU buffers for chain and align phases live in a single contiguous block
(`cudaMalloc` once, sub-allocated via bump pointer `arena_alloc`).

```
plmem_config_batch()   ──► computes max_total_n, buffer_size_long
plmem_malloc_device_mem()
    ├─ dry-run setup_chain_phase()  → chain_size
    ├─ dry-run setup_align_phase()  → align_size
    ├─ arena_size = max(chain_size, align_size) + 4 MB
    ├─ cudaMalloc(&arena_base, arena_size)      // check return code, not pointer
    └─ phase transitions re-use the same block (arena.offset = 0, re-run setup_*)
```

Phase transitions do **no** `cudaFree`/`cudaMalloc`.

---

## `plmem_config_batch` — Per-Anchor Cost Formula

File: `gpu/plmem.cu`, function `plmem_config_batch`.

| Variable | Value | Meaning |
|---|---|---|
| `chain_per_n` | 27 B | ax(4)+ay(4)+sid(1)+xrev(4)+yrev(4)+range(4)+f(4)+p(2) per anchor |
| `bt_per_n` | `mb × 82` B | backtrack arrays per anchor |
| `cub_per_n` | `mb × 16` B | CUB sort temp per anchor |
| `per_long_entry` | **23** B | long-seg buffers; **includes `d_xrev_long`** (was 19 → crash) |
| `long_ratio` | 2.0 | `buffer_size_long = long_ratio × max_total_n` |
| `mb` | usually 4 | `score_kernel_config.micro_batch` |
| `global_reserve` | 256 MB | JSON `global_vram_reserve_mb`; ≥1024 MB for cuda-gdb |

```
per_anchor_total = chain_per_n + bt_per_n + cub_per_n        // voting (mb×61) removed
total_per_n      = per_anchor_total + long_ratio*per_long_entry + overhead_per_n
budget           = (gpu_free_mem - global_reserve) / num_streams * 0.98
max_total_n      = budget / total_per_n
```

**Keep this formula in sync with `setup_chain_phase`.** Per-anchor cost is now
~465 B (was ~709 B before voting removal), so `max_total_n` is ~1.5× higher for the
same budget. `per_long_entry` = 23 must stay (the missing-`xrev` underestimate at 19
was the V100s `invalid argument` crash).

---

## `setup_chain_phase` — Arena Layout (in order)

File: `gpu/plmem.cu` ~line 184. All sizes use `anchor_per_batch = max_total_n`.

```
// Anchor arrays (N):  d_ax,d_ay(8) d_sid(1) d_xrev,d_yrev(8) d_range(4) d_f,d_p(6) = 27 B/N
// Index/cut arrays:   d_start_idx,d_read_end_idx,d_cut_start_idx (×G); d_cut (×C); d_long_seg_count
// Long-seg buffers:   (L = long_ratio×N) ax,ay,sid,range,xrev,f,p = 23 B/L (= per_long_entry)
// Backtrack arrays:   (bt_n = N×mb) in/abs/out arrays = 82 B × mb (= bt_per_n)
//                     (bt_r) d_bt_n_a/offset/ofs_end/num_elements/n_v/n_u + d_bt_cub_tmp
// (Voting arrays removed — the GPU voting/rechain path is gone.)
```

---

## `plmem_malloc_device_mem` — cudaMalloc Safety Rule

On failure the output pointer is **undefined** (not guaranteed NULL). Check the
**return code**:

```cpp
cudaError_t alloc_err = cudaMalloc(&arena_base, arena_size);
if (alloc_err != cudaSuccess || !arena_base) { /* FATAL */ }
```

---

## Long bt_p Dedicated Pool

After arena allocation each stream gets a dedicated `d_align_backtrack_p_long` pool
from remaining VRAM:

```
free_after_arena - (streams_remaining × arena_size) - pool_safety
```

`pool_safety = max(global_reserve, 512 MB)`. Min useful pool = 256 MB. Non-fatal if
it fails — falls back to shared arena bt_p (fewer concurrent slots).

---

## JSON Config Keys (runtime tuning)

| Key | Default | Effect |
|---|---|---|
| `global_vram_reserve_mb` | 256 | VRAM reserved from budget; ≥1024 for cuda-gdb |
| `num_streams` | from JSON | Concurrent pipeline streams |
| `avg_read_n` | 1000 | Avg anchors/read, affects overhead estimate |
| `long_seg_buffer_size` | auto | Override buffer_size_long (total, split by num_streams) |
| `long_cigar_batch` | 0 (auto) | Manual cap on long-task CIGAR batch size |
| `max_align_task_len` | default | Max alignment task length |

---

## GPU Hardware Constants

```
gpu_max_slots  = numSMs × 32   // V100s: 80×32=2560  A100: 108×32=3456
micro_batch    = score_kernel_config.micro_batch  // typically 4
```

---

## Known Issues / History

- **Voting/rechain removed**: no kernel used the `d_vt_*` arrays; their chain-arena
  allocations and the `vt_per_n` budget term were deleted (VRAM reclaimed).
- **Dead align buffers removed**: `d_align_global_buffer` (AGAThA,
  ~14 KB × `max_align_task_len`), `d_align_ez_array`, `d_align_backtrack_n_col`.
- **Z-drop split remainders** are re-aligned on GPU (iterative re-batch through the
  same kernel), not CPU `mm_align1`.
- **`gpu_batch_process_results` region indexing**: uses `reg_idx + reg_offset` to
  follow `mm_insert_reg` shifts — fixes CIGAR mis-mapping on multi-region reads when
  a non-last region z-drop-splits.
- **V100s `invalid argument`**: `per_long_entry` was 19 (missing xrev), must be 23.
- **`plmem_async_*_short_memcpy`**: guard `total_n==0`; **`start_backtrack_impl`**:
  H2D copies under `if (n > 0)`.
