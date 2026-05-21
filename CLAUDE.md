# TurboMap GPU Extension — Developer Reference

This file is auto-loaded by Claude Code at session start. It covers the Extension
(chain+backtrack+voting) GPU memory architecture. Update it when the formulas or
layout change.

---

## Working Branch

Active development branch: `claude/clean-v100-nvidia-only-P39Ok`

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
    └─ arena transitions: setup_*_phase() re-uses same physical block
```

Phase transitions do **no** `cudaFree`/`cudaMalloc` — they just reset
`arena.offset = 0` and re-run `setup_chain_phase` or `setup_align_phase`.

---

## `plmem_config_batch` — Per-Anchor Cost Formula

File: `gpu/plmem.cu`, function `plmem_config_batch` (~line 1207).

### Key variables

| Variable | Value | Meaning |
|---|---|---|
| `chain_per_n` | 27 B | ax(4)+ay(4)+sid(1)+xrev(4)+yrev(4)+range(4)+f(4)+p(2) per anchor |
| `bt_per_n` | `mb × 82` B | backtrack arrays per anchor |
| `vt_per_n` | `mb × 61` B | voting arrays per anchor |
| `cub_per_n` | `mb × 16` B | CUB sort temp per anchor |
| `per_long_entry` | **23** B | ax(4)+ay(4)+sid(1)+range(4)+**xrev(4)**+f(4)+p(2) — long seg buf |
| `long_ratio` | 2.0 | `buffer_size_long = long_ratio × max_total_n` |
| `mb` | `score_kernel_config.micro_batch` | usually 4 |
| `global_reserve` | 256 MB default | JSON key `global_vram_reserve_mb`; set ≥1024 MB for cuda-gdb |

### Formula

```
total_per_n = chain_per_n + mb*(bt_per_n + vt_per_n + cub_per_n)
            + long_ratio * per_long_entry
            + overhead_per_n          // index/cut grid cost, ≈ small

budget      = (gpu_free_mem - global_reserve) / num_streams * 0.98
max_total_n = budget / total_per_n
buffer_size_long = max_total_n * long_ratio
```

**Do not change `per_long_entry` without also checking `setup_chain_phase`.**
The value 23 accounts for `d_xrev_long` (4 bytes × buffer_size_long), which was
the root cause of the V100s `invalid argument` crash (was 19, missing xrev).

### Typical numbers

| GPU | Free VRAM | `max_total_n` | B/anchor |
|---|---|---|---|
| V100s 32 GB | ~31.4 GB | ~45.8 M | ~709 |
| A100 40 GB | ~39 GB | ~57.4 M | ~709 |

---

## `setup_chain_phase` — Arena Layout (in order)

File: `gpu/plmem.cu` ~line 193. All sizes are `anchor_per_batch = max_total_n`.

```
// Short-phase anchor arrays (N entries each)
d_ax, d_ay          int32_t × N       4+4 = 8 B/N
d_sid               int8_t  × N       1 B/N
d_xrev, d_yrev      int32_t × N       4+4 = 8 B/N
d_range             int32_t × N       4 B/N
d_f, d_p            int32_t+uint16_t  4+2 = 6 B/N
                                   ──────────
                                     27 B/N total (= chain_per_n)

// Index/cut arrays (proportional to range_grid_size / num_cut)
d_start_idx, d_read_end_idx, d_cut_start_idx   size_t × G
d_cut                                           size_t × C
d_long_seg_count                                uint    (1 element)

// Long segment buffers (L = buffer_size_long = long_ratio × N)
d_ax_long, d_ay_long      int32_t × L    4+4
d_sid_long                int8_t  × L    1
d_range_long              int32_t × L    4
d_xrev_long               int32_t × L    4      ← was missing from formula
d_total_n_long            size_t  (1 element)
d_f_long                  int32_t × L    4
d_p_long                  uint16_t × L   2
                                     ─────
                                      19+4 = 23 B/L (= per_long_entry)

// Backtrack arrays (bt_n = N × mb, bt_r = range_grid_size × mb)
d_bt_ax_in, d_bt_ay_in, d_bt_xrev_in, d_bt_yrev_in   int32_t  4×4 = 16 B
d_bt_f_in                                              int32_t  4 B
d_bt_p_in                                              uint16_t 2 B
d_bt_zx, d_bt_zy, d_bt_v, d_bt_p_abs                 int64_t  4×8 = 32 B
d_bt_t, d_bt_u                                         int32+uint64  4+8 = 12 B
d_bt_ax_out, d_bt_ay_out, d_bt_xrev_out, d_bt_yrev_out int32_t 4×4 = 16 B
                                                      ────────────────
                                                        82 B × mb  (= bt_per_n)
d_bt_n_a, d_bt_offset, d_bt_ofs_end, d_bt_num_elements, d_bt_n_v, d_bt_n_u
                                                        int × bt_r (6 × 4 B)
d_bt_cub_tmp                                            CUB temp buffer

// Voting arrays (vt_a = N × mb)
d_vt_ax, d_vt_ay, d_vt_bx, d_vt_by   uint64_t  4×8 = 32 B
d_vt_mark, d_vt_anchor_seg, d_vt_out_pos, d_vt_votes  int32_t 4×4 = 16 B
d_vt_keep_bin                           int8_t   1 B
d_vt_seg_start, d_vt_seg_id, d_vt_seg_cnt_flat  int32_t 3×4 = 12 B
                                               ─────────────────────
                                                 61 B × mb  (= vt_per_n)
```

---

## `plmem_malloc_device_mem` — cudaMalloc Safety Rule

File: `gpu/plmem.cu` ~line 599.

CUDA spec: on failure, the output pointer is **undefined** (not guaranteed NULL).
Always check the **return code**:

```cpp
cudaError_t alloc_err = cudaMalloc(&arena_base, arena_size);
if (alloc_err != cudaSuccess || !arena_base) { /* FATAL */ }
```

---

## Long bt_p Dedicated Pool

After arena allocation each stream gets a dedicated `d_align_backtrack_p_long`
pool from remaining VRAM. Sizing:

```
free_after_arena - (streams_remaining × arena_size) - pool_safety
```

`pool_safety = max(global_reserve, 512 MB)`. Minimum useful pool = 256 MB.
Non-fatal if allocation fails — falls back to shared arena bt_p (fewer concurrent slots).

---

## JSON Config Keys (runtime tuning)

| Key | Default | Effect |
|---|---|---|
| `global_vram_reserve_mb` | 256 | VRAM reserved from budget; set ≥1024 for cuda-gdb |
| `num_streams` | from JSON | Number of concurrent pipeline streams |
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

- **V100s `invalid argument` in `plmem_async_h2d_short_memcpy`**: root cause was
  `per_long_entry=19` (missing xrev, should be 23) causing arena underestimate,
  combined with unsafe `cudaMalloc` NULL check. Fixed in commit b8b06d9.
- **`plmem_async_h2d_short_memcpy` / `plmem_async_d2h_short_memcpy`**: guarded
  against `total_n==0` to avoid zero-count CUDA ops (reads with no anchors in
  single-chromosome references like chr3.mmi).
- **`start_backtrack_impl`**: H2D copies wrapped in `if (n > 0)` guard.
