# TurboMap

A GPU-accelerated extension alignment engine for long-read sequencing that delivers high speed with near-lossless accuracy.

## Build

```bash
make -j$(nproc) GPUARCH=sm_80
```

Optional flags:
- `DPX=1` — use the Hopper DPX kernel (requires `GPUARCH=sm_90`).
- `PRINT=1` — verbose logging.
- `RAW=1` / `KERNEL_ONLY=1` — ablation builds.

## Run

```bash
./minimap2 -a -t $(nproc) --gpu --gpu-cfg gpu/gpu_config.json ref.mmi reads.fastq.gz > out.sam
```

## License

MIT (see `LICENSE.txt`), inheriting from upstream minimap2.
