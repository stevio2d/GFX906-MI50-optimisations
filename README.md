# GFX906 / MI50 llama.cpp Optimisations

A collection of patches and fixes for running llama.cpp on AMD Instinct MI50 (gfx906) hardware,
which is no longer officially supported by ROCm 7.x.

## Current Recommendation (2026-03-27)

For MI50 / gfx906, the current best direction for `llama.cpp` with compressed KV
and FlashAttention is:

- keep KV stored as `K=tq3_0`, `V=f16`
- use `--flash-attn on`
- keep decode-like FlashAttention shapes on the conservative vec path
- let prefill-like shapes use the generic tile / `f16` FlashAttention path,
  reconstructing `K` to a temporary `f16` buffer once per kernel launch
- preserve the existing gfx906 `SOLVE_TRI` safeguard

This removes the long-prefill regression without giving up KV savings.

## Latest Validation (ROCm 7.2.0)

Testing on Qwen3.5-27B-abliterated (Q4_K_S) / 1x MI50 / FlashAttention enabled /
3 repetitions per shape.

| Shape | `f16/f16 + FA` | `tq3_0/f16 + FA` | Delta |
|-------|----------------|------------------|-------|
| `pp64` | `167.88 ± 0.11` | `166.11 ± 0.12` | `-1.1%` |
| `pp4096` | `244.79 ± 0.47` | `243.09 ± 0.16` | `-0.7%` |
| `tg256` | `22.08 ± 0.01` | `21.56 ± 0.01` | `-2.4%` |
| `tg256 @ d4096` | `21.61 ± 0.02` | `20.87 ± 0.00` | `-3.4%` |

Interpretation:

- the long-prefill problem is effectively solved
- long prefill is now within about 1% of the `f16/f16` FlashAttention baseline
- decode remains slightly lower than `f16`, but still close
- this makes the dispatch split more promising than older gfx906-specific
  FlashAttention tuning patches that regressed prefill

## Larger-Model Sanity Check

Target model: Qwen3.5-122B-A10B Q4_K_S / 3x MI50 / `--split-mode layer`

- completed a 3-GPU sanity run with `--flash-attn on`, `--cache-type-k tq3_0`,
  `--cache-type-v f16`, `-ngl auto`, `-fit on`
- offloaded `49/49` layers to GPU
- model buffers landed at `23732.80 MiB`, `22334.36 MiB`, `21532.72 MiB`
- KV at `c=4096` was `58.50 MiB` total:
  `K=tq3_0` `10.50 MiB`, `V=f16` `48.00 MiB`
- short `-n 8` completion sanity run succeeded with FlashAttention enabled

This was a runtime sanity check, not a full apples-to-apples benchmark.

## Commands And Proof

Exact commands, environment, and proof excerpts are in
[validation-tq3-fa-2026-03-27.md](validation-tq3-fa-2026-03-27.md).

## Patches

| Directory | Description | Status |
|-----------|-------------|--------|
| [fix39-split-dst-slot-wait-dedupe](fix39-split-dst-slot-wait-dedupe/) | Fixes deterministic wrong output at 64k context under pipeline-parallel (--split-mode layer) by deduplicating destination slot wait/sync per split | Validated 4x MI50 |
| [fix32-fattn-tile-config](fix32-fattn-tile-config/) | Lowers thread count for Flash Attention tile configuration case (256, 256, 32) from 256 down to 128 in `fattn-tile.cuh` | **Regression** (verified slower than upstream) |
| [fix51-stream-k-mmq-gcn](fix51-stream-k-mmq-gcn/) | Enables Stream-K work partitioning for GCN GPUs. | **Regression** for large prompt processing batches. |

## Hardware

- AMD Instinct MI50 / gfx906 (Vega 20)
- 32 GB HBM2 (or 16 GB variants)
- ROCm 6.x / 7.x with `HSA_OVERRIDE_GFX_VERSION=9.0.6`
- If you are using custom gfx906 rocBLAS Tensile libraries, verify
  `ROCBLAS_TENSILE_LIBPATH=/opt/rocm-custom/lib/rocblas/library`
  before treating an early rocBLAS failure as a model/runtime regression.
- For the `tq3_0/f16` KV path documented above, prefer FlashAttention over
  old gfx906-only tuning patches unless a new patch is revalidated.
