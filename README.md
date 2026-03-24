# GFX906 / MI50 llama.cpp Optimisations

A collection of patches and fixes for running llama.cpp on AMD Instinct MI50 (gfx906) hardware,
which is no longer officially supported by ROCm 7.x.

## Patches

| Directory | Description | Status |
|-----------|-------------|--------|
| [fix39-split-dst-slot-wait-dedupe](fix39-split-dst-slot-wait-dedupe/) | Fixes deterministic wrong output at 64k context under pipeline-parallel (--split-mode layer) by deduplicating destination slot wait/sync per split | Validated 4x MI50 |

## Hardware

- AMD Instinct MI50 / gfx906 (Vega 20)
- 32 GB HBM2 per card
- ROCm 6.x / 7.x with `HSA_OVERRIDE_GFX_VERSION=9.0.6`
- Custom rocBLAS with gfx906 Tensile kernels

## Model tested

Qwen3.5-122B-A10B Q4_K_S (MoE, 122B parameters, ~67 GB)
