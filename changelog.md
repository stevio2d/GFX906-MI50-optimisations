# Changelog

## [Unreleased]
- Added `validation-tq3-fa-2026-03-27.md` with exact commands, environment, and proof
  for the MI50 / gfx906 `tq3_0` FlashAttention validation pass.
- Updated `README.md` with the current recommendation for `K=tq3_0`, `V=f16`,
  FlashAttention-enabled MI50 runs, plus a 27B benchmark summary, KV cache savings,
  and a 122B 3-GPU sanity result.
- Added **Fix 32** in `fix32-fattn-tile-config/`. 
  - This fix modifies `ggml/src/ggml-cuda/fattn-tile.cuh` to alter the `GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 32...` threading configuration from `256` threads to `128` threads to improve Flash Attention performance/stability on MI50.
- Updated `README.md` to reflect the newly added Fix 32.
