# TQ3_0 FlashAttention Validation (2026-03-27)

This note records the exact commands and proof used for the MI50 / gfx906
validation pass on the near-final `tq3_0` FlashAttention dispatch change:

- decode-like shapes stay on the vec path
- prefill-like shapes fall through to the generic tile / `f16` FlashAttention path
- `K` stays compressed in KV and is reconstructed to temporary `f16` once per launch

## Scope

- `K=tq3_0`
- `V=f16`
- FlashAttention enabled
- MI50 / gfx906 focus
- preserve correctness, stability, KV savings, and existing `SOLVE_TRI` safeguard

## Environment Used

```bash
export PATH=/opt/rocm/bin:$PATH
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export ROCBLAS_LAYER=0
export ROCBLAS_LOG_LEVEL=0
export HIPBLASLT_LOG_LEVEL=0
export HIP_FORCE_DEV_KERNARG=1
export GPU_MAX_HW_QUEUES=8
export HSA_ENABLE_SDMA=0
export GPU_SINGLE_ALLOC_PERCENT=100
export ROCBLAS_TENSILE_LIBPATH=/opt/rocm-custom/lib/rocblas/library
export LD_LIBRARY_PATH=/opt/rocm-custom/lib:/opt/rocm/lib:/opt/rocm/lib64:${LD_LIBRARY_PATH:-}
```

## Commands Used

### 1. 27B single-GPU benchmark: short + long prefill, shallow decode

```bash
cd /home/stefan/llama.cpp-gfx906

HIP_VISIBLE_DEVICES=0 \
PATH=/opt/rocm/bin:$PATH \
HSA_OVERRIDE_GFX_VERSION=9.0.6 \
ROCBLAS_LAYER=0 \
ROCBLAS_LOG_LEVEL=0 \
HIPBLASLT_LOG_LEVEL=0 \
HIP_FORCE_DEV_KERNARG=1 \
GPU_MAX_HW_QUEUES=8 \
HSA_ENABLE_SDMA=0 \
GPU_SINGLE_ALLOC_PERCENT=100 \
ROCBLAS_TENSILE_LIBPATH=/opt/rocm-custom/lib/rocblas/library \
LD_LIBRARY_PATH=/opt/rocm-custom/lib:/opt/rocm/lib:/opt/rocm/lib64:${LD_LIBRARY_PATH:-} \
./build-mi50/bin/llama-bench \
  -m /home/stefan/.lmstudio/models/mradermacher/Huihui-Qwen3.5-27B-abliterated-GGUF/Huihui-Qwen3.5-27B-abliterated.Q4_K_S.gguf \
  -ngl 99 \
  -sm layer \
  -mg 0 \
  -dev ROCm0 \
  -fa 1 \
  -ctk f16,tq3_0 \
  -ctv f16 \
  -p 64,4096 \
  -n 32,256 \
  -b 2048 \
  -ub 512 \
  -r 3
```

### 2. 27B single-GPU benchmark: decode-at-depth check

```bash
cd /home/stefan/llama.cpp-gfx906

HIP_VISIBLE_DEVICES=0 \
PATH=/opt/rocm/bin:$PATH \
HSA_OVERRIDE_GFX_VERSION=9.0.6 \
ROCBLAS_LAYER=0 \
ROCBLAS_LOG_LEVEL=0 \
HIPBLASLT_LOG_LEVEL=0 \
HIP_FORCE_DEV_KERNARG=1 \
GPU_MAX_HW_QUEUES=8 \
HSA_ENABLE_SDMA=0 \
GPU_SINGLE_ALLOC_PERCENT=100 \
ROCBLAS_TENSILE_LIBPATH=/opt/rocm-custom/lib/rocblas/library \
LD_LIBRARY_PATH=/opt/rocm-custom/lib:/opt/rocm/lib:/opt/rocm/lib64:${LD_LIBRARY_PATH:-} \
./build-mi50/bin/llama-bench \
  -m /home/stefan/.lmstudio/models/mradermacher/Huihui-Qwen3.5-27B-abliterated-GGUF/Huihui-Qwen3.5-27B-abliterated.Q4_K_S.gguf \
  -ngl 99 \
  -sm layer \
  -mg 0 \
  -dev ROCm0 \
  -fa 1 \
  -ctk f16,tq3_0 \
  -ctv f16 \
  -n 256 \
  -d 4096 \
  -b 2048 \
  -ub 512 \
  -r 3
```

### 3. 122B three-GPU sanity check

```bash
cd /home/stefan/llama.cpp-gfx906

PATH=/opt/rocm/bin:$PATH \
HSA_OVERRIDE_GFX_VERSION=9.0.6 \
ROCBLAS_LAYER=0 \
ROCBLAS_LOG_LEVEL=0 \
HIPBLASLT_LOG_LEVEL=0 \
HIP_FORCE_DEV_KERNARG=1 \
GPU_MAX_HW_QUEUES=8 \
HSA_ENABLE_SDMA=0 \
GPU_SINGLE_ALLOC_PERCENT=100 \
ROCBLAS_TENSILE_LIBPATH=/opt/rocm-custom/lib/rocblas/library \
LD_LIBRARY_PATH=/opt/rocm-custom/lib:/opt/rocm/lib:/opt/rocm/lib64:${LD_LIBRARY_PATH:-} \
HIP_VISIBLE_DEVICES=0,1,2 \
./build-mi50/bin/llama-completion \
  -m /home/stefan/.lmstudio/models/unsloth/Qwen3.5-122B-A10B-GGUF/Qwen3.5-122B-A10B-Q4_K_S-00001-of-00003.gguf \
  -dev ROCm0,ROCm1,ROCm2 \
  -sm layer \
  -ngl auto \
  -fit on \
  --flash-attn on \
  --cache-type-k tq3_0 \
  --cache-type-v f16 \
  -c 4096 \
  -n 8 \
  --temp 0 \
  --perf \
  -no-cnv \
  -p "Write one paragraph about TurboQuant."
```

## Proof: 27B Benchmark Output

```text
| model                          |       size |     params | backend    | ngl | type_k | fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -: | ------------ | --------------: | -------------------: |
| qwen35 27B Q4_K - Small        |  14.49 GiB |    26.90 B | ROCm       |  99 |    f16 |  1 | ROCm0        |            pp64 |        167.88 ± 0.11 |
| qwen35 27B Q4_K - Small        |  14.49 GiB |    26.90 B | ROCm       |  99 |    f16 |  1 | ROCm0        |          pp4096 |        244.79 ± 0.47 |
| qwen35 27B Q4_K - Small        |  14.49 GiB |    26.90 B | ROCm       |  99 |    f16 |  1 | ROCm0        |            tg32 |         22.07 ± 0.01 |
| qwen35 27B Q4_K - Small        |  14.49 GiB |    26.90 B | ROCm       |  99 |    f16 |  1 | ROCm0        |           tg256 |         22.08 ± 0.01 |
| qwen35 27B Q4_K - Small        |  14.49 GiB |    26.90 B | ROCm       |  99 |  tq3_0 |  1 | ROCm0        |            pp64 |        166.11 ± 0.12 |
| qwen35 27B Q4_K - Small        |  14.49 GiB |    26.90 B | ROCm       |  99 |  tq3_0 |  1 | ROCm0        |          pp4096 |        243.09 ± 0.16 |
| qwen35 27B Q4_K - Small        |  14.49 GiB |    26.90 B | ROCm       |  99 |  tq3_0 |  1 | ROCm0        |            tg32 |         21.55 ± 0.02 |
| qwen35 27B Q4_K - Small        |  14.49 GiB |    26.90 B | ROCm       |  99 |  tq3_0 |  1 | ROCm0        |           tg256 |         21.56 ± 0.01 |
```

## Proof: 27B Decode-At-Depth Output

```text
| model                          |       size |     params | backend    | ngl | type_k | fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -: | ------------ | --------------: | -------------------: |
| qwen35 27B Q4_K - Small        |  14.49 GiB |    26.90 B | ROCm       |  99 |    f16 |  1 | ROCm0        |   pp512 @ d4096 |        237.86 ± 0.21 |
| qwen35 27B Q4_K - Small        |  14.49 GiB |    26.90 B | ROCm       |  99 |    f16 |  1 | ROCm0        |   tg256 @ d4096 |         21.61 ± 0.02 |
| qwen35 27B Q4_K - Small        |  14.49 GiB |    26.90 B | ROCm       |  99 |  tq3_0 |  1 | ROCm0        |   pp512 @ d4096 |        235.60 ± 0.09 |
| qwen35 27B Q4_K - Small        |  14.49 GiB |    26.90 B | ROCm       |  99 |  tq3_0 |  1 | ROCm0        |   tg256 @ d4096 |         20.87 ± 0.00 |
```

## Proof: 122B Three-GPU Sanity Output

```text
load_tensors: offloading output layer to GPU
load_tensors: offloading 47 repeating layers to GPU
load_tensors: offloaded 49/49 layers to GPU
load_tensors:   CPU_Mapped model buffer size =   772.97 MiB
load_tensors:        ROCm0 model buffer size = 23732.80 MiB
load_tensors:        ROCm1 model buffer size = 22334.36 MiB
load_tensors:        ROCm2 model buffer size = 21532.72 MiB

llama_context: flash_attn    = enabled
llama_kv_cache: size =   58.50 MiB (  4096 cells,  12 layers,  1/1 seqs), K (tq3_0):   10.50 MiB, V (f16):   48.00 MiB

common_perf_print: prompt eval time =     243.41 ms /     7 tokens (   34.77 ms per token,    28.76 tokens per second)
common_perf_print:        eval time =     201.97 ms /     7 runs   (   28.85 ms per token,    34.66 tokens per second)

llama_memory_breakdown_print: |   - ROCm0 (MI50/MI60)  | 32752 =  8518 + (23980 = 23732 +      73 +     174) +         253 |
llama_memory_breakdown_print: |   - ROCm1 (MI50/MI60)  | 32752 =  9940 + (22557 = 22334 +      69 +     154) +         254 |
llama_memory_breakdown_print: |   - ROCm2 (MI50/MI60)  | 32752 = 10364 + (22134 = 21532 +      65 +     537) +         253 |
```

## Notes

- The 122B run above was a load-and-run sanity check, not a benchmark-grade comparison.
- A direct 122B run without the custom `ROCBLAS_TENSILE_LIBPATH` failed with a gfx906
  Tensile lookup error; the environment matters.
- Based on the 27B results and 122B sanity pass, the current recommendation is to
  stop reopening old gfx906-specific FlashAttention tuning patches unless a new
  patch is clearly revalidated against this newer dispatch behavior.
