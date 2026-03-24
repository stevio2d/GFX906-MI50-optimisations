# fix39: Split Destination Slot Wait Dedupe

**File changed:** `ggml/src/ggml-backend.cpp`
**Lines changed:** +17 / -6
**Status:** Validated on 4x AMD MI50 (gfx906), Qwen3.5-122B-A10B Q4_K_S

---

## What problem this solves

When llama.cpp splits a compute graph across multiple GPUs, each "split" has a
list of input tensors that must be copied to the destination GPU before it can
start computing. The upstream code did:

```
for each input tensor in the split:
    wait for destination slot to be free   ← fires N times
    copy tensor to destination GPU
```

The wait is either `event_wait` (GPU-side async fence) or
`ggml_backend_synchronize` (CPU blocks until GPU idle). The bug: **it waits
once per input tensor**, even though all inputs in the same split go to the
**same destination copy slot**. If a split has 10 input tensors, it waits 10
times when it only needs to wait once.

At **long context (64k tokens) under pipeline-parallel mode** (`--split-mode
layer`), splits rotate through a copy-slot ring (0→1→2→3→0→...). Without a
proper fence, a slot could be reused before the previous round's compute
finished — producing **deterministically wrong output** on every run.

At short context or tensor-parallel (`--split-mode row`), the ring stays on
slot 0 the whole time so the corruption was latent and invisible — which is
why baseline was correct at 512 tokens but wrong at 64k PP.

---

## What the fix does

Three changes, all in `ggml_backend_sched_compute_splits()`:

### 1. Synchronize once before the input loop (no-event path)

```cpp
const bool has_split_event = sched->events[split_backend_id][sched->cur_copy] != NULL;
bool split_dst_slot_waited = !has_split_event;
bool split_dst_slot_synced = !has_split_event;

if (!has_split_event) {
    ggml_backend_synchronize(split_backend);  // once, up front
}
```

If there is no GPU event for this slot (e.g. CPU backend), synchronize once
before touching any inputs rather than once per input.

### 2. Dedupe the wait/sync inside the input loop

```cpp
// user inputs (must be synchronous copy):
if (!split_dst_slot_synced) {
    ggml_backend_event_synchronize(...);   // only fires on first user input
    split_dst_slot_waited = true;
    split_dst_slot_synced = true;
}

// internal inputs (async copy, just need a stream-ordered fence):
if (!split_dst_slot_synced && !split_dst_slot_waited) {
    ggml_backend_event_wait(...);          // only fires once per split
    split_dst_slot_waited = true;
}
```

Boolean flags track whether the slot has already been waited on. All
subsequent inputs in the same split skip the wait entirely.

### 3. Synchronize once after the input loop (no-event path)

```cpp
if (!has_split_event) {
    ggml_backend_synchronize(split_backend);  // fence before async compute
}
```

Ensures the destination GPU is fully idle before
`ggml_backend_graph_compute_async` fires for this split, preventing the next
graph execution from overwriting a slot still in use.

---

## How to apply

The patch targets the llama.cpp commit it was validated against. Apply with:

```bash
git apply fix39-split-dst-slot-wait-dedupe.patch
```

Or cherry-pick after adding this repo as a remote.

---

## Benchmark results (4x MI50 / gfx906, validated 2026-03-24)

**Model:** Qwen3.5-122B-A10B Q4_K_S
**Hardware:** 4x AMD Instinct MI50, 32 GB each
**Test:** 4 cases × 3 reps = 12 runs, SHA-256 correctness verified

### Correctness

Three states were measured:

| Case | Original upstream (no fix) | fix/39 only (pre-merge) | fix/39 + upstream merged |
|------|---------------------------|------------------------|--------------------------|
| row / 512 | 3/3 exact | 3/3 exact | 3/3 exact |
| layer / 512 | 3/3 exact | 3/3 exact | 3/3 exact |
| row / 64k | 3/3 exact | 2/3 exact (stochastic) | 3/3 exact |
| layer / 64k | **0/3 exact (deterministic wrong)** | **3/3 exact** | **3/3 exact** |

- **Original upstream** = clean llama.cpp before any of our work. `layer/64k` was
  deterministically broken on every single run — always producing the same wrong
  hash `4b88a9cc...`. This is what this fix targets.
- **fix/39 only** = our patch applied without pulling upstream changes. `layer/64k`
  is now correct, but `row/64k` occasionally produced a stochastic wrong hash
  (`a8f253...`) — a separate latent TP issue that was already there in the original
  upstream and not caused by our patch.
- **fix/39 + upstream merged** = our patch forward-ported onto the latest upstream.
  12/12 exact. The stochastic TP corruption also disappeared, likely fixed by
  upstream sync changes.

Reference hashes:
- short (512): `1d89d99ce29434c19dcc563856bd240b50bd7964b094b2e3de089bafd33f603c`
- long (64k): `473d3c439268c70074d4bc52804395ef64ae276b147678f4b331e69b12316527`

### Throughput (fix/39 only vs fix/39 + upstream merged, eval tok/s)

| Case | fix/39 only | fix/39 + upstream | Delta |
|------|-------------|-------------------|-------|
| row / 512 | 18.3 | 25.6 | +40% |
| layer / 512 | 20.7 | 27.6 | +33% |
| row / 64k | 6.9 | 7.7 | +12% |
| layer / 64k | 7.3 | 8.0 | +11% |

Note: the throughput gains come from upstream changes, not from this fix alone.
This fix's contribution is correctness — specifically the `layer/64k` case.

---

## Command used to run the benchmark

```bash
HSA_OVERRIDE_GFX_VERSION=9.0.6 \
ROCBLAS_TENSILE_LIBPATH=/opt/rocm-custom/lib/rocblas/library \
LD_LIBRARY_PATH=/opt/rocm-custom/lib:/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH \
HIP_VISIBLE_DEVICES=0,1,2,3 \
ROCR_VISIBLE_DEVICES=0,1,2,3 \
./build-opt/bin/llama-completion \
  -m Qwen3.5-122B-A10B-Q4_K_S-00001-of-00003.gguf \
  -c 64000 \
  -ngl 999 \
  --fit off \
  --temp 0 \
  --top-k 1 \
  --seed 123 \
  -n 32 \
  --simple-io \
  --no-display-prompt \
  --no-conversation \
  --no-warmup \
  --verbosity 3 \
  --perf \
  -f prompt-63968.txt \
  -t 16 -tb 16 \
  -b 1024 -ub 256 \
  -ctk q4_0 -ctv q4_0 \
  -fa on \
  --split-mode layer
```

Swap `--split-mode layer` for `row` to test tensor-parallel mode.
Swap `-c 64000` and the prompt file for `-c 512` / `prompt-480.txt` for short context.

### Running as a server (for use with OpenCode or other frontends)

```bash
HSA_OVERRIDE_GFX_VERSION=9.0.6 \
ROCBLAS_TENSILE_LIBPATH=/opt/rocm-custom/lib/rocblas/library \
LD_LIBRARY_PATH=/opt/rocm-custom/lib:/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH \
HIP_VISIBLE_DEVICES=0,1,2,3 \
ROCR_VISIBLE_DEVICES=0,1,2,3 \
./build-opt/bin/llama-server \
  -m Qwen3.5-122B-A10B-Q4_K_S-00001-of-00003.gguf \
  -ngl 999 \
  --split-mode layer \
  -ctk q4_0 -ctv q4_0 \
  -fa on \
  -c 32768 \
  --port 8080 \
  --host 0.0.0.0
```

---

## Hardware notes

- ROCm 7.2 dropped official gfx906 support — requires `HSA_OVERRIDE_GFX_VERSION=9.0.6`
- Custom rocBLAS build required at `/opt/rocm-custom/lib/rocblas/library` with gfx906 Tensile kernels
- `GGML_HIP_GRAPHS=OFF` — HIP graph capture is unstable on gfx906 at long context
- `GGML_HIP_NO_VMM=ON` — virtual memory management not supported on gfx906
