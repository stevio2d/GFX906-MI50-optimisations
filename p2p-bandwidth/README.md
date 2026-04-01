# P2P Bandwidth Tool

Small HIP utility for measuring GPU-to-GPU peer-to-peer bandwidth on multi-GPU ROCm systems.

It does two things in one run:

- validates that a full device-to-device copy actually lands correctly on the destination GPU
- measures copy throughput with HIP events and reports per-direction bandwidth in GiB/s

This is useful when bringing up MI50 or other ROCm boxes where peer access may appear available, but some paths are slower than expected or intermittently corrupt under load.

## Included Files

- `p2p_bandwidth.cpp`: source
- `p2p_bandwidth`: prebuilt binary from this workspace

Binary SHA256:

`d90a05e55a276aafc87fa2df46ae4d48edda6e3351d5263bfe79b643f3c408db`

Source SHA256:

`671927f348248b6ee623ce3669b7071a875d6e253c36f7950238838caa94376f`

## What It Reports

For every supported `src -> dst` GPU direction, the tool:

1. enables HIP peer access where available
2. fills source and destination buffers with different byte patterns
3. performs a peer copy
4. validates the destination buffer byte-for-byte
5. runs timed throughput trials and reports average, minimum, and maximum GiB/s

It also prints:

- GPU index to PCI bus mapping
- optional PCIe slot labels if `dmidecode` data is available
- a peer access matrix showing whether destination GPUs can access source GPUs

## Slot / Topology Labels

The tool tries to label each GPU with its PCIe slot designation by first running:

```bash
sudo -n dmidecode -t slot
```

If passwordless `sudo` is not available, it falls back to an `outputdmi.txt` file in the current directory or next to the executable. The benchmark still works without this data; you just get bus IDs instead of friendlier slot names.

## Usage

```bash
./p2p_bandwidth [--throughput-only] [--payload-mib MiB] [--repeat N] [--pair SRC,DST]
```

### Flags

- `--throughput-only`
  Warn on validation failures and continue reporting bandwidth instead of exiting immediately.
- `--payload-mib <MiB>`
  Transfer size per copy. Default: `512`.
- `--repeat <N>`
  Repeat each tested direction `N` times. Default: `1`.
- `--pair <SRC,DST>`
  Test only one direction, for example `--pair 1,2`.
- `--help`
  Print usage.

## Example Commands

Test every supported direction with defaults:

```bash
./p2p_bandwidth
```

Stress one path with a larger payload and repeated runs:

```bash
./p2p_bandwidth --payload-mib 1024 --repeat 5 --pair 1,2
```

Keep collecting throughput even if validation catches corruption:

```bash
./p2p_bandwidth --throughput-only --repeat 3
```

## Timing Method

The throughput number is based on HIP event timing of repeated `hipMemcpyPeerAsync()` operations on the destination stream.

Current built-in benchmark settings:

- 10 warmup iterations
- 30 measured iterations per trial
- 6 total trials, with the first discarded
- 5 reported trials per direction

The summary prints average, minimum, and maximum GiB/s across the kept trials.

## Build

Compile with ROCm HIP tooling, for example:

```bash
hipcc -O3 -std=c++17 -o p2p_bandwidth p2p_bandwidth.cpp
```

## Notes

- The tool requires at least 2 visible HIP devices.
- It tests directional bandwidth, so `GPU0 -> GPU1` and `GPU1 -> GPU0` are reported separately.
- In default strict mode, the program exits on the first validation failure after printing a partial summary.
- Validation checks correctness before timing, which helps catch broken P2P paths that would otherwise look fast.
