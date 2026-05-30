# ane-perf

Hardware performance characterization of the Apple Neural Engine via IOReport bandwidth histograms. First IOReport-histogram GB/s DRAM-bandwidth method for ANE bandwidth behavior, SRAM boundaries, and dispatch thresholds during LLM inference on Apple Silicon. The SRAM-boundary finding is shared prior art: maderix (Inside the M4 ANE, Part 2, 2026-02-28) independently placed the ANE on-chip SRAM at ~32 MB via the same 24MB-fast / 96MB-slow cliff (TFLOPS wall-clock, not GB/s).

No root. No SIP changes. No entitlements. Runs on any Mac with Apple Silicon.

## Tools

**enumerate.py** -- Extract all 63 hardware counter names from ANE firmware.
```bash
python3 enumerate.py
```
Reads PerfTracer symbols from ANEServices.framework (counter names only -- values are daemon-locked). Also attempts to enumerate 25 `_ANEPerformanceStats` counter names if pyobjc is installed.

**kext_analysis.py** -- ANE kext reverse engineering: dispatch tables, hardware counters, register addresses.
```bash
python3 kext_analysis.py                    # Print full analysis summary
python3 kext_analysis.py --dump-counters    # JSON dump of all counters with registers
```
Static analysis results extracted from the decompressed macOS 26.3 kernelcache. Includes 25 hardware counter names with MMIO register addresses, IOKit selector dispatch tables for both UserClient (11 selectors) and DirectPathClient (9 selectors), entitlement requirements, and the perf counter access architecture.

**measure.py** -- Measure ANE bandwidth, energy, and throttle state during CoreML inference.
```bash
python3 measure.py model.mlpackage              # measure a model
python3 measure.py model.mlpackage --idle -n 200 # with idle baseline, 200 iters
python3 measure.py --discover                    # list all ANE IOReport channels
```
Captures 32-state bandwidth histograms from IOReport's PMP group. Reports utilization, average bandwidth state, peak state, energy, and throttle events.

**profile.py** -- CoreML ANE profiler: per-operation execution scheduling and hardware measurements.
```bash
python3 profile.py model.mlpackage              # Profile any CoreML model
python3 profile.py model.mlpackage --iters 200   # More iterations for accuracy
python3 profile.py model.mlpackage --json         # JSON output
python3 profile.py --build 1024 16 1              # Build and profile a test model
```
Uses ObjC runtime introspection to extract per-operation backend dispatch (ANE vs BNNS vs MPS Graph), estimated run time per backend from CoreML's E5 runtime cost model, and IOReport hardware counters. Shows which ops run on ANE, why (with per-backend cost comparison), and measured energy/bandwidth during execution. No entitlements required.

**experiments.py** -- Reproduce all findings from this repo.
```bash
python3 experiments.py          # all experiments
python3 experiments.py sram     # SRAM boundary
python3 experiments.py conv     # conv vs linear
python3 experiments.py dispatch # dispatch threshold
python3 experiments.py scaling  # scaling behavior
python3 experiments.py calibrate # BW state → GB/s calibration curve
python3 experiments.py int8     # INT8 vs FP16 energy and performance
python3 experiments.py thermal  # 10-minute sustained load throttle test
python3 experiments.py multimodel # multi-model interference test
```
Builds synthetic CoreML models at controlled sizes, measures via IOReport.

## Requirements

- macOS on Apple Silicon (M1 or later)
- Python 3.9+
- `coremltools`, `numpy` (for measure.py and experiments.py)
- `torch` (for experiments.py model building)

```bash
pip3 install coremltools numpy torch
```

enumerate.py has no dependencies beyond the standard library.

## The Three-Zone Performance Model

ANE performance falls into three zones based on per-layer weight size relative to the 32MB on-chip L2 SRAM:

```
      Zone 1: SRAM-resident    Zone 2: SRAM thrashing    Zone 3: DRAM streaming
      weights fit in L2        weights barely spill,     weights clearly exceed SRAM,
        < ~19 MB/layer         futile caching attempts   DMA pipelines efficiently
                                19-34 MB/layer            > ~34 MB/layer

  ms   |                                    .
  per  |                                 .
  pred |                              .
       |                         x    (4x cliff at 32MB)
       |                      .
       |                   .
       |                .
       |  . . . . . .
       +----+----+----+----+----+----+----+----> per-layer weight (MB)
            5   10   15   20   25   30   35
```

Measured on M5 Air (8-layer Conv1d, FP16, 100 iterations each):

| Dim | Per-layer | ms/pred | ANE energy | ANE util | ANE avg BW | Zone |
|-----|-----------|---------|------------|----------|------------|------|
| 1536 | 4.7 MB | 0.735 | 78 | 79.7% | 22.6/31 | SRAM |
| 2048 | 8.4 MB | 1.128 | 125 | 86.4% | 25.3/31 | SRAM |
| 2560 | 13.1 MB | 1.614 | 184 | 90.8% | 27.2/31 | SRAM |
| 3072 | 18.9 MB | 2.256 | 258 | 93.0% | 28.1/31 | SRAM |
| 3584 | 25.7 MB | 3.020 | 347 | 94.7% | 28.8/31 | SRAM |
| **4096** | **33.6 MB** | **12.045** | **1,133** | **98.7%** | **20.2/31** | **Thrashing** |

At 33.6 MB per layer (just over 32 MB SRAM): **4x latency, 3.3x energy**. ANE average bandwidth state *drops* from 28.8 to 20.2 despite 98.7% utilization -- the hardware is spending more time stalled on DRAM weight reloads.

*Measured with synthetic Conv1d models (single matmul per layer). Boundaries may differ for real transformer layers with attention, KV cache access patterns, and mixed operation types.*

## Prior Work

This repo extends and in some cases corrects earlier ANE research:

- [maderix/ANE](https://github.com/maderix/ANE) identified `_ANEPerformanceStats` counters and first measured the SRAM performance cliff from wall-clock timing. His finding that "conv is 3x faster than matmul" on ANE motivated our controlled measurement (Finding 1 below), which shows the difference is CoreML scheduling, not ANE hardware.
- The [Orion paper](https://arxiv.org/abs/2603.06728) catalogued ANE architectural constraints and informed our experiment design.
- [hollance/neural-engine](https://github.com/hollance/neural-engine) documents CoreML compilation behavior and ANE operator support.
- [ANEMLL/Anemll](https://github.com/ANEMLL/Anemll) provides the CoreML conversion pipeline we used for confirmed-ANE reference models.

## Findings

### 1. Conv = Linear on ANE

Widely repeated claim: "Conv is faster than matmul on ANE." This is wrong.

When both operations actually run on ANE, they are identical within measurement noise:

| Dim | Linear ms | Conv1d ms | Conv2d ms | Ratio |
|-----|-----------|-----------|-----------|-------|
| 2048 | 1.125 | 1.133 | 1.134 | 0.99x |
| 2560 | 1.599 | 1.620 | 1.631 | 0.99x |
| 3072 | 2.237 | 2.269 | 2.266 | 0.99x |
| 3584 | 2.981 | 3.013 | 3.024 | 0.99x |

Same energy. Same bandwidth utilization. Same everything. CoreML compiles both to the same ANE operation.

The real difference: CoreML's scheduler routes Conv ops to ANE at lower thresholds than Linear. At dim=1536, Conv gets 79.7% ANE utilization while Linear stays on CPU. At seq_len=1, Linear gets 71% ANE utilization vs Conv at 86%. At seq_len >= 4, both are identical. This is a CoreML scheduling decision, not an ANE hardware property.

### 2. CoreML Dispatch Threshold

CoreML does not always use ANE. Small models run on CPU because ANE dispatch overhead (~0.1ms) exceeds the compute time:

- Multi-layer models: ANE activates above ~37 MB total weight
- Single-layer Conv2d: ANE activates above ~18 MB per op
- Single-layer Linear: ANE activates above ~50 MB per op (higher threshold)

Below the threshold, CPU with AMX is faster. This means ANE profiling requires models large enough to actually trigger ANE dispatch -- check `ANE0 RD+WR` utilization to confirm.

### 3. ANE Bandwidth Profile

During LLM inference (Qwen3.5-0.8B FFN, 24 layers, 1024-dim):

| Metric | Value |
|--------|-------|
| Estimated ANE DRAM bandwidth | ~58 GB/s (see calibration in Finding 6) |
| RD:WR ratio | 26.6:1 |
| ANE RD+WR utilization | 80.2% |
| ANE avg BW state | 24.8 / 31 |
| BW distribution | Bimodal: 22% idle (state 0) + 77% max (state 31) |
| Fabric (NI3) utilization | 96.3% |
| Throttle events | None (13 channels, all zero) |

ANE runs full-throttle or sleeps. No intermediate bandwidth states during steady inference. The 22% idle time is between CoreML dispatch calls, not hardware idle.

Note: an earlier version of this document reported ~123 GB/s, calculated as `(24.8/31) × 153.6 GB/s`. This was wrong -- it assumed state 31 maps to the full 153.6 GB/s system DRAM bandwidth. The calibration experiment (Finding 6) shows state 31 ≈ 75 GB/s. ANE's bandwidth ceiling is roughly half the total system DRAM, consistent with it being one agent on the SoC fabric.

### 4. 32 MB SRAM Boundary

The 4x latency cliff at 32 MB per layer is the sharpest performance boundary we found. Energy per prediction jumps 3.3x. DRAM utilization saturates at 99.2%.

The ~32 MB SRAM placement is shared prior art: maderix (Inside the M4 ANE, Part 2, 2026-02-28) independently placed the ANE on-chip SRAM at ~32 MB via the same 24MB-fast / 96MB-slow cliff. maderix measured this in TFLOPS via wall-clock; this repo's contribution is the IOReport-histogram GB/s DRAM-bandwidth method, not first discovery of the boundary itself.

An interesting anomaly: a single 52 MB layer is *faster* than a single 33.6 MB layer (0.953 ms vs 1.217 ms). When data clearly exceeds SRAM, ANE's DMA engine appears to pipeline weight streaming more efficiently than when the data barely spills.

### 5. INT8 Saves Bandwidth, Not Compute Cycles

INT8 weights on ANE are dequantized to FP16 before compute; INT8 saves memory bandwidth, not compute cycles. This matches maderix's finding and our own Q8 = FP16-speed result. The speedup below is a bandwidth effect (half the weight bytes streamed from DRAM), not a wider/faster compute path:

| Dim | FP16 ms | INT8 ms | Speedup | Energy ratio | BW state |
|-----|---------|---------|---------|--------------|----------|
| 2048 | 1.099 | 0.695 | 1.58x | 0.69x | 25.7 → 21.3 |
| 2560 | 1.621 | 0.954 | 1.70x | 0.65x | 27.2 → 23.8 |
| 3072 | 2.256 | 1.202 | 1.88x | 0.51x | 28.1 → 26.6 |
| 3584 | 3.033 | 1.594 | 1.90x | 0.81x | 28.3 → 27.3 |

Speedup scales with dim (1.58x → 1.90x). INT8 transfers half the bytes, so bandwidth state drops -- ANE finishes faster and spends more time idle between layers. The energy savings (up to 49% at dim=3072) come from moving half the weight bytes, consistent with INT8 weights being dequantized to FP16 for compute (bandwidth saving, not a native INT8 compute path).

### 6. BW State-to-GB/s Calibration

The 32 IOReport bandwidth states map to actual GB/s via empirical calibration (weight_bytes / latency):

| Avg BW state | Est. GB/s | Per-layer weight |
|-------------|-----------|-----------------|
| 22.1 | 51.8 | 40.5 MB |
| 24.5 | 57.4 | 28.9 MB |
| 25.7 | 60.5 | 35.3 MB |
| 26.2 | 62.7 | 98.0 MB |
| 27.5 | 65.0 | 128.0 MB |
| 27.7 | 65.9 | 162.0 MB |
| 19.0 | 44.6 | 200.0 MB |

Linear regression across the 22-28 state range: **GB/s ≈ 2.76 × state - 10.2**. This gives state 31 ≈ 75 GB/s -- ANE's maximum DRAM bandwidth allocation, roughly half the 153.6 GB/s total system DRAM. At the extremes, DRAM saturation (state 19 at 200MB) or dispatch overhead (state 22 at 5MB) distort the mapping.

This calibration corrects the naive formula `(avg_state / 31) × 153.6 GB/s` which overestimates by ~2x because it assumes ANE can consume the full system DRAM bandwidth.

### 7. Multi-Model Interference

Running two CoreML models simultaneously on ANE (both individually confirmed on ANE, util > 0 when run alone):

| Condition | Median ms/pred | Mean ms/pred | CV | Runs |
|-----------|---------------|-------------|-----|------|
| Model B alone (dim=3072) | 2.267 | 2.275 ± 0.046 | 2.0% | 5 × 1000 iters |
| Model B + Model A (dim=2048) bg | 2.240 | 2.596 ± 0.960 | 37.0% | 7 × 1000 iters |

Median slowdown: **0.99x**. Mean slowdown: **1.14x**. The difference is one outlier run (4.773ms, ~2x baseline) out of 7. The other 6 concurrent runs fall within baseline range.

Background model prediction counts vary widely across runs (5,100 to 66,429), suggesting CoreML's dispatch scheduler is non-deterministic under contention. The occasional latency spike is consistent with dispatch-level serialization when both models' CoreML calls collide, not sustained hardware contention.

No measurable interference at median latency, but expect occasional scheduling spikes under concurrent load. True sub-unit parallelism vs fast serialization cannot be distinguished from IOReport data alone -- that would require PerfTracer internal counters.

### 8. No Thermal Throttling Under Sustained Load

10 minutes of continuous ANE inference on the passively-cooled M5 Air (8-layer Conv1d, dim=3072):

| Metric | Value |
|--------|-------|
| Duration | 10 minutes, 120 samples at 5s intervals |
| Latency range | 2.214 - 2.347 ms/pred (6% variance) |
| Energy range | 5,001 - 5,750 per interval |
| ANE utilization | 97.5 - 99.3% (steady) |
| Throttle events | **Zero** across all 13 ANE_THROTTLE channels |

No latency degradation from minute 0 to minute 10. Over this 10-minute test, ANE's power envelope on the passively-cooled M5 Air stayed low enough that no throttling appeared, even at 99%+ utilization. Longer-duration or higher-ambient behavior was not tested; this result is scoped to the 10-minute window measured here.

## Methodology

### IOReport Bandwidth Histograms

IOReport exposes 32-state bandwidth histograms for each SoC agent. Each state represents a bandwidth tier, and the residency value (in nanoseconds) tells you how long the agent spent in that tier during the sample window.

State 0 = idle. State 31 = maximum bandwidth. The average state, weighted by residency, gives an estimate of sustained bandwidth utilization.

We sample before and after a batch of CoreML predictions, compute the delta, and extract ANE-specific channels:

- `ANE0 RD` / `ANE0 WR` at DCS BW: ANE read/write bandwidth at DRAM controller level
- `ANE0 RD+WR` at AF BW: Combined bandwidth at fabric level
- `SOC-NI3 ANE UP`: ANE upstream traffic on SoC network interconnect 3
- `ANE` energy: Energy counter from Energy Model group

### Controlling for ANE Placement

Not every CoreML model runs on ANE. We verify placement by checking `ANE0 RD+WR` utilization > 0. Models that show 0% ANE utilization are running on CPU.

To ensure ANE placement for controlled experiments:
- Use multi-layer models (8+ layers) with total weight > 37 MB
- Conv1d/Conv2d trigger ANE at lower thresholds than Linear
- Check energy counter > 0 and BW utilization > 0 before drawing conclusions

### Hardware

All measurements on M5 Air, 16 GB, macOS 26.3 (Tahoe), 2026-03-20. LPDDR5X @ 9600 MT/s = 153.6 GB/s theoretical DRAM bandwidth.

## Counter Reference

### PerfTracer (63 Metrics)

Counter names extracted from `ANEServices.framework` via `PerfTracerMetricToString()`. These are the names of hardware registers inside the ANE. Values require daemon-level access (see Architecture below).

| # | Category | Counters |
|---|----------|----------|
| 15 | L2 SRAM | read/write cycles, conflicts, stalls across two read ports + one write port |
| 18 | DMA | src1/src2 read + write cycles, conflicts for the DMA engines |
| 5 | Fabric/Bus | idle/active cycles, beat count (x64B), transaction count, latency product |
| 10 | Memory Cache | mcache hit/miss variants (alloc, no-alloc, depri, drop), fabric stall |
| 9 | Texture/Decomp | texture filter/index cache, weight decompression metadata/data cache |
| 6 | Neural Engine | ne_cycle, mac_cycle, ne_stall, nz_count, kernel_cache_miss, kernel_stall |

Hardware topology: 16 NE sub-units (ne_0 through ne_15), plus l2, dma_read, dma_write, kernel_read, pe blocks. See `data/counters.json` for the full annotated list.

### ANEH17Hidra PerfCtr (25 Counters with Register Addresses)

Extracted from the kernelcache `ANEH17Hidra_PerfCtr` table via kext disassembly (`kext_analysis.py`). These are MMIO register addresses on the ANE die. Each counter maps to 1-6 register addresses, likely corresponding to per-cluster or per-configuration views.

| Category | Counter | Index | Registers |
|----------|---------|-------|-----------|
| Clock Cycles | ANE_NE_NOMINAL_CYCLES | 23 | 5 addrs |
| Clock Cycles | ANE_L2_NOMINAL_CYCLES | 33 | 4 addrs |
| Compute | ANE_NE_COMPUTE_CYCLES | 26 | 5 addrs |
| Compute | ANE_L2PE_COMPUTE_CYCLES | 34 | 6 addrs |
| Stall | ANE_NE_INPUT_STALL_CYCLES | 27 | 5 addrs |
| Stall | ANE_NE_OUTPUT_STALL_CYCLES | 28 | 2 addrs |
| Stall | ANE_NE_KERNEL_STALL_CYCLES | 29 | 5 addrs |
| Stall | ANE_L2PE_INPUT_STALL_CYCLES | 35 | 6 addrs |
| Stall | ANE_L2PE_OUTPUT_STALL_CYCLES | 36 | 3 addrs |
| Throttle | ANE_NE_THROTTLE_CYCLES | 24 | 5 addrs |
| Throttle | ANE_L2_THROTTLE_CYCLES | 25 | 2 addrs |
| Thermal | ANE_NE_THERMAL_THROTTLE_CYCLES | 40 | 1 addr |
| Thermal | ANE_L2_THERMAL_THROTTLE_CYCLES | 41 | 2 addrs |
| DMA | ANE_DMA_READWRITE_BYTES | 30 | 5 addrs |
| DMA | ANE_DMA_READ_BYTES | 31 | 5 addrs |
| Energy | ANE_DPE_ENERGY | 32 | 4 addrs |
| Activity | ANE_NE_ACTIVITY_COUNT | 38 | 2 addrs |
| Activity | ANE_NE_ACTIVITY_COUNT_NZD | 39 | 2 addrs |
| MAC | ANE_MAC_THOTTLE_WIN0 | 37 | 6 addrs |
| Datatype | ANE_FP16_CYCLES | - | - |
| Datatype | ANE_INT8_CYCLES | - | - |

Register address ranges: `0x01910xxx` (base config), `0x01914xxx` (alt config), `0x01978xxx` (primary config). The multiple register banks suggest per-NE-cluster counters or different sampling configurations.

Additional counter names found without register mapping: `ANE_KM_STALL_CYCLES`, `ANE_L2_READ_STALL_CYCLES`, `ANE_L2_WRITE_STALL_CYCLES`.

### IOReport (Live, No Root)

Bandwidth histograms from the PMP (Power Management Processor) group:

| Channel | States | What it measures |
|---------|--------|-----------------|
| ANE0 RD / WR / RD+WR (AF BW) | 32 | ANE bandwidth at fabric level |
| ANE0 RD / WR / RD+WR (DCS BW) | 32 | ANE bandwidth at DRAM controller |
| SOC-NI3 ANE UP | 32 | ANE upstream on network interconnect |
| ANE-AF-BW (SOC Floor) | 5 | ANE fabric bandwidth floor state |
| ANE-DCS-BW (DCS Floor) | 8 | ANE DRAM bandwidth floor state |
| ANE (Energy Model) | int | ANE energy consumption |
| ANE_THROTTLE_* (SoC Stats) | 2 | 13 throttle state channels |

## Architecture

ANE performance data flows through two layers:

1. **IOReport** (accessible): Kernel-level SoC counters sampled by the Power Management Processor. 32-state bandwidth histograms, energy, throttle state. No root required. This is what measure.py uses.

2. **PerfTracer / PerfCtr** (names and register addresses only): Fine-grained hardware counters inside the ANE. 63 PerfTracer counter names readable from ANEServices.framework. 25 PerfCtr counter names with MMIO register addresses extracted from kernelcache ANEH17Hidra tables. Values require the `aned` daemon or the `com.apple.ane.iokit-user-access` entitlement.

### Kext Internals (from kext disassembly)

The ANE driver exposes two IOKit user client classes:

- **H11ANEInUserClient**: 11 selectors, general-purpose ANE access
- **H11ANEInDirectPathClient**: 9 selectors, optimized direct inference path

Key dispatch table entries (run `kext_analysis.py` for the full table):

| Selector | Input | Output | Purpose |
|----------|-------|--------|---------|
| 0 | 0x68 | 0x68 | Device handshake |
| 2 | 0x948 | 0x28 | **Program submit** (main inference path) |
| 9 | 0x10 | 0x18 | Performance stats query |

Performance counter values are collected during inference, not via a standalone selector:

1. Client sets `perfStatsMask` in the evaluation request (via `kANEFPerformanceStatsMaskKey`)
2. **Selector 2** submits the program to ANE (0x948-byte input struct)
3. Kernel calls `ANEHWDevice::ReadPerformanceCounters()` during execution
4. `AppleANEPerfCounterReadFunction::callFunction()` reads MMIO registers
5. Results returned to caller via IOSurface shared memory

### Entitlement Barrier

Opening the IOKit connection requires `com.apple.ane.iokit-user-access`. Unsigned processes are SIGKILL'd on `IOServiceOpen()`. Only `aned` (`/usr/libexec/aned`) and Apple-signed frameworks hold this entitlement.

The userspace ANE stack: CoreML -> Espresso -> AppleNeuralEngine.framework -> `aned` daemon -> IOKit driver. The `_ANEPerformanceStats` and `_ANEPerformanceStatsIOSurface` classes in AppleNeuralEngine.framework handle perf stat delivery, but are only usable from entitled processes.

### E5 Runtime (macOS 26+)

On macOS 26 (Tahoe), CoreML uses the E5 runtime (`MLE5Engine`), not the older `MLNeuralNetworkEngine`. The ANE is managed through compiled `e5rt_program_library` and `e5rt_execution_stream_operation` C++ objects -- there is no `_ANEInMemoryModel` ObjC object in the graph. This means:

- `_ANEModel.setPerfStatsMask:` is **unreachable** from the E5 runtime path
- Hardware perf counter values (`_ANEPerformanceStats.performanceCounters`) cannot be accessed through CoreML
- The `profilingOptions` bitmask on `MLModelConfiguration` does not expose per-engine timing

What IS accessible from E5 runtime (used by `profile.py`):

- **Per-operation backend dispatch**: `MLE5ProgramLibrary.segmentationAnalyticsAndReturnError:` returns a dictionary with `SelectedBackend` (ane/bnns/mps_graph/classic_cpu), `EstimatedRunTime` per backend, and `BackendSupport` per backend for every MIL operation
- **ANE cost model**: The `EstimatedRunTime.ane` values are CoreML's internal cost estimates for each op on ANE vs other backends. On a 1024x16 conv model: ANE=1.03ms vs BNNS=30.9ms vs MPS=10.5ms per op (30x estimated speedup)
- **Prediction timing**: `MLPredictionEventMetric._featuresPredictionDuration` gives total wall-clock time
- **IOReport hardware counters**: Energy, bandwidth histograms, throttle state (same as measure.py)

ANE dispatch threshold confirmed via segmentation analytics: models with <16MB total weight dispatch to BNNS. Models ≥32MB or with seq_len≥64 dispatch to ANE. The threshold depends on both total weight AND per-op compute density.

### What This Means

Getting PerfTracer counter values from userspace without SIP changes is not possible on current macOS. On macOS 26, the E5 runtime removes the `_ANEInMemoryModel` ObjC path entirely, making `perfStatsMask` unreachable even if the entitlement barrier were bypassed. The counter names and register addresses documented here are the maximum extractable information.

However, the E5 runtime exposes **per-operation execution scheduling data** including backend selection and cost estimates. Combined with IOReport hardware measurements, this provides a complete profiling picture: which ops run where, why, and at what cost. This is what `profile.py` provides.

## Open Questions

1. **PerfTracer values**: Kext disassembly confirms counter values are read via `ANEHWDevice::ReadPerformanceCounters()` during inference, gated by `perfStatsMask` and the `com.apple.ane.iokit-user-access` entitlement. On macOS 26, the E5 runtime (`MLE5Engine`) bypasses the `_ANEInMemoryModel` ObjC layer entirely, making `perfStatsMask` unreachable even through ObjC runtime introspection. Framework-level probing (6 approaches tested: `profilingOptions`, `_ANEModel` swizzling, Espresso profiling, `MLPredictionEvent`, `MLComputePlan`, `enableInstrumentsTracing`) confirms no path to hardware counter values without the IOKit entitlement. The per-operation cost model from `segmentationAnalytics` (used by `profile.py`) is the deepest profiling data available.

2. **SRAM streaming anomaly**: Why is a 52 MB single layer faster than a 33.6 MB single layer? Is ANE's tiling/scheduling optimized for clear-spill cases?

3. **M5 Pro / M4 Max differences**: Does the SRAM boundary shift on chips with more/fewer NE cores? The 32 MB L2 may vary by chip.

4. **BW state nonlinearity at extremes**: The calibration curve (Finding 6) is roughly linear in the 22-28 state range but distorts at high DRAM saturation. What drives the nonlinearity?

5. **Concurrent execution mechanism**: Multi-model median latency is unaffected but variance increases 18x (Finding 7). Is this dispatch-level serialization (CoreML queue contention) or hardware-level time-slicing? The ANE_NE_COMPUTE_CYCLES and ANE_NE_STALL_CYCLES counters (indices 26, 27) would resolve this, but require entitlement access.

6. **Register bank mapping**: The ANEH17Hidra_PerfCtr table maps each counter to 1-6 MMIO register addresses across three address ranges (0x01910xxx, 0x01914xxx, 0x01978xxx). Are these per-NE-cluster views, per-power-domain views, or different sampling configurations?

## Answered Questions

- ~~INT8 vs FP16 energy~~: **Answered** (Finding 5). INT8 is 1.6-1.9x faster, 0.5-0.8x energy -- a bandwidth saving from moving half the weight bytes, not a native INT8 compute path. INT8 weights are dequantized to FP16 before compute.
- ~~BW state calibration~~: **Answered** (Finding 6). ~2.5 GB/s per state in the linear range (states 22-28), with distortion at extremes.
- ~~Throttle behavior under sustained load~~: **Answered** (Finding 8). Zero throttle events across 10 minutes continuous load on passively-cooled M5 Air.

## License

MIT

## Related

- [maderix/ANE](https://github.com/maderix/ANE) -- ANE reverse engineering, _ANEPerformanceStats discovery, SRAM cliff wall-clock measurements
- [hollance/neural-engine](https://github.com/hollance/neural-engine) -- CoreML compilation behavior and ANE operator documentation
- [ANEMLL/Anemll](https://github.com/ANEMLL/Anemll) -- LLM to CoreML conversion pipeline for ANE inference
- [four-path-mlx](https://github.com/MidasMulli/four-path-mlx) -- Multi-source speculative decoding on Apple Silicon
- [orion-ane](https://github.com/MidasMulli/cognitive-stack-ane) -- Phantom agent with three-tier ANE architecture
- [gdn-coreml](https://github.com/MidasMulli/gdn-coreml) -- GatedDeltaNet SSM to CoreML converter
- [dual-path-inference](https://github.com/MidasMulli/dual-path-inference) -- GPU+ANE concurrency proof-of-concept (archived)
