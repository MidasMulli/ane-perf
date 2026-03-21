# ane-perf

Hardware performance characterization of the Apple Neural Engine via IOReport bandwidth histograms. First published characterization of ANE bandwidth behavior, SRAM boundaries, and dispatch thresholds during LLM inference on Apple Silicon.

No root. No SIP changes. No entitlements. Runs on any Mac with Apple Silicon.

## Tools

**enumerate.py** -- Extract all 63 hardware counter names from ANE firmware.
```bash
python3 enumerate.py
```
Reads PerfTracer symbols from ANEServices.framework (counter names only -- values are daemon-locked). Also attempts to enumerate 25 `_ANEPerformanceStats` counter names if pyobjc is installed.

**measure.py** -- Measure ANE bandwidth, energy, and throttle state during CoreML inference.
```bash
python3 measure.py model.mlpackage              # measure a model
python3 measure.py model.mlpackage --idle -n 200 # with idle baseline, 200 iters
python3 measure.py --discover                    # list all ANE IOReport channels
```
Captures 32-state bandwidth histograms from IOReport's PMP group. Reports utilization, average bandwidth state, peak state, energy, and throttle events.

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

The real difference: CoreML's scheduler routes Conv ops to ANE at lower thresholds than Linear. At dim=1536, Conv gets 79.7% ANE utilization while Linear stays on CPU. This is a CoreML scheduling decision, not an ANE hardware property.

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
| Estimated DRAM bandwidth | ~123 GB/s (80% of 153.6 GB/s theoretical) |
| RD:WR ratio | 26.6:1 |
| ANE RD+WR utilization | 80.2% |
| BW distribution | Bimodal: 22% idle (state 0) + 77% max (state 31) |
| Fabric (NI3) utilization | 96.3% |
| Throttle events | None (13 channels, all zero) |

ANE runs full-throttle or sleeps. No intermediate bandwidth states during steady inference. The 22% idle time is between CoreML dispatch calls, not hardware idle.

### 4. 32 MB SRAM Boundary

The 4x latency cliff at 32 MB per layer is the sharpest performance boundary we found. Energy per prediction jumps 3.3x. DRAM utilization saturates at 99.2%.

An interesting anomaly: a single 52 MB layer is *faster* than a single 33.6 MB layer (0.953 ms vs 1.217 ms). When data clearly exceeds SRAM, ANE's DMA engine appears to pipeline weight streaming more efficiently than when the data barely spills.

### 5. INT8 Has a Real Hardware Path

INT8 quantization on ANE is **not** dequantized to FP16. It runs faster with less energy:

| Dim | FP16 ms | INT8 ms | Speedup | Energy ratio | BW state |
|-----|---------|---------|---------|--------------|----------|
| 2048 | 1.099 | 0.695 | 1.58x | 0.69x | 25.7 → 21.3 |
| 2560 | 1.621 | 0.954 | 1.70x | 0.65x | 27.2 → 23.8 |
| 3072 | 2.256 | 1.202 | 1.88x | 0.51x | 28.1 → 26.6 |
| 3584 | 3.033 | 1.594 | 1.90x | 0.81x | 28.3 → 27.3 |

Speedup scales with dim (1.58x → 1.90x). INT8 transfers half the bytes, so bandwidth state drops -- ANE finishes faster and spends more time idle between layers. The energy savings (up to 49% at dim=3072) confirm the hardware processes INT8 natively rather than widening to FP16.

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

The relationship is roughly linear in the 22-28 state range (~2.5 GB/s per state). At the extremes, DRAM saturation (state 19 at 200MB) or dispatch overhead (state 22 at 5MB) distort the mapping. The calibration curve converts any IOReport state reading to approximate GB/s for this hardware.

### 7. Multi-Model Concurrent Execution

Running two CoreML models simultaneously on ANE shows **zero interference**:

| Condition | Model B ms/pred | ANE util | ANE energy |
|-----------|----------------|----------|------------|
| Model B alone | 2.274 | 93.0% | 258 |
| Model B + Model A background | 2.232 | 99.6% | 2,596 |

Slowdown: **0.98x** (within noise). The background model completed 5,930 predictions while the foreground model ran 100 -- both at full speed. Energy jumped 10x confirming both models are active. ANE utilization rose from 93% to 99.6%.

This means ANE either has concurrent execution capability or serializes models fast enough that per-prediction latency is unaffected. Either way, running multiple CoreML models on ANE does not degrade individual model performance.

### 8. No Thermal Throttling Under Sustained Load

10 minutes of continuous ANE inference on the passively-cooled M5 Air (8-layer Conv1d, dim=3072):

| Metric | Value |
|--------|-------|
| Duration | 10 minutes, 120 samples at 5s intervals |
| Latency range | 2.214 - 2.347 ms/pred (6% variance) |
| Energy range | 5,001 - 5,750 per interval |
| ANE utilization | 97.5 - 99.3% (steady) |
| Throttle events | **Zero** across all 13 ANE_THROTTLE channels |

No latency degradation from minute 0 to minute 10. ANE's power envelope on M5 is low enough that passive cooling never saturates, even at 99%+ utilization. This means ANE workloads can run indefinitely without thermal management concerns on current hardware.

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

2. **PerfTracer** (names only): 63 fine-grained hardware counters inside the ANE. Names are readable from userspace. Values require the `aned` daemon's perf buffer, which is allocated daemon-side and never shared to client processes. IOKit selectors 0-31 all return NotReady from client connections. Getting values would require running as aned (SIP disabled), a kernel extension, or an Apple-signed entitlement.

The ANE driver architecture:
- Load path: client -> XPC -> aned daemon -> IOKit driver
- Eval path: client -> ANEServicesProgramProcessRequestDirect (C++ vtable) -> daemon
- Perf buffer: allocated by daemon, never returned to client
- `ANEDeviceStruct.connection` (ptr[0]) is a C++ vtable of function pointers, not a mach port

## Open Questions

1. **PerfTracer values**: How to read the 63 fine-grained counters without SIP changes? The `initWithRequestPerformanceBuffer:` method signature suggests the daemon could return this data, but it doesn't.

2. **SRAM streaming anomaly**: Why is a 52 MB single layer faster than a 33.6 MB single layer? Is ANE's tiling/scheduling optimized for clear-spill cases?

3. **M5 Pro / M4 Max differences**: Does the SRAM boundary shift on chips with more/fewer NE cores? The 32 MB L2 may vary by chip.

4. **BW state nonlinearity at extremes**: The calibration curve (Finding 6) is roughly linear in the 22-28 state range but distorts at high DRAM saturation. What drives the nonlinearity?

5. **Concurrent execution mechanism**: Multi-model runs show zero interference (Finding 7), but we can't distinguish between true hardware concurrency and fast time-slicing from IOReport data alone. PerfTracer values would resolve this.

## Answered Questions

- ~~INT8 vs FP16 energy~~: **Answered** (Finding 5). ANE has a real INT8 path -- 1.6-1.9x faster, 0.5-0.8x energy. Not dequantized to FP16.
- ~~BW state calibration~~: **Answered** (Finding 6). ~2.5 GB/s per state in the linear range (states 22-28), with distortion at extremes.
- ~~Throttle behavior under sustained load~~: **Answered** (Finding 8). Zero throttle events across 10 minutes continuous load on passively-cooled M5 Air.

## License

MIT

## Related

- [maderix/ANE](https://github.com/maderix/ANE) -- ANE reverse engineering, _ANEPerformanceStats discovery, SRAM cliff wall-clock measurements
- [hollance/neural-engine](https://github.com/hollance/neural-engine) -- CoreML compilation behavior and ANE operator documentation
- [ANEMLL/Anemll](https://github.com/ANEMLL/Anemll) -- LLM to CoreML conversion pipeline for ANE inference
- [four-path-mlx](https://github.com/MidasMulli/four-path-mlx) -- Multi-source speculative decoding on Apple Silicon
- [orion-ane](https://github.com/MidasMulli/orion-ane) -- Phantom agent with three-tier ANE architecture
- [gdn-coreml](https://github.com/MidasMulli/gdn-coreml) -- GatedDeltaNet SSM to CoreML converter
- [dual-path-inference](https://github.com/MidasMulli/dual-path-inference) -- GPU+ANE concurrency proof-of-concept (archived)
