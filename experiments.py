#!/usr/bin/env python3
"""
Reproduce the ANE hardware characterization experiments.

Builds synthetic CoreML models at controlled sizes, measures ANE behavior
via IOReport bandwidth histograms, and reports:

  1. SRAM boundary detection (32MB cliff)
  2. Conv vs Linear comparison (identical on ANE)
  3. CoreML dispatch threshold (when ANE activates)
  4. Scaling with layers and sequence length

Usage:
    python3 experiments.py                # Run all experiments
    python3 experiments.py sram           # Just SRAM boundary
    python3 experiments.py conv           # Just conv vs linear
    python3 experiments.py dispatch       # Just dispatch threshold
    python3 experiments.py scaling        # Just scaling

Requires: macOS with Apple Silicon (M1+), torch, coremltools, numpy
"""

import ctypes
import time
import json
import sys
import os
import numpy as np

# ── IOReport (same bindings as measure.py) ────────────────────

lib = ctypes.cdll.LoadLibrary('/usr/lib/libIOReport.dylib')
CF = ctypes.cdll.LoadLibrary(
    '/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation')

for name, res, args in [
    ('IOReportCopyChannelsInGroup', ctypes.c_void_p, [ctypes.c_void_p]*3),
    ('IOReportGetChannelCount', ctypes.c_int, [ctypes.c_void_p]),
    ('IOReportCreateSubscription', ctypes.c_void_p,
     [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
      ctypes.c_uint64, ctypes.c_void_p]),
    ('IOReportCreateSamples', ctypes.c_void_p, [ctypes.c_void_p]*3),
    ('IOReportCreateSamplesDelta', ctypes.c_void_p, [ctypes.c_void_p]*3),
    ('IOReportSimpleGetIntegerValue', ctypes.c_long, [ctypes.c_void_p]),
    ('IOReportChannelGetChannelName', ctypes.c_void_p, [ctypes.c_void_p]),
    ('IOReportChannelGetSubGroup', ctypes.c_void_p, [ctypes.c_void_p]),
    ('IOReportChannelGetFormat', ctypes.c_int, [ctypes.c_void_p]),
    ('IOReportStateGetCount', ctypes.c_int, [ctypes.c_void_p]),
    ('IOReportStateGetResidency', ctypes.c_uint64,
     [ctypes.c_void_p, ctypes.c_int]),
]:
    fn = getattr(lib, name); fn.restype = res; fn.argtypes = args

for name, res, args in [
    ('CFArrayGetCount', ctypes.c_long, [ctypes.c_void_p]),
    ('CFArrayGetValueAtIndex', ctypes.c_void_p,
     [ctypes.c_void_p, ctypes.c_long]),
    ('CFDictionaryGetValue', ctypes.c_void_p, [ctypes.c_void_p]*2),
    ('CFDictionaryCreateMutableCopy', ctypes.c_void_p,
     [ctypes.c_void_p, ctypes.c_long, ctypes.c_void_p]),
    ('CFStringCreateWithCString', ctypes.c_void_p,
     [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]),
    ('CFStringGetCStringPtr', ctypes.c_char_p,
     [ctypes.c_void_p, ctypes.c_uint32]),
    ('CFStringGetCString', ctypes.c_bool,
     [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]),
]:
    fn = getattr(CF, name); fn.restype = res; fn.argtypes = args

def cfstr(s):
    return CF.CFStringCreateWithCString(None, s.encode(), 0x08000100)

def pystr(p):
    if not p: return '?'
    s = CF.CFStringGetCStringPtr(p, 0x08000100)
    if s: return s.decode()
    buf = ctypes.create_string_buffer(512)
    if CF.CFStringGetCString(p, buf, 512, 0x08000100):
        return buf.value.decode()
    return '?'


ENERGY_NAMES = {'ANE', 'GPU', 'DRAM', 'DCS', 'AMCC', 'CPU Energy'}
BW_KEYS = {
    ('ANE0 RD+WR', 'DCS BW'), ('ANE0 RD', 'DCS BW'),
    ('ANE0 WR', 'DCS BW'), ('AMCC RD+WR', 'DCS BW'),
}


class Sampler:
    def __init__(self):
        self.subs = []
        for g in ['Energy Model', 'PMP']:
            gstr = cfstr(g)
            ch = lib.IOReportCopyChannelsInGroup(gstr, None, None)
            if not ch: continue
            mutable = CF.CFDictionaryCreateMutableCopy(None, 0, ch)
            outSub = ctypes.c_void_p(0)
            sub = lib.IOReportCreateSubscription(
                None, mutable, ctypes.byref(outSub), 0, None)
            if sub:
                self.subs.append((
                    g, sub, outSub.value if outSub.value else mutable))

    def capture(self):
        return [(g, lib.IOReportCreateSamples(sub, sch, None))
                for g, sub, sch in self.subs]

    def delta(self, before, after):
        results = {'energy': {}, 'bw': {}}
        key_cf = cfstr('IOReportChannels')
        for (g1, s1), (g2, s2) in zip(before, after):
            if not s1 or not s2: continue
            d = lib.IOReportCreateSamplesDelta(s1, s2, None)
            if not d: continue
            arr = CF.CFDictionaryGetValue(d, key_cf)
            if not arr: continue
            for i in range(CF.CFArrayGetCount(arr)):
                c = CF.CFArrayGetValueAtIndex(arr, i)
                name = pystr(lib.IOReportChannelGetChannelName(c))
                subgroup = pystr(lib.IOReportChannelGetSubGroup(c))
                fmt = lib.IOReportChannelGetFormat(c)
                if fmt == 1 and name in ENERGY_NAMES:
                    results['energy'][name] = \
                        lib.IOReportSimpleGetIntegerValue(c)
                if fmt == 2 and (name, subgroup) in BW_KEYS:
                    nstates = lib.IOReportStateGetCount(c)
                    res_list = [lib.IOReportStateGetResidency(c, s)
                                for s in range(nstates)]
                    total = sum(res_list)
                    active = sum(res_list[1:])
                    avg_st = (sum(i * r for i, r in enumerate(res_list))
                              / total if total > 0 else 0)
                    peak = max(
                        (i for i, r in enumerate(res_list) if r > 0),
                        default=0)
                    results['bw'][f"{name}|{subgroup}"] = {
                        'util': active / total if total > 0 else 0,
                        'avg': round(avg_st, 1),
                        'peak': peak,
                    }
        return results


# ── Model building ────────────────────────────────────────────

def build_model(dim, n_layers=1, op='conv1d', seq_len=1, quantize=None):
    """Build a CoreML model for ANE testing.

    Args:
        quantize: None for FP16, 'int8' for linear INT8 quantization
    """
    import torch
    import torch.nn as nn
    import coremltools as ct

    if op == 'conv1d':
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList(
                    [nn.Conv1d(dim, dim, 1, bias=False)
                     for _ in range(n_layers)])
            def forward(self, x):
                for l in self.layers: x = l(x)
                return x
        model = M().eval().half()
        example = torch.randn(1, dim, seq_len).half()
    elif op == 'conv2d':
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList(
                    [nn.Conv2d(dim, dim, 1, bias=False)
                     for _ in range(n_layers)])
            def forward(self, x):
                for l in self.layers: x = l(x)
                return x
        model = M().eval().half()
        example = torch.randn(1, dim, 1, 1).half()
    else:  # linear
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList(
                    [nn.Linear(dim, dim, bias=False)
                     for _ in range(n_layers)])
            def forward(self, x):
                for l in self.layers: x = l(x)
                return x
        model = M().eval().half()
        example = torch.randn(1, seq_len, dim).half()

    traced = torch.jit.trace(model, example)
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name='x', shape=example.shape)],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
    )

    if quantize == 'int8':
        from coremltools.optimize.coreml import (
            OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights)
        op_config = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
        config = OptimizationConfig(global_config=op_config)
        ml = linear_quantize_weights(ml, config=config)

    q_tag = f"_{quantize}" if quantize else ""
    path = f"/tmp/ane_perf_{op}_{dim}x{n_layers}_s{seq_len}{q_tag}.mlpackage"
    ml.save(path)

    # Get actual file size
    import subprocess
    size = subprocess.check_output(
        ['du', '-sk', path]).decode().split()[0]
    size_mb = int(size) / 1024

    return path, size_mb


def run_trial(sampler, path, n_iter=100, warmup=10):
    import coremltools as ct
    ml = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    spec = ml.get_spec()
    inputs = {}
    for inp in spec.description.input:
        shape = list(inp.type.multiArrayType.shape)
        inputs[inp.name] = np.random.randn(*shape).astype(np.float16)
    for _ in range(warmup):
        ml.predict(inputs)

    before = sampler.capture()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        ml.predict(inputs)
    elapsed = time.perf_counter() - t0
    after = sampler.capture()

    r = sampler.delta(before, after)
    r['ms'] = round(elapsed * 1000 / n_iter, 3)
    r['pps'] = round(n_iter / elapsed, 1)
    return r


def get_ane(r):
    """Extract key ANE metrics from a trial result."""
    return {
        'energy': r['energy'].get('ANE', 0),
        'util': r['bw'].get('ANE0 RD+WR|DCS BW', {}).get('util', 0),
        'avg': r['bw'].get('ANE0 RD+WR|DCS BW', {}).get('avg', 0),
        'peak': r['bw'].get('ANE0 RD+WR|DCS BW', {}).get('peak', 0),
        'dram_util': r['bw'].get('AMCC RD+WR|DCS BW', {}).get('util', 0),
    }


# ── Experiments ───────────────────────────────────────────────

def exp_sram(sampler):
    """SRAM boundary: 8-layer Conv1d at increasing per-layer size."""
    print("\n" + "=" * 85)
    print("  SRAM BOUNDARY DETECTION")
    print("  8-layer Conv1d, varying dim. ANE L2 SRAM = 32MB.")
    print("  Per-layer weight = dim^2 * 2 bytes (FP16).")
    print("=" * 85)

    dims = [512, 768, 1024, 1536, 2048, 2560, 3072, 3584, 4096]
    results = {}

    print(f"\n  {'Dim':>5}  {'Layer MB':>9}  {'ms/pred':>10}  "
          f"{'ANE nrg':>8}  {'ANE util':>9}  {'ANE avg':>8}  "
          f"{'DRAM util':>10}  Zone")
    print(f"  {'─'*85}")

    for dim in dims:
        layer_mb = dim * dim * 2 / 1e6
        try:
            path, _ = build_model(dim, n_layers=8, op='conv1d')
            r = run_trial(sampler, path, n_iter=100)
            a = get_ane(r)
            results[dim] = {'ms': r['ms'], **a}

            if a['util'] == 0:
                zone = "CPU"
            elif layer_mb < 19:
                zone = "SRAM"
            elif layer_mb < 34:
                zone = "Thrashing"
            else:
                zone = "Streaming"

            print(f"  {dim:>5}  {layer_mb:>8.1f}M  {r['ms']:>10.3f}  "
                  f"{a['energy']:>8,}  {a['util']:>8.1%}  "
                  f"{a['avg']:>8.1f}  {a['dram_util']:>9.1%}  {zone}")
        except Exception as e:
            print(f"  {dim:>5}  ERROR: {e}")

    return results


def exp_conv(sampler):
    """Conv vs Linear: same weights, different op type."""
    print("\n" + "=" * 85)
    print("  CONV vs LINEAR on ANE")
    print("  8 layers, same dim^2 weights. Three op types.")
    print("  Models large enough to trigger ANE dispatch (dim >= 1536).")
    print("=" * 85)

    dims = [1536, 2048, 2560, 3072]
    results = {}

    print(f"\n  {'Op':>8}  {'Dim':>5}  {'ms/pred':>10}  "
          f"{'ANE nrg':>8}  {'ANE util':>9}  {'ANE avg':>8}")
    print(f"  {'─'*60}")

    for dim in dims:
        row = {}
        for op in ['linear', 'conv1d', 'conv2d']:
            try:
                path, _ = build_model(dim, n_layers=8, op=op)
                r = run_trial(sampler, path, n_iter=100)
                a = get_ane(r)
                row[op] = {'ms': r['ms'], **a}
                print(f"  {op:>8}  {dim:>5}  {r['ms']:>10.3f}  "
                      f"{a['energy']:>8,}  {a['util']:>8.1%}  "
                      f"{a['avg']:>8.1f}")
            except Exception as e:
                print(f"  {op:>8}  {dim:>5}  ERROR: {e}")

        if 'linear' in row and 'conv1d' in row:
            ratio = row['linear']['ms'] / row['conv1d']['ms']
            print(f"  {'':>8}  {dim:>5}  "
                  f"Conv1d vs Linear: {ratio:.2f}x")
        print()
        results[dim] = row

    return results


def exp_dispatch(sampler):
    """Find the threshold where CoreML starts using ANE."""
    print("\n" + "=" * 85)
    print("  CoreML ANE DISPATCH THRESHOLD")
    print("  Single Conv2d at increasing dim. Where does ANE activate?")
    print("=" * 85)

    dims = [512, 1024, 1536, 2048, 2560, 3072, 4096, 5120, 6144, 8192]
    results = {}

    print(f"\n  {'Dim':>5}  {'Weight MB':>10}  {'ms/pred':>10}  "
          f"{'ANE nrg':>8}  {'ANE util':>9}  {'on ANE?':>8}")
    print(f"  {'─'*65}")

    for dim in dims:
        weight_mb = dim * dim * 2 / 1e6
        try:
            path, _ = build_model(dim, n_layers=1, op='conv2d')
            r = run_trial(sampler, path, n_iter=200)
            a = get_ane(r)
            results[dim] = {'ms': r['ms'], 'weight_mb': weight_mb, **a}

            on_ane = "YES" if a['energy'] > 0 or a['util'] > 0.01 else "no"
            print(f"  {dim:>5}  {weight_mb:>9.1f}M  {r['ms']:>10.3f}  "
                  f"{a['energy']:>8,}  {a['util']:>8.1%}  {on_ane:>8}")
        except Exception as e:
            print(f"  {dim:>5}  ERROR: {e}")

    return results


def exp_scaling(sampler):
    """How ANE performance scales with layers and sequence length."""
    print("\n" + "=" * 85)
    print("  SCALING BEHAVIOR")
    print("  Fixed dim=2048, 8 layers. Vary sequence length.")
    print("=" * 85)

    results = {}

    print(f"\n  {'SeqLen':>7}  {'ms/pred':>10}  {'ANE nrg':>8}  "
          f"{'ANE util':>9}  {'DRAM util':>10}")
    print(f"  {'─'*55}")

    for seq_len in [1, 4, 16, 64, 256]:
        try:
            path, _ = build_model(2048, n_layers=8, op='conv1d',
                               seq_len=seq_len)
            r = run_trial(sampler, path, n_iter=100)
            a = get_ane(r)
            results[seq_len] = {'ms': r['ms'], **a}
            print(f"  {seq_len:>7}  {r['ms']:>10.3f}  {a['energy']:>8,}  "
                  f"{a['util']:>8.1%}  {a['dram_util']:>9.1%}")
        except Exception as e:
            print(f"  {seq_len:>7}  ERROR: {e}")

    return results


def exp_calibrate(sampler):
    """BW state-to-GB/s calibration curve.

    Build single-layer models at 20 sizes (5MB to 200MB). For each:
    - Record latency and avg BW state
    - Calculate empirical bandwidth: weight_bytes / latency_seconds
    - Plot state vs GB/s for a calibration curve
    """
    print("\n" + "=" * 85)
    print("  BW STATE CALIBRATION")
    print("  Single-layer Conv1d at 20 sizes. BW = weight_bytes / latency.")
    print("  Maps IOReport BW states (0-31) to actual GB/s.")
    print("=" * 85)

    # 20 dims chosen to span ~5MB to ~200MB per layer
    # dim^2 * 2 bytes (FP16): dim=1600→5.1MB, dim=10000→200MB
    dims = [1600, 2000, 2400, 2800, 3000, 3200, 3400, 3600,
            3800, 4000, 4200, 4500, 4800, 5120, 5600, 6144,
            7000, 8000, 9000, 10000]
    results = {}

    print(f"\n  {'Dim':>6}  {'Weight MB':>10}  {'ms/pred':>10}  "
          f"{'ANE avg':>8}  {'Est GB/s':>10}  {'ANE util':>9}  "
          f"{'DRAM util':>10}")
    print(f"  {'─'*80}")

    for dim in dims:
        weight_bytes = dim * dim * 2  # FP16
        weight_mb = weight_bytes / 1e6
        try:
            # Use 8 layers to ensure ANE dispatch, but measure per-layer
            n_layers = max(1, int(40 / weight_mb) + 1)  # enough total weight
            n_layers = min(n_layers, 8)
            path, disk_mb = build_model(dim, n_layers=n_layers, op='conv1d')
            r = run_trial(sampler, path, n_iter=100)
            a = get_ane(r)

            if a['util'] < 0.01:
                print(f"  {dim:>6}  {weight_mb:>9.1f}M  {r['ms']:>10.3f}  "
                      f"{'':>8}  {'':>10}  CPU       {'':>10}")
                continue

            # Total weight transferred = n_layers * weight_bytes
            # (ANE reads all weights from DRAM each prediction)
            total_bytes = n_layers * weight_bytes
            latency_s = r['ms'] / 1000
            est_gbps = total_bytes / latency_s / 1e9

            results[dim] = {
                'weight_mb': round(weight_mb, 1),
                'n_layers': n_layers,
                'total_mb': round(total_bytes / 1e6, 1),
                'ms': r['ms'],
                'avg_state': a['avg'],
                'est_gbps': round(est_gbps, 1),
                'util': a['util'],
                'dram_util': a['dram_util'],
            }

            print(f"  {dim:>6}  {weight_mb:>9.1f}M  {r['ms']:>10.3f}  "
                  f"{a['avg']:>7.1f}  {est_gbps:>9.1f}  "
                  f"{a['util']:>8.1%}  {a['dram_util']:>9.1%}")
        except Exception as e:
            print(f"  {dim:>6}  ERROR: {e}")

    # Print calibration summary
    if results:
        print(f"\n  Calibration curve (BW state → GB/s):")
        print(f"  {'Avg State':>10}  {'Est GB/s':>10}  {'Weight MB':>10}")
        print(f"  {'─'*35}")
        for dim in sorted(results, key=lambda d: results[d]['avg_state']):
            r = results[dim]
            print(f"  {r['avg_state']:>10.1f}  {r['est_gbps']:>10.1f}  "
                  f"{r['weight_mb']:>10.1f}")

    return results


def exp_int8(sampler):
    """INT8 vs FP16: same model, different quantization.

    Tests whether ANE has a real INT8 data path or dequantizes to FP16.
    Compare energy, latency, and bandwidth utilization.
    """
    print("\n" + "=" * 85)
    print("  INT8 vs FP16")
    print("  Same Conv1d models, 8 layers. FP16 vs INT8 linear quantization.")
    print("  If INT8 shows same bandwidth but lower latency → real INT8 path.")
    print("  If identical → ANE dequantizes INT8 to FP16 internally.")
    print("=" * 85)

    dims = [2048, 2560, 3072, 3584]
    results = {}

    print(f"\n  {'Quant':>6}  {'Dim':>5}  {'Disk MB':>8}  {'ms/pred':>10}  "
          f"{'ANE nrg':>8}  {'ANE util':>9}  {'ANE avg':>8}  "
          f"{'DRAM util':>10}")
    print(f"  {'─'*80}")

    for dim in dims:
        row = {}
        for quant in [None, 'int8']:
            label = quant or 'fp16'
            try:
                path, disk_mb = build_model(
                    dim, n_layers=8, op='conv1d', quantize=quant)
                r = run_trial(sampler, path, n_iter=100)
                a = get_ane(r)
                row[label] = {
                    'ms': r['ms'], 'disk_mb': round(disk_mb, 1), **a}

                print(f"  {label:>6}  {dim:>5}  {disk_mb:>7.1f}M  "
                      f"{r['ms']:>10.3f}  {a['energy']:>8,}  "
                      f"{a['util']:>8.1%}  {a['avg']:>8.1f}  "
                      f"{a['dram_util']:>9.1%}")
            except Exception as e:
                print(f"  {label:>6}  {dim:>5}  ERROR: {e}")

        if 'fp16' in row and 'int8' in row:
            speedup = row['fp16']['ms'] / row['int8']['ms']
            energy_ratio = (row['int8']['energy'] / row['fp16']['energy']
                           if row['fp16']['energy'] > 0 else 0)
            size_ratio = (row['int8']['disk_mb'] / row['fp16']['disk_mb']
                         if row['fp16']['disk_mb'] > 0 else 0)
            print(f"  {'':>6}  {dim:>5}  INT8 vs FP16: "
                  f"{speedup:.2f}x speed, "
                  f"{energy_ratio:.2f}x energy, "
                  f"{size_ratio:.2f}x size")
        print()
        results[dim] = row

    return results


def exp_thermal(sampler):
    """Sustained thermal test: 10 minutes continuous ANE load.

    Run an 8-layer Conv1d model continuously, sampling IOReport every 5 seconds.
    Track energy and throttle state over time to detect thermal throttling.
    """
    print("\n" + "=" * 85)
    print("  SUSTAINED THERMAL TEST")
    print("  8-layer Conv1d dim=3072, continuous for 10 minutes.")
    print("  IOReport sampled every 5 seconds. Watching for throttle events.")
    print("=" * 85)

    import coremltools as ct

    path, _ = build_model(3072, n_layers=8, op='conv1d')
    ml = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    spec = ml.get_spec()
    inputs = {}
    for inp in spec.description.input:
        shape = list(inp.type.multiArrayType.shape)
        inputs[inp.name] = np.random.randn(*shape).astype(np.float16)

    # Warmup
    for _ in range(20):
        ml.predict(inputs)

    duration_s = 600  # 10 minutes
    interval_s = 5
    n_samples = duration_s // interval_s
    results = []

    # Also subscribe to SoC Stats for throttle channels
    throttle_sampler = Sampler()
    # Add SoC Stats group for throttle channels
    gstr = cfstr('SoC Stats')
    ch = lib.IOReportCopyChannelsInGroup(gstr, None, None)
    if ch:
        mutable = CF.CFDictionaryCreateMutableCopy(None, 0, ch)
        outSub = ctypes.c_void_p(0)
        sub = lib.IOReportCreateSubscription(
            None, mutable, ctypes.byref(outSub), 0, None)
        if sub:
            throttle_sampler.subs.append((
                'SoC Stats', sub,
                outSub.value if outSub.value else mutable))

    print(f"\n  {'Time':>6}  {'Preds':>6}  {'ms/pred':>10}  "
          f"{'ANE nrg':>8}  {'ANE util':>9}  {'ANE avg':>8}  "
          f"{'Throttle':>10}")
    print(f"  {'─'*70}")

    t_start = time.time()
    for sample_i in range(n_samples):
        before = sampler.capture()
        tb = throttle_sampler.capture()
        t0 = time.perf_counter()

        # Run predictions for interval_s
        count = 0
        while time.perf_counter() - t0 < interval_s:
            ml.predict(inputs)
            count += 1

        elapsed = time.perf_counter() - t0
        after = sampler.capture()
        ta = throttle_sampler.capture()

        r = sampler.delta(before, after)
        a = get_ane(r)
        ms = elapsed * 1000 / count if count > 0 else 0

        # Check throttle from SoC Stats
        throttle_r = throttle_sampler.delta(tb, ta)
        throttle_ns = 0
        key_cf = cfstr('IOReportChannels')
        for (g1, s1), (g2, s2) in zip(tb, ta):
            if not s1 or not s2:
                continue
            d = lib.IOReportCreateSamplesDelta(s1, s2, None)
            if not d:
                continue
            arr = CF.CFDictionaryGetValue(d, key_cf)
            if not arr:
                continue
            for i in range(CF.CFArrayGetCount(arr)):
                c = CF.CFArrayGetValueAtIndex(arr, i)
                name = pystr(lib.IOReportChannelGetChannelName(c))
                fmt = lib.IOReportChannelGetFormat(c)
                if 'THROTTLE' in name and 'ANE' in name and fmt == 2:
                    nstates = lib.IOReportStateGetCount(c)
                    for s in range(1, nstates):
                        throttle_ns += lib.IOReportStateGetResidency(c, s)

        elapsed_min = (time.time() - t_start) / 60
        throttle_str = f"{throttle_ns/1e6:.1f}ms" if throttle_ns > 0 else "none"

        row = {
            'time_s': round(elapsed_min * 60),
            'preds': count,
            'ms': round(ms, 3),
            'energy': a['energy'],
            'util': a['util'],
            'avg': a['avg'],
            'throttle_ns': throttle_ns,
        }
        results.append(row)

        print(f"  {elapsed_min:>5.1f}m  {count:>6}  {ms:>10.3f}  "
              f"{a['energy']:>8,}  {a['util']:>8.1%}  "
              f"{a['avg']:>8.1f}  {throttle_str:>10}")

    # Summary
    if results:
        ms_vals = [r['ms'] for r in results]
        energy_vals = [r['energy'] for r in results]
        throttle_total = sum(r['throttle_ns'] for r in results)
        print(f"\n  Summary ({len(results)} samples over "
              f"{duration_s/60:.0f} minutes):")
        print(f"    Latency:  {min(ms_vals):.3f} - {max(ms_vals):.3f} ms/pred "
              f"(range: {max(ms_vals)-min(ms_vals):.3f}ms)")
        print(f"    Energy:   {min(energy_vals):,} - {max(energy_vals):,} "
              f"per interval")
        if throttle_total > 0:
            print(f"    Throttle: {throttle_total/1e9:.1f}s total")
        else:
            print(f"    Throttle: NONE detected across full run")

    return results


def exp_multimodel(sampler):
    """Multi-model interference: does ANE serialize or allow concurrency?

    Run two CoreML models simultaneously. Model A runs continuously on a
    background thread. Model B runs measured predictions on the main thread.
    Compare Model B's latency with and without Model A running.
    """
    print("\n" + "=" * 85)
    print("  MULTI-MODEL INTERFERENCE")
    print("  Model A (dim=2048) background thread + Model B (dim=3072) measured.")
    print("  Compare Model B latency with/without background load.")
    print("=" * 85)

    import coremltools as ct
    import threading

    # Build two different models
    path_a, _ = build_model(2048, n_layers=8, op='conv1d')
    path_b, _ = build_model(3072, n_layers=8, op='conv1d')

    ml_a = ct.models.MLModel(path_a, compute_units=ct.ComputeUnit.CPU_AND_NE)
    ml_b = ct.models.MLModel(path_b, compute_units=ct.ComputeUnit.CPU_AND_NE)

    def make_inputs(ml):
        spec = ml.get_spec()
        inputs = {}
        for inp in spec.description.input:
            shape = list(inp.type.multiArrayType.shape)
            inputs[inp.name] = np.random.randn(*shape).astype(np.float16)
        return inputs

    inputs_a = make_inputs(ml_a)
    inputs_b = make_inputs(ml_b)

    # Warmup both
    for _ in range(10):
        ml_a.predict(inputs_a)
        ml_b.predict(inputs_b)

    results = {}

    # Baseline: Model B alone
    print(f"\n  Model B alone (100 iterations)...")
    before = sampler.capture()
    t0 = time.perf_counter()
    for _ in range(100):
        ml_b.predict(inputs_b)
    elapsed = time.perf_counter() - t0
    after = sampler.capture()
    r_alone = sampler.delta(before, after)
    a_alone = get_ane(r_alone)
    ms_alone = elapsed * 1000 / 100

    print(f"    ms/pred: {ms_alone:.3f}  ANE util: {a_alone['util']:.1%}  "
          f"ANE avg: {a_alone['avg']:.1f}  energy: {a_alone['energy']:,}")
    results['alone'] = {'ms': round(ms_alone, 3), **a_alone}

    # Background load: Model A on background thread
    stop_flag = threading.Event()
    bg_count = [0]

    def background_load():
        while not stop_flag.is_set():
            ml_a.predict(inputs_a)
            bg_count[0] += 1

    print(f"\n  Model B with Model A background (100 iterations)...")
    bg_thread = threading.Thread(target=background_load, daemon=True)
    bg_thread.start()
    time.sleep(0.5)  # let background warm up

    before = sampler.capture()
    t0 = time.perf_counter()
    for _ in range(100):
        ml_b.predict(inputs_b)
    elapsed = time.perf_counter() - t0
    after = sampler.capture()

    stop_flag.set()
    bg_thread.join(timeout=5)

    r_concurrent = sampler.delta(before, after)
    a_concurrent = get_ane(r_concurrent)
    ms_concurrent = elapsed * 1000 / 100

    print(f"    ms/pred: {ms_concurrent:.3f}  ANE util: {a_concurrent['util']:.1%}  "
          f"ANE avg: {a_concurrent['avg']:.1f}  energy: {a_concurrent['energy']:,}")
    print(f"    Background model completed {bg_count[0]} predictions during test")
    results['concurrent'] = {
        'ms': round(ms_concurrent, 3),
        'bg_preds': bg_count[0],
        **a_concurrent,
    }

    # Analysis
    slowdown = ms_concurrent / ms_alone if ms_alone > 0 else 0
    print(f"\n  Interference: {slowdown:.2f}x slowdown "
          f"({ms_alone:.3f} → {ms_concurrent:.3f} ms/pred)")
    if slowdown > 1.8:
        print(f"  → ANE likely SERIALIZES execution (near 2x = time-shared)")
    elif slowdown > 1.2:
        print(f"  → Partial interference (shared bandwidth, not fully serial)")
    else:
        print(f"  → Minimal interference (separate queues or concurrent execution)")
    results['slowdown'] = round(slowdown, 2)

    return results


# ── Main ──────────────────────────────────────────────────────

def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    sampler = Sampler()
    all_results = {}

    experiments = {
        'sram': ('SRAM boundary', exp_sram),
        'conv': ('Conv vs Linear', exp_conv),
        'dispatch': ('Dispatch threshold', exp_dispatch),
        'scaling': ('Scaling', exp_scaling),
        'calibrate': ('BW state calibration', exp_calibrate),
        'int8': ('INT8 vs FP16', exp_int8),
        'thermal': ('Sustained thermal', exp_thermal),
        'multimodel': ('Multi-model interference', exp_multimodel),
    }

    for key, (name, fn) in experiments.items():
        if which != 'all' and key != which:
            continue
        try:
            all_results[key] = fn(sampler)
        except Exception as e:
            print(f"\n  {name} FAILED: {e}")
            import traceback
            traceback.print_exc()

    out_path = os.path.join(
        os.path.dirname(__file__) or '.', f'data/experiments.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
