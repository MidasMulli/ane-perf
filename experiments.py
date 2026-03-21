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

def build_model(dim, n_layers=1, op='conv1d', seq_len=1):
    """Build a CoreML model for ANE testing."""
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
    path = f"/tmp/ane_perf_{op}_{dim}x{n_layers}_s{seq_len}.mlpackage"
    ml.save(path)
    return path


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
            path = build_model(dim, n_layers=8, op='conv1d')
            r = run_trial(sampler, path, n_iter=100)
            a = get_ane(r)
            results[dim] = {'ms': r['ms'], **a}

            if a['util'] == 0:
                zone = "CPU"
            elif layer_mb < 30:
                zone = "SRAM"
            elif layer_mb < 35:
                zone = "BOUNDARY"
            else:
                zone = "SPILL"

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
                path = build_model(dim, n_layers=8, op=op)
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
            path = build_model(dim, n_layers=1, op='conv2d')
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
            path = build_model(2048, n_layers=8, op='conv1d',
                               seq_len=seq_len)
            r = run_trial(sampler, path, n_iter=100)
            a = get_ane(r)
            results[seq_len] = {'ms': r['ms'], **a}
            print(f"  {seq_len:>7}  {r['ms']:>10.3f}  {a['energy']:>8,}  "
                  f"{a['util']:>8.1%}  {a['dram_util']:>9.1%}")
        except Exception as e:
            print(f"  {seq_len:>7}  ERROR: {e}")

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
