#!/usr/bin/env python3
"""
Measure ANE hardware behavior during CoreML inference via IOReport.

Captures 32-state bandwidth histograms, energy counters, and throttle
state residencies WHILE running a CoreML model. No root required.

Usage:
    python3 measure.py <model.mlpackage>           # Measure a model
    python3 measure.py <model.mlpackage> --idle     # Include idle baseline
    python3 measure.py <model.mlpackage> -n 200     # 200 iterations
    python3 measure.py --discover                   # List all ANE channels

Requires: macOS with Apple Silicon (M1+), coremltools, numpy
"""

import ctypes
import time
import argparse
import json
import sys
import os

# ── IOReport bindings ─────────────────────────────────────────

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
    ('IOReportChannelGetGroup', ctypes.c_void_p, [ctypes.c_void_p]),
    ('IOReportChannelGetSubGroup', ctypes.c_void_p, [ctypes.c_void_p]),
    ('IOReportChannelGetFormat', ctypes.c_int, [ctypes.c_void_p]),
    ('IOReportStateGetCount', ctypes.c_int, [ctypes.c_void_p]),
    ('IOReportStateGetResidency', ctypes.c_uint64,
     [ctypes.c_void_p, ctypes.c_int]),
    ('IOReportStateGetInTransitions', ctypes.c_uint64,
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
    if not p:
        return '?'
    s = CF.CFStringGetCStringPtr(p, 0x08000100)
    if s:
        return s.decode()
    buf = ctypes.create_string_buffer(512)
    if CF.CFStringGetCString(p, buf, 512, 0x08000100):
        return buf.value.decode()
    return '?'


# ── IOReport sampler ──────────────────────────────────────────

ENERGY_NAMES = {'ANE', 'GPU', 'DRAM', 'DCS', 'AMCC', 'CPU Energy'}

ANE_BW_CHANNELS = {
    ('ANE0 RD', 'AF BW'), ('ANE0 WR', 'AF BW'),
    ('ANE0 RD+WR', 'AF BW'),
    ('ANE0 RD', 'DCS BW'), ('ANE0 WR', 'DCS BW'),
    ('ANE0 RD+WR', 'DCS BW'),
    ('SOC-NI3 ANE UP', 'SOC-NI Util BW'),
    ('SOC-NI3 ALL', 'SOC-NI Util BW'),
    ('ANE-AF-BW', 'SOC Floor'), ('ANE-DCS-BW', 'DCS Floor'),
    ('ANE0', 'SOC Floor'), ('ANE0', 'DCS Floor'),
    ('AMCC RD+WR', 'DCS BW'), ('AMCC RD', 'DCS BW'),
    ('AMCC WR', 'DCS BW'),
    ('AGX RD+WR', 'DCS BW'),
}


class Sampler:
    """IOReport sampler for ANE bandwidth and energy channels."""

    def __init__(self):
        self.subs = []
        for g in ['Energy Model', 'PMP', 'SoC Stats']:
            gstr = cfstr(g)
            ch = lib.IOReportCopyChannelsInGroup(gstr, None, None)
            if not ch:
                continue
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
        """Extract energy, bandwidth histograms, and throttle events."""
        results = {'energy': {}, 'bw': {}, 'throttle': {}}
        key_cf = cfstr('IOReportChannels')

        for (g1, s1), (g2, s2) in zip(before, after):
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
                subgroup = pystr(lib.IOReportChannelGetSubGroup(c))
                fmt = lib.IOReportChannelGetFormat(c)

                if fmt == 1 and name in ENERGY_NAMES:
                    results['energy'][name] = \
                        lib.IOReportSimpleGetIntegerValue(c)

                if fmt == 2 and (name, subgroup) in ANE_BW_CHANNELS:
                    nstates = lib.IOReportStateGetCount(c)
                    res_list = [lib.IOReportStateGetResidency(c, s)
                                for s in range(nstates)]
                    trans_list = [lib.IOReportStateGetInTransitions(c, s)
                                 for s in range(nstates)]
                    total = sum(res_list)
                    active = sum(res_list[1:])
                    avg_st = (sum(i * r for i, r in enumerate(res_list))
                              / total if total > 0 else 0)
                    peak = max((i for i, r in enumerate(res_list) if r > 0),
                               default=0)
                    ch_key = f"{name}|{subgroup}"
                    results['bw'][ch_key] = {
                        'utilization': active / total if total > 0 else 0,
                        'avg_state': round(avg_st, 1),
                        'peak_state': peak,
                        'total_transitions': sum(trans_list),
                        'residencies': res_list,
                    }

                if 'THROTTLE' in name and fmt == 2:
                    nstates = lib.IOReportStateGetCount(c)
                    for s in range(1, nstates):
                        r = lib.IOReportStateGetResidency(c, s)
                        if r > 0:
                            results['throttle'][f"{name}_s{s}"] = r

        return results


def discover_channels():
    """List all ANE-related IOReport channels on this system."""
    keywords = ['ane', 'neural', 'dcs', 'fabric', 'ni3']
    channels = []

    for group in ['Energy Model', 'SoC Stats', 'PMP', 'GPU Stats']:
        gstr = cfstr(group)
        ch = lib.IOReportCopyChannelsInGroup(gstr, None, None)
        if not ch:
            continue
        mutable = CF.CFDictionaryCreateMutableCopy(None, 0, ch)
        outSub = ctypes.c_void_p(0)
        sub = lib.IOReportCreateSubscription(
            None, mutable, ctypes.byref(outSub), 0, None)
        if not sub:
            continue
        sampCh = outSub.value if outSub.value else mutable
        samp = lib.IOReportCreateSamples(sub, sampCh, None)
        if not samp:
            continue

        key = cfstr('IOReportChannels')
        arr = CF.CFDictionaryGetValue(samp, key)
        if not arr:
            continue

        for i in range(CF.CFArrayGetCount(arr)):
            c = CF.CFArrayGetValueAtIndex(arr, i)
            name = pystr(lib.IOReportChannelGetChannelName(c))
            if not any(k in name.lower() for k in keywords):
                continue
            subgroup = pystr(lib.IOReportChannelGetSubGroup(c))
            fmt = lib.IOReportChannelGetFormat(c)
            fmt_str = {1: 'integer', 2: 'state_histogram',
                       3: 'histogram'}.get(fmt, f'fmt{fmt}')
            nstates = ''
            if fmt == 2:
                nstates = f" ({lib.IOReportStateGetCount(c)} states)"
            channels.append({
                'name': name, 'group': group, 'subgroup': subgroup,
                'format': fmt_str, 'states': nstates})

    return channels


# ── Display ───────────────────────────────────────────────────

def print_results(results, label, n_iter=0, elapsed=0):
    print(f"\n{'='*70}")
    print(f"  {label}")
    if n_iter > 0:
        print(f"  {n_iter} iterations, {elapsed:.3f}s "
              f"({n_iter/elapsed:.1f}/s, {elapsed*1000/n_iter:.3f} ms/pred)")
    print(f"{'='*70}")

    # Energy
    if any(v > 0 for v in results['energy'].values()):
        print(f"\n  Energy (IOReport units):")
        for name, val in sorted(results['energy'].items()):
            if val > 0:
                print(f"    {name:<20s} {val:>12,}")

    # ANE bandwidth
    ane_keys = [k for k in results['bw']
                if k.startswith('ANE') or 'NI3 ANE' in k]
    if ane_keys:
        print(f"\n  ANE Bandwidth:")
        for k in sorted(ane_keys):
            h = results['bw'][k]
            label_str = k.replace('|', ' @ ')
            print(f"    {label_str:<45s}  "
                  f"util={h['utilization']:.1%}  "
                  f"avg={h['avg_state']:.1f}/31  "
                  f"peak={h['peak_state']}")

    # System bandwidth
    sys_keys = [k for k in results['bw']
                if k.startswith('AMCC') or k.startswith('AGX')]
    if sys_keys:
        print(f"\n  System Bandwidth:")
        for k in sorted(sys_keys):
            h = results['bw'][k]
            label_str = k.replace('|', ' @ ')
            print(f"    {label_str:<45s}  "
                  f"util={h['utilization']:.1%}  "
                  f"avg={h['avg_state']:.1f}")

    # Throttle
    if results['throttle']:
        print(f"\n  Throttle events:")
        for k, v in sorted(results['throttle'].items()):
            print(f"    {k:<50s} {v:,} ns")
    elif not results['throttle']:
        ane_e = results['energy'].get('ANE', 0)
        if ane_e > 0:
            print(f"\n  No throttle events detected.")

    # Summary metrics
    ane_rw = results['bw'].get('ANE0 RD+WR|DCS BW', {})
    ane_rd = results['bw'].get('ANE0 RD|DCS BW', {})
    ane_wr = results['bw'].get('ANE0 WR|DCS BW', {})
    if ane_rw.get('utilization', 0) > 0:
        est_bw = ane_rw['avg_state'] / 31 * 153.6  # M5 Air max
        print(f"\n  Derived metrics:")
        print(f"    Est. ANE DRAM bandwidth:  ~{est_bw:.0f} GB/s "
              f"(avg_state/31 * 153.6)")
        rd = ane_rd.get('avg_state', 0)
        wr = ane_wr.get('avg_state', 0)
        if wr > 0:
            print(f"    RD:WR ratio:             {rd/wr:.1f}:1")
        on_ane = True
    else:
        on_ane = False

    return on_ane


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Measure ANE hardware behavior during CoreML inference')
    parser.add_argument('model', nargs='?', help='Path to .mlpackage')
    parser.add_argument('-n', '--iterations', type=int, default=100)
    parser.add_argument('--idle', action='store_true',
                        help='Include idle baseline measurement')
    parser.add_argument('--discover', action='store_true',
                        help='List all ANE-related IOReport channels')
    parser.add_argument('-o', '--output', help='Save results to JSON')
    args = parser.parse_args()

    if args.discover:
        channels = discover_channels()
        print(f"ANE-related IOReport channels ({len(channels)}):\n")
        print(f"  {'Name':<45s}  {'Group':<15s}  {'Subgroup':<20s}  Format")
        print(f"  {'─'*100}")
        for c in channels:
            print(f"  {c['name']:<45s}  {c['group']:<15s}  "
                  f"{c['subgroup']:<20s}  {c['format']}{c['states']}")
        return

    if not args.model:
        parser.print_help()
        return

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        sys.exit(1)

    import numpy as np

    sampler = Sampler()
    all_results = {}

    # Idle baseline
    if args.idle:
        print("Measuring idle baseline (1s)...")
        before = sampler.capture()
        time.sleep(1.0)
        after = sampler.capture()
        idle = sampler.delta(before, after)
        print_results(idle, "Idle Baseline (1s)")
        all_results['idle'] = idle

    # Load model
    import coremltools as ct
    print(f"\nLoading {args.model}...")
    ml = ct.models.MLModel(args.model, compute_units=ct.ComputeUnit.CPU_AND_NE)
    spec = ml.get_spec()
    inputs = {}
    for inp in spec.description.input:
        shape = list(inp.type.multiArrayType.shape)
        inputs[inp.name] = np.random.randn(*shape).astype(np.float16)

    # Warmup
    print(f"Warming up (10 iterations)...")
    for _ in range(10):
        ml.predict(inputs)

    # Measure
    n = args.iterations
    print(f"Measuring ({n} iterations)...")
    before = sampler.capture()
    t0 = time.perf_counter()
    for _ in range(n):
        ml.predict(inputs)
    elapsed = time.perf_counter() - t0
    after = sampler.capture()

    results = sampler.delta(before, after)
    on_ane = print_results(
        results,
        f"{os.path.basename(args.model)}",
        n_iter=n, elapsed=elapsed)

    if not on_ane:
        print(f"\n  NOTE: ANE bandwidth = 0%. Model is likely running on CPU.")
        print(f"  CoreML routes to ANE only when total model weight > ~37MB")
        print(f"  or per-op compute exceeds dispatch overhead (~0.1ms).")

    all_results['inference'] = results
    all_results['timing'] = {
        'model': args.model, 'n_iter': n,
        'elapsed_s': round(elapsed, 3),
        'ms_per_pred': round(elapsed * 1000 / n, 3),
    }

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
