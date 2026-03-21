#!/usr/bin/env python3
"""
CoreML ANE profiler — per-operation execution scheduling and hardware measurements.

For any CoreML model, shows:
  1. Per-op backend dispatch (ANE vs BNNS vs MPS Graph vs CPU)
  2. Per-op estimated run time per backend (CoreML cost model)
  3. IOReport hardware counters during inference (energy, bandwidth, throttle)
  4. Total execution timing and predictions/sec

This is the framework-level path to ANE profiling — no entitlements required.

Usage:
    python3 profile.py model.mlpackage              # Profile a CoreML model
    python3 profile.py model.mlpackage --iters 200   # More iterations for accuracy
    python3 profile.py model.mlpackage --json         # JSON output
    python3 profile.py --build 1024 16 1              # Build and profile a test model

Requires: macOS with Apple Silicon, coremltools (for --build only), numpy
"""

import ctypes
import ctypes.util
import os
import sys
import time
import json as json_mod
import numpy as np

# ── ObjC Runtime ─────────────────────────────────────────────

objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library('objc'))

for name, res, args in [
    ('objc_getClass', ctypes.c_void_p, [ctypes.c_char_p]),
    ('sel_registerName', ctypes.c_void_p, [ctypes.c_char_p]),
    ('objc_msgSend', ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p]),
    ('class_copyIvarList', ctypes.POINTER(ctypes.c_void_p),
     [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]),
    ('ivar_getName', ctypes.c_char_p, [ctypes.c_void_p]),
    ('ivar_getTypeEncoding', ctypes.c_char_p, [ctypes.c_void_p]),
    ('ivar_getOffset', ctypes.c_long, [ctypes.c_void_p]),
    ('class_getName', ctypes.c_char_p, [ctypes.c_void_p]),
    ('object_getClass', ctypes.c_void_p, [ctypes.c_void_p]),
    ('class_getInstanceVariable', ctypes.c_void_p,
     [ctypes.c_void_p, ctypes.c_char_p]),
    ('class_getSuperclass', ctypes.c_void_p, [ctypes.c_void_p]),
]:
    fn = getattr(objc, name)
    fn.restype = res
    fn.argtypes = args

def _send(ret_type, obj, sel_name, *arg_pairs):
    sel = objc.sel_registerName(sel_name.encode())
    arg_types = [t for t, v in arg_pairs]
    f = ctypes.cast(objc.objc_msgSend, ctypes.CFUNCTYPE(
        ret_type, ctypes.c_void_p, ctypes.c_void_p, *arg_types))
    return f(obj, sel, *[v for t, v in arg_pairs])

def msg(obj, sel_name):
    sel = objc.sel_registerName(sel_name.encode())
    return objc.objc_msgSend(obj, sel)

def msg_id(obj, sel_name, arg):
    return _send(ctypes.c_void_p, obj, sel_name, (ctypes.c_void_p, arg))

def msg_cstr(obj, sel_name, s):
    return _send(ctypes.c_void_p, obj, sel_name,
                 (ctypes.c_char_p, s.encode() if isinstance(s, str) else s))

def msg_int(obj, sel_name):
    return _send(ctypes.c_long, obj, sel_name)

def msg_double(obj, sel_name):
    return _send(ctypes.c_double, obj, sel_name)

def msg_set_int64(obj, sel_name, v):
    _send(None, obj, sel_name, (ctypes.c_int64, ctypes.c_int64(v)))

def get_class(name):
    return objc.objc_getClass(name.encode())

def get_class_name(ptr):
    if not ptr or ptr < 0x1000: return "nil"
    try:
        cls = objc.object_getClass(ptr)
        if cls:
            n = objc.class_getName(cls)
            if n: return n.decode()
    except: pass
    return "?"

def read_ptr(obj, offset):
    return ctypes.c_void_p.from_address(obj + offset).value

def read_double(obj, offset):
    return ctypes.c_double.from_address(obj + offset).value

def nsstring_to_py(nsstr):
    if not nsstr: return None
    CF = ctypes.cdll.LoadLibrary(
        '/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation')
    CF.CFStringGetCStringPtr.restype = ctypes.c_char_p
    CF.CFStringGetCStringPtr.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    s = CF.CFStringGetCStringPtr(nsstr, 0x08000100)
    if s: return s.decode()
    CF.CFStringGetCString.restype = ctypes.c_bool
    CF.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
    buf = ctypes.create_string_buffer(4096)
    if CF.CFStringGetCString(nsstr, buf, 4096, 0x08000100):
        return buf.value.decode()
    return None

def make_nsstring(s):
    return msg_cstr(get_class("NSString"), "stringWithUTF8String:", s)

def make_nsurl(path):
    ns = make_nsstring(path)
    return msg_id(get_class("NSURL"), "fileURLWithPath:", ns)


# ── Load frameworks ──────────────────────────────────────────

for p in [
    '/System/Library/Frameworks/CoreML.framework/CoreML',
    '/System/Library/PrivateFrameworks/ANEServices.framework/ANEServices',
    '/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine',
    '/System/Library/PrivateFrameworks/Espresso.framework/Espresso',
]:
    try: ctypes.cdll.LoadLibrary(p)
    except: pass


# ── IOReport ─────────────────────────────────────────────────

lib = ctypes.cdll.LoadLibrary('/usr/lib/libIOReport.dylib')
CF = ctypes.cdll.LoadLibrary(
    '/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation')

for name, res, args in [
    ('IOReportCopyChannelsInGroup', ctypes.c_void_p, [ctypes.c_void_p]*3),
    ('IOReportCreateSubscription', ctypes.c_void_p,
     [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
      ctypes.c_uint64, ctypes.c_void_p]),
    ('IOReportCreateSamples', ctypes.c_void_p, [ctypes.c_void_p]*3),
    ('IOReportCreateSamplesDelta', ctypes.c_void_p, [ctypes.c_void_p]*3),
    ('IOReportChannelGetChannelName', ctypes.c_void_p, [ctypes.c_void_p]),
    ('IOReportChannelGetSubGroup', ctypes.c_void_p, [ctypes.c_void_p]),
    ('IOReportChannelGetFormat', ctypes.c_int, [ctypes.c_void_p]),
    ('IOReportStateGetCount', ctypes.c_int, [ctypes.c_void_p]),
    ('IOReportStateGetResidency', ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_int]),
    ('IOReportSimpleGetIntegerValue', ctypes.c_long, [ctypes.c_void_p]),
]:
    fn = getattr(lib, name); fn.restype = res; fn.argtypes = args

for name, res, args in [
    ('CFArrayGetCount', ctypes.c_long, [ctypes.c_void_p]),
    ('CFArrayGetValueAtIndex', ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_long]),
    ('CFDictionaryGetValue', ctypes.c_void_p, [ctypes.c_void_p]*2),
    ('CFDictionaryCreateMutableCopy', ctypes.c_void_p,
     [ctypes.c_void_p, ctypes.c_long, ctypes.c_void_p]),
    ('CFStringCreateWithCString', ctypes.c_void_p,
     [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]),
    ('CFStringGetCStringPtr', ctypes.c_char_p, [ctypes.c_void_p, ctypes.c_uint32]),
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
    if CF.CFStringGetCString(p, buf, 512, 0x08000100): return buf.value.decode()
    return '?'

ENERGY_NAMES = {'ANE', 'GPU', 'DRAM', 'CPU Energy'}
BW_KEYS = {
    ('ANE0 RD+WR', 'DCS BW'), ('ANE0 RD', 'DCS BW'), ('ANE0 WR', 'DCS BW'),
}


class IOReportSampler:
    def __init__(self):
        self.subs = []
        for g in ['Energy Model', 'PMP']:
            ch = lib.IOReportCopyChannelsInGroup(cfstr(g), None, None)
            if not ch: continue
            m = CF.CFDictionaryCreateMutableCopy(None, 0, ch)
            out = ctypes.c_void_p(0)
            sub = lib.IOReportCreateSubscription(None, m, ctypes.byref(out), 0, None)
            if sub: self.subs.append((g, sub, out.value or m))

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
                    results['energy'][name] = lib.IOReportSimpleGetIntegerValue(c)
                if fmt == 2 and (name, subgroup) in BW_KEYS:
                    nstates = lib.IOReportStateGetCount(c)
                    res_list = [lib.IOReportStateGetResidency(c, s)
                                for s in range(nstates)]
                    total = sum(res_list)
                    active = sum(res_list[1:])
                    peak = max((i for i, r in enumerate(res_list) if r > 0), default=0)
                    results['bw'][name] = {
                        'util': active / total if total > 0 else 0,
                        'peak': peak,
                    }
        return results


# ── Model loading and prediction via ObjC ────────────────────

def load_model(path):
    """Load CoreML model via ObjC with profilingOptions enabled."""
    url = make_nsurl(path)
    config = msg(msg(get_class("MLModelConfiguration"), "alloc"), "init")
    msg_set_int64(config, "setComputeUnits:", 2)  # all
    msg_set_int64(config, "setProfilingOptions:", 0xFF)

    error = ctypes.c_void_p(0)
    sel = objc.sel_registerName(b"compileModelAtURL:error:")
    f = ctypes.cast(objc.objc_msgSend, ctypes.CFUNCTYPE(
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)))
    compiled_url = f(get_class("MLModel"), sel, url, ctypes.byref(error))
    if not compiled_url:
        print(f"Compile error: {nsstring_to_py(msg(error.value, 'localizedDescription')) if error.value else 'unknown'}")
        return None

    error = ctypes.c_void_p(0)
    sel = objc.sel_registerName(b"modelWithContentsOfURL:configuration:error:")
    f = ctypes.cast(objc.objc_msgSend, ctypes.CFUNCTYPE(
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)))
    model = f(get_class("MLModel"), sel, compiled_url, config, ctypes.byref(error))
    if not model:
        print(f"Load error: {nsstring_to_py(msg(error.value, 'localizedDescription')) if error.value else 'unknown'}")
    return model


def get_input_spec(model):
    """Get input feature names and shapes from model description."""
    desc = msg(model, "modelDescription")
    if not desc: return {}
    input_desc = msg(desc, "inputDescriptionsByName")
    if not input_desc: return {}

    keys = msg(input_desc, "allKeys")
    count = msg_int(keys, "count")
    inputs = {}

    for i in range(count):
        key = _send(ctypes.c_void_p, keys, "objectAtIndex:",
                    (ctypes.c_ulong, ctypes.c_ulong(i)))
        key_str = nsstring_to_py(key)
        feat = msg_id(input_desc, "objectForKey:", key)
        if not feat: continue

        # Get multiarray constraint
        ma_constraint = msg(feat, "multiArrayConstraint")
        if ma_constraint:
            shape_arr = msg(ma_constraint, "shape")
            if shape_arr:
                shape_count = msg_int(shape_arr, "count")
                shape = []
                for j in range(shape_count):
                    dim = _send(ctypes.c_void_p, shape_arr, "objectAtIndex:",
                                (ctypes.c_ulong, ctypes.c_ulong(j)))
                    shape.append(msg_int(dim, "intValue"))
                inputs[key_str] = shape
    return inputs


def predict(model, inputs_spec, n=10):
    """Run n predictions."""
    nsnum = get_class("NSNumber")
    nsmut = get_class("NSMutableArray")
    mla = get_class("MLMultiArray")

    # Build feature provider from first input
    providers = {}
    for name, shape in inputs_spec.items():
        shape_arr = msg(msg(nsmut, "alloc"), "init")
        for d in shape:
            n_obj = _send(ctypes.c_void_p, nsnum, "numberWithInt:", (ctypes.c_int, d))
            msg_id(shape_arr, "addObject:", n_obj)

        error = ctypes.c_void_p(0)
        sel = objc.sel_registerName(b"initWithShape:dataType:error:")
        f = ctypes.cast(objc.objc_msgSend, ctypes.CFUNCTYPE(
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)))
        arr = f(msg(mla, "alloc"), sel, shape_arr, ctypes.c_int(65568), ctypes.byref(error))
        if arr:
            providers[name] = msg_id(get_class("MLFeatureValue"),
                                     "featureValueWithMultiArray:", arr)

    if not providers:
        return

    # Build NSDictionary
    if len(providers) == 1:
        name, fv = list(providers.items())[0]
        key = make_nsstring(name)
        sel = objc.sel_registerName(b"dictionaryWithObject:forKey:")
        f = ctypes.cast(objc.objc_msgSend, ctypes.CFUNCTYPE(
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p))
        d = f(get_class("NSDictionary"), sel, fv, key)
    else:
        d = msg(msg(get_class("NSMutableDictionary"), "alloc"), "init")
        for name, fv in providers.items():
            key = make_nsstring(name)
            sel = objc.sel_registerName(b"setObject:forKey:")
            f = ctypes.cast(objc.objc_msgSend, ctypes.CFUNCTYPE(
                None, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p))
            f(d, sel, fv, key)

    error = ctypes.c_void_p(0)
    sel = objc.sel_registerName(b"initWithDictionary:error:")
    f = ctypes.cast(objc.objc_msgSend, ctypes.CFUNCTYPE(
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)))
    provider = f(msg(get_class("MLDictionaryFeatureProvider"), "alloc"),
                 sel, d, ctypes.byref(error))
    if not provider: return

    sel = objc.sel_registerName(b"predictionFromFeatures:error:")
    f = ctypes.cast(objc.objc_msgSend, ctypes.CFUNCTYPE(
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)))

    for _ in range(n):
        e = ctypes.c_void_p(0)
        f(model, sel, provider, ctypes.byref(e))


# ── Segmentation analytics ───────────────────────────────────

def get_op_schedule(model):
    """Get per-op execution schedule from segmentation analytics."""
    engine_ptr = read_ptr(model, 88)  # MLDelegateModel._internalEngine
    if not engine_ptr: return []
    proglib_ptr = read_ptr(engine_ptr, 104)  # MLE5Engine._programLibrary
    if not proglib_ptr: return []

    error = ctypes.c_void_p(0)
    sel = objc.sel_registerName(b"segmentationAnalyticsAndReturnError:")
    f = ctypes.cast(objc.objc_msgSend, ctypes.CFUNCTYPE(
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p)))
    analytics = f(proglib_ptr, sel, ctypes.byref(error))
    if not analytics: return []

    keys = msg(analytics, "allKeys")
    if not keys: return []
    count = msg_int(keys, "count")

    ops = []
    for i in range(count):
        key = _send(ctypes.c_void_p, keys, "objectAtIndex:",
                    (ctypes.c_ulong, ctypes.c_ulong(i)))
        val = msg_id(analytics, "objectForKey:", key)

        op = {}
        for field in ['OpType', 'SelectedBackend', 'OpIndex', 'DebugName']:
            fk = make_nsstring(field)
            fv = msg_id(val, "objectForKey:", fk)
            if fv:
                cls = get_class_name(fv)
                if 'String' in cls:
                    v = nsstring_to_py(fv)
                    op[field] = v.strip('"') if v else v
                elif 'Number' in cls:
                    op[field] = msg_int(fv, "longLongValue")

        # EstimatedRunTime per backend
        ert_key = make_nsstring("EstimatedRunTime")
        ert_val = msg_id(val, "objectForKey:", ert_key)
        if ert_val and 'Dict' in get_class_name(ert_val):
            ert = {}
            for be in ['ane', 'bnns', 'classic_cpu', 'mps_graph']:
                bk = make_nsstring(be)
                bv = msg_id(ert_val, "objectForKey:", bk)
                if bv: ert[be] = msg_double(bv, "doubleValue")
            op['estimated_time'] = ert

        # BackendSupport
        bs_key = make_nsstring("BackendSupport")
        bs_val = msg_id(val, "objectForKey:", bs_key)
        if bs_val and 'Dict' in get_class_name(bs_val):
            support = {}
            for be in ['ane', 'bnns', 'classic_cpu', 'mps_graph']:
                bk = make_nsstring(be)
                bv = msg_id(bs_val, "objectForKey:", bk)
                if bv: support[be] = msg_int(bv, "longLongValue")
            op['backend_support'] = support

        ops.append(op)

    return sorted(ops, key=lambda x: x.get('OpIndex', 0))


# ── Pretty printing ──────────────────────────────────────────

def print_profile(model_path, ops, hw, elapsed, n_iter, inputs_spec):
    """Print the full profile report."""
    import subprocess
    size = subprocess.check_output(['du', '-sk', model_path]).decode().split()[0]
    size_mb = int(size) / 1024

    print()
    print("=" * 78)
    print(f"  CoreML ANE Profile: {os.path.basename(model_path)}")
    print(f"  Size: {size_mb:.1f} MB | Inputs: {inputs_spec}")
    print("=" * 78)

    # Backend summary
    backends = {}
    for op in ops:
        be = op.get('SelectedBackend', '?')
        backends[be] = backends.get(be, 0) + 1

    print(f"\n  Backend dispatch: ", end="")
    for be, count in sorted(backends.items()):
        print(f"{be}={count} ", end="")
    total_ops = sum(backends.values())
    ane_pct = backends.get('ane', 0) / total_ops * 100 if total_ops > 0 else 0
    print(f"({total_ops} ops, {ane_pct:.0f}% ANE)")

    # Per-op table
    ane_ops = [op for op in ops if op.get('SelectedBackend') == 'ane']
    non_ane_ops = [op for op in ops if op.get('SelectedBackend') != 'ane']

    if ops:
        print(f"\n  {'Op':<40s} {'Backend':>8s} {'ERT(ane)':>10s} {'ERT(bnns)':>10s} "
              f"{'ERT(mps)':>10s} {'Speedup':>8s}")
        print("  " + "-" * 88)

        for op in ops:
            be = op.get('SelectedBackend', '?')
            ert = op.get('estimated_time', {})
            ane_t = ert.get('ane', 0)
            bnns_t = ert.get('bnns', 0)
            mps_t = ert.get('mps_graph', 0)
            speedup = bnns_t / ane_t if ane_t > 0 else 0

            name = op.get('OpType', '?')
            idx = op.get('OpIndex', '')
            label = f"[{idx}] {name}"

            ane_str = f"{ane_t*1000:.3f}ms" if ane_t > 0 else "—"
            bnns_str = f"{bnns_t*1000:.3f}ms" if bnns_t > 0 else "—"
            mps_str = f"{mps_t*1000:.3f}ms" if mps_t > 0 else "—"
            sp_str = f"{speedup:.1f}x" if speedup > 0 else "—"

            marker = " *" if be == 'ane' else ""
            print(f"  {label:<40s} {be:>8s} {ane_str:>10s} {bnns_str:>10s} "
                  f"{mps_str:>10s} {sp_str:>8s}{marker}")

    # Hardware measurements
    ane_energy = hw['energy'].get('ANE', 0)
    gpu_energy = hw['energy'].get('GPU', 0)
    cpu_energy = hw['energy'].get('CPU Energy', 0)
    ane_bw = hw['bw'].get('ANE0 RD+WR', {})
    ms_per = elapsed * 1000 / n_iter

    print(f"\n  Hardware measurements ({n_iter} iterations, {elapsed:.3f}s):")
    print(f"    Latency:    {ms_per:.3f} ms/iter ({n_iter/elapsed:.0f} predictions/sec)")
    print(f"    ANE energy: {ane_energy:,d} | GPU energy: {gpu_energy:,d} | CPU energy: {cpu_energy:,d}")
    if ane_bw:
        print(f"    ANE BW:     util={ane_bw['util']*100:.2f}%, peak_state={ane_bw['peak']}")
    else:
        print(f"    ANE BW:     not active")

    uses_ane = ane_energy > 0 or ane_bw.get('util', 0) > 0 or 'ane' in backends
    print(f"\n  ANE active: {'YES' if uses_ane else 'NO'}")
    if not uses_ane and any(op.get('backend_support', {}).get('ane', 0) for op in ops):
        print(f"  Note: ANE is supported but not selected. Model may be too small for ANE dispatch.")

    print()


# ── Build test model ─────────────────────────────────────────

def build_test_model(dim, n_layers, seq_len):
    import torch
    import torch.nn as nn
    import coremltools as ct

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(
                [nn.Conv1d(dim, dim, 1, bias=False) for _ in range(n_layers)])
        def forward(self, x):
            for l in self.layers: x = l(x)
            return x

    path = f"/tmp/ane_perf_test_{dim}x{n_layers}_s{seq_len}.mlpackage"
    model = M().eval().half()
    example = torch.randn(1, dim, seq_len).half()
    traced = torch.jit.trace(model, example)
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name='x', shape=example.shape)],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
    )
    ml.save(path)
    return path


# ── Main ─────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="CoreML ANE Profiler")
    parser.add_argument('model', nargs='?', help='Path to .mlpackage or .mlmodelc')
    parser.add_argument('--build', nargs=3, type=int, metavar=('DIM', 'LAYERS', 'SEQ'),
                        help='Build a test model: dim layers seq_len')
    parser.add_argument('--iters', type=int, default=100, help='Number of iterations')
    parser.add_argument('--warmup', type=int, default=20, help='Warmup iterations')
    parser.add_argument('--json', action='store_true', help='JSON output')
    args = parser.parse_args()

    if args.build:
        dim, layers, seq = args.build
        print(f"Building test model: {dim}x{layers} seq={seq}...")
        model_path = build_test_model(dim, layers, seq)
        print(f"Saved: {model_path}")
    elif args.model:
        model_path = args.model
    else:
        parser.print_help()
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"Not found: {model_path}")
        sys.exit(1)

    # Load
    model = load_model(model_path)
    if not model:
        sys.exit(1)

    # Get input spec
    inputs_spec = get_input_spec(model)
    if not inputs_spec:
        print("Could not determine input spec")
        sys.exit(1)

    # Get op schedule
    ops = get_op_schedule(model)

    # Warmup
    predict(model, inputs_spec, args.warmup)

    # Measure
    sampler = IOReportSampler()
    before = sampler.capture()
    t0 = time.perf_counter()
    predict(model, inputs_spec, args.iters)
    elapsed = time.perf_counter() - t0
    after = sampler.capture()
    hw = sampler.delta(before, after)

    if args.json:
        result = {
            'model': model_path,
            'inputs': inputs_spec,
            'ops': ops,
            'hardware': hw,
            'timing': {
                'elapsed_s': elapsed,
                'iterations': args.iters,
                'ms_per_iter': elapsed * 1000 / args.iters,
            }
        }
        print(json_mod.dumps(result, indent=2, default=str))
    else:
        print_profile(model_path, ops, hw, elapsed, args.iters, inputs_spec)


if __name__ == "__main__":
    main()
