#!/usr/bin/env python3
"""
Enumerate Apple Neural Engine hardware counter names.

Two independent counter systems:
  1. PerfTracer: 63 metrics + 22 hardware block categories
     (via ANEServices.framework C++ symbols)
  2. _ANEPerformanceStats: 25 named counters with kANE_ prefix
     (via AppleNeuralEngine.framework ObjC runtime)

Counter NAMES are readable from any userspace process on macOS.
Counter VALUES require daemon-level access (see ARCHITECTURE.md).

Requires: macOS with Apple Silicon (M1+)
No root, no SIP changes, no entitlements needed.
"""

import ctypes
import json
import sys

# ── PerfTracer (ANEServices.framework) ─────────────────────────

def enumerate_perftracer():
    """Extract 63 metric names and 22+ hardware block categories."""
    try:
        svc = ctypes.cdll.LoadLibrary(
            '/System/Library/PrivateFrameworks/ANEServices.framework/ANEServices')
    except OSError:
        return [], []

    _dl = ctypes.cdll.LoadLibrary(None)
    _dlsym = _dl.dlsym
    _dlsym.restype = ctypes.c_void_p
    _dlsym.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

    metric_ptr = _dlsym(svc._handle, b'_Z24PerfTracerMetricToStringi')
    cat_ptr = _dlsym(svc._handle, b'_Z26PerfTracerCategoryToStringi')

    if not metric_ptr or not cat_ptr:
        return [], []

    metric_fn = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_int)(metric_ptr)
    cat_fn = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_int)(cat_ptr)

    # Metrics (1-63)
    metrics = []
    for i in range(1, 64):
        name = metric_fn(i)
        if name and len(name) > 0:
            metrics.append({'id': i, 'name': name.decode()})

    # Categories (1-63)
    categories = []
    for i in range(1, 64):
        name = cat_fn(i)
        if name and len(name) > 0:
            categories.append({'id': i, 'name': name.decode()})

    return metrics, categories


# ── _ANEPerformanceStats (AppleNeuralEngine.framework) ─────────

def enumerate_perfstats():
    """Extract 25 counter names from _ANEPerformanceStats ObjC class."""
    try:
        ctypes.cdll.LoadLibrary(
            '/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine')
        import objc
        cls = objc.lookUpClass('_ANEPerformanceStats')
    except (OSError, ImportError, Exception):
        return []

    try:
        stats = cls.statsWithHardwareExecutionNS_(0)
        counters = []
        for i in range(64):
            name = stats.stringForPerfCounter_(i)
            if name and name != 'kANE_UKNOWN':
                counters.append({'id': i, 'name': str(name)})
            elif name == 'kANE_UKNOWN':
                break
        return counters
    except Exception:
        return []


# ── Annotate ───────────────────────────────────────────────────

PERFTRACER_CATEGORIES = {
    range(1, 16): 'L2 SRAM',
    range(16, 34): 'DMA',
    range(34, 39): 'Fabric/Bus',
    range(39, 49): 'Memory Cache',
    range(49, 58): 'Texture/Decomp',
    range(58, 64): 'Neural Engine',
}

PERFTRACER_DESCRIPTIONS = {
    'l2_cycle': 'Total L2 SRAM clock cycles',
    'l2_src1_read_active_cycle': 'L2 source-1 read port active cycles',
    'l2_src1_read_idle_cycle': 'L2 source-1 read port idle cycles',
    'l2_src2_read_active_cycle': 'L2 source-2 read port active cycles',
    'l2_src2_read_idle_cycle': 'L2 source-2 read port idle cycles',
    'l2_write_active_cycle': 'L2 write port active cycles',
    'l2_write_idle_cycle': 'L2 write port idle cycles',
    'l2_read_conflict_cycle': 'L2 read bank conflict cycles',
    'l2_read_intra_conflict_cycle': 'L2 read intra-bank conflict cycles',
    'l2_read_conflict_count': 'L2 read bank conflict events',
    'l2_read_intra_conflict_count': 'L2 read intra-bank conflict events',
    'l2_write_conflict_cycle': 'L2 write bank conflict cycles',
    'l2_write_intra_conflict_cycle': 'L2 write intra-bank conflict cycles',
    'l2_write_conflict_count': 'L2 write bank conflict events',
    'l2_write_intra_conflict_count': 'L2 write intra-bank conflict events',
    'dma_src1_read_active_cycle': 'DMA source-1 read active cycles',
    'dma_src1_read_idle_cycle': 'DMA source-1 read idle cycles',
    'dma_src1_read_conflict_cycle': 'DMA source-1 read conflict cycles',
    'dma_src1_read_intra_conflict_cycle': 'DMA source-1 read intra-conflict cycles',
    'dma_src1_read_conflict_count': 'DMA source-1 read conflict events',
    'dma_src1_read_intra_conflict_count': 'DMA source-1 read intra-conflict events',
    'dma_src2_read_active_cycle': 'DMA source-2 read active cycles',
    'dma_src2_read_idle_cycle': 'DMA source-2 read idle cycles',
    'dma_src2_read_conflict_cycle': 'DMA source-2 read conflict cycles',
    'dma_src2_read_intra_conflict_cycle': 'DMA source-2 read intra-conflict cycles',
    'dma_src2_read_conflict_count': 'DMA source-2 read conflict events',
    'dma_src2_read_intra_conflict_count': 'DMA source-2 read intra-conflict events',
    'dma_write_active_cycle': 'DMA write active cycles',
    'dma_write_idle_cycle': 'DMA write idle cycles',
    'dma_write_conflict_cycle': 'DMA write conflict cycles',
    'dma_write_intra_conflict_cycle': 'DMA write intra-conflict cycles',
    'dma_write_conflict_count': 'DMA write conflict events',
    'dma_write_intra_conflict_count': 'DMA write intra-conflict events',
    'idle_cycle': 'Fabric/bus total idle cycles',
    'active_cycle': 'Fabric/bus total active cycles',
    'beat_count_in_64B': 'Fabric data transfer volume (x 64 bytes)',
    'transaction_count': 'Fabric transaction count',
    'latency_product': 'Fabric latency-bandwidth product',
    'latency_threshold': 'Memory cache latency threshold counter',
    'fabric_stall': 'SoC fabric stall cycles',
    'mcache_hit_no_dealloc': 'Memory cache hit (no deallocation)',
    'mcache_miss_alloc': 'Memory cache miss (allocation triggered)',
    'mcache_non_coh_agent': 'Memory cache non-coherent agent access',
    'mcache_coh_agent': 'Memory cache coherent agent access',
    'mcache_miss_no_alloc': 'Memory cache miss (no allocation)',
    'mcache_hit_depri': 'Memory cache hit (deprioritized)',
    'mcache_hit_drop': 'Memory cache hit (dropped)',
    'mcache_hit_no_alloc': 'Memory cache hit (no allocation)',
    'l2_stall': 'L2 SRAM stall cycles (texture/decomp)',
    'tex_filter_cache_miss': 'Texture filter cache misses',
    'tex_filter_cache_total': 'Texture filter cache total accesses',
    'tex_index_cache_miss': 'Texture index cache misses',
    'tex_index_cache_total': 'Texture index cache total accesses',
    'decomp_metadata_cache_miss': 'Weight decompression metadata cache misses',
    'decomp_metadata_cache_hit': 'Weight decompression metadata cache hits',
    'decomp_data_cache_miss': 'Weight decompression data cache misses',
    'decomp_data_cache_hit': 'Weight decompression data cache hits',
    'ne_stall': 'Neural engine pipeline stall cycles',
    'kernel_cache_miss': 'Kernel/weight cache misses',
    'ne_cycle': 'Neural engine active cycles',
    'mac_cycle': 'MAC (multiply-accumulate) unit active cycles',
    'nz_count': 'Non-zero element count (sparsity metric)',
    'kernel_stall': 'Kernel/weight loading stall cycles',
}

PERFSTATS_DESCRIPTIONS = {
    'kANE_AF_TO_L2_DATA': 'Bytes transferred from Accelerator Framework to L2 SRAM',
    'kANE_AF_TO_KM_DATA': 'Bytes transferred from AF to Kernel Memory (DRAM)',
    'kANE_L2_TO_AF_DATA': 'Bytes transferred from L2 SRAM to AF',
    'kANE_L2_TO_NE_DATA': 'Bytes transferred from L2 SRAM to NE cores',
    'kANE_NE_TO_L2_DATA': 'Bytes transferred from NE cores to L2 SRAM',
    'kANE_INT8_CYCLES': 'Cycles spent on INT8 operations',
    'kANE_FP16_CYCLES': 'Cycles spent on FP16 operations',
    'kANE_L2_READ_STALL_CYCLES': 'L2 read stalls (data not ready from DRAM)',
    'kANE_L2_WRITE_STALL_CYCLES': 'L2 write stalls (write buffer full)',
    'kANE_KM_STALL_CYCLES': 'Kernel memory (DRAM) stall cycles',
    'kANE_NE_NOMINAL_CYCLES': 'Total NE clock cycles (wall-clock reference)',
    'kANE_NE_THROTTLE_CYCLES': 'Cycles NE was throttled (thermal/power)',
    'kANE_L2_THROTTLE_CYCLES': 'Cycles L2 was throttled',
    'kANE_NE_COMPUTE_CYCLES': 'Actual NE compute cycles (utilization = this/nominal)',
    'kANE_NE_INPUT_STALL_CYCLES': 'Cycles NE waiting for input data',
    'kANE_NE_OUTPUT_STALL_CYCLES': 'Cycles NE waiting to write output',
    'kANE_NE_KERNEL_STALL_CYCLES': 'Cycles NE waiting for weight/kernel data',
    'kANE_DMA_READWRITE_BYTES': 'Total DMA bytes transferred (read + write)',
    'kANE_DMA_READ_BYTES': 'DMA read bytes only',
    'kANE_DPE_ENERGY': 'Data Processing Element energy consumption',
    'kANE_L2_NOMINAL_CYCLES': 'L2 total clock cycles (wall-clock reference)',
    'kANE_L2PE_COMPUTE_CYCLES': 'L2 Processing Element compute cycles',
    'kANE_L2PE_INPUT_STALL_CYCLES': 'L2PE cycles waiting for input',
    'kANE_L2PE_OUTPUT_STALL_CYCLES': 'L2PE cycles waiting to write output',
}

CATEGORY_ANNOTATIONS = {
    'l2': 'L2 SRAM (32MB on-chip)',
    'pe': 'Processing Element',
    'ne_all': 'All 16 Neural Engine sub-units',
    'dma_read': 'DMA read engine',
    'dma_write': 'DMA write engine',
    'kernel_read': 'Kernel/weight loading unit',
}


def annotate_metric(m):
    """Add category and description to a PerfTracer metric."""
    for id_range, cat in PERFTRACER_CATEGORIES.items():
        if m['id'] in id_range:
            m['category'] = cat
            break
    m['description'] = PERFTRACER_DESCRIPTIONS.get(m['name'], '')
    return m


def annotate_perfstat(c):
    """Add description to a _ANEPerformanceStats counter."""
    c['description'] = PERFSTATS_DESCRIPTIONS.get(c['name'], '')
    return c


def annotate_category(c):
    """Add annotation to a PerfTracer category."""
    c['annotation'] = CATEGORY_ANNOTATIONS.get(c['name'], '')
    if c['name'].startswith('ne_') and c['name'][3:].isdigit():
        c['annotation'] = f"Neural Engine sub-unit {c['name'][3:]}"
    return c


# ── Main ───────────────────────────────────────────────────────

def main():
    out = {'perftracer': {}, 'perfstats': {}}

    # PerfTracer
    metrics, categories = enumerate_perftracer()
    metrics = [annotate_metric(m) for m in metrics]
    categories = [annotate_category(c) for c in categories]

    print(f"PerfTracer: {len(metrics)} metrics, {len(categories)} categories")
    print()

    if metrics:
        print(f"{'ID':>3}  {'Name':<42}  {'Category':<16}  Description")
        print(f"{'─'*3}  {'─'*42}  {'─'*16}  {'─'*40}")
        for m in metrics:
            print(f"{m['id']:>3}  {m['name']:<42}  {m.get('category', ''):>16}  "
                  f"{m.get('description', '')}")
        print()

    if categories:
        print(f"Hardware blocks ({len(categories)}):")
        for c in categories:
            ann = f"  -- {c['annotation']}" if c.get('annotation') else ""
            print(f"  [{c['id']:>2}] {c['name']:<16}{ann}")
        print()

    out['perftracer']['metrics'] = metrics
    out['perftracer']['categories'] = categories

    # _ANEPerformanceStats
    perfstats = enumerate_perfstats()
    perfstats = [annotate_perfstat(c) for c in perfstats]

    print(f"_ANEPerformanceStats: {len(perfstats)} counters")
    if perfstats:
        print()
        print(f"{'ID':>3}  {'Name':<36}  Description")
        print(f"{'─'*3}  {'─'*36}  {'─'*50}")
        for c in perfstats:
            print(f"{c['id']:>3}  {c['name']:<36}  {c.get('description', '')}")

    out['perfstats']['counters'] = perfstats

    # Save
    with open('data/counters.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to data/counters.json")


if __name__ == '__main__':
    main()
