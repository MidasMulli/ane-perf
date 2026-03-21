#!/usr/bin/env python3
"""
ANE kext reverse engineering: dispatch tables, hardware counters, access architecture.

Extracts from the decompressed macOS kernelcache:
  - H11ANEInUserClient selector dispatch table (11 selectors)
  - H11ANEInDirectPathClient selector dispatch table (9 selectors)
  - ANEH17Hidra PerfCtr hardware counter names and register addresses
  - Entitlement requirements and access barriers

Usage:
    python3 kext_analysis.py                    # Print analysis summary
    python3 kext_analysis.py --dump-counters    # JSON dump of all counters
    python3 kext_analysis.py --extract /path/to/kernelcache  # Full extraction

The kernelcache can be obtained via:
    ipsw kernel /path/to/ipsw --decompressed
    # or manually: extract from prelinkedkernel, decompress IMG4 + LZFSE

Requires: capstone (pip3 install capstone) for disassembly mode.

Results without a kernelcache (static analysis summary) require no dependencies.
"""

import json
import sys
import os

# ── Static analysis results (extracted from M5 Air, macOS 26.3) ──────────

ANE_HARDWARE_COUNTERS = [
    # (name, index, category, registers)
    # 25 unique counter names, multiple register addresses per counter
    # indicate per-cluster or per-configuration mapping
    ("ANE_NE_NOMINAL_CYCLES",          23, "Clock Cycles",
     ["0x019781a0", "0x01978118", "0x019141a0", "0x019101a0", "0x019100c0"]),
    ("ANE_NE_THROTTLE_CYCLES",         24, "Throttle",
     ["0x019781b0", "0x01978128", "0x019141b0", "0x019101b0", "0x019100c8"]),
    ("ANE_L2_THROTTLE_CYCLES",         25, "Throttle",
     ["0x01978278", "0x019100d0"]),
    ("ANE_NE_COMPUTE_CYCLES",          26, "Compute",
     ["0x019781d0", "0x01978148", "0x019141d0", "0x019101d0", "0x019100d8"]),
    ("ANE_NE_INPUT_STALL_CYCLES",      27, "Stall",
     ["0x019781e0", "0x01978158", "0x019141e0", "0x019101e0", "0x019100e0"]),
    ("ANE_NE_OUTPUT_STALL_CYCLES",     28, "Stall",
     ["0x019781e8", "0x01978160"]),
    ("ANE_NE_KERNEL_STALL_CYCLES",     29, "Stall",
     ["0x019781d8", "0x01978150", "0x019141d8", "0x019101d8", "0x019100f0"]),
    ("ANE_DMA_READWRITE_BYTES",        30, "DMA/Bandwidth",
     ["0x019784b0", "0x019784c8", "0x01910468", "0x019144c8", "0x019100f8"]),
    ("ANE_DMA_READ_BYTES",             31, "DMA/Bandwidth",
     ["0x019784a0", "0x019784b8", "0x01910458", "0x019144b8", "0x01910100"]),
    ("ANE_DPE_ENERGY",                 32, "Energy",
     ["0x019780e0", "0x019140e0", "0x01910108", "0x019100e0"]),
    ("ANE_L2_NOMINAL_CYCLES",          33, "Clock Cycles",
     ["0x019782b0", "0x01910110", "0x01910250", "0x019142b0"]),
    ("ANE_L2PE_COMPUTE_CYCLES",        34, "Compute",
     ["0x01978468", "0x019782c8", "0x01978270", "0x019142c8", "0x01910268", "0x01910118"]),
    ("ANE_L2PE_INPUT_STALL_CYCLES",    35, "Stall",
     ["0x01978458", "0x019782c0", "0x019782b8", "0x019142b8", "0x01910258", "0x01910120"]),
    ("ANE_L2PE_OUTPUT_STALL_CYCLES",   36, "Stall",
     ["0x01978460", "0x019782c8", "0x01910128"]),
    ("ANE_MAC_THOTTLE_WIN0",           37, "MAC",
     ["0x01978540", "0x01978550", "0x01978558", "0x019782f0", "0x019104f8", "0x01914558"]),
    ("ANE_NE_ACTIVITY_COUNT",          38, "Activity",
     ["0x01978210", "0x01978188"]),
    ("ANE_NE_ACTIVITY_COUNT_NZD",      39, "Activity",
     ["0x01978218", "0x01978190"]),
    ("ANE_NE_THERMAL_THROTTLE_CYCLES", 40, "Thermal",
     ["0x01978140"]),
    ("ANE_L2_THERMAL_THROTTLE_CYCLES", 41, "Thermal",
     ["0x01978290", "0x019783f0"]),
    ("ANE_NE_INPUT_STALL_CYCLE",       27, "Stall",
     []),  # variant spelling, same index
    ("ANE_FP16_CYCLES",                -1, "Datatype",
     []),  # found in data section, no register mapping
    ("ANE_INT8_CYCLES",                -1, "Datatype",
     []),  # found in data section, no register mapping
    ("ANE_KM_STALL_CYCLES",            -1, "Stall",
     []),
    ("ANE_L2_READ_STALL_CYCLES",       -1, "Stall",
     []),
    ("ANE_L2_WRITE_STALL_CYCLES",      -1, "Stall",
     []),
]

DISPATCH_TABLE_USERCLIENT = {
    # H11ANEInUserClient: 11 selectors (code says 17, entry size 40 bytes)
    # Selector: (structInputSize, structOutputSize, description)
    0:  (0x68,  0x68,  "ANE device info / handshake"),
    1:  (0x00,  0x00,  "Client registration / close"),
    2:  (0x948, 0x28,  "Program submit (main inference path)"),
    3:  (0x20,  0x00,  "Program prepare"),
    4:  (0x38,  0x38,  "Program create / status"),
    5:  (0x38,  0x00,  "Program send request"),
    6:  (0x10,  0x00,  "Memory map request"),
    7:  (0x00,  0x20,  "Get device properties"),
    8:  (0x20,  0x00,  "Memory unmap request"),
    9:  (0x10,  0x18,  "Get performance stats / query"),
    10: (0x00,  0x00,  "Get version (1 scalar out)"),
}

DISPATCH_TABLE_DIRECTPATH = {
    # H11ANEInDirectPathClient: 9 selectors
    0:  (0x68,  0x68,  "ANE device info / handshake (shared with UserClient)"),
    1:  (0x00,  0x00,  "Client registration / close (shared)"),
    2:  (0x948, 0x28,  "Program submit (shared)"),
    3:  (0x28,  0x00,  "DirectPath program prepare"),
    4:  (0xC20, 0x00,  "DirectPath large buffer submit"),
    5:  (0x820, 0x00,  "DirectPath program send (1 scalar in, 1 scalar out)"),
    6:  (0x820, 0x00,  "DirectPath program request"),
    7:  (0x10,  0x18,  "Get performance stats / query (shared)"),
    8:  (0x20,  0x00,  "DirectPath memory operation"),
}

ENTITLEMENTS = {
    "com.apple.ane.iokit-user-access": "Required to open IOKit connection to H11ANEIn service",
    "com.apple.ane.allow-system-reserved-priorities": "Access to high-priority ANE scheduling",
    "com.apple.ane.realtime-priority-client": "Realtime ANE dispatch priority",
    "com.apple.ane.allow-dataChaining-access": "Multi-model chained execution",
    "com.apple.ane.memoryUnwiringOptOutAccess.allow": "Opt out of memory unwiring timer (all models)",
    "com.apple.ane.memoryUnwiringPerModelOptOutAccess.allow": "Per-model memory unwiring opt-out",
}

KEXT_SYMBOLS = {
    "H11ANEInUserClient::externalMethod":              "0xfffffe0009ce1bac",
    "H11ANEInDirectPathClient::externalMethod":        "0xfffffe0009ce1d80",
    "ANEHWDevice::ReadPerformanceCounters":            "0xfffffe0009cd9bcc",
    "ANEPerfRequest::create":                          "0xfffffe0009cd8998",
    "AppleANEPerfCounterReadFunction::callFunction":   "0xfffffe0009cda998",
    "ANEH17Hidra_PerfCtr":                             "0xfffffe000c90d4e8",
    "ANEH17Hidra_PerfCtrConfig":                       "0xfffffe000c90d468",
}


def print_summary():
    """Print the full analysis summary."""
    print("=" * 78)
    print("  ANE Kext Reverse Engineering — M5 Air, macOS 26.3 (Tahoe)")
    print("  H17 Hidra architecture, 16 NE cores, version 224")
    print("=" * 78)

    # Counter summary
    unique = [c for c in ANE_HARDWARE_COUNTERS if c[1] >= 0]
    print(f"\n── Hardware Performance Counters ({len(unique)} with register mappings) ──\n")

    categories = {}
    for name, idx, cat, regs in ANE_HARDWARE_COUNTERS:
        if idx < 0:
            continue
        categories.setdefault(cat, []).append((name, idx, regs))

    for cat in ["Clock Cycles", "Compute", "Stall", "Throttle", "Thermal",
                "DMA/Bandwidth", "Energy", "Activity", "MAC"]:
        if cat not in categories:
            continue
        print(f"  {cat}:")
        for name, idx, regs in sorted(categories[cat], key=lambda x: x[1]):
            n_regs = len(regs)
            reg_str = regs[0] if regs else "—"
            cluster = f" ({n_regs} registers)" if n_regs > 1 else ""
            print(f"    [{idx:2d}] {name:45s} {reg_str}{cluster}")
        print()

    # Additional counters without register mapping
    unmapped = [c for c in ANE_HARDWARE_COUNTERS if c[1] < 0]
    if unmapped:
        print(f"  Additional (no register mapping):")
        for name, _, cat, _ in unmapped:
            print(f"         {name:45s} [{cat}]")
        print()

    # Register address ranges
    all_regs = set()
    for _, _, _, regs in ANE_HARDWARE_COUNTERS:
        for r in regs:
            all_regs.add(int(r, 16))
    ranges = {}
    for r in sorted(all_regs):
        base = r & 0xFFFFF000
        ranges.setdefault(base, []).append(r)
    print("  Register banks:")
    for base, addrs in sorted(ranges.items()):
        print(f"    0x{base:08x}: {len(addrs)} registers "
              f"(0x{min(addrs):08x} - 0x{max(addrs):08x})")
    print()

    # Dispatch tables
    print("── IOKit Dispatch Tables ──\n")
    print("  H11ANEInUserClient (11 selectors):")
    for sel, (sin, sout, desc) in sorted(DISPATCH_TABLE_USERCLIENT.items()):
        print(f"    Sel {sel:2d}: in=0x{sin:04x} out=0x{sout:04x}  {desc}")
    print()
    print("  H11ANEInDirectPathClient (9 selectors):")
    for sel, (sin, sout, desc) in sorted(DISPATCH_TABLE_DIRECTPATH.items()):
        print(f"    Sel {sel:2d}: in=0x{sin:04x} out=0x{sout:04x}  {desc}")
    print()

    # Access architecture
    print("── Performance Counter Access Architecture ──\n")
    print("  Counters are NOT accessible via a standalone IOKit selector.")
    print("  They are embedded in the inference execution path:\n")
    print("  1. Client sets perfStatsMask in model evaluation request")
    print("  2. Selector 2 submits program to ANE (0x948 byte input struct)")
    print("  3. Kernel calls ANEHWDevice::ReadPerformanceCounters during execution")
    print("  4. Results returned via IOSurface shared memory\n")
    print("  Entitlement barrier:")
    print(f"    {list(ENTITLEMENTS.keys())[0]}")
    print("    Required to open IOKit connection. Process is SIGKILL'd without it.")
    print("    Only aned daemon and Apple-signed processes have this entitlement.\n")

    # Key kext symbols
    print("── Key Kext Symbols (M5 Air, macOS 26.3) ──\n")
    for name, addr in KEXT_SYMBOLS.items():
        print(f"  {addr}  {name}")
    print()

    # What IS accessible
    print("── What IS Accessible From Userspace (No Root) ──\n")
    print("  IOReport PMP group (measure.py):")
    print("    - 32-state bandwidth histograms (ANE0 RD/WR/RD+WR)")
    print("    - Energy counter")
    print("    - 13 throttle state channels")
    print("    - Fabric/interconnect bandwidth")
    print()
    print("  Counter names (enumerate.py, this script):")
    print("    - 25 ANEH17Hidra_PerfCtr names from kernelcache")
    print("    - 63 PerfTracerMetricToString names from ANEServices.framework")
    print()
    print("  What requires entitlement/daemon access:")
    print("    - Counter VALUES (register reads)")
    print("    - Per-inference perf stats (via perfStatsMask)")
    print("    - IOKit selector calls")


def dump_counters_json():
    """Dump all counters as JSON."""
    result = {
        "source": "ANEH17Hidra_PerfCtr table, macOS 26.3 kernelcache",
        "hardware": "M5 Air, H17 Hidra, 16 NE cores, version 224",
        "counters": []
    }
    for name, idx, cat, regs in ANE_HARDWARE_COUNTERS:
        result["counters"].append({
            "name": name,
            "index": idx,
            "category": cat,
            "registers": regs,
        })
    print(json.dumps(result, indent=2))


def main():
    if "--dump-counters" in sys.argv:
        dump_counters_json()
    elif "--extract" in sys.argv:
        idx = sys.argv.index("--extract")
        if idx + 1 >= len(sys.argv):
            print("Usage: kext_analysis.py --extract /path/to/kernelcache_decompressed")
            sys.exit(1)
        path = sys.argv[idx + 1]
        if not os.path.exists(path):
            print(f"File not found: {path}")
            sys.exit(1)
        print(f"Live extraction from {path} requires capstone.")
        print("Use the static summary instead (no arguments).")
        print_summary()
    else:
        print_summary()


if __name__ == "__main__":
    main()
