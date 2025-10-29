#!/usr/bin/env python3
"""
pxi.py — minimal dummy PXI script for Autoprobe movement robustness testing.
- No hardware calls
- No file I/O
- Deterministic, fast, and safe

Usage example:
  python pxi.py settings.json WaferX DevY --gpib "GPIB0::22::INSTR" --duration 2.0 --invert
"""

import argparse
import json
import sys
import time
from datetime import datetime

def main():
    ap = argparse.ArgumentParser(description="Dummy PXI placeholder (no hardware).")
    # keep these to mirror your real script signature
    ap.add_argument("settings", help="settings filename (unused in dummy)")
    ap.add_argument("chip", help="chip name (logged only)")
    ap.add_argument("device", help="device name (logged only)")
    # common flags you may pass from the launcher
    ap.add_argument("--gpib", default="", help="VISA resource string, e.g. GPIB0::22::INSTR (unused)")
    ap.add_argument("--polarity", default="PMOS", choices=["PMOS","NMOS","pmos","nmos"])
    ap.add_argument("--start-module", default="", help="module name from Autoprobe (logged only)")
    ap.add_argument("--invert", action="store_true", help="dummy invert flag (logged)")
    ap.add_argument("--duration", type=float, default=1.0, help="seconds to pretend we are running a measurement")
    ap.add_argument("--fail", action="store_true", help="if set, exit with non-zero status for failure-path testing")
    ap.add_argument("--seed", type=int, default=0, help="optional seed for any randomized dummy behavior (unused)")

    args = ap.parse_args()

    # minimal banner so your launcher logs show something useful
    meta = {
        "ts_start": datetime.now().isoformat(timespec="seconds"),
        "settings": args.settings,
        "chip": args.chip,
        "device": args.device,
        "gpib": args.gpib,
        "polarity": args.polarity.upper(),
        "start_module": args.start_module,
        "invert": bool(args.invert),
        "duration_s": float(args.duration),
        "mode": "DUMMY_NO_HW_NO_IO",
        "version": "pxi_dummy_0.1",
    }
    print("[pxi] start", json.dumps(meta), flush=True)

    # simulate a few checkpoints so your caller can stream logs
    t0 = time.perf_counter()
    checkpoints = max(1, int(args.duration / 0.25))
    for i in range(checkpoints):
        time.sleep(args.duration / checkpoints if args.duration > 0 else 0)
        elapsed = time.perf_counter() - t0
        print(f"[pxi] progress {{\"elapsed_s\": {elapsed:.3f}, \"step\": {i+1}, \"total\": {checkpoints}}}", flush=True)

    # pretend we measured something (but do NOT write files)
    dummy_result = {
        "ok": not args.fail,
        "vt_dummy_V": -0.15 if args.polarity.lower()=="pmos" else 0.15,
        "id_read_A": 1.0e-7,
        "notes": "dummy result; no hardware or file I/O performed",
    }
    print("[pxi] result", json.dumps(dummy_result), flush=True)

    print("[pxi] done", json.dumps({"ts_end": datetime.now().isoformat(timespec="seconds")}), flush=True)

    # exit code to let Autoprobe test failure handling paths
    sys.exit(1 if args.fail else 0)


if __name__ == "__main__":
    main()