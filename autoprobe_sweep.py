#!/usr/bin/env python3
import argparse, json, time, shlex, subprocess, sys
from pathlib import Path
import sys

# TOML loader with fallback
try:
    import tomllib  # Python 3.11+
    def _load_toml_bytes(p: Path) -> dict:
        with open(p, "rb") as f:
            return tomllib.load(f)
except ModuleNotFoundError:
    import toml  # pip install toml
    def _load_toml_bytes(p: Path) -> dict:
        return toml.loads(p.read_text())

from autoprobe.cascade import CascadeAutoProbe

CONFIG_DIR = Path(r"C:\Users\RSG\Documents\CMU_BTI_Testing\autoprobe_config")
WAFER_JSON   = CONFIG_DIR / "wafer.json"
MODULES_JSON = CONFIG_DIR / "skywater_cnfet_sputter_scaling_v1_fet_modules.json"
INVERT_DIRECTION = True  # flip sign on both X and Y if your die indexing is opposite stage axes
# Global start position (die indices + module name)
START_POS = {
    "die_x": 0,   # <-- set your actual starting die X
    "die_y": 0,   # <-- set your actual starting die Y
    "module": "mod_dummy_fet_tlm_nmos_lc_0.40_lch_0.40_lov_0.06_gateasym_0.00_w_4.0_3"
}

# ---------- Helpers ----------
def load_toml(p: Path) -> dict:
    return _load_toml_bytes(p)

def load_MODULES_JSON(MODULES_JSON: Path) -> dict:
    with open(MODULES_JSON, "r") as f:
        return json.load(f)

def validate_modules_exist(MODULES_JSON: Path, requested: list[str]) -> list[str]:
    m = load_MODULES_JSON(MODULES_JSON)
    all_names = set(m.keys())
    return [nm for nm in requested if nm not in all_names]

def load_sweep_toml(sweep_toml: Path) -> tuple[list[tuple[int,int]], list[str]]:
    cfg = load_toml(sweep_toml)
    dies = cfg.get("dies") or [[0, 0]]
    dies_norm = [(int(dx), int(dy)) for dx, dy in dies]
    modules = [str(x) for x in (cfg.get("sweep", {}).get("modules") or [])]
    if not modules:
        raise ValueError("No modules found in sweep TOML ([sweep].modules).")
    return dies_norm, modules

def build_pxi_cmd_from_measurement(
    tcfg: dict, module_name: str, device_name: str, cli_gpib: str | None
) -> list[str]:
    meas = tcfg.get("measurement", {}) or {}
    overrides = (tcfg.get("module_overrides", {}) or {}).get(module_name, {}) or {}
    v = {**meas, **overrides}

    script   = str(v.get("script", "pxi.py"))
    settings = str(v.get("settings", "settings.json"))
    chip     = str(v.get("chip", "CHIP"))
    toml_gpib = v.get("gpib")
    duration = float(v.get("duration", 2.0))
    invert   = bool(v.get("invert", False))
    extra    = [str(x) for x in (v.get("extra", []) or [])]

    cmd = [sys.executable, script, settings, chip, device_name]
    gpib = cli_gpib or toml_gpib
    if gpib:
        cmd += ["--gpib", gpib]
    cmd += ["--start-module", module_name, "--duration", str(duration)]
    if invert:
        cmd.append("--invert")
    if extra:
        cmd += extra
    return cmd

def run_pxi(cmd: list[str]) -> int:
    print(f"[pxi] START: {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in proc.stdout:
            print(line.rstrip())
    finally:
        proc.wait()
    print(f"[pxi] END rc={proc.returncode}", flush=True)
    return proc.returncode

# ---------- Die pitch resolve (accepts your wafer.json) ----------
# ---------- Die pitch resolve (from wafer.json only) ----------
def resolve_die_pitch(wafer_json: Path) -> tuple[float, float]:
    """
    Load die pitch (dx, dy) directly from wafer.json.
    Expects keys:
        die_size_x  -> die pitch in X
        die_size_y  -> die pitch in Y
    """
    with open(wafer_json, "r") as f:
        w = json.load(f)

    if "die_size_x" not in w or "die_size_y" not in w:
        raise KeyError(
            f"wafer.json must define both 'die_size_x' and 'die_size_y' (found keys: {list(w.keys())})"
        )

    dx = float(w["die_size_x"])
    dy = float(w["die_size_y"])

    if dx == 0 or dy == 0:
        raise ValueError("die_size_x and die_size_y must be non-zero values in wafer.json.")

    return dx, dy


# ---------- Robust relative die move + re-home ----------
def _maybe_wait_motion(probe, fallback_s=0.5):
    # If your wrapper exposes a wait method, use it; otherwise short sleep.
    if hasattr(probe, "wait_for_motion_complete") and callable(probe.wait_for_motion_complete):
        try:
            probe.wait_for_motion_complete()
            return
        except Exception:
            pass
    time.sleep(fallback_s)

def safe_move_to_die_relative(probe: CascadeAutoProbe,
                              target_die: tuple[int,int],
                              state: dict,
                              die_pitch: tuple[float,float]):
    cur_x, cur_y = state["current_die_x"], state["current_die_y"]
    die_x, die_y = map(int, target_die)
    if (die_x, die_y) == (cur_x, cur_y):
        return

    die_dx, die_dy = die_pitch

    # sign flips if needed
    s = -1.0 if INVERT_DIRECTION else 1.0
    steps_x = die_x - cur_x
    steps_y = die_y - cur_y
    dx_to_die = s * steps_x * die_dx
    dy_to_die = s * steps_y * die_dy

    print("Contacts UP before die move…")
    probe.move_contacts_up(); time.sleep(0.5)

    print(f"[die=({die_x},{die_y})] Relative chuck move: ΔX={dx_to_die}, ΔY={dy_to_die}")
    # Use high-level relative move instead of raw GPIB
    probe.move_chuck_relative_to_home(x=dx_to_die, y=dy_to_die)
    _maybe_wait_motion(probe, fallback_s=0.8)

    # Re-home at the new die so modules remain local to (0,0)
    if hasattr(probe, "set_chuck_home") and callable(probe.set_chuck_home):
        probe.set_chuck_home()
    elif hasattr(probe, "set_home") and callable(probe.set_home):
        probe.set_home()
    else:
        # If no explicit home-set method exists, we can at least pause to settle
        print("[warn] No set_chuck_home()/set_home() method found; continuing without re-home.")
        time.sleep(0.2)

    state["current_die_x"] = die_x
    state["current_die_y"] = die_y
    time.sleep(0.2)

# ----------NBTI CONTROLS SCRIPT----------

def _maybe_add_boolean_optional(args_list, flag_name, value, default):
    """
    Handle argparse.BooleanOptionalAction flags.
    If value differs from default, emit --flag (for True) or --no-flag (for False).
    """
    if value is None:
        return
    if bool(value) is bool(default):
        return
    args_list.append(f"--{flag_name}" if value else f"--no-{flag_name}")

def _maybe_add_scalar(args_list, flag_name, value):
    """Append '--flag value' if value is not None."""
    if value is None:
        return
    args_list += [f"--{flag_name.replace('_','-')}", str(value)]

def _maybe_add_list(args_list, flag_name, value_list):
    """Append '--flag a b c' for list-like values (e.g., initial_sweep)."""
    if not value_list:
        return
    args_list.append(f"--{flag_name.replace('_','-')}")
    args_list += [str(v) for v in value_list]

def _merge_measurement_block(tcfg: dict, module_name: str) -> dict:
    meas = (tcfg.get("measurement") or {})
    overrides = ((tcfg.get("module_overrides") or {}).get(module_name) or {})
    v = {**meas, **overrides}
    return v

def build_nbti_cmd_from_toml(
    tcfg: dict,
    module_name: str,
    device_name: str,
) -> list[str]:
    """
    Build a sys.executable command list for ./scripts/nbti.py using
    a TOML config with [measurement] and [module_overrides.<module_name>].

    Positional args: settings, chip, device
    """
    v = _merge_measurement_block(tcfg, module_name)

    # Positional bits
    script   = str(v.get("script", "scripts/nbti.py"))
    settings = str(v.get("settings", "settings.json"))
    chip     = str(v.get("chip", "CHIP"))

    cmd = [sys.executable, script, settings, chip, device_name]

    # --- Simple string/float/int args ---
    _maybe_add_scalar(cmd, "polarity",         v.get("polarity"))
    _maybe_add_scalar(cmd, "tstart",           v.get("tstart"))
    _maybe_add_scalar(cmd, "tend",             v.get("tend"))
    _maybe_add_scalar(cmd, "tstart-relax",     v.get("tstart_relax"))
    _maybe_add_scalar(cmd, "tend-relax",       v.get("tend_relax"))
    _maybe_add_scalar(cmd, "samples",          v.get("samples"))
    _maybe_add_scalar(cmd, "samples-relax",    v.get("samples_relax"))
    _maybe_add_scalar(cmd, "stress-relax-cycles", v.get("stress_relax_cycles"))
    _maybe_add_scalar(cmd, "read-bias",        v.get("read_bias"))
    _maybe_add_scalar(cmd, "read-gate-bias",   v.get("read_gate_bias"))
    _maybe_add_scalar(cmd, "gate-bias",        v.get("gate_bias"))
    _maybe_add_scalar(cmd, "boost-voltage",    v.get("boost_voltage"))
    _maybe_add_scalar(cmd, "boost-sleep",      v.get("boost_sleep"))
    _maybe_add_scalar(cmd, "efield",           v.get("efield"))
    _maybe_add_scalar(cmd, "eot",              v.get("eot"))
    _maybe_add_scalar(cmd, "vt-min",           v.get("vt_min"))
    _maybe_add_scalar(cmd, "vt-max",           v.get("vt_max"))
    _maybe_add_scalar(cmd, "initial-sweep-sleep", v.get("initial_sweep_sleep"))
    _maybe_add_scalar(cmd, "ac-freq",          v.get("ac_freq"))
    _maybe_add_scalar(cmd, "ac-mode",          v.get("ac_mode"))
    _maybe_add_scalar(cmd, "dutycycle",        v.get("dutycycle"))
    _maybe_add_scalar(cmd, "pattern",          v.get("pattern"))
    _maybe_add_scalar(cmd, "t-unit",           v.get("t_unit"))
    _maybe_add_scalar(cmd, "data-folder", v.get("data_folder"))

    # --- List args ---
    _maybe_add_list(cmd, "initial-sweep", v.get("initial_sweep"))

    # --- BooleanOptionalAction flags & their argparse defaults ---
    # Defaults per your parser:
    # stress_iv default False
    # relax     default True
    # relax_iv  default False
    # ac        default False
    # clamp_vt  default False
    _maybe_add_boolean_optional(cmd, "stress-iv", v.get("stress_iv"), default=False)
    _maybe_add_boolean_optional(cmd, "relax",     v.get("relax"),     default=True)
    _maybe_add_boolean_optional(cmd, "relax-iv",  v.get("relax_iv"),  default=False)
    _maybe_add_boolean_optional(cmd, "ac",        v.get("ac"),        default=False)
    _maybe_add_boolean_optional(cmd, "clamp-vt",  v.get("clamp_vt"),  default=False)

    # Optional extra passthroughs
    extra = v.get("extra") or []
    cmd += [str(x) for x in extra]

    return cmd


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Autoprobe sweep: dies + modules from TOML; PXI from measurement TOML.")
    # Motion/bootstrap
    ap.add_argument("--data-folder", default="C:\\Users\\RSG\\Documents\\CMU_BTI_Testing\\_DATA\\Wafer9_PMOS_NBTI_AC_1kHz_Duty_0.5_Vgs_-1.2V", type=Path, help="Run root; also used by CascadeAutoProbe")
    ap.add_argument("--gpib", default="GPIB1::22::INSTR", help="GPIB VISA (CLI override for TOML gpib).")

    # TOML configs
    ap.add_argument("--config-toml", default="C:\\Users\\RSG\\Documents\\CMU_BTI_Testing\\build\\measurements\\nbti_AC.toml", type=Path, help="Measurement TOML (pxi.py + args).")
    ap.add_argument("--sweep-toml",  default="C:\\Users\\RSG\\Documents\\CMU_BTI_Testing\\build\\sweeps\\pmos_nbti.toml", type=Path, help="Sweep TOML (dies + [sweep].modules).")

    # Behavior
    ap.add_argument("--stop-on-fail", action="store_true", help="Abort sweep on the first PXI failure.")


    args = ap.parse_args()

    if not WAFER_JSON.exists():
        raise FileNotFoundError(f"Cannot find wafer.json at {WAFER_JSON}")
    if not MODULES_JSON.exists():
        raise FileNotFoundError(f"Cannot find modules.json at {MODULES_JSON}")
    
    start_die_x = int(START_POS["die_x"])
    start_die_y = int(START_POS["die_y"])
    start_module = str(START_POS["module"])


    meas_tcfg = load_toml(args.config_toml)

    dies_list, modules_to_sweep = load_sweep_toml(args.sweep_toml)
    if not dies_list:
        dies_list = [(start_die_x, start_die_y)]

    missing = validate_modules_exist(MODULES_JSON, modules_to_sweep)
    if missing:
        print("[sweep] ERROR: The following modules are not defined in modules.json:")
        for nm in missing:
            print("  -", nm)
        sys.exit(2)

    die_pitch = resolve_die_pitch(WAFER_JSON)
    print(f"[sweep] Using die pitch: dx={die_pitch[0]}, dy={die_pitch[1]}")

    if not args.gpib:
        raise ValueError("You must specify a GPIB address using --gpib (no fallback from TOML).")
    gpib_for_motion = args.gpib


    print("\n=== Motion + PXI (TOML-driven) ===")
    probe = CascadeAutoProbe(
        gpib_address=gpib_for_motion,
        data_folder=str(args.data_folder),
        die_x=start_die_x,
        die_y=start_die_y,
        current_module=start_module,
        invert_direction=INVERT_DIRECTION,
        path_wafer_metadata=WAFER_JSON,
        path_die_modules=MODULES_JSON,
    )

    state = {"current_die_x": start_die_x, "current_die_y": start_die_y}


    overall_rc = 0
    try:
        print("Contacts UP (safety before any XY move)…")
        probe.move_contacts_up(); time.sleep(0.2)

        print("Move CHUCK HOME…")
        probe.move_chuck_home(); time.sleep(0.2)

        for (dx, dy) in dies_list:
            print(f"\n=== Move to die ({dx},{dy}) ===")
            # robust relative move + re-home at new die
            safe_move_to_die_relative(probe, (int(dx), int(dy)), state, die_pitch)

            # module loop in local die frame
            for idx, mod_name in enumerate(modules_to_sweep):
                device_name = f"{mod_name}"
                print(f"\n- Move to module: {mod_name} (device={device_name})")
                probe.move_contacts_up(); time.sleep(0.2)
                probe.move_to_module(mod_name); time.sleep(0.2)

                print("  Contacts DOWN…")
                probe.move_contacts_down(); time.sleep(0.1)

                effective_gpib = args.gpib or None
                cmd = build_nbti_cmd_from_toml(
                    meas_tcfg, module_name=mod_name, device_name=device_name
                )
                rc = run_pxi(cmd)
                overall_rc = rc if rc != 0 else overall_rc

                print("  Contacts UP…")
                probe.move_contacts_up(); time.sleep(0.2)

                if rc != 0 and args.stop_on_fail:
                    print(f"[sweep] PXI failed on module '{mod_name}' (rc={rc}); stopping early.")
                    sys.exit(rc)

        print("\nSweep complete. Leaving contacts UP.")
        probe.move_contacts_up()
    finally:
        pass

    sys.exit(overall_rc)

if __name__ == "__main__":
    main()