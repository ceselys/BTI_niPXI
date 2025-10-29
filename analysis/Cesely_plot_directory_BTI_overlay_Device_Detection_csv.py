import os
from pathlib import Path
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.lines import Line2D

# ===================== USER CONFIG =====================
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR       = WORKSPACE_ROOT / "CMU_BTI_Testing" / "analysis" / "Wafer9_PMOS_NBTI_EOT_2.73_EF_6.0" / "lc_0.20_lch_0.40" / "DATA"
OUT_DIR        = WORKSPACE_ROOT / "CMU_BTI_Testing" / "analysis" / "Wafer9_PMOS_NBTI_EOT_2.73_EF_6.0" / "lc_0.20_lch_0.40" / "Plots"

# Force a branch: "left" (pMOS-like), "right" (nMOS-like), or None for auto
BRANCH_OVERRIDE = "left"

# Make overlay plots by groups (polarity, nf)
MAKE_OVERLAYS = True

# =======================================================

OUT_DIR.mkdir(parents=True, exist_ok=True)

def is_device_dir(p: Path) -> bool:
    required = [
        "initial_iv.csv",
        "nbti_id_vs_time_stress.csv",
        "nbti_id_vs_time_relax.csv",
        "config.json",
    ]
    missing = [f for f in required if not (p / f).exists()]
    if missing:
        print(f"[skip] {p.name}: missing {missing}")
        return False
    return True

def is_device_dir_og(p: Path) -> bool:
    # A “device dir” is any folder that contains the required files
    required = [ "initial_iv.csv", "nbti_id_vs_time_stress.csv", "nbti_id_vs_time_relax.csv", "config.json", ] 
    return all((p / fname).exists() for fname in required)

def loglog_slope(x, y):
    """
    Return slope m from linear fit in log10 space: log10(y) = m*log10(x) + b.
    Returns None if insufficient usable points.
    """
    x = np.asarray(x); y = np.asarray(y)
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return None
    X = np.log10(x[mask]); Y = np.log10(y[mask])
    m, b = np.polyfit(X, Y, 1)
    Yh = m*X + b
    ss_res = np.sum((Y - Yh)**2)
    ss_tot = np.sum((Y - np.mean(Y))**2)
    R2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return float(m), float(R2)

def pick_branch(vgs, id_abs, branch="left"):
    i_min = np.argmin(id_abs)
    if branch == "left":
        V = vgs[:i_min+1]; I = id_abs[:i_min+1]
    else:
        V = vgs[i_min:];   I = id_abs[i_min:]
    s = np.argsort(V); V, I = V[s], I[s]
    return V, I

def fit_piecewise_models(V, I, n_sub=8):
    idx_sub = np.argsort(I)[:max(3, min(n_sub, len(I)//2))]
    V_sub, I_sub = V[idx_sub], I[idx_sub]
    s = np.argsort(V_sub); V_sub, I_sub = V_sub[s], I_sub[s]
    a, b = np.polyfit(V_sub, np.log(I_sub + 1e-30), 1)
    sub_params = (a, b)

    mask_super = np.ones(len(V), dtype=bool)
    mask_super[idx_sub] = False
    V_super, I_super = V[mask_super], I[mask_super]

    if len(V_super) < 3:
        return sub_params, None, float(I_sub.max())

    c2, c1, c0 = np.polyfit(V_super, I_super, 2)
    quad_params = (c2, c1, c0)
    I_sub_edge = float(np.exp(a * V_sub[-1] + b))
    I_knee = max(I_sub_edge, float(np.percentile(I_super, 5)))
    return sub_params, quad_params, I_knee

def invert_piecewise(I_target, sub_params, quad_params, I_knee, V_range):
    a, b = sub_params
    I_target = float(max(I_target, 1e-30))
    if (quad_params is None) or (I_target <= I_knee):
        return (np.log(I_target) - b) / (a + 1e-30)

    c2, c1, c0 = quad_params
    A, B, C = c2, c1, c0 - I_target
    disc = B*B - 4*A*C
    Vmin, Vmax = V_range
    center = 0.5 * (Vmin + Vmax)

    if disc < 0:
        vtx = -B / (2*A + 1e-30)
        return float(np.clip(vtx, Vmin, Vmax))

    r1 = (-B + np.sqrt(disc)) / (2*A + 1e-30)
    r2 = (-B - np.sqrt(disc)) / (2*A + 1e-30)
    candidates = np.array([r1, r2])
    in_range = (candidates >= Vmin) & (candidates <= Vmax)
    if in_range.any():
        return float(candidates[in_range][np.argmin(np.abs(candidates[in_range] - center))])
    return float(np.clip(candidates[np.argmin(np.abs(candidates - center))], Vmin, Vmax))

def choose_branch_auto(vgs_curve, id_abs_curve, vgs_stress_series):
    i_min = np.argmin(id_abs_curve)
    V_left  = vgs_curve[:i_min+1]
    V_right = vgs_curve[i_min:]
    Vg_med = float(np.median(vgs_stress_series))
    dist_left  = min(abs(Vg_med - V_left.min()),  abs(Vg_med - V_left.max()))
    dist_right = min(abs(Vg_med - V_right.min()), abs(Vg_med - V_right.max()))
    return "left" if dist_left <= dist_right else "right"

def load_bti_time_csv(p: Path):
    """
    nbti_id_vs_time_*.csv with columns: t, i_d
    Returns time (np.ndarray) and |i_d| (np.ndarray).
    """
    df = pd.read_csv(p)
    t = df["t"].to_numpy()
    i_abs = np.abs(df["i_d"].to_numpy())
    return t, i_abs

def load_initial_iv_csv(p: Path):
    """
    initial_iv.csv columns: v_wl, i_wl, v_sl, i_sl, v_bl, i_bl
    wl=gate, sl=source, bl=drain
    Returns: vgs_curve, id_abs_curve, vds_compare
    """
    df = pd.read_csv(p)
    Vgs = df["v_wl"].to_numpy() - df["v_sl"].to_numpy()
    Id  = np.abs(df["i_bl"].to_numpy())
    Vds_compare = df["v_bl"].to_numpy() - df["v_sl"].to_numpy()  # per your request
    s = np.argsort(Vgs)
    return Vgs[s], Id[s], Vds_compare[s]

def load_config_json(path: Path) -> dict:
    """
    Reads device_dir/config.json and returns a dict with:
      vgs_stress, vgs_read, vds, polarity (lowercased: 'nmos'/'pmos' if present)
    Computes Vgs values w.r.t. source: Vgs = Vg - Vs; Vds = Vd - Vs.
    """
    with open(path, "r") as f:
        cfg = json.load(f)

    # Default 0 if not present
    v_s = float(cfg.get("v_s", 0.0))
    v_d = float(cfg.get("v_d", 0.0))
    v_g_stress = float(cfg.get("v_g_stress", cfg.get("v_gate", 0.0)))
    v_g_read   = float(cfg.get("v_g_read", cfg.get("v_read_gate_bias", 0.0)))

    vgs_stress = v_g_stress - v_s
    vgs_read   = v_g_read   - v_s
    vds        = v_d        - v_s

    pol = cfg.get("polarity", "").strip().lower()  # e.g., "PMOS" -> "pmos"
    if pol not in ("pmos", "nmos"):
        pol = None

    return {
        "vgs_stress": vgs_stress,
        "vgs_read": vgs_read,
        "vds": vds,
        "polarity": pol,
        "raw": cfg,  # keep raw in case you want other fields later
    }

def process_device_dir(device_dir: Path, out_dir: Path, branch_override=None):
    """
    Requires these files in each device folder:
      - initial_iv.csv                    (v_wl, i_wl, v_sl, i_sl, v_bl, i_bl)
      - nbti_id_vs_time_stress.csv        (t, i_d)
      - nbti_id_vs_time_relax.csv         (t, i_d)
      - config.json                       (provides vgs_stress, vgs_read, vds, polarity, etc.)

    Returns: dict with per-device results for overlays, or None if skipped/QC-fail.
      {
        'name': str,
        'meta': {'polarity':..., 'nf': ...},
        'stress': {'t': array, 'abs_dvt': array},
        'relax' : {'t': array, 'abs_dvt': array},
        'qc'    : {'m_stress': float, 'm_relax': float, 'keep': bool}
      }
    Also saves the per-device 2×2 panel PNG to out_dir.
    """
    # Required files
    csv_iv  = device_dir / "initial_iv.csv"
    csv_str = device_dir / "nbti_id_vs_time_stress.csv"
    csv_rel = device_dir / "nbti_id_vs_time_relax.csv"
    cfg_p   = device_dir / "config.json"
    if not (csv_iv.exists() and csv_str.exists() and csv_rel.exists() and cfg_p.exists()):
        return None

    try:
        # ---------- Load config for Vgs/Vds and metadata ----------
        cfg = load_config_json(cfg_p)
        vgs_stress_scalar = cfg["vgs_stress"]
        vgs_read_scalar   = cfg["vgs_read"]
        vds_scalar        = cfg["vds"]

        # ---------- Baseline IV from initial_iv.csv ----------
        ivdf = pd.read_csv(csv_iv)
        vgs_curve = (ivdf["v_wl"].to_numpy() - ivdf["v_sl"].to_numpy())
        id_abs_curve = np.abs(ivdf["i_bl"].to_numpy())
        vds_compare  = (ivdf["v_bl"].to_numpy() - ivdf["v_sl"].to_numpy())  # kept if you want QC on it

        # Sort baseline by Vgs
        s = np.argsort(vgs_curve)
        vgs_curve   = vgs_curve[s]
        id_abs_curve= id_abs_curve[s]
        vds_compare = vds_compare[s]

        # ---------- Time series from CSVs (t, |i_d|) ----------
        t_stress, iabs_stress = load_bti_time_csv(csv_str)
        t_relax,  iabs_relax  = load_bti_time_csv(csv_rel)

        # ---------- Branch selection ----------
        # Prefer measured read Vgs from config if nonzero, else stress Vgs, else parse dir token
        def nz(x): return (x is not None) and np.isfinite(x) and (abs(x) > 0)
        if nz(vgs_read_scalar):
            vgs_for_branch = float(vgs_read_scalar)
        elif nz(vgs_stress_scalar):
            vgs_for_branch = float(vgs_stress_scalar)
        else:
            vgs_for_branch = parse_stress_vgs_from_path(device_dir)

        if branch_override in ("left", "right"):
            branch = branch_override
        else:
            branch = choose_branch_auto(vgs_curve, id_abs_curve, vgs_for_branch)

        V_branch, I_branch = pick_branch(vgs_curve, id_abs_curve, branch=branch)

        # ---------- Piecewise fit of baseline |Id|(Vgs) ----------
        sub_params, quad_params, I_knee = fit_piecewise_models(V_branch, I_branch, n_sub=8)
        Vmin, Vmax = float(V_branch.min()), float(V_branch.max())

        def vgs_equiv_from_abs_id(I_target):
            return invert_piecewise(I_target, sub_params, quad_params, I_knee, (Vmin, Vmax))

        # ---------- ΔVt(t) ----------
        # Same preference policy as before: use vgs_read if given, else vgs_stress, else vgs_for_branch
        def compute_dvt(iabs_series, prefer_read=True):
            vgs_equiv = np.array([vgs_equiv_from_abs_id(x) for x in iabs_series])
            if prefer_read and nz(vgs_read_scalar):
                vgs_meas = vgs_read_scalar
            elif nz(vgs_stress_scalar):
                vgs_meas = vgs_stress_scalar
            else:
                vgs_meas = vgs_for_branch
            return np.abs(vgs_meas - vgs_equiv)

        dvt_stress = compute_dvt(iabs_stress, prefer_read=True)
        dvt_relax  = compute_dvt(iabs_relax,  prefer_read=True)

        # ---------- QC ----------
        m_stress, R2_stress = loglog_slope(t_stress, dvt_stress)
        m_relax,  R2_relax  = loglog_slope(t_relax,  dvt_relax)


        # ---------- Fraction remaining vs FINAL stress ΔVt ----------
        eps = 1e-30
        idx_final_stress = int(np.argmax(t_stress))
        baseline_stress_final = max(float(dvt_stress[idx_final_stress]), eps)
        frac_remaining = dvt_relax / baseline_stress_final

        # ---------- Forward fit curve for panel ----------
        V_dense = np.linspace(Vmin, Vmax, 400)
        I_fit = []
        a, b = sub_params
        for V in V_dense:
            I_sub = np.exp(a * V + b)
            if quad_params is None:
                I_fit.append(I_sub)
            else:
                c2, c1, c0 = quad_params
                I_quad = c2 * V * V + c1 * V + c0
                I_fit.append(I_quad if I_sub > I_knee else I_sub)
        I_fit = np.array(I_fit)

        # ---------- 2×2 panel ----------
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        # Optional: show config-derived biases in the title
        fig.suptitle(
            f"{device_dir.name}  |  branch={branch}  |  Vgs_read={vgs_read_scalar:.3f} V  Vgs_stress={vgs_stress_scalar:.3f} V  Vds={vds_scalar:.3f} V",
            fontsize=10
        )

        ax = axes[0, 0]
        ax.loglog(t_stress, dvt_stress, 'o-')
        ax.set_xlabel("Time (s)"); ax.set_ylabel("|ΔVt| (V)")
        ax.set_title("Stress |ΔVt| vs Time"); ax.grid(True, which="both")

        ax = axes[0, 1]
        ax.loglog(t_relax, dvt_relax, 'o-')
        ax.set_xlabel("Time (s)"); ax.set_ylabel("|ΔVt| (V)")
        ax.set_title("Relax |ΔVt| vs Time"); ax.grid(True, which="both")

        ax = axes[1, 0]
        ax.semilogx(t_relax, frac_remaining, 'o-')
        ax.set_xlabel("Time (s)"); ax.set_ylabel("ΔVt Fraction Remaining")
        ax.set_ylim(0, 1.05); ax.set_title("Relax Fraction Remaining"); ax.grid(True, which="both")

        ax = axes[1, 1]
        ax.plot(vgs_curve, id_abs_curve, 'o', label="Baseline |Id| (both)")
        ax.plot(V_branch, I_branch, 'o', label=f"Branch: {branch}")
        ax.plot(V_dense, I_fit, '-', label="Piecewise fit")
        ax.set_yscale("log")
        ax.set_xlabel("Vgs (V)"); ax.set_ylabel("|Id| (A)")
        ax.set_title("|Id|-Vgs Baseline (fit)"); ax.grid(True, which="both")
        ax.legend(fontsize=8)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"bti_{device_dir.name}.png", dpi=300)
        plt.close(fig)


        return {
            "name": device_dir.name,
            "stress": {"t": t_stress, "abs_dvt": dvt_stress},
            "relax":  {"t": t_relax,  "abs_dvt": dvt_relax},
            "qc": {"m_stress": m_stress, "m_relax": m_relax},
            "derived": {
                "Vds_compare": vds_compare,   # from IV sweep
                "Vds_cfg": vds_scalar,        # from config.json
                "Vgs_read_cfg": vgs_read_scalar,
                "Vgs_stress_cfg": vgs_stress_scalar,
            },
        }

    except Exception as e:
        print(f"[skip] {device_dir.name}: {e}")
        return None


# --- replace your helper with this version (adds line handle return) ---
def powerlaw_fit_and_plot(ax, x_all, y_all, label_prefix="fit", linestyle="--"):
    """
    Fit y = A * x^m in log10 space and draw it.
    Returns (m, A, R2, line_handle).
    """
    mask = (x_all > 0) & (y_all > 0) & np.isfinite(x_all) & np.isfinite(y_all)
    X = np.log10(x_all[mask]); Y = np.log10(y_all[mask])
    if len(X) < 2:
        return None, None, None, None

    m, b = np.polyfit(X, Y, 1)           # Y = m*X + b  =>  y = 10^b * x^m
    y_pred = m*X + b
    ss_res = np.sum((Y - y_pred)**2)
    ss_tot = np.sum((Y - np.mean(Y))**2)
    R2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    A = 10**b

    x_fit = np.linspace(x_all[mask].min(), x_all[mask].max(), 200)
    y_fit = A * (x_fit**m)
    (line,) = ax.loglog(
        x_fit, y_fit, linestyle, linewidth=2,
        label=f"{label_prefix}: ΔVt = {A:.2e}·t^{m:.3f}  (R²={R2:.3f})"
    )
    return m, A, R2, line


def semilog_fit_and_plot(ax, x_all, y_all, label_prefix="Combined fit (fraction)", linestyle="--"):
    """
    Linear fit on semilog-x axes: y = m*log10(t) + b
    Returns (m, b, R2, line_handle).
    """
    mask = (x_all > 0) & np.isfinite(x_all) & np.isfinite(y_all)
    X = np.log10(x_all[mask]); Y = y_all[mask]
    if len(X) < 2:
        return None, None, None, None

    m, b = np.polyfit(X, Y, 1)
    y_pred = m*X + b
    ss_res = np.sum((Y - y_pred)**2)
    ss_tot = np.sum((Y - np.mean(Y))**2)
    R2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

    x_fit = np.linspace(x_all[mask].min(), x_all[mask].max(), 200)
    y_fit = m*np.log10(x_fit) + b
    y_fit = np.clip(y_fit, 0.0, 1.0)  # fraction is [0,1]

    (line,) = ax.semilogx(
        x_fit, y_fit, linestyle, linewidth=2,
        label=f"{label_prefix}: y = {m:.3f}·log10(t) + {b:.3f}  (R²={R2:.3f})"
    )
    return m, b, R2, line


def make_overlays(results_by_group, out_dir: Path):
    """
    results_by_group[(polarity, nf)] -> list of per-device dicts
    Makes two overlays per group: stress and relax (|ΔVt| vs time).
    """
    for (pol, nf), items in results_by_group.items():
        # ----- STRESS OVERLAY -----
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        x_all, y_all = [], []
        for d in items:
            t = d["stress"]["t"]; y = d["stress"]["abs_dvt"]
            ax.loglog(t, y, 'o', alpha=0.6, label=d["name"])
            x_all.append(t); y_all.append(y)
        if x_all:
            x_all = np.concatenate(x_all); y_all = np.concatenate(y_all)
            powerlaw_fit_and_plot(ax, x_all, y_all, label_prefix="Group fit")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("|ΔVt| (V)")
        ax.set_title(f"STRESS overlay: {pol}, nf={nf}  (all devices)")
        ax.grid(True, which="both"); ax.legend(fontsize=7, ncol=1)
        fig.tight_layout()
        fig.savefig(out_dir / f"overlay_{pol}_nf{nf}_stress.png", dpi=300)
        plt.close(fig)

        # ----- RELAX OVERLAY -----
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        x_all, y_all = [], []
        for d in items:
            t = d["relax"]["t"]; y = d["relax"]["abs_dvt"]
            ax.loglog(t, y, 'o', alpha=0.6, label=d["name"])
            x_all.append(t); y_all.append(y)
        if x_all:
            x_all = np.concatenate(x_all); y_all = np.concatenate(y_all)
            powerlaw_fit_and_plot(ax, x_all, y_all, label_prefix="Group fit")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("|ΔVt| (V)")
        ax.set_title(f"RELAX overlay: {pol}, nf={nf}  (all devices)")
        ax.grid(True, which="both"); ax.legend(fontsize=7, ncol=1)
        fig.tight_layout()
        fig.savefig(out_dir / f"overlay_{pol}_nf{nf}_relax.png", dpi=300)
        plt.close(fig)


def main():
    # Allow BASE_DIR to be either the parent (with many device subfolders)
    # or a single device folder itself.

    if is_device_dir(BASE_DIR):
        device_dirs = [BASE_DIR]
    else:
        device_dirs = sorted([p for p in BASE_DIR.iterdir() if p.is_dir() and is_device_dir(p)])

    if not device_dirs:
        print(f"No device directories under: {BASE_DIR}")
        return

    # Keep the same data structure shape: dict -> list
    # but use a single, simple key instead of (polarity, nf)
    results_by_group = defaultdict(list)

    n_ok = 0
    for dd in device_dirs:
        res = process_device_dir(dd, OUT_DIR, branch_override=BRANCH_OVERRIDE)
        if res is None:
            continue
        n_ok += 1

        # Single bucket for everything (no meta-based grouping)
        key = ("all", None)
        results_by_group[key].append(res)

    print(f"Generated {n_ok} per-device panel PNG(s) to: {OUT_DIR}")

    # Overlays still work — they’ll just be computed on the single "(all, None)" group
    if MAKE_OVERLAYS and results_by_group:
        make_overlays(results_by_group, OUT_DIR)
        print(f"Overlay PNGs written to: {OUT_DIR}")

    #make_nf_combined_overlays(results_by_group, OUT_DIR)
    #print(f"Combined overlays written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
