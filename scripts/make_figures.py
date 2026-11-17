"""Figures of the computational study, generated from the campaign outputs.

    python scripts/make_figures.py --det results/final_det --stoch results/final_stoch results/final_stoch_ind3d --out figures

Produces (PDF, print-ready):
  fig_regime.pdf      structural ratios of every instance, marker = certified gap regime
  fig_detgap.pdf      certified gap vs size, ours vs CP-SAT
  fig_policygap.pdf   gap of each policy to the PI certificate vs size (synthetic seeds + plant)
  fig_paired.pdf      per-scenario paired differences to the static plan, one panel per size class
  fig_cvar.pdf        CVaR95 against mean makespan per policy, both normalized by the PI certificate
  fig_plant.pdf       the two plant windows: policies against the PI band
  fig_ecdf.pdf        per-scenario regret ratio C(policy)/Phi_omega as ECDF, one panel per plant window
  fig_hierarchy.pdf   bound hierarchy per instance (energetic, head-tail, Lagrangian, CG, destructive, CP-SAT) / best UB
  fig_convergence.pdf anytime traces (certified LB, RMP value, Lagrangian, primal UB) parsed from the campaign log (--log)
"""
import argparse, csv, glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

# validated categorical order (blue, orange, aqua, yellow, magenta, green); text in ink, never series color
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
POLS = [("static", "static plan"), ("reactive-plan", "plan-order repair"), ("reactive-MWR", "MWR"),
        ("reactive-SPT", "SPT"), ("replan", "list re-plan"), ("replan-opt", "CP re-optimization")]
PCOL = {k: C[i] for i, (k, _) in enumerate(POLS)}
plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "grid.color": GRID, "grid.linewidth": 0.5, "legend.frameon": False, "figure.dpi": 150})


def fnum(r, k):
    v = r.get(k, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def load_det(prefix):
    rows = []
    for f in sorted(glob.glob(prefix + "_*.csv")):
        g = re.sub(r"\d+$", "", os.path.basename(f)[len(os.path.basename(prefix)) + 1:-4])
        for r in csv.DictReader(open(f)):
            lb = max(x for x in [fnum(r, "lb_ht"), fnum(r, "lb_cg"), fnum(r, "lb_destructive")] if x == x)
            ub = fnum(r, "best_ub"); lbcp = fnum(r, "lb_cpsat")
            cp_ok = lbcp == lbcp and lbcp > 0 and r.get("cpsat_status") != "UNKNOWN"
            rows.append(dict(group=g, name=r["instance"], ops=int(fnum(r, "ops")), wl=fnum(r, "ratio_workload"),
                             path=fnum(r, "ratio_path"), gap=100 * (ub - lb) / lb,
                             gap_cp=(100 * (ub - lbcp) / lbcp) if cp_ok else float("nan"), limit=fnum(r, "time_limit")))
    return rows


def load_stoch(prefixes):
    out = []
    for p in prefixes:
        for f in sorted(glob.glob(p + "_*.json")):
            d = json.load(open(f)); r = d["row"]
            d["cert"] = r.get("pi_cp_lcb", r["pi_ht_lcb"]); d["pi_mean"] = r.get("pi_cp_mean", r["pi_ht_mean"])
            d["M"] = {k: np.array(v) for k, v in d["makespans"].items()}
            d["plant"] = r["instance"].startswith("ind")
            if any(o["row"]["instance"] == r["instance"] and o["row"]["scenarios"] >= r["scenarios"] for o in out):
                continue   # the same instance from an overlapping prefix: keep the larger run
            out = [o for o in out if o["row"]["instance"] != r["instance"]] + [d]
    return sorted(out, key=lambda d: (d["plant"], d["row"]["ops"]))


def logx(ax, ticks):
    ax.set_xscale("log"); ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks]); ax.xaxis.set_minor_formatter(NullFormatter())


def fig_regime(det, out):
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    grp_marker = {"bench": "o", "balanced": "s", "coupled": "^", "square": "D", "industrial": "P"}
    grp_label = {"bench": "benchmark variants", "balanced": "balanced plants", "coupled": "coupling-dominated",
                 "square": "square plants", "industrial": "plant windows"}
    for g, mk in grp_marker.items():
        rs = [r for r in det if r["group"] == g]
        if not rs: continue
        closed = [r for r in rs if r["gap"] < 0.05]; open_ = [r for r in rs if r["gap"] >= 0.05]
        ax.scatter([r["wl"] for r in closed], [r["path"] for r in closed], marker=mk, s=34, facecolor="white",
                   edgecolor=INK, linewidth=0.9, label=grp_label[g], zorder=3)
        if open_:
            sizes = [30 + 6 * r["gap"] for r in open_]
            ax.scatter([r["wl"] for r in open_], [r["path"] for r in open_], marker=mk, s=sizes, facecolor=C[1],
                       edgecolor=INK, linewidth=0.6, alpha=0.9, zorder=4)
            for r in open_:
                dx = (8, -12) if r["name"].startswith("cpl_c60") else (5, 3)
                ax.annotate(f"{r['gap']:.0f}%", (r["wl"], r["path"]), textcoords="offset points", xytext=dx, fontsize=7, color=INK2)
    ax.plot([], [], marker="o", ls="", mfc="white", mec=INK, label="certified optimal")
    ax.plot([], [], marker="o", ls="", mfc=C[1], mec=INK, label="open gap (area $\\propto$ gap)")
    ax.set_xlabel("workload ratio  $\\max_c W_c / UB$"); ax.set_ylabel("path ratio  $\\max_j L_j / UB$")
    ax.set_xlim(0.3, 1.02); ax.set_ylim(0.1, 1.05); ax.grid(True, alpha=0.6)
    ax.legend(fontsize=6.5, loc="lower left", ncol=1)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_regime.pdf")); plt.close(fig)


def fig_detgap(det, out):
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    syn = [r for r in det if r["group"] != "industrial"]
    ax.scatter([r["ops"] for r in syn], [r["gap"] for r in syn], s=26, color=C[0], label="cell-wise certificate (ours)", zorder=3)
    cp = [r for r in syn if r["gap_cp"] == r["gap_cp"]]
    ax.scatter([r["ops"] for r in cp], [np.minimum(r["gap_cp"], 120) for r in cp], s=26, marker="s", facecolor="white",
               edgecolor=C[1], linewidth=1.2, label="CP-SAT plant model", zorder=3)
    for r in cp:
        if r["gap_cp"] > 120:
            ax.annotate(f"{r['gap_cp']:.0f}%", (r["ops"], 120), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7, color=INK2)
    nob = [r for r in syn if r["gap_cp"] != r["gap_cp"]]
    for r in nob:
        ax.annotate("no CP bound", (r["ops"], r["gap"]), textcoords="offset points", xytext=(6, -10), fontsize=7, color=INK2)
    ind = [r for r in det if r["group"] == "industrial"]
    ax.scatter([r["ops"] for r in ind], [r["gap"] for r in ind], s=26, marker="P", color=C[2], label="plant windows (ours)", zorder=3)
    logx(ax, [50, 100, 200, 500, 1000, 2000, 4500]); ax.set_ylim(-3, 128)
    ax.set_xlabel("operations"); ax.set_ylabel("certified gap  $(UB-LB)/LB$  (%)"); ax.grid(True, axis="y", alpha=0.6)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_detgap.pdf")); plt.close(fig)


def fig_policygap(stoch, out):
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    for k, lab in POLS:
        xs, ys = [], []
        for d in stoch:
            if k in d["M"]:
                xs.append(d["row"]["ops"]); ys.append(100 * (d["M"][k].mean() - d["cert"]) / d["cert"])
        syn = [(x, y) for x, y, d in zip(xs, ys, [d for d in stoch if k in d["M"]]) if not d["plant"]]
        pl = [(x, y) for x, y, d in zip(xs, ys, [d for d in stoch if k in d["M"]]) if d["plant"]]
        if syn: ax.scatter(*zip(*syn), s=22, color=PCOL[k], label=lab, zorder=3)
        if pl: ax.scatter(*zip(*pl), s=40, marker="P", color=PCOL[k], edgecolor=INK, linewidth=0.5, zorder=4)
    ax.plot([], [], marker="P", ls="", color="white", mec=INK, label="plant windows (marker)")
    logx(ax, [33, 50, 100, 160, 230, 440]); ax.set_ylim(0, 65)
    ax.set_xlabel("operations"); ax.set_ylabel("gap to PI certificate (%)"); ax.grid(True, axis="y", alpha=0.6)
    ax.legend(fontsize=6.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_policygap.pdf"), bbox_inches="tight"); plt.close(fig)


def fig_paired(stoch, out):
    classes = [("33--37 ops", lambda d: d["row"]["ops"] < 100 and not d["plant"]),
               ("150--164 ops", lambda d: 100 <= d["row"]["ops"] < 300 and not d["plant"]),
               ("434--448 ops", lambda d: d["row"]["ops"] >= 300 and not d["plant"]),
               ("plant windows", lambda d: d["plant"])]
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.6), sharey=True)
    for ax, (title, sel) in zip(axs, classes):
        ds = [d for d in stoch if sel(d)]
        data, labels, cols = [], [], []
        for k, lab in POLS[1:]:
            v = np.concatenate([100 * (d["M"][k] - d["M"]["static"]) / d["M"]["static"] for d in ds if k in d["M"]]) if any(k in d["M"] for d in ds) else np.array([])
            data.append(v); labels.append(lab); cols.append(PCOL[k])
        pos = np.arange(len(data))
        bp = ax.boxplot([v if len(v) else [np.nan] for v in data], positions=pos, widths=0.55, showfliers=False, patch_artist=True,
                        medianprops=dict(color=INK, lw=1.2), whiskerprops=dict(color=INK2, lw=0.8), capprops=dict(color=INK2, lw=0.8))
        for patch, c in zip(bp["boxes"], cols):
            patch.set(facecolor=c, edgecolor=INK, linewidth=0.6, alpha=0.85)
        ax.axhline(0, color=INK, lw=0.6); ax.set_title(title, fontsize=8.5); ax.set_xticks([])
        ax.grid(True, axis="y", alpha=0.6)
    axs[0].set_ylabel("makespan vs static plan (%)"); axs[0].set_ylim(-45, 45)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=PCOL[k], edgecolor=INK, lw=0.6) for k, _ in POLS[1:]]
    fig.legend(handles, [lab for _, lab in POLS[1:]], loc="lower center", ncol=5, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(os.path.join(out, "fig_paired.pdf")); plt.close(fig)


def cvar(x, q=0.05):
    m = max(1, int(np.ceil(q * len(x)))); return float(np.sort(x)[-m:].mean())


def fig_cvar(stoch, out):
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for k, lab in POLS:
        xs = [d["M"][k].mean() / d["cert"] for d in stoch if k in d["M"]]
        ys = [cvar(d["M"][k]) / d["cert"] for d in stoch if k in d["M"]]
        mk = ["P" if d["plant"] else "o" for d in stoch if k in d["M"]]
        for x, y, m in zip(xs, ys, mk):
            ax.scatter(x, y, s=24 if m == "o" else 40, marker=m, color=PCOL[k], edgecolor=INK if m == "P" else "none", linewidth=0.5, zorder=3)
        ax.scatter([], [], s=24, color=PCOL[k], label=lab)
    lim = [1.0, 2.6]; ax.plot(lim, lim, color=INK2, lw=0.6, ls="--"); ax.text(1.55, 1.49, "CVaR = mean", fontsize=7, color=INK2, rotation=20)
    ax.set_xlabel("mean makespan / PI certificate"); ax.set_ylabel("CVaR$_{0.95}$ / PI certificate")
    ax.set_xlim(1.0, 1.75); ax.set_ylim(1.0, 3.6); ax.grid(True, alpha=0.6)
    ax.scatter([], [], s=40, marker="P", color="white", edgecolor=INK, label="plant windows")
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_cvar.pdf"), bbox_inches="tight"); plt.close(fig)


def fig_plant(stoch, out):
    pl = [d for d in stoch if d["plant"]]
    if not pl: return
    fig, axs = plt.subplots(1, len(pl), figsize=(3.4 * len(pl), 3.0), sharey=False)
    axs = np.atleast_1d(axs)
    for ax, d in zip(axs, pl):
        keys = [k for k, _ in POLS if k in d["M"]]
        means = [d["M"][k].mean() / d["pi_mean"] for k in keys]
        err = [1.96 * d["M"][k].std(ddof=1) / np.sqrt(len(d["M"][k])) / d["pi_mean"] for k in keys]
        pos = np.arange(len(keys))
        ax.bar(pos, means, width=0.62, color=[PCOL[k] for k in keys], edgecolor="white", linewidth=1.0, zorder=3)
        ax.errorbar(pos, means, yerr=err, fmt="none", ecolor=INK, elinewidth=0.8, capsize=2, zorder=4)
        ax.axhspan(d["cert"] / d["pi_mean"], 1.0, color=C[6], alpha=0.18, lw=0, zorder=1)
        ax.axhline(1.0, color=INK, lw=0.7, zorder=2); ax.text(-0.45, 1.006, "PI mean (band: 95% LCB)", ha="left", va="bottom", fontsize=6.5, color=INK2)
        for p_, m in zip(pos, means):
            ax.text(p_, m + err[list(pos).index(p_)] + 0.008, f"{100*(m-1):+.0f}%", ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_xticks(pos); ax.set_xticklabels([dict(POLS)[k] for k in keys], rotation=35, ha="right", fontsize=7)
        r = d["row"]; ax.set_title(f"{r['instance'].split('_a')[0]}: {r['ops']} ops, {r['machines']} machines", fontsize=8.5)
        ax.set_ylim(0.95, max(means) + 0.16); ax.grid(True, axis="y", alpha=0.6)
    axs[0].set_ylabel("expected makespan / PI mean")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_plant.pdf")); plt.close(fig)


def fig_ecdf(stoch, out):
    """ECDF of the per-scenario ratio makespan / perfect-information value, policies on the plant windows."""
    pl = [d for d in stoch if d["plant"]]
    if not pl: return
    fig, axs = plt.subplots(1, len(pl), figsize=(3.5 * len(pl), 3.0), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, d in zip(axs, pl):
        phi = np.array(d.get("phi_cp") or d["phi_ht"], dtype=float)
        for k, lab in POLS:
            if k not in d["M"]: continue
            r = np.sort(d["M"][k] / phi); y = np.arange(1, len(r) + 1) / len(r)
            ax.step(np.concatenate([[r[0]], r]), np.concatenate([[0], y]), where="post", color=PCOL[k], lw=1.6 if k in ("replan-opt", "reactive-MWR", "static") else 1.1, label=lab, zorder=3)
        ax.axvline(1.0, color=INK, lw=0.6, zorder=2)
        ax.text(1.004, 0.03, "clairvoyant", fontsize=6.5, color=INK2, rotation=90, va="bottom")
        ax.axhline(0.95, color=INK2, lw=0.5, ls=":"); ax.text(1.78, 0.905, "95th percentile", fontsize=6.5, color=INK2, ha="right")
        r = d["row"]; ax.set_title(f"{r['instance'].split('_a')[0]}: {r['ops']} ops, {r['scenarios']} scenarios", fontsize=8.5)
        ax.set_xlim(0.98, 1.8); ax.set_ylim(0, 1.0); ax.grid(True, alpha=0.6)
        ax.set_xlabel("makespan / perfect-information value of the scenario")
    axs[0].set_ylabel("fraction of scenarios")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.8, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(os.path.join(out, "fig_ecdf.pdf")); plt.close(fig)


def load_det_full(prefix):
    rows = []
    for f in sorted(glob.glob(prefix + "_*.csv")):
        g = re.sub(r"\d+$", "", os.path.basename(f)[len(os.path.basename(prefix)) + 1:-4])
        for r in csv.DictReader(open(f)):
            r = dict(r); r["group"] = g; rows.append(r)
    return rows


def fig_hierarchy(prefix, out):
    """Every bound of the hierarchy on every instance, normalized by the best known solution."""
    rows = load_det_full(prefix)
    order = {"bench": 0, "balanced": 1, "coupled": 2, "square": 3, "industrial": 4}
    rows.sort(key=lambda r: (order[r["group"]], int(float(r["ops"])), r["instance"]))
    # (column, label, marker, color, size, filled, zorder)
    layers = [("lb_ht_energetic", "energetic", "v", C[4], 24, True, 3), ("lb_sg", "Lagrangian (supergradient)", "x", C[3], 26, True, 3),
              ("lb_cg_lagr", "Lagrangian (column generation)", "+", C[1], 30, True, 3), ("lb_ht", "head--tail", "o", C[0], 30, True, 4),
              ("lb_cpsat", "CP-SAT plant model", "s", INK2, 46, False, 5), ("lb_destructive", "destructive (shaving)", "D", C[2], 30, False, 6)]
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    for i, r in enumerate(rows):
        ub = fnum(r, "best_ub")
        vals = {k: fnum(r, k) / ub for k, *_ in layers}
        if r.get("cpsat_status") == "UNKNOWN" or vals["lb_cpsat"] <= 0: vals["lb_cpsat"] = float("nan")
        ok = [v for v in vals.values() if v == v and v > 0]
        ax.plot([i, i], [min(ok), 1.0], color=GRID, lw=1.2, zorder=1)
        ax.plot([i - 0.35, i + 0.35], [1.0, 1.0], color=INK, lw=1.0, zorder=7)
        for k, lab, mk, col, sz, filled, z in layers:
            v = vals[k]
            if v != v or v <= 0: continue
            if filled: ax.scatter([i], [v], marker=mk, s=sz, color=col, linewidth=1.1, zorder=z, label=lab if i == 0 else None)
            else: ax.scatter([i], [v], marker=mk, s=sz, facecolor="none", edgecolor=col, linewidth=1.0, zorder=z, label=lab if i == 0 else None)
    bounds = []; last = None
    for i, r in enumerate(rows):
        if r["group"] != last: bounds.append(i); last = r["group"]
    for b in bounds[1:]: ax.axvline(b - 0.5, color=INK2, lw=0.5, ls=":")
    names = {"bench": "benchmark variants", "balanced": "balanced", "coupled": "coupling-dominated", "square": "square", "industrial": "plant windows"}
    ends = bounds[1:] + [len(rows)]
    for b, e in zip(bounds, ends):
        ax.text((b + e - 1) / 2, 1.045, names[rows[b]["group"]], ha="center", fontsize=7, color=INK2)
    ax.set_xticks(np.arange(len(rows))); ax.set_xticklabels([f"{int(float(r['ops']))}" for r in rows], rotation=90, fontsize=5.8)
    ax.set_xlabel("instances of the deterministic campaign (operations), grouped by structure")
    ax.set_ylabel("bound / best known solution"); ax.set_ylim(0.0, 1.08); ax.set_xlim(-0.8, len(rows) - 0.2); ax.grid(True, axis="y", alpha=0.6)
    ax.plot([], [], color=INK, lw=1.0, label="best known solution")
    ax.legend(fontsize=6.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.3))
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_hierarchy.pdf"), bbox_inches="tight"); plt.close(fig)


def parse_log(path):
    """Per-instance anytime traces from the campaign log (UTF-8 or UTF-16 as written by PowerShell)."""
    raw = open(path, "rb").read()
    txt = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", "replace")
    runs = {}; cur = None; t_ht = None
    for line in txt.splitlines():
        line = line.rstrip("\r")
        m = re.search(r"=== (\S+): jobs (\d+) machines (\d+) ops (\d+)", line)
        if m:
            cur = runs.setdefault(m.group(1), dict(ops=int(m.group(4)), cg=[], ub=[], destr=[], ht=None, t_ht=None, sg=[], cp=None, t_cp=None, lb_final=None)); continue
        if cur is None: continue
        m = re.search(r"HT energetic (\d+), HT best (\d+) \(([\d.]+)s\)", line)
        if m: cur["ht"] = float(m.group(2)); cur["t_ht"] = float(m.group(3)); continue
        m = re.search(r"it\s+(\d+)\s+t=\s*([\d.]+)s\s+z_RMP=([\d.]+)\s+Phi=(-?[\d.]+)\s+LB=([\d.]+)\s+UB=(\d+)", line)
        if m: cur["cg"].append(tuple(float(m.group(i)) for i in range(2, 7))); continue
        m = re.search(r"destructive: T=(\d+) -> (open|infeasible).*\[([\d.]+)s\]", line)
        if m: cur["destr"].append((float(m.group(3)), float(m.group(1)), m.group(2))); continue
        m = re.search(r"destructive bound \(shaving\): (\d+) \[(\d+) tests, ([\d.]+)s\]", line)
        if m: cur["lb_final"] = float(m.group(1)); cur["t_destr"] = float(m.group(3)); continue
        m = re.search(r"sg it\s+(\d+)\s+t=\s*([\d.]+)s\s+Phi=(-?[\d.]+)\s+best=(-?[\d.]+)", line)
        if m: cur["sg"].append((float(m.group(2)), float(m.group(4)))); continue
        m = re.search(r"CP-SAT plant: UB (\S+) LB (\S+) \[(\w+), ([\d.]+)s\]", line)
        if m: cur["cp"] = (None if m.group(1) == "None" else float(m.group(1)), float(m.group(2)), m.group(3)); cur["t_cp"] = float(m.group(4)); continue
    return runs


def fig_convergence(logpath, out, names=("mk10_c4", "gen_c8_n200_s3", "cpl_c20_n60_s1", "cpl_c40_n40_s1")):
    runs = parse_log(logpath)
    names = [n for n in names if n in runs]
    if not names: return
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.0))
    for ax, n in zip(axs.ravel(), names):
        R = runs[n]; cg = np.array(R["cg"]); t_cg_end = cg[-1, 0] if len(cg) else R["t_ht"]
        ub_best = min([cg[:, 4].min() if len(cg) else np.inf] + ([R["cp"][0]] if R["cp"] and R["cp"][0] else []))
        f = lambda v: np.asarray(v) / ub_best
        # certified lower bound: head--tail from t_ht, column generation LB after, destructive at the end
        t_lb = [R["t_ht"]] + list(cg[:, 0]); v_lb = [R["ht"]] + list(cg[:, 3])
        if R["lb_final"] is not None and R["destr"]:
            t_lb.append(t_cg_end + R["destr"][-1][0]); v_lb.append(R["lb_final"])
        ax.step(t_lb, f(v_lb), where="post", color=C[2], lw=2.0, label="certified lower bound", zorder=4)
        ax.plot(cg[:, 0], f(cg[:, 1]), color=C[3], lw=1.0, label="restricted master value $z_{RMP}$", zorder=3)
        phi_best = np.maximum.accumulate(cg[:, 2])
        ax.plot(cg[:, 0], f(phi_best), color=C[1], lw=1.3, ls="--", label="Lagrangian $\\Phi$ (best so far, CG)", zorder=3)
        if R["sg"]:
            sg = np.array(R["sg"]); ax.plot(sg[:, 0], f(sg[:, 1]), color=C[4], lw=1.4, ls=":", label="Lagrangian $\\Phi$ (supergradient)", zorder=3)
        ax.step(cg[:, 0], f(cg[:, 4]), where="post", color=C[0], lw=1.6, label="primal solution (CG + LNS)", zorder=4)
        for t, T, res in R["destr"]:
            ax.scatter([t_cg_end + t], f([T]), marker="|" , s=40, color=C[2] if res == "infeasible" else INK2, lw=1.0, zorder=5)
        if R["cp"]:
            ub, lb, st = R["cp"]
            if ub: ax.scatter([R["t_cp"]], f([ub]), marker="s", s=28, facecolor="white", edgecolor=INK, zorder=6, label="CP-SAT plant model (UB, LB)")
            if lb > 0: ax.scatter([R["t_cp"]], f([lb]), marker="s", s=28, facecolor="white", edgecolor=INK, zorder=6)
            if ub and lb > 0: ax.plot([R["t_cp"]] * 2, f([lb, ub]), color=INK, lw=0.6, zorder=5)
        ax.set_xscale("log"); ax.grid(True, alpha=0.6)
        ax.annotate("head--tail", (R["t_ht"], R["ht"] / ub_best), textcoords="offset points", xytext=(2, -9), fontsize=6.5, color=INK2)
        tmax = max([t_cg_end + (R["destr"][-1][0] if R["destr"] else 0), R["t_cp"] or 0]) * 1.4
        ax.set_xlim(max(0.5, R["t_ht"] * 0.6), tmax)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")); ax.xaxis.set_minor_formatter(NullFormatter())
        lo = min(0.55, float(np.nanmin(f(phi_best[phi_best > 0]))) if (phi_best > 0).any() else 0.55)
        ax.set_ylim(max(0.0, lo - 0.05), 1.22)
        gap = 100 * (ub_best - (R["lb_final"] or cg[-1, 3])) / (R["lb_final"] or cg[-1, 3])
        ax.set_title(f"{n}: {R['ops']} ops, certified gap {gap:.1f}%", fontsize=8.5)
    for ax in axs[1]: ax.set_xlabel("time (s, log)")
    for ax in axs[:, 0]: ax.set_ylabel("value / best known solution")
    h, l = axs[0, 0].get_legend_handles_labels()
    if R["sg"] == [] or len(l) < 7:
        for ax in axs.ravel():
            h2, l2 = ax.get_legend_handles_labels()
            for hh, ll in zip(h2, l2):
                if ll not in l: h.append(hh); l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=6.8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(os.path.join(out, "fig_convergence.pdf")); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True); ap.add_argument("--stoch", nargs="+", required=True); ap.add_argument("--out", default="figures"); ap.add_argument("--log", default=None, help="campaign log for the convergence figure")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    det = load_det(a.det); stoch = load_stoch(a.stoch)
    fig_regime(det, a.out); fig_detgap(det, a.out); fig_policygap(stoch, a.out); fig_paired(stoch, a.out); fig_cvar(stoch, a.out); fig_plant(stoch, a.out)
    fig_ecdf(stoch, a.out); fig_hierarchy(a.det, a.out)
    if a.log: fig_convergence(a.log, a.out)
    print("figures written to", a.out)


if __name__ == "__main__":
    main()
