"""All FloodWing figures. Each function draws, saves a 200-dpi PNG to the
figure directory, and returns the matplotlib Figure (so notebooks display it).
"""
from __future__ import annotations

import textwrap
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import os

import matplotlib

if "inline" not in matplotlib.get_backend().lower() and not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")   # headless scripts / CI; notebooks keep their inline backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

from .qubo import PlacementProblem, all_energies, bits_from_key, valid_mask
from .utils import cell_coords, cell_name

# Okabe-Ito colour-blind-safe palette.
OKABE_ITO = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "vermillion": "#D55E00",
             "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "black": "#000000", "grey": "#8C8C8C"}
METHOD_COLORS = {"Exact": OKABE_ITO["black"], "QAOA": OKABE_ITO["blue"], "QAOA-shots": OKABE_ITO["sky"],
                 "QAOA-best-sample": OKABE_ITO["purple"], "Greedy": OKABE_ITO["orange"],
                 "SA": OKABE_ITO["green"], "Random": OKABE_ITO["grey"]}
HEAT_CMAP = "cividis"
DPI = 200


@contextmanager
def style() -> Iterator[None]:
    """Consistent, non-default look for every figure."""
    rc = {
        "figure.facecolor": "white", "axes.facecolor": "#FAFAF7", "axes.edgecolor": "#3A3A3A",
        "axes.grid": True, "grid.color": "#E3E3DC", "grid.linewidth": 0.7, "axes.spines.top": False,
        "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 11,
        "axes.labelsize": 10, "font.family": "DejaVu Sans", "font.size": 9.5, "legend.frameon": False,
        "axes.prop_cycle": matplotlib.cycler(color=list(OKABE_ITO.values())),
        "savefig.bbox": "tight", "savefig.dpi": DPI,
    }
    with plt.rc_context(rc):
        yield


def _save(fig: plt.Figure, fig_dir: Path, name: str) -> plt.Figure:
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{name}.png", dpi=DPI, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Core drawing helper
# ---------------------------------------------------------------------------
def draw_placement(ax: plt.Axes, problem: PlacementProblem, x: np.ndarray | None, title: str = "",
                   w: np.ndarray | None = None, vmax: float | None = None, show_names: bool = True,
                   colorbar: bool = False) -> Any:
    """Heatmap of w with drones, footprints, radio links, base station and blocked cells."""
    w = problem.w if w is None else w
    R, C = problem.rows, problem.cols
    im = ax.imshow(w.reshape(R, C), cmap=HEAT_CMAP, vmin=0, vmax=vmax or max(1.0, float(np.max(w))),
                   extent=(-0.5, C - 0.5, R - 0.5, -0.5), zorder=0)
    ax.grid(False)
    for i in problem.blocked:
        r, c = divmod(i, C)
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, hatch="xxx", edgecolor=OKABE_ITO["vermillion"],
                               linewidth=1.2, zorder=2))
    if show_names:
        for i in range(R * C):
            r, c = divmod(i, C)
            ax.text(c - 0.42, r - 0.3, cell_name(i, C), fontsize=7, color="white", alpha=0.85, zorder=3)
    by, bx = problem.base_pos
    ax.scatter([bx], [by], marker="s", s=110, color=OKABE_ITO["vermillion"], edgecolor="black", zorder=6,
               label="base station")
    if x is not None:
        sel = np.flatnonzero(x)
        xy = cell_coords(R, C)
        pts = np.vstack([xy[sel], np.array([[by, bx]])])
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                if np.linalg.norm(pts[a] - pts[b]) <= problem.radio_range + 1e-9:
                    ax.plot([pts[a, 1], pts[b, 1]], [pts[a, 0], pts[b, 0]], color=OKABE_ITO["sky"], lw=1.1,
                            alpha=0.8, zorder=4)
        for i in sel:
            r, c = divmod(int(i), C)
            ax.add_patch(Circle((c, r), 0.75, fill=False, ls="--", lw=1.0, edgecolor=OKABE_ITO["orange"], zorder=4))
            ax.scatter([c], [r], marker="^", s=120, color=OKABE_ITO["orange"], edgecolor="black", zorder=7)
    ax.set_xlim(-0.9, C - 0.1)
    ax.set_ylim(R - 0.1, -1.6)
    ax.set_xticks(range(C), [str(c + 1) for c in range(C)])
    ax.set_yticks(range(R), [chr(65 + r) for r in range(R)])
    ax.set_title(title)
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="survivor likelihood w")
    return im


# ---------------------------------------------------------------------------
# Figures 1-9: core
# ---------------------------------------------------------------------------
def fig_data_overview(ds, idx: int, fig_dir: Path) -> plt.Figure:
    """Fig 1: RGB composite, water mask, NDVI, built-up proxy, edge proximity."""
    with style():
        fig, axes = plt.subplots(1, 5, figsize=(16, 3.6))
        panels = [(ds.rgb[idx], "RGB composite (B4-B3-B2)", None), (ds.X[idx, 0], "Water fraction (label)", "Blues"),
                  (ds.X[idx, 1], "NDVI (scaled 0-1)", "Greens"), (ds.X[idx, 2], "Built-up proxy (NDBI)", "Oranges"),
                  (ds.X[idx, 3], "Flood-edge proximity", "magma")]
        for ax, (img, title, cmap) in zip(axes, panels):
            ax.imshow(img, cmap=cmap) if cmap else ax.imshow(img)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        fig.suptitle(f"Data overview: chip {ds.chip_ids[idx]} (source: {ds.sources[idx]})", fontweight="bold")
        return _save(fig, fig_dir, "01_data_overview")


def fig_heatmap_prediction(ds, train, fig_dir: Path, n_show: int = 3) -> plt.Figure:
    """Fig 2: proxy truth vs CNN prediction on the grid for validation chips + scatter."""
    va = train.val_idx[:n_show]
    with style():
        fig = plt.figure(figsize=(13, 3.2 * len(va)))
        gs = fig.add_gridspec(len(va), 3, width_ratios=[1, 1, 1.2])
        for r, idx in enumerate(va):
            for c, (data, lab) in enumerate([(ds.Y[idx], "Proxy truth"), (train.predictions[idx], "CNN prediction")]):
                ax = fig.add_subplot(gs[r, c])
                im = ax.imshow(data.reshape(ds.rows, ds.cols), cmap=HEAT_CMAP, vmin=0, vmax=1)
                ax.grid(False)
                for i in range(ds.rows * ds.cols):
                    rr, cc = divmod(i, ds.cols)
                    ax.text(cc, rr, f"{data[i]:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if data[i] < 0.6 else "black")
                ax.set_xticks(range(ds.cols), [str(i + 1) for i in range(ds.cols)])
                ax.set_yticks(range(ds.rows), [chr(65 + i) for i in range(ds.rows)])
                corr = np.corrcoef(ds.Y[idx], train.predictions[idx])[0, 1]
                ax.set_title(f"{lab} | {ds.chip_ids[idx]}" + (f"\nr = {corr:.2f}" if c == 1 else ""))
        ax = fig.add_subplot(gs[:, 2])
        yt, yp = ds.Y[train.val_idx].ravel(), train.predictions[train.val_idx].ravel()
        ax.scatter(yt, yp, s=14, alpha=0.6, color=OKABE_ITO["blue"], edgecolor="none")
        ax.plot([0, 1], [0, 1], color=OKABE_ITO["grey"], ls="--", lw=1)
        m = train.metrics
        ax.set_title(f"All validation cells\nMAE {m['mae']:.3f} | Pearson {m['pearson_r']:.2f} | "
                     f"Spearman {m['spearman_rho']:.2f}")
        ax.set_xlabel("proxy truth")
        ax.set_ylabel("prediction")
        fig.colorbar(im, ax=fig.axes[:-1], fraction=0.02, pad=0.02, label="likelihood")
        return _save(fig, fig_dir, "02_heatmap_prediction")


def fig_training_curves(history: Mapping[str, Sequence[float]], fig_dir: Path) -> plt.Figure:
    """Fig 3: train/val loss and validation Spearman per epoch."""
    with style():
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
        ep = np.arange(1, len(history["train_loss"]) + 1)
        a1.plot(ep, history["train_loss"], marker="o", label="train", color=OKABE_ITO["blue"])
        a1.plot(ep, history["val_loss"], marker="s", label="validation", color=OKABE_ITO["orange"])
        a1.set_xlabel("epoch")
        a1.set_ylabel("MSE loss")
        a1.set_title("CNN loss")
        a1.legend()
        a2.plot(ep, history["val_spearman"], marker="o", color=OKABE_ITO["green"])
        a2.set_xlabel("epoch")
        a2.set_ylabel("Spearman rho")
        a2.set_title("Validation rank correlation")
        a2.set_ylim(-0.05, 1.0)
        return _save(fig, fig_dir, "03_cnn_training_curves")


def fig_placement_map(problem: PlacementProblem, x: np.ndarray, fig_dir: Path, title: str,
                      name: str = "04_placement_map") -> plt.Figure:
    """Fig 4: single placement map with full legend."""
    with style():
        fig, ax = plt.subplots(figsize=(6.6, 5.4))
        draw_placement(ax, problem, x, title, colorbar=True)
        handles = [plt.Line2D([], [], marker="^", ls="", color=OKABE_ITO["orange"], markeredgecolor="black",
                              markersize=10, label="drone"),
                   plt.Line2D([], [], ls="--", color=OKABE_ITO["orange"], label="footprint"),
                   plt.Line2D([], [], color=OKABE_ITO["sky"], label=f"radio link (<= {problem.radio_range} cells)"),
                   plt.Line2D([], [], marker="s", ls="", color=OKABE_ITO["vermillion"], markeredgecolor="black",
                              markersize=9, label="base station"),
                   Rectangle((0, 0), 1, 1, fill=False, hatch="xxx", edgecolor=OKABE_ITO["vermillion"],
                             label="blocked cell")]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=8)
        return _save(fig, fig_dir, name)


def fig_method_comparison(problem: PlacementProblem, results: Mapping[str, Any], fig_dir: Path,
                          name: str = "05_method_comparison") -> plt.Figure:
    """Fig 5: same instance, placements from several methods side by side."""
    with style():
        fig, axes = plt.subplots(1, len(results), figsize=(5.0 * len(results), 4.6))
        axes = np.atleast_1d(axes)
        opt = results["Exact"].objective if "Exact" in results else None
        for ax, (m, r) in zip(axes, results.items()):
            g = "" if opt is None else f" | gap {(r.objective - opt) / abs(opt):.1%}"
            draw_placement(ax, problem, r.x, f"{m}\nobjective {r.objective:.3f}{g}")
        fig.suptitle(f"Method comparison on {problem.name}", fontweight="bold")
        return _save(fig, fig_dir, name)


def _pad(histories: Sequence[Sequence[float]]) -> np.ndarray:
    L = max(len(h) for h in histories)
    return np.array([np.concatenate([h, np.full(L - len(h), h[-1])]) for h in histories if len(h)])


def fig_qaoa_convergence(histories: Sequence[Sequence[float]], optima: Sequence[float], fig_dir: Path) -> plt.Figure:
    """Fig 6: expected energy minus exact optimum versus optimiser iteration (mean +- std band)."""
    rel = [np.asarray(h) - o for h, o in zip(histories, optima) if len(h)]
    H = _pad(rel)
    with style():
        fig, ax = plt.subplots(figsize=(7.5, 4))
        it = np.arange(1, H.shape[1] + 1)
        for h in H:
            ax.plot(it, h, color=OKABE_ITO["sky"], alpha=0.25, lw=0.8)
        mu, sd = H.mean(axis=0), H.std(axis=0)
        ax.plot(it, mu, color=OKABE_ITO["blue"], lw=2, label=f"mean over {len(H)} runs")
        ax.fill_between(it, mu - sd, mu + sd, color=OKABE_ITO["blue"], alpha=0.18, label="+-1 std")
        ax.axhline(0, color="black", ls=":", lw=1, label="exact optimum")
        ax.set_xlabel("objective evaluation")
        ax.set_ylabel("<QUBO> - f*  (shot estimate)")
        ax.set_title("QAOA convergence")
        ax.legend()
        return _save(fig, fig_dir, "06_qaoa_convergence")


def fig_bitstring_probs(problem: PlacementProblem, p: np.ndarray, fig_dir: Path, top: int = 20,
                        name: str = "07_bitstring_probabilities") -> plt.Figure:
    """Fig 7: top bitstrings by probability, coloured optimal / valid / invalid."""
    E = all_energies(problem)
    vm = valid_mask(problem)
    opt = E[vm].min()
    order = np.argsort(-p)[:top]
    colors, labels = [], []
    for k in order:
        if vm[k] and np.isclose(E[k], opt):
            colors.append(OKABE_ITO["green"])
        elif vm[k]:
            colors.append(OKABE_ITO["blue"])
        else:
            colors.append(OKABE_ITO["vermillion"])
        labels.append("-".join(cell_name(i, problem.cols) for i in np.flatnonzero(bits_from_key(int(k), problem.n))) or "none")
    with style():
        fig, ax = plt.subplots(figsize=(11, 4.4))
        ax.bar(range(len(order)), p[order], color=colors, edgecolor="black", linewidth=0.4)
        ax.set_xticks(range(len(order)), labels, rotation=60, ha="right", fontsize=7.5)
        ax.set_ylabel("probability (exact)")
        ax.set_title(f"Top-{top} QAOA bitstrings | P(valid) = {p[vm].sum():.3f}, "
                     f"P(optimal) = {p[vm & np.isclose(E, opt)].sum():.4f}, uniform P(optimal) = "
                     f"{(vm & np.isclose(E, opt)).sum() / len(p):.4f}")
        ax.legend(handles=[Rectangle((0, 0), 1, 1, color=OKABE_ITO["green"], label="optimal"),
                           Rectangle((0, 0), 1, 1, color=OKABE_ITO["blue"], label="valid"),
                           Rectangle((0, 0), 1, 1, color=OKABE_ITO["vermillion"], label="invalid (sum x != k)")])
        return _save(fig, fig_dir, name)


def fig_gap_distribution(df: pd.DataFrame, fig_dir: Path) -> plt.Figure:
    """Fig 8: optimality gap per method across instances."""
    methods = [m for m in METHOD_COLORS if m in set(df["method"].astype(str)) and m != "Exact"]
    data = [df[df.method.astype(str) == m]["gap"].to_numpy() * 100 for m in methods]
    with style():
        fig, ax = plt.subplots(figsize=(9, 4.2))
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
        for patch, m in zip(bp["boxes"], methods):
            patch.set_facecolor(METHOD_COLORS[m])
            patch.set_alpha(0.55)
        rng = np.random.default_rng(0)
        for i, (d, m) in enumerate(zip(data, methods), start=1):
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=16, color=METHOD_COLORS[m], edgecolor="black",
                       linewidth=0.3, zorder=3)
        ax.set_xticks(range(1, len(methods) + 1), methods)
        ax.set_ylabel("optimality gap (%)  [lower is better]")
        ax.set_title(f"Optimality gap across {df['instance'].nunique()} instances (Exact = 0%)")
        return _save(fig, fig_dir, "08_optimality_gap")


def fig_agent_before_after(steps: Sequence[Mapping[str, Any]], fig_dir: Path) -> plt.Figure:
    """Fig 9: heatmap + QAOA placement before (top) and after (bottom) each report."""
    n = len(steps)
    vmax = max(max(float(np.max(s["problem_before"].w)), float(np.max(s["problem_after"].w))) for s in steps)
    with style():
        fig, axes = plt.subplots(2, n, figsize=(3.9 * n, 7.6))
        axes = np.atleast_2d(axes).reshape(2, n)
        for j, s in enumerate(steps):
            draw_placement(axes[0, j], s["problem_before"], s["x_before"], "before", vmax=vmax, show_names=False)
            ups = ", ".join(f"{u['cell']}:{u['action']}" for u in s["updates"]) or "no valid update"
            draw_placement(axes[1, j], s["problem_after"], s["x_after"], f"after: {ups}", vmax=vmax, show_names=False)
            axes[0, j].set_title("\n".join(textwrap.wrap(f"Report {j + 1}: {s['report']}", 34)) + "\n\nbefore",
                                 fontsize=8.5)
        fig.suptitle("Agent loop: field report -> heatmap update -> QAOA re-solve", fontweight="bold", y=1.03)
        return _save(fig, fig_dir, "09_agent_before_after")


# ---------------------------------------------------------------------------
# Figures 10-17: bonus
# ---------------------------------------------------------------------------
def fig_depth_study(df: pd.DataFrame, fig_dir: Path) -> plt.Figure:
    g = df.groupby("p")
    with style():
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        for ax, col, lab, color in zip(axes, ["gap", "p_optimal", "evals"],
                                       ["optimality gap (%)", "P(optimal)", "objective evaluations"],
                                       [OKABE_ITO["blue"], OKABE_ITO["green"], OKABE_ITO["orange"]]):
            scale = 100 if col == "gap" else 1
            ax.errorbar(g[col].mean().index, g[col].mean() * scale, yerr=g[col].std(ddof=0) * scale, marker="o",
                        capsize=4, color=color, lw=2)
            ax.set_xlabel("QAOA depth p")
            ax.set_ylabel(lab)
            ax.set_xticks(sorted(df["p"].unique()))
        axes[0].set_title("Gap vs depth")
        axes[1].set_title("P(optimal) vs depth")
        axes[2].set_title("Cost vs depth")
        fig.suptitle(f"QAOA depth study ({df['instance'].nunique()} instances, mean +- std)", fontweight="bold")
        return _save(fig, fig_dir, "10_depth_study")


def fig_penalty_sensitivity(df: pd.DataFrame, fig_dir: Path) -> plt.Figure:
    g = df.groupby("multiplier")
    with style():
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.9))
        m = g["p_valid"].mean()
        a1.errorbar(m.index, m, yerr=g["p_valid"].std(ddof=0), marker="o", capsize=4, color=OKABE_ITO["blue"],
                    lw=2, label="P(valid) of QAOA state")
        a1.plot(m.index, g["ground_state_valid"].mean(), marker="s", ls="--", color=OKABE_ITO["vermillion"],
                label="QUBO ground state valid (fraction)")
        a1.set_xscale("log", base=2)
        a1.set_xlabel("P / P_default")
        a1.set_ylabel("rate")
        a1.set_ylim(-0.05, 1.05)
        a1.set_title("Validity vs penalty weight")
        a1.legend(fontsize=8)
        gm = g["gap"].mean() * 100
        a2.errorbar(gm.index, gm, yerr=g["gap"].std(ddof=0) * 100, marker="o", capsize=4,
                    color=OKABE_ITO["orange"], lw=2)
        a2.set_xscale("log", base=2)
        a2.set_xlabel("P / P_default")
        a2.set_ylabel("optimality gap (%)")
        a2.set_title("Decoded gap vs penalty weight")
        return _save(fig, fig_dir, "11_penalty_sensitivity")


def fig_warm_start(ws: Mapping[str, Any], fig_dir: Path) -> plt.Figure:
    """Fig 12: convergence after an agent update, random init vs warm start."""
    with style():
        fig, ax = plt.subplots(figsize=(8, 4))
        for i, c in enumerate(ws["cold"]):
            h = np.minimum.accumulate(c.extras["history"])
            ax.plot(np.arange(1, len(h) + 1), h, color=OKABE_ITO["grey"], lw=1.4, alpha=0.8,
                    label="random init (running min)" if i == 0 else None)
        h = np.minimum.accumulate(ws["warm"].extras["history"])
        ax.plot(np.arange(1, len(h) + 1), h, color=OKABE_ITO["blue"], lw=2.4, label="warm start (running min)")
        w, cs = ws["warm_summary"], pd.DataFrame(ws["cold_summary"]).mean(numeric_only=True)
        ax.set_xlabel("objective evaluation")
        ax.set_ylabel("<QUBO> (shot estimate)")
        ax.set_title(f"Re-solve after report: warm {w['evals_to_converge']} evals to converge "
                     f"vs random {cs['evals_to_converge']:.1f} (tol {ws['tolerance']})")
        ax.legend()
        return _save(fig, fig_dir, "12_warm_start")


def fig_optimizer_comparison(df: pd.DataFrame, fig_dir: Path) -> plt.Figure:
    g = df.groupby("optimizer")
    cols = [("gap", "optimality gap (%)", 100), ("evals", "objective evaluations", 1), ("runtime_s", "wall time (s)", 1)]
    colors = [OKABE_ITO["blue"], OKABE_ITO["orange"], OKABE_ITO["green"]]
    with style():
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        for ax, (col, lab, sc) in zip(axes, cols):
            mu, sd = g[col].mean() * sc, g[col].std(ddof=0) * sc
            ax.bar(mu.index, mu, yerr=sd, capsize=4, color=colors[: len(mu)], edgecolor="black", linewidth=0.5)
            ax.set_ylabel(lab)
            ax.set_title(lab)
        fig.suptitle("Classical optimiser comparison (same instances, same initial point)", fontweight="bold")
        return _save(fig, fig_dir, "13_optimizer_comparison")


def fig_noise(df: pd.DataFrame, fig_dir: Path) -> plt.Figure:
    with style():
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
        a1.plot(df["level"], df["p_optimal"], marker="o", color=OKABE_ITO["green"], lw=2, label="P(optimal)")
        a1.axhline(df.attrs.get("uniform_p_optimal", np.nan), color=OKABE_ITO["grey"], ls=":",
                   label="uniform random guess")
        a1.set_xlabel("noise level (x base: 0.1% 1q, 1% 2q, 1% readout)")
        a1.set_ylabel("P(optimal)")
        a1.set_title("P(optimal) under noise (shot estimate)")
        a1.legend(fontsize=8)
        a2.plot(df["level"], df["p_valid"], marker="s", color=OKABE_ITO["blue"], lw=2)
        a2.set_xlabel("noise level")
        a2.set_ylabel("P(valid)")
        a2.set_title(f"P(valid) under noise ({int(df['cx_count'].iloc[0])} CX gates)")
        return _save(fig, fig_dir, "14_noise_robustness")


def fig_landscape(L: Mapping[str, Any], fig_dir: Path) -> plt.Figure:
    with style():
        fig, ax = plt.subplots(figsize=(7, 5.4))
        cs = ax.contourf(L["gammas"], L["betas"], L["energy"], levels=30, cmap="viridis")
        fig.colorbar(cs, ax=ax, label="<QUBO>")
        path = L["path"]
        ax.plot(path[:, 0], path[:, 1], color="white", lw=1.4, marker=".", ms=4, label="COBYLA path (Qiskit QAOA)")
        ax.scatter(path[0, 0], path[0, 1], color=OKABE_ITO["orange"], s=60, zorder=5, label="start", edgecolor="black")
        ax.scatter(path[-1, 0], path[-1, 1], color=OKABE_ITO["vermillion"], s=80, marker="*", zorder=5,
                   label="end", edgecolor="black")
        ax.set_xlabel("gamma")
        ax.set_ylabel("beta")
        ax.set_title("p = 1 energy landscape")
        ax.grid(False)
        ax.legend(loc="upper right", fontsize=8)
        return _save(fig, fig_dir, "15_landscape_p1")


def fig_scaling(df: pd.DataFrame, fig_dir: Path) -> plt.Figure:
    with style():
        fig, ax = plt.subplots(figsize=(7.5, 4))
        ax.plot(df["n_qubits"], df["time_per_eval_s"], marker="o", color=OKABE_ITO["blue"], lw=2,
                label="QAOA time / evaluation")
        ax.plot(df["n_qubits"], df["exact_time_s"], marker="s", color=OKABE_ITO["black"], lw=1.5,
                label="exact brute force (total)")
        ax.set_yscale("log")
        ax.set_xlabel("qubits n")
        ax.set_ylabel("seconds (log)")
        ax2 = ax.twinx()
        ax2.plot(df["n_qubits"], df["statevector_bytes"] / 1024, ls="--", color=OKABE_ITO["vermillion"],
                 label="statevector memory (KiB)")
        ax2.set_yscale("log", base=2)
        ax2.set_ylabel("KiB (log2)")
        ax2.grid(False)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
        ax.set_title("Scaling probe: statevector cost grows as 2^n")
        return _save(fig, fig_dir, "16_scaling")


def fig_circuit(circuit, fig_dir: Path) -> plt.Figure:
    """Fig 17: QAOA circuit diagram (mpl drawer, text fallback)."""
    with style():
        try:
            fig = circuit.draw("mpl", fold=40, style="clifford")
        except Exception:  # noqa: BLE001 - pylatexenc missing etc.
            txt = str(circuit.draw("text", fold=120))
            fig, ax = plt.subplots(figsize=(14, 0.22 * len(txt.splitlines()) + 1))
            ax.text(0, 1, txt, family="monospace", fontsize=6.5, va="top")
            ax.axis("off")
        return _save(fig, fig_dir, "17_circuit_diagram")


def fig_architecture(fig_dir: Path) -> plt.Figure:
    """Fig 18: pipeline architecture (3 x 4 grid of components, QAOA accented)."""
    W, H = 2.2, 0.72
    cols = {"a": 0.24, "b": 2.70, "c": 5.16}
    rows = {"1": 0.56, "2": 1.68, "3": 2.80, "4": 3.92}
    boxes = {
        "data": ("a", "1", "Flood chips", "Sen1Floods11 S2 + water mask", "or synthetic fallback"),
        "feat": ("b", "1", "Feature maps 64 x 64", "water, NDVI, NDBI,", "shoreline proximity"),
        "cnn": ("c", "1", "CNN heatmap", "23,905 parameters", "one score per cell, 0 to 1"),
        "report": ("a", "2", "Field report", "plain text from rescuers", "roof at A4, B3 collapsed"),
        "agent": ("b", "2", "Keyword agent", "boost x1.5, reduce x0.5,", "block sets w = 0"),
        "w": ("c", "2", "Weights w, blocked B", "12 cells, rescaled 0 to 1", "rebuilt after every report"),
        "qubo": ("c", "3", "QUBO builder", "coverage, overlap, radio links", "plus P(sum x - 4)^2, P = 3.8"),
        "qaoa": ("b", "3", "Qiskit QAOA, p = 2", "12 qubits, Aer sampler", "COBYLA tunes 4 angles"),
        "base": ("c", "4", "Classical baselines", "exact, greedy,", "simulated annealing, random"),
        "out": ("b", "4", "Placement + comparison", "4 hover cells, gap vs exact,", "P(optimal), link to base"),
    }
    with style():
        fig, ax = plt.subplots(figsize=(10.2, 6.6))
        ax.set_xlim(0, 7.6)
        ax.set_ylim(4.88, 0)
        ax.axis("off")
        for key, (c, r, name, l1, l2) in boxes.items():
            x, y = cols[c], rows[r]
            main = key == "qaoa"
            ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0,rounding_size=0.08",
                                        facecolor=OKABE_ITO["blue"] if main else "white", alpha=0.14 if main else 1,
                                        edgecolor="none"))
            ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0,rounding_size=0.08", fill=False,
                                        edgecolor=OKABE_ITO["blue"] if main else "#9A9A92",
                                        linewidth=2 if main else 1.2))
            ax.text(x + W / 2, y + 0.24, name, ha="center", va="center", fontsize=10.5, fontweight="bold", color="#1F1F1F")
            ax.text(x + W / 2, y + 0.44, l1, ha="center", va="center", fontsize=9, color="#1F1F1F" if main else "#555555")
            ax.text(x + W / 2, y + 0.60, l2, ha="center", va="center", fontsize=9, color="#1F1F1F" if main else "#555555")

        def arrow(x0, y0, x1, y1):
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color="#8C8C84", lw=1.3, shrinkA=0, shrinkB=0))

        mid = H / 2
        for r in ("1", "2"):
            arrow(cols["a"] + W, rows[r] + mid, cols["b"], rows[r] + mid)
            arrow(cols["b"] + W, rows[r] + mid, cols["c"], rows[r] + mid)
        cx_c, cx_b = cols["c"] + W / 2, cols["b"] + W / 2
        arrow(cx_c, rows["1"] + H, cx_c, rows["2"])
        arrow(cx_c, rows["2"] + H, cx_c, rows["3"])
        arrow(cols["c"], rows["3"] + mid, cols["b"] + W, rows["3"] + mid)
        arrow(cx_c, rows["3"] + H, cx_c, rows["4"])
        arrow(cx_b, rows["3"] + H, cx_b, rows["4"])
        arrow(cols["c"], rows["4"] + mid, cols["b"] + W, rows["4"] + mid)
        ax.text(0.24, 0.30, "ML and the agent build the problem; QAOA chooses the drone cells",
                fontsize=13, fontweight="bold", color="#1F1F1F", va="center")
        ax.text(0.24, 3.30, "Each new report rebuilds w and B,", fontsize=9, color="#555555")
        ax.text(0.24, 3.50, "then QAOA re-solves from its last", fontsize=9, color="#555555")
        ax.text(0.24, 3.70, "angles (warm start).", fontsize=9, color="#555555")
        return _save(fig, fig_dir, "18_architecture")


def fig_xy_mixer(df: pd.DataFrame, fig_dir: Path) -> plt.Figure:
    """Extra figure: standard X mixer versus Dicke + XY ring mixer."""
    g = df.groupby("mixer")
    cols = [("p_valid", "P(valid)", 1), ("p_optimal", "P(optimal)", 1), ("gap", "optimality gap (%)", 100)]
    with style():
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        for ax, (col, lab, sc) in zip(axes, cols):
            mu, sd = g[col].mean() * sc, g[col].std(ddof=0) * sc
            ax.bar(mu.index, mu, yerr=sd, capsize=4, color=[OKABE_ITO["blue"], OKABE_ITO["green"]][: len(mu)],
                   edgecolor="black", linewidth=0.5)
            ax.set_title(lab)
            ax.tick_params(axis="x", labelsize=8)
        fig.suptitle("Constraint-preserving mixer (same depth p)", fontweight="bold")
        return _save(fig, fig_dir, "19_xy_mixer")


def make_agent_gif(steps: Sequence[Mapping[str, Any]], path: Path, fps: float = 0.8) -> Path | None:
    """Optional animated GIF of the placement map after each report."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    frames = [(steps[0]["problem_before"], steps[0]["x_before"], "Initial placement")]
    frames += [(s["problem_after"], s["x_after"], "\n".join(textwrap.wrap(f"Report {i + 1}: {s['report']}", 48)))
               for i, s in enumerate(steps)]
    vmax = max(float(np.max(p.w)) for p, _, _ in frames)
    with style():
        fig, ax = plt.subplots(figsize=(5.6, 5.2))

        def update(i: int) -> list:
            ax.clear()
            p, x, t = frames[i]
            draw_placement(ax, p, x, t, vmax=vmax)
            return []

        anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
        try:
            anim.save(path, writer=PillowWriter(fps=fps), dpi=110)
        except Exception:  # noqa: BLE001 - Pillow missing
            plt.close(fig)
            return None
        plt.close(fig)
    return path
