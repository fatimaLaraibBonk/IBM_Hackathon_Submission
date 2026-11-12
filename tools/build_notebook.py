"""Build notebook/floodwing_kaggle.ipynb from the package sources.

The notebook is self-contained for Kaggle: it writes the package and tests to
disk with %%writefile cells (generated from src/ so there is one source of
truth), then runs each pipeline stage in its own section.

Usage:  python tools/build_notebook.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "floodwing"
MODULES = ("__init__", "config", "utils", "data", "qubo", "solvers", "heatmap", "agent", "evaluate",
           "bonus", "viz", "pipeline")


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str, hidden: bool = False):
    cell = nbf.v4.new_code_cell(text.strip())
    if hidden:
        cell.metadata["jupyter"] = {"source_hidden": True}
        cell.metadata["collapsed"] = True
    return cell


def build() -> nbf.NotebookNode:
    reqs = [ln.split("#")[0].strip() for ln in (ROOT / "requirements.txt").read_text().splitlines()]
    pins = " ".join(f'"{r}"' for r in reqs if r and not r.startswith(("numpy", "scipy", "pandas", "matplotlib",
                                                                          "torch", "pillow", "tqdm", "pytest",
                                                                          "scikit", "seaborn")))
    cells = [
        md("""
# FloodWing: hybrid CNN + agent + QAOA drone placement for flood search-and-rescue

**Qiskit Fall Fest 2026 | Domain: UAVs & Robotics (Disaster Response)**

After a flood, a coordinator has `k = 4` drones and a grid of candidate hover cells. FloodWing chooses where
they hover so that likely-survivor cells are covered, footprints do not overlap, and drones stay within radio
range of each other:

1. **CNN** predicts a (proxy) survivor-likelihood heatmap from Sentinel-2 flood features (Sen1Floods11).
2. **Agent** turns plain-text field reports into heatmap updates (boost / block / reduce).
3. **QUBO + QAOA (Qiskit)** chooses the placement; it is compared with exact brute force, greedy, simulated
   annealing and random baselines.

No quantum speedup is claimed: 12 qubits are solved exactly in milliseconds. The goal is a correct,
reproducible hybrid formulation with an honest analysis. Run top to bottom with **Internet ON**.
`FAST_MODE=True` targets < 30 min on Kaggle CPU.
"""),
        md("""
## 1. Setup and config
Installs the pinned Qiskit stack, writes the `floodwing` package to disk (one `%%writefile` cell per module,
generated from the repository so the code is identical), prints versions and the chosen Qiskit primitives.
"""),
        code(f"""
import importlib.util, subprocess, sys
# Pinned Qiskit stack + data/plot extras. Kaggle's numpy/torch builds are kept.
!pip install -q {pins}
if importlib.util.find_spec("torch") is None:      # PyTorch is preinstalled on Kaggle; install if missing
    !pip install -q torch
"""),
        code("""
from pathlib import Path
PKG_DIR = Path("floodwing_pkg")                  # package is written here (relative to the working dir)
for sub in ("src/floodwing", "tests"):
    (PKG_DIR / sub).mkdir(parents=True, exist_ok=True)
print("package dir:", PKG_DIR.resolve())
"""),
        md("### 1.1 Package source (written to disk; collapse this section)"),
    ]
    for name in MODULES:
        body = (SRC / f"{name}.py").read_text()
        cells.append(code(f"%%writefile {{PKG_DIR}}/src/floodwing/{name}.py\n{body}", hidden=True))
    cells.append(code("%%writefile {PKG_DIR}/tests/test_core.py\n" + (ROOT / "tests" / "test_core.py").read_text(),
                      hidden=True))
    cells += [
        md("### 1.2 Configuration\nEvery hyperparameter lives in the frozen `Config` dataclass. Change values with "
           "`cfg.with_overrides(...)`. `FAST_MODE=False` runs the full experiment (< 90 min)."),
        code("""
import sys, warnings, importlib
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str((PKG_DIR / "src").resolve()))
for m in [m for m in list(sys.modules) if m.startswith("floodwing")]:
    del sys.modules[m]                            # pick up freshly written sources on re-runs

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image, Markdown
from floodwing.config import Config
from floodwing import pipeline as P, viz, agent
from floodwing.utils import cell_name

FAST_MODE = True
WORK_DIR = "/kaggle/working" if Path("/kaggle/working").exists() else (
    str(Path.cwd().parent) if Path.cwd().name == "notebook" else str(Path.cwd()))
cfg = Config(fast_mode=FAST_MODE, work_dir=WORK_DIR)   # set use_synthetic=True to skip the download
st = P.stage_setup(cfg)                                # seeds, folders, versions, primitives
pd.set_option("display.precision", 4)
"""),
        md("""
## 2. Data download (Sen1Floods11) and preprocessing
A small subset of hand-labelled Sen1Floods11 chips (Bonafilia et al., 2020) is downloaded anonymously from
`gs://sen1floods11` (paths are discovered by listing the bucket). If the bucket is unreachable, a procedural
generator with the same schema is used, so nothing downstream changes. Each chip becomes a 64x64 tensor with
water, NDVI, built-up (NDBI) and flood-edge proximity, then is pooled onto the 3x4 planning grid.
**The survivor target is a hand-defined proxy** (`0.45 flood + 0.35 built-up + 0.20 edge + noise`).
"""),
        code("""
ds = P.stage_data(st)
print(f"X {ds.X.shape} | Y {ds.Y.shape} | sources: {ds.source_summary}")
viz.fig_data_overview(ds, 0, cfg.fig_dir); plt.show()
"""),
        md("""
## 3. Heatmap model (CNN)
Three Conv-BN-ReLU blocks (16/32/64 channels), adaptive average pooling to the grid, 1x1 head, sigmoid
(< 100k parameters). MSE loss, Adam, 80/20 split by chip. Metrics on the validation chips: MAE, Pearson,
Spearman and top-k overlap. Predictions are min-max scaled per chip to give the QUBO weights `w`.
"""),
        code("""
train = P.stage_heatmap(st)
viz.fig_training_curves(train.history, cfg.fig_dir); plt.show()
viz.fig_heatmap_prediction(ds, train, cfg.fig_dir); plt.show()
"""),
        md("""
## 4. Agent layer
A deterministic keyword parser (no API keys) maps reports to updates. Cells are named letter = row,
number = column (A1..C4). Out-of-range cells are ignored and logged. Five sample reports are applied **in
sequence** to the demo heatmap and QAOA re-solves after each one. A local-LLM parser exists behind
`Config.use_local_llm` (default OFF) and falls back to the rules on any malformed output.
"""),
        code("""
P.stage_instances(st)
for text in agent.SAMPLE_REPORTS:
    print(f"{text}\\n  -> {agent.parse_report(text, cfg.rows, cfg.cols, cfg)}")
steps = P.stage_agent(st)
viz.fig_agent_before_after(steps, cfg.fig_dir); plt.show()
"""),
        md("""
## 5. QUBO formulation
`f(x) = -sum w_i x_i + a sum O_ij x_i x_j - b sum C_ij x_i x_j`, plus `P (sum x - k)^2` and a linear penalty for
blocked cells. The model is built as a `QuadraticProgram`, converted with `QuadraticProgramToQubo`, and checked
against an explicit QUBO matrix; the unit tests run here too.
"""),
        code("""
qubo_info = P.stage_qubo(st)
import pytest
exit_code = pytest.main(["-q", "-p", "no:cacheprovider", "-W", "ignore", str(PKG_DIR / "tests" / "test_core.py")])
print("pytest exit code:", int(exit_code))
"""),
        md("""
## 6. QAOA solve + baselines
`qiskit_algorithms.QAOA` (p = 2, COBYLA, Aer sampler) on the demo scenario, against exact brute force,
greedy, simulated annealing and random placements. Three QAOA read-outs are reported: **QAOA** (most probable
valid bitstring of the exact output state), **QAOA-shots** (most frequent valid bitstring in 4096 shots, what
hardware would give) and **QAOA-best-sample** (best valid bitstring among the shots, close to brute force).
"""),
        code("""
demo_df = P.stage_demo_solve(st)
display(demo_df[["method", "objective", "gap", "valid", "coverage", "overlap", "connected", "p_valid",
                 "p_optimal", "n_evals", "runtime_s", "cells_named"]])
q = st.demo_results["QAOA"]
viz.fig_placement_map(st.demo.problem, q.x, cfg.fig_dir, f"QAOA placement: {st.demo.name}"); plt.show()
viz.fig_method_comparison(st.demo.problem, {m: st.demo_results[m] for m in ("QAOA", "Exact", "Greedy")}, cfg.fig_dir); plt.show()
viz.fig_bitstring_probs(st.demo.problem, q.extras["p_exact"], cfg.fig_dir); plt.show()
"""),
        md("""
## 7. Evaluation across instances
Every method on every instance (one instance per chip heatmap, validation chips first). Outputs
`results/tables/main_results.csv`, `results/tables/summary.csv` and a paired Wilcoxon test of QAOA vs greedy.
"""),
        code("""
main_df, summary_df = P.stage_evaluate(st)
display(summary_df[["method", "n", "gap_mean", "gap_median", "valid_rate", "p_valid_mean", "p_optimal_mean",
                    "connected_rate", "evals_mean", "runtime_mean_s", "tie_vs_exact", "loss_vs_exact"]])
print(st.wilcoxon["note"])
hist = [v["qaoa"].extras["history"] for v in st.raw_eval.values()]
opts = [v["exact"].objective for v in st.raw_eval.values()]
viz.fig_qaoa_convergence(hist, opts, cfg.fig_dir); plt.show()
viz.fig_gap_distribution(main_df, cfg.fig_dir); plt.show()
"""),
        md("""
## 8. Bonus quantum experiments
Each experiment is gated by a `Config` flag and wrapped so a failure is logged (and reported in Section 10)
instead of stopping the run: depth study, penalty sweep, warm start after an agent update, optimizer
comparison, noise robustness, (gamma, beta) landscape, scaling probe, and a constraint-preserving XY mixer.
"""),
        code("""
bonus = P.stage_bonus(st)
GROUP_BY = {"depth": "p", "penalty": "multiplier", "optimizers": "optimizer", "xy_mixer": "mixer"}
for key in ("depth", "penalty", "optimizers", "xy_mixer", "noise", "scaling"):
    df_b = bonus.get(key)
    if not isinstance(df_b, pd.DataFrame):
        continue
    display(Markdown(f"**{key}**"))
    display(df_b.groupby(GROUP_BY[key]).mean(numeric_only=True) if key in GROUP_BY else df_b)
if bonus.get("warmstart"):
    print("warm start:", bonus["warmstart"]["warm_summary"])
    print("random init:", pd.DataFrame(bonus["warmstart"]["cold_summary"]).mean().to_dict())
"""),
        md("## 9. Visualisations gallery\nAll figures are regenerated at 200 dpi in `results/figures/`."),
        code("""
figs = P.stage_figures(st)
for name in figs:
    display(Markdown(f"**{name}**")); display(Image(filename=str(cfg.fig_dir / name), width=900))
gif = cfg.fig_dir / "agent_updates.gif"
if gif.exists():
    display(Image(filename=str(gif)))
"""),
        md("""
## 10. Limitations and honest reflection
* 12-16 variables are solved exactly in milliseconds, so there is **no quantum advantage claim**.
* With the standard X mixer, QAOA mostly learns the `sum x = k` constraint: P(optimal) stays a small multiple of
  uniform guessing at p <= 3 and depends on the penalty weight (Section 8). The XY mixer keeps every sample valid.
* The headline **QAOA** read-out uses the exact statevector distribution (simulator-only). The
  hardware-realistic **QAOA-shots** read-out is much noisier; **QAOA-best-sample** is close to brute force.
* The survivor likelihood is a **proxy**, so the CNN learns a known function plus noise.
* The pairwise radio reward does not guarantee a connected relay chain to base (post-hoc connectivity is reported).
* The agent is a keyword parser and is brittle to unusual phrasing.
* Simulator and simple noise-model results do not predict real hardware. Scaling beyond 16 qubits is untested.

Items recorded during this run (data source, skipped or failed bonus items, figure failures):
"""),
        code("""
for msg in st.issues or ["(none)"]:
    print("-", msg)
"""),
        md("## 11. Export\nWrites `results/summary.json` (every headline number used in the report) and zips the figures. "
           "`P.run_all(cfg)` reproduces Sections 1-11 in one call."),
        code("""
summary = P.stage_export(st)
print(f"total runtime {st.elapsed():.0f}s")
display(pd.Series(st.timings, name="seconds").round(1))
"""),
    ]
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb


if __name__ == "__main__":
    out = ROOT / "notebook" / "floodwing_kaggle.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build(), out)
    print(f"wrote {out}")
