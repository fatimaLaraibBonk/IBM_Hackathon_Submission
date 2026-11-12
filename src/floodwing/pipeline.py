"""Stage-by-stage orchestration. ``run_all(cfg)`` reproduces every table,
figure and ``results/summary.json`` in one call; the notebook calls the same
stage functions one section at a time.
"""
from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import bonus, viz
from .agent import SAMPLE_REPORTS, apply_updates, blocked_cells, make_parser
from .config import Config, print_versions, set_seeds
from .data import FloodDataset, build_dataset
from .evaluate import Instance, build_instances, evaluate_instances, gap, paired_test, summarise
from .qubo import build_problem, build_quadratic_program, qubo_matrix, to_ising, to_qubo_program
from .solvers import (SolverResult, describe_primitives, exact_solver, greedy_solver, numpy_eigensolver_check,
                      qaoa_variants, qubo_ground_state, random_solver, simulated_annealing, solve_qaoa)
from .utils import cell_name, finite_or_none, get_logger, save_json, timed


@dataclass
class PipelineState:
    """Everything produced by the stages (passed explicitly, never global)."""

    cfg: Config
    logger: logging.Logger
    t_start: float = field(default_factory=time.perf_counter)
    timings: dict[str, float] = field(default_factory=dict)
    versions: dict[str, str] = field(default_factory=dict)
    primitives: str = ""
    dataset: FloodDataset | None = None
    train: Any = None
    instances: list[Instance] = field(default_factory=list)
    demo: Instance | None = None
    demo_results: dict[str, SolverResult] = field(default_factory=dict)
    qubo_info: dict[str, Any] = field(default_factory=dict)
    agent_steps: list[dict[str, Any]] = field(default_factory=list)
    main_df: pd.DataFrame | None = None
    summary_df: pd.DataFrame | None = None
    wilcoxon: dict[str, Any] = field(default_factory=dict)
    raw_eval: dict[str, Any] = field(default_factory=dict)
    bonus: dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    figures: list[str] = field(default_factory=list)

    def elapsed(self) -> float:
        return time.perf_counter() - self.t_start

    def note(self, msg: str) -> None:
        """Record a limitation / skipped item; it ends up in summary.json."""
        self.logger.warning(msg)
        self.issues.append(msg)


def _budget_s(cfg: Config) -> float:
    return cfg.time_budget_fast_s if cfg.fast_mode else cfg.time_budget_full_s


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def stage_setup(cfg: Config, logger: logging.Logger | None = None) -> PipelineState:
    """Seeds, directories, version printout, primitive selection."""
    logger = logger or get_logger()
    set_seeds(cfg.seed)
    cfg.ensure_dirs()
    st = PipelineState(cfg=cfg, logger=logger)
    st.versions = print_versions()
    st.primitives = describe_primitives(cfg)
    print("Qiskit primitives ->", st.primitives)
    print(f"FAST_MODE={cfg.fast_mode} | grid {cfg.rows}x{cfg.cols} ({cfg.n_cells} qubits) | k={cfg.k} | "
          f"work_dir={cfg.work_dir}")
    return st


def stage_data(st: PipelineState) -> FloodDataset:
    with timed("data", st.timings, st.logger):
        st.dataset = build_dataset(st.cfg, st.logger)
    if "synthetic" in st.dataset.sources:
        st.note(f"data source: {st.dataset.source_summary} (synthetic chips were used)")
    print(f"Data source in use -> {st.dataset.source_summary}")
    return st.dataset


def stage_heatmap(st: PipelineState):
    from .heatmap import train_heatmap_model

    with timed("heatmap_cnn", st.timings, st.logger):
        st.train = train_heatmap_model(st.dataset, st.cfg, st.logger)
    print(f"CNN: {st.train.n_params} parameters on {st.train.device} | validation {st.train.metrics} | "
          f"mean-predictor MAE {st.train.extras['mean_baseline_mae']:.4f}")
    return st.train


def stage_instances(st: PipelineState) -> list[Instance]:
    """Instances from predicted heatmaps, validation chips first; instance 0 is the demo."""
    order = list(st.train.val_idx) + list(st.train.train_idx)
    st.instances = build_instances(st.train.weights, st.dataset.chip_ids, order, st.cfg)
    d = st.instances[0]
    name = f"demo_{d.chip_id}"
    st.demo = Instance(name=name, chip_id=d.chip_id, chip_index=d.chip_index, w=d.w,
                       problem=build_problem(d.w, st.cfg, name=name))
    print(f"{len(st.instances)} evaluation instances; demo scenario = {name}")
    return st.instances


def stage_qubo(st: PipelineState) -> dict[str, Any]:
    """Build and verify the QUBO of the demo instance."""
    prob = st.demo.problem
    with timed("qubo", st.timings, st.logger):
        qp = build_quadratic_program(prob)
        qubo_qp = to_qubo_program(prob)
        op, offset = to_ising(prob)
        Q, c = qubo_matrix(prob)
        rng = np.random.default_rng(st.cfg.seed)
        X = rng.integers(0, 2, size=(50, prob.n))
        diffs = [abs(qubo_qp.objective.evaluate(x) - (x @ Q @ x + c)) for x in X]
        gs = qubo_ground_state(prob)
        mes = numpy_eigensolver_check(prob)
        ex = exact_solver(prob)
    st.qubo_info = {
        "n_qubits": prob.n, "k": prob.k, "a": prob.a, "b": prob.b, "P": prob.P, "P_block": prob.P_block,
        "radio_range": prob.radio_range, "n_pauli_terms": len(op), "ising_offset": float(offset),
        "max_abs_qp_vs_matrix": float(max(diffs)), "qubo_ground_state_valid": bool(gs["valid"]),
        "numpy_eigensolver_objective": mes["objective"], "exact_objective": ex.objective,
        "eigensolver_matches_exact": bool(np.isclose(mes["objective"], ex.objective)),
        "blocked_handling": "linear penalty P_block * x_i (variable kept so qubit map is fixed)",
        "n_quadratic_terms": len(qp.objective.quadratic.to_dict()),
    }
    print(qp.prettyprint()[:1200] + "\n...")
    for k_, v in st.qubo_info.items():
        print(f"  {k_}: {v}")
    if st.qubo_info["max_abs_qp_vs_matrix"] > 1e-8:
        st.note("QUBO matrix and QuadraticProgram disagree; check qubo.py")
    return st.qubo_info


def solve_all_methods(problem, cfg: Config, seed: int) -> dict[str, SolverResult]:
    rng = np.random.default_rng(seed)
    ex = exact_solver(problem)
    q = solve_qaoa(problem, cfg, seed=seed, exact_optimum=ex.objective)
    shots, best = qaoa_variants(q, problem)
    return {"Exact": ex, "QAOA": q, "QAOA-shots": shots, "QAOA-best-sample": best,
            "Greedy": greedy_solver(problem), "SA": simulated_annealing(problem, cfg.sa_steps, rng, cfg.sa_t0, cfg.sa_t1),
            "Random": random_solver(problem, cfg.n_random, rng)}


def stage_demo_solve(st: PipelineState) -> pd.DataFrame:
    """QAOA + baselines on the demo scenario."""
    with timed("demo_solve", st.timings, st.logger):
        st.demo_results = solve_all_methods(st.demo.problem, st.cfg, st.cfg.seed)
    opt = st.demo_results["Exact"].objective
    df = pd.DataFrame([{**r.row(), "gap": gap(r.objective, opt),
                        "cells_named": ", ".join(cell_name(i, st.cfg.cols) for i in np.flatnonzero(r.x))}
                       for r in st.demo_results.values()])
    q = st.demo_results["QAOA"]
    print(f"QAOA primitive: {q.extras['primitive']} | decode: {q.extras['decode_rule']} | "
          f"{q.extras['n_pauli_terms']} Pauli terms | optimal params {np.round(q.extras['optimal_point'], 3)}")
    return df


def stage_agent(st: PipelineState, reports: tuple[str, ...] = SAMPLE_REPORTS) -> list[dict[str, Any]]:
    """Apply the sample reports sequentially to the demo heatmap; re-solve with QAOA after each."""
    cfg = st.cfg
    parser = make_parser(cfg, st.logger)
    with timed("agent", st.timings, st.logger):
        w = st.demo.w.copy()
        blocked: set[int] = set()
        prob = st.demo.problem
        prev = st.demo_results.get("QAOA") or solve_qaoa(prob, cfg)
        steps = []
        for i, text in enumerate(reports):
            updates = parser(text, cfg.rows, cfg.cols)
            w_new = apply_updates(w, updates, cfg.w_cap)
            blocked_new = blocked | blocked_cells(updates)
            prob_new = build_problem(w_new, cfg, blocked_new, name=f"{st.demo.name}_after_r{i + 1}")
            ex = exact_solver(prob_new)
            res = solve_qaoa(prob_new, cfg, seed=cfg.seed + 100 + i, exact_optimum=ex.objective)
            steps.append({"report": text, "updates": updates, "problem_before": prob, "x_before": prev.x,
                          "problem_after": prob_new, "x_after": res.x, "qaoa": res, "exact": ex,
                          "gap": gap(res.objective, ex.objective)})
            print(f"Report {i + 1}: {text}\n   updates: {[(u['cell'], u['action']) for u in updates]}\n"
                  f"   QAOA placement {[cell_name(j, cfg.cols) for j in np.flatnonzero(res.x)]} "
                  f"(gap {steps[-1]['gap']:.3f})")
            w, blocked, prob, prev = w_new, blocked_new, prob_new, res
    st.agent_steps = steps
    return steps


def stage_evaluate(st: PipelineState) -> tuple[pd.DataFrame, pd.DataFrame]:
    with timed("evaluation", st.timings, st.logger):
        st.main_df, st.raw_eval = evaluate_instances(st.instances, st.cfg, st.logger)
        st.summary_df = summarise(st.main_df)
        st.wilcoxon = paired_test(st.main_df, "QAOA", "Greedy")
    st.main_df.to_csv(st.cfg.table_dir / "main_results.csv", index=False)
    st.summary_df.to_csv(st.cfg.table_dir / "summary.csv", index=False)
    print(f"Wilcoxon QAOA vs Greedy: {st.wilcoxon}")
    return st.main_df, st.summary_df


def _run_bonus(st: PipelineState, key: str, flag: bool, fn: Callable[[], Any]) -> Any:
    if not flag:
        st.note(f"bonus '{key}' skipped (disabled in Config)")
        return None
    if st.elapsed() > _budget_s(st.cfg):
        st.note(f"bonus '{key}' skipped: time budget {_budget_s(st.cfg):.0f}s exceeded")
        return None
    try:
        with timed(f"bonus_{key}", st.timings, st.logger):
            out = fn()
        st.bonus[key] = out
        return out
    except Exception as exc:  # noqa: BLE001 - a failed bonus must never stop the core
        st.note(f"bonus '{key}' FAILED: {type(exc).__name__}: {exc}")
        return None


def stage_bonus(st: PipelineState) -> dict[str, Any]:
    """Run every enabled bonus experiment in priority order (1,2,3 then 4,6,7,8 then 5)."""
    cfg, lg = st.cfg, st.logger
    insts = st.instances[: cfg.bonus_instances]
    parser = make_parser(cfg, lg)
    _run_bonus(st, "depth", cfg.run_bonus_depth, lambda: bonus.depth_study(insts, cfg, lg))
    _run_bonus(st, "penalty", cfg.run_bonus_penalty, lambda: bonus.penalty_sweep(insts, cfg, lg))
    _run_bonus(st, "warmstart", cfg.run_bonus_warmstart,
               lambda: bonus.warm_start_study(st.demo, cfg, SAMPLE_REPORTS[0], parser, apply_updates,
                                              blocked_cells, logger=lg))
    _run_bonus(st, "optimizers", cfg.run_bonus_optimizers, lambda: bonus.optimizer_comparison(insts, cfg, logger=lg))
    _run_bonus(st, "noise", cfg.run_bonus_noise,
               lambda: bonus.noise_study(st.demo_results["QAOA"], st.demo.problem, cfg, lg))
    _run_bonus(st, "landscape", cfg.run_bonus_landscape, lambda: bonus.landscape_p1(st.demo.problem, cfg))
    _run_bonus(st, "scaling", cfg.run_bonus_scaling, lambda: bonus.scaling_probe(cfg, lg))
    _run_bonus(st, "xy_mixer", cfg.run_bonus_xy_mixer, lambda: bonus.xy_mixer_study(insts, cfg, lg))
    for key in ("depth", "penalty", "optimizers", "noise", "scaling", "xy_mixer"):
        if isinstance(st.bonus.get(key), pd.DataFrame):
            st.bonus[key].to_csv(cfg.table_dir / f"bonus_{key}.csv", index=False)
    return st.bonus


def _toy_circuit(cfg: Config):
    """6-qubit (2x3, k=2) QAOA p=1 circuit, decomposed to readable gates."""
    from qiskit.circuit.library import QAOAAnsatz

    toy = build_problem(np.linspace(0.2, 1.0, 6), cfg.with_overrides(rows=2, cols=3, k=2), name="toy")
    op, _ = to_ising(toy)
    return QAOAAnsatz(op, reps=1).decompose().decompose()


def stage_figures(st: PipelineState) -> list[str]:
    """Generate every figure whose inputs exist; failures are recorded, not raised."""
    cfg, fd = st.cfg, st.cfg.fig_dir
    jobs: list[tuple[str, Callable[[], Any]]] = []
    ds, tr = st.dataset, st.train
    if ds is not None:
        demo_idx = st.demo.chip_index if st.demo else 0
        jobs.append(("01", lambda: viz.fig_data_overview(ds, demo_idx, fd)))
    if tr is not None:
        jobs += [("02", lambda: viz.fig_heatmap_prediction(ds, tr, fd)), ("03", lambda: viz.fig_training_curves(tr.history, fd))]
    if st.demo_results:
        q = st.demo_results["QAOA"]
        jobs += [("04", lambda: viz.fig_placement_map(st.demo.problem, q.x, fd, f"QAOA placement: {st.demo.name}")),
                 ("05", lambda: viz.fig_method_comparison(st.demo.problem, {m: st.demo_results[m] for m in
                                                                             ("QAOA", "Exact", "Greedy")}, fd)),
                 ("07", lambda: viz.fig_bitstring_probs(st.demo.problem, q.extras["p_exact"], fd))]
    if st.raw_eval:
        hist = [v["qaoa"].extras["history"] for v in st.raw_eval.values()]
        opts = [v["exact"].objective for v in st.raw_eval.values()]
        jobs.append(("06", lambda: viz.fig_qaoa_convergence(hist, opts, fd)))
    if st.main_df is not None:
        jobs.append(("08", lambda: viz.fig_gap_distribution(st.main_df, fd)))
    if st.agent_steps:
        jobs.append(("09", lambda: viz.fig_agent_before_after(st.agent_steps, fd)))
    b = st.bonus
    for key, fn in (("depth", viz.fig_depth_study), ("penalty", viz.fig_penalty_sensitivity),
                    ("warmstart", viz.fig_warm_start), ("optimizers", viz.fig_optimizer_comparison),
                    ("noise", viz.fig_noise), ("landscape", viz.fig_landscape), ("scaling", viz.fig_scaling),
                    ("xy_mixer", viz.fig_xy_mixer)):
        if b.get(key) is not None:
            jobs.append((key, lambda fn=fn, key=key: fn(b[key], fd)))
    jobs += [("17", lambda: viz.fig_circuit(_toy_circuit(cfg), fd)), ("18", lambda: viz.fig_architecture(fd))]

    import matplotlib.pyplot as plt

    made = []
    with timed("figures", st.timings, st.logger):
        for tag, job in jobs:
            try:
                fig = job()
                made.append(tag)
                plt.close(fig)
            except Exception as exc:  # noqa: BLE001
                st.note(f"figure {tag} failed: {type(exc).__name__}: {exc}")
        if cfg.make_gif and st.agent_steps:
            try:
                if viz.make_agent_gif(st.agent_steps, fd / "agent_updates.gif") is None:
                    st.note("GIF not written (Pillow unavailable)")
            except Exception as exc:  # noqa: BLE001
                st.note(f"GIF failed: {exc}")
    st.figures = sorted(p.name for p in fd.glob("*.png"))
    print(f"{len(st.figures)} figures in {fd}")
    return st.figures


LIMITATIONS: tuple[str, ...] = (
    "n <= 16 variables are solved exactly in milliseconds by brute force, so no quantum advantage is claimed.",
    "QAOA with the standard X mixer spends most of its effort on the sum x = k penalty: P(optimal) stays a "
    "small multiple of uniform guessing at p <= 3, and results depend on the penalty weight (see the sweep).",
    "The QAOA placement is decoded from the exact statevector distribution (simulator-only). The hardware-"
    "realistic most-frequent-shot decoding (QAOA-shots) is far noisier and is reported separately.",
    "QAOA-best-sample (best of 4096 shots over 495 valid states) is close to brute force and is not evidence "
    "of quantum quality on its own.",
    "Survivor likelihood is a hand-defined proxy (0.45 flood + 0.35 built-up + 0.20 edge + noise), not real "
    "survivor data, so the CNN learns a known function.",
    "The pairwise radio reward does not guarantee a connected relay chain to base; the post-hoc connectivity "
    "rate is reported per method.",
    "The agent is a keyword parser and is brittle to unusual phrasing; a human confirmation step is needed in "
    "practice.",
    "Simulator and simple noise-model results do not predict real-hardware behaviour.",
    "Scaling beyond 16 qubits was not tested.",
    "Default radio range is 2.25 cells, not 1.5: with R = 1.5 the radio matrix equals the 8-neighbour overlap "
    "matrix and the overlap/connectivity terms partly cancel.",
)


def build_summary(st: PipelineState) -> dict[str, Any]:
    """Headline numbers for the report (``results/summary.json``)."""
    cfg = st.cfg
    out: dict[str, Any] = {
        "project": "FloodWing", "fast_mode": cfg.fast_mode, "seed": cfg.seed, "versions": st.versions,
        "primitives": st.primitives, "config": cfg.to_dict(), "timings_s": st.timings,
        "total_runtime_s": st.elapsed(), "issues_and_skips": st.issues, "limitations": list(LIMITATIONS),
    }
    if st.dataset is not None:
        out["data"] = {"n_chips": st.dataset.n, "sources": st.dataset.source_summary,
                       "proxy_target": "clip(0.45*flood + 0.35*built_up + 0.20*edge + N(0, 0.03), 0, 1)"}
    if st.train is not None:
        out["cnn"] = {"n_params": st.train.n_params, "device": st.train.device, "validation": st.train.metrics,
                      "train": st.train.extras["train_metrics"],
                      "mean_predictor_val_mae": st.train.extras["mean_baseline_mae"],
                      "epochs": len(st.train.history["train_loss"])}
    out["qubo"] = st.qubo_info
    if st.demo_results:
        opt = st.demo_results["Exact"].objective
        out["demo"] = {"name": st.demo.name, **{m: {"cells": [cell_name(i, cfg.cols) for i in np.flatnonzero(r.x)],
                                                    "objective": r.objective, "gap": gap(r.objective, opt),
                                                    "connected": r.connected, "p_optimal": r.p_optimal,
                                                    "p_valid": r.p_valid}
                                                for m, r in st.demo_results.items()}}
    if st.summary_df is not None:
        out["evaluation"] = {"n_instances": int(st.main_df["instance"].nunique()),
                             "per_method": st.summary_df.replace({np.nan: None}).to_dict(orient="records"),
                             "wilcoxon_qaoa_vs_greedy": st.wilcoxon}
    if st.agent_steps:
        out["agent"] = [{"report": s["report"], "updates": [(u["cell"], u["action"]) for u in s["updates"]],
                         "placement_after": [cell_name(i, cfg.cols) for i in np.flatnonzero(s["x_after"])],
                         "gap_after": s["gap"]} for s in st.agent_steps]
    b: dict[str, Any] = {}
    if isinstance(st.bonus.get("depth"), pd.DataFrame):
        b["depth"] = st.bonus["depth"].groupby("p")[["gap", "p_optimal", "p_valid", "evals"]].mean().reset_index().to_dict(orient="records")
    if isinstance(st.bonus.get("penalty"), pd.DataFrame):
        b["penalty"] = st.bonus["penalty"].groupby("multiplier")[["p_valid", "gap", "ground_state_valid"]].mean().reset_index().to_dict(orient="records")
    if st.bonus.get("warmstart"):
        ws = st.bonus["warmstart"]
        b["warm_start"] = {"warm": ws["warm_summary"],
                           "random_init_mean": pd.DataFrame(ws["cold_summary"]).mean(numeric_only=True).to_dict()}
    if isinstance(st.bonus.get("optimizers"), pd.DataFrame):
        b["optimizers"] = st.bonus["optimizers"].groupby("optimizer")[["gap", "evals", "runtime_s"]].mean().reset_index().to_dict(orient="records")
    if isinstance(st.bonus.get("noise"), pd.DataFrame):
        b["noise"] = st.bonus["noise"][["level", "p_optimal", "p_valid", "mean_energy"]].to_dict(orient="records")
    if st.bonus.get("landscape"):
        L = st.bonus["landscape"]
        b["landscape"] = {"grid_min": L["grid_min"], "qaoa_p1_energy": L["qaoa_energy"],
                          "optimum_gamma_beta": [float(v[0]) for v in L["optimum_point"]]}
    if isinstance(st.bonus.get("scaling"), pd.DataFrame):
        b["scaling"] = st.bonus["scaling"][["n_qubits", "time_per_eval_s", "statevector_bytes", "exact_time_s"]].to_dict(orient="records")
    if isinstance(st.bonus.get("xy_mixer"), pd.DataFrame):
        b["xy_mixer"] = st.bonus["xy_mixer"].groupby("mixer")[["p_valid", "p_optimal", "gap", "evals"]].mean().reset_index().to_dict(orient="records")
    out["bonus"] = b
    out["figures"] = st.figures
    return out


def stage_export(st: PipelineState) -> dict[str, Any]:
    summary = build_summary(st)
    save_json(summary, st.cfg.results_dir / "summary.json")
    archive = shutil.make_archive(str(st.cfg.results_dir / "figures"), "zip", st.cfg.fig_dir)
    print(f"wrote {st.cfg.results_dir / 'summary.json'} and {archive}")
    return summary


def run_all(cfg: Config | None = None, logger: logging.Logger | None = None) -> PipelineState:
    """Reproduce everything: data -> CNN -> agent -> QUBO -> QAOA + baselines ->
    evaluation -> bonus -> figures -> summary.json."""
    cfg = cfg or Config()
    st = stage_setup(cfg, logger)
    stage_data(st)
    stage_heatmap(st)
    stage_instances(st)
    stage_qubo(st)
    stage_demo_solve(st)
    stage_agent(st)
    stage_evaluate(st)
    stage_bonus(st)
    stage_figures(st)
    stage_export(st)
    print(f"total runtime {st.elapsed():.0f}s | timings: " +
          ", ".join(f"{k}={v:.0f}s" for k, v in st.timings.items()))
    return st
