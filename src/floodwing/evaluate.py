"""Experiment loops: build instances from predicted heatmaps, run every method,
and summarise with optimality gaps, win/tie/loss and a paired Wilcoxon test.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .config import Config
from .qubo import PlacementProblem, build_problem
from .solvers import (SolverResult, exact_solver, greedy_solver, qaoa_variants, qubo_ground_state,
                      random_solver, simulated_annealing, solve_qaoa)
from .utils import finite_or_none, get_logger

METHOD_ORDER = ("Exact", "QAOA", "QAOA-shots", "QAOA-best-sample", "Greedy", "SA", "Random")


@dataclass(frozen=True)
class Instance:
    name: str
    chip_id: str
    chip_index: int
    w: np.ndarray
    problem: PlacementProblem


def build_instances(weights: np.ndarray, chip_ids: Sequence[str], order: Sequence[int], cfg: Config,
                    n: int | None = None, blocked: Sequence[int] = ()) -> list[Instance]:
    """One placement instance per chip heatmap, taken in ``order`` (validation chips first)."""
    n = cfg.n_instances if n is None else n
    out = []
    for rank, idx in enumerate(list(order)[:n]):
        name = f"inst{rank:02d}_{chip_ids[idx]}"
        out.append(Instance(name=name, chip_id=chip_ids[idx], chip_index=int(idx), w=weights[idx],
                            problem=build_problem(weights[idx], cfg, blocked, name=name)))
    return out


def gap(value: float, optimum: float) -> float:
    """Relative optimality gap (f - f*) / |f*|."""
    return float((value - optimum) / max(abs(optimum), 1e-9))


def run_instance(inst: Instance, cfg: Config, seed: int) -> tuple[list[SolverResult], dict[str, Any]]:
    """All methods on one instance. Returns results and the raw QAOA result extras."""
    rng = np.random.default_rng(seed)
    exact = exact_solver(inst.problem)
    qaoa = solve_qaoa(inst.problem, cfg, seed=seed, exact_optimum=exact.objective)
    results = [exact, qaoa, *qaoa_variants(qaoa, inst.problem), greedy_solver(inst.problem),
               simulated_annealing(inst.problem, cfg.sa_steps, rng, cfg.sa_t0, cfg.sa_t1),
               random_solver(inst.problem, cfg.n_random, rng)]
    return results, {"qaoa": qaoa, "exact": exact}


def results_to_rows(inst: Instance, results: Sequence[SolverResult]) -> list[dict[str, Any]]:
    opt = next(r for r in results if r.method == "Exact").objective
    rows = []
    for r in results:
        row = {"instance": inst.name, "chip_id": inst.chip_id, **r.row(), "optimum": opt,
               "gap": gap(r.objective, opt)}
        if r.method == "Random":
            row["objective_std"] = r.extras.get("objective_std")
        rows.append(row)
    return rows


def evaluate_instances(instances: Sequence[Instance], cfg: Config, logger: logging.Logger | None = None
                       ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run every method on every instance. Returns (main_results, raw per-instance QAOA/exact)."""
    logger = logger or get_logger()
    rows: list[dict[str, Any]] = []
    raw: dict[str, Any] = {}
    for i, inst in enumerate(instances):
        gs = qubo_ground_state(inst.problem)
        if not gs["valid"]:
            logger.warning("%s: QUBO ground state is INVALID at P=%.2f (penalty too small)",
                           inst.name, inst.problem.P)
        results, extra = run_instance(inst, cfg, seed=cfg.seed + i)
        raw[inst.name] = extra
        rows.extend(results_to_rows(inst, results))
        q = extra["qaoa"]
        logger.info("%s: QAOA gap %.3f | P(valid) %.2f | P(opt) %.4f | evals %d | %.1fs", inst.name,
                    gap(q.objective, extra["exact"].objective), q.p_valid, q.p_optimal, q.n_evals, q.runtime_s)
    df = pd.DataFrame(rows)
    df["method"] = pd.Categorical(df["method"], categories=list(METHOD_ORDER), ordered=True)
    return df.sort_values(["instance", "method"]).reset_index(drop=True), raw


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/std/median per method plus win/tie/loss against Exact."""
    tol = 1e-6
    out = []
    for method, g in df.groupby("method", observed=True):
        diff = g["objective"].to_numpy() - g["optimum"].to_numpy()
        valid = g["valid"].to_numpy().astype(bool)
        out.append({
            "method": method, "n": len(g),
            "objective_mean": g["objective"].mean(), "objective_std": g["objective"].std(ddof=0),
            "gap_mean": g["gap"].mean(), "gap_std": g["gap"].std(ddof=0), "gap_median": g["gap"].median(),
            "valid_rate": valid.mean(), "coverage_mean": g["coverage"].mean(),
            "overlap_mean": g["overlap"].mean(), "connected_rate": g["connected"].astype(float).mean(),
            "p_valid_mean": pd.to_numeric(g["p_valid"], errors="coerce").mean(),
            "p_optimal_mean": pd.to_numeric(g["p_optimal"], errors="coerce").mean(),
            "evals_mean": pd.to_numeric(g["n_evals"], errors="coerce").mean(),
            "runtime_mean_s": g["runtime_s"].mean(),
            "win_vs_exact": int(np.sum((diff < -tol) & valid)),
            "tie_vs_exact": int(np.sum((np.abs(diff) <= tol) & valid)),
            "loss_vs_exact": int(np.sum((diff > tol) | ~valid)),
        })
    return pd.DataFrame(out)


def paired_test(df: pd.DataFrame, a: str = "QAOA", b: str = "Greedy") -> dict[str, Any]:
    """Two-sided Wilcoxon signed-rank test on objective values (paired by instance)."""
    pa = df[df.method == a].set_index("instance")["objective"]
    pb = df[df.method == b].set_index("instance")["objective"]
    common = pa.index.intersection(pb.index)
    d = (pa[common] - pb[common]).to_numpy()
    res: dict[str, Any] = {"a": a, "b": b, "n": int(len(d)), "mean_diff": finite_or_none(d.mean() if len(d) else None),
                           "a_better": int(np.sum(d < -1e-9)), "b_better": int(np.sum(d > 1e-9)),
                           "ties": int(np.sum(np.abs(d) <= 1e-9))}
    nz = d[np.abs(d) > 1e-9]
    if len(nz) < 1:
        res.update(statistic=None, p_value=None,
                   note="all paired differences are zero: the methods are indistinguishable on these instances")
        return res
    try:
        t = stats.wilcoxon(nz, alternative="two-sided")
        res.update(statistic=float(t.statistic), p_value=float(t.pvalue))
    except ValueError as exc:
        res.update(statistic=None, p_value=None, note=f"test not computable: {exc}")
        return res
    sig = res["p_value"] < 0.05
    better = a if res["mean_diff"] < 0 else b
    res["note"] = (f"significant at 5%: {better} has lower (better) objective on average" if sig else
                   f"not significant at 5% (p={res['p_value']:.3f}, {len(nz)} non-tied pairs); "
                   "no reliable difference between the methods on these instances")
    return res
