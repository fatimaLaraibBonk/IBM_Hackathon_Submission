"""Bonus quantum experiments. Each function is independent, gated by a Config
flag in the pipeline, and returns plain data (DataFrame / dict) for plotting.

1. depth_study           QAOA depth p versus solution quality
2. penalty_sweep         penalty weight P versus validity and gap
3. warm_start_study      re-solve after an agent update: random vs warm init
4. optimizer_comparison  COBYLA vs SPSA vs Nelder-Mead
5. xy_mixer_study        Dicke initial state + ring XY mixer (stays in sum x = k)
6. noise_study           depolarising + readout noise on the trained circuit
7. landscape_p1          (gamma, beta) energy surface with the optimiser path
8. scaling_probe         runtime and memory versus qubit count
"""
from __future__ import annotations

import logging
import time
from math import comb
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import Config
from .evaluate import Instance, gap
from .qubo import (PlacementProblem, all_energies, build_problem, bits_from_key, decode, valid_mask,
                   with_penalty)
from .solvers import (SolverResult, exact_solver, ising_diagonal, merge_params, numpy_qaoa_energy,
                      qubo_ground_state, scaled_ising, solve_qaoa, split_params)
from .utils import get_logger


def _qaoa_row(inst: Instance, res: SolverResult, opt: float, **extra: Any) -> dict[str, Any]:
    return {"instance": inst.name, "gap": gap(res.objective, opt), "p_valid": res.p_valid,
            "p_optimal": res.p_optimal, "evals": res.n_evals, "runtime_s": res.runtime_s,
            "expected_energy": res.extras["expected_energy"], "valid": res.valid, **extra}


# 1 -------------------------------------------------------------------------
def depth_study(instances: Sequence[Instance], cfg: Config, logger: logging.Logger | None = None) -> pd.DataFrame:
    """QAOA at each depth in ``cfg.depth_values`` on the bonus instances."""
    logger = logger or get_logger()
    rows = []
    for inst in instances:
        opt = exact_solver(inst.problem).objective
        for p in cfg.depth_values:
            res = solve_qaoa(inst.problem, cfg, reps=p, restarts=cfg.bonus_restarts, exact_optimum=opt)
            rows.append(_qaoa_row(inst, res, opt, p=p))
            logger.info("depth p=%d %s: gap %.3f P(opt) %.4f evals %d", p, inst.name, rows[-1]["gap"],
                        res.p_optimal, res.n_evals)
    return pd.DataFrame(rows)


# 2 -------------------------------------------------------------------------
def penalty_sweep(instances: Sequence[Instance], cfg: Config, logger: logging.Logger | None = None) -> pd.DataFrame:
    """Sweep P over multiples of the default; record validity, gap and ground-state validity."""
    logger = logger or get_logger()
    rows = []
    for inst in instances:
        opt = exact_solver(inst.problem).objective
        for m in cfg.penalty_multipliers:
            prob = with_penalty(inst.problem, m * inst.problem.P)
            res = solve_qaoa(prob, cfg, restarts=cfg.bonus_restarts, exact_optimum=opt)
            gs = qubo_ground_state(prob)
            rows.append(_qaoa_row(inst, res, opt, multiplier=m, P=prob.P, ground_state_valid=gs["valid"],
                                  most_probable_valid=res.extras["most_probable_valid"]))
            logger.info("penalty x%.2f %s: P(valid) %.2f gap %.3f ground-valid %s", m, inst.name,
                        res.p_valid, rows[-1]["gap"], gs["valid"])
    return pd.DataFrame(rows)


# 3 -------------------------------------------------------------------------
def evals_to_converge(history: Sequence[float], tol: float) -> int:
    """First evaluation (1-based) whose running minimum is within ``tol`` of the final best."""
    h = np.minimum.accumulate(np.asarray(history, dtype=float))
    if len(h) == 0:
        return 0
    return int(np.argmax(h <= h[-1] + tol) + 1)


def warm_start_study(inst: Instance, cfg: Config, report: str,
                     parser: Callable[[str, int, int], list[dict]], apply_fn: Callable[..., np.ndarray],
                     blocked_fn: Callable[[list[dict]], set[int]], n_cold: int = 3,
                     tol: float = 0.05, logger: logging.Logger | None = None) -> dict[str, Any]:
    """Solve the base instance, apply a field report, then re-solve from (a)
    random parameters and (b) the previous optimal (gamma, beta)."""
    logger = logger or get_logger()
    base = solve_qaoa(inst.problem, cfg, restarts=1)
    updates = parser(report, cfg.rows, cfg.cols)
    w_new = apply_fn(inst.w, updates, cfg.w_cap)
    new_prob = build_problem(w_new, cfg, blocked_fn(updates), name=inst.name + "_updated")
    opt = exact_solver(new_prob).objective
    warm = solve_qaoa(new_prob, cfg, restarts=1, initial_point=base.extras["optimal_point"], exact_optimum=opt)
    rng = np.random.default_rng(cfg.seed + 99)
    names = base.extras["param_names"]
    cold_runs = []
    for _ in range(n_cold):
        x0 = merge_params(names, rng.uniform(0, np.pi / 2, cfg.qaoa_reps), rng.uniform(-np.pi / 2, np.pi / 2, cfg.qaoa_reps))
        cold_runs.append(solve_qaoa(new_prob, cfg, restarts=1, initial_point=x0, exact_optimum=opt))

    def summary(r: SolverResult) -> dict[str, Any]:
        return {"evals": r.n_evals, "evals_to_converge": evals_to_converge(r.extras["history"], tol),
                "final_energy": r.extras["expected_energy"], "gap": gap(r.objective, opt),
                "p_optimal": r.p_optimal, "p_valid": r.p_valid}

    out = {"report": report, "updates": updates, "base": base, "warm": warm, "cold": cold_runs,
           "warm_summary": summary(warm), "cold_summary": [summary(c) for c in cold_runs],
           "w_before": inst.w, "w_after": w_new, "problem_after": new_prob, "tolerance": tol}
    cs = pd.DataFrame(out["cold_summary"]).mean(numeric_only=True)
    logger.info("warm start: %d evals to converge (gap %.3f) vs random init mean %.1f (gap %.3f)",
                out["warm_summary"]["evals_to_converge"], out["warm_summary"]["gap"],
                cs["evals_to_converge"], cs["gap"])
    return out


# 4 -------------------------------------------------------------------------
def optimizer_comparison(instances: Sequence[Instance], cfg: Config,
                         optimizers: Sequence[str] = ("COBYLA", "SPSA", "NELDER_MEAD"),
                         logger: logging.Logger | None = None) -> pd.DataFrame:
    """Same instances, same initial point, different classical optimisers."""
    logger = logger or get_logger()
    rows = []
    for inst in instances:
        opt = exact_solver(inst.problem).objective
        for name in optimizers:
            res = solve_qaoa(inst.problem, cfg, optimizer=name, restarts=1, exact_optimum=opt)
            rows.append(_qaoa_row(inst, res, opt, optimizer=name))
            logger.info("optimizer %s %s: gap %.3f evals %d %.1fs", name, inst.name, rows[-1]["gap"],
                        res.n_evals, res.runtime_s)
    return pd.DataFrame(rows)


# 5 -------------------------------------------------------------------------
def dicke_vector(n: int, k: int) -> np.ndarray:
    """|D^n_k>: equal superposition of all weight-k basis states."""
    keys = np.arange(2**n)
    weight = np.array([bin(int(x)).count("1") for x in keys])
    v = (weight == k).astype(complex)
    return v / np.sqrt(comb(n, k))


def ring_pairs(n: int) -> list[tuple[int, int]]:
    """Ring edges ordered as even pairs, odd pairs, then the wrap-around edge."""
    even = [(i, i + 1) for i in range(0, n - 1, 2)]
    odd = [(i, i + 1) for i in range(1, n - 1, 2)]
    return even + odd + [(n - 1, 0)]


def xy_qaoa_circuit(op, n: int, k: int, reps: int, include_state_prep: bool = False):
    """QAOA layers with a ring XY mixer, applied to the Dicke state |D^n_k>.

    Cost layer: exp(-i gamma H) built from RZ/RZZ gates of the (scaled) Ising op.
    Mixer layer: prod over ring edges of exp(-i beta (XX + YY)) = RXX(2b) RYY(2b);
    every factor conserves Hamming weight, so the state never leaves sum x = k.

    By default the circuit starts from |D^n_k> supplied as a Statevector (see
    ``solve_xy_qaoa``) because re-synthesising a generic StatePreparation on
    every evaluation is slow; ``include_state_prep=True`` prepends it as a gate
    for drawing or export.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.circuit.library import StatePreparation

    gam = ParameterVector("γ", reps)
    bet = ParameterVector("β", reps)
    qc = QuantumCircuit(n)
    if include_state_prep:
        qc.append(StatePreparation(dicke_vector(n, k), label="Dicke"), range(n))
    for layer in range(reps):
        for pauli, coeff in zip(op.paulis, op.coeffs):
            idx = np.flatnonzero(pauli.z)
            c = float(np.real(coeff))
            if len(idx) == 1:
                qc.rz(2 * c * gam[layer], int(idx[0]))
            elif len(idx) == 2:
                qc.rzz(2 * c * gam[layer], int(idx[0]), int(idx[1]))
        for i, j in ring_pairs(n):
            qc.rxx(2 * bet[layer], i, j)
            qc.ryy(2 * bet[layer], i, j)
    return qc, gam, bet


def solve_xy_qaoa(problem: PlacementProblem, cfg: Config, reps: int | None = None,
                  maxiter: int | None = None, starts: int = 2, seed: int | None = None) -> dict[str, Any]:
    """Custom variational loop (scipy COBYLA) on the XY-mixer circuit, exact statevector energies."""
    from qiskit.quantum_info import Statevector

    reps = reps or cfg.qaoa_reps
    maxiter = maxiter or cfg.maxiter
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    t0 = time.perf_counter()
    op, offset, scale = scaled_ising(problem, cfg.scale_hamiltonian)
    qc, gam, bet = xy_qaoa_circuit(op, problem.n, problem.k, reps)
    E = all_energies(problem)
    evals = 0
    history: list[float] = []

    psi0 = Statevector(dicke_vector(problem.n, problem.k))

    def probs(x: np.ndarray) -> np.ndarray:
        bind = {**{gam[i]: x[i] for i in range(reps)}, **{bet[i]: x[reps + i] for i in range(reps)}}
        return np.asarray(psi0.evolve(qc.assign_parameters(bind)).probabilities())

    def energy(x: np.ndarray) -> float:
        nonlocal evals
        evals += 1
        e = float(probs(x) @ E)
        history.append(e)
        return e

    s = (np.arange(reps) + 0.5) / reps
    inits = [np.concatenate([s * cfg.ramp_dt, sign * (1 - s) * cfg.ramp_dt]) for sign in (-1.0, 1.0)]
    inits += [np.concatenate([rng.uniform(0, 1, reps), rng.uniform(-1, 1, reps)]) for _ in range(max(0, starts - 2))]
    best = None
    for x0 in inits[:max(starts, 1)]:
        r = minimize(energy, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": cfg.cobyla_rhobeg})
        if best is None or r.fun < best.fun:
            best = r
    p = probs(best.x)
    vmask = valid_mask(problem)
    opt = float(E[vmask].min())
    om = vmask & np.isclose(E, opt, atol=1e-7)
    vk = np.flatnonzero(vmask)
    key = int(vk[np.argmax(p[vk])])
    d = decode(problem, key)
    return {"x": d["x"], "objective": d["objective"], "valid": d["valid"], "p_valid": float(p[vmask].sum()),
            "p_optimal": float(p[om].sum()), "evals": evals, "runtime_s": time.perf_counter() - t0,
            "gap": gap(d["objective"], opt), "params": best.x, "expected_energy": float(best.fun),
            "history": history, "circuit": qc, "probs": p}


def xy_mixer_study(instances: Sequence[Instance], cfg: Config, logger: logging.Logger | None = None) -> pd.DataFrame:
    """Standard X-mixer QAOA versus Dicke + XY-mixer QAOA at the same depth."""
    logger = logger or get_logger()
    rows = []
    for inst in instances:
        opt = exact_solver(inst.problem).objective
        x_res = solve_qaoa(inst.problem, cfg, restarts=cfg.bonus_restarts, exact_optimum=opt)
        rows.append(_qaoa_row(inst, x_res, opt, mixer="X (standard)"))
        xy = solve_xy_qaoa(inst.problem, cfg)
        rows.append({"instance": inst.name, "gap": xy["gap"], "p_valid": xy["p_valid"], "p_optimal": xy["p_optimal"],
                     "evals": xy["evals"], "runtime_s": xy["runtime_s"], "expected_energy": xy["expected_energy"],
                     "valid": xy["valid"], "mixer": "XY ring + Dicke"})
        logger.info("mixer %s: X P(opt) %.4f gap %.3f | XY P(opt) %.4f gap %.3f", inst.name, x_res.p_optimal,
                    rows[-2]["gap"], xy["p_optimal"], xy["gap"])
    return pd.DataFrame(rows)


# 6 -------------------------------------------------------------------------
def make_noise_model(p1: float, p2: float, readout: float):
    """Depolarising noise on 1q/2q gates plus symmetric readout error."""
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    nm = NoiseModel()
    if p1 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["sx", "x"])
    if p2 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    if readout > 0:
        nm.add_all_qubit_readout_error(ReadoutError([[1 - readout, readout], [readout, 1 - readout]]))
    return nm


def noise_study(qaoa_res: SolverResult, problem: PlacementProblem, cfg: Config,
                logger: logging.Logger | None = None) -> pd.DataFrame:
    """Re-run the trained circuit (fixed parameters) under scaled noise levels."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    logger = logger or get_logger()
    circ = qaoa_res.extras["optimal_circuit"].assign_parameters(qaoa_res.extras["optimal_parameters"])
    circ = circ.remove_final_measurements(inplace=False)
    circ.measure_all()
    tc = transpile(circ, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1, seed_transpiler=cfg.seed)
    E = all_energies(problem)
    vm = valid_mask(problem)
    opt = float(E[vm].min())
    om = vm & np.isclose(E, opt, atol=1e-7)
    rows = []
    for lvl in cfg.noise_levels:
        nm = make_noise_model(lvl * cfg.noise_p1, lvl * cfg.noise_p2, lvl * cfg.noise_readout)
        sim = AerSimulator(noise_model=nm if lvl > 0 else None, seed_simulator=cfg.seed)
        counts = sim.run(tc, shots=cfg.noise_shots).result().get_counts()
        p = np.zeros(2**problem.n)
        for k_, v in counts.items():
            p[int(k_.replace(" ", ""), 2)] += v
        p /= p.sum()
        rows.append({"level": lvl, "p1": lvl * cfg.noise_p1, "p2": lvl * cfg.noise_p2,
                     "readout": lvl * cfg.noise_readout, "p_optimal": float(p[om].sum()),
                     "p_valid": float(p[vm].sum()), "mean_energy": float(p @ E),
                     "cx_count": int(tc.count_ops().get("cx", 0)), "depth": int(tc.depth())})
        logger.info("noise x%.1f: P(opt) %.4f P(valid) %.3f", lvl, rows[-1]["p_optimal"], rows[-1]["p_valid"])
    rows_df = pd.DataFrame(rows)
    rows_df.attrs["uniform_p_optimal"] = float(om.sum() / 2**problem.n)
    return rows_df


# 7 -------------------------------------------------------------------------
def landscape_p1(problem: PlacementProblem, cfg: Config, gamma_range: tuple[float, float] = (0.0, 1.0),
                 beta_range: tuple[float, float] = (-np.pi / 4, np.pi / 4)) -> dict[str, Any]:
    """Expected QUBO energy over a (gamma, beta) grid for p = 1 (fast NumPy
    statevector, validated against Qiskit in tests) plus the path of a real
    qiskit_algorithms.QAOA p = 1 optimisation."""
    op, offset, scale = scaled_ising(problem, cfg.scale_hamiltonian)
    diag = ising_diagonal(op)
    m = cfg.landscape_points
    G = np.linspace(*gamma_range, m)
    B = np.linspace(*beta_range, m)
    Z = np.array([[numpy_qaoa_energy(diag, [g], [b]) * scale + offset for g in G] for b in B])
    res = solve_qaoa(problem, cfg, reps=1, restarts=1)
    names = res.extras["param_names"]
    path = np.array([split_params(names, p_) for p_ in res.extras["params_history"]])[:, :, 0]  # (T, 2): gamma, beta
    return {"gammas": G, "betas": B, "energy": Z, "path": path, "path_energy": np.array(res.extras["history"]),
            "optimum_point": split_params(names, res.extras["optimal_point"]),
            "grid_min": float(Z.min()), "qaoa_energy": res.extras["expected_energy"]}


# 8 -------------------------------------------------------------------------
def scaling_probe(cfg: Config, logger: logging.Logger | None = None) -> pd.DataFrame:
    """QAOA (p = 1) time per objective evaluation and statevector memory versus n."""
    logger = logger or get_logger()
    rows = []
    rng = np.random.default_rng(cfg.seed + 5)
    for r, c in cfg.scaling_grids:
        n = r * c
        if n > 16:
            continue
        k = max(1, min(cfg.k, n - 1))
        sub = cfg.with_overrides(rows=r, cols=c, k=k)
        prob = build_problem(rng.uniform(0, 1, n), sub)
        t0 = time.perf_counter()
        ex = exact_solver(prob)
        t_exact = time.perf_counter() - t0
        res = solve_qaoa(prob, sub, reps=1, restarts=1, maxiter=cfg.scaling_evals)
        per_eval = res.runtime_s / max(res.n_evals, 1)
        rows.append({"n_qubits": n, "grid": f"{r}x{c}", "k": k, "qaoa_time_s": res.runtime_s, "evals": res.n_evals,
                     "time_per_eval_s": per_eval, "statevector_bytes": 16 * 2**n,
                     "exact_time_s": t_exact, "exact_candidates": ex.n_evals, "gap": gap(res.objective, ex.objective)})
        logger.info("scaling n=%d: %.3fs/eval, statevector %.1f KiB, exact %.4fs", n, per_eval,
                    16 * 2**n / 1024, t_exact)
    return pd.DataFrame(rows)
