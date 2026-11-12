"""QAOA (core quantum solver) plus exact, greedy, random and simulated-annealing
baselines. Every solver returns the same :class:`SolverResult` schema.
"""
from __future__ import annotations

import itertools
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .config import Config
from .qubo import (PlacementProblem, all_energies, batch_objective, bits_from_key, decode,
                   objective, to_ising, valid_mask)

warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"qiskit.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*Sampler.*")


@dataclass
class SolverResult:
    """Common result schema for every method."""

    method: str
    x: np.ndarray
    objective: float
    valid: bool
    coverage: float
    overlap: int
    connected: bool
    n_components: int
    runtime_s: float
    n_evals: int | None = None
    p_valid: float | None = None
    p_optimal: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def row(self) -> dict[str, Any]:
        """Flat dict for tables (no arrays)."""
        return {
            "method": self.method, "objective": self.objective, "valid": self.valid,
            "coverage": self.coverage, "overlap": self.overlap, "connected": self.connected,
            "n_components": self.n_components, "runtime_s": self.runtime_s, "n_evals": self.n_evals,
            "p_valid": self.p_valid, "p_optimal": self.p_optimal,
            "cells": "-".join(str(i) for i in np.flatnonzero(self.x)),
        }


def _result(method: str, problem: PlacementProblem, x: np.ndarray, runtime: float, **kw: Any) -> SolverResult:
    d = decode(problem, np.asarray(x, dtype=np.int8))
    return SolverResult(method=method, x=d["x"], objective=d["objective"], valid=d["valid"],
                        coverage=d["coverage"], overlap=d["overlap_count"], connected=d["connected_to_base"],
                        n_components=d["n_components"], runtime_s=runtime, **kw)


def _allowed_cells(problem: PlacementProblem) -> list[int]:
    allowed = [i for i in range(problem.n) if i not in problem.blocked]
    if len(allowed) < problem.k:
        raise ValueError(f"only {len(allowed)} unblocked cells for k={problem.k} drones")
    return allowed


# ---------------------------------------------------------------------------
# Classical baselines
# ---------------------------------------------------------------------------
def exact_solver(problem: PlacementProblem) -> SolverResult:
    """Brute force over all C(n_allowed, k) placements (ground-truth optimum)."""
    t0 = time.perf_counter()
    allowed = _allowed_cells(problem)
    combos = np.array(list(itertools.combinations(allowed, problem.k)))
    X = np.zeros((len(combos), problem.n), dtype=np.int8)
    X[np.arange(len(combos))[:, None], combos] = 1
    vals = batch_objective(problem, X)
    best = int(np.argmin(vals))
    n_ties = int(np.sum(np.isclose(vals, vals[best], atol=1e-9)))
    return _result("Exact", problem, X[best], time.perf_counter() - t0, n_evals=len(combos),
                   extras={"n_optimal_ties": n_ties, "n_candidates": len(combos)})


def qubo_ground_state(problem: PlacementProblem) -> dict[str, Any]:
    """Minimum of the full QUBO over all 2^n states (valid or not).

    Used to verify that the penalty P makes the constrained optimum the global one.
    """
    E = all_energies(problem)
    key = int(np.argmin(E))
    d = decode(problem, key)
    return {"key": key, "energy": float(E[key]), "valid": d["valid"], "x": d["x"]}


def numpy_eigensolver_check(problem: PlacementProblem) -> dict[str, Any]:
    """Cross-check the optimum with Qiskit's NumPyMinimumEigensolver."""
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    try:  # qiskit-optimization >= 0.7 ships its own eigensolvers
        from qiskit_optimization.minimum_eigensolvers import NumPyMinimumEigensolver
    except ImportError:
        from qiskit_algorithms import NumPyMinimumEigensolver

    from .qubo import build_quadratic_program

    t0 = time.perf_counter()
    res = MinimumEigenOptimizer(NumPyMinimumEigensolver(), penalty=problem.P).solve(
        build_quadratic_program(problem))
    x = np.round(res.x).astype(np.int8)
    return {"x": x, "objective": objective(problem, x), "runtime_s": time.perf_counter() - t0,
            "status": str(res.status)}


def greedy_solver(problem: PlacementProblem) -> SolverResult:
    """Add the cell with the best marginal objective gain, k times."""
    t0 = time.perf_counter()
    allowed = _allowed_cells(problem)
    chosen: list[int] = []
    pair = problem.a * problem.O - problem.b * problem.C
    evals = 0
    for _ in range(problem.k):
        best_i, best_delta = -1, np.inf
        for i in allowed:
            if i in chosen:
                continue
            delta = -problem.w[i] + sum(pair[i, j] for j in chosen)
            evals += 1
            if delta < best_delta - 1e-12:
                best_i, best_delta = i, delta
        chosen.append(best_i)
    x = np.zeros(problem.n, dtype=np.int8)
    x[chosen] = 1
    return _result("Greedy", problem, x, time.perf_counter() - t0, n_evals=evals)


def random_solver(problem: PlacementProblem, n_samples: int, rng: np.random.Generator) -> SolverResult:
    """Mean and std over random valid placements. ``objective`` is the MEAN;
    ``x`` is the median-objective sample (a representative placement)."""
    t0 = time.perf_counter()
    allowed = np.array(_allowed_cells(problem))
    X = np.zeros((n_samples, problem.n), dtype=np.int8)
    for r in range(n_samples):
        X[r, rng.choice(allowed, size=problem.k, replace=False)] = 1
    vals = batch_objective(problem, X)
    med = int(np.argsort(vals)[len(vals) // 2])
    res = _result("Random", problem, X[med], time.perf_counter() - t0, n_evals=n_samples,
                  extras={"objective_std": float(vals.std()), "objective_best": float(vals.min())})
    res.objective = float(vals.mean())
    return res


def simulated_annealing(problem: PlacementProblem, steps: int, rng: np.random.Generator,
                        t0: float = 1.0, t1: float = 0.01) -> SolverResult:
    """Metropolis SA on the same QUBO with swap moves that keep sum x = k."""
    start = time.perf_counter()
    allowed = np.array(_allowed_cells(problem))
    x = np.zeros(problem.n, dtype=np.int8)
    x[rng.choice(allowed, size=problem.k, replace=False)] = 1
    e = objective(problem, x)
    best_x, best_e = x.copy(), e
    temps = t0 * (t1 / t0) ** (np.arange(steps) / max(steps - 1, 1))
    for T in temps:
        on = np.flatnonzero(x)
        off = np.array([i for i in allowed if x[i] == 0])
        i, j = rng.choice(on), rng.choice(off)
        x[i], x[j] = 0, 1
        e_new = objective(problem, x)
        if e_new <= e or rng.random() < np.exp(-(e_new - e) / T):
            e = e_new
            if e < best_e - 1e-12:
                best_x, best_e = x.copy(), e
        else:
            x[i], x[j] = 1, 0
    return _result("SA", problem, best_x, time.perf_counter() - start, n_evals=steps)


# ---------------------------------------------------------------------------
# Qiskit primitives selection (runtime version detection)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PrimitiveInfo:
    name: str
    version: str          # "V1" or "V2"


def _algorithms_major_minor() -> tuple[int, int]:
    import qiskit_algorithms

    parts = qiskit_algorithms.__version__.split(".")
    return int(parts[0]), int(parts[1])


def sampler_api() -> str:
    """'V2' for qiskit-algorithms >= 0.4 (BaseSamplerV2), else 'V1'."""
    return "V2" if _algorithms_major_minor() >= (0, 4) else "V1"


def make_sampler(shots: int, seed: int | None = None):
    """Return (sampler, transpiler_or_None, PrimitiveInfo) matching the installed
    qiskit-algorithms. V2: Aer SamplerV2 (+ preset pass manager), falling back
    to qiskit.primitives.StatevectorSampler. V1: Aer V1 Sampler, falling back to
    qiskit.primitives.Sampler.
    """
    if sampler_api() == "V2":
        try:
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
            from qiskit_aer import AerSimulator
            from qiskit_aer.primitives import SamplerV2 as AerSamplerV2

            pm = generate_preset_pass_manager(optimization_level=1, backend=AerSimulator(method="statevector"))
            return (AerSamplerV2(default_shots=shots, seed=seed), pm,
                    PrimitiveInfo(f"qiskit_aer.primitives.SamplerV2 ({shots} shots)", "V2"))
        except Exception:  # noqa: BLE001
            from qiskit.primitives import StatevectorSampler

            return (StatevectorSampler(default_shots=shots, seed=seed), None,
                    PrimitiveInfo(f"qiskit.primitives.StatevectorSampler ({shots} shots)", "V2"))
    try:
        from qiskit_aer.primitives import Sampler as AerSampler

        return (AerSampler(run_options={"shots": shots, "seed": seed}), None,
                PrimitiveInfo(f"qiskit_aer.primitives.Sampler (V1, {shots} shots)", "V1"))
    except Exception:  # noqa: BLE001
        from qiskit.primitives import Sampler

        return (Sampler(options={"shots": shots, "seed": seed}), None,
                PrimitiveInfo(f"qiskit.primitives.Sampler (V1 reference, {shots} shots)", "V1"))


def describe_primitives(cfg: Config) -> str:
    """Human-readable description of the primitives this run will use."""
    _, pm, info = make_sampler(cfg.opt_shots, cfg.seed)
    tp = "preset pass manager (AerSimulator)" if pm is not None else "none"
    return (f"qiskit-algorithms sampler API: {info.version} | sampler: {info.name} | transpiler: {tp} | "
            f"final sample: {cfg.shots} shots | exact probabilities: qiskit.quantum_info.Statevector")


def make_optimizer(name: str, maxiter: int, rhobeg: float = 0.15):
    """qiskit_algorithms optimizer by name: COBYLA, SPSA or NELDER_MEAD."""
    from qiskit_algorithms import optimizers as opt

    name = name.upper()
    if name == "COBYLA":
        return opt.COBYLA(maxiter=maxiter, rhobeg=rhobeg)
    if name == "SPSA":
        return opt.SPSA(maxiter=maxiter)
    if name in ("NELDER_MEAD", "NELDER-MEAD", "NM"):
        return opt.NELDER_MEAD(maxiter=maxiter, maxfev=maxiter)
    raise ValueError(f"unknown optimizer {name!r}")


# ---------------------------------------------------------------------------
# QAOA
# ---------------------------------------------------------------------------
def scaled_ising(problem: PlacementProblem, scale_hamiltonian: bool = True):
    """(operator, offset, scale) where QUBO(x) = scale * <x|op|x> + offset.

    Dividing by the largest |coefficient| does not change the ground state but
    keeps sensible (gamma, beta) ranges comparable across instances.
    """
    op, offset = to_ising(problem)
    scale = float(np.max(np.abs(op.coeffs.real))) if scale_hamiltonian else 1.0
    return (op / scale).simplify(), float(offset), scale


def parameter_layout(op, reps: int) -> list[str]:
    """Parameter names in the order QAOAAnsatz (and therefore QAOA) uses."""
    from qiskit.circuit.library import QAOAAnsatz

    return [p.name for p in QAOAAnsatz(op, reps=reps).parameters]


def _is_gamma(name: str) -> bool:
    return name.startswith("γ") or "gamma" in name.lower()


def split_params(names: Sequence[str], values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """(gammas, betas) from a flat parameter vector in QAOAAnsatz order."""
    g = [v for n, v in zip(names, values) if _is_gamma(n)]
    b = [v for n, v in zip(names, values) if not _is_gamma(n)]
    return np.array(g, dtype=float), np.array(b, dtype=float)


def merge_params(names: Sequence[str], gammas: Sequence[float], betas: Sequence[float]) -> np.ndarray:
    """Flat vector in QAOAAnsatz order from (gammas, betas)."""
    gi, bi = iter(gammas), iter(betas)
    return np.array([next(gi) if _is_gamma(n) else next(bi) for n in names], dtype=float)


def ramp_initial_point(names: Sequence[str], reps: int, dt: float) -> np.ndarray:
    """Linear-ramp (Trotterised annealing) initial point.

    With cost layer exp(-i gamma H_C) and mixer exp(-i beta sum X), annealing
    from the |+> ground state of -sum X gives gamma_l = s_l dt and
    beta_l = -(1 - s_l) dt with s_l = (l + 1/2) / p. The negative beta matters:
    positive-beta ramps start in the wrong basin.
    """
    s = (np.arange(reps) + 0.5) / reps
    return merge_params(names, s * dt, -(1 - s) * dt)


def exact_probabilities(circuit, params: dict) -> np.ndarray:
    """Exact |amplitude|^2 for every integer key (qubit 0 = least significant bit)."""
    from qiskit.quantum_info import Statevector

    bound = circuit.assign_parameters(params).remove_final_measurements(inplace=False)
    return np.asarray(Statevector(bound).probabilities(), dtype=float)


def _counts_to_vector(dist: dict, n: int) -> np.ndarray:
    """Probability vector from a dict keyed by int or Qiskit bitstring."""
    p = np.zeros(2**n)
    for key, v in dist.items():
        k = int(key, 2) if isinstance(key, str) else int(key)
        p[k] += max(float(v), 0.0)
    s = p.sum()
    return p / s if s > 0 else p


def sample_circuit(circuit, params: dict, n: int, shots: int, seed: int) -> tuple[np.ndarray, str]:
    """Shot sample of the bound circuit as a probability vector over integer keys."""
    bound = circuit.assign_parameters(params).remove_final_measurements(inplace=False)
    bound.measure_all()
    sampler, pm, info = make_sampler(shots, seed)
    if info.version == "V2":
        if pm is not None:
            bound = pm.run(bound)
        counts = sampler.run([bound]).result()[0].data.meas.get_counts()
        return _counts_to_vector(counts, n), info.name
    quasi = sampler.run([bound]).result().quasi_dists[0]
    return _counts_to_vector(dict(quasi), n), info.name


def solve_qaoa(problem: PlacementProblem, cfg: Config, *, reps: int | None = None, maxiter: int | None = None,
               restarts: int | None = None, optimizer: str = "COBYLA",
               initial_point: Sequence[float] | None = None, seed: int | None = None,
               shots: int | None = None, method_name: str = "QAOA",
               exact_optimum: float | None = None) -> SolverResult:
    """Run QAOA (qiskit_algorithms.QAOA) and decode a placement.

    Decoding (three views of the same optimised circuit):

    * ``QAOA`` (returned): highest-probability valid bitstring of the exact
      output distribution (statevector, simulator-only). If no valid state has
      non-zero probability, the most probable bitstring (flagged invalid).
    * ``extras['shots_x']``: most frequent valid bitstring in the final
      ``shots``-shot sample (what hardware would give); falls back to the
      best-energy valid sample on a tie-free basis, then to the most frequent.
    * ``extras['best_sampled_x']``: lowest-objective valid bitstring among
      the shots ("QAOA-best-sample").

    P(valid) and P(optimal) are exact at the optimised parameters; the
    shot-based versions are in ``extras``.
    """
    from qiskit_algorithms import QAOA

    reps = reps or cfg.qaoa_reps
    maxiter = maxiter or cfg.maxiter
    restarts = restarts or cfg.restarts
    shots = shots or cfg.shots
    seed = cfg.seed if seed is None else seed
    rng = np.random.default_rng(seed)

    t0 = time.perf_counter()
    op, offset, scale = scaled_ising(problem, cfg.scale_hamiltonian)
    names = parameter_layout(op, reps)
    sampler, pm, info = make_sampler(cfg.opt_shots, seed)

    best: dict[str, Any] | None = None
    all_histories: list[list[float]] = []
    total_evals = 0
    for r in range(restarts):
        if r == 0 and initial_point is not None:
            x0 = np.asarray(initial_point, dtype=float)
        elif r == 0:
            x0 = ramp_initial_point(names, reps, cfg.ramp_dt)
        else:
            x0 = ramp_initial_point(names, reps, rng.uniform(0.3, 1.3)) + rng.normal(0, 0.05, len(names))
        history: list[float] = []
        params_hist: list[np.ndarray] = []

        def callback(count: int, params: np.ndarray, mean: float, meta: Any,
                     _h: list = history, _p: list = params_hist) -> None:
            _h.append(float(np.real(mean)) * scale + offset)
            _p.append(np.array(params, dtype=float))

        kwargs: dict[str, Any] = {"aggregation": cfg.cvar_alpha} if cfg.cvar_alpha else {}
        if info.version == "V2" and pm is not None:
            kwargs["transpiler"] = pm
        qaoa = QAOA(sampler, make_optimizer(optimizer, maxiter, cfg.cobyla_rhobeg), reps=reps,
                    initial_point=x0, callback=callback, **kwargs)
        res = qaoa.compute_minimum_eigenvalue(op)
        total_evals += int(res.cost_function_evals or len(history))
        all_histories.append(history)
        # Rank restarts by the exact energy at their optimum (removes shot noise).
        params = dict(res.optimal_parameters)
        p_ex = exact_probabilities(res.optimal_circuit, params)
        energy = float(p_ex @ all_energies(problem))
        if best is None or energy < best["energy"]:
            best = {"energy": energy, "res": res, "params": params, "p_exact": p_ex, "history": history,
                    "params_hist": params_hist, "x0": x0, "restart": r}

    assert best is not None
    res = best["res"]
    p_exact = best["p_exact"]
    opt_point = np.array([best["params"][p] for p in res.optimal_circuit.parameters], dtype=float)
    p_shots, shot_name = sample_circuit(res.optimal_circuit, best["params"], problem.n, shots, seed)

    vmask = valid_mask(problem)
    E = all_energies(problem)
    opt_val = float(E[vmask].min()) if exact_optimum is None else float(exact_optimum)
    opt_mask = vmask & np.isclose(E, opt_val, atol=1e-7)

    valid_keys = np.flatnonzero(vmask)
    if p_exact[valid_keys].max() > 0:
        key = int(valid_keys[np.argmax(p_exact[valid_keys])])
        decode_rule = "most_probable_valid (exact distribution)"
    else:
        key = int(np.argmax(p_exact))
        decode_rule = "most_probable_overall (no valid probability mass)"
    sampled = np.flatnonzero(p_shots > 0)
    sampled_valid = sampled[vmask[sampled]]
    if len(sampled_valid):
        top = p_shots[sampled_valid].max()
        ties = sampled_valid[np.isclose(p_shots[sampled_valid], top)]
        shots_key = int(ties[np.argmin(E[ties])])          # break count ties by objective
        best_sample_key = int(sampled_valid[np.argmin(E[sampled_valid])])
    else:
        shots_key = best_sample_key = int(np.argmax(p_shots))

    runtime = time.perf_counter() - t0
    gammas, betas = split_params(names, opt_point)
    result = _result(method_name, problem, bits_from_key(key, problem.n), runtime, n_evals=total_evals,
                     p_valid=float(p_exact[vmask].sum()), p_optimal=float(p_exact[opt_mask].sum()))
    result.extras.update({
        "decode_rule": decode_rule,
        "shots_x": bits_from_key(shots_key, problem.n),
        "best_sampled_x": bits_from_key(best_sample_key, problem.n),
        "best_sampled_objective": float(E[best_sample_key]),
        "p_valid_shots": float(p_shots[vmask].sum()),
        "p_optimal_shots": float(p_shots[opt_mask].sum()),
        "p_exact": p_exact, "p_shots": p_shots,
        "optimal_point": opt_point, "param_names": names, "gammas": gammas, "betas": betas,
        "history": best["history"], "all_histories": all_histories,
        "params_history": best["params_hist"], "initial_point": best["x0"],
        "expected_energy": best["energy"], "scale": scale, "offset": offset,
        "reps": reps, "optimizer": optimizer, "restarts": restarts, "best_restart": best["restart"],
        "primitive": info.name, "shot_primitive": shot_name, "n_pauli_terms": len(op),
        "most_probable_key": int(np.argmax(p_exact)),
        "most_probable_valid": bool(vmask[int(np.argmax(p_exact))]),
        "optimal_circuit": res.optimal_circuit, "optimal_parameters": best["params"],
    })
    return result


def qaoa_variants(qaoa_res: SolverResult, problem: PlacementProblem) -> list[SolverResult]:
    """'QAOA-shots' and 'QAOA-best-sample' results derived from one QAOA run.

    QAOA-best-sample picks the best of 4096 shots over only C(12,4)=495 valid
    states, which is close to brute force; it is reported for completeness and
    is not a fair measure of the QAOA state quality on its own.
    """
    out = []
    for name, key in (("QAOA-shots", "shots_x"), ("QAOA-best-sample", "best_sampled_x")):
        out.append(_result(name, problem, qaoa_res.extras[key], qaoa_res.runtime_s, n_evals=qaoa_res.n_evals,
                           p_valid=qaoa_res.extras["p_valid_shots"], p_optimal=qaoa_res.extras["p_optimal_shots"]))
    return out


# ---------------------------------------------------------------------------
# Fast NumPy statevector QAOA (validated against Qiskit in tests); used for
# dense parameter scans such as the (gamma, beta) landscape.
# ---------------------------------------------------------------------------
def ising_diagonal(op) -> np.ndarray:
    """Diagonal of a Z-only SparsePauliOp in the computational basis (index = int key)."""
    n = op.num_qubits
    bits = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(np.int8)
    zsign = 1 - 2 * bits                      # +1 for |0>, -1 for |1>
    diag = np.zeros(2**n)
    for pauli, coeff in zip(op.paulis, op.coeffs):
        if pauli.x.any():
            raise ValueError("operator is not diagonal")
        idx = np.flatnonzero(pauli.z)
        diag += np.real(coeff) * (np.prod(zsign[:, idx], axis=1) if len(idx) else 1.0)
    return diag


def numpy_qaoa_state(diag: np.ndarray, gammas: Sequence[float], betas: Sequence[float]) -> np.ndarray:
    """|psi(gamma, beta)> for the standard X-mixer QAOA with cost diagonal ``diag``."""
    N = diag.shape[0]
    n = int(np.log2(N))
    psi = np.full(N, 1 / np.sqrt(N), dtype=complex)
    for g, b in zip(gammas, betas):
        psi = psi * np.exp(-1j * g * diag)
        c, s = np.cos(b), -1j * np.sin(b)
        t = psi.reshape((2,) * n)
        for q in range(n):
            ax = n - 1 - q                     # qubit q is the least-significant bit
            t0, t1 = np.take(t, 0, axis=ax), np.take(t, 1, axis=ax)
            t = np.stack([c * t0 + s * t1, s * t0 + c * t1], axis=ax)
        psi = t.reshape(N)
    return psi


def numpy_qaoa_energy(diag: np.ndarray, gammas: Sequence[float], betas: Sequence[float]) -> float:
    psi = numpy_qaoa_state(diag, gammas, betas)
    return float(np.real(np.vdot(psi, diag * psi)))


def run_all_classical(problem: PlacementProblem, cfg: Config, rng: np.random.Generator) -> list[SolverResult]:
    """Exact, Greedy, Random and SA on one instance."""
    return [exact_solver(problem), greedy_solver(problem), random_solver(problem, cfg.n_random, rng),
            simulated_annealing(problem, cfg.sa_steps, rng, cfg.sa_t0, cfg.sa_t1)]

