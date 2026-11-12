"""QUBO formulation of drone placement, Qiskit conversion, and decoding.

Variables x_i in {0,1}: x_i = 1 means a drone hovers over cell i.

    f(x) = - sum_i w_i x_i + a sum_{i<j} O_ij x_i x_j - b sum_{i<j} C_ij x_i x_j
    QUBO(x) = f(x) + P (sum_i x_i - k)^2 + P_block sum_{i in B} x_i

Blocked cells are handled with a linear penalty rather than by removing the
variable. That keeps the qubit-to-cell map fixed across agent updates, so a
warm start after a field report reuses the same parameters on the same qubits.

Qubit i corresponds to variable i (Qiskit little-endian: in a bitstring the
right-most character is qubit 0).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .config import Config
from .utils import cell_coords, cell_name


@dataclass(frozen=True)
class PlacementProblem:
    """A fully specified drone-placement instance."""

    w: np.ndarray                 # (n,) survivor likelihood weights
    rows: int
    cols: int
    k: int
    O: np.ndarray                 # (n, n) symmetric overlap matrix, zero diagonal
    C: np.ndarray                 # (n, n) symmetric radio-link matrix, zero diagonal
    a: float
    b: float
    P: float                      # equality-constraint penalty
    P_block: float                # linear penalty per blocked cell selected
    blocked: tuple[int, ...]
    base_pos: tuple[float, float]
    radio_range: float
    name: str = "instance"

    @property
    def n(self) -> int:
        return int(self.w.shape[0])


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
def overlap_matrix(rows: int, cols: int, mode: str = "neighbors8") -> np.ndarray:
    """Footprint overlap. 'neighbors8': 1 for 8-neighbours. 'decay': 1 for
    4-neighbours, 0.5 for diagonal neighbours, 0 otherwise."""
    xy = cell_coords(rows, cols)
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    if mode == "neighbors8":
        O = ((d > 0) & (d < 1.5)).astype(float)
    elif mode == "decay":
        O = np.where((d > 0) & (d < 1.01), 1.0, np.where((d > 1.01) & (d < 1.5), 0.5, 0.0))
    else:
        raise ValueError(f"unknown overlap mode {mode!r}")
    return O


def radio_matrix(rows: int, cols: int, radio_range: float) -> np.ndarray:
    """1 if two cells are within ``radio_range`` (Euclidean, cell units)."""
    xy = cell_coords(rows, cols)
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    return ((d > 0) & (d <= radio_range + 1e-9)).astype(float)


def default_penalty(w: np.ndarray, a: float, b: float, scale: float = 2.0) -> float:
    """P = scale * (max_i w_i + a + b)."""
    return float(scale * (np.max(w) + a + b))


def build_problem(w: Sequence[float] | np.ndarray, cfg: Config, blocked: Iterable[int] = (),
                  *, a: float | None = None, b: float | None = None, P: float | None = None,
                  k: int | None = None, rows: int | None = None, cols: int | None = None,
                  name: str = "instance") -> PlacementProblem:
    """Build a PlacementProblem from weights and the config defaults."""
    rows = rows or cfg.rows
    cols = cols or cfg.cols
    w = np.asarray(w, dtype=float).copy()
    if w.shape != (rows * cols,):
        raise ValueError(f"w has shape {w.shape}, expected ({rows * cols},)")
    a = cfg.a if a is None else a
    b = cfg.b if b is None else b
    k = cfg.k if k is None else k
    P = default_penalty(w, a, b, cfg.penalty_scale) if P is None else float(P)
    blocked_t = tuple(sorted({int(i) for i in blocked if 0 <= int(i) < rows * cols}))
    return PlacementProblem(
        w=w, rows=rows, cols=cols, k=k,
        O=overlap_matrix(rows, cols, cfg.overlap_mode), C=radio_matrix(rows, cols, cfg.radio_range),
        a=float(a), b=float(b), P=P, P_block=cfg.block_penalty_scale * default_penalty(w, a, b, cfg.penalty_scale),
        blocked=blocked_t, base_pos=tuple(cfg.base_pos), radio_range=cfg.radio_range, name=name,
    )


def with_penalty(problem: PlacementProblem, P: float) -> PlacementProblem:
    """Copy of ``problem`` with a different constraint penalty."""
    from dataclasses import replace

    return replace(problem, P=float(P))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def qubo_matrix(problem: PlacementProblem) -> tuple[np.ndarray, float]:
    """Upper-triangular Q and constant c with QUBO(x) = x^T Q x + c.

    Diagonal: -w_i + P(1 - 2k) + P_block [i in B]
    Off-diagonal (i<j): a O_ij - b C_ij + 2P
    Constant: P k^2
    """
    n, P, k = problem.n, problem.P, problem.k
    Q = np.triu(problem.a * problem.O - problem.b * problem.C + 2.0 * P, k=1)
    diag = -problem.w + P * (1 - 2 * k)
    for i in problem.blocked:
        diag[i] += problem.P_block
    Q[np.diag_indices(n)] = diag
    return Q, float(P * k * k)


def objective(problem: PlacementProblem, x: np.ndarray) -> float:
    """Full QUBO value (objective plus penalties) of one bit vector."""
    Q, c = qubo_matrix(problem)
    x = np.asarray(x, dtype=float)
    return float(x @ Q @ x + c)


def batch_objective(problem: PlacementProblem, X: np.ndarray) -> np.ndarray:
    """QUBO value for each row of ``X`` (m, n)."""
    Q, c = qubo_matrix(problem)
    X = np.asarray(X, dtype=float)
    return np.einsum("mi,ij,mj->m", X, Q, X) + c


def all_bitstrings(n: int) -> np.ndarray:
    """(2^n, n) array; row r holds the bits of integer r with x_i = (r >> i) & 1."""
    r = np.arange(2**n, dtype=np.int64)[:, None]
    return ((r >> np.arange(n)) & 1).astype(np.int8)


def all_energies(problem: PlacementProblem) -> np.ndarray:
    """QUBO value of every one of the 2^n states, indexed by integer key."""
    return batch_objective(problem, all_bitstrings(problem.n))


def coverage(problem: PlacementProblem, x: np.ndarray) -> float:
    """sum_i w_i x_i (survivor likelihood covered)."""
    return float(np.dot(problem.w, np.asarray(x, dtype=float)))


# ---------------------------------------------------------------------------
# Qiskit conversion
# ---------------------------------------------------------------------------
def build_quadratic_program(problem: PlacementProblem):
    """QuadraticProgram with the objective and the ``sum x = k`` constraint."""
    from qiskit_optimization import QuadraticProgram

    qp = QuadraticProgram(name=problem.name)
    names = [f"x{i}" for i in range(problem.n)]
    for nm in names:
        qp.binary_var(nm)
    linear = {names[i]: -float(problem.w[i]) for i in range(problem.n)}
    for i in problem.blocked:
        linear[names[i]] += problem.P_block
    quad: dict[tuple[str, str], float] = {}
    for i in range(problem.n):
        for j in range(i + 1, problem.n):
            v = problem.a * problem.O[i, j] - problem.b * problem.C[i, j]
            if v != 0.0:
                quad[(names[i], names[j])] = float(v)
    qp.minimize(linear=linear, quadratic=quad)
    qp.linear_constraint(linear={nm: 1 for nm in names}, sense="==", rhs=problem.k, name="k_drones")
    return qp


def to_qubo_program(problem: PlacementProblem):
    """Unconstrained QUBO QuadraticProgram via QuadraticProgramToQubo(penalty=P)."""
    from qiskit_optimization.converters import QuadraticProgramToQubo

    return QuadraticProgramToQubo(penalty=problem.P).convert(build_quadratic_program(problem))


def to_ising(problem: PlacementProblem):
    """(SparsePauliOp, offset) with QUBO(x) = <x|H|x> + offset."""
    return to_qubo_program(problem).to_ising()


# ---------------------------------------------------------------------------
# Decoding and quality metrics
# ---------------------------------------------------------------------------
def bits_from_key(key: int | str | Sequence[int] | np.ndarray, n: int) -> np.ndarray:
    """Normalise an int key, a Qiskit bitstring (qubit 0 right-most) or an array."""
    if isinstance(key, (int, np.integer)):
        return np.array([(int(key) >> i) & 1 for i in range(n)], dtype=np.int8)
    if isinstance(key, str):
        s = key.replace(" ", "")
        if len(s) != n:
            raise ValueError(f"bitstring length {len(s)} != {n}")
        return np.array([int(ch) for ch in reversed(s)], dtype=np.int8)
    arr = np.asarray(key, dtype=np.int8)
    if arr.shape != (n,):
        raise ValueError(f"bit array shape {arr.shape} != ({n},)")
    return arr


def key_from_bits(x: np.ndarray) -> int:
    """Inverse of ``bits_from_key`` for arrays."""
    return int(sum(int(v) << i for i, v in enumerate(np.asarray(x).astype(int))))


def _components(points: np.ndarray, radius: float) -> list[int]:
    """Union-find connected-component labels for points linked within ``radius``."""
    m = len(points)
    parent = list(range(m))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(m):
        for j in range(i + 1, m):
            if np.linalg.norm(points[i] - points[j]) <= radius + 1e-9:
                parent[find(i)] = find(j)
    return [find(i) for i in range(m)]


def radio_connectivity(problem: PlacementProblem, x: np.ndarray) -> tuple[int, bool]:
    """(components among selected drones, whether drones + base form one component).

    This is a post-hoc quality metric; the QUBO only rewards linked pairs and
    does not guarantee a connected relay chain back to base.
    """
    sel = np.flatnonzero(np.asarray(x))
    if len(sel) == 0:
        return 0, False
    xy = cell_coords(problem.rows, problem.cols)[sel]
    n_comp = len(set(_components(xy, problem.radio_range)))
    with_base = np.vstack([xy, np.array(problem.base_pos, dtype=float)[None]])
    connected = len(set(_components(with_base, problem.radio_range))) == 1
    return n_comp, connected


def decode(problem: PlacementProblem, bitstring: int | str | Sequence[int] | np.ndarray) -> dict:
    """Interpret a bitstring as a placement and score it."""
    x = bits_from_key(bitstring, problem.n)
    sel = np.flatnonzero(x)
    overlap = int(sum(problem.O[i, j] > 0 for a_i, i in enumerate(sel) for j in sel[a_i + 1:]))
    n_comp, connected = radio_connectivity(problem, x)
    return {
        "x": x,
        "cells": sel.tolist(),
        "cell_names": [cell_name(i, problem.cols) for i in sel],
        "objective": objective(problem, x),
        "valid": bool(x.sum() == problem.k and not set(sel.tolist()) & set(problem.blocked)),
        "count_ok": bool(x.sum() == problem.k),
        "coverage": coverage(problem, x),
        "overlap_count": overlap,
        "n_components": n_comp,
        "connected_to_base": connected,
    }


def valid_mask(problem: PlacementProblem) -> np.ndarray:
    """Boolean mask over integer keys: exactly k ones and no blocked cell."""
    bits = all_bitstrings(problem.n)
    ok = bits.sum(axis=1) == problem.k
    if problem.blocked:
        ok &= bits[:, list(problem.blocked)].sum(axis=1) == 0
    return ok
