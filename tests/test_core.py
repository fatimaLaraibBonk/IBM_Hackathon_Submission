"""Core unit tests (pytest). Also callable from the notebook via ``pytest.main``."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from floodwing.agent import apply_updates, blocked_cells, parse_report  # noqa: E402
from floodwing.config import Config  # noqa: E402
from floodwing.qubo import (bits_from_key, build_problem, decode, key_from_bits, qubo_matrix,  # noqa: E402
                            to_ising, to_qubo_program)
from floodwing.solvers import (exact_solver, greedy_solver, ising_diagonal, numpy_qaoa_energy,  # noqa: E402
                               qubo_ground_state, simulated_annealing, solve_qaoa)

CFG = Config(work_dir=str(Path(__file__).resolve().parents[1] / ".pytest_work"))


@pytest.fixture(scope="module")
def problem():
    rng = np.random.default_rng(0)
    return build_problem(rng.uniform(0, 1, CFG.n_cells), CFG, blocked=[5])


def test_qubo_matrix_matches_quadratic_program(problem):
    Q, c = qubo_matrix(problem)
    qp = to_qubo_program(problem)
    rng = np.random.default_rng(1)
    for x in rng.integers(0, 2, size=(50, problem.n)):
        assert qp.objective.evaluate(x) == pytest.approx(float(x @ Q @ x + c), abs=1e-9)


def test_ising_diagonal_matches_qubo(problem):
    op, offset = to_ising(problem)
    diag = ising_diagonal(op)
    Q, c = qubo_matrix(problem)
    for key in (0, 1, 37, 2**problem.n - 1):
        x = bits_from_key(key, problem.n).astype(float)
        assert diag[key] + offset == pytest.approx(float(x @ Q @ x + c), abs=1e-8)


def test_exact_is_valid_and_not_worse_than_greedy(problem):
    ex, gr = exact_solver(problem), greedy_solver(problem)
    assert ex.x.sum() == problem.k and ex.valid
    assert ex.objective <= gr.objective + 1e-12
    assert not set(np.flatnonzero(ex.x)) & set(problem.blocked)
    sa = simulated_annealing(problem, 2000, np.random.default_rng(0))
    assert ex.objective <= sa.objective + 1e-12


def test_default_penalty_makes_ground_state_valid(problem):
    assert qubo_ground_state(problem)["valid"]


def test_decode_flags_invalid_bitstrings(problem):
    assert not decode(problem, "0" * problem.n)["valid"]
    x = np.zeros(problem.n, dtype=int)
    x[[0, 1, 2]] = 1                        # k - 1 drones
    assert not decode(problem, x)["valid"]
    x[[3]] = 1
    assert decode(problem, x)["valid"]
    x_blocked = np.zeros(problem.n, dtype=int)
    x_blocked[[0, 1, 2, 5]] = 1             # includes blocked cell 5
    assert not decode(problem, x_blocked)["valid"]


def test_bitstring_order_round_trip(problem):
    x = np.zeros(problem.n, dtype=np.int8)
    x[[0, 3]] = 1
    key = key_from_bits(x)
    qiskit_str = format(key, f"0{problem.n}b")          # qubit 0 is the right-most character
    assert np.array_equal(bits_from_key(qiskit_str, problem.n), x)
    assert np.array_equal(bits_from_key(key, problem.n), x)


@pytest.mark.parametrize("text,expected", [
    ("Survivors on a rooftop at C4.", [("C4", "boost")]),
    ("Bridge down near A2.", [("A2", "block")]),
    ("Boat spotted at A1; sector B3 has been evacuated.", [("A1", "boost"), ("B3", "reduce")]),
])
def test_parse_report_valid(text, expected):
    got = [(u["cell"], u["action"]) for u in parse_report(text, 3, 4)]
    assert got == expected


def test_parse_report_out_of_range_and_no_cell():
    assert parse_report("Survivors spotted at D7 and Z1.", 3, 4) == []
    assert parse_report("Survivors spotted near the river.", 3, 4) == []


def test_parse_report_two_cells_one_report():
    ups = parse_report("Family stranded at A1 and boat seen at C2.", 3, 4)
    assert {(u["cell"], u["action"]) for u in ups} == {("A1", "boost"), ("C2", "boost")}


def test_apply_updates_is_pure():
    w = np.linspace(0, 1, 12)
    w_copy = w.copy()
    ups = parse_report("Survivors at A1. Bridge down at B2.", 3, 4)
    out = apply_updates(w, ups)
    assert np.array_equal(w, w_copy)
    assert out is not w
    assert out[5] == 0.0 and blocked_cells(ups) == {5}


def test_numpy_qaoa_matches_qiskit_statevector():
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.quantum_info import Statevector

    small = build_problem(np.linspace(0.1, 0.9, 6), CFG.with_overrides(rows=2, cols=3, k=2))
    op, _ = to_ising(small)
    op = (op / np.max(np.abs(op.coeffs.real))).simplify()
    ans = QAOAAnsatz(op, reps=1)
    vals = {p: (0.4 if p.name.startswith("γ") else -0.3) for p in ans.parameters}
    p = Statevector(ans.assign_parameters(vals)).probabilities()
    diag = ising_diagonal(op)
    assert float(p @ diag) == pytest.approx(numpy_qaoa_energy(diag, [0.4], [-0.3]), abs=1e-9)


def test_same_seed_same_qaoa_result():
    small_cfg = CFG.with_overrides(rows=2, cols=3, k=2, maxiter_fast=15, restarts_fast=1, shots=512,
                                   opt_shots=1024)
    prob = build_problem(np.linspace(0.1, 0.9, 6), small_cfg)
    r1 = solve_qaoa(prob, small_cfg, seed=7)
    r2 = solve_qaoa(prob, small_cfg, seed=7)
    assert np.array_equal(r1.x, r2.x)
    assert r1.extras["optimal_point"] == pytest.approx(r2.extras["optimal_point"], abs=1e-6)
    assert r1.p_optimal == pytest.approx(r2.p_optimal, abs=1e-6)
