"""Central configuration, seeding, device selection and version reporting.

Every hyperparameter lives in :class:`Config`. Modules read from a Config
instance that is passed in explicitly; nothing reads module-level mutable state.
"""
from __future__ import annotations

import importlib
import os
import platform
import random
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

# Packages whose versions are printed and pinned.
TRACKED_PACKAGES: tuple[str, ...] = (
    "numpy", "scipy", "pandas", "matplotlib", "seaborn", "sklearn", "torch",
    "qiskit", "qiskit_aer", "qiskit_algorithms", "qiskit_optimization",
    "rasterio", "gcsfs",
)


def _default_work_dir() -> str:
    """Return /kaggle/working on Kaggle, else the current directory."""
    kaggle = Path("/kaggle/working")
    return str(kaggle) if kaggle.exists() else str(Path.cwd())


@dataclass(frozen=True)
class Config:
    """All FloodWing hyperparameters. Frozen: use ``dataclasses.replace`` to change.

    Fields ending in ``_full`` / ``_fast`` are resolved through the matching
    property according to ``fast_mode``.
    """

    # ---- global -------------------------------------------------------
    seed: int = 42
    fast_mode: bool = True
    work_dir: str = field(default_factory=_default_work_dir)

    # ---- data ---------------------------------------------------------
    use_synthetic: bool = False          # True skips the Sen1Floods11 download
    n_chips_full: int = 60
    n_chips_fast: int = 24
    feature_size: int = 64               # feature tensor resolution (pixels)
    synthetic_size: int = 128            # synthetic chip resolution (multiple of feature_size)
    min_valid_frac: float = 0.05
    flood_frac_min: float = 0.02
    flood_frac_max: float = 0.90
    gcs_bucket: str = "sen1floods11"
    gcs_version: str = "v1.1"
    download_budget_s: float = 420.0     # stop downloading after this many seconds
    max_download_attempts_factor: int = 4  # try up to factor * n_chips candidates

    # ---- planning grid / QUBO ------------------------------------------
    rows: int = 3
    cols: int = 4
    k: int = 4
    base_pos: tuple[float, float] = (-1.0, 0.0)   # base station: just north of cell A1
    radio_range: float = 2.25            # cell units (see README: avoids C == O degeneracy)
    overlap_mode: str = "neighbors8"     # "neighbors8" or "decay"
    a: float = 0.6                       # overlap penalty weight
    b: float = 0.3                       # radio-link reward weight
    penalty_scale: float = 2.0           # P = penalty_scale * (max w + a + b)
    block_penalty_scale: float = 1.0     # blocked-cell linear penalty = scale * P

    # ---- proxy survivor target ----------------------------------------
    target_weights: tuple[float, float, float] = (0.45, 0.35, 0.20)  # flood, built-up, edge
    target_noise: float = 0.03
    edge_scale_px: float = 6.0           # proximity decays to 0 at this distance (64px grid)

    # ---- CNN ------------------------------------------------------------
    epochs_full: int = 30
    epochs_fast: int = 8
    batch_size: int = 8
    lr: float = 1e-3
    val_frac: float = 0.2

    # ---- QAOA -----------------------------------------------------------
    qaoa_reps: int = 2
    maxiter_full: int = 150
    maxiter_fast: int = 60
    restarts_full: int = 3
    restarts_fast: int = 1
    shots: int = 4096                    # final sample from the optimised circuit
    opt_shots: int = 8192                # shots per energy evaluation during optimisation (V2 sampler)
    scale_hamiltonian: bool = True       # divide the Ising operator by its largest |coeff|
    ramp_dt: float = 0.5                 # linear-ramp step per layer (first restart)
    cobyla_rhobeg: float = 0.15          # COBYLA initial step; the default 1.0 overshoots the basin
    cvar_alpha: float | None = None      # set e.g. 0.25 for CVaR-QAOA; None = standard expectation

    # ---- classical baselines ------------------------------------------
    n_random: int = 1000
    sa_steps: int = 3000
    sa_t0: float = 1.0
    sa_t1: float = 0.01

    # ---- evaluation ------------------------------------------------------
    n_instances_full: int = 20
    n_instances_fast: int = 8

    # ---- agent ------------------------------------------------------------
    boost_factor: float = 1.5
    reduce_factor: float = 0.5
    w_cap: float = 2.0
    use_local_llm: bool = False
    local_llm_model: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # ---- bonus experiments (each independently gated) --------------------
    run_bonus_depth: bool = True
    run_bonus_penalty: bool = True
    run_bonus_warmstart: bool = True
    run_bonus_optimizers: bool = True
    run_bonus_xy_mixer: bool = True
    run_bonus_noise: bool = True
    run_bonus_landscape: bool = True
    run_bonus_scaling: bool = True
    bonus_restarts: int = 1              # restarts inside bonus studies (bounded runtime)
    bonus_instances_full: int = 6
    bonus_instances_fast: int = 3
    depth_values_full: tuple[int, ...] = (1, 2, 3, 4)
    depth_values_fast: tuple[int, ...] = (1, 2, 3)
    penalty_multipliers: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
    noise_levels: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0)
    noise_p1: float = 0.001              # base 1-qubit depolarising
    noise_p2: float = 0.01               # base 2-qubit depolarising
    noise_readout: float = 0.01          # base readout flip probability
    noise_shots: int = 4096
    landscape_points_full: int = 30
    landscape_points_fast: int = 20
    scaling_grids: tuple[tuple[int, int], ...] = ((2, 4), (2, 5), (3, 4), (2, 7), (4, 4))
    scaling_evals: int = 5
    make_gif: bool = True
    time_budget_fast_s: float = 1500.0   # bonus experiments are skipped (and logged) past this
    time_budget_full_s: float = 5100.0

    # ---- resolution helpers -----------------------------------------------
    def __post_init__(self) -> None:
        n = self.rows * self.cols
        if n > 16:
            raise ValueError(f"rows*cols = {n} exceeds the 16-qubit limit")
        if not 1 <= self.k < n:
            raise ValueError(f"k={self.k} must be in [1, {n - 1}]")
        if self.overlap_mode not in ("neighbors8", "decay"):
            raise ValueError(f"unknown overlap_mode {self.overlap_mode!r}")
        if self.synthetic_size % self.feature_size:
            raise ValueError("synthetic_size must be a multiple of feature_size")

    @property
    def n_cells(self) -> int:
        return self.rows * self.cols

    def _pick(self, name: str) -> Any:
        return getattr(self, f"{name}_fast" if self.fast_mode else f"{name}_full")

    @property
    def n_chips(self) -> int:
        return int(self._pick("n_chips"))

    @property
    def epochs(self) -> int:
        return int(self._pick("epochs"))

    @property
    def maxiter(self) -> int:
        return int(self._pick("maxiter"))

    @property
    def restarts(self) -> int:
        return int(self._pick("restarts"))

    @property
    def n_instances(self) -> int:
        return int(self._pick("n_instances"))

    @property
    def bonus_instances(self) -> int:
        return int(self._pick("bonus_instances"))

    @property
    def depth_values(self) -> tuple[int, ...]:
        return tuple(self._pick("depth_values"))

    @property
    def landscape_points(self) -> int:
        return int(self._pick("landscape_points"))

    # ---- paths --------------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return Path(self.work_dir) / "data"

    @property
    def results_dir(self) -> Path:
        return Path(self.work_dir) / "results"

    @property
    def fig_dir(self) -> Path:
        return self.results_dir / "figures"

    @property
    def table_dir(self) -> Path:
        return self.results_dir / "tables"

    @property
    def model_dir(self) -> Path:
        return self.results_dir / "model"

    def ensure_dirs(self) -> None:
        """Create every output directory."""
        for p in (self.data_dir, self.fig_dir, self.table_dir, self.model_dir):
            p.mkdir(parents=True, exist_ok=True)

    def with_overrides(self, **kwargs: Any) -> "Config":
        """Return a copy with fields replaced."""
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(n_chips=self.n_chips, epochs=self.epochs, maxiter=self.maxiter,
                 restarts=self.restarts, n_instances=self.n_instances)
        return d


def set_seeds(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device() -> str:
    """'cuda' if a GPU is visible to PyTorch, else 'cpu'."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def package_versions() -> dict[str, str]:
    """Installed versions of the tracked packages ('missing' if absent)."""
    out: dict[str, str] = {"python": platform.python_version()}
    for name in TRACKED_PACKAGES:
        try:
            mod = importlib.import_module(name)
            out[name] = str(getattr(mod, "__version__", "unknown"))
        except Exception:  # noqa: BLE001 - any import failure means unavailable
            out[name] = "missing"
    return out


def print_versions() -> dict[str, str]:
    """Print and return package versions."""
    versions = package_versions()
    width = max(len(k) for k in versions)
    print(f"Platform: {platform.platform()} | executable: {sys.executable}")
    for k, v in versions.items():
        print(f"  {k:<{width}}  {v}")
    return versions
