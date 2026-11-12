"""Sen1Floods11 subset download, synthetic fallback, and feature engineering.

Output of this module (``FloodDataset``) is identical whether the chips came
from Sen1Floods11 or from the procedural generator, so every downstream stage
runs unchanged.

Survivor likelihood is NOT in Sen1Floods11. ``proxy_target`` defines a
hand-written proxy per grid cell; the CNN therefore learns a known function
plus noise. This is stated in the README, the report and the limitations.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from scipy import ndimage

from .config import Config
from .utils import get_logger

# Sentinel-2 band order in Sen1Floods11 S2Hand chips (13 bands, 0-based index).
S2_BANDS: tuple[str, ...] = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A",
                             "B9", "B10", "B11", "B12")
B2, B3, B4, B8, B11 = (S2_BANDS.index(b) for b in ("B2", "B3", "B4", "B8", "B11"))
FEATURE_NAMES: tuple[str, ...] = ("water", "ndvi", "built_up", "edge_proximity")
EPS = 1e-6


@dataclass(frozen=True)
class RawChip:
    """One flood chip before feature engineering."""

    chip_id: str
    s2: np.ndarray       # (13, H, W) float32 reflectance-like values
    label: np.ndarray    # (H, W) int16: -1 invalid, 0 non-water, 1 water
    source: str          # "sen1floods11" or "synthetic"


@dataclass(frozen=True)
class FloodDataset:
    """Feature tensors and proxy targets for every kept chip."""

    X: np.ndarray            # (N, 4, F, F) float32 input channels
    Y: np.ndarray            # (N, n_cells) float32 proxy survivor likelihood
    valid: np.ndarray        # (N, F, F) float32 fraction of valid label pixels
    rgb: np.ndarray          # (N, F, F, 3) float32 display composite in [0, 1]
    cell_feats: np.ndarray   # (N, 3, n_cells): flood_frac, built_up, edge_prox per cell
    chip_ids: tuple[str, ...]
    sources: tuple[str, ...]
    rows: int
    cols: int

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def source_summary(self) -> str:
        uniq, counts = np.unique(np.array(self.sources), return_counts=True)
        return ", ".join(f"{u}: {c}" for u, c in zip(uniq, counts))


# ---------------------------------------------------------------------------
# Generic array helpers
# ---------------------------------------------------------------------------
def block_mean(arr: np.ndarray, out: int) -> np.ndarray:
    """Average-pool the last two axes of ``arr`` down to ``out x out``.

    Crops to a multiple of ``out`` first, so any input size works.
    """
    h, w = arr.shape[-2:]
    fh, fw = h // out, w // out
    if fh < 1 or fw < 1:
        raise ValueError(f"cannot downsample {h}x{w} to {out}x{out}")
    a = arr[..., : fh * out, : fw * out]
    new_shape = a.shape[:-2] + (out, fh, out, fw)
    return a.reshape(new_shape).mean(axis=(-3, -1))


def adaptive_avg_pool2d(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """NumPy port of torch.nn.AdaptiveAvgPool2d on the last two axes."""
    h, w = arr.shape[-2:]
    out = np.empty(arr.shape[:-2] + (out_h, out_w), dtype=np.float64)
    for i in range(out_h):
        r0, r1 = (i * h) // out_h, -(-((i + 1) * h) // out_h)
        for j in range(out_w):
            c0, c1 = (j * w) // out_w, -(-((j + 1) * w) // out_w)
            out[..., i, j] = arr[..., r0:r1, c0:c1].mean(axis=(-2, -1))
    return out


# ---------------------------------------------------------------------------
# Feature engineering (shared by real and synthetic chips)
# ---------------------------------------------------------------------------
def chip_passes_filter(label: np.ndarray, cfg: Config) -> tuple[bool, float, float]:
    """Return (keep, valid_fraction, flood_fraction_of_valid)."""
    valid = label >= 0
    valid_frac = float(valid.mean())
    flood_frac = float((label == 1).sum() / max(valid.sum(), 1))
    keep = valid_frac >= cfg.min_valid_frac and cfg.flood_frac_min <= flood_frac <= cfg.flood_frac_max
    return keep, valid_frac, flood_frac


def compute_features(chip: RawChip, cfg: Config) -> dict[str, np.ndarray]:
    """Turn one raw chip into a (4, F, F) feature tensor plus display data.

    Channels: water fraction (invalid pixels count as non-water), NDVI in
    [0, 1], NDBI-like built-up proxy in [0, 1], flood-edge proximity in [0, 1].
    """
    F = cfg.feature_size
    s2 = np.nan_to_num(chip.s2.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    label = chip.label
    valid_px = (label >= 0).astype(np.float32)
    water_px = (label == 1).astype(np.float32)

    ndvi = (s2[B8] - s2[B4]) / (s2[B8] + s2[B4] + EPS)
    ndbi = (s2[B11] - s2[B8]) / (s2[B11] + s2[B8] + EPS)
    ndvi01 = np.clip((ndvi + 1.0) / 2.0, 0.0, 1.0)
    built01 = np.clip((ndbi + 1.0) / 2.0, 0.0, 1.0)

    water = block_mean(water_px, F)
    valid = block_mean(valid_px, F)
    ndvi_f = block_mean(ndvi01, F)
    built_f = block_mean(built01, F)

    # Flood-edge proximity: distance (in feature pixels) to the water/land boundary.
    wmask = water > 0.5
    if wmask.any() and (~wmask).any():
        d_out = ndimage.distance_transform_edt(~wmask)   # land pixels: distance to water
        d_in = ndimage.distance_transform_edt(wmask)     # water pixels: distance to land
        dist = np.where(wmask, d_in, d_out)
        edge = np.clip(1.0 - dist / cfg.edge_scale_px, 0.0, 1.0)
    else:
        edge = np.zeros_like(water)

    rgb = np.stack([s2[B4], s2[B3], s2[B2]], axis=-1)
    rgb = block_mean(np.moveaxis(rgb, -1, 0), F)
    rgb = np.moveaxis(rgb, 0, -1)
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    rgb = np.clip((rgb - lo) / max(hi - lo, EPS), 0.0, 1.0)

    feats = np.stack([water, ndvi_f, built_f, edge]).astype(np.float32)
    return {"features": feats, "valid": valid.astype(np.float32), "rgb": rgb.astype(np.float32)}


def cell_aggregates(features: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """(3, n_cells): flood fraction, built-up, edge proximity per planning cell."""
    pooled = adaptive_avg_pool2d(features[[0, 2, 3]], rows, cols)   # (3, rows, cols)
    return pooled.reshape(3, rows * cols).astype(np.float32)


def proxy_target(cell_feats: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """PROXY survivor likelihood per cell (hand-defined, not real survivor data).

    target_i = clip(0.45*flood_i + 0.35*built_i + 0.20*edge_i + noise, 0, 1).
    The weights sum to 1 and every feature is in [0, 1], so the fixed global
    normalisation is the identity followed by clipping.
    """
    wf, wb, we = cfg.target_weights
    t = wf * cell_feats[0] + wb * cell_feats[1] + we * cell_feats[2]
    t = t + rng.normal(0.0, cfg.target_noise, size=t.shape)
    return np.clip(t, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Synthetic generator (mandatory fallback)
# ---------------------------------------------------------------------------
# Reflectance signatures (B2, B3, B4, B8, B11) by surface class.
_SIGNATURES = {
    "water": (0.06, 0.05, 0.03, 0.02, 0.01),
    "veg": (0.04, 0.07, 0.04, 0.38, 0.17),
    "built": (0.12, 0.13, 0.15, 0.20, 0.32),
    "bare": (0.10, 0.12, 0.14, 0.22, 0.27),
}


def generate_synthetic_chip(rng: np.random.Generator, cfg: Config, idx: int) -> RawChip:
    """Procedural flood chip with the same schema as a Sen1Floods11 chip.

    Smooth random elevation (Gaussian-filtered noise plus a tilt), flood mask
    from ``elevation < threshold``, Gaussian building-density blobs on land,
    vegetation elsewhere, and a few invalid (cloud) blobs in the label.
    """
    S = cfg.synthetic_size
    yy, xx = np.mgrid[0:S, 0:S] / S
    for _ in range(50):
        noise = ndimage.gaussian_filter(rng.normal(size=(S, S)), sigma=S / 10)
        noise = (noise - noise.mean()) / (noise.std() + EPS)
        angle = rng.uniform(0, 2 * np.pi)
        tilt = np.cos(angle) * xx + np.sin(angle) * yy
        elev = noise + rng.uniform(0.5, 2.0) * tilt
        thr = np.quantile(elev, rng.uniform(0.15, 0.55))
        water = elev < thr

        built = np.zeros((S, S))
        for _ in range(int(rng.integers(2, 7))):
            cy, cx = rng.uniform(0, 1, size=2)
            sig = rng.uniform(0.04, 0.12)
            built += rng.uniform(0.5, 1.0) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sig**2))
        built = np.clip(built, 0, 1)
        # Buildings partly submerged near the shoreline are what make edges interesting.
        built *= np.where(water, 0.35, 1.0)

        label = water.astype(np.int16)
        cloud = ndimage.gaussian_filter(rng.normal(size=(S, S)), sigma=S / 16)
        label[cloud > np.quantile(cloud, 1 - rng.uniform(0.0, 0.08))] = -1

        keep, _, _ = chip_passes_filter(label, cfg)
        if keep:
            break

    sig = {k: np.array(v)[:, None, None] for k, v in _SIGNATURES.items()}
    veg_frac = (1 - built) * rng.uniform(0.5, 0.9)
    bare_frac = 1 - built - veg_frac
    land = sig["built"] * built + sig["veg"] * veg_frac + sig["bare"] * bare_frac
    five = np.where(water[None], sig["water"], land)
    five = five * (1 + 0.05 * rng.normal(size=five.shape))
    s2 = np.zeros((13, S, S), dtype=np.float32)
    for band_i, src_i in zip((B2, B3, B4, B8, B11), range(5)):
        s2[band_i] = five[src_i]
    # Fill the other bands with plausible blends so the schema is complete.
    for band_i in range(13):
        if band_i not in (B2, B3, B4, B8, B11):
            s2[band_i] = 0.5 * (s2[B4] + s2[B8])
    s2 = np.clip(s2, 0, None) * 10000.0
    return RawChip(chip_id=f"synthetic_{idx:03d}", s2=s2.astype(np.float32), label=label, source="synthetic")


def iter_synthetic_chips(cfg: Config, n: int, start_idx: int = 0) -> Iterator[RawChip]:
    """Yield ``n`` filtered synthetic chips (deterministic given cfg.seed)."""
    rng = np.random.default_rng(cfg.seed + 1000 + start_idx)
    for i in range(n):
        yield generate_synthetic_chip(rng, cfg, start_idx + i)


# ---------------------------------------------------------------------------
# Sen1Floods11 download (anonymous GCS)
# ---------------------------------------------------------------------------
_PREFERRED_SUBPATH = ("data", "flood_events", "HandLabeled")
_SKIP_DIRS = {"catalog", "checkpoints", "WeaklyLabeled", "S1Hand", "JRCWaterHand", "S1OtsuLabelHand"}


def _find_dir(fs, root: str, target: str, logger: logging.Logger, max_depth: int = 5) -> str | None:
    """Locate a directory named ``target`` under ``root`` (bounded, verified at runtime)."""
    root = root.rstrip("/")
    guess = "/".join((root,) + _PREFERRED_SUBPATH + (target,))
    try:
        if fs.isdir(guess):
            return guess
    except Exception as exc:  # noqa: BLE001
        logger.info("guess %s not verified (%s)", guess, exc)
    logger.info("searching %s for '%s' (guess path missing)", root, target)
    stack: list[tuple[str, int]] = [(root, 0)]
    while stack:
        path, depth = stack.pop()
        try:
            entries = fs.ls(path, detail=True)
        except Exception as exc:  # noqa: BLE001
            logger.info("cannot list %s: %s", path, exc)
            continue
        for e in entries:
            if e.get("type") != "directory":
                continue
            name = e["name"].rstrip("/").split("/")[-1]
            if name == target:
                return e["name"].rstrip("/")
            if depth + 1 < max_depth and name not in _SKIP_DIRS:
                stack.append((e["name"], depth + 1))
    return None


def _read_tif(path: Path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as src:
        return src.read()


def bucket_reachable(cfg: Config, timeout_s: float = 15.0) -> tuple[bool, str]:
    """Fast anonymous probe of the public bucket through the GCS JSON API.

    gcsfs retries with long back-offs when the network is blocked, so the
    pipeline checks reachability first and falls back to synthetic data in
    seconds rather than minutes.
    """
    import json
    import urllib.request

    url = (f"https://storage.googleapis.com/storage/v1/b/{cfg.gcs_bucket}/o"
           f"?prefix={cfg.gcs_version}/&delimiter=/&maxResults=10")
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:   # noqa: S310 - fixed https URL
            body = json.loads(resp.read().decode())
        prefixes = body.get("prefixes", [])
        return bool(prefixes or body.get("items")), f"top-level prefixes: {prefixes}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def iter_sen1floods11_chips(cfg: Config, n: int, logger: logging.Logger) -> Iterator[RawChip]:
    """Yield up to ``n`` filtered Sen1Floods11 hand-labelled chips.

    Paths are discovered by listing the public bucket anonymously; nothing is
    assumed without verification. Downloads are cached in ``cfg.data_dir``.
    Raises on fatal problems so the caller can fall back to synthetic data.
    """
    ok, detail = bucket_reachable(cfg)
    if not ok:
        raise ConnectionError(f"gs://{cfg.gcs_bucket} not reachable ({detail})")
    logger.info("bucket reachable: %s", detail)
    import gcsfs

    fs = gcsfs.GCSFileSystem(token="anon", timeout=60)
    root = f"{cfg.gcs_bucket}/{cfg.gcs_version}"
    logger.info("bucket listing %s: %s", root, [p.split("/")[-1] for p in fs.ls(root)])
    s2_dir = _find_dir(fs, root, "S2Hand", logger)
    lbl_dir = _find_dir(fs, root, "LabelHand", logger)
    if not s2_dir or not lbl_dir:
        raise FileNotFoundError(f"S2Hand/LabelHand not found under gs://{root}")
    logger.info("verified S2Hand=%s LabelHand=%s", s2_dir, lbl_dir)

    s2_files = {p.split("/")[-1] for p in fs.ls(s2_dir) if p.endswith(".tif")}
    lbl_files = sorted(p.split("/")[-1] for p in fs.ls(lbl_dir) if p.endswith("_LabelHand.tif"))
    pairs = [(lf.replace("_LabelHand.tif", ""), lf) for lf in lbl_files
             if lf.replace("_LabelHand.tif", "_S2Hand.tif") in s2_files]
    if not pairs:
        raise FileNotFoundError("no matching S2Hand/LabelHand pairs")
    logger.info("found %d hand-labelled chip pairs", len(pairs))
    order = np.random.default_rng(cfg.seed).permutation(len(pairs))

    local = cfg.data_dir / "sen1floods11"
    local.mkdir(parents=True, exist_ok=True)
    kept, tried, t0 = 0, 0, time.perf_counter()
    for idx in order:
        if kept >= n or tried >= cfg.max_download_attempts_factor * n:
            break
        if time.perf_counter() - t0 > cfg.download_budget_s:
            logger.warning("download budget %.0fs exhausted after %d chips", cfg.download_budget_s, kept)
            break
        chip_id, lbl_name = pairs[idx]
        tried += 1
        lbl_local = local / lbl_name
        s2_local = local / f"{chip_id}_S2Hand.tif"
        try:
            if not lbl_local.exists():
                fs.get(f"{lbl_dir}/{lbl_name}", str(lbl_local))
            label = _read_tif(lbl_local)[0].astype(np.int16)
            keep, vf, ff = chip_passes_filter(label, cfg)
            if not keep:
                logger.debug("skip %s (valid %.2f, flood %.2f)", chip_id, vf, ff)
                continue
            if not s2_local.exists():
                fs.get(f"{s2_dir}/{chip_id}_S2Hand.tif", str(s2_local))
            s2 = _read_tif(s2_local).astype(np.float32)
        except Exception as exc:  # noqa: BLE001 - one bad chip must not kill the run
            logger.warning("chip %s failed: %s", chip_id, exc)
            continue
        if s2.shape[0] != 13:
            logger.warning("chip %s has %d bands, expected 13; skipped", chip_id, s2.shape[0])
            continue
        kept += 1
        yield RawChip(chip_id=chip_id, s2=s2, label=label, source="sen1floods11")


# ---------------------------------------------------------------------------
# Dataset assembly with caching
# ---------------------------------------------------------------------------
def _cache_path(cfg: Config) -> Path:
    tag = "syn" if cfg.use_synthetic else "auto"
    return cfg.data_dir / f"features_{tag}_n{cfg.n_chips}_f{cfg.feature_size}_g{cfg.rows}x{cfg.cols}_s{cfg.seed}.npz"


def build_dataset(cfg: Config, logger: logging.Logger | None = None, use_cache: bool = True) -> FloodDataset:
    """Download (or synthesise) chips, engineer features and proxy targets.

    Never raises because of the network: any download problem logs a warning
    and the remainder is filled with synthetic chips.
    """
    logger = logger or get_logger()
    cfg.ensure_dirs()
    cache = _cache_path(cfg)
    if use_cache and cache.exists():
        z = np.load(cache, allow_pickle=False)
        logger.info("loaded cached features from %s", cache)
        return FloodDataset(X=z["X"], Y=z["Y"], valid=z["valid"], rgb=z["rgb"], cell_feats=z["cell_feats"],
                            chip_ids=tuple(z["chip_ids"].tolist()), sources=tuple(z["sources"].tolist()),
                            rows=cfg.rows, cols=cfg.cols)

    records: list[tuple[str, str, dict[str, np.ndarray]]] = []   # (chip_id, source, features)
    if not cfg.use_synthetic:
        try:
            for chip in iter_sen1floods11_chips(cfg, cfg.n_chips, logger):
                records.append((chip.chip_id, chip.source, compute_features(chip, cfg)))
        except Exception as exc:  # noqa: BLE001 - network, auth, missing package...
            logger.warning("Sen1Floods11 download unavailable (%s: %s); using synthetic fallback",
                           type(exc).__name__, exc)
    missing = cfg.n_chips - len(records)
    if missing > 0:
        if records:
            logger.warning("only %d real chips; topping up with %d synthetic chips", len(records), missing)
        for chip in iter_synthetic_chips(cfg, missing, start_idx=len(records)):
            records.append((chip.chip_id, chip.source, compute_features(chip, cfg)))

    rng = np.random.default_rng(cfg.seed + 7)
    feats = np.stack([r[2]["features"] for r in records])
    cell = np.stack([cell_aggregates(f, cfg.rows, cfg.cols) for f in feats])
    Y = np.stack([proxy_target(c, cfg, rng) for c in cell])
    ds = FloodDataset(
        X=feats, Y=Y,
        valid=np.stack([r[2]["valid"] for r in records]),
        rgb=np.stack([r[2]["rgb"] for r in records]),
        cell_feats=cell,
        chip_ids=tuple(r[0] for r in records),
        sources=tuple(r[1] for r in records),
        rows=cfg.rows, cols=cfg.cols,
    )
    np.savez_compressed(cache, X=ds.X, Y=ds.Y, valid=ds.valid, rgb=ds.rgb, cell_feats=ds.cell_feats,
                        chip_ids=np.array(ds.chip_ids), sources=np.array(ds.sources))
    logger.info("dataset ready: %d chips (%s)", ds.n, ds.source_summary)
    return ds
