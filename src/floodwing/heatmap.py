"""Small CNN that predicts the per-cell (proxy) survivor-likelihood heatmap."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import stats

from .config import Config, get_device
from .data import FloodDataset
from .utils import get_logger

try:  # torch is required for this module; the notebook installs it if missing
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyTorch is required for floodwing.heatmap: pip install torch") from exc


class HeatmapCNN(nn.Module):
    """3 x (Conv-BN-ReLU) blocks (16/32/64), adaptive pool to the grid, 1x1 head, sigmoid."""

    def __init__(self, in_channels: int = 4, rows: int = 3, cols: int = 4) -> None:
        super().__init__()

        def block(cin: int, cout: int, pool: bool) -> list[nn.Module]:
            layers: list[nn.Module] = [nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
            return layers + ([nn.MaxPool2d(2)] if pool else [])

        self.features = nn.Sequential(*block(in_channels, 16, True), *block(16, 32, True), *block(32, 64, False))
        self.pool = nn.AdaptiveAvgPool2d((rows, cols))
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, H, W) -> (B, rows*cols)
        return torch.sigmoid(self.head(self.pool(self.features(x)))).flatten(1)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


@dataclass
class TrainResult:
    model: HeatmapCNN
    history: dict[str, list[float]]
    train_idx: np.ndarray
    val_idx: np.ndarray
    metrics: dict[str, float]
    predictions: np.ndarray            # (N, n_cells) raw sigmoid output
    weights: np.ndarray                # (N, n_cells) per-chip min-max scaled to [0, 1]
    n_params: int
    device: str
    train_time_s: float
    extras: dict = field(default_factory=dict)


def split_indices(n: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """80/20 train/val split by chip."""
    perm = np.random.default_rng(seed).permutation(n)
    n_val = max(1, int(round(val_frac * n)))
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def topk_overlap(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Mean fraction of the true top-k cells that the model also ranks top-k."""
    vals = []
    for t, p in zip(y_true, y_pred):
        vals.append(len(set(np.argsort(-t)[:k]) & set(np.argsort(-p)[:k])) / k)
    return float(np.mean(vals))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> dict[str, float]:
    """MAE, Pearson r, Spearman rho (pooled over cells) and top-k overlap."""
    t, p = y_true.ravel(), y_pred.ravel()
    pr = float(stats.pearsonr(t, p)[0]) if np.std(p) > 0 else float("nan")
    sr = float(stats.spearmanr(t, p)[0]) if np.std(p) > 0 else float("nan")
    return {"mae": float(np.mean(np.abs(t - p))), "pearson_r": pr, "spearman_rho": sr,
            "topk_overlap": topk_overlap(y_true, y_pred, k)}


def minmax_rows(a: np.ndarray) -> np.ndarray:
    """Scale each row to [0, 1] (constant rows map to 0.5)."""
    lo, hi = a.min(axis=1, keepdims=True), a.max(axis=1, keepdims=True)
    rng = np.where(hi - lo > 1e-9, hi - lo, 1.0)
    out = (a - lo) / rng
    out[(hi - lo).ravel() <= 1e-9] = 0.5
    return out


def _predict(model: HeatmapCNN, X: np.ndarray, device: str, batch: int = 64) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            outs.append(model(torch.from_numpy(X[i:i + batch]).to(device)).cpu().numpy())
    return np.concatenate(outs)


def train_heatmap_model(ds: FloodDataset, cfg: Config, logger: logging.Logger | None = None,
                        use_cache: bool = True) -> TrainResult:
    """Train (or load cached) CNN, evaluate on the validation split, predict all chips."""
    logger = logger or get_logger()
    device = get_device()
    torch.manual_seed(cfg.seed)
    tr, va = split_indices(ds.n, cfg.val_frac, cfg.seed)
    model = HeatmapCNN(ds.X.shape[1], cfg.rows, cfg.cols).to(device)
    n_params = count_parameters(model)
    if n_params >= 100_000:
        raise RuntimeError(f"CNN has {n_params} parameters; the spec caps it at 100k")

    tag = f"{'fast' if cfg.fast_mode else 'full'}_n{ds.n}_g{cfg.rows}x{cfg.cols}_s{cfg.seed}"
    w_path: Path = cfg.model_dir / f"heatmap_cnn_{tag}.pt"
    hist_path = cfg.model_dir / f"heatmap_history_{tag}.npz"
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_spearman": []}
    t0 = time.perf_counter()
    if use_cache and w_path.exists() and hist_path.exists():
        model.load_state_dict(torch.load(w_path, map_location=device))
        h = np.load(hist_path)
        history = {k: h[k].tolist() for k in history}
        logger.info("loaded cached CNN weights from %s", w_path)
    else:
        Xtr = torch.from_numpy(ds.X[tr]).to(device)
        Ytr = torch.from_numpy(ds.Y[tr]).to(device)
        Xva = torch.from_numpy(ds.X[va]).to(device)
        Yva = torch.from_numpy(ds.Y[va]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        loss_fn = nn.MSELoss()
        gen = torch.Generator().manual_seed(cfg.seed)
        for epoch in range(cfg.epochs):
            model.train()
            perm = torch.randperm(len(tr), generator=gen)
            run_loss = 0.0
            for i in range(0, len(tr), cfg.batch_size):
                idx = perm[i:i + cfg.batch_size]
                if len(idx) < 2:   # BatchNorm needs >1 sample in train mode
                    continue
                opt.zero_grad()
                loss = loss_fn(model(Xtr[idx]), Ytr[idx])
                loss.backward()
                opt.step()
                run_loss += float(loss) * len(idx)
            model.eval()
            with torch.no_grad():
                pv = model(Xva)
                vloss = float(loss_fn(pv, Yva))
            sp = stats.spearmanr(ds.Y[va].ravel(), pv.cpu().numpy().ravel())[0]
            history["train_loss"].append(run_loss / len(tr))
            history["val_loss"].append(vloss)
            history["val_spearman"].append(float(sp) if np.isfinite(sp) else float("nan"))
            logger.info("epoch %2d/%d  train %.4f  val %.4f  spearman %.3f", epoch + 1, cfg.epochs,
                        history["train_loss"][-1], vloss, history["val_spearman"][-1])
        torch.save(model.state_dict(), w_path)
        np.savez(hist_path, **{k: np.array(v) for k, v in history.items()})
    train_time = time.perf_counter() - t0

    preds = _predict(model, ds.X, device)
    metrics = regression_metrics(ds.Y[va], preds[va], cfg.k)
    metrics_train = regression_metrics(ds.Y[tr], preds[tr], cfg.k)
    # Baseline for context: predict the training-set mean per cell.
    mean_pred = np.repeat(ds.Y[tr].mean(axis=0, keepdims=True), len(va), axis=0)
    baseline = {"mae": float(np.mean(np.abs(ds.Y[va] - mean_pred)))}
    logger.info("validation: MAE %.4f | Pearson %.3f | Spearman %.3f | top-%d overlap %.2f",
                metrics["mae"], metrics["pearson_r"], metrics["spearman_rho"], cfg.k, metrics["topk_overlap"])
    np.savez(cfg.model_dir / f"heatmap_predictions_{tag}.npz", preds=preds, chip_ids=np.array(ds.chip_ids))
    return TrainResult(model=model, history=history, train_idx=tr, val_idx=va, metrics=metrics,
                       predictions=preds, weights=minmax_rows(preds), n_params=n_params, device=device,
                       train_time_s=train_time,
                       extras={"train_metrics": metrics_train, "mean_baseline_mae": baseline["mae"]})
