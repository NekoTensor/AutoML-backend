"""
Phase 2 — Neural Architecture Search (NAS)
-------------------------------------------
A deliberately *simple* NAS: it defines a small search space of
MLP architectures (depth, width, activation, dropout) and trains
each candidate for a few quick epochs to rank them. This is not
DARTS or ENAS-grade NAS, but the interface (`run`) is where you'd
swap in a heavier search strategy without touching the rest of the
pipeline.
"""

from __future__ import annotations

import itertools
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}

SEARCH_SPACE = [
    {"layers": [64, 32], "activation": "relu", "dropout": 0.2},
    {"layers": [128, 64], "activation": "relu", "dropout": 0.3},
    {"layers": [128, 64, 64, 32], "activation": "gelu", "dropout": 0.25},
    {"layers": [256, 128, 64], "activation": "gelu", "dropout": 0.3},
    {"layers": [64, 64, 32], "activation": "tanh", "dropout": 0.2},
    {"layers": [128, 128, 64, 32], "activation": "relu", "dropout": 0.35},
]


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, layers: list[int], activation: str, dropout: float):
        super().__init__()
        act = ACTIVATIONS[activation]
        modules = []
        prev = in_dim
        for width in layers:
            modules += [nn.Linear(prev, width), act(), nn.Dropout(dropout)]
            prev = width
        modules.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


def _emit(progress, **kwargs):
    if progress:
        progress(kwargs)


def _quick_train_eval(X_train, y_train, X_val, y_val, arch, task_type, out_dim, epochs=6):
    torch.manual_seed(0)
    model = MLP(X_train.shape[1], out_dim, arch["layers"], arch["activation"], arch["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss() if task_type == "classification" else nn.MSELoss()

    Xt = torch.tensor(X_train, dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    if task_type == "classification":
        yt = torch.tensor(y_train, dtype=torch.long)
        yv = torch.tensor(y_val, dtype=torch.long)
    else:
        yt = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        yv = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(Xt)
        loss = loss_fn(out, yt)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        val_out = model(Xv)
        if task_type == "classification":
            preds = val_out.argmax(dim=1)
            score = (preds == yv).float().mean().item()
        else:
            mse = nn.functional.mse_loss(val_out, yv).item()
            score = 1.0 / (1.0 + mse)  # higher is better, keeps report format uniform

    return score, model


def run(prepared: dict, progress=None) -> dict:
    df = prepared["df"]
    feature_cols = prepared["feature_cols"]
    target_col = prepared["target_col"]
    task_type = prepared["task_type"]

    X = df[feature_cols].values.astype(float)
    y = df[target_col].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    out_dim = len(np.unique(y)) if task_type == "classification" else 1

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if task_type == "classification" else None,
    )

    _emit(progress, phase="phase2", status="running", message="Searching architectures...", total=len(SEARCH_SPACE))

    results = []
    for i, arch in enumerate(SEARCH_SPACE, start=1):
        score, _ = _quick_train_eval(X_train, y_train, X_val, y_val, arch, task_type, out_dim)
        results.append({"arch": arch, "score": score})
        _emit(
            progress,
            phase="phase2",
            status="candidate",
            index=i,
            total=len(SEARCH_SPACE),
            architecture=f"{len(arch['layers'])} layers ({'-'.join(map(str, arch['layers']))})",
            accuracy=round(score * 100, 1) if task_type == "classification" else None,
            score=round(score, 4),
        )

    best = max(results, key=lambda r: r["score"])

    _emit(
        progress,
        phase="phase2",
        status="done",
        message="Best architecture found.",
        best_architecture=best["arch"],
        best_score=round(best["score"], 4),
    )

    return {
        "best_architecture": best["arch"],
        "all_results": results,
        "scaler": scaler,
        "out_dim": out_dim,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
    }
