"""
Phase 3 — Hyperparameter Optimization (HPO)
---------------------------------------------
Uses Optuna if it's installed (TPE sampler → smarter trials), and
transparently falls back to plain random search otherwise so the
project still runs with a slimmer dependency set.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn

from src.phases.phase2_nas import MLP

try:
    import optuna
    from optuna.samplers import TPESampler
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False


N_TRIALS = 25


def _emit(progress, **kwargs):
    if progress:
        progress(kwargs)


def _train_eval_once(X_train, y_train, X_val, y_val, arch, task_type, out_dim, lr, dropout, batch_size, epochs=10):
    torch.manual_seed(0)
    arch = dict(arch)
    arch["dropout"] = dropout
    model = MLP(X_train.shape[1], out_dim, arch["layers"], arch["activation"], dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss() if task_type == "classification" else nn.MSELoss()

    Xt = torch.tensor(X_train, dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    if task_type == "classification":
        yt = torch.tensor(y_train, dtype=torch.long)
        yv = torch.tensor(y_val, dtype=torch.long)
    else:
        yt = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        yv = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    n = len(Xt)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            out = model(Xt[idx])
            loss = loss_fn(out, yt[idx])
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
            score = 1.0 / (1.0 + mse)

    return score


def run(prepared: dict, nas_result: dict, progress=None) -> dict:
    task_type = prepared["task_type"]
    arch = nas_result["best_architecture"]
    out_dim = nas_result["out_dim"]
    X_train, X_val = nas_result["X_train"], nas_result["X_val"]
    y_train, y_val = nas_result["y_train"], nas_result["y_val"]

    _emit(progress, phase="phase3", status="running", message="Starting hyperparameter optimization...", total=N_TRIALS)

    trials_log = []

    def objective(trial_idx, lr, dropout, batch_size):
        score = _train_eval_once(X_train, y_train, X_val, y_val, arch, task_type, out_dim, lr, dropout, batch_size)
        trials_log.append({"trial": trial_idx, "lr": lr, "dropout": dropout, "batch_size": batch_size, "score": score})
        _emit(
            progress,
            phase="phase3",
            status="trial",
            trial=trial_idx,
            total=N_TRIALS,
            lr=round(lr, 5),
            dropout=round(dropout, 2),
            batch_size=batch_size,
            accuracy=round(score * 100, 1) if task_type == "classification" else None,
            score=round(score, 4),
        )
        return score

    if _HAS_OPTUNA:
        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def optuna_objective(trial):
            lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
            dropout = trial.suggest_float("dropout", 0.1, 0.5)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
            return objective(trial.number + 1, lr, dropout, batch_size)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(optuna_objective, n_trials=N_TRIALS)
        best = max(trials_log, key=lambda t: t["score"])
    else:
        for i in range(1, N_TRIALS + 1):
            lr = 10 ** random.uniform(-4, -2.3)
            dropout = random.uniform(0.1, 0.5)
            batch_size = random.choice([16, 32, 64])
            objective(i, lr, dropout, batch_size)
        best = max(trials_log, key=lambda t: t["score"])

    _emit(
        progress,
        phase="phase3",
        status="done",
        message=f"{N_TRIALS} trials completed.",
        best_trial=best,
    )

    return {"best_hparams": best, "all_trials": trials_log}
