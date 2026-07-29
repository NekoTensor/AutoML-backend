"""
Phase 4 — Full Training
-------------------------
Trains the best (architecture, hyperparameter) combo for real, epoch
by epoch, streaming loss/accuracy back to the frontend.

Two safety nets, both aimed at the same failure mode (real-world val
curves are noisy, so training often quietly drifts past its best point
without ever showing N *strictly consecutive* worsening epochs):

  1. Best-checkpoint tracking — whenever val loss hits a new low, we
     snapshot the model's weights. Whatever epoch 40 looks like, the
     model actually handed to Phase 5 is the best checkpoint seen, not
     just "whatever the last epoch produced."
  2. Trend-based overfitting detection — instead of requiring
     `PATIENCE` consecutive rising epochs (too strict for noisy val
     loss, since it resets to 0 on any single down-tick), we compare
     current val loss to a rolling average from `TREND_WINDOW` epochs
     ago. A sustained upward drift trips this even when the curve
     wiggles up and down along the way.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from src.phases.phase2_nas import MLP

MAX_EPOCHS = 40
TREND_WINDOW = 5  # compare current val loss vs. this many epochs back
OVERFIT_MARGIN = 0.02  # how much higher val loss must be than train loss (normalized)
TREND_MARGIN = 0.01  # how much higher current val loss must be vs. the rolling-back value
INTERVENTION_COOLDOWN = 5  # epochs to wait after an intervention before re-triggering


def _emit(progress, **kwargs):
    if progress:
        progress(kwargs)


def run(prepared: dict, nas_result: dict, hpo_result: dict, progress=None) -> dict:
    task_type = prepared["task_type"]
    arch = dict(nas_result["best_architecture"])
    out_dim = nas_result["out_dim"]
    X_train, X_val = nas_result["X_train"], nas_result["X_val"]
    y_train, y_val = nas_result["y_train"], nas_result["y_val"]

    best_hp = hpo_result["best_hparams"]
    lr = best_hp["lr"]
    dropout = best_hp["dropout"]
    batch_size = best_hp["batch_size"]
    arch["dropout"] = dropout

    torch.manual_seed(0)
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
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    interventions = []
    last_intervention_epoch = -INTERVENTION_COOLDOWN

    best_val_loss = float("inf")
    best_state_dict = copy.deepcopy(model.state_dict())
    best_epoch = 0

    _emit(progress, phase="phase4", status="running", message="Training started.", total_epochs=MAX_EPOCHS)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n)
        running_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            out = model(Xt[idx])
            loss = loss_fn(out, yt[idx])
            loss.backward()
            opt.step()
            running_loss += loss.item() * len(idx)
        train_loss = running_loss / n

        model.eval()
        with torch.no_grad():
            train_out = model(Xt)
            val_out = model(Xv)
            val_loss = loss_fn(val_out, yv).item()
            if task_type == "classification":
                train_acc = (train_out.argmax(1) == yt).float().mean().item()
                val_acc = (val_out.argmax(1) == yv).float().mean().item()
            else:
                train_acc = None
                val_acc = None

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        if train_acc is not None:
            history["train_acc"].append(round(train_acc * 100, 2))
            history["val_acc"].append(round(val_acc * 100, 2))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        _emit(
            progress,
            phase="phase4",
            status="epoch",
            epoch=epoch,
            total_epochs=MAX_EPOCHS,
            train_loss=round(train_loss, 4),
            val_loss=round(val_loss, 4),
            train_acc=round(train_acc * 100, 1) if train_acc is not None else None,
            val_acc=round(val_acc * 100, 1) if val_acc is not None else None,
        )

        # --- overfitting detection: rolling-window trend, not strict-consecutive ---
        gap = val_loss - train_loss
        epochs_since_intervention = epoch - last_intervention_epoch
        if epoch > TREND_WINDOW and epochs_since_intervention >= INTERVENTION_COOLDOWN:
            val_loss_then = history["val_loss"][-(TREND_WINDOW + 1)]
            drifted_up = (val_loss - val_loss_then) > TREND_MARGIN
            if drifted_up and gap > OVERFIT_MARGIN:
                new_dropout = min(dropout + 0.1, 0.6)
                new_lr = lr * 0.5
                for m in model.net:
                    if isinstance(m, nn.Dropout):
                        m.p = new_dropout
                for g in opt.param_groups:
                    g["lr"] = new_lr
                interventions.append({"epoch": epoch, "new_dropout": round(new_dropout, 2), "new_lr": round(new_lr, 6)})
                _emit(
                    progress,
                    phase="phase4",
                    status="overfitting_detected",
                    epoch=epoch,
                    message="Overfitting detected — increasing dropout and reducing learning rate.",
                    new_dropout=round(new_dropout, 2),
                    new_lr=round(new_lr, 6),
                )
                dropout, lr = new_dropout, new_lr
                last_intervention_epoch = epoch

    # restore the best checkpoint seen — not just whatever the final epoch produced
    model.load_state_dict(best_state_dict)
    final_val_acc = history["val_acc"][best_epoch - 1] if history["val_acc"] else None
    final_val_loss = history["val_loss"][best_epoch - 1]

    _emit(
        progress,
        phase="phase4",
        status="done",
        message="Training complete.",
        final_val_acc=final_val_acc,
        final_val_loss=final_val_loss,
        interventions=interventions,
        best_epoch=best_epoch,
    )

    return {
        "model": model,
        "history": history,
        "interventions": interventions,
        "input_dim": X_train.shape[1],
        "final_val_acc": final_val_acc,
        "final_val_loss": final_val_loss,
        "best_epoch": best_epoch,
    }
