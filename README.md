---
title: AutoML Backend
emoji: 🧠
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# Autonomous Deep Learning Framework (AutoML Platform)

Upload a CSV, pick a task type and target column, click **Start AutoML** —
everything else (data prep, neural architecture search, hyperparameter
optimization, training with live overfitting correction, and
pruning/distillation/quantization + ONNX export) runs on its own, streaming
live progress to the browser over a WebSocket.

## Project layout

```
automl/
├── app.py                     # FastAPI app: upload endpoint, /ws/run websocket, downloads
├── requirements.txt
├── src/
│   ├── orchestrator.py        # run_pipeline(): calls phases 1→5 in order, then builds the notebook
│   ├── notebook_export.py     # packages one run into a self-contained .ipynb
│   └── phases/
│       ├── phase1_data.py     # validation, class-balance check, synthetic oversampling
│       ├── phase2_nas.py      # architecture search over a small MLP search space
│       ├── phase3_hpo.py      # hyperparameter search (Optuna TPE, or random-search fallback)
│       ├── phase4_train.py    # full training loop, live overfitting detection + correction
│       └── phase5_compress.py # pruning → distillation → quantization → ONNX export
├── static/index.html          # single-page frontend (vanilla JS, no build step)
├── uploads/                   # saved user CSVs (uploads/sample_churn.csv included as a demo)
├── models/                    # exported .onnx models + JSON reports land here
└── reports/                   # (reserved for future report artifacts)
```

Every phase module has exactly one job and one function:

```python
def run(..., progress=None) -> dict:
    ...
```

`progress` is an optional callback invoked with small JSON-serializable
dicts throughout the phase. `orchestrator.py` is the only file that knows
all five phases exist — it wires phase N's output into phase N+1's input.
`app.py` wraps the callback so events get pushed straight to the browser
over the websocket, with no polling.

## Continuous deployment

This repo's `main` branch auto-deploys to a live Hugging Face Space on
every merge, via `.github/workflows/sync-to-hf.yml` — Spaces don't watch
GitHub natively, so this Action pushes `main` to the Space's own git
remote whenever it changes. See `CONTRIBUTING.md` for the full
fork → PR → merge → live flow and which repo secrets a maintainer needs
to set up for it.

## Setup

```bash
cd automl
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.10+ (uses `list[int]` / `X | Y` type hints).

If you don't want the Optuna dependency, just don't install it — phase 3
detects its absence at import time and transparently falls back to plain
random search over the same 25 trials.

## Run

```bash
uvicorn app:app --reload --port 8000
```

Open **http://localhost:8000** in a browser. Upload a CSV (there's a demo
one at `uploads/sample_churn.csv`), choose a task type, pick the target
column, and click Start AutoML. Watch phases 1 → 5 run live, then download
the ONNX model, the JSON report, and a self-contained Jupyter notebook
from the final dashboard.

## What you get at the end

Every run produces three downloadable files:

- **`{job_id}_model.onnx`** — the compressed, exported model.
- **`{job_id}_report.json`** — every number from every phase, machine-readable.
- **`{job_id}_notebook.ipynb`** — the same run as an actual notebook: each
  phase's real source code as a runnable cell, followed by a markdown
  cell showing what *that specific run* produced (dataset stats, NAS
  candidates, HPO trials, training curves, compression numbers), plus a
  training-curve plot cell and a cell that loads the ONNX file with
  `onnxruntime` and runs it. Built by `src/notebook_export.py`, called
  automatically at the end of `orchestrator.run_pipeline()`. To run the
  notebook yourself: drop it in the project root (next to `src/`)
  alongside its matching `_report.json` and `_model.onnx`, activate the
  same venv, `pip install matplotlib onnxruntime` (not in the base
  requirements since the notebook is optional), and run top to bottom.

## Phase 4 training details worth knowing

`phase4_train.py` does two things beyond a plain training loop:

- **Best-checkpoint tracking**: whenever validation loss hits a new low,
  it snapshots the model's weights. The model handed to Phase 5 (and
  therefore the ONNX export) is the *best* checkpoint seen across all
  epochs, not just whatever the final epoch happened to produce.
- **Trend-based overfitting detection**: rather than requiring `N`
  strictly consecutive epochs of rising validation loss, it compares
  current val loss to a rolling value from `TREND_WINDOW` epochs back —
  catching a real, sustained drift even when the curve wiggles along the
  way. A short cooldown (`INTERVENTION_COOLDOWN`) after each intervention
  stops it from re-triggering every epoch once dropout/LR are adjusted.

## What's real vs. simplified (read this before you demo it)

This is a genuine, runnable pipeline — not a scripted fake progress bar.
Every number on screen comes from an actual model being trained. That
said, a few things are intentionally lightweight so it runs in a few
minutes on a laptop CPU instead of hours on a GPU cluster:

- **NAS (phase 2)** searches a fixed shortlist of 6 hand-picked
  architectures (varying depth/width/activation/dropout), each trained
  for 6 quick epochs to rank them — not a differentiable/evolutionary
  search over an open-ended space. Swapping in DARTS, ENAS, or an
  evolutionary search is a matter of replacing `SEARCH_SPACE` and
  `_quick_train_eval` in `phase2_nas.py`; nothing else in the pipeline
  needs to change.
- **HPO (phase 3)** runs 25 trials of Optuna's TPE sampler (or random
  search without Optuna) over learning rate, dropout, and batch size.
- **Synthetic oversampling (phase 1)** is a from-scratch SMOTE-style
  k-NN interpolation, not `imbalanced-learn`, to avoid an extra
  dependency — same idea, smaller footprint.
- **Compression (phase 5)** does real `torch.nn.utils.prune` L1
  magnitude pruning, a lightweight distillation pass (student = pruned
  net retrained against the pre-prune model's soft outputs), and real
  dynamic int8 quantization. The ONNX export ships the pruned +
  distilled fp32 graph rather than the quantized graph, because ONNX's
  opset support for PyTorch's dynamic-quantized linear ops is patchy —
  documented in a comment at the top of the export step in
  `phase5_compress.py`.
- **Overfitting detection (phase 4)** is a simple, transparent rule
  (validation loss rising for `PATIENCE` epochs while trailing the
  train/val gap) rather than a learned early-stopping policy — easy to
  read, easy to tune the two constants at the top of the file.

None of this is fake — it's genuinely training and returning real
scores — but if someone asks "is this doing DARTS-style NAS and Bayesian
optimization end to end," the honest answer is: it's NAS-flavored search
and HPO-flavored search, built for interface clarity over search-strategy
sophistication. The architecture is built so you can drop in a heavier
search strategy per-phase later without touching the orchestrator, the
API, or the frontend.

## Extending it

- **Swap the model family**: everything downstream of `phase2_nas.MLP`
  only cares that a phase returns a `torch.nn.Module`, so you could
  replace the whole NAS module with a search over tree ensembles,
  CNN/tabular-attention blocks, etc.
- **Add "Run Inference"**: the ONNX file + `scaler`/`encoders` from
  `phase1_data.run()` are all you need — a `/api/predict/{job_id}`
  endpoint that loads the ONNX graph with `onnxruntime` and applies the
  same preprocessing would complete the "Run Inference" button shown on
  the dashboard mockup (not yet wired up in this drop).
- **Persist jobs**: right now everything is in-memory per websocket
  connection; for multi-user or resumable jobs, back `orchestrator.py`'s
  state with a job table (SQLite is plenty) keyed by `job_id`.
