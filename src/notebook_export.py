"""
Notebook export — packages one pipeline run into a single .ipynb.

The notebook is not just a report dump: each phase's section embeds
that phase's actual source file as a runnable code cell (so opening
the notebook and hitting "Run All" — from the project root, with the
same venv active — reproduces the pipeline), interleaved with markdown
cells showing what that specific run produced.

We build the notebook as a plain dict following the nbformat v4
schema directly, so this has no dependency on the `nbformat` package.
"""

from __future__ import annotations

import json
import os

PHASE_FILES = {
    "phase1_data.py": "Phase 1 — Data Understanding & Augmentation",
    "phase2_nas.py": "Phase 2 — Neural Architecture Search",
    "phase3_hpo.py": "Phase 3 — Hyperparameter Optimization",
    "phase4_train.py": "Phase 4 — Training",
    "phase5_compress.py": "Phase 5 — Compression & ONNX Export",
}


def _md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def _code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def _read_phase_source(phases_dir: str, filename: str) -> str:
    path = os.path.join(phases_dir, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_notebook(report: dict, phases_dir: str, output_dir: str, job_id: str) -> str:
    cells = []

    # --- Title / run summary ---
    ds = report["dataset"]
    nas = report["nas"]
    hpo = report["hpo"]
    train = report["training"]
    comp = report["compression"]

    final_acc_line = (
        f"**Final validation accuracy:** {train['final_val_acc']}%"
        if train["final_val_acc"] is not None
        else f"**Final validation loss:** {train['final_val_loss']}"
    )

    summary_md = f"""# Autonomous AutoML Run — `{report['job_id']}`

Generated automatically at the end of the pipeline. This notebook is a
full record of one run: every phase's source code as it executed,
followed by that phase's actual output for this job.

| | |
|---|---|
| Task type | `{report['task_type']}` |
| Target column | `{report['target_col']}` |
| Total pipeline time | {report['elapsed_seconds']}s |
| Dataset rows (final / original) | {ds['final_rows']} / {ds['original_rows']} |
| Synthetic rows added | {ds['synthetic_added']} |
| Best architecture | {'-'.join(map(str, nas['best_architecture']['layers']))} ({nas['best_architecture']['activation']}) |
| Best learning rate | {hpo['best_hparams']['lr']} |
| {final_acc_line.replace('**', '')} | |
| Model size (original → compressed) | {comp['original_size_mb']} MB → {comp['compressed_size_mb']} MB |
| Accuracy change after compression | {comp['accuracy_loss']} |

**How to run this notebook:** place it in the project root (next to
`src/`), activate the same virtualenv used for the original run
(`pip install -r requirements.txt`), and run cells top to bottom. Each
phase cell defines that phase's `run(...)` function exactly as it
executed — nothing is abridged.
"""
    cells.append(_md(summary_md))

    # --- Setup cell ---
    cells.append(_code(
        "import json\nimport pandas as pd\nimport numpy as np\nimport torch\n\n"
        f"REPORT_PATH = \"{job_id}_report.json\"\n"
        f"with open(REPORT_PATH) as f:\n    report = json.load(f)\nreport"
    ))

    # --- Phase sections ---
    phase_result_snippets = {
        "phase1_data.py": (
            "## Phase 1 results for this run\n\n"
            f"- Rows: {ds['original_rows']} → {ds['final_rows']} ({ds['synthetic_added']} synthetic added)\n"
            f"- Features: {ds['features']}\n"
            f"- Class balance: {json.dumps(ds['class_balance'])}\n"
        ),
        "phase2_nas.py": (
            "## Phase 2 results for this run\n\n"
            f"**Best architecture:** `{nas['best_architecture']}`\n\n"
            "All candidates evaluated:\n\n"
            + "\n".join(
                f"- {c['architecture']} → score {round(c['score'], 4)}"
                for c in nas["candidates"]
            )
        ),
        "phase3_hpo.py": (
            "## Phase 3 results for this run\n\n"
            f"**Best hyperparameters (trial {hpo['best_hparams']['trial']}):** "
            f"lr={hpo['best_hparams']['lr']}, dropout={hpo['best_hparams']['dropout']}, "
            f"batch_size={hpo['best_hparams']['batch_size']}\n\n"
            f"Ran {hpo['n_trials']} trials total."
        ),
        "phase4_train.py": (
            "## Phase 4 results for this run\n\n"
            f"- {final_acc_line}\n"
            f"- Restored best checkpoint from epoch {train.get('best_epoch', 'N/A')} "
            f"(final training epoch was {len(train['history']['val_loss'])})\n"
            f"- Overfitting interventions triggered: {len(train['interventions'])}\n"
            + ("".join(f"  - epoch {iv['epoch']}: dropout→{iv['new_dropout']}, lr→{iv['new_lr']}\n"
                        for iv in train["interventions"]) if train["interventions"] else "  (none — training stayed stable)\n")
        ),
        "phase5_compress.py": (
            "## Phase 5 results for this run\n\n"
            f"- Model size: {comp['original_size_mb']} MB → {comp['compressed_size_mb']} MB\n"
            f"- Accuracy change from compression: {comp['accuracy_loss']}\n"
            f"- Exported ONNX file: `{job_id}_model.onnx`"
        ),
    }

    for filename, title in PHASE_FILES.items():
        cells.append(_md(f"---\n# {title}\n\nSource, exactly as it ran:"))
        source = _read_phase_source(phases_dir, filename)
        cells.append(_code(source))
        cells.append(_md(phase_result_snippets[filename]))

    # --- Training curve plot ---
    cells.append(_md("---\n# Training Curves"))
    cells.append(_code(
        "import matplotlib.pyplot as plt\n\n"
        "hist = report['training']['history']\n"
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
        "axes[0].plot(hist['train_loss'], label='train loss')\n"
        "axes[0].plot(hist['val_loss'], label='val loss')\n"
        "axes[0].set_title('Loss'); axes[0].set_xlabel('epoch'); axes[0].legend()\n"
        "if hist.get('train_acc'):\n"
        "    axes[1].plot(hist['train_acc'], label='train acc')\n"
        "    axes[1].plot(hist['val_acc'], label='val acc')\n"
        "    axes[1].set_title('Accuracy (%)'); axes[1].set_xlabel('epoch'); axes[1].legend()\n"
        "plt.tight_layout(); plt.show()"
    ))

    # --- Load ONNX and run inference ---
    cells.append(_md(
        "---\n# Loading the exported ONNX model\n\n"
        "Requires `onnxruntime` (`pip install onnxruntime`). This loads the "
        "compressed model exported by Phase 5 and runs it on a dummy input "
        "of the right shape — replace `dummy_input` with real preprocessed "
        "feature rows to get real predictions."
    ))
    cells.append(_code(
        "import onnxruntime as ort\n\n"
        f"session = ort.InferenceSession(\"{job_id}_model.onnx\")\n"
        "input_name = session.get_inputs()[0].name\n"
        "input_shape = session.get_inputs()[0].shape\n"
        "print('Expected input shape:', input_shape)\n\n"
        "n_features = input_shape[1] if isinstance(input_shape[1], int) else "
        f"{len(ds)}  # fallback\n"
        "dummy_input = np.random.randn(1, n_features).astype(np.float32)\n"
        "outputs = session.run(None, {input_name: dummy_input})\n"
        "print('Model output:', outputs[0])"
    ))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    notebook_path = os.path.join(output_dir, f"{job_id}_notebook.ipynb")
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)

    return notebook_path
