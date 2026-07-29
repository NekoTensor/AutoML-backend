"""
Orchestrator — glues Phases 1-5 together.

This is the ONLY place that knows all five phases exist. Each phase
module only knows about its own inputs/outputs, so you can swap any
phase's internals independently.
"""

from __future__ import annotations

import json
import os
import time

from src.phases import phase1_data, phase2_nas, phase3_hpo, phase4_train, phase5_compress
from src.notebook_export import build_notebook


def run_pipeline(job_id: str, dataset_path: str, task_type: str, target_col: str,
                  output_dir: str, progress=None) -> dict:
    """Runs the full 5-phase AutoML pipeline synchronously.

    `progress` is a callable(event: dict) invoked throughout — the
    FastAPI layer wraps this to push events over a websocket.
    """
    t0 = time.time()

    prepared = phase1_data.run(dataset_path, task_type, target_col, progress=progress)
    nas_result = phase2_nas.run(prepared, progress=progress)
    hpo_result = phase3_hpo.run(prepared, nas_result, progress=progress)
    train_result = phase4_train.run(prepared, nas_result, hpo_result, progress=progress)
    compress_result = phase5_compress.run(prepared, nas_result, train_result, output_dir, job_id, progress=progress)

    elapsed = round(time.time() - t0, 1)

    report = {
        "job_id": job_id,
        "task_type": task_type,
        "target_col": target_col,
        "elapsed_seconds": elapsed,
        "dataset": {
            "original_rows": prepared["original_rows"],
            "final_rows": len(prepared["df"]),
            "features": len(prepared["feature_cols"]),
            "class_balance": prepared["class_balance"],
            "synthetic_added": prepared["synthetic_added"],
        },
        "nas": {
            "best_architecture": nas_result["best_architecture"],
            "candidates": [
                {"architecture": r["arch"], "score": r["score"]} for r in nas_result["all_results"]
            ],
        },
        "hpo": {
            "best_hparams": hpo_result["best_hparams"],
            "n_trials": len(hpo_result["all_trials"]),
        },
        "training": {
            "history": train_result["history"],
            "interventions": train_result["interventions"],
            "final_val_acc": train_result["final_val_acc"],
            "final_val_loss": train_result["final_val_loss"],
            "best_epoch": train_result["best_epoch"],
        },
        "compression": {
            "original_size_mb": compress_result["original_size_mb"],
            "compressed_size_mb": compress_result["compressed_size_mb"],
            "accuracy_loss": compress_result["accuracy_loss"],
        },
        "onnx_path": compress_result["onnx_path"],
    }

    report_path = os.path.join(output_dir, f"{job_id}_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    report["report_path"] = report_path

    phases_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phases")
    notebook_path = build_notebook(report, phases_dir, output_dir, job_id)
    report["notebook_path"] = notebook_path

    if progress:
        progress({"phase": "pipeline", "status": "complete", "report": report})

    return report
