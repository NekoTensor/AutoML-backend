"""
FastAPI backend for the Autonomous AutoML Framework.

Endpoints
---------
POST /api/upload            -> saves the CSV, returns column names + preview
WS   /ws/run                -> client sends job config, server streams progress
                                events for all 5 phases, then a final "complete"
                                event with the report + download links
GET  /api/download/onnx/{job_id}
GET  /api/download/report/{job_id}
GET  /                       -> serves the frontend
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.orchestrator import run_pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

for d in (UPLOAD_DIR, OUTPUT_DIR, REPORT_DIR):
    os.makedirs(d, exist_ok=True)

app = FastAPI(title="Autonomous AutoML Framework")

# The Next.js frontend now lives on its own origin (localhost:3000 in dev,
# your Vercel domain in production), so this backend needs to explicitly
# allow cross-origin requests. Set ALLOWED_ORIGINS as a comma-separated
# env var in production (e.g. "https://your-app.vercel.app"); defaults to
# "*" for easy local development.
_allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _allowed_origins.split(",")] if _allowed_origins != "*" else ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
executor = ThreadPoolExecutor(max_workers=2)


@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    job_id = uuid.uuid4().hex[:10]
    dest_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
    with open(dest_path, "wb") as f:
        f.write(await file.read())

    df = pd.read_csv(dest_path)
    return JSONResponse({
        "job_id": job_id,
        "filename": file.filename,
        "path": dest_path,
        "columns": list(df.columns),
        "rows": len(df),
        "preview": df.head(5).to_dict(orient="records"),
    })


@app.websocket("/ws/run")
async def ws_run(websocket: WebSocket):
    """
    Client sends a JSON config once connected:
        {"path": "...", "task_type": "classification", "target_col": "churn", "job_id": "..."}
    Server streams one JSON message per progress event, ending with
    {"phase": "pipeline", "status": "complete", "report": {...}}
    or {"phase": "pipeline", "status": "error", "message": "..."}
    """
    await websocket.accept()
    loop = asyncio.get_event_loop()

    try:
        config_raw = await websocket.receive_text()
        config = json.loads(config_raw)

        dataset_path = config["path"]
        task_type = config["task_type"]
        target_col = config["target_col"]
        job_id = config.get("job_id") or uuid.uuid4().hex[:10]

        queue: asyncio.Queue = asyncio.Queue()

        def progress_cb(event: dict):
            # Called from the worker thread — hand off to the event loop safely.
            asyncio.run_coroutine_threadsafe(queue.put(event), loop)

        def run_job():
            return run_pipeline(job_id, dataset_path, task_type, target_col, OUTPUT_DIR, progress=progress_cb)

        future = loop.run_in_executor(executor, run_job)

        while not future.done() or not queue.empty():
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.5)
                await websocket.send_text(json.dumps(event))
            except asyncio.TimeoutError:
                continue

        report = future.result()  # re-raises any exception from the worker thread
        await websocket.send_text(json.dumps({
            "phase": "pipeline",
            "status": "complete",
            "job_id": job_id,
            "onnx_url": f"/api/download/onnx/{job_id}",
            "report_url": f"/api/download/report/{job_id}",
            "notebook_url": f"/api/download/notebook/{job_id}",
            "report": report,
        }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"phase": "pipeline", "status": "error", "message": str(e)}))
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@app.get("/api/download/onnx/{job_id}")
async def download_onnx(job_id: str):
    path = os.path.join(OUTPUT_DIR, f"{job_id}_model.onnx")
    if not os.path.exists(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, filename=f"model_{job_id}.onnx")


@app.get("/api/download/report/{job_id}")
async def download_report(job_id: str):
    path = os.path.join(OUTPUT_DIR, f"{job_id}_report.json")
    if not os.path.exists(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, filename=f"report_{job_id}.json")


@app.get("/api/download/notebook/{job_id}")
async def download_notebook(job_id: str):
    path = os.path.join(OUTPUT_DIR, f"{job_id}_notebook.ipynb")
    if not os.path.exists(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, filename=f"automl_run_{job_id}.ipynb")


app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "static"), html=True), name="static")
