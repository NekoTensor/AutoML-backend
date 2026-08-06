"""
FastAPI backend for the Autonomous AutoML Framework.

Endpoints
---------
POST /api/upload            -> saves the CSV, returns column names + preview
WS   /ws/run                -> client sends job config, server streams progress
                                events for all 5 phases, then a final "complete"
                                event with the report + download links
WS   /ws/attach/{job_id}    -> re-attach to a job already in flight; replays
                                every event emitted so far, then follows live
POST /api/cancel/{job_id}   -> cooperatively stops a running job
GET  /api/job/{job_id}      -> job status, for deciding whether to re-attach
GET  /api/download/onnx/{job_id}
GET  /api/download/report/{job_id}
GET  /health                -> liveness probe (used to warm a sleeping Space)
GET  /                       -> redirects to the frontend; this service is API-only
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

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


# ---------------------------------------------------------------------------
# Job registry
#
# A run used to live entirely inside the websocket handler, which meant the
# socket *was* the job: closing the tab orphaned a pipeline that kept burning
# CPU with nobody listening, and a reload could never get back to it. Jobs are
# now first-class and keyed by job_id, with the socket demoted to one of
# possibly-several subscribers. That single change is what makes both cancel
# and resume expressible.
#
# Everything here is in-process and deliberately so — this backend already
# assumes one worker (the ThreadPoolExecutor is module-level state), and a
# Redis-backed registry would be scaffolding for a scale this doesn't have.
# ---------------------------------------------------------------------------

TERMINAL_STATUSES = {"complete", "error", "cancelled"}


class JobCancelled(Exception):
    """Raised inside the worker thread to unwind a cancelled pipeline."""


class Job:
    def __init__(self, job_id: str, loop: asyncio.AbstractEventLoop):
        self.job_id = job_id
        self.loop = loop
        self.status = "running"
        # Full ordered event log. Replaying this to a late subscriber is what
        # lets the frontend rebuild its entire UI state from scratch after a
        # reload, instead of us having to serialize and version that state.
        self.events: list[dict] = []
        self.subscribers: set[asyncio.Queue] = set()
        # Checked from the worker thread on every progress callback, set from
        # the event loop by the cancel endpoint. A plain bool guarded by the
        # GIL would do, but Event makes the cross-thread intent explicit.
        self.cancel_requested = threading.Event()

    def publish(self, event: dict):
        """Append to the log and fan out. Must run on the event loop."""
        self.events.append(event)
        for q in self.subscribers:
            q.put_nowait(event)

    def publish_threadsafe(self, event: dict):
        """Called from the pipeline's worker thread."""
        self.loop.call_soon_threadsafe(self.publish, event)


JOBS: dict[str, Job] = {}
# Bounds memory: each finished job retains its whole event log for replay, so
# without a cap a long-lived Space would accumulate them indefinitely.
MAX_RETAINED_JOBS = 32


def _reap_finished_jobs():
    finished = [jid for jid, j in JOBS.items() if j.status in TERMINAL_STATUSES]
    while len(JOBS) > MAX_RETAINED_JOBS and finished:
        JOBS.pop(finished.pop(0), None)


async def _stream_job(websocket: WebSocket, job: Job):
    """Replay the job's history to this socket, then follow it live.

    Used by both /ws/run and /ws/attach, so a re-attached client goes through
    exactly the same code path as one that watched from the beginning.
    """
    queue: asyncio.Queue = asyncio.Queue()
    # Snapshot the backlog and subscribe in the same synchronous step. There is
    # no await between the two lines, and publish() only ever runs on this same
    # event loop, so no event can slip through the gap: everything already
    # emitted is in `backlog`, everything after it lands in `queue`, with no
    # overlap and nothing dropped.
    backlog = list(job.events)
    job.subscribers.add(queue)
    try:
        for event in backlog:
            await websocket.send_text(json.dumps(event))

        while True:
            if job.status in TERMINAL_STATUSES and queue.empty():
                break
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            await websocket.send_text(json.dumps(event))
    finally:
        job.subscribers.discard(queue)


@app.get("/health")
async def health():
    """Cheap endpoint the frontend can hit to wake a sleeping Space."""
    return {"status": "ok"}


@app.get("/api/job/{job_id}")
async def job_status(job_id: str):
    job = JOBS.get(job_id)
    # "unknown" rather than a 404 so the frontend can treat "never existed"
    # and "swept after finishing" identically: both mean don't re-attach.
    return {"job_id": job_id, "status": job.status if job else "unknown"}


@app.post("/api/cancel/{job_id}")
async def cancel_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None or job.status in TERMINAL_STATUSES:
        return {"job_id": job_id, "cancelled": False}
    # Cooperative: the pipeline notices at its next progress callback. There's
    # no safe way to kill a thread mid-tensor-op, and every phase reports
    # progress frequently enough that the delay is at most one epoch/trial.
    job.cancel_requested.set()
    return {"job_id": job_id, "cancelled": True}


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
    loop = asyncio.get_running_loop()

    try:
        config = json.loads(await websocket.receive_text())
        dataset_path = config["path"]
        task_type = config["task_type"]
        target_col = config["target_col"]
        job_id = config.get("job_id") or uuid.uuid4().hex[:10]
    except (WebSocketDisconnect, json.JSONDecodeError, KeyError) as e:
        await _send_and_close(websocket, {"phase": "pipeline", "status": "error",
                                          "message": f"Invalid job config: {e}"})
        return

    job = Job(job_id, loop)
    JOBS[job_id] = job
    _reap_finished_jobs()

    def progress_cb(event: dict):
        # Called from the worker thread. Raising here unwinds the pipeline
        # wherever it currently is — every phase calls back often enough that
        # a cancel takes effect within one trial or epoch.
        if job.cancel_requested.is_set():
            raise JobCancelled()
        job.publish_threadsafe(event)

    def run_job():
        return run_pipeline(job_id, dataset_path, task_type, target_col, OUTPUT_DIR, progress=progress_cb)

    # Deliberately not awaited here: the job's lifetime is owned by the
    # registry, not by this socket. If the client vanishes, _finalize_job still
    # runs and the result stays available to a client that re-attaches.
    future = loop.run_in_executor(executor, run_job)
    future.add_done_callback(lambda f: _finalize_job(job, f))

    await _follow(websocket, job)


def _finalize_job(job: Job, future):
    """Turn the worker thread's outcome into the job's terminal event."""
    try:
        report = future.result()
        event = {
            "phase": "pipeline",
            "status": "complete",
            "job_id": job.job_id,
            "onnx_url": f"/api/download/onnx/{job.job_id}",
            "report_url": f"/api/download/report/{job.job_id}",
            "notebook_url": f"/api/download/notebook/{job.job_id}",
            "report": report,
        }
        job.status = "complete"
    except JobCancelled:
        event = {"phase": "pipeline", "status": "cancelled", "job_id": job.job_id}
        job.status = "cancelled"
    except Exception as e:
        event = {"phase": "pipeline", "status": "error", "message": str(e)}
        job.status = "error"
    job.publish(event)


async def _follow(websocket: WebSocket, job: Job):
    """Stream a job to a socket, tolerating the client going away."""
    try:
        await _stream_job(websocket, job)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


async def _send_and_close(websocket: WebSocket, payload: dict):
    try:
        await websocket.send_text(json.dumps(payload))
    except Exception:
        pass
    try:
        await websocket.close()
    except Exception:
        pass


@app.websocket("/ws/attach/{job_id}")
async def ws_attach(websocket: WebSocket, job_id: str):
    """Re-attach to an existing job after a reload or a dropped connection.

    Replays the job's full event log before following it live, so the client
    ends up in exactly the state it would have reached had it never left.
    """
    await websocket.accept()
    job = JOBS.get(job_id)
    if job is None:
        await _send_and_close(websocket, {
            "phase": "pipeline",
            "status": "error",
            "message": "That run is no longer available.",
        })
        return
    await _follow(websocket, job)


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


# ---------------------------------------------------------------------------
# This service is API-only.
#
# It used to also mount static/index.html at "/", which meant the product had
# two complete, independently-written frontends: this one and the Next.js app
# on Vercel. Two implementations of the same UI drift the moment either side
# changes, and the copy served from here was already behind. The Next.js app
# is the single frontend; "/" now just points people at it.
#
# Set FRONTEND_URL to override the redirect target (e.g. a preview deploy).
# ---------------------------------------------------------------------------

FRONTEND_URL = os.environ.get("FRONTEND_URL", "https://nekocortex.vercel.app")


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(FRONTEND_URL, status_code=307)
