"""
main.py — Brain Tumour Detection API v3
========================================
FastAPI server exposing the full HADF → YOLO-NAS → En-DeNet v3 pipeline
with Grad-CAM explainability, 3D reconstruction, and clinical PDF reports.

Endpoints
---------
GET  /health                 — liveness + readiness probe
GET  /api/model-info         — architecture summary
POST /api/predict            — full analysis → JSON (base64 images + 3D data)
POST /api/report             — full analysis → PDF file download
GET  /api/predict/stream     — SSE progress stream while analysing
POST /api/batch              — analyse up to 10 images, returns list of results
GET  /api/classes            — tumour class names, colours, descriptions
GET  /docs                   — Swagger UI (auto-generated)
GET  /redoc                  — ReDoc UI (auto-generated)

Usage
-----
    # Development (auto-reload):
    python main.py --reload

    # Production:
    python main.py --host 0.0.0.0 --port 8000 --workers 2

    # Custom model directory:
    python main.py --model-dir /opt/models

Environment variables (override CLI defaults)
---------------------------------------------
    MODEL_DIR   path to trained_models/   (default: ./trained_models)
    HOST        bind address               (default: 127.0.0.1)
    PORT        bind port                  (default: 8000)
    LOG_LEVEL   debug|info|warning|error   (default: info)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import tempfile
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import (BackgroundTasks, FastAPI, File, Form,
                     HTTPException, Request, UploadFile)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import (FileResponse, JSONResponse,
                                StreamingResponse)
from pydantic import BaseModel, Field

from models.inference import BrainTumorPipeline

# ── AMD / ROCm env flags (safe to set on all platforms) ───────────────────────
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION",        "11.0.2")
os.environ.setdefault("MIOPEN_DISABLE_CACHE",            "1")
os.environ.setdefault("MIOPEN_DEBUG_DISABLE_SQL",        "1")
os.environ.setdefault("MIOPEN_DISABLE_USERDB",           "1")
os.environ.setdefault("MIOPEN_DEBUG_CONV_IMPLICIT_GEMM", "1")
os.environ.setdefault("AMD_LOG_LEVEL",                   "0")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("brain_tumor_api")

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB   = 50
MAX_BATCH_SIZE     = 10
ALLOWED_MIME_TYPES = {
    "image/jpeg", "image/jpg", "image/png",
    "image/bmp",  "image/tiff", "image/webp",
}
CLASS_INFO = {
    "Glioma": {
        "color":       "#FF4444",
        "description": "Most common primary brain tumour. Arises from glial cells. "
                        "Grades I–IV; WHO grade IV (glioblastoma) is most aggressive.",
        "prevalence":  "~45% of all primary brain tumours",
    },
    "Meningioma": {
        "color":       "#FF8C00",
        "description": "Arises from the meninges. Usually benign (grade I), slow-growing. "
                        "Often incidentally discovered.",
        "prevalence":  "~37% of all primary brain tumours",
    },
    "Pituitary": {
        "color":       "#FFD700",
        "description": "Benign adenoma of the pituitary gland. May cause hormonal "
                        "imbalances and visual disturbances.",
        "prevalence":  "~15% of all primary brain tumours",
    },
    "No Tumor": {
        "color":       "#44CC44",
        "description": "No tumour detected. Normal brain parenchyma.",
        "prevalence":  "N/A",
    },
}

# ── Global pipeline instance ──────────────────────────────────────────────────
_pipeline: Optional[BrainTumorPipeline] = None


# ── Pydantic response schemas ─────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:       str
    models_ready: bool
    demo_mode:    bool
    device:       str
    uptime_s:     float

class ClassInfo(BaseModel):
    name:        str
    color:       str
    description: str
    prevalence:  str

class BatchItem(BaseModel):
    filename:    str
    success:     bool
    result:      Optional[dict] = None
    error:       Optional[str]  = None

class BatchResponse(BaseModel):
    total:      int
    succeeded:  int
    failed:     int
    results:    List[BatchItem]
    elapsed_s:  float


# ── Application lifespan ──────────────────────────────────────────────────────

_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load models. Shutdown: log goodbye."""
    global _pipeline
    model_dir = os.environ.get("MODEL_DIR", "./trained_models")
    logger.info("=" * 60)
    logger.info("  Brain Tumour Detection API v3  —  starting up")
    logger.info("  Model dir : %s", model_dir)
    logger.info("=" * 60)

    try:
        _pipeline = BrainTumorPipeline(model_dir=model_dir)
        mode = "DEMO (no weights)" if _pipeline.demo_mode else "LIVE"
        logger.info("✅ Pipeline ready | mode=%s | device=%s",
                    mode, _pipeline.device)
    except Exception:
        logger.exception("❌ Pipeline initialisation failed — running in stub mode")
        _pipeline = None

    yield  # ── server is running ──────────────────────────────────────────────

    logger.info("🛑  Brain Tumour Detection API shutting down")


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Brain Tumour Detection API",
    description=(
        "**AI-assisted clinical decision support.**\n\n"
        "Pipeline: `HADF preprocessing` → `YOLO-NAS classification` → "
        "`En-DeNet v3 segmentation` → `Grad-CAM / SegGrad-CAM++ / EigenCAM` "
        "→ `3D reconstruction` → `clinical PDF report`.\n\n"
        "> ⚠️ **Not a substitute for professional medical diagnosis.**"
    ),
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow any origin in dev; tighten for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gzip large JSON responses (base64 images compress very well)
app.add_middleware(GZipMiddleware, minimum_size=2048)


# ── Request-ID middleware ─────────────────────────────────────────────────────

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Attach a unique X-Request-ID to every response for traceability."""
    rid = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    start = time.perf_counter()
    response = await call_next(request)
    elapsed  = round(time.perf_counter() - start, 3)
    response.headers["X-Request-ID"]    = rid
    response.headers["X-Process-Time"]  = str(elapsed)
    return response


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_pipeline() -> BrainTumorPipeline:
    """Return the pipeline or raise 503 if not ready."""
    if _pipeline is None:
        raise HTTPException(503, "Pipeline not initialised — check server logs")
    return _pipeline


def _decode_image(contents: bytes) -> np.ndarray:
    """
    Decode raw file bytes to a NumPy BGR/grayscale image.
    Handles uint8 and uint16 DICOM-style images.
    """
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise HTTPException(400, "Cannot decode image — unsupported format or corrupted file")
    # Normalise 16-bit to 8-bit (common for raw MRI exports)
    if image.dtype == np.uint16:
        image = (image / 256).astype(np.uint8)
    return image


async def _read_upload(file: UploadFile) -> bytes:
    """Read upload, enforcing file-size and MIME-type limits."""
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            400,
            f"Unsupported file type '{file.content_type}'. "
            f"Accepted: {', '.join(sorted(ALLOWED_MIME_TYPES))}",
        )
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {MAX_FILE_SIZE_MB} MB limit")
    return contents


def _make_temp_pdf(prefix: str = "report_") -> str:
    """Return a path for a temporary PDF that will survive the request."""
    fd, path = tempfile.mkstemp(suffix=".pdf", prefix=prefix)
    os.close(fd)
    return path


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health / readiness probe",
    tags=["System"],
)
async def health():
    """
    Kubernetes-style liveness + readiness probe.
    Returns HTTP 200 when the pipeline is ready; 503 if it failed to load.
    """
    p = _pipeline
    if p is None:
        raise HTTPException(503, "Pipeline not ready")
    return HealthResponse(
        status       = "healthy",
        models_ready = p.models_ready,
        demo_mode    = p.demo_mode,
        device       = str(p.device),
        uptime_s     = round(time.time() - _start_time, 1),
    )


@app.get(
    "/api/model-info",
    summary="Model architecture summary",
    tags=["System"],
)
async def model_info():
    """Return a description of every component in the inference pipeline."""
    return {
        "pipeline_version": "3.0.0",
        "classifier": {
            "name":         "YOLO-NAS (custom)",
            "input":        "1 × 224 × 224 grayscale",
            "backbone":     "QARepVGG stages + SPPF + CBAM",
            "normalisation":"GroupNorm (AMD/ROCm-safe)",
            "loss":         "Focal Loss (γ=2, α=0.25)",
            "classes":      list(CLASS_INFO.keys()),
        },
        "segmenter": {
            "name":    "En-DeNet v3",
            "input":   "1 × 512 × 512 grayscale",
            "features": [
                "ASPP bottleneck (multi-scale context)",
                "Attention gates on skip connections",
                "Residual encoder blocks",
                "Deep supervision (auxiliary heads)",
                "Tversky–Focal combined loss",
                "Dropout2d spatial regularisation",
            ],
        },
        "preprocessing": {
            "name":   "HADF (Hybrid Anisotropic Diffusion Filter)",
            "method": "Perona-Malik with kurtosis-based stopping",
        },
        "explainability": {
            "GradCAM":    "Grad-CAM++ on YOLO-NAS SPPF — JET colourmap",
            "SegGradCAM": "Grad-CAM++ on En-DeNet bottleneck — HOT colourmap",
            "EigenCAM":   "Gradient-free SVD on bottleneck — VIRIDIS colourmap",
            "note":       "Multi-region aware — highlights all lesion foci",
        },
        "reconstruction": {
            "method":  "Gaussian depth extrusion → marching cubes surface mesh",
            "output":  "Interactive Plotly Mesh3d dict (JSON-serialisable)",
            "library": "scikit-image marching_cubes + scipy gaussian_filter",
        },
        "report": {
            "format":  "Clinical PDF (ReportLab)",
            "pages":   "3 — diagnosis, measurements + 3D, explainability",
        },
    }


@app.get(
    "/api/classes",
    summary="Tumour class definitions",
    tags=["System"],
)
async def get_classes():
    """Return class names, accent colours, and clinical descriptions."""
    return {
        name: ClassInfo(name=name, **info)
        for name, info in CLASS_INFO.items()
    }


@app.post(
    "/api/predict",
    summary="Full analysis — returns JSON with base64 images",
    tags=["Inference"],
    responses={
        200: {"description": "Analysis result with all images as base64 PNG"},
        400: {"description": "Bad request (invalid image, wrong MIME type)"},
        413: {"description": "File too large"},
        503: {"description": "Pipeline not ready"},
    },
)
async def predict(
    file:            UploadFile = File(...,  description="MRI scan (JPEG/PNG/BMP/TIFF/WebP, ≤50 MB)"),
    generate_report: bool       = Form(False, description="Include base64-encoded PDF in response"),
    patient_id:      str        = Form("Unknown", description="Patient identifier for report header"),
    patient_name:    str        = Form("N/A",     description="Patient name for report header"),
    radiologist:     str        = Form("AI System", description="Radiologist name for report header"),
):
    """
    **Main inference endpoint.**

    Upload a brain MRI scan and receive:
    - `classification` — tumour class + confidence + probability distribution
    - `segmentation`   — binary mask + coloured overlay (base64 PNG)
    - `tumor_3d`       — shape metrics + Plotly 3D surface dict + volume
    - `hadf_image_b64` — HADF-preprocessed image (base64 PNG)
    - `overlay_image_b64`       — detection overlay with tumour contour
    - `gradcam_overlay_b64`     — Grad-CAM++ heatmap on classifier input
    - `seg_gradcam_overlay_b64` — SegGrad-CAM++ on segmenter bottleneck
    - `eigencam_overlay_b64`    — EigenCAM on segmenter bottleneck
    - `report_b64`  *(optional)* — full clinical PDF as base64 string

    Set `generate_report=true` to include the PDF.
    """
    p        = _get_pipeline()
    contents = await _read_upload(file)
    image    = _decode_image(contents)

    logger.info("predict | file=%s  size=%dx%d  generate_report=%s  patient=%s",
                file.filename, image.shape[1], image.shape[0],
                generate_report, patient_id)

    try:
        result = p.predict(
            image,
            generate_report = generate_report,
            patient_id      = patient_id,
            patient_name    = patient_name,
        )
    except Exception as exc:
        logger.exception("Inference error for %s", file.filename)
        raise HTTPException(500, f"Inference failed: {exc}") from exc

    logger.info("predict | done in %.3fs | class=%s (%.1f%%)",
                result.get("processing_time", 0),
                result.get("classification", {}).get("class_name", "?"),
                result.get("classification", {}).get("confidence", 0))

    return JSONResponse(content=result)


@app.post(
    "/api/report",
    summary="Full analysis — returns PDF file download",
    tags=["Inference"],
    response_class=FileResponse,
    responses={
        200: {"description": "PDF file", "content": {"application/pdf": {}}},
        400: {"description": "Bad request"},
        500: {"description": "Report generation failed"},
        503: {"description": "Pipeline not ready"},
    },
)
async def get_report(
    background_tasks: BackgroundTasks,
    file:         UploadFile = File(...),
    patient_id:   str        = Form("Unknown"),
    patient_name: str        = Form("N/A"),
    radiologist:  str        = Form("AI System"),
):
    """
    **PDF report endpoint.**

    Upload a brain MRI → run full pipeline → stream the clinical PDF back
    as a file download (`Content-Disposition: attachment`).

    The PDF contains:
    - Patient header + diagnosis block
    - Probability distribution table
    - 6-panel imaging analysis (HADF, overlay, 3× CAM heatmaps, mask)
    - Tumour measurements + shape metrics table
    - 3D reconstruction (3 static views)
    - 3× coloured CAM heatmaps (Grad-CAM / SegGrad-CAM / EigenCAM)
    - Clinical disclaimer
    """
    p        = _get_pipeline()
    contents = await _read_upload(file)
    image    = _decode_image(contents)

    logger.info("report | file=%s  patient=%s", file.filename, patient_id)

    try:
        result = p.predict(
            image,
            generate_report = True,
            patient_id      = patient_id,
            patient_name    = patient_name,
        )
    except Exception as exc:
        logger.exception("Pipeline error for report request")
        raise HTTPException(500, f"Pipeline failed: {exc}") from exc

    report_path = result.get("report_path")
    if not report_path or not Path(report_path).exists():
        err = result.get("report_error", "unknown error")
        logger.error("Report generation failed: %s", err)
        raise HTTPException(500, f"Report generation failed: {err}")

    # Schedule temp-file cleanup after response is sent
    background_tasks.add_task(_cleanup_file, report_path)

    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in patient_id)
    return FileResponse(
        path         = report_path,
        media_type   = "application/pdf",
        filename     = f"brain_tumor_report_{safe_id}.pdf",
        headers      = {"X-Patient-ID": patient_id,
                        "X-Tumor-Class": result.get("classification",{})
                                                .get("class_name","Unknown")},
    )


@app.post(
    "/api/batch",
    response_model=BatchResponse,
    summary=f"Batch analysis — up to {MAX_BATCH_SIZE} images",
    tags=["Inference"],
)
async def batch_predict(
    files:       List[UploadFile] = File(..., description=f"Up to {MAX_BATCH_SIZE} MRI scans"),
    patient_ids: str              = Form("",  description="Comma-separated patient IDs (optional)"),
):
    """
    **Batch inference endpoint.**

    Submit up to 10 MRI scans in one request.
    Each file is analysed independently; partial failures do not abort the batch.

    `patient_ids` is an optional comma-separated list that maps positionally
    to the uploaded files (e.g. `P001,P002,P003`).

    Returns a list of per-file results (or error messages).
    """
    p = _get_pipeline()

    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            400, f"Batch size {len(files)} exceeds limit of {MAX_BATCH_SIZE}")

    ids = [s.strip() for s in patient_ids.split(",") if s.strip()]
    t0  = time.perf_counter()
    results: List[BatchItem] = []

    for idx, upload in enumerate(files):
        pid = ids[idx] if idx < len(ids) else f"batch_{idx+1:03d}"
        try:
            contents = await _read_upload(upload)
            image    = _decode_image(contents)
            result   = p.predict(image, patient_id=pid)
            results.append(BatchItem(
                filename = upload.filename or f"file_{idx}",
                success  = True,
                result   = result,
            ))
            logger.info("batch[%d/%d] %s → %s (%.2f%%)",
                        idx + 1, len(files),
                        upload.filename,
                        result.get("classification",{}).get("class_name","?"),
                        result.get("classification",{}).get("confidence", 0))
        except HTTPException as exc:
            results.append(BatchItem(
                filename = upload.filename or f"file_{idx}",
                success  = False,
                error    = exc.detail,
            ))
        except Exception as exc:
            logger.exception("Batch item %d failed", idx)
            results.append(BatchItem(
                filename = upload.filename or f"file_{idx}",
                success  = False,
                error    = str(exc),
            ))

    elapsed   = round(time.perf_counter() - t0, 3)
    succeeded = sum(1 for r in results if r.success)

    return BatchResponse(
        total     = len(files),
        succeeded = succeeded,
        failed    = len(files) - succeeded,
        results   = results,
        elapsed_s = elapsed,
    )


@app.get(
    "/api/predict/stream",
    summary="SSE progress stream during inference",
    tags=["Inference"],
)
async def predict_stream(request: Request, filename: str = "uploaded_scan"):
    """
    **Server-Sent Events (SSE) endpoint.**

    Streams progress updates while inference runs so the frontend can
    show a live progress bar.  The final event carries `event: result`
    with the full JSON payload identical to `/api/predict`.

    Usage (JavaScript):
    ```js
    const es = new EventSource('/api/predict/stream?filename=scan.png');
    es.addEventListener('progress', e => console.log(JSON.parse(e.data)));
    es.addEventListener('result',   e => { const r = JSON.parse(e.data); ... });
    es.addEventListener('error',    e => console.error(e.data));
    ```

    > Note: this endpoint expects the image to be passed as a POST body
    > but EventSource only supports GET.  For a real integration use
    > `fetch()` + `ReadableStream` from the `/api/predict` endpoint,
    > or send the image in a separate request and poll via SSE.
    """
    p = _get_pipeline()

    async def generate():
        steps = [
            (10,  "HADF preprocessing…"),
            (25,  "Running YOLO-NAS classification…"),
            (50,  "En-DeNet segmentation…"),
            (70,  "Computing Grad-CAM heatmaps…"),
            (85,  "Building 3D reconstruction…"),
            (95,  "Generating clinical report…"),
            (100, "Done"),
        ]

        for pct, msg in steps:
            if await request.is_disconnected():
                break
            payload = json.dumps({"percent": pct, "message": msg,
                                   "demo_mode": p.demo_mode})
            yield f"event: progress\ndata: {payload}\n\n"
            await asyncio.sleep(0.25)

        # In a real integration, run inference here and stream the result.
        # For this reference implementation we emit a placeholder result event.
        yield (
            "event: result\n"
            "data: " + json.dumps({
                "message":  "Connect via POST /api/predict for full results",
                "demo_mode": p.demo_mode,
            }) + "\n\n"
        )

    return StreamingResponse(
        generate(),
        media_type = "text/event-stream",
        headers    = {
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",      # Nginx: disable buffering
            "Access-Control-Allow-Origin": "*",
        },
    )


# ── Background task helpers ───────────────────────────────────────────────────

def _cleanup_file(path: str) -> None:
    """Delete a temporary file silently after the response has been sent."""
    try:
        Path(path).unlink(missing_ok=True)
        logger.debug("Cleaned up temp file: %s", path)
    except Exception:
        pass


# ── Exception handlers ────────────────────────────────────────────────────────

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code = 404,
        content     = {
            "error": "Not found",
            "path":  str(request.url.path),
            "hint":  "See /docs for available endpoints",
        },
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    logger.exception("Unhandled server error on %s", request.url.path)
    return JSONResponse(
        status_code = 500,
        content     = {
            "error":   "Internal server error",
            "detail":  str(exc),
            "request": str(request.url.path),
        },
    )


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog        = "main.py",
        description = "Brain Tumour Detection API v3",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host",      default=os.environ.get("HOST", "127.0.0.1"),
                   help="Bind address (use 0.0.0.0 for Docker/LAN)")
    p.add_argument("--port",      type=int,
                   default=int(os.environ.get("PORT", 8000)),
                   help="Bind port")
    p.add_argument("--workers",   type=int, default=1,
                   help="Number of Uvicorn workers (>1 disables --reload)")
    p.add_argument("--reload",    action="store_true",
                   help="Auto-reload on source changes (dev only)")
    p.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "./trained_models"),
                   help="Directory containing yolonas_best.pth + endenet_best.pth")
    p.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "info"),
                   choices=["debug", "info", "warning", "error"],
                   help="Uvicorn log level")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Push CLI value into env so the lifespan handler picks it up
    os.environ["MODEL_DIR"] = args.model_dir

    # Workers > 1 is incompatible with --reload
    workers = 1 if args.reload else args.workers

    logger.info("=" * 60)
    logger.info("  Brain Tumour Detection API  v3.0.0")
    logger.info("  Swagger UI  →  http://%s:%d/docs", args.host, args.port)
    logger.info("  ReDoc       →  http://%s:%d/redoc", args.host, args.port)
    logger.info("  Health      →  http://%s:%d/health", args.host, args.port)
    logger.info("  Model dir   →  %s", args.model_dir)
    logger.info("  Workers     →  %d  |  Reload: %s", workers, args.reload)
    logger.info("=" * 60)

    uvicorn.run(
        "main:app",
        host      = args.host,
        port      = args.port,
        reload    = args.reload,
        workers   = workers,
        log_level = args.log_level,
        access_log= True,
    )