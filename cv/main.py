"""
KrishiDrishti CV Module — FastAPI Application

Endpoints:
  POST /analyze          — Run the full CV grading pipeline on a wheat grain photo
  POST /upload-video     — Store a buyer visual record video (no CV processing)
  GET  /health           — Health check
"""

import os
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse

from cv.config import (
    ALLOWED_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    MAX_FILE_SIZE_MB,
    MAX_VIDEO_SIZE_MB,
)
from cv.pipeline import run_pipeline

app = FastAPI(
    title="KrishiDrishti CV Service",
    description=(
        "Deterministic computer vision grading for agricultural produce. "
        "Accepts a photo of grain spread on a white sheet with a ₹10 coin "
        "for reference, and returns an AGMARK grade."
    ),
    version="0.1.0",
)

# Directory for storing uploaded videos
VIDEO_STORAGE_DIR = Path(os.environ.get("VIDEO_STORAGE_DIR", "./uploaded_videos"))
VIDEO_STORAGE_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "service": "krishidrishti-cv"}


@app.post("/analyze")
async def analyze_grain(
    file: UploadFile = File(..., description="Grain photo (JPEG/PNG, max 10MB)"),
    detailed: bool = Query(
        default=False,
        description="If true, include per-metric grade breakdown in the response",
    ),
    commodity: str = Query(
        default="wheat",
        description="Commodity name for AGMARK threshold lookup",
    ),
):
    """
    Run the full CV grading pipeline on a grain photo.

    The photo should show wheat grains spread out on a white sheet
    of paper with a ₹10 coin placed alongside as a size reference.

    Returns an AGMARK grade (A/B/C/Reject) with measurements.
    """
    # --- Input Validation ---
    if file.content_type not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={"status": "invalid_format"},
        )

    # Read the file
    image_bytes = await file.read()

    # Check file size
    file_size_mb = len(image_bytes) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        return JSONResponse(
            status_code=400,
            content={"status": "file_too_large"},
        )

    # --- Run Pipeline ---
    result = run_pipeline(
        image_bytes=image_bytes,
        detailed=detailed,
        commodity=commodity,
    )

    # Determine HTTP status code based on result
    if result["status"] == "success":
        return JSONResponse(status_code=200, content=result)
    elif result["status"] in ("invalid_format", "file_too_large"):
        return JSONResponse(status_code=400, content=result)
    else:
        # Pipeline-level errors (blurry, tilted, no coin, etc.)
        # These are user-actionable, not server errors → 422
        return JSONResponse(status_code=422, content=result)


@app.post("/upload-video")
async def upload_video(
    file: UploadFile = File(..., description="Grain video (MP4/MOV, max 50MB)"),
):
    """
    Upload a short video of the grain for buyer visual reference.

    No CV processing is performed — the video is stored as a blob
    and a URL is returned. Link this URL to the listing record.
    """
    # --- Input Validation ---
    if file.content_type not in ALLOWED_VIDEO_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={"status": "invalid_format"},
        )

    video_bytes = await file.read()

    file_size_mb = len(video_bytes) / (1024 * 1024)
    if file_size_mb > MAX_VIDEO_SIZE_MB:
        return JSONResponse(
            status_code=400,
            content={"status": "file_too_large"},
        )

    # Generate a unique filename
    ext = ".mp4" if file.content_type == "video/mp4" else ".mov"
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = VIDEO_STORAGE_DIR / filename

    # Store the video
    with open(filepath, "wb") as f:
        f.write(video_bytes)

    # In production, this would return a cloud storage URL
    video_url = f"/videos/{filename}"

    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "video_url": video_url,
        },
    )
