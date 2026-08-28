"""
KrishiDrishti CV Module — Pydantic Response Schemas

Defines the exact JSON contract between the CV service and the backend.
"""

from pydantic import BaseModel, Field
from typing import Optional


class Measurements(BaseModel):
    """Aggregate grain measurements from the CV pipeline."""

    avg_length_mm: float = Field(..., description="Average grain length in mm")
    avg_width_mm: float = Field(..., description="Average grain width in mm")
    avg_area_mm2: float = Field(..., description="Average grain area in mm^2")
    avg_lw_ratio: float = Field(..., description="Average length/width ratio")
    median_length_mm: float = Field(..., description="Median grain length in mm")
    broken_pct: float = Field(..., description="Percentage of broken grains")
    discolored_pct: float = Field(..., description="Percentage of discolored/damaged grain area")
    foreign_matter_pct: float = Field(..., description="Percentage of foreign matter by area")


class PerMetricGrades(BaseModel):
    """Grade breakdown per individual AGMARK parameter."""

    length: str = Field(..., description="Grade based on average length")
    width: str = Field(..., description="Grade based on average width")
    area: str = Field(..., description="Grade based on average grain area")
    lw_ratio: str = Field(..., description="Grade based on length/width aspect-ratio band")
    broken: str = Field(..., description="Grade based on broken kernel percentage")
    discolored: str = Field(..., description="Grade based on discoloration percentage")
    foreign_matter: str = Field(..., description="Grade based on foreign matter percentage")


class AnalyzeSuccessResponse(BaseModel):
    """Successful analysis response."""

    status: str = Field(default="success")
    grain_count: int = Field(..., description="Number of valid grains detected")
    measurements: Measurements
    grade: str = Field(..., description="Overall AGMARK grade (A/B/C/Reject)")
    hash: str = Field(..., description="SHA-256 hash of the result JSON for QR verification")
    per_metric_grades: Optional[PerMetricGrades] = Field(
        default=None,
        description="Per-parameter grade breakdown. Only included when ?detailed=true",
    )


class AnalyzeErrorResponse(BaseModel):
    """Error response for failed analysis."""

    status: str = Field(
        ...,
        description=(
            "Error code: too_blurry | too_tilted | coin_not_found | "
            "paper_not_detected | too_few_grains | invalid_format | "
            "file_too_large | processing_error"
        ),
    )
    reason: str = Field(..., description="Deterministic explanation for the failure")


class VideoUploadResponse(BaseModel):
    """Response after storing a buyer visual record video."""

    status: str = Field(default="success")
    video_url: str = Field(..., description="URL to the stored video blob")
    duration_seconds: Optional[float] = Field(
        default=None, description="Duration of the uploaded video"
    )
