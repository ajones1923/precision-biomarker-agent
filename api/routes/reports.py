"""Report generation routes for the Precision Biomarker Agent.

Provides endpoints to:
  - Generate full 12-section patient report
  - Download report as PDF
  - Export as FHIR R4 DiagnosticReport

Author: Adam Jones
Date: March 2026
"""

from __future__ import annotations

import io
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix="/v1/report", tags=["reports"])


# =====================================================================
# In-memory report store (placeholder for production persistence)
# =====================================================================

_report_store: Dict[str, Dict[str, Any]] = {}


# =====================================================================
# Request / Response Schemas
# =====================================================================

class PatientProfileRequest(BaseModel):
    """Patient profile for report generation."""
    patient_id: str = Field(..., description="Unique patient identifier")
    age: int = Field(..., ge=0, le=150)
    sex: str = Field(..., max_length=10)
    biomarkers: Dict[str, float] = Field(default_factory=dict)
    genotypes: Dict[str, str] = Field(default_factory=dict)
    star_alleles: Dict[str, str] = Field(default_factory=dict)


class ReportGenerateRequest(BaseModel):
    """Request to generate a full 12-section report."""
    patient_profile: PatientProfileRequest
    format: str = Field("markdown", description="Output format: markdown, json")


class ReportMeta(BaseModel):
    """Metadata about a generated report."""
    report_id: str
    patient_id: str
    format: str
    generated_at: str
    sections: int = 12
    processing_time_ms: float = 0.0


class ReportGenerateResponse(BaseModel):
    """Response from report generation."""
    meta: ReportMeta
    report: str = Field(..., description="Full report content (markdown or JSON)")


class FHIRExportRequest(BaseModel):
    """Request for FHIR R4 DiagnosticReport export."""
    patient_profile: PatientProfileRequest


# =====================================================================
# Endpoints
# =====================================================================

@router.post("/generate", response_model=ReportGenerateResponse)
async def generate_report(request: ReportGenerateRequest, req: Request):
    """Generate a full 12-section precision biomarker patient report.

    Runs all analysis modules (biological age, disease trajectories,
    pharmacogenomics, genotype adjustments) and produces a structured
    report in markdown or JSON format.

    The report is stored in memory and can be later downloaded as PDF
    via the /v1/report/{report_id}/pdf endpoint.
    """
    agent = getattr(req.app.state, "agent", None)
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    t0 = time.perf_counter()

    try:
        from src.models import PatientProfile
        from src.report_generator import ReportGenerator

        profile = PatientProfile(
            patient_id=request.patient_profile.patient_id,
            age=request.patient_profile.age,
            sex=request.patient_profile.sex,
            biomarkers=request.patient_profile.biomarkers,
            genotypes=request.patient_profile.genotypes,
            star_alleles=request.patient_profile.star_alleles,
        )

        # Run full analysis
        analysis = agent.analyze_patient(profile)

        # Generate report
        generator = ReportGenerator()
        report_markdown = generator.generate(analysis, profile)

        elapsed = (time.perf_counter() - t0) * 1000

        # Generate report ID and store
        report_id = uuid.uuid4().hex[:12]
        timestamp = datetime.now(timezone.utc).isoformat()

        _report_store[report_id] = {
            "report_id": report_id,
            "patient_id": profile.patient_id,
            "markdown": report_markdown,
            "analysis": analysis,
            "profile": profile,
            "generated_at": timestamp,
        }

        # Format output
        if request.format == "json":
            from src.export import export_json
            report_content = export_json(
                analysis_result=analysis,
                query=f"Full report for patient {profile.patient_id}",
            )
        else:
            report_content = report_markdown

        meta = ReportMeta(
            report_id=report_id,
            patient_id=profile.patient_id,
            format=request.format,
            generated_at=timestamp,
            sections=12,
            processing_time_ms=round(elapsed, 1),
        )

        return ReportGenerateResponse(
            meta=meta,
            report=report_content,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")


@router.get("/{report_id}/pdf")
async def download_pdf(report_id: str):
    """Download a previously generated report as a styled PDF.

    The report must have been generated first via POST /v1/report/generate.
    Uses reportlab for professional PDF formatting with HCLS AI Factory branding.
    """
    if report_id not in _report_store:
        raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found")

    try:
        from src.export import export_pdf

        stored = _report_store[report_id]
        markdown = stored["markdown"]
        pdf_bytes = export_pdf(markdown)

        patient_id = stored["patient_id"]
        filename = f"biomarker_report_{patient_id}_{report_id}.pdf"

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF export failed: {e}")


@router.post("/fhir")
async def export_fhir(request: FHIRExportRequest, req: Request):
    """Export patient analysis as a FHIR R4 DiagnosticReport JSON bundle.

    Creates a FHIR Bundle containing:
    - DiagnosticReport resource (main report)
    - Observation resources (biological age, disease trajectories, PGx)
    - Patient resource reference

    Useful for EHR integration and interoperability.
    """
    agent = getattr(req.app.state, "agent", None)
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        from src.models import PatientProfile
        from src.export import export_fhir_diagnostic_report

        profile = PatientProfile(
            patient_id=request.patient_profile.patient_id,
            age=request.patient_profile.age,
            sex=request.patient_profile.sex,
            biomarkers=request.patient_profile.biomarkers,
            genotypes=request.patient_profile.genotypes,
            star_alleles=request.patient_profile.star_alleles,
        )

        # Run analysis
        analysis = agent.analyze_patient(profile)

        # Export as FHIR
        fhir_json = export_fhir_diagnostic_report(analysis, profile)

        return JSONResponse(
            content={
                "format": "fhir_r4",
                "resource_type": "Bundle",
                "patient_id": profile.patient_id,
                "bundle": __import__("json").loads(fhir_json),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FHIR export failed: {e}")
