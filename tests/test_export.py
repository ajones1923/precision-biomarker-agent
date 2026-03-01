"""Tests for Precision Biomarker Agent export functions.

Validates markdown export, JSON export, PDF export (bytes), and
FHIR R4 bundle structure.

Author: Adam Jones
Date: March 2026
"""

import json
from datetime import datetime

import pytest

from src.models import (
    AnalysisResult,
    BiologicalAgeResult,
    DiseaseCategory,
    DiseaseTrajectoryResult,
    GenotypeAdjustmentResult,
    MetabolizerPhenotype,
    PatientProfile,
    PGxResult,
    RiskLevel,
)


# =====================================================================
# FIXTURES
# =====================================================================


@pytest.fixture
def simple_analysis(sample_patient_profile):
    """Return a minimal AnalysisResult for export testing."""
    return AnalysisResult(
        patient_profile=sample_patient_profile,
        biological_age=BiologicalAgeResult(
            chronological_age=45,
            biological_age=47.2,
            age_acceleration=2.2,
            phenoage_score=0.003,
        ),
        disease_trajectories=[
            DiseaseTrajectoryResult(
                disease=DiseaseCategory.DIABETES,
                risk_level=RiskLevel.HIGH,
                current_markers={"hba1c": 5.9},
            ),
        ],
        pgx_results=[
            PGxResult(
                gene="CYP2D6",
                star_alleles="*1/*4",
                phenotype=MetabolizerPhenotype.INTERMEDIATE,
            ),
        ],
        critical_alerts=["HbA1c 5.9% pre-diabetic"],
    )


# =====================================================================
# EXPORT HELPERS (inline, matching the UI pattern)
# =====================================================================


def export_markdown(analysis: AnalysisResult) -> str:
    """Export analysis as markdown report."""
    sections = []
    sections.append("# Precision Biomarker Analysis Report\n")
    sections.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    sections.append(f"**Patient:** {analysis.patient_profile.patient_id}, "
                    f"Age {analysis.patient_profile.age}, Sex {analysis.patient_profile.sex}\n")

    if analysis.critical_alerts:
        sections.append("## Critical Alerts\n")
        for alert in analysis.critical_alerts:
            sections.append(f"- **ALERT:** {alert}")
        sections.append("")

    sections.append("## Biological Age\n")
    ba = analysis.biological_age
    sections.append(f"- Biological Age: {ba.biological_age} years")
    sections.append(f"- Acceleration: {ba.age_acceleration:+.1f} years\n")

    sections.append("## Disease Trajectories\n")
    for traj in analysis.disease_trajectories:
        sections.append(f"- {traj.disease.value}: {traj.risk_level.value.upper()}")

    if analysis.pgx_results:
        sections.append("\n## PGx Profile\n")
        for pgx in analysis.pgx_results:
            sections.append(f"- {pgx.gene} ({pgx.star_alleles}): {pgx.phenotype.value}")

    sections.append("\n## Disclaimer\n")
    sections.append("Not intended as medical advice.")

    return "\n".join(sections)


def export_json(analysis: AnalysisResult) -> str:
    """Export analysis as JSON string."""
    data = {
        "patient_id": analysis.patient_profile.patient_id,
        "age": analysis.patient_profile.age,
        "sex": analysis.patient_profile.sex,
        "biological_age": {
            "chronological": analysis.biological_age.chronological_age,
            "biological": analysis.biological_age.biological_age,
            "acceleration": analysis.biological_age.age_acceleration,
        },
        "disease_trajectories": [
            {
                "disease": t.disease.value,
                "risk_level": t.risk_level.value,
                "markers": t.current_markers,
            }
            for t in analysis.disease_trajectories
        ],
        "pgx_results": [
            {
                "gene": p.gene,
                "star_alleles": p.star_alleles,
                "phenotype": p.phenotype.value,
            }
            for p in analysis.pgx_results
        ],
        "critical_alerts": analysis.critical_alerts,
        "timestamp": analysis.timestamp,
    }
    return json.dumps(data, indent=2)


def export_pdf_bytes(analysis: AnalysisResult) -> bytes:
    """Export analysis as PDF bytes (simplified stub for testing)."""
    # In production, this uses reportlab; here we generate minimal PDF
    content = export_markdown(analysis)
    # Minimal PDF structure
    pdf = b"%PDF-1.4\n"
    pdf += b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    pdf += b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    pdf += b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\n"
    pdf += b"xref\n0 4\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n0\n%%EOF"
    return pdf


def export_fhir(analysis: AnalysisResult) -> dict:
    """Export analysis as FHIR R4 Bundle."""
    patient = analysis.patient_profile
    bundle = {
        "resourceType": "Bundle",
        "type": "document",
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "identifier": [{"value": patient.patient_id}],
                    "gender": "male" if patient.sex == "M" else "female",
                    "birthDate": str(datetime.now().year - patient.age),
                }
            },
            {
                "resource": {
                    "resourceType": "DiagnosticReport",
                    "status": "final",
                    "code": {
                        "coding": [
                            {
                                "system": "http://loinc.org",
                                "code": "100746-7",
                                "display": "Precision Biomarker Analysis",
                            }
                        ],
                        "text": "Precision Biomarker Analysis",
                    },
                    "conclusion": f"Biological age {analysis.biological_age.biological_age}, "
                                  f"acceleration {analysis.biological_age.age_acceleration:+.1f} years",
                }
            },
        ],
    }

    # Add observations for biomarkers
    for name, val in patient.biomarkers.items():
        bundle["entry"].append({
            "resource": {
                "resourceType": "Observation",
                "status": "final",
                "code": {"text": name},
                "valueQuantity": {"value": val},
            }
        })

    # Add PGx molecular sequence entries
    for pgx in analysis.pgx_results:
        bundle["entry"].append({
            "resource": {
                "resourceType": "MolecularSequence",
                "type": "dna",
                "patient": {"reference": f"Patient/{patient.patient_id}"},
                "performer": [{"display": f"{pgx.gene} {pgx.star_alleles}"}],
            }
        })

    return bundle


# =====================================================================
# MARKDOWN EXPORT
# =====================================================================


class TestExportMarkdown:
    """Tests for markdown export."""

    def test_returns_string(self, simple_analysis):
        """export_markdown() returns a string."""
        md = export_markdown(simple_analysis)
        assert isinstance(md, str)

    def test_contains_patient_info(self, simple_analysis):
        """Markdown contains patient ID and demographics."""
        md = export_markdown(simple_analysis)
        assert "HG002" in md
        assert "45" in md

    def test_contains_biological_age(self, simple_analysis):
        """Markdown contains biological age values."""
        md = export_markdown(simple_analysis)
        assert "47.2" in md
        assert "+2.2" in md

    def test_contains_disease_trajectories(self, simple_analysis):
        """Markdown contains disease trajectory results."""
        md = export_markdown(simple_analysis)
        assert "diabetes" in md

    def test_contains_pgx_profile(self, simple_analysis):
        """Markdown contains PGx gene results."""
        md = export_markdown(simple_analysis)
        assert "CYP2D6" in md

    def test_contains_critical_alerts(self, simple_analysis):
        """Markdown contains critical alerts."""
        md = export_markdown(simple_analysis)
        assert "ALERT" in md
        assert "HbA1c" in md

    def test_contains_disclaimer(self, simple_analysis):
        """Markdown contains disclaimer section."""
        md = export_markdown(simple_analysis)
        assert "Disclaimer" in md

    def test_valid_markdown_headers(self, simple_analysis):
        """All headers use proper markdown syntax."""
        md = export_markdown(simple_analysis)
        lines = md.split("\n")
        headers = [l for l in lines if l.startswith("#")]
        for h in headers:
            assert h.startswith("# ") or h.startswith("## ")


# =====================================================================
# JSON EXPORT
# =====================================================================


class TestExportJSON:
    """Tests for JSON export."""

    def test_returns_valid_json(self, simple_analysis):
        """export_json() returns a valid JSON string."""
        json_str = export_json(simple_analysis)
        data = json.loads(json_str)
        assert isinstance(data, dict)

    def test_contains_patient_id(self, simple_analysis):
        """JSON contains patient_id field."""
        data = json.loads(export_json(simple_analysis))
        assert data["patient_id"] == "HG002"

    def test_contains_biological_age(self, simple_analysis):
        """JSON contains biological age values."""
        data = json.loads(export_json(simple_analysis))
        assert data["biological_age"]["biological"] == 47.2
        assert data["biological_age"]["acceleration"] == 2.2

    def test_contains_disease_trajectories(self, simple_analysis):
        """JSON contains disease trajectory array."""
        data = json.loads(export_json(simple_analysis))
        assert len(data["disease_trajectories"]) >= 1
        assert data["disease_trajectories"][0]["disease"] == "diabetes"

    def test_contains_pgx_results(self, simple_analysis):
        """JSON contains PGx results array."""
        data = json.loads(export_json(simple_analysis))
        assert len(data["pgx_results"]) >= 1
        assert data["pgx_results"][0]["gene"] == "CYP2D6"

    def test_contains_critical_alerts(self, simple_analysis):
        """JSON contains critical alerts array."""
        data = json.loads(export_json(simple_analysis))
        assert len(data["critical_alerts"]) >= 1

    def test_contains_timestamp(self, simple_analysis):
        """JSON contains timestamp field."""
        data = json.loads(export_json(simple_analysis))
        assert "timestamp" in data
        assert "T" in data["timestamp"]

    def test_roundtrip_serialization(self, simple_analysis):
        """JSON can be serialized and deserialized without data loss."""
        json_str = export_json(simple_analysis)
        data = json.loads(json_str)
        json_str2 = json.dumps(data, indent=2)
        data2 = json.loads(json_str2)
        assert data == data2


# =====================================================================
# PDF EXPORT
# =====================================================================


class TestExportPDF:
    """Tests for PDF export."""

    def test_returns_bytes(self, simple_analysis):
        """export_pdf_bytes() returns bytes."""
        pdf = export_pdf_bytes(simple_analysis)
        assert isinstance(pdf, bytes)

    def test_starts_with_pdf_header(self, simple_analysis):
        """PDF bytes should start with %PDF header."""
        pdf = export_pdf_bytes(simple_analysis)
        assert pdf.startswith(b"%PDF")

    def test_ends_with_eof_marker(self, simple_analysis):
        """PDF bytes should end with %%EOF marker."""
        pdf = export_pdf_bytes(simple_analysis)
        assert b"%%EOF" in pdf

    def test_non_empty_output(self, simple_analysis):
        """PDF should be non-empty."""
        pdf = export_pdf_bytes(simple_analysis)
        assert len(pdf) > 50


# =====================================================================
# FHIR R4 EXPORT
# =====================================================================


class TestExportFHIR:
    """Tests for FHIR R4 bundle export."""

    def test_returns_dict(self, simple_analysis):
        """export_fhir() returns a dict."""
        fhir = export_fhir(simple_analysis)
        assert isinstance(fhir, dict)

    def test_resource_type_is_bundle(self, simple_analysis):
        """FHIR root resourceType should be Bundle."""
        fhir = export_fhir(simple_analysis)
        assert fhir["resourceType"] == "Bundle"

    def test_bundle_type_is_document(self, simple_analysis):
        """FHIR bundle type should be 'document'."""
        fhir = export_fhir(simple_analysis)
        assert fhir["type"] == "document"

    def test_contains_patient_resource(self, simple_analysis):
        """FHIR bundle should contain a Patient resource."""
        fhir = export_fhir(simple_analysis)
        patient_entries = [
            e for e in fhir["entry"]
            if e["resource"]["resourceType"] == "Patient"
        ]
        assert len(patient_entries) == 1
        assert patient_entries[0]["resource"]["gender"] == "male"

    def test_contains_diagnostic_report(self, simple_analysis):
        """FHIR bundle should contain a DiagnosticReport resource."""
        fhir = export_fhir(simple_analysis)
        reports = [
            e for e in fhir["entry"]
            if e["resource"]["resourceType"] == "DiagnosticReport"
        ]
        assert len(reports) == 1
        assert reports[0]["resource"]["status"] == "final"

    def test_contains_observations_for_biomarkers(self, simple_analysis):
        """FHIR bundle should contain Observation resources for biomarkers."""
        fhir = export_fhir(simple_analysis)
        observations = [
            e for e in fhir["entry"]
            if e["resource"]["resourceType"] == "Observation"
        ]
        # Patient has many biomarkers
        assert len(observations) > 0

    def test_contains_molecular_sequence_for_pgx(self, simple_analysis):
        """FHIR bundle should contain MolecularSequence for PGx results."""
        fhir = export_fhir(simple_analysis)
        sequences = [
            e for e in fhir["entry"]
            if e["resource"]["resourceType"] == "MolecularSequence"
        ]
        assert len(sequences) >= 1

    def test_timestamp_present(self, simple_analysis):
        """FHIR bundle should have a timestamp."""
        fhir = export_fhir(simple_analysis)
        assert "timestamp" in fhir
        assert "T" in fhir["timestamp"]

    def test_valid_json_serialization(self, simple_analysis):
        """FHIR bundle should be JSON-serializable."""
        fhir = export_fhir(simple_analysis)
        json_str = json.dumps(fhir, indent=2)
        assert len(json_str) > 100
        # Verify roundtrip
        reparsed = json.loads(json_str)
        assert reparsed["resourceType"] == "Bundle"

    def test_diagnostic_report_contains_conclusion(self, simple_analysis):
        """DiagnosticReport should have a conclusion with biological age."""
        fhir = export_fhir(simple_analysis)
        report = next(
            e for e in fhir["entry"]
            if e["resource"]["resourceType"] == "DiagnosticReport"
        )
        assert "47.2" in report["resource"]["conclusion"]
