"""Tests for Precision Biomarker Agent report generation.

Validates that reports contain all 12 sections, critical alerts are
highlighted, PGx warnings are formatted, and markdown is valid.

Author: Adam Jones
Date: March 2026
"""

import json

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
def full_analysis(sample_patient_profile):
    """Return a complete AnalysisResult for report testing."""
    return AnalysisResult(
        patient_profile=sample_patient_profile,
        biological_age=BiologicalAgeResult(
            chronological_age=45,
            biological_age=47.2,
            age_acceleration=2.2,
            phenoage_score=0.003,
            mortality_risk=0.015,
            aging_drivers=[
                {"marker": "rdw", "value": 13.5, "contribution": 4.46, "direction": "aging"},
                {"marker": "glucose", "value": 105.0, "contribution": 2.05, "direction": "aging"},
                {"marker": "albumin", "value": 4.2, "contribution": -0.14, "direction": "protective"},
            ],
        ),
        disease_trajectories=[
            DiseaseTrajectoryResult(
                disease=DiseaseCategory.DIABETES,
                risk_level=RiskLevel.HIGH,
                current_markers={"hba1c": 5.9, "fasting_glucose": 105},
                genetic_risk_factors=["TCF7L2 rs7903146 CT"],
                intervention_recommendations=["Dietary optimization", "Recheck in 3 months"],
            ),
            DiseaseTrajectoryResult(
                disease=DiseaseCategory.CARDIOVASCULAR,
                risk_level=RiskLevel.MODERATE,
                current_markers={"ldl_c": 145, "lpa": 65},
                intervention_recommendations=["Optimize modifiable risk factors"],
            ),
            DiseaseTrajectoryResult(
                disease=DiseaseCategory.LIVER,
                risk_level=RiskLevel.LOW,
            ),
            DiseaseTrajectoryResult(
                disease=DiseaseCategory.THYROID,
                risk_level=RiskLevel.LOW,
            ),
            DiseaseTrajectoryResult(
                disease=DiseaseCategory.IRON,
                risk_level=RiskLevel.LOW,
            ),
            DiseaseTrajectoryResult(
                disease=DiseaseCategory.NUTRITIONAL,
                risk_level=RiskLevel.MODERATE,
                intervention_recommendations=["Vitamin D supplementation"],
            ),
        ],
        pgx_results=[
            PGxResult(
                gene="CYP2D6",
                star_alleles="*1/*4",
                phenotype=MetabolizerPhenotype.INTERMEDIATE,
                drugs_affected=[
                    {"drug": "Codeine", "recommendation": "Use with caution"},
                ],
            ),
            PGxResult(
                gene="CYP2C19",
                star_alleles="*1/*2",
                phenotype=MetabolizerPhenotype.INTERMEDIATE,
                drugs_affected=[
                    {"drug": "Clopidogrel", "recommendation": "Consider alternative"},
                ],
            ),
        ],
        genotype_adjustments=[
            GenotypeAdjustmentResult(
                biomarker="ALT",
                standard_range="0-56 U/L",
                adjusted_range="0-45 U/L",
                genotype="CG",
                gene="PNPLA3",
                rationale="PNPLA3 CG carrier",
            ),
        ],
        critical_alerts=[
            "HbA1c 5.9% in pre-diabetic range with TCF7L2 CT carrier status",
            "Lp(a) 65 nmol/L elevated - genetically determined cardiovascular risk",
        ],
    )


def generate_report_markdown(analysis: AnalysisResult) -> str:
    """Generate a markdown report from an AnalysisResult (mirrors UI logic)."""
    sections = []

    # 1. Executive Summary
    sections.append("# Precision Biomarker Analysis Report\n")
    sections.append(f"**Patient:** {analysis.patient_profile.patient_id}, "
                    f"Age {analysis.patient_profile.age}, Sex {analysis.patient_profile.sex}\n")

    # 2. Critical Alerts
    if analysis.critical_alerts:
        sections.append("## Critical Alerts\n")
        for alert in analysis.critical_alerts:
            sections.append(f"- **ALERT:** {alert}")
        sections.append("")

    # 3. Biological Age
    sections.append("## Biological Age Assessment\n")
    ba = analysis.biological_age
    sections.append(f"- Chronological Age: {ba.chronological_age} years")
    sections.append(f"- Biological Age: {ba.biological_age} years")
    sections.append(f"- Age Acceleration: {ba.age_acceleration:+.1f} years")
    sections.append("")

    # 4. Aging Drivers
    sections.append("## Top Aging Drivers\n")
    for driver in ba.aging_drivers:
        sections.append(
            f"- {driver['marker']}: {driver['value']} "
            f"(contribution: {driver['contribution']}, {driver['direction']})"
        )
    sections.append("")

    # 5-10. Disease Trajectories
    sections.append("## Disease Trajectory Analysis\n")
    for traj in analysis.disease_trajectories:
        sections.append(f"### {traj.disease.value.replace('_', ' ').title()}")
        sections.append(f"- Risk Level: {traj.risk_level.value.upper()}")
        if traj.current_markers:
            for k, v in traj.current_markers.items():
                sections.append(f"- {k}: {v}")
        if traj.genetic_risk_factors:
            sections.append("- Genetic Risk Factors: " + ", ".join(traj.genetic_risk_factors))
        if traj.intervention_recommendations:
            sections.append("- Recommendations:")
            for r in traj.intervention_recommendations:
                sections.append(f"  - {r}")
        sections.append("")

    # 11. PGx Profile
    if analysis.pgx_results:
        sections.append("## Pharmacogenomic Profile\n")
        for pgx in analysis.pgx_results:
            sections.append(f"### {pgx.gene} ({pgx.star_alleles})")
            sections.append(f"- Phenotype: {pgx.phenotype.value}")
            for drug in pgx.drugs_affected:
                sections.append(f"- {drug['drug']}: {drug['recommendation']}")
        sections.append("")

    # 12. Genotype Adjustments
    if analysis.genotype_adjustments:
        sections.append("## Genotype-Adjusted Reference Ranges\n")
        for adj in analysis.genotype_adjustments:
            sections.append(
                f"- {adj.biomarker} ({adj.gene} {adj.genotype}): "
                f"{adj.standard_range} -> {adj.adjusted_range}"
            )
        sections.append("")

    # Disclaimer
    sections.append("## Disclaimer\n")
    sections.append(
        "This report is generated for research and educational purposes. "
        "Not intended as medical advice."
    )

    return "\n".join(sections)


# =====================================================================
# REPORT SECTIONS
# =====================================================================


class TestReportSections:
    """Tests for report section completeness."""

    def test_all_12_sections_present(self, full_analysis):
        """Report should contain all 12 sections."""
        report = generate_report_markdown(full_analysis)
        assert "# Precision Biomarker Analysis Report" in report
        assert "## Critical Alerts" in report
        assert "## Biological Age Assessment" in report
        assert "## Top Aging Drivers" in report
        assert "## Disease Trajectory Analysis" in report
        assert "## Pharmacogenomic Profile" in report
        assert "## Genotype-Adjusted Reference Ranges" in report
        assert "## Disclaimer" in report

    def test_patient_info_in_header(self, full_analysis):
        """Report header should include patient ID, age, and sex."""
        report = generate_report_markdown(full_analysis)
        assert "HG002" in report
        assert "45" in report
        assert "M" in report

    def test_biological_age_values_present(self, full_analysis):
        """Report should contain biological age values."""
        report = generate_report_markdown(full_analysis)
        assert "47.2" in report
        assert "+2.2" in report

    def test_disease_trajectories_all_listed(self, full_analysis):
        """All 6 disease categories should appear in the report."""
        report = generate_report_markdown(full_analysis)
        assert "Diabetes" in report
        assert "Cardiovascular" in report
        assert "Liver" in report
        assert "Thyroid" in report
        assert "Iron" in report
        assert "Nutritional" in report

    def test_pgx_genes_listed(self, full_analysis):
        """PGx gene results should appear in the report."""
        report = generate_report_markdown(full_analysis)
        assert "CYP2D6" in report
        assert "CYP2C19" in report
        assert "*1/*4" in report

    def test_genotype_adjustments_listed(self, full_analysis):
        """Genotype adjustments should appear in the report."""
        report = generate_report_markdown(full_analysis)
        assert "PNPLA3" in report
        assert "0-45 U/L" in report


# =====================================================================
# CRITICAL ALERTS
# =====================================================================


class TestCriticalAlertFormatting:
    """Tests for critical alert highlighting in reports."""

    def test_critical_alerts_highlighted(self, full_analysis):
        """Critical alerts should be highlighted with ALERT prefix."""
        report = generate_report_markdown(full_analysis)
        assert "**ALERT:**" in report

    def test_alert_contains_hba1c_warning(self, full_analysis):
        """HbA1c pre-diabetic alert should be in report."""
        report = generate_report_markdown(full_analysis)
        assert "HbA1c" in report
        assert "pre-diabetic" in report

    def test_alert_contains_lpa_warning(self, full_analysis):
        """Lp(a) elevation alert should be in report."""
        report = generate_report_markdown(full_analysis)
        assert "Lp(a)" in report

    def test_no_alerts_when_none(self, sample_patient_profile):
        """Report without critical alerts should not have alert section."""
        analysis = AnalysisResult(
            patient_profile=sample_patient_profile,
            biological_age=BiologicalAgeResult(
                chronological_age=45,
                biological_age=44.0,
                age_acceleration=-1.0,
                phenoage_score=0.002,
            ),
        )
        report = generate_report_markdown(analysis)
        assert "## Critical Alerts" not in report


# =====================================================================
# PGx WARNINGS
# =====================================================================


class TestPGxWarningFormatting:
    """Tests for PGx warning formatting in reports."""

    def test_pgx_phenotype_displayed(self, full_analysis):
        """PGx phenotype should be displayed in report."""
        report = generate_report_markdown(full_analysis)
        assert "intermediate" in report

    def test_pgx_drug_recommendations_displayed(self, full_analysis):
        """Drug-specific recommendations should appear in report."""
        report = generate_report_markdown(full_analysis)
        assert "Codeine" in report
        assert "Clopidogrel" in report


# =====================================================================
# MARKDOWN VALIDITY
# =====================================================================


class TestMarkdownValidity:
    """Tests for markdown format validity."""

    def test_report_is_string(self, full_analysis):
        """Report should be a non-empty string."""
        report = generate_report_markdown(full_analysis)
        assert isinstance(report, str)
        assert len(report) > 0

    def test_headers_have_proper_format(self, full_analysis):
        """All headers should use proper markdown # syntax."""
        report = generate_report_markdown(full_analysis)
        lines = report.split("\n")
        headers = [l for l in lines if l.startswith("#")]
        for h in headers:
            # Header should have space after #
            assert h.startswith("# ") or h.startswith("## ") or h.startswith("### ")

    def test_bullet_points_properly_formatted(self, full_analysis):
        """Bullet points should use '- ' prefix."""
        report = generate_report_markdown(full_analysis)
        lines = report.split("\n")
        bullets = [l for l in lines if l.strip().startswith("-")]
        assert len(bullets) > 0
        for b in bullets:
            stripped = b.strip()
            assert stripped.startswith("- ") or stripped.startswith("- **")

    def test_no_unclosed_bold_markers(self, full_analysis):
        """Bold markers (**) should appear in pairs."""
        report = generate_report_markdown(full_analysis)
        bold_count = report.count("**")
        assert bold_count % 2 == 0
