"""Genotype-adjusted reference range engine.

Adjusts standard biomarker reference ranges based on individual genotype,
enabling earlier detection of clinically meaningful deviations. Pure
computation — no LLM or database calls.

Author: Adam Jones
Date: March 2026
"""

from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


# ---------------------------------------------------------------------------
# Adjustment rule definitions: genotype -> biomarker reference range shifts
# ---------------------------------------------------------------------------

ADJUSTMENT_RULES = {
    "PNPLA3_rs738409": {
        "display_name": "PNPLA3 (rs738409)",
        "description": "Patatin-like phospholipase domain-containing 3 — hepatic lipid metabolism",
        "risk_allele": "G",
        "affected_biomarkers": {
            "alt": {
                "unit": "U/L",
                "standard_range": {"lower": 7, "upper": 56},
                "adjustments": {
                    "CC": {
                        "adjusted_range": {"lower": 7, "upper": 56},
                        "rationale": "No PNPLA3 risk alleles. Standard ALT reference range applies.",
                    },
                    "CG": {
                        "adjusted_range": {"lower": 7, "upper": 45},
                        "rationale": (
                            "PNPLA3 CG heterozygote: ~1.5x increased NAFLD risk. "
                            "ALT upper limit adjusted to 45 U/L for earlier detection "
                            "of hepatic steatosis."
                        ),
                    },
                    "GG": {
                        "adjusted_range": {"lower": 7, "upper": 35},
                        "rationale": (
                            "PNPLA3 GG homozygote: 2-3x increased NAFLD risk. "
                            "ALT upper limit adjusted to 35 U/L. Values above this "
                            "threshold warrant liver ultrasound evaluation even if "
                            "within standard reference range."
                        ),
                    },
                },
            },
        },
    },
    "TCF7L2_rs7903146": {
        "display_name": "TCF7L2 (rs7903146)",
        "description": "Transcription factor 7-like 2 — beta-cell function and glucose homeostasis",
        "risk_allele": "T",
        "affected_biomarkers": {
            "fasting_glucose": {
                "unit": "mg/dL",
                "standard_range": {"lower": 70, "upper": 100},
                "adjustments": {
                    "CC": {
                        "adjusted_range": {"lower": 70, "upper": 100},
                        "rationale": "No TCF7L2 risk alleles. Standard fasting glucose range applies.",
                    },
                    "CT": {
                        "adjusted_range": {"lower": 70, "upper": 95},
                        "rationale": (
                            "TCF7L2 CT heterozygote: ~1.4x increased T2D risk. "
                            "Fasting glucose threshold tightened to 95 mg/dL for "
                            "earlier detection of metabolic shift."
                        ),
                    },
                    "TT": {
                        "adjusted_range": {"lower": 70, "upper": 90},
                        "rationale": (
                            "TCF7L2 TT homozygote: ~2x increased T2D risk (strongest "
                            "common genetic risk factor for T2D). Fasting glucose "
                            "threshold tightened to 90 mg/dL. Values 90-100 mg/dL "
                            "in this genotype may indicate early beta-cell dysfunction."
                        ),
                    },
                },
            },
            "hba1c": {
                "unit": "%",
                "standard_range": {"lower": 4.0, "upper": 6.0},
                "adjustments": {
                    "CC": {
                        "adjusted_range": {"lower": 4.0, "upper": 6.0},
                        "rationale": "No TCF7L2 risk alleles. Standard HbA1c range applies.",
                    },
                    "CT": {
                        "adjusted_range": {"lower": 4.0, "upper": 5.8},
                        "rationale": (
                            "TCF7L2 CT heterozygote: HbA1c threshold tightened to 5.8%. "
                            "Earlier intervention recommended given genetic predisposition."
                        ),
                    },
                    "TT": {
                        "adjusted_range": {"lower": 4.0, "upper": 5.5},
                        "rationale": (
                            "TCF7L2 TT homozygote: HbA1c threshold tightened to 5.5%. "
                            "HbA1c 5.5-5.7% in this genotype warrants proactive lifestyle "
                            "intervention and insulin sensitivity testing (HOMA-IR)."
                        ),
                    },
                },
            },
        },
    },
    "APOE": {
        "display_name": "APOE",
        "description": "Apolipoprotein E — lipid transport and Alzheimer's disease risk",
        "risk_allele": "E4",
        "affected_biomarkers": {
            "ldl_c": {
                "unit": "mg/dL",
                "standard_range": {"lower": 0, "upper": 130},
                "adjustments": {
                    "E2/E2": {
                        "adjusted_range": {"lower": 0, "upper": 130},
                        "rationale": (
                            "APOE E2/E2: LDL-C may be artificially low due to impaired "
                            "LDL receptor binding. If LDL is elevated despite E2/E2, "
                            "evaluate for type III dyslipidemia (dysbetalipoproteinemia). "
                            "Check ApoB and triglycerides."
                        ),
                    },
                    "E2/E3": {
                        "adjusted_range": {"lower": 0, "upper": 130},
                        "rationale": "APOE E2/E3: Standard LDL-C reference range applies.",
                    },
                    "E3/E3": {
                        "adjusted_range": {"lower": 0, "upper": 130},
                        "rationale": "APOE E3/E3: Standard LDL-C reference range applies.",
                    },
                    "E3/E4": {
                        "adjusted_range": {"lower": 0, "upper": 115},
                        "rationale": (
                            "APOE E3/E4: One E4 allele increases LDL-C levels and "
                            "Alzheimer's disease risk (~3x). LDL target adjusted to "
                            "<115 mg/dL. More aggressive cardiovascular risk management "
                            "recommended."
                        ),
                    },
                    "E4/E4": {
                        "adjusted_range": {"lower": 0, "upper": 100},
                        "rationale": (
                            "APOE E4/E4: Two E4 alleles significantly increase both "
                            "cardiovascular disease and Alzheimer's disease risk (~12x). "
                            "LDL target adjusted to <100 mg/dL. Aggressive lipid "
                            "management strongly recommended for neuroprotection."
                        ),
                    },
                    "E2/E4": {
                        "adjusted_range": {"lower": 0, "upper": 115},
                        "rationale": (
                            "APOE E2/E4: Mixed effects — E2 may lower and E4 may raise "
                            "LDL-C. LDL target adjusted to <115 mg/dL due to E4 presence."
                        ),
                    },
                },
            },
        },
    },
    "DIO2_rs225014": {
        "display_name": "DIO2 (rs225014, Thr92Ala)",
        "description": "Type 2 deiodinase — T4 to T3 conversion in tissues",
        "risk_allele": "A",
        "affected_biomarkers": {
            "tsh": {
                "unit": "mIU/L",
                "standard_range": {"lower": 0.4, "upper": 4.0},
                "adjustments": {
                    "GG": {
                        "adjusted_range": {"lower": 0.4, "upper": 4.0},
                        "rationale": "DIO2 GG (Thr/Thr): Normal T4-to-T3 conversion. Standard TSH range applies.",
                    },
                    "GA": {
                        "adjusted_range": {"lower": 0.4, "upper": 3.5},
                        "rationale": (
                            "DIO2 GA (Thr/Ala) heterozygote: Mildly impaired T4-to-T3 "
                            "conversion. TSH upper limit adjusted to 3.5 mIU/L. "
                            "Free T3 measurement recommended for complete assessment."
                        ),
                    },
                    "AA": {
                        "adjusted_range": {"lower": 0.4, "upper": 3.0},
                        "rationale": (
                            "DIO2 AA (Ala/Ala) homozygote: Significantly impaired T4-to-T3 "
                            "conversion. TSH upper limit adjusted to 3.0 mIU/L. Patient "
                            "may have normal TSH and free T4 but symptomatic hypothyroidism "
                            "due to low tissue T3. Free T3 measurement is essential. "
                            "May benefit from combination T4+T3 therapy."
                        ),
                    },
                },
            },
            "free_t3": {
                "unit": "pg/mL",
                "standard_range": {"lower": 2.3, "upper": 4.2},
                "adjustments": {
                    "GG": {
                        "adjusted_range": {"lower": 2.3, "upper": 4.2},
                        "rationale": "DIO2 GG: Normal deiodinase activity. Standard free T3 range applies.",
                    },
                    "GA": {
                        "adjusted_range": {"lower": 2.5, "upper": 4.2},
                        "rationale": (
                            "DIO2 GA heterozygote: Free T3 lower limit adjusted to 2.5 pg/mL. "
                            "Values 2.3-2.5 pg/mL may indicate suboptimal T3 production "
                            "in this genotype."
                        ),
                    },
                    "AA": {
                        "adjusted_range": {"lower": 2.8, "upper": 4.2},
                        "rationale": (
                            "DIO2 AA homozygote: Free T3 lower limit adjusted to 2.8 pg/mL. "
                            "Values below 2.8 pg/mL despite normal TSH strongly suggest "
                            "impaired tissue T3 availability. Consider T3 supplementation "
                            "if symptomatic (fatigue, cognitive issues, cold intolerance)."
                        ),
                    },
                },
            },
        },
    },
    "HFE_rs1800562": {
        "display_name": "HFE (rs1800562, C282Y)",
        "description": "Homeostatic iron regulator — hereditary hemochromatosis",
        "risk_allele": "A",
        "affected_biomarkers": {
            "ferritin": {
                "unit": "ng/mL",
                "standard_range_male": {"lower": 30, "upper": 400},
                "standard_range_female": {"lower": 15, "upper": 150},
                "adjustments_male": {
                    "GG": {
                        "adjusted_range": {"lower": 30, "upper": 400},
                        "rationale": "HFE C282Y GG (wildtype): No hemochromatosis risk alleles. Standard ferritin range applies.",
                    },
                    "GA": {
                        "adjusted_range": {"lower": 30, "upper": 300},
                        "rationale": (
                            "HFE C282Y GA (carrier): Upper ferritin limit adjusted to "
                            "300 ng/mL. Carrier status confers mild increased risk of "
                            "iron accumulation, especially if compound heterozygous "
                            "with H63D. Annual monitoring recommended."
                        ),
                    },
                    "AA": {
                        "adjusted_range": {"lower": 30, "upper": 200},
                        "rationale": (
                            "HFE C282Y AA (homozygous): Hereditary hemochromatosis genotype. "
                            "Upper ferritin limit adjusted to 200 ng/mL. Values >200 ng/mL "
                            "warrant evaluation for iron overload. Ferritin >500 ng/mL "
                            "requires liver MRI for iron quantification. Consider "
                            "therapeutic phlebotomy."
                        ),
                    },
                },
                "adjustments_female": {
                    "GG": {
                        "adjusted_range": {"lower": 15, "upper": 150},
                        "rationale": "HFE C282Y GG (wildtype): Standard ferritin range applies.",
                    },
                    "GA": {
                        "adjusted_range": {"lower": 15, "upper": 120},
                        "rationale": (
                            "HFE C282Y GA (carrier, female): Upper ferritin limit adjusted "
                            "to 120 ng/mL. Menstruation provides partial protection against "
                            "iron overload, but monitoring is still recommended."
                        ),
                    },
                    "AA": {
                        "adjusted_range": {"lower": 15, "upper": 100},
                        "rationale": (
                            "HFE C282Y AA (homozygous, female): Hereditary hemochromatosis "
                            "genotype. Upper ferritin limit adjusted to 100 ng/mL. Risk "
                            "of iron overload increases post-menopause. Ferritin >100 ng/mL "
                            "warrants evaluation."
                        ),
                    },
                },
            },
        },
    },
    "FADS1_rs174546": {
        "display_name": "FADS1 (rs174546)",
        "description": "Fatty acid desaturase 1 — omega-3/omega-6 conversion",
        "risk_allele": "C",
        "affected_biomarkers": {
            "omega3_index": {
                "unit": "%",
                "standard_range": {"lower": 8, "upper": 12},
                "adjustments": {
                    "TT": {
                        "adjusted_range": {"lower": 8, "upper": 12},
                        "rationale": (
                            "FADS1 TT: Normal fatty acid desaturase activity. "
                            "Standard omega-3 index target of >=8% applies. "
                            "Plant-based ALA can be converted to EPA/DHA."
                        ),
                    },
                    "CT": {
                        "adjusted_range": {"lower": 6, "upper": 12},
                        "rationale": (
                            "FADS1 CT heterozygote: Reduced ALA-to-EPA/DHA conversion. "
                            "Omega-3 index target adjusted to >=6%. Increased dietary "
                            "intake of preformed EPA/DHA (fatty fish 2-3x/week or "
                            "fish oil 1-2g/day) recommended."
                        ),
                    },
                    "CC": {
                        "adjusted_range": {"lower": 5, "upper": 12},
                        "rationale": (
                            "FADS1 CC homozygote: Severely reduced ALA-to-EPA/DHA "
                            "conversion. Plant-based omega-3 sources (flax, chia, "
                            "walnuts) are insufficient. Requires direct EPA/DHA "
                            "supplementation: fish oil 2-4g/day or algal DHA. "
                            "Omega-3 index target adjusted to >=5%."
                        ),
                    },
                },
            },
        },
    },
}


class GenotypeAdjuster:
    """Adjusts biomarker reference ranges based on individual genotype.

    Standard laboratory reference ranges are population-level norms that
    do not account for individual genetic variation. This engine applies
    genotype-specific adjustments to enable earlier, more precise detection
    of clinically meaningful biomarker deviations.

    Supports 6 gene-biomarker adjustment rules covering liver function,
    glucose metabolism, lipid management, thyroid function, iron metabolism,
    and fatty acid conversion.
    """

    def __init__(self) -> None:
        self._rules = ADJUSTMENT_RULES

    def adjust_single(
        self,
        biomarker: str,
        value: float,
        genotype_key: str,
        genotype_value: str,
        sex: str = "male",
    ) -> Optional[Dict[str, Any]]:
        """Adjust a single biomarker's reference range based on one genotype.

        Args:
            biomarker: Biomarker name (e.g., 'alt', 'ldl_c', 'tsh').
            value: Biomarker measurement value.
            genotype_key: Genotype identifier (e.g., 'PNPLA3_rs738409', 'APOE').
            genotype_value: Genotype value (e.g., 'GG', 'E3/E4').
            sex: Patient sex ('male' or 'female') for sex-stratified ranges.

        Returns:
            Adjustment dict or None if no applicable rule exists.
        """
        rule = self._rules.get(genotype_key)
        if not rule:
            return None

        biomarker_config = rule.get("affected_biomarkers", {}).get(biomarker)
        if not biomarker_config:
            return None

        # Handle sex-stratified adjustments (e.g., ferritin)
        if f"adjustments_{sex.lower()}" in biomarker_config:
            adjustments = biomarker_config[f"adjustments_{sex.lower()}"]
            standard_range = biomarker_config.get(f"standard_range_{sex.lower()}")
        else:
            adjustments = biomarker_config.get("adjustments", {})
            standard_range = biomarker_config.get("standard_range")

        if not standard_range:
            return None

        adjustment = adjustments.get(genotype_value)
        if not adjustment:
            return None

        adjusted_range = adjustment["adjusted_range"]
        unit = biomarker_config.get("unit", "")

        # Determine if value is within standard range but outside adjusted range
        in_standard = standard_range["lower"] <= value <= standard_range["upper"]
        in_adjusted = adjusted_range["lower"] <= value <= adjusted_range["upper"]

        if in_standard and not in_adjusted:
            status = "GENOTYPE_FLAGGED"
            interpretation = (
                f"{biomarker.upper()} {value} {unit} is within standard range "
                f"({standard_range['lower']}-{standard_range['upper']} {unit}) "
                f"but OUTSIDE genotype-adjusted range "
                f"({adjusted_range['lower']}-{adjusted_range['upper']} {unit}). "
                f"This may be clinically significant given {genotype_key} {genotype_value} genotype."
            )
        elif not in_standard and not in_adjusted:
            status = "OUT_OF_RANGE"
            interpretation = (
                f"{biomarker.upper()} {value} {unit} is outside both standard and "
                f"genotype-adjusted reference ranges. Clinical attention recommended."
            )
        elif not in_standard and in_adjusted:
            status = "STANDARD_FLAGGED_ONLY"
            interpretation = (
                f"{biomarker.upper()} {value} {unit} is outside standard range but "
                f"within genotype-adjusted range. May be acceptable for {genotype_key} "
                f"{genotype_value} genotype."
            )
        else:
            status = "NORMAL"
            interpretation = (
                f"{biomarker.upper()} {value} {unit} is within both standard and "
                f"genotype-adjusted reference ranges."
            )

        return {
            "biomarker": biomarker,
            "value": value,
            "unit": unit,
            "genotype_key": genotype_key,
            "genotype_value": genotype_value,
            "gene_display_name": rule["display_name"],
            "gene_description": rule["description"],
            "standard_range": standard_range,
            "adjusted_range": adjusted_range,
            "status": status,
            "interpretation": interpretation,
            "rationale": adjustment["rationale"],
            "range_changed": standard_range != adjusted_range,
        }

    def adjust_all(
        self,
        biomarkers: Dict[str, float],
        genotypes: Dict[str, str],
        sex: str = "male",
    ) -> Dict[str, Any]:
        """Adjust all applicable biomarker reference ranges based on genotypes.

        Args:
            biomarkers: Dict of biomarker name -> value.
            genotypes: Dict of genotype key -> genotype value.
            sex: Patient sex for sex-stratified ranges.

        Returns:
            Dict with adjustments list, flagged items, and summary.
        """
        adjustments = []
        flagged = []
        genotype_flags_count = 0

        for genotype_key, genotype_value in genotypes.items():
            rule = self._rules.get(genotype_key)
            if not rule:
                continue

            for biomarker in rule.get("affected_biomarkers", {}):
                if biomarker not in biomarkers:
                    continue

                result = self.adjust_single(
                    biomarker=biomarker,
                    value=biomarkers[biomarker],
                    genotype_key=genotype_key,
                    genotype_value=genotype_value,
                    sex=sex,
                )

                if result:
                    adjustments.append(result)
                    if result["status"] == "GENOTYPE_FLAGGED":
                        flagged.append(result)
                        genotype_flags_count += 1
                    elif result["status"] == "OUT_OF_RANGE":
                        flagged.append(result)

        # Sort flagged items: GENOTYPE_FLAGGED first (these are the novel findings),
        # then OUT_OF_RANGE
        status_order = {"GENOTYPE_FLAGGED": 0, "OUT_OF_RANGE": 1}
        flagged.sort(key=lambda r: status_order.get(r["status"], 2))

        logger.info(
            f"Genotype adjustment complete: {len(adjustments)} rules applied, "
            f"{genotype_flags_count} genotype-specific flags, "
            f"{len(flagged)} total flagged items"
        )

        return {
            "adjustments": adjustments,
            "flagged": flagged,
            "total_adjustments": len(adjustments),
            "genotype_flags": genotype_flags_count,
            "total_flagged": len(flagged),
            "genotypes_used": list(genotypes.keys()),
            "biomarkers_evaluated": list(biomarkers.keys()),
        }

    def get_adjusted_ranges(
        self,
        genotypes: Dict[str, str],
        sex: str = "male",
    ) -> Dict[str, Dict[str, Any]]:
        """Get all adjusted reference ranges for a patient's genotype profile.

        Useful for displaying personalized reference ranges on a lab report.

        Args:
            genotypes: Dict of genotype key -> genotype value.
            sex: Patient sex for sex-stratified ranges.

        Returns:
            Dict of biomarker -> adjusted range info.
        """
        ranges = {}

        for genotype_key, genotype_value in genotypes.items():
            rule = self._rules.get(genotype_key)
            if not rule:
                continue

            for biomarker, biomarker_config in rule.get("affected_biomarkers", {}).items():
                # Handle sex-stratified adjustments
                if f"adjustments_{sex.lower()}" in biomarker_config:
                    adjustments = biomarker_config[f"adjustments_{sex.lower()}"]
                    standard_range = biomarker_config.get(f"standard_range_{sex.lower()}")
                else:
                    adjustments = biomarker_config.get("adjustments", {})
                    standard_range = biomarker_config.get("standard_range")

                adjustment = adjustments.get(genotype_value)
                if not adjustment or not standard_range:
                    continue

                ranges[biomarker] = {
                    "unit": biomarker_config.get("unit", ""),
                    "standard_range": standard_range,
                    "adjusted_range": adjustment["adjusted_range"],
                    "genotype_key": genotype_key,
                    "genotype_value": genotype_value,
                    "gene_display_name": rule["display_name"],
                    "rationale": adjustment["rationale"],
                    "range_changed": standard_range != adjustment["adjusted_range"],
                }

        return ranges

    def list_supported_adjustments(self) -> List[Dict[str, Any]]:
        """List all supported genotype-biomarker adjustment rules.

        Returns:
            List of dicts describing each adjustment rule.
        """
        supported = []
        for genotype_key, rule in self._rules.items():
            for biomarker, config in rule.get("affected_biomarkers", {}).items():
                # Determine available genotype options
                if "adjustments" in config:
                    genotype_options = list(config["adjustments"].keys())
                elif "adjustments_male" in config:
                    genotype_options = list(config["adjustments_male"].keys())
                else:
                    genotype_options = []

                supported.append({
                    "genotype_key": genotype_key,
                    "gene_display_name": rule["display_name"],
                    "gene_description": rule["description"],
                    "biomarker": biomarker,
                    "risk_allele": rule["risk_allele"],
                    "genotype_options": genotype_options,
                    "sex_stratified": f"adjustments_male" in config,
                })

        return supported
