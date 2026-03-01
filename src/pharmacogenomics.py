"""Pharmacogenomic mapping engine.

Maps star alleles and genotypes to drug recommendations following CPIC
Level 1A guidelines. Pure computation — no LLM or database calls.

Author: Adam Jones
Date: March 2026
"""

from typing import Any, Dict, List, Optional

from loguru import logger


# ---------------------------------------------------------------------------
# PGx gene configurations: star alleles -> phenotype -> drug recommendations
# All mappings follow CPIC Level 1A guidelines
# ---------------------------------------------------------------------------

PGX_GENE_CONFIGS = {
    "CYP2D6": {
        "display_name": "CYP2D6",
        "description": "Cytochrome P450 2D6 — metabolizes ~25% of drugs",
        "allele_to_phenotype": {
            "*1/*1": "Normal Metabolizer",
            "*1/*2": "Normal Metabolizer",
            "*2/*2": "Normal Metabolizer",
            "*1/*4": "Intermediate Metabolizer",
            "*1/*5": "Intermediate Metabolizer",
            "*1/*10": "Intermediate Metabolizer",
            "*2/*4": "Intermediate Metabolizer",
            "*4/*10": "Intermediate Metabolizer",
            "*10/*10": "Intermediate Metabolizer",
            "*4/*4": "Poor Metabolizer",
            "*4/*5": "Poor Metabolizer",
            "*5/*5": "Poor Metabolizer",
            "*4/*6": "Poor Metabolizer",
            "*1/*1xN": "Ultra-rapid Metabolizer",
            "*1/*2xN": "Ultra-rapid Metabolizer",
            "*2/*2xN": "Ultra-rapid Metabolizer",
        },
        "drug_recommendations": {
            "codeine": {
                "Normal Metabolizer": {
                    "recommendation": "Use codeine per standard dosing",
                    "action": "STANDARD_DOSING",
                    "alert_level": "INFO",
                },
                "Intermediate Metabolizer": {
                    "recommendation": "Use codeine with caution; consider alternative analgesics (morphine, non-opioid). Reduced conversion to morphine expected",
                    "action": "CONSIDER_ALTERNATIVE",
                    "alert_level": "WARNING",
                },
                "Poor Metabolizer": {
                    "recommendation": "AVOID codeine — no conversion to morphine, will be ineffective. Use morphine, non-opioid analgesics, or non-tramadol opioids",
                    "action": "AVOID",
                    "alert_level": "CRITICAL",
                },
                "Ultra-rapid Metabolizer": {
                    "recommendation": "AVOID codeine — excess conversion to morphine, risk of fatal respiratory depression. Use non-opioid analgesics or non-tramadol alternatives",
                    "action": "AVOID",
                    "alert_level": "CRITICAL",
                },
            },
            "tramadol": {
                "Normal Metabolizer": {
                    "recommendation": "Use tramadol per standard dosing",
                    "action": "STANDARD_DOSING",
                    "alert_level": "INFO",
                },
                "Intermediate Metabolizer": {
                    "recommendation": "Use tramadol with caution; may have reduced efficacy",
                    "action": "CONSIDER_ALTERNATIVE",
                    "alert_level": "WARNING",
                },
                "Poor Metabolizer": {
                    "recommendation": "AVOID tramadol — reduced efficacy due to impaired O-demethylation. Use alternative analgesics",
                    "action": "AVOID",
                    "alert_level": "CRITICAL",
                },
                "Ultra-rapid Metabolizer": {
                    "recommendation": "AVOID tramadol — risk of respiratory depression from excess O-desmethyltramadol. Use non-opioid analgesics",
                    "action": "AVOID",
                    "alert_level": "CRITICAL",
                },
            },
            "tamoxifen": {
                "Normal Metabolizer": {
                    "recommendation": "Use tamoxifen per standard dosing (20 mg/day)",
                    "action": "STANDARD_DOSING",
                    "alert_level": "INFO",
                },
                "Intermediate Metabolizer": {
                    "recommendation": "Consider higher tamoxifen dose (40 mg/day) or switch to aromatase inhibitor if postmenopausal",
                    "action": "DOSE_ADJUSTMENT",
                    "alert_level": "WARNING",
                },
                "Poor Metabolizer": {
                    "recommendation": "AVOID tamoxifen — greatly reduced conversion to active endoxifen. Use aromatase inhibitor if postmenopausal; consider alternative endocrine therapy if premenopausal",
                    "action": "AVOID",
                    "alert_level": "CRITICAL",
                },
                "Ultra-rapid Metabolizer": {
                    "recommendation": "Use tamoxifen per standard dosing; may have increased active metabolite levels",
                    "action": "STANDARD_DOSING",
                    "alert_level": "INFO",
                },
            },
            "ondansetron": {
                "Normal Metabolizer": {
                    "recommendation": "Use ondansetron per standard dosing",
                    "action": "STANDARD_DOSING",
                    "alert_level": "INFO",
                },
                "Intermediate Metabolizer": {
                    "recommendation": "Use ondansetron per standard dosing",
                    "action": "STANDARD_DOSING",
                    "alert_level": "INFO",
                },
                "Poor Metabolizer": {
                    "recommendation": "Ondansetron may have increased exposure. Consider dose reduction or alternative antiemetic (granisetron)",
                    "action": "DOSE_ADJUSTMENT",
                    "alert_level": "WARNING",
                },
                "Ultra-rapid Metabolizer": {
                    "recommendation": "Ondansetron may have reduced efficacy. Consider alternative antiemetic or higher dose",
                    "action": "CONSIDER_ALTERNATIVE",
                    "alert_level": "WARNING",
                },
            },
        },
    },
    "CYP2C19": {
        "display_name": "CYP2C19",
        "description": "Cytochrome P450 2C19 — critical for clopidogrel activation",
        "allele_to_phenotype": {
            "*1/*1": "Normal Metabolizer",
            "*1/*2": "Intermediate Metabolizer",
            "*1/*3": "Intermediate Metabolizer",
            "*2/*17": "Intermediate Metabolizer",
            "*2/*2": "Poor Metabolizer",
            "*2/*3": "Poor Metabolizer",
            "*3/*3": "Poor Metabolizer",
            "*1/*17": "Rapid Metabolizer",
            "*17/*17": "Ultra-rapid Metabolizer",
        },
        "drug_recommendations": {
            "clopidogrel": {
                "Normal Metabolizer": {
                    "recommendation": "Use clopidogrel per standard dosing",
                    "action": "STANDARD_DOSING",
                    "alert_level": "INFO",
                },
                "Intermediate Metabolizer": {
                    "recommendation": "Clopidogrel may have reduced efficacy. Consider alternative antiplatelet (prasugrel or ticagrelor) especially for ACS/PCI",
                    "action": "CONSIDER_ALTERNATIVE",
                    "alert_level": "WARNING",
                },
                "Poor Metabolizer": {
                    "recommendation": "AVOID clopidogrel — significantly reduced platelet inhibition. Use prasugrel or ticagrelor (FDA boxed warning)",
                    "action": "AVOID",
                    "alert_level": "CRITICAL",
                },
                "Rapid Metabolizer": {
                    "recommendation": "Use clopidogrel per standard dosing; expected normal response",
                    "action": "STANDARD_DOSING",
                    "alert_level": "INFO",
                },
                "Ultra-rapid Metabolizer": {
                    "recommendation": "Use clopidogrel per standard dosing; may have enhanced response",
                    "action": "STANDARD_DOSING",
                    "alert_level": "INFO",
                },
            },
            "citalopram": {
                "Normal Metabolizer": {
                    "recommendation": "Use citalopram per standard dosing (20-40 mg/day)",
                    "action": "STANDARD_DOSING",
                    "alert_level": "INFO",
                },
                "Intermediate Metabolizer": {
                    "recommendation": "Use citalopram per standard dosing; monitor for side effects",
                    "action": "STANDARD_DOSING",
                    "alert_level": "INFO",
                },
                "Poor Metabolizer": {
                    "recommendation": "Reduce citalopram dose by 50% (max 20 mg/day). QTc prolongation risk at standard doses. Consider alternative SSRI (sertraline)",
                    "action": "DOSE_REDUCTION",
                    "alert_level": "WARNING",
                },
                "Rapid Metabolizer": {
                    "recommendation": "Use citalopram per standard dosing",
                    "action": "STANDARD_DOSING",
                    "alert_level": "INFO",
                },
                "Ultra-rapid Metabolizer": {
                    "recommendation": "Citalopram may have reduced efficacy. Consider alternative SSRI or dose increase with monitoring",
                    "action": "CONSIDER_ALTERNATIVE",
                    "alert_level": "WARNING",
                },
            },
            "escitalopram": {
                "Normal Metabolizer": {
                    "recommendation": "Use escitalopram per standard dosing (10-20 mg/day)",
                    "action": "STANDARD_DOSING",
                    "alert_level": "INFO",
                },
                "Intermediate Metabolizer": {
                    "recommendation": "Use escitalopram per standard dosing",
                    "action": "STANDARD_DOSING",
                    "alert_level": "INFO",
                },
                "Poor Metabolizer": {
                    "recommendation": "Reduce escitalopram dose by 50% (max 10 mg/day). Consider alternative SSRI (sertraline)",
                    "action": "DOSE_REDUCTION",
                    "alert_level": "WARNING",
                },
                "Rapid Metabolizer": {
                    "recommendation": "Use escitalopram per standard dosing",
                    "action": "STANDARD_DOSING",
                    "alert_level": "INFO",
                },
                "Ultra-rapid Metabolizer": {
                    "recommendation": "Escitalopram may have reduced efficacy. Consider alternative SSRI or dose increase",
                    "action": "CONSIDER_ALTERNATIVE",
                    "alert_level": "WARNING",
                },
            },
            "omeprazole": {
                "Normal Metabolizer": {
                    "recommendation": "Use omeprazole per standard dosing",
                    "action": "STANDARD_DOSING",
                    "alert_level": "INFO",
                },
                "Intermediate Metabolizer": {
                    "recommendation": "Use omeprazole per standard dosing",
                    "action": "STANDARD_DOSING",
                    "alert_level": "INFO",
                },
                "Poor Metabolizer": {
                    "recommendation": "Omeprazole exposure increased 4-12x. Consider 50% dose reduction for long-term use. Monitor for adverse effects",
                    "action": "DOSE_REDUCTION",
                    "alert_level": "WARNING",
                },
                "Rapid Metabolizer": {
                    "recommendation": "May need increased omeprazole dose for H. pylori eradication",
                    "action": "DOSE_ADJUSTMENT",
                    "alert_level": "INFO",
                },
                "Ultra-rapid Metabolizer": {
                    "recommendation": "Increase omeprazole dose (2-3x standard) or switch to rabeprazole (less CYP2C19 dependent)",
                    "action": "DOSE_ADJUSTMENT",
                    "alert_level": "WARNING",
                },
            },
            "voriconazole": {
                "Normal Metabolizer": {
                    "recommendation": "Use voriconazole per standard dosing",
                    "action": "STANDARD_DOSING",
                    "alert_level": "INFO",
                },
                "Intermediate Metabolizer": {
                    "recommendation": "Use voriconazole per standard dosing; therapeutic drug monitoring recommended",
                    "action": "STANDARD_DOSING",
                    "alert_level": "INFO",
                },
                "Poor Metabolizer": {
                    "recommendation": "Voriconazole exposure significantly increased. Reduce dose and perform therapeutic drug monitoring. Consider alternative antifungal",
                    "action": "DOSE_REDUCTION",
                    "alert_level": "WARNING",
                },
                "Rapid Metabolizer": {
                    "recommendation": "Voriconazole may be subtherapeutic. Consider alternative antifungal or therapeutic drug monitoring with dose adjustment",
                    "action": "CONSIDER_ALTERNATIVE",
                    "alert_level": "WARNING",
                },
                "Ultra-rapid Metabolizer": {
                    "recommendation": "AVOID voriconazole — likely subtherapeutic levels. Use alternative antifungal (isavuconazole, posaconazole)",
                    "action": "AVOID",
                    "alert_level": "CRITICAL",
                },
            },
        },
    },
    "SLCO1B1": {
        "display_name": "SLCO1B1",
        "description": "Solute carrier organic anion transporter — statin hepatic uptake",
        "genotype_field": "SLCO1B1_rs4149056",
        "genotype_to_phenotype": {
            "TT": "Normal Function",
            "TC": "Intermediate Function",
            "CC": "Poor Function",
        },
        "drug_recommendations": {
            "simvastatin": {
                "Normal Function": {
                    "recommendation": "Use simvastatin per standard dosing",
                    "action": "STANDARD_DOSING",
                    "alert_level": "INFO",
                },
                "Intermediate Function": {
                    "recommendation": "Simvastatin max 20 mg/day. AVOID 80mg dose. Consider switching to rosuvastatin or pitavastatin (less SLCO1B1 dependent)",
                    "action": "DOSE_REDUCTION",
                    "alert_level": "WARNING",
                },
                "Poor Function": {
                    "recommendation": "AVOID simvastatin — high risk of myopathy/rhabdomyolysis. Use rosuvastatin or pitavastatin instead",
                    "action": "AVOID",
                    "alert_level": "CRITICAL",
                },
            },
            "atorvastatin": {
                "Normal Function": {
                    "recommendation": "Use atorvastatin per standard dosing",
                    "action": "STANDARD_DOSING",
                    "alert_level": "INFO",
                },
                "Intermediate Function": {
                    "recommendation": "Use atorvastatin with caution; lower starting dose. Monitor for myopathy symptoms",
                    "action": "DOSE_ADJUSTMENT",
                    "alert_level": "WARNING",
                },
                "Poor Function": {
                    "recommendation": "Atorvastatin at reduced dose with close monitoring, or switch to rosuvastatin/pitavastatin. Monitor CK levels",
                    "action": "DOSE_REDUCTION",
                    "alert_level": "WARNING",
                },
            },
            "rosuvastatin": {
                "Normal Function": {
                    "recommendation": "Use rosuvastatin per standard dosing",
                    "action": "STANDARD_DOSING",
                    "alert_level": "INFO",
                },
                "Intermediate Function": {
                    "recommendation": "Use rosuvastatin per standard dosing; less affected by SLCO1B1 variants",
                    "action": "STANDARD_DOSING",
                    "alert_level": "INFO",
                },
                "Poor Function": {
                    "recommendation": "Use rosuvastatin at standard dose; preferred statin for SLCO1B1 poor function. Alternatively consider pitavastatin",
                    "action": "STANDARD_DOSING",
                    "alert_level": "INFO",
                },
            },
        },
    },
    "VKORC1": {
        "display_name": "VKORC1",
        "description": "Vitamin K epoxide reductase — warfarin target",
        "genotype_field": "VKORC1_rs9923231",
        "genotype_to_phenotype": {
            "GG": "Normal Sensitivity",
            "AG": "Intermediate Sensitivity",
            "GA": "Intermediate Sensitivity",
            "AA": "High Sensitivity",
        },
        "drug_recommendations": {
            "warfarin": {
                "Normal Sensitivity": {
                    "recommendation": "Standard warfarin dosing range (5-7 mg/day). Titrate based on INR",
                    "action": "STANDARD_DOSING",
                    "alert_level": "INFO",
                },
                "Intermediate Sensitivity": {
                    "recommendation": "Reduce warfarin dose (3-4 mg/day expected maintenance). VKORC1 AG: ~25% dose reduction. Titrate based on INR",
                    "action": "DOSE_REDUCTION",
                    "alert_level": "WARNING",
                },
                "High Sensitivity": {
                    "recommendation": "Low warfarin dose required (1-2 mg/day expected maintenance). VKORC1 AA: ~50% dose reduction. High bleeding risk — close INR monitoring essential",
                    "action": "DOSE_REDUCTION",
                    "alert_level": "WARNING",
                },
            },
        },
    },
    "MTHFR": {
        "display_name": "MTHFR",
        "description": "Methylenetetrahydrofolate reductase — folate metabolism",
        "genotype_field": "MTHFR_rs1801133",
        "genotype_to_phenotype": {
            "CC": "Normal Activity",
            "CT": "Intermediate Activity",
            "TC": "Intermediate Activity",
            "TT": "Reduced Activity",
        },
        "drug_recommendations": {
            "methotrexate": {
                "Normal Activity": {
                    "recommendation": "Use methotrexate per standard dosing with standard folate supplementation",
                    "action": "STANDARD_DOSING",
                    "alert_level": "INFO",
                },
                "Intermediate Activity": {
                    "recommendation": "Monitor for methotrexate toxicity. Consider dose adjustment if adverse effects occur. Ensure adequate folate supplementation",
                    "action": "DOSE_ADJUSTMENT",
                    "alert_level": "WARNING",
                },
                "Reduced Activity": {
                    "recommendation": "5-7x increased methotrexate toxicity risk. Dose reduction recommended. Use leucovorin rescue. Supplement with L-methylfolate (NOT folic acid). Close CBC monitoring",
                    "action": "DOSE_REDUCTION",
                    "alert_level": "CRITICAL",
                },
            },
            "folate_supplementation": {
                "Normal Activity": {
                    "recommendation": "Standard folic acid supplementation is effective",
                    "action": "STANDARD_DOSING",
                    "alert_level": "INFO",
                },
                "Intermediate Activity": {
                    "recommendation": "Consider L-methylfolate instead of folic acid for optimal folate status",
                    "action": "DOSE_ADJUSTMENT",
                    "alert_level": "INFO",
                },
                "Reduced Activity": {
                    "recommendation": "Use L-methylfolate (1-5 mg/day) instead of folic acid. Folic acid cannot be efficiently converted to active form (5-MTHF). Unmetabolized folic acid may accumulate",
                    "action": "DOSE_ADJUSTMENT",
                    "alert_level": "WARNING",
                },
            },
        },
    },
    "TPMT": {
        "display_name": "TPMT",
        "description": "Thiopurine S-methyltransferase — thiopurine metabolism",
        "allele_to_phenotype": {
            "*1/*1": "Normal Metabolizer",
            "*1/*3A": "Intermediate Metabolizer",
            "*1/*3C": "Intermediate Metabolizer",
            "*1/*2": "Intermediate Metabolizer",
            "*3A/*3A": "Poor Metabolizer",
            "*3A/*3C": "Poor Metabolizer",
            "*3C/*3C": "Poor Metabolizer",
            "*2/*3A": "Poor Metabolizer",
        },
        "drug_recommendations": {
            "azathioprine": {
                "Normal Metabolizer": {
                    "recommendation": "Use azathioprine per standard dosing (2-3 mg/kg/day)",
                    "action": "STANDARD_DOSING",
                    "alert_level": "INFO",
                },
                "Intermediate Metabolizer": {
                    "recommendation": "Reduce azathioprine dose by 50% (1-1.5 mg/kg/day). Monitor CBC weekly for first month, then monthly",
                    "action": "DOSE_REDUCTION",
                    "alert_level": "WARNING",
                },
                "Poor Metabolizer": {
                    "recommendation": "Reduce azathioprine to 10-15% of standard dose or AVOID. Severe myelosuppression risk (life-threatening). If used, start at 0.5 mg/kg/day with intensive CBC monitoring",
                    "action": "AVOID",
                    "alert_level": "CRITICAL",
                },
            },
            "6-mercaptopurine": {
                "Normal Metabolizer": {
                    "recommendation": "Use 6-mercaptopurine per standard dosing",
                    "action": "STANDARD_DOSING",
                    "alert_level": "INFO",
                },
                "Intermediate Metabolizer": {
                    "recommendation": "Reduce 6-mercaptopurine dose by 50%. Monitor CBC closely",
                    "action": "DOSE_REDUCTION",
                    "alert_level": "WARNING",
                },
                "Poor Metabolizer": {
                    "recommendation": "Reduce 6-mercaptopurine to 10-15% of standard dose. Severe myelosuppression risk. Consider alternative therapy",
                    "action": "AVOID",
                    "alert_level": "CRITICAL",
                },
            },
            "thioguanine": {
                "Normal Metabolizer": {
                    "recommendation": "Use thioguanine per standard dosing",
                    "action": "STANDARD_DOSING",
                    "alert_level": "INFO",
                },
                "Intermediate Metabolizer": {
                    "recommendation": "Reduce thioguanine dose by 50%. Monitor for myelosuppression",
                    "action": "DOSE_REDUCTION",
                    "alert_level": "WARNING",
                },
                "Poor Metabolizer": {
                    "recommendation": "Reduce thioguanine to 10-15% of standard dose. Severe myelosuppression risk",
                    "action": "AVOID",
                    "alert_level": "CRITICAL",
                },
            },
        },
    },
    "HLA-B*57:01": {
        "display_name": "HLA-B*57:01",
        "description": "Human leukocyte antigen B*57:01 — abacavir hypersensitivity",
        "genotype_field": "HLA_B5701",
        "genotype_to_phenotype": {
            "negative": "Negative",
            "Negative": "Negative",
            "positive": "Positive",
            "Positive": "Positive",
            "carrier": "Positive",
            "Carrier": "Positive",
        },
        "drug_recommendations": {
            "abacavir": {
                "Negative": {
                    "recommendation": "Abacavir use is permitted. HLA-B*57:01 negative — low risk of hypersensitivity reaction",
                    "action": "STANDARD_DOSING",
                    "alert_level": "INFO",
                },
                "Positive": {
                    "recommendation": "CONTRAINDICATED: Do NOT prescribe abacavir. HLA-B*57:01 positive — 5-8% risk of severe, potentially fatal hypersensitivity reaction. FDA-mandated screening. Use alternative NRTI (tenofovir)",
                    "action": "CONTRAINDICATED",
                    "alert_level": "CRITICAL",
                },
            },
        },
    },
}


class PharmacogenomicMapper:
    """Maps star alleles and genotypes to drug recommendations.

    Follows CPIC Level 1A guidelines for 7 pharmacogenes covering
    major drug classes including analgesics, antiplatelets, statins,
    anticoagulants, antimetabolites, thiopurines, and antiretrovirals.
    """

    def __init__(self) -> None:
        self._gene_configs = PGX_GENE_CONFIGS

    def _resolve_phenotype(
        self,
        gene: str,
        star_alleles: Optional[str] = None,
        genotype: Optional[str] = None,
    ) -> Optional[str]:
        """Resolve phenotype from star alleles or genotype.

        Args:
            gene: Gene name (e.g., 'CYP2D6').
            star_alleles: Star allele notation (e.g., '*1/*4').
            genotype: Genotype notation (e.g., 'TC').

        Returns:
            Phenotype string or None if not resolved.
        """
        config = self._gene_configs.get(gene)
        if not config:
            return None

        # Try star allele lookup first
        if star_alleles and "allele_to_phenotype" in config:
            phenotype = config["allele_to_phenotype"].get(star_alleles)
            if phenotype:
                return phenotype

        # Try genotype lookup
        if genotype and "genotype_to_phenotype" in config:
            phenotype = config["genotype_to_phenotype"].get(genotype)
            if phenotype:
                return phenotype

        return None

    def map_gene(
        self,
        gene: str,
        star_alleles: Optional[str] = None,
        genotype: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Map a single gene to drug recommendations.

        Args:
            gene: Gene name (e.g., 'CYP2D6', 'SLCO1B1').
            star_alleles: Star allele notation (e.g., '*1/*4').
            genotype: Genotype notation (e.g., 'TC' for SLCO1B1).

        Returns:
            Dict with gene, phenotype, affected drugs, and recommendations.
        """
        config = self._gene_configs.get(gene)
        if not config:
            logger.warning(f"Unknown PGx gene: {gene}")
            return {
                "gene": gene,
                "phenotype": None,
                "error": f"Gene {gene} not in PGx database",
                "affected_drugs": [],
                "critical_alerts": [],
            }

        phenotype = self._resolve_phenotype(gene, star_alleles, genotype)
        if not phenotype:
            allele_info = star_alleles or genotype or "not provided"
            logger.warning(f"Could not resolve phenotype for {gene} ({allele_info})")
            return {
                "gene": gene,
                "display_name": config["display_name"],
                "description": config["description"],
                "star_alleles": star_alleles,
                "genotype": genotype,
                "phenotype": None,
                "error": f"Unrecognized allele/genotype: {allele_info}",
                "affected_drugs": [],
                "critical_alerts": [],
            }

        affected_drugs = []
        critical_alerts = []

        for drug, phenotype_recs in config.get("drug_recommendations", {}).items():
            rec = phenotype_recs.get(phenotype)
            if rec:
                drug_result = {
                    "drug": drug,
                    "recommendation": rec["recommendation"],
                    "action": rec["action"],
                    "alert_level": rec["alert_level"],
                }
                affected_drugs.append(drug_result)

                if rec["alert_level"] == "CRITICAL":
                    critical_alerts.append({
                        "drug": drug,
                        "gene": gene,
                        "phenotype": phenotype,
                        "action": rec["action"],
                        "message": rec["recommendation"],
                    })

        return {
            "gene": gene,
            "display_name": config["display_name"],
            "description": config["description"],
            "star_alleles": star_alleles,
            "genotype": genotype,
            "phenotype": phenotype,
            "affected_drugs": affected_drugs,
            "critical_alerts": critical_alerts,
        }

    def map_all(
        self,
        star_alleles: Optional[Dict[str, str]] = None,
        genotypes: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Map all provided PGx genes to drug recommendations.

        Args:
            star_alleles: Dict of gene -> star alleles (e.g.,
                {'CYP2D6': '*1/*4', 'TPMT': '*1/*3A'}).
            genotypes: Dict of rsID or gene -> genotype (e.g.,
                {'SLCO1B1_rs4149056': 'TC', 'VKORC1_rs9923231': 'AG',
                 'MTHFR_rs1801133': 'CT', 'HLA_B5701': 'positive'}).

        Returns:
            Dict with gene_results list, all critical_alerts, and summary.
        """
        star_alleles = star_alleles or {}
        genotypes = genotypes or {}

        gene_results = []
        all_critical_alerts = []
        drugs_to_avoid = []
        drugs_to_adjust = []

        # Process star allele genes
        for gene, alleles in star_alleles.items():
            result = self.map_gene(gene, star_alleles=alleles)
            gene_results.append(result)
            all_critical_alerts.extend(result.get("critical_alerts", []))

        # Process genotype-based genes
        # Map genotype field names to gene names
        genotype_field_to_gene = {}
        for gene, config in self._gene_configs.items():
            field = config.get("genotype_field")
            if field:
                genotype_field_to_gene[field] = gene

        for field, gt in genotypes.items():
            gene = genotype_field_to_gene.get(field)
            if gene:
                # Skip if already processed via star alleles
                if gene not in star_alleles:
                    result = self.map_gene(gene, genotype=gt)
                    gene_results.append(result)
                    all_critical_alerts.extend(result.get("critical_alerts", []))

        # Compile drug action lists
        for result in gene_results:
            for drug_rec in result.get("affected_drugs", []):
                if drug_rec["action"] in ("AVOID", "CONTRAINDICATED"):
                    drugs_to_avoid.append({
                        "drug": drug_rec["drug"],
                        "gene": result["gene"],
                        "phenotype": result["phenotype"],
                        "reason": drug_rec["recommendation"],
                    })
                elif drug_rec["action"] in ("DOSE_REDUCTION", "DOSE_ADJUSTMENT", "CONSIDER_ALTERNATIVE"):
                    drugs_to_adjust.append({
                        "drug": drug_rec["drug"],
                        "gene": result["gene"],
                        "phenotype": result["phenotype"],
                        "action": drug_rec["action"],
                        "reason": drug_rec["recommendation"],
                    })

        logger.info(
            f"PGx mapping complete: {len(gene_results)} genes analyzed, "
            f"{len(all_critical_alerts)} critical alerts, "
            f"{len(drugs_to_avoid)} drugs to avoid"
        )

        return {
            "gene_results": gene_results,
            "critical_alerts": all_critical_alerts,
            "drugs_to_avoid": drugs_to_avoid,
            "drugs_to_adjust": drugs_to_adjust,
            "genes_analyzed": len(gene_results),
            "has_critical_findings": len(all_critical_alerts) > 0,
        }

    def check_drug(
        self,
        drug_name: str,
        star_alleles: Optional[Dict[str, str]] = None,
        genotypes: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Check a specific drug against all available PGx data.

        Args:
            drug_name: Drug name to check (case-insensitive).
            star_alleles: Dict of gene -> star alleles.
            genotypes: Dict of rsID/gene -> genotype.

        Returns:
            Dict with drug name, safety assessment, and gene-specific findings.
        """
        drug_lower = drug_name.lower()
        star_alleles = star_alleles or {}
        genotypes = genotypes or {}

        findings = []
        is_safe = True
        overall_action = "STANDARD_DOSING"

        # Check each gene config for this drug
        for gene, config in self._gene_configs.items():
            if drug_lower not in config.get("drug_recommendations", {}):
                continue

            # Resolve phenotype
            alleles = star_alleles.get(gene)
            gt = None
            field = config.get("genotype_field")
            if field:
                gt = genotypes.get(field)

            if not alleles and not gt:
                findings.append({
                    "gene": gene,
                    "status": "NOT_TESTED",
                    "message": f"{gene} genotype not available — cannot assess {drug_name} safety",
                })
                continue

            phenotype = self._resolve_phenotype(gene, alleles, gt)
            if not phenotype:
                continue

            rec = config["drug_recommendations"][drug_lower].get(phenotype)
            if rec:
                findings.append({
                    "gene": gene,
                    "phenotype": phenotype,
                    "alleles": alleles or gt,
                    "action": rec["action"],
                    "alert_level": rec["alert_level"],
                    "recommendation": rec["recommendation"],
                })

                if rec["action"] in ("AVOID", "CONTRAINDICATED"):
                    is_safe = False
                    overall_action = rec["action"]
                elif rec["action"] in ("DOSE_REDUCTION", "DOSE_ADJUSTMENT") and overall_action == "STANDARD_DOSING":
                    overall_action = rec["action"]

        return {
            "drug": drug_name,
            "is_safe": is_safe,
            "overall_action": overall_action,
            "findings": findings,
            "genes_checked": len(findings),
        }
