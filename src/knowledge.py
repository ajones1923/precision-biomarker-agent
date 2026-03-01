"""Precision Biomarker Agent -- Knowledge Graph.

Extends the Clinker pattern from rag-chat-pipeline/src/knowledge.py and
mirrors cart_intelligence_agent/src/knowledge.py, adapted for precision
biomarker analysis. Contains:

1. BIOMARKER_DOMAINS: 6 disease domains with biomarkers, genetic modifiers, and
   intervention targets
2. PHENOAGE_KNOWLEDGE: PhenoAge clock biomarker descriptions, coefficients,
   and clinical interpretation
3. PGX_KNOWLEDGE: 7 pharmacogenes with key drug interactions and CPIC guidance
4. CROSS_MODAL_LINKS: Mapping of biomarker findings to triggers for other
   HCLS AI Factory agents (imaging, oncology, genomics)

Author: Adam Jones
Date: March 2026
"""

from typing import Any, Dict, List, Optional


# =============================================================================
# 1. BIOMARKER_DOMAINS -- 6 disease domain knowledge graphs
# =============================================================================

BIOMARKER_DOMAINS: Dict[str, Dict[str, Any]] = {
    "diabetes": {
        "name": "Diabetes / Metabolic Syndrome",
        "key_biomarkers": {
            "HbA1c": {
                "unit": "%",
                "normal_range": "4.0-5.6",
                "pre_disease": "5.7-6.4",
                "disease": ">=6.5",
                "clinical_note": "Glycated hemoglobin reflects 2-3 month average glucose.",
            },
            "fasting_glucose": {
                "unit": "mg/dL",
                "normal_range": "70-99",
                "pre_disease": "100-125",
                "disease": ">=126",
                "clinical_note": "Fasting plasma glucose; confirm on repeat testing.",
            },
            "fasting_insulin": {
                "unit": "uIU/mL",
                "normal_range": "2.6-24.9",
                "elevated": ">25",
                "clinical_note": "Elevated insulin with normal glucose suggests insulin resistance.",
            },
            "HOMA_IR": {
                "unit": "index",
                "normal_range": "<2.0",
                "pre_disease": "2.0-2.9",
                "disease": ">=3.0",
                "clinical_note": "Homeostatic model assessment; (glucose x insulin) / 405.",
            },
            "triglycerides": {
                "unit": "mg/dL",
                "normal_range": "<150",
                "borderline": "150-199",
                "elevated": ">=200",
                "clinical_note": "Part of metabolic syndrome criteria.",
            },
        },
        "genetic_modifiers": [
            {"gene": "TCF7L2", "rs_id": "rs7903146", "risk_allele": "T",
             "effect": "1.4x increased T2DM risk per allele; strongest common T2DM variant"},
            {"gene": "SLC30A8", "rs_id": "rs13266634", "risk_allele": "C",
             "effect": "Zinc transporter affecting insulin secretion"},
            {"gene": "PPARG", "rs_id": "rs1801282", "risk_allele": "C",
             "effect": "Pro12Ala; affects insulin sensitivity and adipogenesis"},
        ],
        "intervention_targets": [
            "Lifestyle modification (diet, exercise) -- 58% risk reduction (DPP trial)",
            "Metformin for pre-diabetes with HOMA-IR >3.0",
            "Berberine for insulin sensitization (TCF7L2 risk carriers)",
            "Chromium picolinate 200-1000 mcg/day for insulin sensitivity",
        ],
        "clinical_context": (
            "Pre-diabetes is detectable 5-10 years before clinical T2DM diagnosis. "
            "The biomarker trajectory of rising fasting insulin followed by rising "
            "glucose followed by rising HbA1c provides a window for intervention. "
            "Genetic risk from TCF7L2 can accelerate this trajectory."
        ),
    },

    "cardiovascular": {
        "name": "Cardiovascular Risk",
        "key_biomarkers": {
            "Lp(a)": {
                "unit": "nmol/L",
                "normal_range": "<75",
                "elevated": "75-125",
                "high_risk": ">125",
                "clinical_note": "Genetically determined; 90% of variation due to LPA gene.",
            },
            "ApoB": {
                "unit": "mg/dL",
                "normal_range": "<90",
                "borderline": "90-130",
                "elevated": ">130",
                "clinical_note": "Better predictor of cardiovascular events than LDL-C.",
            },
            "hs_CRP": {
                "unit": "mg/L",
                "normal_range": "<1.0",
                "borderline": "1.0-3.0",
                "elevated": ">3.0",
                "clinical_note": "Marker of systemic inflammation and vascular risk.",
            },
            "homocysteine": {
                "unit": "umol/L",
                "normal_range": "5-15",
                "elevated": ">15",
                "clinical_note": "Elevated by MTHFR variants; independent CVD risk factor.",
            },
            "LDL_C": {
                "unit": "mg/dL",
                "normal_range": "<100",
                "borderline": "100-159",
                "elevated": ">=160",
                "clinical_note": "Primary target for statin therapy; PCSK9 variants affect levels.",
            },
        },
        "genetic_modifiers": [
            {"gene": "APOE", "rs_id": "rs429358/rs7412", "risk_allele": "e4",
             "effect": "APOE4 increases LDL-C by 10-15 mg/dL; 2-3x Alzheimer risk"},
            {"gene": "LPA", "rs_id": "rs10455872", "risk_allele": "G",
             "effect": "KIV-2 repeats determine Lp(a) levels; not modifiable by statins"},
            {"gene": "PCSK9", "rs_id": "rs11591147", "risk_allele": "T",
             "effect": "Loss-of-function variants reduce LDL-C and CVD risk by ~50%"},
            {"gene": "MTHFR", "rs_id": "rs1801133", "risk_allele": "T",
             "effect": "C677T reduces enzyme activity 30-70%; raises homocysteine"},
        ],
        "intervention_targets": [
            "PCSK9 inhibitors for high Lp(a) with established CVD",
            "Niacin 1-3g/day may reduce Lp(a) by 20-30%",
            "Methylfolate + B12 for MTHFR-related hyperhomocysteinemia",
            "ApoB-guided statin therapy (target <80 mg/dL for high risk)",
            "Omega-3 fatty acids for triglyceride reduction",
        ],
        "clinical_context": (
            "Lp(a) is the single most heritable cardiovascular risk factor. "
            "ApoB better captures atherogenic particle burden than LDL-C alone. "
            "APOE genotype affects both lipid metabolism and Alzheimer risk, "
            "requiring integrative counseling. MTHFR C677T homozygotes need "
            "methylfolate rather than folic acid."
        ),
    },

    "liver": {
        "name": "Liver Health / NAFLD / MASLD",
        "key_biomarkers": {
            "ALT": {
                "unit": "U/L",
                "normal_range": "7-56",
                "mildly_elevated": "56-100",
                "elevated": ">100",
                "clinical_note": "Primary marker of hepatocyte injury.",
            },
            "AST": {
                "unit": "U/L",
                "normal_range": "10-40",
                "elevated": ">40",
                "clinical_note": "AST:ALT ratio >2 suggests alcoholic liver disease.",
            },
            "GGT": {
                "unit": "U/L",
                "normal_range": "9-48",
                "elevated": ">48",
                "clinical_note": "Sensitive marker for biliary disease and alcohol use.",
            },
            "FIB4_index": {
                "unit": "index",
                "low_risk": "<1.3",
                "intermediate": "1.3-2.67",
                "high_risk": ">2.67",
                "clinical_note": "Non-invasive fibrosis score: (age x AST) / (platelets x sqrt(ALT)).",
            },
            "ferritin": {
                "unit": "ng/mL",
                "normal_range_male": "24-336",
                "normal_range_female": "11-307",
                "elevated": ">500",
                "clinical_note": "Elevated in iron overload, inflammation, and NAFLD.",
            },
        },
        "genetic_modifiers": [
            {"gene": "PNPLA3", "rs_id": "rs738409", "risk_allele": "G",
             "effect": "I148M variant; 2-3x NAFLD/NASH risk; strongest genetic predictor"},
            {"gene": "TM6SF2", "rs_id": "rs58542926", "risk_allele": "T",
             "effect": "E167K variant; increases hepatic steatosis but may protect against CVD"},
            {"gene": "HSD17B13", "rs_id": "rs72613567", "risk_allele": "TA",
             "effect": "Loss-of-function protects against NASH progression (therapeutic target)"},
        ],
        "intervention_targets": [
            "Weight loss 7-10% for NAFLD regression",
            "Mediterranean diet reduces hepatic steatosis",
            "Vitamin E 800 IU/day for non-diabetic NASH (PNPLA3 GG carriers benefit most)",
            "Avoid excess fructose (exacerbates PNPLA3-mediated steatosis)",
            "GLP-1 receptor agonists for NASH with diabetes",
        ],
        "clinical_context": (
            "PNPLA3 I148M is the strongest genetic determinant of NAFLD/NASH risk. "
            "Carriers of the G allele have 2-3x higher risk and progress faster to "
            "fibrosis. FIB-4 index is the recommended first-line non-invasive fibrosis "
            "assessment. The combination of PNPLA3 genotype + FIB-4 provides superior "
            "risk stratification compared to either alone."
        ),
    },

    "thyroid": {
        "name": "Thyroid Function",
        "key_biomarkers": {
            "TSH": {
                "unit": "mIU/L",
                "normal_range": "0.4-4.0",
                "subclinical_hypo": "4.0-10.0",
                "overt_hypo": ">10.0",
                "subclinical_hyper": "0.1-0.4",
                "overt_hyper": "<0.1",
                "clinical_note": "Most sensitive marker of thyroid dysfunction.",
            },
            "free_T4": {
                "unit": "ng/dL",
                "normal_range": "0.8-1.8",
                "clinical_note": "Low in hypothyroidism; high in hyperthyroidism.",
            },
            "free_T3": {
                "unit": "pg/mL",
                "normal_range": "2.3-4.2",
                "clinical_note": "Active thyroid hormone; may be normal in early hypothyroidism.",
            },
            "anti_TPO": {
                "unit": "IU/mL",
                "normal_range": "<35",
                "elevated": ">35",
                "clinical_note": "Thyroid peroxidase antibodies; marker of Hashimoto's thyroiditis.",
            },
            "reverse_T3": {
                "unit": "ng/dL",
                "normal_range": "10-24",
                "elevated": ">24",
                "clinical_note": "Elevated in euthyroid sick syndrome, chronic stress, and T4-to-T3 conversion issues.",
            },
        },
        "genetic_modifiers": [
            {"gene": "DIO2", "rs_id": "rs225014", "risk_allele": "A",
             "effect": "Thr92Ala; impaired T4-to-T3 conversion; may need combination T4/T3 therapy"},
            {"gene": "DIO1", "rs_id": "rs2235544", "risk_allele": "A",
             "effect": "Affects peripheral T4-to-T3 conversion efficiency"},
        ],
        "intervention_targets": [
            "Levothyroxine dose optimization guided by DIO2 genotype",
            "Selenium 200 mcg/day for elevated anti-TPO (reduces antibody titers)",
            "Combination T4/T3 therapy for DIO2 Thr92Ala homozygotes with persistent symptoms",
            "Iron and selenium optimization (cofactors for deiodinase enzymes)",
        ],
        "clinical_context": (
            "DIO2 Thr92Ala polymorphism affects T4-to-T3 conversion and may explain "
            "persistent hypothyroid symptoms despite normal TSH on levothyroxine. "
            "Thyroid autoimmunity (anti-TPO positive) can precede clinical "
            "hypothyroidism by years, providing an intervention window."
        ),
    },

    "iron": {
        "name": "Iron Metabolism",
        "key_biomarkers": {
            "ferritin": {
                "unit": "ng/mL",
                "normal_range_male": "24-336",
                "normal_range_female": "11-307",
                "iron_deficiency": "<15",
                "iron_overload": ">500",
                "clinical_note": "Acute phase reactant; interpret with CRP.",
            },
            "iron_saturation": {
                "unit": "%",
                "normal_range": "20-50",
                "low": "<20",
                "elevated": ">45",
                "high_risk": ">60",
                "clinical_note": "Transferrin saturation >45% prompts HFE genotyping.",
            },
            "serum_iron": {
                "unit": "ug/dL",
                "normal_range": "60-170",
                "clinical_note": "Diurnal variation; best measured fasting AM.",
            },
            "TIBC": {
                "unit": "ug/dL",
                "normal_range": "250-370",
                "clinical_note": "Total iron-binding capacity; elevated in iron deficiency.",
            },
            "hepcidin": {
                "unit": "ng/mL",
                "normal_range": "1-20",
                "elevated": ">20",
                "clinical_note": "Master regulator of iron homeostasis; elevated suppresses absorption.",
            },
        },
        "genetic_modifiers": [
            {"gene": "HFE", "rs_id": "rs1800562", "risk_allele": "A",
             "effect": "C282Y homozygosity causes hereditary hemochromatosis (iron overload)"},
            {"gene": "HFE", "rs_id": "rs1799945", "risk_allele": "G",
             "effect": "H63D; mild iron overload risk, especially compound heterozygotes C282Y/H63D"},
            {"gene": "TMPRSS6", "rs_id": "rs855791", "risk_allele": "T",
             "effect": "Affects hepcidin regulation; associated with iron-refractory iron deficiency anemia"},
        ],
        "intervention_targets": [
            "Therapeutic phlebotomy for HFE C282Y homozygotes (target ferritin 50-100 ng/mL)",
            "Iron chelation therapy (deferasirox) for transfusion-dependent iron overload",
            "IV iron (ferric carboxymaltose) for TMPRSS6-related oral iron resistance",
            "Vitamin C 200 mg with iron supplements to enhance absorption",
            "Avoid iron supplements and vitamin C supplementation in HFE carriers with elevated ferritin",
        ],
        "clinical_context": (
            "HFE C282Y homozygosity is the most common genetic cause of iron overload "
            "(hereditary hemochromatosis) in Northern European populations. Early detection "
            "via transferrin saturation screening followed by HFE genotyping prevents "
            "irreversible organ damage (cirrhosis, cardiomyopathy, diabetes). "
            "TMPRSS6 variants can cause iron deficiency that is refractory to oral iron."
        ),
    },

    "nutritional": {
        "name": "Nutritional Genomics",
        "key_biomarkers": {
            "vitamin_D_25OH": {
                "unit": "ng/mL",
                "deficient": "<20",
                "insufficient": "20-30",
                "optimal": "30-50",
                "clinical_note": "VDR polymorphisms affect vitamin D receptor sensitivity.",
            },
            "folate_serum": {
                "unit": "ng/mL",
                "deficient": "<3.0",
                "normal_range": "3.0-20.0",
                "clinical_note": "MTHFR C677T carriers may need higher levels for homocysteine control.",
            },
            "vitamin_B12": {
                "unit": "pg/mL",
                "deficient": "<200",
                "borderline": "200-300",
                "normal_range": "300-900",
                "clinical_note": "FUT2 non-secretors have higher B12 requirements.",
            },
            "omega3_index": {
                "unit": "%",
                "low_risk": ">8",
                "moderate_risk": "4-8",
                "high_risk": "<4",
                "clinical_note": "FADS1/2 variants affect omega-3 fatty acid metabolism.",
            },
            "methylmalonic_acid": {
                "unit": "nmol/L",
                "normal_range": "<270",
                "elevated": ">270",
                "clinical_note": "Functional B12 marker; more specific than serum B12.",
            },
        },
        "genetic_modifiers": [
            {"gene": "VDR", "rs_id": "rs2228570", "risk_allele": "T",
             "effect": "FokI polymorphism; affects vitamin D receptor function"},
            {"gene": "MTHFR", "rs_id": "rs1801133", "risk_allele": "T",
             "effect": "C677T; 30-70% reduced enzyme activity; need methylfolate not folic acid"},
            {"gene": "FADS1", "rs_id": "rs174537", "risk_allele": "T",
             "effect": "Reduced desaturase activity; impaired conversion of ALA to EPA/DHA"},
            {"gene": "FUT2", "rs_id": "rs602662", "risk_allele": "A",
             "effect": "Non-secretor status; reduced B12 absorption and higher requirements"},
            {"gene": "BCMO1", "rs_id": "rs7501331", "risk_allele": "T",
             "effect": "Reduced beta-carotene to vitamin A conversion by 30-70%"},
        ],
        "intervention_targets": [
            "Methylfolate (not folic acid) for MTHFR C677T carriers",
            "Preformed EPA/DHA for FADS1 poor converters (2-4 g/day)",
            "Vitamin D3 2000-5000 IU/day for VDR FokI T carriers",
            "Methylcobalamin for FUT2 non-secretors (sublingual preferred)",
            "Preformed vitamin A (retinol) for BCMO1 poor converters",
        ],
        "clinical_context": (
            "Nutritional genomics reveals why standard RDAs are insufficient for "
            "many individuals. MTHFR C677T is the most clinically relevant nutritional "
            "variant, affecting folate metabolism and methylation capacity. FADS1/2 "
            "variants are especially important for vegetarians who rely on ALA-to-EPA "
            "conversion. VDR variants affect the threshold for vitamin D sufficiency."
        ),
    },
}


# =============================================================================
# 2. PHENOAGE_KNOWLEDGE -- PhenoAge clock interpretation
# =============================================================================

PHENOAGE_KNOWLEDGE: Dict[str, Any] = {
    "description": (
        "PhenoAge is a biological age estimator developed by Levine et al. (2018) "
        "using 9 routine blood biomarkers and chronological age. It was trained on "
        "NHANES III mortality data and validated against lifespan, healthspan, and "
        "disease outcomes. PhenoAge acceleration (PhenoAge - chronological age) is "
        "associated with all-cause mortality, cardiovascular disease, cancer, and "
        "cognitive decline."
    ),
    "reference": "Levine ME et al. Aging (Albany NY). 2018;10(4):573-591.",
    "biomarkers": {
        "albumin": {
            "coefficient": -0.0336,
            "unit": "g/dL",
            "direction": "protective",
            "interpretation": (
                "Higher albumin is associated with slower aging. Albumin reflects "
                "hepatic synthetic function and nutritional status. Levels <3.5 g/dL "
                "are a strong predictor of mortality."
            ),
        },
        "creatinine": {
            "coefficient": 0.0095,
            "unit": "mg/dL",
            "direction": "aging",
            "interpretation": (
                "Higher creatinine contributes to accelerated aging, reflecting "
                "declining kidney function. Must be interpreted with muscle mass."
            ),
        },
        "glucose": {
            "coefficient": 0.1953,
            "unit": "mg/dL",
            "direction": "aging",
            "interpretation": (
                "Fasting glucose has the largest coefficient. Elevated glucose drives "
                "advanced glycation end-products (AGEs), oxidative stress, and "
                "accelerated aging across multiple organ systems."
            ),
        },
        "ln_crp": {
            "coefficient": 0.0954,
            "unit": "ln(mg/L)",
            "direction": "aging",
            "interpretation": (
                "Log-transformed hs-CRP captures chronic low-grade inflammation "
                "(inflammaging). Elevated CRP accelerates biological aging and "
                "is modifiable through diet, exercise, and anti-inflammatory interventions."
            ),
        },
        "lymphocyte_pct": {
            "coefficient": -0.0120,
            "unit": "%",
            "direction": "protective",
            "interpretation": (
                "Higher lymphocyte percentage is protective. Low lymphocytes "
                "(lymphopenia) reflects immunosenescence and is associated with "
                "increased infection risk and mortality."
            ),
        },
        "mcv": {
            "coefficient": 0.0268,
            "unit": "fL",
            "direction": "aging",
            "interpretation": (
                "Higher MCV (macrocytosis) contributes to accelerated aging. "
                "Can reflect B12/folate deficiency, liver disease, or myelodysplasia."
            ),
        },
        "rdw": {
            "coefficient": 0.3306,
            "unit": "%",
            "direction": "aging",
            "interpretation": (
                "Red cell distribution width has the second-largest coefficient. "
                "Elevated RDW (anisocytosis) is one of the strongest predictors of "
                "mortality and reflects oxidative stress, chronic inflammation, and "
                "impaired erythropoiesis."
            ),
        },
        "alkaline_phosphatase": {
            "coefficient": 0.0019,
            "unit": "U/L",
            "direction": "aging",
            "interpretation": (
                "Higher ALP contributes modestly to accelerated aging. Reflects "
                "hepatobiliary or bone disease. Elevated ALP is associated with "
                "cardiovascular and all-cause mortality."
            ),
        },
        "wbc": {
            "coefficient": 0.0554,
            "unit": "10^3/uL",
            "direction": "aging",
            "interpretation": (
                "Higher WBC count reflects chronic inflammation and immune activation. "
                "Even high-normal WBC (>7.0) is associated with accelerated aging."
            ),
        },
    },
    "clinical_interpretation": {
        "acceleration_positive": (
            "Positive age acceleration (PhenoAge > chronological age) indicates "
            "biological aging faster than expected. Each year of acceleration is "
            "associated with approximately 9% increased mortality risk."
        ),
        "acceleration_negative": (
            "Negative age acceleration (PhenoAge < chronological age) indicates "
            "biological aging slower than expected, suggesting resilience and "
            "favorable metabolic health."
        ),
        "actionable_drivers": (
            "The most modifiable PhenoAge drivers are glucose (diet/exercise), "
            "hs-CRP (anti-inflammatory interventions), RDW (nutritional optimization), "
            "and albumin (protein intake). Targeting these can reduce PhenoAge "
            "by 2-5 years within 6-12 months."
        ),
    },
}


# =============================================================================
# 3. PGX_KNOWLEDGE -- 7 key pharmacogenes
# =============================================================================

PGX_KNOWLEDGE: Dict[str, Dict[str, Any]] = {
    "CYP2D6": {
        "full_name": "Cytochrome P450 2D6",
        "chromosome": "22q13.1",
        "phenotypes": {
            "ultra_rapid": "Gene duplication/amplification (*1xN, *2xN); increased metabolism",
            "normal": "*1/*1, *1/*2; standard enzyme activity",
            "intermediate": "*1/*4, *1/*5, *41/*41; reduced activity (50-75%)",
            "poor": "*4/*4, *5/*5, *4/*5; absent or minimal activity (<10%)",
        },
        "key_drugs": [
            {"drug": "codeine", "effect": "Ultra-rapid: risk of toxicity/respiratory depression; Poor: no analgesic effect",
             "recommendation": "Avoid codeine in ultra-rapid and poor metabolizers; use morphine alternative",
             "cpic_level": "1A"},
            {"drug": "tamoxifen", "effect": "Poor: reduced conversion to endoxifen; compromised efficacy",
             "recommendation": "Consider aromatase inhibitor for poor metabolizers",
             "cpic_level": "1A"},
            {"drug": "tramadol", "effect": "Ultra-rapid: toxicity risk; Poor: reduced efficacy",
             "recommendation": "Avoid in ultra-rapid and poor metabolizers",
             "cpic_level": "1A"},
            {"drug": "ondansetron", "effect": "Ultra-rapid: reduced efficacy",
             "recommendation": "Select alternative antiemetic for ultra-rapid metabolizers",
             "cpic_level": "1B"},
        ],
        "clinical_context": (
            "CYP2D6 is the most polymorphic CYP enzyme with >100 allelic variants. "
            "It metabolizes ~25% of clinically used drugs. Gene deletions (*5) and "
            "duplications (*1xN) are common. Ethnicity-based frequency: PM prevalence "
            "5-10% European, 1-2% East Asian."
        ),
    },
    "CYP2C19": {
        "full_name": "Cytochrome P450 2C19",
        "chromosome": "10q23.33",
        "phenotypes": {
            "ultra_rapid": "*17/*17; increased enzyme activity",
            "normal": "*1/*1; standard activity",
            "intermediate": "*1/*2, *1/*3; reduced activity",
            "poor": "*2/*2, *2/*3, *3/*3; absent activity",
        },
        "key_drugs": [
            {"drug": "clopidogrel", "effect": "Poor/IM: reduced active metabolite; increased CVD event risk",
             "recommendation": "Use prasugrel or ticagrelor for poor/intermediate metabolizers",
             "cpic_level": "1A"},
            {"drug": "omeprazole", "effect": "Ultra-rapid: reduced efficacy; Poor: increased drug levels",
             "recommendation": "Increase dose for ultra-rapid; reduce for poor metabolizers",
             "cpic_level": "1A"},
            {"drug": "escitalopram", "effect": "Poor: increased drug levels and side effects; Ultra-rapid: subtherapeutic",
             "recommendation": "Reduce dose 50% for poor metabolizers; consider alternative for ultra-rapid",
             "cpic_level": "1A"},
            {"drug": "voriconazole", "effect": "Poor: toxicity risk from increased exposure",
             "recommendation": "Choose alternative antifungal or reduce dose with TDM",
             "cpic_level": "1A"},
        ],
        "clinical_context": (
            "CYP2C19 is critical for clopidogrel activation. The *2 allele (loss-of-function) "
            "is found in 15% of Europeans and 30% of East Asians. *17 (gain-of-function) is "
            "present in 20-30% of Europeans. Genotyping before PCI is recommended by AHA/ACC."
        ),
    },
    "CYP2C9": {
        "full_name": "Cytochrome P450 2C9",
        "chromosome": "10q23.33",
        "phenotypes": {
            "normal": "*1/*1; standard activity",
            "intermediate": "*1/*2, *1/*3; reduced activity (50-80%)",
            "poor": "*2/*2, *3/*3, *2/*3; markedly reduced activity (<20%)",
        },
        "key_drugs": [
            {"drug": "warfarin", "effect": "Poor/IM: reduced clearance; increased bleeding risk",
             "recommendation": "Reduce initial dose 25-50%; use pharmacogenomic dosing algorithm",
             "cpic_level": "1A"},
            {"drug": "phenytoin", "effect": "Poor: increased drug levels and toxicity",
             "recommendation": "Reduce dose by 25%; monitor levels closely",
             "cpic_level": "1A"},
            {"drug": "celecoxib", "effect": "Poor: increased exposure; GI bleeding risk",
             "recommendation": "Reduce dose by 50% or use alternative NSAID",
             "cpic_level": "1B"},
        ],
        "clinical_context": (
            "CYP2C9 is critical for warfarin metabolism. Combined CYP2C9 + VKORC1 "
            "genotyping explains ~40% of warfarin dose variability. CYP2C9*2 and *3 "
            "are the most common reduced-function alleles in Europeans (10-15% each)."
        ),
    },
    "VKORC1": {
        "full_name": "Vitamin K Epoxide Reductase Complex Subunit 1",
        "chromosome": "16p11.2",
        "phenotypes": {
            "normal_sensitivity": "GG genotype; standard warfarin dose",
            "intermediate_sensitivity": "GA genotype; reduced dose needed",
            "high_sensitivity": "AA genotype; significantly reduced dose needed",
        },
        "key_drugs": [
            {"drug": "warfarin", "effect": "AA: 2-3 mg/day typical (vs 5-7 mg/day for GG)",
             "recommendation": "Use pharmacogenomic dosing algorithm combining CYP2C9 + VKORC1",
             "cpic_level": "1A"},
        ],
        "clinical_context": (
            "VKORC1 rs9923231 (-1639G>A) is the strongest genetic predictor of warfarin "
            "dose requirement, explaining ~25% of dose variability. The A allele frequency "
            "is ~40% in Europeans and ~90% in East Asians."
        ),
    },
    "SLCO1B1": {
        "full_name": "Solute Carrier Organic Anion Transporter Family Member 1B1",
        "chromosome": "12p12.1",
        "phenotypes": {
            "normal_function": "TT (rs4149056); standard statin transport",
            "intermediate_function": "TC; reduced hepatic uptake of statins",
            "poor_function": "CC; markedly reduced transport; myopathy risk",
        },
        "key_drugs": [
            {"drug": "simvastatin", "effect": "CC: 17x increased myopathy risk; avoid high doses",
             "recommendation": "Use max 20 mg simvastatin or switch to rosuvastatin/pravastatin",
             "cpic_level": "1A"},
            {"drug": "atorvastatin", "effect": "CC: increased myopathy risk (lower than simvastatin)",
             "recommendation": "Monitor for myopathy; consider lower dose",
             "cpic_level": "2A"},
        ],
        "clinical_context": (
            "SLCO1B1*5 (rs4149056 T>C) reduces hepatic statin uptake, increasing systemic "
            "exposure and myopathy risk. The C allele is present in ~15% of Europeans. "
            "The SEARCH trial showed 17x increased myopathy risk with 80 mg simvastatin "
            "in CC homozygotes."
        ),
    },
    "DPYD": {
        "full_name": "Dihydropyrimidine Dehydrogenase",
        "chromosome": "1p21.3",
        "phenotypes": {
            "normal": "*1/*1; standard 5-FU metabolism",
            "intermediate": "*1/*2A, *1/*13; reduced activity (30-70%)",
            "poor": "*2A/*2A; absent activity (<10%); life-threatening toxicity risk",
        },
        "key_drugs": [
            {"drug": "5-fluorouracil", "effect": "Poor: fatal toxicity risk (neutropenia, mucositis, death)",
             "recommendation": "Contraindicated in poor metabolizers; reduce 50% for intermediate",
             "cpic_level": "1A"},
            {"drug": "capecitabine", "effect": "Same as 5-FU (capecitabine is 5-FU prodrug)",
             "recommendation": "Pre-treatment DPYD testing is mandatory in EU (EMA 2020)",
             "cpic_level": "1A"},
        ],
        "clinical_context": (
            "DPYD deficiency causes life-threatening fluoropyrimidine toxicity. "
            "Pre-treatment testing is mandatory in the EU and recommended by CPIC. "
            "DPYD*2A (IVS14+1G>A) completely eliminates enzyme activity. "
            "Approximately 3-5% of the population carries a reduced-function allele."
        ),
    },
    "CYP3A5": {
        "full_name": "Cytochrome P450 3A5",
        "chromosome": "7q22.1",
        "phenotypes": {
            "expresser": "*1/*1 or *1/*3; functional enzyme (more common in African descent)",
            "non_expresser": "*3/*3; absent expression (most Europeans)",
        },
        "key_drugs": [
            {"drug": "tacrolimus", "effect": "Expressers: rapid metabolism; need higher starting dose",
             "recommendation": "Start 1.5-2x standard dose for *1/*1; titrate with TDM",
             "cpic_level": "1A"},
        ],
        "clinical_context": (
            "CYP3A5 expression is highly ethnicity-dependent: *1 (functional) allele "
            "frequency is 60-90% in African populations vs 5-15% in Europeans. This "
            "explains racial disparities in tacrolimus dosing requirements after "
            "organ transplantation."
        ),
    },
}


# =============================================================================
# 4. CROSS_MODAL_LINKS -- Triggers to other HCLS AI Factory agents
# =============================================================================

CROSS_MODAL_LINKS: Dict[str, Dict[str, Any]] = {
    "elevated_lpa": {
        "trigger_condition": "Lp(a) > 125 nmol/L",
        "target_agent": "imaging_intelligence_agent",
        "target_url": "not_deployed",
        "action": "cardiovascular_imaging_assessment",
        "description": (
            "Elevated Lp(a) is the strongest heritable cardiovascular risk factor. "
            "When detected, trigger the Imaging Intelligence Agent to assess coronary "
            "artery calcium score (CAC), carotid intima-media thickness (CIMT), and "
            "cardiac CT angiography for subclinical atherosclerosis."
        ),
        "urgency": "high",
        "payload": {
            "biomarker": "Lp(a)",
            "threshold_exceeded": ">125 nmol/L",
            "imaging_modalities": ["CAC_score", "CIMT", "CCTA"],
            "clinical_rationale": "Screen for subclinical atherosclerosis in high Lp(a)",
        },
    },
    "pgx_critical": {
        "trigger_condition": "Critical PGx finding (DPYD poor metabolizer or CYP2D6 ultra-rapid on opioids)",
        "target_agent": "cart_intelligence_agent",
        "target_url": "http://localhost:8525/api/v1/drug-interaction-check",
        "action": "drug_interaction_check",
        "description": (
            "When a critical pharmacogenomic finding is detected (e.g., DPYD poor "
            "metabolizer on 5-FU, CYP2D6 ultra-rapid on codeine), trigger the "
            "CAR-T/Oncology Intelligence Agent to check for drug interaction conflicts "
            "in any active treatment protocols."
        ),
        "urgency": "critical",
        "payload": {
            "pgx_gene": "variable",
            "metabolizer_status": "variable",
            "flagged_drugs": [],
            "clinical_rationale": "Cross-check oncology treatment protocols for PGx conflicts",
        },
    },
    "pre_diabetic_trajectory": {
        "trigger_condition": "HOMA-IR > 2.5 AND HbA1c > 5.5% AND TCF7L2 risk carrier",
        "target_agent": "genomics_pipeline",
        "target_url": "http://localhost:5001/api/v1/reanalyze",
        "action": "vcf_reanalysis_diabetes_genes",
        "description": (
            "When biomarker trajectory analysis detects pre-diabetic pattern combined "
            "with TCF7L2 risk genotype, trigger the Genomics Pipeline to re-analyze "
            "the patient VCF for additional diabetes-associated rare variants in "
            "HNF1A, HNF4A, GCK, and other MODY genes."
        ),
        "urgency": "moderate",
        "payload": {
            "gene_panel": ["TCF7L2", "HNF1A", "HNF4A", "GCK", "SLC30A8", "KCNJ11", "ABCC8"],
            "analysis_type": "rare_variant_screening",
            "clinical_rationale": "Screen for monogenic diabetes (MODY) in pre-diabetic trajectory",
        },
    },
    "iron_overload_detected": {
        "trigger_condition": "Ferritin > 500 AND transferrin saturation > 45%",
        "target_agent": "imaging_intelligence_agent",
        "target_url": "not_deployed",
        "action": "liver_iron_quantification",
        "description": (
            "Iron overload detected by biomarkers. Trigger imaging assessment for "
            "hepatic iron concentration (HIC) via MRI R2* and liver stiffness "
            "measurement via FibroScan to assess fibrosis stage."
        ),
        "urgency": "high",
        "payload": {
            "biomarkers": {"ferritin": ">500", "transferrin_saturation": ">45%"},
            "imaging_modalities": ["MRI_R2star", "FibroScan"],
            "hfe_genotype": "check_required",
            "clinical_rationale": "Quantify hepatic iron and assess fibrosis in iron overload",
        },
    },
    "accelerated_aging_detected": {
        "trigger_condition": "PhenoAge acceleration > 5 years",
        "target_agent": "genomics_pipeline",
        "target_url": "http://localhost:5001/api/v1/reanalyze",
        "action": "epigenetic_variant_analysis",
        "description": (
            "Significant biological age acceleration detected. Trigger genomic pipeline "
            "to analyze telomere-related variants (TERT, TERC) and DNA repair genes "
            "(ATM, BRCA1) for germline variants that may explain accelerated aging."
        ),
        "urgency": "moderate",
        "payload": {
            "gene_panel": ["TERT", "TERC", "ATM", "BRCA1", "LMNA", "WRN"],
            "analysis_type": "aging_variant_screening",
            "clinical_rationale": "Screen for germline aging-associated variants",
        },
    },
}


# =============================================================================
# 5. HELPER FUNCTIONS
# =============================================================================


def get_domain_context(domain: str) -> str:
    """Return formatted knowledge context for a disease domain.

    Args:
        domain: One of 'diabetes', 'cardiovascular', 'liver', 'thyroid', 'iron', 'nutritional'.

    Returns:
        Formatted string with domain knowledge or empty string if not found.
    """
    info = BIOMARKER_DOMAINS.get(domain.lower())
    if not info:
        return ""

    lines = [f"Domain: {info['name']}"]

    # Key biomarkers
    lines.append("Key Biomarkers:")
    for name, details in info["key_biomarkers"].items():
        unit = details.get("unit", "")
        normal = details.get("normal_range", details.get("normal_range_male", ""))
        note = details.get("clinical_note", "")
        lines.append(f"  - {name} ({unit}): normal {normal}. {note}")

    # Genetic modifiers
    if info.get("genetic_modifiers"):
        lines.append("Genetic Modifiers:")
        for mod in info["genetic_modifiers"]:
            lines.append(
                f"  - {mod['gene']} {mod['rs_id']} ({mod['risk_allele']}): {mod['effect']}"
            )

    # Intervention targets
    if info.get("intervention_targets"):
        lines.append("Intervention Targets:")
        for target in info["intervention_targets"]:
            lines.append(f"  - {target}")

    # Clinical context
    if info.get("clinical_context"):
        lines.append(f"Clinical Context: {info['clinical_context']}")

    return "\n".join(lines)


def get_pgx_context(gene: str) -> str:
    """Return formatted PGx knowledge context for a pharmacogene.

    Args:
        gene: Gene symbol (e.g., 'CYP2D6', 'DPYD').

    Returns:
        Formatted string with PGx knowledge or empty string if not found.
    """
    gene_upper = gene.upper()
    info = PGX_KNOWLEDGE.get(gene_upper)
    if not info:
        # Try case-insensitive lookup
        for key, val in PGX_KNOWLEDGE.items():
            if key.upper() == gene_upper:
                info = val
                break
    if not info:
        return ""

    lines = [f"Pharmacogene: {info['full_name']} ({gene_upper})"]

    # Phenotypes
    lines.append("Phenotypes:")
    for phenotype, desc in info.get("phenotypes", {}).items():
        lines.append(f"  - {phenotype}: {desc}")

    # Key drugs
    lines.append("Key Drug Interactions:")
    for drug in info.get("key_drugs", []):
        lines.append(
            f"  - {drug['drug']}: {drug['effect']} "
            f"(CPIC Level {drug['cpic_level']})"
        )
        lines.append(f"    Recommendation: {drug['recommendation']}")

    # Clinical context
    if info.get("clinical_context"):
        lines.append(f"Clinical Context: {info['clinical_context']}")

    return "\n".join(lines)


def get_biomarker_context(biomarker_name: str) -> str:
    """Return formatted knowledge context for a specific biomarker.

    Searches across all disease domains for the biomarker.

    Args:
        biomarker_name: Biomarker name (e.g., 'HbA1c', 'Lp(a)', 'ferritin').

    Returns:
        Formatted string with biomarker knowledge or empty string if not found.
    """
    name_lower = biomarker_name.lower().replace(" ", "_")

    # Search all domains
    for domain_key, domain_info in BIOMARKER_DOMAINS.items():
        for marker_key, marker_details in domain_info["key_biomarkers"].items():
            if marker_key.lower() == name_lower or biomarker_name.lower() in marker_key.lower():
                lines = [
                    f"Biomarker: {marker_key}",
                    f"Domain: {domain_info['name']}",
                    f"Unit: {marker_details.get('unit', 'N/A')}",
                ]
                # Add all range info
                for rk, rv in marker_details.items():
                    if rk not in ("unit", "clinical_note"):
                        lines.append(f"  {rk}: {rv}")
                if marker_details.get("clinical_note"):
                    lines.append(f"Clinical Note: {marker_details['clinical_note']}")

                # Add relevant genetic modifiers from same domain
                modifiers = [
                    m for m in domain_info.get("genetic_modifiers", [])
                ]
                if modifiers:
                    lines.append("Related Genetic Modifiers:")
                    for mod in modifiers:
                        lines.append(f"  - {mod['gene']} {mod['rs_id']}: {mod['effect']}")

                return "\n".join(lines)

    # Check PhenoAge biomarkers
    for marker_key, marker_details in PHENOAGE_KNOWLEDGE.get("biomarkers", {}).items():
        if marker_key.lower() == name_lower or biomarker_name.lower() in marker_key.lower():
            return (
                f"Biomarker: {marker_key} (PhenoAge Clock)\n"
                f"Coefficient: {marker_details['coefficient']}\n"
                f"Unit: {marker_details['unit']}\n"
                f"Direction: {marker_details['direction']}\n"
                f"Interpretation: {marker_details['interpretation']}"
            )

    return ""


def get_cross_modal_context(trigger_key: str) -> str:
    """Return formatted cross-modal link context.

    Args:
        trigger_key: Key from CROSS_MODAL_LINKS (e.g., 'elevated_lpa').

    Returns:
        Formatted string with cross-modal link details.
    """
    link = CROSS_MODAL_LINKS.get(trigger_key)
    if not link:
        return ""

    return (
        f"Cross-Modal Trigger: {trigger_key}\n"
        f"Condition: {link['trigger_condition']}\n"
        f"Target Agent: {link['target_agent']}\n"
        f"Action: {link['action']}\n"
        f"Urgency: {link['urgency']}\n"
        f"Description: {link['description']}"
    )


def get_knowledge_stats() -> Dict[str, int]:
    """Return statistics about the biomarker knowledge graph.

    Returns:
        Dict with counts of domains, biomarkers, genetic modifiers,
        pharmacogenes, PhenoAge markers, and cross-modal links.
    """
    total_biomarkers = sum(
        len(d["key_biomarkers"]) for d in BIOMARKER_DOMAINS.values()
    )
    total_genetic_modifiers = sum(
        len(d.get("genetic_modifiers", [])) for d in BIOMARKER_DOMAINS.values()
    )
    total_pgx_drugs = sum(
        len(g.get("key_drugs", [])) for g in PGX_KNOWLEDGE.values()
    )

    return {
        "disease_domains": len(BIOMARKER_DOMAINS),
        "total_biomarkers": total_biomarkers,
        "total_genetic_modifiers": total_genetic_modifiers,
        "pharmacogenes": len(PGX_KNOWLEDGE),
        "pgx_drug_interactions": total_pgx_drugs,
        "phenoage_markers": len(PHENOAGE_KNOWLEDGE.get("biomarkers", {})),
        "cross_modal_links": len(CROSS_MODAL_LINKS),
    }
