"""Precision Biomarker Agent -- Streamlit UI v1.0.

Full-featured UI with 6 tabs:
- Biomarker Analysis: full patient analysis pipeline
- Biological Age: PhenoAge calculator
- Disease Risk: focused disease trajectory analysis
- PGx Profile: pharmacogenomic drug interaction mapping
- Evidence Explorer: RAG Q&A with collection filtering
- Reports: PDF and FHIR R4 export

Port: 8528 (assigned to Precision Biomarker Agent)

Usage:
    streamlit run app/biomarker_ui.py --server.port 8528

Author: Adam Jones
Date: March 2026
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# Add project root to path (must happen before src imports)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load API key from rag-chat-pipeline .env if not already set
if not os.environ.get("ANTHROPIC_API_KEY"):
    from config.settings import settings

    if settings.ANTHROPIC_API_KEY:
        os.environ["ANTHROPIC_API_KEY"] = settings.ANTHROPIC_API_KEY
    else:
        env_path = settings.RAG_PIPELINE_ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    os.environ["ANTHROPIC_API_KEY"] = (
                        line.split("=", 1)[1].strip().strip('"')
                    )
                    break


# =====================================================================
# ENGINE INITIALIZATION
# =====================================================================


@st.cache_resource
def init_engine():
    """Initialize the Precision Biomarker analysis engine (cached across reruns)."""
    try:
        from src.collections import BiomarkerCollectionManager
        from src.biological_age import BiologicalAgeCalculator
        from src.disease_trajectory import DiseaseTrajectoryAnalyzer

        manager = BiomarkerCollectionManager()
        manager.connect()

        bio_age_calc = BiologicalAgeCalculator()
        disease_analyzer = DiseaseTrajectoryAnalyzer()

        try:
            from sentence_transformers import SentenceTransformer

            class SimpleEmbedder:
                def __init__(self):
                    self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")

                def embed_text(self, text):
                    return self.model.encode(text).tolist()

                def encode(self, texts):
                    return self.model.encode(texts).tolist()

            embedder = SimpleEmbedder()
        except ImportError:
            embedder = None

        try:
            import anthropic
            from config.settings import settings as cfg

            class SimpleLLMClient:
                def __init__(self):
                    self.client = anthropic.Anthropic()

                def generate(
                    self,
                    prompt,
                    system_prompt="",
                    max_tokens=2048,
                    temperature=0.7,
                ):
                    msg = self.client.messages.create(
                        model=cfg.LLM_MODEL,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        system=system_prompt,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return msg.content[0].text

                def generate_stream(
                    self,
                    prompt,
                    system_prompt="",
                    max_tokens=2048,
                    temperature=0.7,
                ):
                    with self.client.messages.stream(
                        model=cfg.LLM_MODEL,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        system=system_prompt,
                        messages=[{"role": "user", "content": prompt}],
                    ) as stream:
                        for text in stream.text_stream:
                            yield text

            llm_client = SimpleLLMClient()
        except (ImportError, Exception):
            llm_client = None

        return {
            "manager": manager,
            "bio_age_calc": bio_age_calc,
            "disease_analyzer": disease_analyzer,
            "embedder": embedder,
            "llm_client": llm_client,
        }
    except Exception as e:
        st.error(f"Failed to initialize engine: {e}")
        return None


engine = init_engine()


# =====================================================================
# PAGE CONFIG
# =====================================================================

st.set_page_config(
    page_title="Precision Biomarker Agent -- HCLS AI Factory",
    page_icon="\U0001fa78",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# CUSTOM CSS -- NVIDIA Black + Green theme
# =====================================================================

st.markdown(
    """
<style>
    .stApp { background-color: #0a0a0f; }
    .stApp, .stApp p, .stApp span, .stApp li, .stApp td, .stApp th,
    .stApp label, .stApp .stMarkdown {
        color: #ffffff;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #ffffff;
    }
    section[data-testid="stSidebar"] {
        background-color: #12121a;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: #e0e0e8;
    }
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #76B900;
    }
    .stTextInput input, .stTextArea textarea {
        background-color: #1a1a24 !important;
        color: #ffffff !important;
        border: 1px solid #333 !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #76B900 !important;
        box-shadow: 0 0 0 1px #76B900 !important;
    }
    .stSelectbox > div > div {
        background-color: #1a1a24 !important;
        color: #ffffff !important;
    }
    .stNumberInput input {
        background-color: #1a1a24 !important;
        color: #ffffff !important;
    }
    .stButton > button {
        background-color: #76B900 !important;
        color: #000000 !important;
        font-weight: 600;
        border: none;
    }
    .stButton > button:hover {
        background-color: #5a9100 !important;
    }
    .risk-critical { color: #ff4444; font-weight: bold; }
    .risk-high { color: #ff8800; font-weight: bold; }
    .risk-moderate { color: #ffcc00; font-weight: bold; }
    .risk-low { color: #76B900; font-weight: bold; }
    .risk-normal { color: #76B900; font-weight: bold; }
</style>
""",
    unsafe_allow_html=True,
)


# =====================================================================
# SIDEBAR
# =====================================================================

with st.sidebar:
    st.markdown("## Precision Biomarker Agent")
    st.caption("HCLS AI Factory | v1.0")

    st.markdown("---")
    st.markdown("### Collection Stats")

    if engine and engine.get("manager"):
        try:
            stats = engine["manager"].get_collection_stats()
            total = sum(stats.values())
            for name, count in stats.items():
                short = name.replace("biomarker_", "").replace("_", " ").title()
                st.text(f"{short}: {count:,}")
            st.markdown(f"**Total: {total:,} vectors**")
        except Exception:
            st.warning("Milvus not connected")
    else:
        st.warning("Engine not initialized")

    st.markdown("---")
    st.markdown("### Service Status")

    status_items = {
        "Milvus": engine is not None and engine.get("manager") is not None,
        "Embedder": engine is not None and engine.get("embedder") is not None,
        "LLM": engine is not None and engine.get("llm_client") is not None,
    }
    for svc, ok in status_items.items():
        icon = "+" if ok else "x"
        st.text(f"[{icon}] {svc}")

    st.markdown("---")
    st.markdown("### About")
    st.caption(
        "Genotype-aware biomarker interpretation, biological age estimation, "
        "disease trajectory detection, and pharmacogenomic profiling. "
        "Part of the HCLS AI Factory precision medicine platform."
    )


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================


def risk_badge(level: str) -> str:
    """Return an HTML span with risk-level coloring."""
    css_class = f"risk-{level.lower()}"
    return f'<span class="{css_class}">{level.upper()}</span>'


def get_pgx_phenotype(gene: str, star_alleles: str) -> dict:
    """Determine metabolizer phenotype from star alleles (simplified)."""
    poor_alleles = {"*2", "*3", "*4", "*5", "*6", "*7", "*8"}
    alleles = [a.strip() for a in star_alleles.split("/")]

    poor_count = sum(1 for a in alleles if a in poor_alleles)

    if poor_count == 2:
        return {"phenotype": "Poor Metabolizer", "level": "critical"}
    elif poor_count == 1:
        return {"phenotype": "Intermediate Metabolizer", "level": "moderate"}
    elif "*17" in alleles:
        return {"phenotype": "Ultra-Rapid Metabolizer", "level": "high"}
    else:
        return {"phenotype": "Normal Metabolizer", "level": "normal"}


# PGx drug mapping (gene -> list of drugs with recommendations)
PGX_DRUG_MAP = {
    "CYP2D6": [
        {"drug": "Codeine", "poor": "AVOID - no analgesic effect", "ultra_rapid": "AVOID - risk of toxicity", "normal": "Standard dosing"},
        {"drug": "Tramadol", "poor": "AVOID - reduced efficacy", "ultra_rapid": "AVOID - risk of respiratory depression", "normal": "Standard dosing"},
        {"drug": "Tamoxifen", "poor": "Consider aromatase inhibitor", "ultra_rapid": "Standard dosing", "normal": "Standard dosing"},
        {"drug": "Amitriptyline", "poor": "50% dose reduction", "ultra_rapid": "Avoid or increase monitoring", "normal": "Standard dosing"},
    ],
    "CYP2C19": [
        {"drug": "Clopidogrel", "poor": "AVOID - use prasugrel/ticagrelor", "ultra_rapid": "Standard dosing", "normal": "Standard dosing"},
        {"drug": "Omeprazole", "poor": "50% dose reduction", "ultra_rapid": "Increase dose 2-3x", "normal": "Standard dosing"},
        {"drug": "Escitalopram", "poor": "50% dose reduction", "ultra_rapid": "Consider dose increase", "normal": "Standard dosing"},
        {"drug": "Voriconazole", "poor": "Use alternative antifungal", "ultra_rapid": "Increase dose", "normal": "Standard dosing"},
    ],
    "SLCO1B1": [
        {"drug": "Simvastatin", "poor": "AVOID high doses - myopathy risk", "ultra_rapid": "Standard dosing", "normal": "Standard dosing"},
        {"drug": "Atorvastatin", "poor": "Use lower dose", "ultra_rapid": "Standard dosing", "normal": "Standard dosing"},
        {"drug": "Rosuvastatin", "poor": "Use lower starting dose", "ultra_rapid": "Standard dosing", "normal": "Standard dosing"},
    ],
    "VKORC1": [
        {"drug": "Warfarin", "poor": "Reduce dose 25-50%", "ultra_rapid": "Increase dose", "normal": "Standard dosing"},
    ],
    "MTHFR": [
        {"drug": "Methotrexate", "poor": "Dose reduction; supplement with leucovorin", "ultra_rapid": "Standard dosing", "normal": "Standard dosing"},
        {"drug": "Fluorouracil (5-FU)", "poor": "Monitor for toxicity", "ultra_rapid": "Standard dosing", "normal": "Standard dosing"},
    ],
    "TPMT": [
        {"drug": "Azathioprine", "poor": "AVOID or 90% dose reduction", "ultra_rapid": "Standard dosing", "normal": "Standard dosing"},
        {"drug": "Mercaptopurine", "poor": "AVOID or 90% dose reduction", "ultra_rapid": "Standard dosing", "normal": "Standard dosing"},
    ],
    "CYP2C9": [
        {"drug": "Warfarin", "poor": "Reduce dose 25-50%", "ultra_rapid": "Standard dosing", "normal": "Standard dosing"},
        {"drug": "Celecoxib", "poor": "50% dose reduction", "ultra_rapid": "Standard dosing", "normal": "Standard dosing"},
        {"drug": "Phenytoin", "poor": "Reduce dose; monitor levels", "ultra_rapid": "Standard dosing", "normal": "Standard dosing"},
    ],
}


# =====================================================================
# TABS
# =====================================================================

st.markdown("# Precision Biomarker Agent")
st.caption("Genotype-aware biomarker interpretation | HCLS AI Factory")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Biomarker Analysis",
        "Biological Age",
        "Disease Risk",
        "PGx Profile",
        "Evidence Explorer",
        "Reports",
    ]
)


# =====================================================================
# TAB 1: BIOMARKER ANALYSIS
# =====================================================================

with tab1:
    st.markdown("## Full Patient Biomarker Analysis")
    st.caption(
        "Enter patient information, biomarker values, and genotype data for "
        "a comprehensive precision medicine analysis."
    )

    col_info1, col_info2 = st.columns(2)
    with col_info1:
        patient_age = st.number_input("Age", min_value=0, max_value=150, value=45, key="t1_age")
    with col_info2:
        patient_sex = st.selectbox("Sex", ["M", "F"], key="t1_sex")

    # -- Biomarker value inputs --
    biomarkers = {}

    with st.expander("CBC (Complete Blood Count)", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            v = st.number_input("WBC (10^3/uL)", min_value=0.0, value=0.0, step=0.1, key="t1_wbc")
            if v > 0:
                biomarkers["wbc"] = v
        with c2:
            v = st.number_input("Lymphocyte %", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="t1_lymph")
            if v > 0:
                biomarkers["lymphocyte_pct"] = v
        with c3:
            v = st.number_input("RDW %", min_value=0.0, value=0.0, step=0.1, key="t1_rdw")
            if v > 0:
                biomarkers["rdw"] = v
        c1, c2, c3 = st.columns(3)
        with c1:
            v = st.number_input("MCV (fL)", min_value=0.0, value=0.0, step=0.1, key="t1_mcv")
            if v > 0:
                biomarkers["mcv"] = v
        with c2:
            v = st.number_input("Platelets (10^3/uL)", min_value=0.0, value=0.0, step=1.0, key="t1_plt")
            if v > 0:
                biomarkers["platelets"] = v

    with st.expander("CMP (Comprehensive Metabolic Panel)", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            v = st.number_input("Albumin (g/dL)", min_value=0.0, value=0.0, step=0.1, key="t1_alb")
            if v > 0:
                biomarkers["albumin"] = v
        with c2:
            v = st.number_input("Creatinine (mg/dL)", min_value=0.0, value=0.0, step=0.01, key="t1_cr")
            if v > 0:
                biomarkers["creatinine"] = v
        with c3:
            v = st.number_input("Glucose (mg/dL)", min_value=0.0, value=0.0, step=1.0, key="t1_glu")
            if v > 0:
                biomarkers["glucose"] = v
                biomarkers["fasting_glucose"] = v
        c1, c2, c3 = st.columns(3)
        with c1:
            v = st.number_input("ALT (U/L)", min_value=0.0, value=0.0, step=1.0, key="t1_alt")
            if v > 0:
                biomarkers["alt"] = v
        with c2:
            v = st.number_input("AST (U/L)", min_value=0.0, value=0.0, step=1.0, key="t1_ast")
            if v > 0:
                biomarkers["ast"] = v
        with c3:
            v = st.number_input("Alk Phos (U/L)", min_value=0.0, value=0.0, step=1.0, key="t1_alp")
            if v > 0:
                biomarkers["alkaline_phosphatase"] = v

    with st.expander("Lipids", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            v = st.number_input("Total Cholesterol (mg/dL)", min_value=0.0, value=0.0, step=1.0, key="t1_tc")
            if v > 0:
                biomarkers["total_cholesterol"] = v
        with c2:
            v = st.number_input("LDL-C (mg/dL)", min_value=0.0, value=0.0, step=1.0, key="t1_ldl")
            if v > 0:
                biomarkers["ldl_c"] = v
        with c3:
            v = st.number_input("HDL-C (mg/dL)", min_value=0.0, value=0.0, step=1.0, key="t1_hdl")
            if v > 0:
                biomarkers["hdl_c"] = v
        c1, c2, _ = st.columns(3)
        with c1:
            v = st.number_input("Triglycerides (mg/dL)", min_value=0.0, value=0.0, step=1.0, key="t1_trig")
            if v > 0:
                biomarkers["triglycerides"] = v
        with c2:
            v = st.number_input("Lp(a) (nmol/L)", min_value=0.0, value=0.0, step=1.0, key="t1_lpa")
            if v > 0:
                biomarkers["lpa"] = v

    with st.expander("Thyroid", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            v = st.number_input("TSH (mIU/L)", min_value=0.0, value=0.0, step=0.01, key="t1_tsh")
            if v > 0:
                biomarkers["tsh"] = v
        with c2:
            v = st.number_input("Free T4 (ng/dL)", min_value=0.0, value=0.0, step=0.01, key="t1_ft4")
            if v > 0:
                biomarkers["free_t4"] = v
        with c3:
            v = st.number_input("Free T3 (pg/mL)", min_value=0.0, value=0.0, step=0.01, key="t1_ft3")
            if v > 0:
                biomarkers["free_t3"] = v

    with st.expander("Iron Studies", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            v = st.number_input("Ferritin (ng/mL)", min_value=0.0, value=0.0, step=1.0, key="t1_ferr")
            if v > 0:
                biomarkers["ferritin"] = v
        with c2:
            v = st.number_input("Transferrin Saturation (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="t1_tsat")
            if v > 0:
                biomarkers["transferrin_saturation"] = v

    with st.expander("Inflammation", expanded=False):
        v = st.number_input("hs-CRP (mg/L)", min_value=0.0, value=0.0, step=0.01, key="t1_crp")
        if v > 0:
            biomarkers["hs_crp"] = v

    with st.expander("Diabetes", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            v = st.number_input("HbA1c (%)", min_value=0.0, value=0.0, step=0.1, key="t1_hba1c")
            if v > 0:
                biomarkers["hba1c"] = v
        with c2:
            v = st.number_input("Fasting Insulin (uIU/mL)", min_value=0.0, value=0.0, step=0.1, key="t1_ins")
            if v > 0:
                biomarkers["fasting_insulin"] = v
        with c3:
            v = st.number_input("HOMA-IR", min_value=0.0, value=0.0, step=0.1, key="t1_homa")
            if v > 0:
                biomarkers["homa_ir"] = v

    with st.expander("Vitamins & Minerals", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            v = st.number_input("Vitamin D (ng/mL)", min_value=0.0, value=0.0, step=0.1, key="t1_vitd")
            if v > 0:
                biomarkers["vitamin_d"] = v
        with c2:
            v = st.number_input("Vitamin B12 (pg/mL)", min_value=0.0, value=0.0, step=1.0, key="t1_b12")
            if v > 0:
                biomarkers["vitamin_b12"] = v
        with c3:
            v = st.number_input("Folate (ng/mL)", min_value=0.0, value=0.0, step=0.1, key="t1_fol")
            if v > 0:
                biomarkers["folate"] = v
        c1, c2, c3 = st.columns(3)
        with c1:
            v = st.number_input("Omega-3 Index (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="t1_o3")
            if v > 0:
                biomarkers["omega3_index"] = v
        with c2:
            v = st.number_input("Magnesium (mg/dL)", min_value=0.0, value=0.0, step=0.1, key="t1_mag")
            if v > 0:
                biomarkers["magnesium"] = v
        with c3:
            v = st.number_input("Zinc (mcg/dL)", min_value=0.0, value=0.0, step=1.0, key="t1_zinc")
            if v > 0:
                biomarkers["zinc"] = v

    # -- Genotype inputs --
    genotypes = {}
    with st.expander("Genotype Inputs (rsIDs)", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            v = st.text_input("TCF7L2 rs7903146", placeholder="e.g., CT", key="t1_tcf7l2")
            if v:
                genotypes["TCF7L2_rs7903146"] = v
        with c2:
            v = st.text_input("PNPLA3 rs738409", placeholder="e.g., CG", key="t1_pnpla3")
            if v:
                genotypes["PNPLA3_rs738409"] = v
        with c3:
            v = st.text_input("DIO2 rs225014", placeholder="e.g., GA", key="t1_dio2")
            if v:
                genotypes["DIO2_rs225014"] = v
        c1, c2, c3 = st.columns(3)
        with c1:
            v = st.text_input("APOE", placeholder="e.g., E3/E4", key="t1_apoe")
            if v:
                genotypes["APOE"] = v
        with c2:
            v = st.text_input("HFE rs1800562", placeholder="e.g., GA", key="t1_hfe")
            if v:
                genotypes["HFE_rs1800562"] = v
        with c3:
            v = st.text_input("MTHFR rs1801133", placeholder="e.g., CT", key="t1_mthfr")
            if v:
                genotypes["MTHFR_rs1801133"] = v
        c1, c2, _ = st.columns(3)
        with c1:
            v = st.text_input("FADS1 rs174546", placeholder="e.g., CT", key="t1_fads1")
            if v:
                genotypes["FADS1_rs174546"] = v
        with c2:
            v = st.text_input("HSD17B13 rs72613567", placeholder="e.g., T/TA", key="t1_hsd")
            if v:
                genotypes["HSD17B13_rs72613567"] = v

    # -- Star allele inputs --
    star_alleles = {}
    with st.expander("Star Allele Inputs (PGx)", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            v = st.text_input("CYP2D6", placeholder="e.g., *1/*2", key="t1_cyp2d6")
            if v:
                star_alleles["CYP2D6"] = v
        with c2:
            v = st.text_input("CYP2C19", placeholder="e.g., *1/*1", key="t1_cyp2c19")
            if v:
                star_alleles["CYP2C19"] = v
        with c3:
            v = st.text_input("SLCO1B1", placeholder="e.g., *1/*1", key="t1_slco1b1")
            if v:
                star_alleles["SLCO1B1"] = v
        c1, c2, c3 = st.columns(3)
        with c1:
            v = st.text_input("VKORC1", placeholder="e.g., A/G", key="t1_vkorc1")
            if v:
                star_alleles["VKORC1"] = v
        with c2:
            v = st.text_input("MTHFR (star)", placeholder="e.g., CT", key="t1_mthfr_star")
            if v:
                star_alleles["MTHFR"] = v
        with c3:
            v = st.text_input("TPMT", placeholder="e.g., *1/*1", key="t1_tpmt")
            if v:
                star_alleles["TPMT"] = v

    # -- Run Analysis --
    if st.button("Run Full Analysis", type="primary", key="t1_run"):
        if not biomarkers:
            st.warning("Please enter at least one biomarker value.")
        elif engine is None:
            st.error("Engine not initialized. Check Milvus connection.")
        else:
            with st.spinner("Running comprehensive analysis..."):
                results = {}

                # Biological age
                try:
                    bio_age = engine["bio_age_calc"].calculate(
                        chronological_age=float(patient_age),
                        biomarkers=biomarkers,
                    )
                    results["biological_age"] = bio_age
                except Exception as e:
                    st.warning(f"Biological age calculation error: {e}")

                # Disease trajectories
                try:
                    trajectories = engine["disease_analyzer"].analyze_all(
                        biomarkers=biomarkers,
                        genotypes=genotypes,
                        age=float(patient_age),
                        sex="male" if patient_sex == "M" else "female",
                    )
                    results["disease_trajectories"] = trajectories
                except Exception as e:
                    st.warning(f"Disease trajectory error: {e}")

                # Store results in session state
                st.session_state["analysis_results"] = results
                st.session_state["patient_info"] = {
                    "age": patient_age,
                    "sex": patient_sex,
                    "biomarkers": biomarkers,
                    "genotypes": genotypes,
                    "star_alleles": star_alleles,
                }

            # Display results
            if "biological_age" in results:
                ba = results["biological_age"]
                pheno = ba.get("phenoage", ba)
                st.markdown("### Biological Age")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Chronological Age", f"{patient_age} years")
                with c2:
                    bio_val = pheno.get("biological_age", ba.get("biological_age", "N/A"))
                    st.metric("Biological Age", f"{bio_val} years")
                with c3:
                    accel = pheno.get("age_acceleration", ba.get("age_acceleration", 0))
                    risk = pheno.get("mortality_risk", ba.get("mortality_risk", "NORMAL"))
                    st.metric("Acceleration", f"{accel:+.1f} years")
                    st.markdown(f"Risk: {risk_badge(risk)}", unsafe_allow_html=True)

            if "disease_trajectories" in results:
                st.markdown("### Disease Trajectories")
                for traj in results["disease_trajectories"]:
                    level = traj.get("risk_level", "LOW")
                    name = traj.get("display_name", traj.get("disease", ""))
                    stage = traj.get("stage", "")
                    st.markdown(
                        f"**{name}**: {risk_badge(level)} | Stage: {stage}",
                        unsafe_allow_html=True,
                    )
                    if traj.get("findings"):
                        for f in traj["findings"]:
                            st.markdown(f"- {f}")
                    if traj.get("recommendations"):
                        with st.expander(f"Recommendations for {name}"):
                            for r in traj["recommendations"]:
                                st.markdown(f"- {r}")


# =====================================================================
# TAB 2: BIOLOGICAL AGE
# =====================================================================

with tab2:
    st.markdown("## Biological Age Calculator")
    st.caption(
        "Quick PhenoAge estimation using 9 routine blood biomarkers "
        "(Levine et al., Aging 2018)."
    )

    t2_age = st.number_input("Chronological Age", min_value=0, max_value=150, value=45, key="t2_age")

    st.markdown("#### PhenoAge Biomarkers")
    c1, c2, c3 = st.columns(3)
    with c1:
        t2_alb = st.number_input("Albumin (g/dL)", min_value=0.0, value=4.2, step=0.1, key="t2_alb")
        t2_cr = st.number_input("Creatinine (mg/dL)", min_value=0.0, value=0.9, step=0.01, key="t2_cr")
        t2_glu = st.number_input("Glucose (mg/dL)", min_value=0.0, value=95.0, step=1.0, key="t2_glu")
    with c2:
        t2_crp = st.number_input("hs-CRP (mg/L)", min_value=0.0, value=1.5, step=0.1, key="t2_crp")
        t2_lymph = st.number_input("Lymphocyte %", min_value=0.0, max_value=100.0, value=30.0, step=0.1, key="t2_lymph")
        t2_mcv = st.number_input("MCV (fL)", min_value=0.0, value=89.0, step=0.1, key="t2_mcv")
    with c3:
        t2_rdw = st.number_input("RDW %", min_value=0.0, value=13.0, step=0.1, key="t2_rdw")
        t2_alp = st.number_input("Alk Phos (U/L)", min_value=0.0, value=65.0, step=1.0, key="t2_alp")
        t2_wbc = st.number_input("WBC (10^3/uL)", min_value=0.0, value=6.0, step=0.1, key="t2_wbc")

    if st.button("Calculate Biological Age", type="primary", key="t2_run"):
        if engine is None:
            st.error("Engine not initialized.")
        else:
            pheno_biomarkers = {
                "albumin": t2_alb,
                "creatinine": t2_cr,
                "glucose": t2_glu,
                "hs_crp": t2_crp,
                "lymphocyte_pct": t2_lymph,
                "mcv": t2_mcv,
                "rdw": t2_rdw,
                "alkaline_phosphatase": t2_alp,
                "wbc": t2_wbc,
            }

            result = engine["bio_age_calc"].calculate_phenoage(
                chronological_age=float(t2_age),
                biomarkers=pheno_biomarkers,
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Biological Age", f"{result['biological_age']} years")
            with c2:
                st.metric("Age Acceleration", f"{result['age_acceleration']:+.1f} years")
            with c3:
                st.markdown(
                    f"Risk Level: {risk_badge(result['mortality_risk'])}",
                    unsafe_allow_html=True,
                )

            st.markdown("#### Top Aging Drivers")
            driver_data = []
            for d in result.get("top_aging_drivers", []):
                driver_data.append({
                    "Biomarker": d["biomarker"],
                    "Value": d["value"],
                    "Contribution": d["contribution"],
                    "Direction": d["direction"],
                })
            if driver_data:
                st.table(driver_data)


# =====================================================================
# TAB 3: DISEASE RISK
# =====================================================================

with tab3:
    st.markdown("## Disease Risk Analysis")
    st.caption("Focused disease trajectory analysis with genotype-stratified thresholds.")

    disease_options = {
        "Type 2 Diabetes": "type2_diabetes",
        "Cardiovascular Disease": "cardiovascular",
        "Liver Disease (NAFLD/Fibrosis)": "liver",
        "Thyroid Dysfunction": "thyroid",
        "Iron Metabolism Disorder": "iron",
        "Nutritional Deficiency": "nutritional",
    }

    selected_disease = st.selectbox(
        "Disease Category",
        list(disease_options.keys()),
        key="t3_disease",
    )
    disease_key = disease_options[selected_disease]

    st.markdown(f"#### Relevant Biomarkers for {selected_disease}")
    t3_biomarkers = {}
    t3_genotypes = {}

    if disease_key == "type2_diabetes":
        c1, c2 = st.columns(2)
        with c1:
            v = st.number_input("HbA1c (%)", min_value=0.0, value=0.0, step=0.1, key="t3_hba1c")
            if v > 0:
                t3_biomarkers["hba1c"] = v
            v = st.number_input("Fasting Glucose (mg/dL)", min_value=0.0, value=0.0, step=1.0, key="t3_glu")
            if v > 0:
                t3_biomarkers["fasting_glucose"] = v
        with c2:
            v = st.number_input("Fasting Insulin (uIU/mL)", min_value=0.0, value=0.0, step=0.1, key="t3_ins")
            if v > 0:
                t3_biomarkers["fasting_insulin"] = v
            gt = st.text_input("TCF7L2 rs7903146", placeholder="e.g., CT", key="t3_tcf7l2")
            if gt:
                t3_genotypes["TCF7L2_rs7903146"] = gt

    elif disease_key == "cardiovascular":
        c1, c2 = st.columns(2)
        with c1:
            v = st.number_input("LDL-C (mg/dL)", min_value=0.0, value=0.0, step=1.0, key="t3_ldl")
            if v > 0:
                t3_biomarkers["ldl_c"] = v
            v = st.number_input("Lp(a) (nmol/L)", min_value=0.0, value=0.0, step=1.0, key="t3_lpa")
            if v > 0:
                t3_biomarkers["lpa"] = v
            v = st.number_input("hs-CRP (mg/L)", min_value=0.0, value=0.0, step=0.1, key="t3_crp")
            if v > 0:
                t3_biomarkers["hs_crp"] = v
        with c2:
            v = st.number_input("ApoB (mg/dL)", min_value=0.0, value=0.0, step=1.0, key="t3_apob")
            if v > 0:
                t3_biomarkers["apob"] = v
            gt = st.text_input("APOE", placeholder="e.g., E3/E4", key="t3_apoe")
            if gt:
                t3_genotypes["APOE"] = gt

    elif disease_key == "liver":
        c1, c2 = st.columns(2)
        with c1:
            v = st.number_input("ALT (U/L)", min_value=0.0, value=0.0, step=1.0, key="t3_alt")
            if v > 0:
                t3_biomarkers["alt"] = v
            v = st.number_input("AST (U/L)", min_value=0.0, value=0.0, step=1.0, key="t3_ast")
            if v > 0:
                t3_biomarkers["ast"] = v
            v = st.number_input("Platelets (10^3/uL)", min_value=0.0, value=0.0, step=1.0, key="t3_plt")
            if v > 0:
                t3_biomarkers["platelets"] = v
        with c2:
            v = st.number_input("Ferritin (ng/mL)", min_value=0.0, value=0.0, step=1.0, key="t3_ferr")
            if v > 0:
                t3_biomarkers["ferritin"] = v
            gt = st.text_input("PNPLA3 rs738409", placeholder="e.g., CG", key="t3_pnpla3")
            if gt:
                t3_genotypes["PNPLA3_rs738409"] = gt

    elif disease_key == "thyroid":
        c1, c2 = st.columns(2)
        with c1:
            v = st.number_input("TSH (mIU/L)", min_value=0.0, value=0.0, step=0.01, key="t3_tsh")
            if v > 0:
                t3_biomarkers["tsh"] = v
            v = st.number_input("Free T4 (ng/dL)", min_value=0.0, value=0.0, step=0.01, key="t3_ft4")
            if v > 0:
                t3_biomarkers["free_t4"] = v
        with c2:
            v = st.number_input("Free T3 (pg/mL)", min_value=0.0, value=0.0, step=0.01, key="t3_ft3")
            if v > 0:
                t3_biomarkers["free_t3"] = v
            gt = st.text_input("DIO2 rs225014", placeholder="e.g., GA", key="t3_dio2")
            if gt:
                t3_genotypes["DIO2_rs225014"] = gt

    elif disease_key == "iron":
        c1, c2 = st.columns(2)
        with c1:
            v = st.number_input("Ferritin (ng/mL)", min_value=0.0, value=0.0, step=1.0, key="t3_ferr_iron")
            if v > 0:
                t3_biomarkers["ferritin"] = v
            v = st.number_input("Transferrin Sat (%)", min_value=0.0, value=0.0, step=0.1, key="t3_tsat")
            if v > 0:
                t3_biomarkers["transferrin_saturation"] = v
        with c2:
            gt = st.text_input("HFE rs1800562 (C282Y)", placeholder="e.g., GA", key="t3_hfe")
            if gt:
                t3_genotypes["HFE_rs1800562"] = gt

    elif disease_key == "nutritional":
        c1, c2 = st.columns(2)
        with c1:
            v = st.number_input("Vitamin D (ng/mL)", min_value=0.0, value=0.0, step=0.1, key="t3_vitd")
            if v > 0:
                t3_biomarkers["vitamin_d"] = v
            v = st.number_input("Vitamin B12 (pg/mL)", min_value=0.0, value=0.0, step=1.0, key="t3_b12")
            if v > 0:
                t3_biomarkers["vitamin_b12"] = v
            v = st.number_input("Folate (ng/mL)", min_value=0.0, value=0.0, step=0.1, key="t3_fol")
            if v > 0:
                t3_biomarkers["folate"] = v
        with c2:
            v = st.number_input("Omega-3 Index (%)", min_value=0.0, value=0.0, step=0.1, key="t3_o3")
            if v > 0:
                t3_biomarkers["omega3_index"] = v
            gt = st.text_input("MTHFR rs1801133", placeholder="e.g., CT", key="t3_mthfr")
            if gt:
                t3_genotypes["MTHFR_rs1801133"] = gt
            gt2 = st.text_input("FADS1 rs174546", placeholder="e.g., CT", key="t3_fads1")
            if gt2:
                t3_genotypes["FADS1_rs174546"] = gt2

    if st.button("Analyze Risk", type="primary", key="t3_run"):
        if not t3_biomarkers:
            st.warning("Please enter at least one biomarker value.")
        elif engine is None:
            st.error("Engine not initialized.")
        else:
            analyzer = engine["disease_analyzer"]
            method_map = {
                "type2_diabetes": analyzer.analyze_type2_diabetes,
                "cardiovascular": analyzer.analyze_cardiovascular,
                "liver": lambda b, g: analyzer.analyze_liver(b, g, age=float(st.session_state.get("t1_age", 45))),
                "thyroid": analyzer.analyze_thyroid,
                "iron": lambda b, g: analyzer.analyze_iron(b, g, sex="male"),
                "nutritional": analyzer.analyze_nutritional,
            }

            result = method_map[disease_key](t3_biomarkers, t3_genotypes)

            st.markdown(f"### {result['display_name']}")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    f"Risk Level: {risk_badge(result['risk_level'])}",
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(f"Stage: **{result['stage']}**")

            if result.get("findings"):
                st.markdown("#### Findings")
                for f in result["findings"]:
                    st.markdown(f"- {f}")

            if result.get("recommendations"):
                st.markdown("#### Recommendations")
                for r in result["recommendations"]:
                    st.markdown(f"- {r}")


# =====================================================================
# TAB 4: PGx PROFILE
# =====================================================================

with tab4:
    st.markdown("## Pharmacogenomic Profile")
    st.caption("Star allele interpretation for 7 pharmacogenes + HLA-B*57:01 status.")

    t4_star_alleles = {}
    c1, c2, c3 = st.columns(3)
    with c1:
        v = st.text_input("CYP2D6", placeholder="e.g., *1/*4", key="t4_cyp2d6")
        if v:
            t4_star_alleles["CYP2D6"] = v
        v = st.text_input("CYP2C19", placeholder="e.g., *2/*2", key="t4_cyp2c19")
        if v:
            t4_star_alleles["CYP2C19"] = v
        v = st.text_input("CYP2C9", placeholder="e.g., *1/*1", key="t4_cyp2c9")
        if v:
            t4_star_alleles["CYP2C9"] = v
    with c2:
        v = st.text_input("SLCO1B1", placeholder="e.g., *5/*5", key="t4_slco1b1")
        if v:
            t4_star_alleles["SLCO1B1"] = v
        v = st.text_input("VKORC1", placeholder="e.g., A/A", key="t4_vkorc1")
        if v:
            t4_star_alleles["VKORC1"] = v
        v = st.text_input("MTHFR", placeholder="e.g., TT", key="t4_mthfr")
        if v:
            t4_star_alleles["MTHFR"] = v
    with c3:
        v = st.text_input("TPMT", placeholder="e.g., *3A/*3A", key="t4_tpmt")
        if v:
            t4_star_alleles["TPMT"] = v
        t4_hla = st.checkbox("HLA-B*57:01 Positive", key="t4_hla")

    if st.button("Map Drug Interactions", type="primary", key="t4_run"):
        if not t4_star_alleles and not t4_hla:
            st.warning("Please enter at least one star allele or check HLA-B*57:01.")
        else:
            st.markdown("### Drug Interaction Map")

            table_data = []
            critical_alerts = []

            for gene, alleles in t4_star_alleles.items():
                pheno_info = get_pgx_phenotype(gene, alleles)
                drugs = PGX_DRUG_MAP.get(gene, [])

                for drug_info in drugs:
                    phenotype = pheno_info["phenotype"]
                    if "Poor" in phenotype:
                        rec = drug_info.get("poor", "Standard dosing")
                    elif "Ultra" in phenotype:
                        rec = drug_info.get("ultra_rapid", "Standard dosing")
                    else:
                        rec = drug_info.get("normal", "Standard dosing")

                    if "AVOID" in rec:
                        critical_alerts.append(f"{gene} {alleles} + {drug_info['drug']}: {rec}")

                    table_data.append({
                        "Gene": gene,
                        "Star Alleles": alleles,
                        "Phenotype": phenotype,
                        "Drug": drug_info["drug"],
                        "Recommendation": rec,
                    })

            if t4_hla:
                critical_alerts.append("HLA-B*57:01 Positive + Abacavir: CONTRAINDICATED")
                table_data.append({
                    "Gene": "HLA-B",
                    "Star Alleles": "*57:01",
                    "Phenotype": "Positive",
                    "Drug": "Abacavir",
                    "Recommendation": "CONTRAINDICATED - risk of hypersensitivity reaction",
                })

            if critical_alerts:
                st.markdown("#### Critical Alerts")
                for alert in critical_alerts:
                    st.markdown(
                        f'<p class="risk-critical">WARNING: {alert}</p>',
                        unsafe_allow_html=True,
                    )

            if table_data:
                st.table(table_data)


# =====================================================================
# TAB 5: EVIDENCE EXPLORER
# =====================================================================

with tab5:
    st.markdown("## Evidence Explorer")
    st.caption("RAG-powered Q&A across all biomarker knowledge collections.")

    # Sidebar collection filters (using columns inside the tab)
    collection_names = [
        "biomarker_reference",
        "biomarker_genetic_variants",
        "biomarker_pgx_rules",
        "biomarker_disease_trajectories",
        "biomarker_clinical_evidence",
        "biomarker_nutrition",
        "biomarker_drug_interactions",
        "biomarker_aging_markers",
        "biomarker_genotype_adjustments",
        "biomarker_monitoring",
    ]

    with st.expander("Collection Filters", expanded=False):
        selected_collections = []
        c1, c2 = st.columns(2)
        for i, coll in enumerate(collection_names):
            short = coll.replace("biomarker_", "").replace("_", " ").title()
            col = c1 if i % 2 == 0 else c2
            with col:
                if st.checkbox(short, value=True, key=f"t5_coll_{coll}"):
                    selected_collections.append(coll)

    # Initialize conversation history
    if "evidence_history" not in st.session_state:
        st.session_state["evidence_history"] = []

    question = st.text_input(
        "Ask a question about biomarkers, genetics, or pharmacogenomics:",
        placeholder="e.g., How does MTHFR C677T affect folate metabolism?",
        key="t5_question",
    )

    if st.button("Search Evidence", type="primary", key="t5_run"):
        if not question:
            st.warning("Please enter a question.")
        elif engine is None or engine.get("embedder") is None:
            st.error("Engine or embedder not initialized.")
        else:
            with st.spinner("Searching evidence..."):
                # Embed the question
                prefix = "Represent this sentence for searching relevant passages: "
                query_vec = engine["embedder"].embed_text(prefix + question)

                # Search selected collections
                all_hits = []
                for coll in selected_collections:
                    try:
                        hits = engine["manager"].search(
                            collection_name=coll,
                            query_embedding=query_vec,
                            top_k=5,
                            score_threshold=0.4,
                        )
                        all_hits.extend(hits)
                    except Exception:
                        pass

                # Sort by score
                all_hits.sort(key=lambda x: x.get("score", 0), reverse=True)

                # Build evidence context
                evidence_text = ""
                for i, hit in enumerate(all_hits[:15]):
                    coll_name = hit.get("collection", "unknown")
                    score = hit.get("score", 0)
                    text = hit.get("text_chunk", hit.get("text_summary", hit.get("finding", "")))
                    evidence_text += f"\n[{i+1}] ({coll_name}, score={score:.3f}): {text}\n"

                # Generate LLM response if available
                if engine.get("llm_client") and evidence_text:
                    system_prompt = (
                        "You are a precision medicine biomarker expert. Answer the question "
                        "using the provided evidence. Cite sources by their number [1], [2], etc. "
                        "Be concise and clinically accurate."
                    )
                    prompt = f"Question: {question}\n\nEvidence:\n{evidence_text}\n\nAnswer:"

                    response_container = st.empty()
                    full_response = ""
                    for chunk in engine["llm_client"].generate_stream(
                        prompt, system_prompt=system_prompt
                    ):
                        full_response += chunk
                        response_container.markdown(full_response)

                    # Store in history
                    st.session_state["evidence_history"].append({
                        "question": question,
                        "answer": full_response,
                        "hits": len(all_hits),
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                    })
                elif evidence_text:
                    st.markdown("### Evidence Found")
                    st.markdown(evidence_text)
                else:
                    st.info("No matching evidence found in selected collections.")

    # Show conversation history
    if st.session_state.get("evidence_history"):
        with st.expander("Conversation History", expanded=False):
            for entry in reversed(st.session_state["evidence_history"]):
                st.markdown(f"**Q** ({entry['timestamp']}): {entry['question']}")
                st.markdown(f"**A**: {entry['answer'][:200]}...")
                st.markdown("---")


# =====================================================================
# TAB 6: REPORTS
# =====================================================================

with tab6:
    st.markdown("## Report Generation")
    st.caption("Generate and export comprehensive biomarker analysis reports.")

    results = st.session_state.get("analysis_results")
    patient = st.session_state.get("patient_info")

    if not results or not patient:
        st.info(
            "No analysis results available. Run a Full Analysis in the "
            "Biomarker Analysis tab first."
        )
    else:
        if st.button("Generate Full Report", type="primary", key="t6_gen"):
            with st.spinner("Generating report..."):
                report_sections = []

                # Section 1: Executive Summary
                report_sections.append("# Precision Biomarker Analysis Report")
                report_sections.append(
                    f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                )
                report_sections.append(
                    f"**Patient:** Age {patient['age']}, Sex {patient['sex']}"
                )

                # Section 2: Biological Age
                if "biological_age" in results:
                    ba = results["biological_age"]
                    pheno = ba.get("phenoage", ba)
                    report_sections.append("\n## 1. Biological Age Assessment")
                    report_sections.append(
                        f"- Chronological Age: {patient['age']} years"
                    )
                    report_sections.append(
                        f"- Biological Age (PhenoAge): {pheno.get('biological_age', 'N/A')} years"
                    )
                    report_sections.append(
                        f"- Age Acceleration: {pheno.get('age_acceleration', 'N/A')} years"
                    )
                    report_sections.append(
                        f"- Mortality Risk: {pheno.get('mortality_risk', 'N/A')}"
                    )

                # Section 3-8: Disease Trajectories
                if "disease_trajectories" in results:
                    report_sections.append("\n## 2. Disease Trajectory Analysis")
                    for i, traj in enumerate(results["disease_trajectories"]):
                        report_sections.append(
                            f"\n### 2.{i+1} {traj.get('display_name', '')}"
                        )
                        report_sections.append(
                            f"- Risk Level: {traj.get('risk_level', 'N/A')}"
                        )
                        report_sections.append(
                            f"- Stage: {traj.get('stage', 'N/A')}"
                        )
                        if traj.get("findings"):
                            report_sections.append("\n**Findings:**")
                            for f in traj["findings"]:
                                report_sections.append(f"- {f}")
                        if traj.get("recommendations"):
                            report_sections.append("\n**Recommendations:**")
                            for r in traj["recommendations"]:
                                report_sections.append(f"- {r}")

                # Section 9: Biomarker Summary
                report_sections.append("\n## 3. Biomarker Values")
                for name, val in patient.get("biomarkers", {}).items():
                    report_sections.append(f"- {name}: {val}")

                # Section 10: Genotypes
                if patient.get("genotypes"):
                    report_sections.append("\n## 4. Genotype Summary")
                    for rsid, gt in patient["genotypes"].items():
                        report_sections.append(f"- {rsid}: {gt}")

                # Section 11: Star Alleles / PGx
                if patient.get("star_alleles"):
                    report_sections.append("\n## 5. Pharmacogenomic Profile")
                    for gene, alleles in patient["star_alleles"].items():
                        report_sections.append(f"- {gene}: {alleles}")

                # Section 12: Disclaimer
                report_sections.append("\n## Disclaimer")
                report_sections.append(
                    "This report is generated by the Precision Biomarker Agent "
                    "(HCLS AI Factory) for research and educational purposes. "
                    "It is not intended as medical advice. All clinical decisions "
                    "should be made in consultation with qualified healthcare providers."
                )

                full_report = "\n".join(report_sections)
                st.session_state["full_report"] = full_report

        # Display report
        if "full_report" in st.session_state:
            st.markdown(st.session_state["full_report"])

            st.markdown("---")
            st.markdown("### Export Options")

            # Export Markdown
            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "Export PDF",
                    data=st.session_state["full_report"].encode("utf-8"),
                    file_name=f"biomarker_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown",
                    key="t6_pdf",
                )
            with c2:
                # FHIR R4 export
                fhir_bundle = {
                    "resourceType": "Bundle",
                    "type": "document",
                    "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "entry": [
                        {
                            "resource": {
                                "resourceType": "Patient",
                                "gender": "male" if patient["sex"] == "M" else "female",
                                "birthDate": str(datetime.now().year - patient["age"]),
                            }
                        },
                        {
                            "resource": {
                                "resourceType": "DiagnosticReport",
                                "status": "final",
                                "code": {
                                    "text": "Precision Biomarker Analysis"
                                },
                                "conclusion": st.session_state["full_report"][:500],
                            }
                        },
                    ],
                }
                for name, val in patient.get("biomarkers", {}).items():
                    fhir_bundle["entry"].append({
                        "resource": {
                            "resourceType": "Observation",
                            "status": "final",
                            "code": {"text": name},
                            "valueQuantity": {"value": val},
                        }
                    })

                st.download_button(
                    "Export FHIR R4",
                    data=json.dumps(fhir_bundle, indent=2),
                    file_name=f"biomarker_fhir_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                    key="t6_fhir",
                )
