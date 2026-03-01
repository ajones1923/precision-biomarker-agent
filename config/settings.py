"""Precision Biomarker Agent configuration.

Follows the same Pydantic BaseSettings pattern as cart_intelligence_agent/config/settings.py.
"""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class PrecisionBiomarkerSettings(BaseSettings):
    """Configuration for Precision Biomarker Agent."""

    # ── Paths ──
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    CACHE_DIR: Path = DATA_DIR / "cache"
    REFERENCE_DIR: Path = DATA_DIR / "reference"

    # ── RAG Pipeline (reuse existing) ──
    RAG_PIPELINE_ROOT: Path = Path(
        "/home/adam/projects/hcls-ai-factory/rag-chat-pipeline"
    )

    # ── Milvus ──
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530

    # Collection names (10 biomarker-specific + 1 read-only genomic)
    COLLECTION_BIOMARKER_REF: str = "biomarker_reference"
    COLLECTION_GENETIC_VARIANTS: str = "biomarker_genetic_variants"
    COLLECTION_PGX_RULES: str = "biomarker_pgx_rules"
    COLLECTION_DISEASE_TRAJECTORIES: str = "biomarker_disease_trajectories"
    COLLECTION_CLINICAL_EVIDENCE: str = "biomarker_clinical_evidence"
    COLLECTION_NUTRITION: str = "biomarker_nutrition"
    COLLECTION_DRUG_INTERACTIONS: str = "biomarker_drug_interactions"
    COLLECTION_AGING_MARKERS: str = "biomarker_aging_markers"
    COLLECTION_GENOTYPE_ADJUSTMENTS: str = "biomarker_genotype_adjustments"
    COLLECTION_MONITORING: str = "biomarker_monitoring"
    COLLECTION_GENOMIC: str = "genomic_evidence"  # Existing shared collection

    # ── Embeddings ──
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_BATCH_SIZE: int = 32

    # ── LLM ──
    LLM_PROVIDER: str = "anthropic"
    LLM_MODEL: str = "claude-sonnet-4-20250514"
    ANTHROPIC_API_KEY: Optional[str] = None

    # ── RAG Search ──
    TOP_K_PER_COLLECTION: int = 5
    SCORE_THRESHOLD: float = 0.4

    # Collection search weights (must sum to ~1.0)
    WEIGHT_BIOMARKER_REF: float = 0.15
    WEIGHT_GENETIC_VARIANTS: float = 0.15
    WEIGHT_PGX_RULES: float = 0.12
    WEIGHT_DISEASE_TRAJECTORIES: float = 0.12
    WEIGHT_CLINICAL_EVIDENCE: float = 0.12
    WEIGHT_NUTRITION: float = 0.08
    WEIGHT_DRUG_INTERACTIONS: float = 0.08
    WEIGHT_AGING_MARKERS: float = 0.08
    WEIGHT_GENOTYPE_ADJUSTMENTS: float = 0.05
    WEIGHT_MONITORING: float = 0.05

    # ── API Server ──
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8529

    # ── Streamlit ──
    STREAMLIT_PORT: int = 8528

    # ── Prometheus Metrics ──
    METRICS_ENABLED: bool = True

    # ── Conversation Memory ──
    MAX_CONVERSATION_CONTEXT: int = 3

    # ── Citation Scoring ──
    CITATION_HIGH_THRESHOLD: float = 0.75
    CITATION_MEDIUM_THRESHOLD: float = 0.60

    # ── CORS ──
    CORS_ORIGINS: str = "http://localhost:8080,http://localhost:8528,http://localhost:8529"

    # ── Request Limits ──
    MAX_REQUEST_SIZE_MB: int = 10

    model_config = SettingsConfigDict(
        env_prefix="BIOMARKER_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = PrecisionBiomarkerSettings()
