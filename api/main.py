"""Precision Biomarker Agent -- FastAPI REST API.

Wraps the multi-collection RAG engine and analysis modules as a
production-ready REST API with CORS, health checks, Prometheus-compatible
metrics, and Pydantic request/response schemas.

Endpoints:
    GET  /health           -- Service health with collection and vector counts
    GET  /collections      -- Collection names and record counts
    POST /query            -- Full RAG query (retrieve + LLM synthesis)
    POST /search           -- Evidence-only retrieval (no LLM, fast)
    POST /analyze          -- Full patient analysis
    POST /biological-age   -- Biological age calculation only
    GET  /knowledge/stats  -- Knowledge graph statistics
    GET  /metrics          -- Prometheus-compatible metrics (placeholder)

Port: 8529 (from config/settings.py)

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 8529 --reload

Author: Adam Jones
Date: March 2026
"""

import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

# =====================================================================
# Path setup -- ensure project root is importable
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load API key from rag-chat-pipeline .env if not already set
if not os.environ.get("ANTHROPIC_API_KEY"):
    _env_path = Path("/home/adam/projects/hcls-ai-factory/rag-chat-pipeline/.env")
    if _env_path.exists():
        for _line in _env_path.read_text().splitlines():
            if _line.startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = _line.split("=", 1)[1].strip().strip('"')
                break

from config.settings import settings
from src.biological_age import BiologicalAgeCalculator
from src.disease_trajectory import DiseaseTrajectoryAnalyzer
from src.genotype_adjustment import GenotypeAdjuster
from src.knowledge import get_knowledge_stats
from src.models import AgentQuery, CrossCollectionResult, PatientProfile, SearchHit
from src.pharmacogenomics import PharmacogenomicMapper
from src.rag_engine import BiomarkerRAGEngine

# Route modules
from api.routes.analysis import router as analysis_router
from api.routes.reports import router as reports_router
from api.routes.events import router as events_router

# =====================================================================
# Module-level state (populated during lifespan startup)
# =====================================================================

_engine: Optional[BiomarkerRAGEngine] = None
_agent = None  # PrecisionBiomarkerAgent
_manager = None  # Collection manager
_bio_age_calc: Optional[BiologicalAgeCalculator] = None
_trajectory_analyzer: Optional[DiseaseTrajectoryAnalyzer] = None
_pgx_mapper: Optional[PharmacogenomicMapper] = None
_genotype_adjuster: Optional[GenotypeAdjuster] = None

# Simple request counters for /metrics
_metrics: Dict[str, int] = {
    "requests_total": 0,
    "query_requests_total": 0,
    "search_requests_total": 0,
    "analyze_requests_total": 0,
    "bio_age_requests_total": 0,
    "errors_total": 0,
}


# =====================================================================
# Lifespan -- initialize engine on startup, disconnect on shutdown
# =====================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the RAG engine, analysis modules, and Milvus on startup."""
    global _engine, _agent, _manager
    global _bio_age_calc, _trajectory_analyzer, _pgx_mapper, _genotype_adjuster

    # -- Collection manager --
    try:
        from src.collections import BiomarkerCollectionManager
        _manager = BiomarkerCollectionManager(
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT,
        )
        _manager.connect()
    except Exception:
        _manager = None

    # -- Embedder --
    try:
        from sentence_transformers import SentenceTransformer

        class _Embedder:
            def __init__(self):
                self.model = SentenceTransformer(settings.EMBEDDING_MODEL)

            def embed_text(self, text: str) -> List[float]:
                return self.model.encode(text).tolist()

        embedder = _Embedder()
    except ImportError:
        embedder = None

    # -- LLM client --
    try:
        import anthropic

        class _LLMClient:
            def __init__(self):
                self.client = anthropic.Anthropic()

            def generate(
                self, prompt: str, system_prompt: str = "",
                max_tokens: int = 2048, temperature: float = 0.7,
            ) -> str:
                msg = self.client.messages.create(
                    model=settings.LLM_MODEL,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                )
                return msg.content[0].text

            def generate_stream(
                self, prompt: str, system_prompt: str = "",
                max_tokens: int = 2048, temperature: float = 0.7,
            ):
                with self.client.messages.stream(
                    model=settings.LLM_MODEL,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                ) as stream:
                    for text in stream.text_stream:
                        yield text

        llm_client = _LLMClient()
    except (ImportError, Exception):
        llm_client = None

    # -- Knowledge module --
    from src import knowledge as kg

    # -- Analysis modules --
    _bio_age_calc = BiologicalAgeCalculator()
    _trajectory_analyzer = DiseaseTrajectoryAnalyzer()
    _pgx_mapper = PharmacogenomicMapper()
    _genotype_adjuster = GenotypeAdjuster()

    # -- Build RAG engine --
    _engine = BiomarkerRAGEngine(
        collection_manager=_manager,
        embedder=embedder,
        llm_client=llm_client,
        knowledge=kg,
    )

    # -- Build agent --
    from src.agent import PrecisionBiomarkerAgent
    _agent = PrecisionBiomarkerAgent(
        rag_engine=_engine,
        bio_age_calc=_bio_age_calc,
        trajectory_analyzer=_trajectory_analyzer,
        pgx_mapper=_pgx_mapper,
        genotype_adjuster=_genotype_adjuster,
    )

    # Store references on app.state for route access
    app.state.engine = _engine
    app.state.agent = _agent
    app.state.manager = _manager
    app.state.bio_age_calc = _bio_age_calc
    app.state.trajectory_analyzer = _trajectory_analyzer
    app.state.pgx_mapper = _pgx_mapper
    app.state.genotype_adjuster = _genotype_adjuster
    app.state.metrics = _metrics

    yield

    # -- Shutdown --
    if _manager:
        _manager.disconnect()


# =====================================================================
# FastAPI app
# =====================================================================

app = FastAPI(
    title="Precision Biomarker Agent API",
    description=(
        "REST API for the Precision Biomarker Intelligence Agent -- multi-collection "
        "RAG engine with biological age calculation, disease trajectory analysis, "
        "pharmacogenomic mapping, and genotype-adjusted reference ranges."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# -- CORS middleware --
_cors_origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- Request size limit middleware --
@app.middleware("http")
async def _limit_request_size(request: Request, call_next):
    """Reject request bodies that exceed the configured size limit."""
    content_length = request.headers.get("content-length")
    max_bytes = settings.MAX_REQUEST_SIZE_MB * 1024 * 1024
    if content_length and int(content_length) > max_bytes:
        return JSONResponse(status_code=413, content={"detail": "Request body too large"})
    return await call_next(request)


# -- Include route modules --
app.include_router(analysis_router)
app.include_router(reports_router)
app.include_router(events_router)


# =====================================================================
# Pydantic request / response schemas
# =====================================================================

class HealthResponse(BaseModel):
    """Response schema for GET /health."""
    status: str = "healthy"
    collections: int = Field(..., description="Number of active collections")
    total_vectors: int = Field(..., description="Total vectors across all collections")
    agent_ready: bool = Field(..., description="Whether the full agent is initialized")


class CollectionInfo(BaseModel):
    """Single collection metadata."""
    name: str
    record_count: int


class CollectionsResponse(BaseModel):
    """Response schema for GET /collections."""
    collections: List[CollectionInfo]
    total: int


class QueryRequest(BaseModel):
    """Request schema for POST /query and POST /search."""
    question: str = Field(..., min_length=1, description="Natural-language question")
    disease_area: Optional[str] = Field(None, max_length=50, description="Filter by disease area")
    collections: Optional[List[str]] = Field(None, description="Restrict search to specific collections")
    year_min: Optional[int] = Field(None, ge=1990, le=2030, description="Minimum publication year")
    year_max: Optional[int] = Field(None, ge=1990, le=2030, description="Maximum publication year")


class EvidenceItem(BaseModel):
    """A single piece of evidence returned to the client."""
    collection: str
    id: str
    score: float
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """Response schema for POST /query (RAG with LLM)."""
    question: str
    answer: str
    evidence: List[EvidenceItem]
    knowledge_context: str = ""
    collections_searched: int = 0
    search_time_ms: float = 0.0


class SearchResponse(BaseModel):
    """Response schema for POST /search (evidence only, no LLM)."""
    question: str
    evidence: List[EvidenceItem]
    knowledge_context: str = ""
    collections_searched: int = 0
    search_time_ms: float = 0.0


class BiologicalAgeRequest(BaseModel):
    """Request schema for POST /biological-age."""
    age: int = Field(..., ge=0, le=150, description="Chronological age")
    biomarkers: Dict[str, float] = Field(..., description="Biomarker name -> value mapping")


class BiologicalAgeResponse(BaseModel):
    """Response schema for POST /biological-age."""
    chronological_age: int
    biological_age: float
    age_acceleration: float
    mortality_risk: str
    top_aging_drivers: List[Dict]


class KnowledgeStatsResponse(BaseModel):
    """Response schema for GET /knowledge/stats."""
    disease_domains: int
    total_biomarkers: int
    total_genetic_modifiers: int
    pharmacogenes: int
    pgx_drug_interactions: int
    phenoage_markers: int
    cross_modal_links: int


# =====================================================================
# Helper -- convert internal SearchHit to API EvidenceItem
# =====================================================================

def _hit_to_evidence(hit: SearchHit) -> EvidenceItem:
    """Convert an internal SearchHit to the API EvidenceItem schema."""
    return EvidenceItem(
        collection=hit.collection,
        id=hit.id,
        score=hit.score,
        text=hit.text,
        metadata=hit.metadata,
    )


# =====================================================================
# Endpoints
# =====================================================================

@app.get("/health", response_model=HealthResponse, tags=["status"])
async def health():
    """Return service health with collection count and total vector count."""
    _metrics["requests_total"] += 1

    agent_ready = _engine is not None and _agent is not None

    if not _manager:
        return HealthResponse(
            status="degraded",
            collections=0,
            total_vectors=0,
            agent_ready=agent_ready,
        )

    try:
        stats = _manager.get_collection_stats()
        total_collections = sum(1 for v in stats.values() if v > 0)
        total_vectors = sum(stats.values())
        return HealthResponse(
            status="healthy",
            collections=total_collections,
            total_vectors=total_vectors,
            agent_ready=agent_ready,
        )
    except Exception as e:
        _metrics["errors_total"] += 1
        raise HTTPException(status_code=503, detail=f"Milvus unavailable: {e}")


@app.get("/collections", response_model=CollectionsResponse, tags=["status"])
async def list_collections():
    """Return all collection names and their record counts."""
    _metrics["requests_total"] += 1

    if not _manager:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        stats = _manager.get_collection_stats()
        items = [
            CollectionInfo(name=name, record_count=count)
            for name, count in stats.items()
        ]
        return CollectionsResponse(
            collections=items,
            total=len(items),
        )
    except Exception as e:
        _metrics["errors_total"] += 1
        raise HTTPException(status_code=500, detail=f"Failed to fetch collection stats: {e}")


@app.post("/query", response_model=QueryResponse, tags=["rag"])
async def query(request: QueryRequest):
    """Full RAG query: retrieve evidence from Milvus, augment with the
    knowledge graph, and synthesize an LLM response.
    """
    _metrics["requests_total"] += 1
    _metrics["query_requests_total"] += 1

    if not _engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    if not _engine.llm:
        raise HTTPException(status_code=503, detail="LLM client not available")
    if not _engine.embedder:
        raise HTTPException(status_code=503, detail="Embedding model not loaded")

    try:
        agent_query = AgentQuery(question=request.question)

        evidence: CrossCollectionResult = _engine.retrieve(
            query=agent_query,
            collections_filter=request.collections,
            year_min=request.year_min,
            year_max=request.year_max,
        )

        from src.rag_engine import BIOMARKER_SYSTEM_PROMPT
        prompt_text = _engine._build_prompt(request.question, evidence)
        answer = _engine.llm.generate(
            prompt=prompt_text,
            system_prompt=BIOMARKER_SYSTEM_PROMPT,
            max_tokens=2048,
            temperature=0.7,
        )

        return QueryResponse(
            question=request.question,
            answer=answer,
            evidence=[_hit_to_evidence(h) for h in evidence.hits],
            knowledge_context=evidence.knowledge_context,
            collections_searched=evidence.total_collections_searched,
            search_time_ms=evidence.search_time_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        _metrics["errors_total"] += 1
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


@app.post("/search", response_model=SearchResponse, tags=["rag"])
async def search(request: QueryRequest):
    """Evidence-only retrieval (no LLM). Fast retrieval for evidence snippets."""
    _metrics["requests_total"] += 1
    _metrics["search_requests_total"] += 1

    if not _engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    if not _engine.embedder:
        raise HTTPException(status_code=503, detail="Embedding model not loaded")

    try:
        agent_query = AgentQuery(question=request.question)

        evidence: CrossCollectionResult = _engine.retrieve(
            query=agent_query,
            collections_filter=request.collections,
            year_min=request.year_min,
            year_max=request.year_max,
        )

        return SearchResponse(
            question=request.question,
            evidence=[_hit_to_evidence(h) for h in evidence.hits],
            knowledge_context=evidence.knowledge_context,
            collections_searched=evidence.total_collections_searched,
            search_time_ms=evidence.search_time_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        _metrics["errors_total"] += 1
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


@app.post("/analyze", tags=["analysis"])
async def analyze(profile: PatientProfile):
    """Full patient analysis using all modules (bio age, trajectories, PGx, adjustments)."""
    _metrics["requests_total"] += 1
    _metrics["analyze_requests_total"] += 1

    if not _agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        result = _agent.analyze_patient(profile)
        return JSONResponse(content=result.model_dump())
    except HTTPException:
        raise
    except Exception as e:
        _metrics["errors_total"] += 1
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


@app.post("/biological-age", response_model=BiologicalAgeResponse, tags=["analysis"])
async def biological_age(request: BiologicalAgeRequest):
    """Calculate biological age from blood biomarkers using PhenoAge algorithm."""
    _metrics["requests_total"] += 1
    _metrics["bio_age_requests_total"] += 1

    if not _bio_age_calc:
        raise HTTPException(status_code=503, detail="Bio age calculator not initialized")

    try:
        result = _bio_age_calc.calculate(request.age, request.biomarkers)
        phenoage = result.get("phenoage", {})
        return BiologicalAgeResponse(
            chronological_age=request.age,
            biological_age=result.get("biological_age", request.age),
            age_acceleration=result.get("age_acceleration", 0.0),
            mortality_risk=phenoage.get("mortality_risk", "UNKNOWN"),
            top_aging_drivers=phenoage.get("top_aging_drivers", []),
        )
    except Exception as e:
        _metrics["errors_total"] += 1
        raise HTTPException(status_code=500, detail=f"Biological age calculation failed: {e}")


@app.get("/knowledge/stats", response_model=KnowledgeStatsResponse, tags=["knowledge"])
async def knowledge_stats():
    """Return statistics about the biomarker knowledge graph."""
    _metrics["requests_total"] += 1

    try:
        stats = get_knowledge_stats()
        return KnowledgeStatsResponse(**stats)
    except Exception as e:
        _metrics["errors_total"] += 1
        raise HTTPException(status_code=500, detail=f"Knowledge stats failed: {e}")


@app.get("/metrics", response_class=PlainTextResponse, tags=["monitoring"])
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    lines = [
        "# HELP biomarker_api_requests_total Total API requests",
        "# TYPE biomarker_api_requests_total counter",
        f'biomarker_api_requests_total {_metrics["requests_total"]}',
        "",
        "# HELP biomarker_api_query_requests_total Total /query requests",
        "# TYPE biomarker_api_query_requests_total counter",
        f'biomarker_api_query_requests_total {_metrics["query_requests_total"]}',
        "",
        "# HELP biomarker_api_search_requests_total Total /search requests",
        "# TYPE biomarker_api_search_requests_total counter",
        f'biomarker_api_search_requests_total {_metrics["search_requests_total"]}',
        "",
        "# HELP biomarker_api_analyze_requests_total Total /analyze requests",
        "# TYPE biomarker_api_analyze_requests_total counter",
        f'biomarker_api_analyze_requests_total {_metrics["analyze_requests_total"]}',
        "",
        "# HELP biomarker_api_bio_age_requests_total Total /biological-age requests",
        "# TYPE biomarker_api_bio_age_requests_total counter",
        f'biomarker_api_bio_age_requests_total {_metrics["bio_age_requests_total"]}',
        "",
        "# HELP biomarker_api_errors_total Total error responses",
        "# TYPE biomarker_api_errors_total counter",
        f'biomarker_api_errors_total {_metrics["errors_total"]}',
        "",
    ]

    # Add collection vector counts if available
    if _manager:
        try:
            stats = _manager.get_collection_stats()
            lines.append("# HELP biomarker_collection_vectors Number of vectors per collection")
            lines.append("# TYPE biomarker_collection_vectors gauge")
            for name, count in stats.items():
                lines.append(f'biomarker_collection_vectors{{collection="{name}"}} {count}')
            lines.append("")
        except Exception:
            pass

    return "\n".join(lines) + "\n"


# =====================================================================
# Entrypoint for direct execution
# =====================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
    )
