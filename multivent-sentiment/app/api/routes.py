from fastapi import APIRouter, HTTPException, Request

from app.artifacts import read_json
from app.preprocessing.text_cleaner import prepare_text
from app.schemas.sentiment import (AnalysisResponse, FeedbackRequest, HealthResponse,
                                    RetrievalRequest, RetrievalResult, TopicInfo)
from app.sentiment.base_provider import ProviderError

router = APIRouter()


def require(request: Request, component: str):
    runtime = request.app.state.runtime
    value = getattr(runtime, component)
    if value is None:
        raise HTTPException(503, detail=runtime.health().model_dump())
    return value


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    return request.app.state.runtime.health()


@router.post("/analyze", response_model=AnalysisResponse, response_model_exclude_none=True)
def analyze(body: FeedbackRequest, request: Request, debug: bool = False):
    service = require(request, "analysis")
    try:
        return service.analyze(body.feedback, debug=debug)
    except ProviderError as exc:
        raise HTTPException(502, detail=str(exc)) from exc


@router.post("/retrieve", response_model=RetrievalResult)
def retrieve(body: RetrievalRequest, request: Request):
    service = require(request, "retrieval")
    top_k = body.top_k or request.app.state.runtime.settings.rag_top_k
    return RetrievalResult(retrieved_examples=service.retrieve(body.feedback, top_k))


@router.post("/topic", response_model=TopicInfo)
def topic(body: FeedbackRequest, request: Request):
    service = require(request, "lda")
    settings = request.app.state.runtime.settings
    return service.infer(prepare_text(body.feedback, settings.enable_pii_sanitization))


@router.get("/topics")
def topics(request: Request) -> list[dict]:
    return require(request, "lda").all_topics()


@router.get("/dataset/stats")
def dataset_stats(request: Request) -> dict:
    path = request.app.state.runtime.settings.corpus_dir / "manifest.json"
    try:
        manifest = read_json(path)
        return {key: manifest[key] for key in ("binary_sentiment", "source_stats", "total_source_records_examined",
                                               "neutral_excluded", "sizes", "threshold_statistics")}
    except FileNotFoundError as exc:
        raise HTTPException(503, "Dataset report missing. Run python scripts/prepare_multivent_dataset.py") from exc
