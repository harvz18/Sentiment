from hmac import compare_digest

from fastapi import APIRouter, Header, HTTPException, Request

from app.artifacts import read_json
from app.preprocessing.text_cleaner import prepare_text
from app.preprocessing.text_normalizer import normalize_informal_text
from app.schemas.sentiment import (AnalysisResponse, FeedbackRequest,
                                    FeedbackSummaryRequest, FeedbackSummaryResponse,
                                    HealthResponse, NormalizationRequest,
                                    NormalizationResponse, RetrievalRequest,
                                    RetrievalResult, TopicInfo)
from app.sentiment.base_provider import ProviderError

router = APIRouter()


def require_api_key(request: Request, supplied: str | None) -> None:
    configured = request.app.state.runtime.settings.analysis_api_key.get_secret_value().strip()
    if configured and (supplied is None or not compare_digest(configured, supplied)):
        raise HTTPException(401, detail="Invalid or missing analysis API key")


def require(request: Request, component: str):
    runtime = request.app.state.runtime
    value = getattr(runtime, component)
    if value is None:
        raise HTTPException(503, detail=runtime.health().model_dump())
    return value


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    return request.app.state.runtime.health()


@router.post("/normalize", response_model=NormalizationResponse)
def normalize(body: NormalizationRequest, request: Request,
              x_analysis_key: str | None = Header(None)):
    require_api_key(request, x_analysis_key)
    result = normalize_informal_text(body.text)
    normalizations = [change.__dict__ for change in result.changes]
    return NormalizationResponse(
        original=result.original,
        normalized=result.normalized,
        changes=[f"{change.original} → {change.normalized}" for change in result.changes],
        normalizations=normalizations,
    )


@router.post("/analyze", response_model=AnalysisResponse, response_model_exclude_none=True)
def analyze(body: FeedbackRequest, request: Request, debug: bool = False,
            x_analysis_key: str | None = Header(None)):
    require_api_key(request, x_analysis_key)
    service = require(request, "analysis")
    try:
        return service.analyze(body.feedback, debug=debug)
    except ProviderError as exc:
        raise HTTPException(502, detail=str(exc)) from exc


@router.post("/summarize", response_model=FeedbackSummaryResponse)
def summarize(body: FeedbackSummaryRequest, request: Request,
              x_analysis_key: str | None = Header(None)):
    require_api_key(request, x_analysis_key)
    provider = require(request, "provider")
    try:
        return provider.summarize(body)
    except ProviderError as exc:
        raise HTTPException(502, detail=str(exc)) from exc


@router.post("/retrieve", response_model=RetrievalResult)
def retrieve(body: RetrievalRequest, request: Request, x_analysis_key: str | None = Header(None)):
    require_api_key(request, x_analysis_key)
    service = require(request, "retrieval")
    top_k = body.top_k or request.app.state.runtime.settings.rag_top_k
    return RetrievalResult(retrieved_examples=service.retrieve(body.feedback, top_k))


@router.post("/topic", response_model=TopicInfo)
def topic(body: FeedbackRequest, request: Request, x_analysis_key: str | None = Header(None)):
    require_api_key(request, x_analysis_key)
    service = require(request, "lda")
    settings = request.app.state.runtime.settings
    normalized = normalize_informal_text(body.feedback).normalized
    return service.infer(prepare_text(normalized, settings.enable_pii_sanitization))


@router.get("/topics")
def topics(request: Request, x_analysis_key: str | None = Header(None)) -> list[dict]:
    require_api_key(request, x_analysis_key)
    return require(request, "lda").all_topics()


@router.get("/dataset/stats")
def dataset_stats(request: Request, x_analysis_key: str | None = Header(None)) -> dict:
    require_api_key(request, x_analysis_key)
    path = request.app.state.runtime.settings.corpus_dir / "manifest.json"
    try:
        manifest = read_json(path)
        return {key: manifest[key] for key in ("binary_sentiment", "source_stats", "total_source_records_examined",
                                               "neutral_excluded", "sizes", "threshold_statistics")}
    except FileNotFoundError as exc:
        raise HTTPException(503, "Dataset report missing. Run python scripts/prepare_multivent_dataset.py") from exc
