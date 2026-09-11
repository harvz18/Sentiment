import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.main import create_app
from app.schemas.sentiment import BinaryScores, ReferenceExample, TopicInfo
from app.sentiment.base_provider import ProviderError
from app.sentiment.factory import create_provider
from app.sentiment.heuristic_provider import HeuristicSentimentProvider
from app.sentiment.openai_provider import OpenAISentimentProvider, OpenAIVerdict


def example(sentiment, score):
    return ReferenceExample(id="a", text="reference", sentiment=sentiment,
                            similarity=score, split="train", source_dataset="fixture", source_name="fixture",
                            source_record_id="a", source_domain="hospitality", source_url="https://example.test",
                            license="CC-BY-4.0", language="english", aspect="general", pool="domain",
                            domain_relevance_score=1)


def test_fallback_vote_tie_and_no_evidence():
    topic = TopicInfo(id=0, name="Unassigned", probability=.8, keywords=[])
    provider = HeuristicSentimentProvider()
    result = provider.analyze("feedback", topic, [example("positive", .8), example("negative", .2)])
    assert result.sentiment == "positive" and result.confidence == pytest.approx(.8)
    assert result.provider == "rag_weighted_fallback" and result.language == "unknown"
    for references in ([], [example("negative", -.5)], [example("positive", .5), example("negative", .5)]):
        verdict = provider.analyze("feedback", topic, references)
        assert verdict.sentiment == "negative"
        assert verdict.scores == BinaryScores(positive=.5, negative=.5)
        assert verdict.confidence == (0.5 if len(references) == 2 else 0)


def test_missing_key_and_mock_selection(settings):
    settings.ai_provider = "openai"
    assert create_provider(settings).name == "rag_weighted_fallback"
    settings.ai_provider = "mock"
    assert create_provider(settings).name == "mock"
    settings.ai_provider = "rag_fallback"
    settings.openai_api_key = SecretStr("test-key")
    assert create_provider(settings).name == "rag_weighted_fallback"


@pytest.mark.parametrize("body", [{}, {"feedback": " "}, {"feedback": "!!!"}, {"feedback": 42},
                                    {"feedback": "x" * 5001}, {"feedback": "ok", "extra": 1}])
def test_api_validation(runtime, body):
    with TestClient(create_app(runtime=runtime)) as client:
        assert client.post("/analyze", json=body).status_code == 422


def test_all_routes(runtime):
    with TestClient(create_app(runtime=runtime)) as client:
        assert client.get("/").json()["status"] == "ok"
        assert "Sentiment Lab" in client.get("/", headers={"Accept": "text/html"}).text
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/health").json()["lda_loaded"] is True
        assert len(client.get("/topics").json()) == 2
        text = "  Nami gid ang food.  "
        result = client.post("/analyze", json={"feedback": text})
        assert result.status_code == 200, result.text
        assert result.json()["feedback"] == text
        assert result.json()["provider"] == "rag_weighted_fallback"
        assert result.json()["sentiment"] in {"positive", "negative"}
        assert "debug" not in result.json()
        debug = client.post("/analyze?debug=true", json={"feedback": text}).json()["debug"]
        assert len(debug["evidence"]) == 6
        assert debug["positive_score"] + debug["negative_score"] > 0
        assert all(r["split"] == "train" for r in result.json()["rag"]["retrieved_examples"])
        assert client.post("/topic", json={"feedback": text}).status_code == 200
        assert len(client.post("/retrieve", json={"feedback": text, "top_k": 1}).json()["retrieved_examples"]) == 1
        assert client.post("/retrieve", json={"feedback": text, "top_k": 0}).status_code == 422


def test_missing_artifacts_still_serves_health(settings):
    with TestClient(create_app(settings=settings)) as client:
        assert client.get("/health").json()["status"] == "degraded"
        assert client.post("/analyze", json={"feedback": "Nami gid"}).status_code == 503
        assert client.get("/dataset/stats").status_code == 503


def test_experiment_logging(runtime):
    runtime.settings.enable_analysis_logging = True
    runtime.analysis.analyze("Nami gid", ground_truth="positive")
    event = json.loads((runtime.settings.root / "logs/analysis.jsonl").read_text())
    assert event["ground_truth"] == "positive" and event["latency_ms"] > 0
    assert event["retrieved_example_ids"] and "openai_api_key" not in event


def test_openai_adapter_contract_without_network(runtime):
    provider = OpenAISentimentProvider.__new__(OpenAISentimentProvider)
    provider.model = "test-model"
    verdict = OpenAIVerdict(sentiment="negative", confidence=.8, scores=BinaryScores(positive=.2, negative=.8), language="Hiligaynon", reason="Test verdict")
    parse = Mock(return_value=SimpleNamespace(status="completed", output_parsed=verdict))
    provider.client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
    result = provider.analyze("Nami pero dugay.", runtime.lda.infer("dugay"), [])
    assert result.provider == "openai"
    request = parse.call_args.kwargs
    assert request["store"] is False and request["text_format"] is OpenAIVerdict
    assert json.loads(request["input"][1]["content"])["feedback"] == "Nami pero dugay."
    parse.return_value = SimpleNamespace(status="completed", output_parsed=None)
    with pytest.raises(ProviderError):
        provider.analyze("feedback", runtime.lda.infer("dugay"), [])
