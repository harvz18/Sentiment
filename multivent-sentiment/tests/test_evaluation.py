import pytest

from scripts.evaluate import binary_metrics


def test_binary_metric_definitions():
    report = binary_metrics(["positive", "positive", "negative", "negative"],
                            ["positive", "negative", "negative", "negative"])
    assert report["accuracy"] == .75
    assert report["precision"] == 1
    assert report["recall"] == .5
    assert report["f1"] == pytest.approx(2 / 3)
    assert report["confusion_matrix"] == [[2, 0], [1, 1]]


def test_evaluation_rejects_neutral_predictions():
    with pytest.raises(ValueError):
        binary_metrics(["positive"], ["neutral"])
