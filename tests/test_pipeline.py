import pytest

from pulso_transmi.pipeline import PipelineError, stable_idempotency_key, validate_predictions


def target(station_id: str, target_at: str) -> dict[str, str]:
    return {"station_id": station_id, "target_at": target_at}


def prediction(station_id: str, target_at: str, value: float = 10.0) -> dict[str, object]:
    return {"station_id": station_id, "target_at": target_at, "value": value}


def test_validate_predictions_requires_exact_targets() -> None:
    targets = [target("02300", "2026-09-10T06:15:00Z")]
    predictions = [prediction("02300", "2026-09-10T06:15:00Z")]

    validate_predictions(predictions, targets, expected=1)


def test_validate_predictions_rejects_duplicate_targets() -> None:
    targets = [
        target("02300", "2026-09-10T06:15:00Z"),
        target("02300", "2026-09-10T06:30:00Z"),
    ]
    predictions = [
        prediction("02300", "2026-09-10T06:15:00Z"),
        prediction("02300", "2026-09-10T06:15:00Z"),
    ]

    with pytest.raises(PipelineError, match="exactly one row"):
        validate_predictions(predictions, targets, expected=2)


def test_idempotency_key_is_stable_and_changes_with_content() -> None:
    predictions = [prediction("02300", "2026-09-10T06:15:00Z")]

    first = stable_idempotency_key("cycle-1", "model-v1", predictions)
    second = stable_idempotency_key("cycle-1", "model-v1", predictions)
    changed = stable_idempotency_key(
        "cycle-1", "model-v1", [prediction("02300", "2026-09-10T06:15:00Z", 11.0)]
    )

    assert first == second
    assert first != changed
    assert len(first) >= 8
