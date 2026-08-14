"""Reusable inference helpers for the AI text detector."""

from .classifiers import (
    CausalTextClassifier,
    EncoderTextClassifier,
    SklearnTextClassifier,
    build_causal_batch,
    load_classifier,
    score_text,
)
from .registry import (
    PROJECT_DIR,
    ArtifactNotReadyError,
    ModelSpec,
    artifact_status,
    ensure_artifact_ready,
    model_registry,
)
from .scoring import score_payload, validate_text


__all__ = [
    "PROJECT_DIR",
    "ArtifactNotReadyError",
    "CausalTextClassifier",
    "EncoderTextClassifier",
    "ModelSpec",
    "SklearnTextClassifier",
    "artifact_status",
    "build_causal_batch",
    "ensure_artifact_ready",
    "load_classifier",
    "model_registry",
    "score_payload",
    "score_text",
    "validate_text",
]
