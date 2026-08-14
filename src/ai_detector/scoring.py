"""Input validation and public score formatting."""

import math


def validate_text(text):
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        raise ValueError("text cannot be empty")
    return text


def score_payload(ai_probability):
    probability = float(ai_probability)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("AI probability must be between 0 and 1")
    return {"score": round(probability * 100.0, 4)}
