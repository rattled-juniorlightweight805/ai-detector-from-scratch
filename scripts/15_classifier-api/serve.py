#!/usr/bin/env python3
"""Keep one AI-text classifier in memory behind a local HTTP API."""

import argparse
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

from fastapi import Body, FastAPI, HTTPException
from pydantic import Field, ValidationError, create_model

from ai_detector import (
    load_classifier,
    model_registry,
    score_payload,
    validate_text,
)


CheckRequest = create_model(
    "CheckRequest",
    text=(str, Field(min_length=1)),
)
ScoreResponse = create_model("ScoreResponse", score=(float, ...))


def create_app(
    *,
    model_name,
    artifact_path = None,
    device = "auto",
    classifier = None,
):
    """Create an app that loads its classifier once during startup."""

    @asynccontextmanager
    async def lifespan(application):
        application.state.classifier = (
            classifier
            if classifier is not None
            else load_classifier(
                model_name,
                artifact_path=artifact_path,
                device=device,
            )
        )
        application.state.inference_lock = Lock()
        yield
        del application.state.classifier
        del application.state.inference_lock

    application = FastAPI(
        title="Local AI Detector API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.get("/api/health")
    def health():
        return {"status": "ok", "model": model_name}

    @application.post("/api/check", response_model=ScoreResponse)
    def check_text(payload=Body(...)):
        try:
            payload = CheckRequest(**payload)
        except (TypeError, ValidationError) as error:
            detail = error.errors() if isinstance(error, ValidationError) else str(error)
            raise HTTPException(status_code=422, detail=detail) from error
        try:
            text = validate_text(payload.text)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        with application.state.inference_lock:
            probability = application.state.classifier.score_many(
                [text], batch_size=1
            )[0]
        return ScoreResponse(**score_payload(probability))

    return application


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load one AI-text classifier and keep it in memory behind a "
            "local HTTP API."
        )
    )
    parser.add_argument(
        "--model",
        choices=sorted(model_registry()),
        required=True,
        help="Classifier to load during server startup",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        help="Override the model's default exported artifact path",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Torch inference device (default: auto)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="TCP port (default: 8000)",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65_535:
        parser.error("--port must be between 1 and 65535")
    return args


def main():
    import uvicorn

    args = parse_args()
    application = create_app(
        model_name=args.model,
        artifact_path=args.artifact,
        device=args.device,
    )
    uvicorn.run(application, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
