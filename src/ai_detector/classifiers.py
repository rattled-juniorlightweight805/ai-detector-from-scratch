"""Model loaders for the trained AI-text classifiers."""

import json
from pathlib import Path

from .registry import ensure_artifact_ready, model_registry
from .scoring import score_payload, validate_text


def _load_metadata(artifact_path):
    metadata_path = artifact_path / "detector-config.json"
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _resolve_device(requested):
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: auto, cpu, cuda, mps")
    return torch.device(requested)


class SklearnTextClassifier:
    def __init__(self, spec):
        import numpy as np
        from joblib import load

        self.spec = spec
        self.model = load(spec.artifact_path)
        classes = np.asarray(self.model.classes_)
        matches = np.flatnonzero(classes == 1)
        if matches.size != 1:
            raise ValueError("The saved classifier must contain AI class 1")
        self.ai_column = int(matches[0])

    def score_many(
        self, texts, *, batch_size = 8
    ):
        del batch_size
        validated = [validate_text(text) for text in texts]
        if not validated:
            return []
        probabilities = self.model.predict_proba(validated)
        return [
            float(probability)
            for probability in probabilities[:, self.ai_column]
        ]


class EncoderTextClassifier:
    def __init__(
        self, spec, *, device = "auto"
    ):
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self.spec = spec
        self.metadata = _load_metadata(spec.artifact_path)
        self.device = _resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(spec.artifact_path)
        if spec.kind == "peft":
            from peft import AutoPeftModelForSequenceClassification

            self.model = (
                AutoPeftModelForSequenceClassification.from_pretrained(
                    spec.artifact_path
                )
            )
        else:
            self.model = (
                AutoModelForSequenceClassification.from_pretrained(
                    spec.artifact_path
                )
            )
        self.model.to(self.device)
        self.model.eval()

        self.max_length = int(self.metadata["max_length"])
        self.temperature = float(self.metadata.get("temperature", 1.0))
        self.ai_index = int(
            self.metadata.get("label_mapping", {}).get("ai", 1)
        )
        if self.temperature <= 0:
            raise ValueError("Saved temperature must be positive")

    def score_many(
        self, texts, *, batch_size = 8
    ):
        import torch

        validated = [validate_text(text) for text in texts]
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        probabilities = []
        for start in range(0, len(validated), batch_size):
            batch = validated[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(self.device)
                for key, value in encoded.items()
            }
            with torch.inference_mode():
                logits = self.model(**encoded).logits
                batch_probabilities = (
                    (logits / self.temperature)
                    .softmax(dim=1)[:, self.ai_index]
                    .float()
                    .cpu()
                    .tolist()
                )
            probabilities.extend(batch_probabilities)
        return probabilities


def build_causal_batch(
    encoded_input_ids,
    *,
    pad_token_id,
    eos_token_id,
    context_length,
    readout_position,
):
    if readout_position not in {"variable", "fixed"}:
        raise ValueError("readout_position must be 'variable' or 'fixed'")
    if context_length < 1:
        raise ValueError("context length must be positive")
    if any(len(input_ids) + 1 > context_length for input_ids in encoded_input_ids):
        raise ValueError("encoded input exceeds the configured context length")
    if not encoded_input_ids:
        return [], []

    input_rows = []
    mask_rows = []
    if readout_position == "variable":
        batch_length = max(len(input_ids) + 1 for input_ids in encoded_input_ids)
        for input_ids in encoded_input_ids:
            padding_length = batch_length - len(input_ids) - 1
            input_rows.append(
                list(input_ids)
                + [eos_token_id]
                + [pad_token_id] * padding_length
            )
            mask_rows.append(
                [1] * (len(input_ids) + 1) + [0] * padding_length
            )
    else:
        for input_ids in encoded_input_ids:
            padding_length = context_length - len(input_ids) - 1
            input_rows.append(
                list(input_ids)
                + [pad_token_id] * padding_length
                + [eos_token_id]
            )
            mask_rows.append(
                [1] * len(input_ids)
                + [0] * padding_length
                + [1]
            )
    return input_rows, mask_rows


class CausalTextClassifier:
    def __init__(
        self, spec, *, device = "auto"
    ):
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self.spec = spec
        self.metadata = _load_metadata(spec.artifact_path)
        self.device = _resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(spec.artifact_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            spec.artifact_path
        )
        self.model.to(self.device)
        self.model.eval()

        self.max_text_length = int(self.metadata["max_text_length"])
        self.context_length = int(self.metadata["context_length"])
        self.readout_position = str(self.metadata["readout_position"])
        self.temperature = float(self.metadata.get("temperature", 1.0))
        self.ai_index = int(
            self.metadata.get("label_mapping", {}).get("ai", 1)
        )
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Saved tokenizer does not define a padding token")
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Saved tokenizer does not define an EOS token")
        if self.temperature <= 0:
            raise ValueError("Saved temperature must be positive")

    def _prepare_batch(self, texts):
        import torch

        encoded = self.tokenizer(
            list(texts),
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_text_length,
        )
        input_rows, mask_rows = build_causal_batch(
            encoded["input_ids"],
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            context_length=self.context_length,
            readout_position=self.readout_position,
        )
        return {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "attention_mask": torch.tensor(mask_rows, dtype=torch.long),
        }

    def score_many(
        self, texts, *, batch_size = 1
    ):
        import torch

        validated = [validate_text(text) for text in texts]
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        probabilities = []
        for start in range(0, len(validated), batch_size):
            batch = validated[start : start + batch_size]
            encoded = {
                key: value.to(self.device)
                for key, value in self._prepare_batch(batch).items()
            }
            with torch.inference_mode():
                logits = self.model(**encoded).logits
                batch_probabilities = (
                    (logits / self.temperature)
                    .softmax(dim=1)[:, self.ai_index]
                    .float()
                    .cpu()
                    .tolist()
                )
            probabilities.extend(batch_probabilities)
        return probabilities


def load_classifier(
    model_name,
    *,
    artifact_path = None,
    device = "auto",
):
    registry = model_registry()
    if model_name not in registry:
        available = ", ".join(sorted(registry))
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available}"
        )
    spec = registry[model_name]
    if artifact_path is not None:
        spec = spec._replace(
            artifact_path=Path(artifact_path).expanduser().resolve()
        )
    ensure_artifact_ready(spec)

    if spec.kind == "sklearn":
        return SklearnTextClassifier(spec)
    if spec.kind in {"encoder", "peft"}:
        return EncoderTextClassifier(spec, device=device)
    return CausalTextClassifier(spec, device=device)


def score_text(
    text,
    *,
    model_name,
    artifact_path = None,
    device = "auto",
):
    classifier = load_classifier(
        model_name, artifact_path=artifact_path, device=device
    )
    probability = classifier.score_many([text], batch_size=1)[0]
    return score_payload(probability)
