
import importlib.util
import json
from pathlib import Path
import sys

from fastapi.testclient import TestClient


PROJECT_DIR = Path(__file__).resolve().parents[1]
API_DIR = PROJECT_DIR / "scripts" / "15_classifier-api"
SERVER_PATH = API_DIR / "serve.py"
CLI_PATH = API_DIR / "classify.py"


def load_module(name, path):
    if str(API_DIR) not in sys.path:
        sys.path.insert(0, str(API_DIR))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeClassifier:
    def __init__(self, probability):
        self.probability = probability
        self.received_texts = []

    def score_many(self, texts, *, batch_size=8):
        assert batch_size == 1
        self.received_texts.extend(texts)
        return [self.probability for _ in texts]


def test_server_loads_once_and_reuses_classifier(monkeypatch):
    module = load_module("classifier_server", SERVER_PATH)
    classifier = FakeClassifier(0.87341)
    load_calls = []

    def fake_load_classifier(model_name, *, artifact_path, device):
        load_calls.append((model_name, artifact_path, device))
        return classifier

    monkeypatch.setattr(module, "load_classifier", fake_load_classifier)
    app = module.create_app(model_name="distilbert", device="cpu")

    with TestClient(app) as client:
        first = client.post("/api/check", json={"text": "first text"})
        second = client.post("/api/check", json={"text": "second text"})
        health = client.get("/api/health")

    assert load_calls == [("distilbert", None, "cpu")]
    assert first.json() == {"score": 87.341}
    assert second.json() == {"score": 87.341}
    assert health.json() == {"status": "ok", "model": "distilbert"}
    assert classifier.received_texts == ["first text", "second text"]


def test_server_rejects_whitespace_only_text():
    module = load_module("classifier_server_whitespace", SERVER_PATH)
    app = module.create_app(
        model_name="logreg",
        classifier=FakeClassifier(0.5),
    )

    with TestClient(app) as client:
        response = client.post("/api/check", json={"text": "   \n"})

    assert response.status_code == 422
    assert response.json()["detail"] == "text cannot be empty"


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_cli_server_client_posts_text_and_preserves_score(monkeypatch):
    module = load_module("classifier_cli", CLI_PATH)
    requests = []

    def fake_urlopen(request, *, timeout):
        requests.append((request, timeout))
        return FakeResponse({"score": 73.4567})

    monkeypatch.setattr(module, "urlopen", fake_urlopen)
    payload = module.request_server_score(
        "http://127.0.0.1:8000",
        "Text to classify.",
        timeout=12.0,
    )

    request, timeout = requests[0]
    assert request.full_url == "http://127.0.0.1:8000/api/check"
    assert json.loads(request.data) == {"text": "Text to classify."}
    assert timeout == 12.0
    assert payload == {"score": 73.4567}
