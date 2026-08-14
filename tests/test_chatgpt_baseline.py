
import importlib.util
import sys
from pathlib import Path

import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    PROJECT_DIR
    / "scripts"
    / "06_chatgpt-baseline"
    / "benchmark_inference.py"
)
SPEC = importlib.util.spec_from_file_location("chatgpt_benchmark", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def completed_response(score):
    return {
        "id": "response-1",
        "model": "gpt-5.6-luna",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": f'{{"ai_score": {score}}}',
                    }
                ],
            }
        ],
    }


def test_prompt_document_is_the_payload_source_of_truth():
    prompt = MODULE.load_prompt_config(
        PROJECT_DIR / "scripts" / "06_chatgpt-baseline" / "PROMPT.md"
    )
    payload = MODULE.build_payload(
        prompt,
        "A sample to classify.",
        model="gpt-5.6-luna",
        reasoning_effort="medium",
        max_output_tokens=2_048,
    )

    assert payload["model"] == "gpt-5.6-luna"
    assert payload["reasoning"] == {"effort": "medium"}
    assert payload["store"] is False
    assert "temperature" not in payload
    assert "A sample to classify." in payload["input"]
    assert payload["text"]["format"]["strict"] is True
    assert payload["text"]["format"]["schema"] == prompt.output_schema


def test_score_parser_enforces_integer_range():
    assert MODULE.parse_ai_score(completed_response(73)) == 73

    with pytest.raises(MODULE.APIError, match="outside 0 to 100"):
        MODULE.parse_ai_score(completed_response(101))


def test_pass_summary_reports_confusion_retries_and_usage():
    rows = [
        {
            "true_label": 0,
            "prediction": 0,
            "latency_seconds": 1.0,
            "attempts": 1,
            "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        },
        {
            "true_label": 1,
            "prediction": 0,
            "latency_seconds": 3.0,
            "attempts": 2,
            "usage": {"input_tokens": 20, "output_tokens": 3, "total_tokens": 23},
        },
    ]

    summary = MODULE.summarize_pass(rows, duration_seconds=2.0)

    assert summary["accuracy"] == 0.5
    assert summary["confusion"] == {
        "true_negative": 1,
        "false_positive": 0,
        "false_negative": 1,
        "true_positive": 0,
    }
    assert summary["retries"] == 1
    assert summary["usage"]["input_tokens"] == 30
    assert summary["usage"]["total_tokens"] == 35


def test_five_pass_summary_measures_repeat_agreement():
    pass_payloads = []
    for pass_index in range(1, 6):
        rows = [
            {
                "sample_id": "human-1",
                "true_label": 0,
                "prediction": 0,
                "ai_score": 10 + pass_index,
            },
            {
                "sample_id": "ai-1",
                "true_label": 1,
                "prediction": 1,
                "ai_score": 90 - pass_index,
            },
        ]
        pass_payloads.append(
            {
                "pass_index": pass_index,
                "results": rows,
                "summary": {
                    "duration_seconds": float(pass_index),
                    "accuracy": 1.0,
                    "usage": {
                        "input_tokens": 20,
                        "cached_input_tokens": 0,
                        "output_tokens": 2,
                        "reasoning_tokens": 0,
                        "total_tokens": 22,
                    },
                },
            }
        )

    summary = MODULE.overall_summary(pass_payloads, {"threshold": 50})

    assert summary["passes"] == 5
    assert summary["total_classifications"] == 10
    assert summary["mean_accuracy"] == 1.0
    assert summary["unanimous_prediction_rate"] == 1.0
    assert summary["majority_vote_accuracy"] == 1.0
    assert summary["majority_vote_confusion"] == {
        "true_negative": 1,
        "false_positive": 0,
        "false_negative": 0,
        "true_positive": 1,
    }
    assert summary["total_usage"]["total_tokens"] == 110


def test_even_pass_tie_uses_the_configured_threshold():
    rows = [
        {
            "sample_id": "sample-1",
            "true_label": 1,
            "prediction": 0,
            "ai_score": 55,
        }
    ]
    second_rows = [
        {
            "sample_id": "sample-1",
            "true_label": 1,
            "prediction": 1,
            "ai_score": 75,
        }
    ]
    usage = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
    }
    pass_payloads = [
        {
            "results": rows,
            "summary": {"duration_seconds": 1.0, "accuracy": 0.0, "usage": usage},
        },
        {
            "results": second_rows,
            "summary": {"duration_seconds": 1.0, "accuracy": 1.0, "usage": usage},
        },
    ]

    summary = MODULE.overall_summary(pass_payloads, {"threshold": 70})

    assert summary["majority_vote_accuracy"] == 0.0
