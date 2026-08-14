import argparse
import importlib.util
import io
import json
import multiprocessing
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
from pathlib import Path
from unittest import mock


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "02_ai-data"
    / "_generate_ai_texts.py"
)
SPEC = importlib.util.spec_from_file_location("generate_ai_texts", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def ollama_args(data_dir, *, count):
    return argparse.Namespace(
        provider="ollama",
        model="test-model",
        meta=data_dir / "meta.json",
        output_folder=data_dir / "ai",
        count=count,
        seed=17,
        timeout=10.0,
        max_retries=2,
        length_tolerance=0.20,
        workers=1,
        api_base="http://localhost:11434",
        api_key=None,
        temperature=0.9,
        top_p=0.95,
        keep_alive="1m",
    )


def openai_repair_args(data_dir, *, count):
    args = ollama_args(data_dir, count=count)
    args.provider = "openai"
    args.model = "gpt-5.6-luna"
    args.model_selection = "luna-medium"
    args.reasoning_effort = "medium"
    args.source_model = "gpt-5.6-sol"
    args.source_model_selection = "sol-medium"
    args.source_reasoning_effort = "medium"
    return args


def commit_worker(data_dir_string, model, seed_id):
    data_dir = Path(data_dir_string)
    args = ollama_args(data_dir, count=1)
    args.model = model
    seed_sample = {
        "id": seed_id,
        "file": f"human/{seed_id}.txt",
        "label": "human",
        "collection": "test",
        "source": f"source-{seed_id}",
        "word_count": 50,
        "public_hub_eligible": True,
    }
    question = (
        f"How can example number {seed_id} illustrate safe concurrent dataset updates?"
    )
    prompt_relative = MODULE.prompt_relative_path(args, seed_id, recovered=False)
    MODULE.atomic_write_text(
        data_dir / prompt_relative, MODULE.text_for_file(question)
    )
    response = " ".join(f"response{seed_id}_{index}" for index in range(50))
    MODULE.commit_fresh_response(
        args,
        seed_sample,
        data_dir / "ai",
        response,
        MODULE.ModelResult(response, model, {}),
        prompt_relative,
        question,
        None,
    )


def repair_worker(data_dir_string, model, sample_id):
    data_dir = Path(data_dir_string)
    args = ollama_args(data_dir, count=1)
    args.model = model
    question = (
        f"How can repair example {sample_id} verify concurrent metadata updates safely?"
    )
    prompt_relative = MODULE.prompt_relative_path(args, sample_id, recovered=True)
    MODULE.atomic_write_text(
        data_dir / prompt_relative, MODULE.text_for_file(question)
    )
    response = " ".join(f"repair{sample_id}_{index}" for index in range(50))
    MODULE.commit_repair_response(
        args,
        sample_id,
        response,
        MODULE.ModelResult(response, model, {}),
        prompt_relative,
        question,
    )


class DatasetGenerationTests(unittest.TestCase):
    def test_every_repair_parser_accepts_workers(self):
        for provider in ("ollama", "openai", "openrouter", "gemini"):
            parser = MODULE.build_parser(provider, "repair")
            arguments = ["--workers", "4"]
            if provider != "gemini":
                model = "sol-medium" if provider == "openai" else "test-model"
                arguments.extend(["--model", model])
            args = parser.parse_args(arguments)
            self.assertEqual(args.workers, 4)

    def test_openai_repair_cli_keeps_source_and_target_models_separate(self):
        parser = MODULE.build_parser("openai", "repair")
        args = parser.parse_args(
            [
                "--source-model",
                "sol-medium",
                "--model",
                "luna-medium",
                "--count",
                "5071",
            ]
        )
        with mock.patch.dict(MODULE.os.environ, {"OPENAI_API_KEY": "test-key"}):
            MODULE.validate_args(args, "openai", "repair")

        self.assertEqual(args.source_model_selection, "sol-medium")
        self.assertEqual(args.source_model, "gpt-5.6-sol")
        self.assertEqual(args.model_selection, "luna-medium")
        self.assertEqual(args.model, "gpt-5.6-luna")
        self.assertEqual(MODULE.model_job_name(args), "openai-sol-medium")

    def test_http_error_preserves_openai_bio_policy_code(self):
        body = json.dumps(
            {"error": {"message": "blocked", "code": "bio_policy"}}
        ).encode("utf-8")
        http_error = urllib.error.HTTPError(
            "https://example.invalid",
            400,
            "Bad Request",
            {},
            io.BytesIO(body),
        )
        with mock.patch.object(
            MODULE.urllib.request, "urlopen", side_effect=http_error
        ):
            with self.assertRaises(MODULE.GenerationError) as raised:
                MODULE.http_json(
                    "https://example.invalid",
                    {},
                    {},
                    10.0,
                    "OpenAI",
                )

        self.assertEqual(raised.exception.code, "bio_policy")
        self.assertFalse(raised.exception.retryable)

    def test_incomplete_http_response_is_retryable(self):
        incomplete_read = MODULE.http.client.IncompleteRead(b"partial response", 42)
        with mock.patch.object(
            MODULE.urllib.request, "urlopen", side_effect=incomplete_read
        ):
            with self.assertRaises(MODULE.GenerationError) as raised:
                MODULE.http_json(
                    "https://example.invalid",
                    {},
                    {},
                    10.0,
                    "OpenRouter",
                )

        self.assertIn(
            "OpenRouter returned an incomplete response", str(raised.exception)
        )
        self.assertTrue(raised.exception.retryable)

    def test_missing_model_selection_does_not_match_another_model(self):
        generator = {
            "provider": "openrouter",
            "requested_model": "deepseek-v4-flash-0731",
            "model": "deepseek/deepseek-v4-flash-0731",
        }

        self.assertFalse(
            MODULE.generator_matches_model(
                generator,
                "openrouter",
                "moonshotai/kimi-k3",
                None,
                None,
            )
        )

    def test_openai_uses_more_headroom_for_reasoning_tokens(self):
        args = openai_repair_args(Path("/tmp/test-data"), count=1)
        args.api_base = "https://example.invalid/v1"
        args.api_key = "test-key"
        args.timeout = 10.0
        payload = {
            "status": "completed",
            "model": "gpt-5.6-luna",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "answer"}],
                }
            ],
            "usage": {},
        }
        with mock.patch.object(MODULE, "http_json", return_value=payload) as call:
            MODULE.openai_text(args, "instructions", "question", 100)

        self.assertEqual(call.call_args.args[1]["max_output_tokens"], 4096)

    def test_short_response_retry_aims_for_the_upper_bound(self):
        instructions = MODULE.response_system(100, 0.20, 58)

        self.assertIn("between 80 and 120 words", instructions)
        self.assertIn("aiming for about 120", instructions)
        self.assertIn("had only 58 words", instructions)

    def test_overlong_response_is_trimmed_at_complete_sentence(self):
        sentences = []
        for sentence_index in range(4):
            words = [
                f"sentence{sentence_index}_{word_index}" for word_index in range(35)
            ]
            words[-1] += "."
            sentences.append(" ".join(words))
        overlong_response = " ".join(sentences)
        args = ollama_args(Path("/tmp/test-data"), count=1)
        with mock.patch.object(
            MODULE,
            "call_model",
            return_value=MODULE.ModelResult(overlong_response, "test-model", {}),
        ) as call:
            response, result, error = MODULE.generate_response(
                args, "Why does this work?", 100, 1
            )

        self.assertIsNone(error)
        self.assertIsNotNone(result)
        self.assertEqual(MODULE.word_count(response), 105)
        self.assertTrue(response.endswith("."))
        self.assertEqual(call.call_count, 1)
        self.assertEqual(
            result.usage["response_postprocessing"]["type"],
            "sentence-boundary-trim",
        )

    def test_openrouter_length_finish_returns_usable_content(self):
        args = argparse.Namespace(
            model="test/model",
            api_base="https://example.invalid/v1",
            api_key="test-key",
            timeout=10.0,
            temperature=0.9,
        )
        payload = {
            "model": "test/model",
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": "usable partial response"},
                }
            ],
            "usage": {"completion_tokens": 10},
        }
        with mock.patch.object(MODULE, "http_json", return_value=payload):
            result = MODULE.openrouter_text(args, "system", "prompt", 100)

        self.assertEqual(result.text, "usable partial response")
        self.assertEqual(result.usage["finish_reason"], "length")

    def test_dataset_topic_is_a_valid_question(self):
        question = (
            "When analyzing a dataset, how should a researcher decide whether an "
            "outlier should be removed or retained?"
        )
        self.assertEqual(MODULE.question_is_valid(question), (True, ""))

    def test_repair_accepts_nonempty_legacy_prompt_shape(self):
        legacy_prompt = (
            "Discuss several practical considerations when selecting a training "
            "dataset, including quality, coverage, and possible sources of bias."
        )
        self.assertEqual(MODULE.stored_prompt_is_valid(legacy_prompt), (True, ""))
        self.assertEqual(MODULE.stored_prompt_is_valid("  "), (False, "is empty"))

    def test_duplicate_model_job_lock_fails_fast(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "job.lock"
            with MODULE.exclusive_file_lock(lock_path):
                with self.assertRaises(SystemExit):
                    with MODULE.exclusive_file_lock(lock_path, wait=False):
                        self.fail("A duplicate job unexpectedly acquired the lock")

    def test_parallel_jobs_claim_disjoint_human_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "data"
            (data_dir / "human").mkdir(parents=True)
            samples = []
            for sample_id in range(1, 5):
                (data_dir / "human" / f"{sample_id}.txt").write_text(
                    "human source material " * 20, encoding="utf-8"
                )
                samples.append(
                    {
                        "id": sample_id,
                        "file": f"human/{sample_id}.txt",
                        "label": "human",
                        "word_count": 50,
                    }
                )
            metadata = {
                "counts": {"total": 4, "human": 4, "ai": 0},
                "samples": samples,
            }
            (data_dir / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
            first_args = ollama_args(data_dir, count=2)
            first_args.model = "model-a"
            second_args = ollama_args(data_dir, count=2)
            second_args.model = "model-b"

            first, _ = MODULE.claim_fresh_candidates(first_args)
            second, _ = MODULE.claim_fresh_candidates(second_args)
            first_ids = {int(sample["id"]) for sample in first}
            second_ids = {int(sample["id"]) for sample in second}

            self.assertEqual(len(first_ids), 2)
            self.assertEqual(len(second_ids), 2)
            self.assertFalse(first_ids & second_ids)
            self.assertEqual(first_ids | second_ids, {1, 2, 3, 4})

    def test_repair_workers_generate_concurrently_and_commit_serially(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "data"
            (data_dir / "ai").mkdir(parents=True)
            samples = []
            for sample_id in range(1, 5):
                question = (
                    f"How can concurrency example {sample_id} improve a network-bound "
                    "dataset repair job?"
                )
                (data_dir / "ai" / f"{sample_id}.txt").write_text(
                    question + "\n", encoding="utf-8"
                )
                samples.append(
                    {
                        "id": sample_id,
                        "file": f"ai/{sample_id}.txt",
                        "label": "ai",
                        "sample_type": "generated-question",
                        "target_response_words": 50,
                        "generator": {
                            "provider": "ollama",
                            "requested_model": "test-model",
                            "model": "test-model",
                        },
                    }
                )
            metadata = {
                "counts": {"total": 4, "human": 0, "ai": 4},
                "samples": samples,
            }
            (data_dir / "meta.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            args = ollama_args(data_dir, count=4)
            args.workers = 2
            active = 0
            maximum_active = 0
            activity_lock = threading.Lock()

            def delayed_response(*_arguments):
                nonlocal active, maximum_active
                with activity_lock:
                    active += 1
                    maximum_active = max(maximum_active, active)
                time.sleep(0.03)
                with activity_lock:
                    active -= 1
                response = " ".join(f"parallel{i}" for i in range(50))
                return MODULE.ModelResult(response, "test-model", {})

            with mock.patch.object(
                MODULE, "call_model", side_effect=delayed_response
            ):
                MODULE.repair_dataset(args)

            updated = json.loads(
                (data_dir / "meta.json").read_text(encoding="utf-8")
            )
            self.assertEqual(maximum_active, 2)
            self.assertEqual(
                [sample["sample_type"] for sample in updated["samples"]],
                ["generated-response"] * 4,
            )

    def test_parallel_repair_drains_in_flight_requests_after_interrupt(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "data"
            data_dir.mkdir(parents=True)
            args = ollama_args(data_dir, count=2)
            args.workers = 2
            jobs = [
                MODULE.RepairJob(
                    sample={"id": sample_id},
                    sample_id=sample_id,
                    target_words=50,
                    prompt_relative=f"prompts/{sample_id}.txt",
                    question=f"Question {sample_id}?",
                )
                for sample_id in (1, 2)
            ]
            activity_lock = threading.Lock()
            active = 0
            both_started = threading.Event()
            release_workers = threading.Event()

            def blocked_response(
                *_arguments,
            ):
                nonlocal active
                with activity_lock:
                    active += 1
                    if active == 2:
                        both_started.set()
                release_workers.wait(1.0)
                response = " ".join(f"interrupt{i}" for i in range(50))
                return response, MODULE.ModelResult(response, "test-model", {}), None

            real_wait = MODULE.wait
            wait_calls = 0

            def interrupt_first_wait(*arguments, **keywords):
                nonlocal wait_calls
                wait_calls += 1
                if wait_calls == 1:
                    self.assertTrue(both_started.wait(1.0))
                    release_workers.set()
                    raise KeyboardInterrupt
                return real_wait(*arguments, **keywords)

            with (
                mock.patch.object(MODULE, "generate_response", side_effect=blocked_response),
                mock.patch.object(
                    MODULE, "complete_repair_job", return_value="generated"
                ) as complete,
                mock.patch.object(MODULE, "wait", side_effect=interrupt_first_wait),
            ):
                completed, failures, interrupted = MODULE.run_repair_jobs(
                    args, data_dir, jobs, 0, 2
                )

            self.assertTrue(interrupted)
            self.assertEqual(completed, 2)
            self.assertEqual(failures, 0)
            self.assertEqual(complete.call_count, 2)

    def test_parallel_repairs_preserve_both_metadata_updates(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "data"
            (data_dir / "ai").mkdir(parents=True)
            samples = []
            for sample_id, model in ((1, "model-a"), (2, "model-b")):
                question = (
                    f"How can repair example {sample_id} verify concurrent metadata updates safely?"
                )
                (data_dir / "ai" / f"{sample_id}.txt").write_text(
                    question + "\n", encoding="utf-8"
                )
                samples.append(
                    {
                        "id": sample_id,
                        "file": f"ai/{sample_id}.txt",
                        "label": "ai",
                        "sample_type": "generated-question",
                        "target_response_words": 50,
                        "generator": {
                            "provider": "ollama",
                            "requested_model": model,
                            "model": model,
                        },
                    }
                )
            metadata = {
                "counts": {"total": 2, "human": 0, "ai": 2},
                "samples": samples,
            }
            (data_dir / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
            context = multiprocessing.get_context("fork")
            processes = [
                context.Process(
                    target=repair_worker,
                    args=(str(data_dir), "model-a", 1),
                ),
                context.Process(
                    target=repair_worker,
                    args=(str(data_dir), "model-b", 2),
                ),
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(10)
                self.assertEqual(process.exitcode, 0)

            updated = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(
                [sample["sample_type"] for sample in updated["samples"]],
                ["generated-response", "generated-response"],
            )
            for sample in updated["samples"]:
                self.assertEqual(sample["word_count"], 50)

    def test_parallel_commits_preserve_both_metadata_updates(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "data"
            (data_dir / "human").mkdir(parents=True)
            (data_dir / "ai").mkdir()
            samples = []
            for sample_id in (1, 2):
                (data_dir / "human" / f"{sample_id}.txt").write_text(
                    "human source material " * 20, encoding="utf-8"
                )
                samples.append(
                    {
                        "id": sample_id,
                        "file": f"human/{sample_id}.txt",
                        "label": "human",
                        "word_count": 50,
                    }
                )
            metadata = {
                "counts": {"total": 2, "human": 2, "ai": 0},
                "samples": samples,
            }
            (data_dir / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
            context = multiprocessing.get_context("fork")
            processes = [
                context.Process(
                    target=commit_worker,
                    args=(str(data_dir), "model-a", 1),
                ),
                context.Process(
                    target=commit_worker,
                    args=(str(data_dir), "model-b", 2),
                ),
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(10)
                self.assertEqual(process.exitcode, 0)

            updated = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
            ai_samples = [
                sample for sample in updated["samples"] if sample.get("label") == "ai"
            ]
            self.assertEqual(len(ai_samples), 2)
            self.assertEqual(len({sample["id"] for sample in ai_samples}), 2)
            self.assertEqual(len({sample["file"] for sample in ai_samples}), 2)
            self.assertEqual(updated["counts"], {"total": 4, "human": 2, "ai": 2})

    def test_fresh_generation_saves_answer_as_ai_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "data"
            (data_dir / "human").mkdir(parents=True)
            (data_dir / "human" / "1.txt").write_text(
                "human source material " * 30, encoding="utf-8"
            )
            metadata = {
                "counts": {"total": 1, "human": 1, "ai": 0},
                "samples": [
                    {
                        "id": 1,
                        "file": "human/1.txt",
                        "label": "human",
                        "collection": "test",
                        "source": "test-source",
                        "word_count": 50,
                        "public_hub_eligible": True,
                    }
                ],
            }
            (data_dir / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
            question = "How can a small software team improve code reviews without slowing delivery?"
            answer = " ".join(f"answer{i}" for i in range(50))
            results = [
                MODULE.ModelResult(question, "test-model", {"phase": "question"}),
                MODULE.ModelResult(answer, "test-model", {"phase": "answer"}),
            ]
            with mock.patch.object(MODULE, "call_model", side_effect=results):
                MODULE.fresh_dataset(ollama_args(data_dir, count=1))

            updated = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
            ai_sample = updated["samples"][1]
            ai_text = (data_dir / ai_sample["file"]).read_text(encoding="utf-8").strip()
            prompt_text = (data_dir / ai_sample["prompt_file"]).read_text(
                encoding="utf-8"
            ).strip()
            self.assertEqual(ai_sample["sample_type"], "generated-response")
            self.assertEqual(ai_text, answer)
            self.assertNotEqual(ai_text, question)
            self.assertEqual(prompt_text, question)
            self.assertEqual(updated["counts"], {"total": 2, "human": 1, "ai": 1})

    def test_repair_preserves_question_and_replaces_ai_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "data"
            (data_dir / "ai").mkdir(parents=True)
            question = "Why can careful batching improve training throughput on modern accelerators?"
            (data_dir / "ai" / "1.txt").write_text(question + "\n", encoding="utf-8")
            metadata = {
                "counts": {"total": 1, "human": 0, "ai": 1},
                "samples": [
                    {
                        "id": 1,
                        "file": "ai/1.txt",
                        "label": "ai",
                        "sample_type": "generated-question",
                        "target_response_words": 50,
                        "word_count": len(question.split()),
                        "generator": {
                            "provider": "ollama",
                            "requested_model": "test-model",
                            "model": "test-model",
                        },
                        "batch_usage": {"phase": "question"},
                        "created_at_utc": "2026-01-01T00:00:00+00:00",
                    }
                ],
            }
            (data_dir / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
            answer = " ".join(f"response{i}" for i in range(50))
            with mock.patch.object(
                MODULE,
                "call_model",
                return_value=MODULE.ModelResult(answer, "test-model", {"phase": "answer"}),
            ):
                MODULE.repair_dataset(ollama_args(data_dir, count=1))

            updated = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
            ai_sample = updated["samples"][0]
            ai_text = (data_dir / "ai" / "1.txt").read_text(encoding="utf-8").strip()
            prompt_text = (data_dir / ai_sample["prompt_file"]).read_text(
                encoding="utf-8"
            ).strip()
            self.assertEqual(ai_sample["sample_type"], "generated-response")
            self.assertEqual(ai_text, answer)
            self.assertEqual(prompt_text, question)
            self.assertEqual(ai_sample["prompt_batch_usage"], {"phase": "question"})
            self.assertEqual(ai_sample["batch_usage"], {"phase": "answer"})

    def test_repair_restores_invalid_interrupted_output_and_regenerates(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "data"
            (data_dir / "ai").mkdir(parents=True)
            args = ollama_args(data_dir, count=1)
            question = (
                "How can checkpointing make a long-running dataset generation job "
                "safer to resume?"
            )
            (data_dir / "ai" / "1.txt").write_text(
                "incomplete output from an interrupted request\n", encoding="utf-8"
            )
            prompt_relative = MODULE.prompt_relative_path(args, 1, recovered=True)
            MODULE.atomic_write_text(
                data_dir / prompt_relative, MODULE.text_for_file(question)
            )
            metadata = {
                "counts": {"total": 1, "human": 0, "ai": 1},
                "samples": [
                    {
                        "id": 1,
                        "file": "ai/1.txt",
                        "label": "ai",
                        "sample_type": "generated-question",
                        "target_response_words": 50,
                        "generator": {
                            "provider": "ollama",
                            "requested_model": "test-model",
                            "model": "test-model",
                        },
                    }
                ],
            }
            (data_dir / "meta.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            answer = " ".join(f"recovered{i}" for i in range(50))
            with mock.patch.object(
                MODULE,
                "call_model",
                return_value=MODULE.ModelResult(answer, "test-model", {}),
            ) as call:
                MODULE.repair_dataset(args)

            updated = json.loads(
                (data_dir / "meta.json").read_text(encoding="utf-8")
            )
            self.assertEqual(updated["samples"][0]["sample_type"], "generated-response")
            self.assertEqual(
                (data_dir / "ai" / "1.txt").read_text(encoding="utf-8").strip(),
                answer,
            )
            self.assertEqual(call.call_args.args[2], question)

    def test_openai_repair_switch_preserves_completed_source_responses(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "data"
            (data_dir / "ai").mkdir(parents=True)
            sol_response = " ".join(f"sol{i}" for i in range(50))
            question = (
                "How can a repair job switch models without replacing responses "
                "that already completed?"
            )
            (data_dir / "ai" / "1.txt").write_text(
                sol_response + "\n", encoding="utf-8"
            )
            (data_dir / "ai" / "2.txt").write_text(
                question + "\n", encoding="utf-8"
            )
            sol_generator = {
                "provider": "openai",
                "requested_model": "gpt-5.6-sol",
                "model": "gpt-5.6-sol",
                "model_selection": "sol-medium",
                "reasoning_effort": "medium",
            }
            metadata = {
                "counts": {"total": 2, "human": 0, "ai": 2},
                "samples": [
                    {
                        "id": 1,
                        "file": "ai/1.txt",
                        "label": "ai",
                        "sample_type": "generated-response",
                        "target_response_words": 50,
                        "word_count": 50,
                        "generator": sol_generator,
                        "prompt_generator": sol_generator,
                    },
                    {
                        "id": 2,
                        "file": "ai/2.txt",
                        "label": "ai",
                        "sample_type": "generated-question",
                        "target_response_words": 50,
                        "generator": sol_generator,
                    },
                ],
            }
            (data_dir / "meta.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            luna_response = " ".join(f"luna{i}" for i in range(50))
            args = openai_repair_args(data_dir, count=2)
            with mock.patch.object(
                MODULE,
                "call_model",
                return_value=MODULE.ModelResult(
                    luna_response, "gpt-5.6-luna", {}
                ),
            ) as call:
                MODULE.repair_dataset(args)

            updated = json.loads(
                (data_dir / "meta.json").read_text(encoding="utf-8")
            )
            first, second = updated["samples"]
            self.assertEqual(
                (data_dir / "ai" / "1.txt").read_text(encoding="utf-8").strip(),
                sol_response,
            )
            self.assertEqual(first["generator"]["model_selection"], "sol-medium")
            self.assertEqual(
                (data_dir / "ai" / "2.txt").read_text(encoding="utf-8").strip(),
                luna_response,
            )
            self.assertEqual(second["generator"]["model_selection"], "luna-medium")
            self.assertEqual(
                second["prompt_generator"]["requested_model"], "gpt-5.6-sol"
            )
            self.assertIn("openai-gpt-5.6-sol", second["prompt_file"])
            self.assertEqual(call.call_count, 1)

    def test_biology_prompt_is_replaced_and_preserved_during_repair(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "data"
            (data_dir / "ai").mkdir(parents=True)
            question = (
                "How do proteins and enzymes interact during a biological process?"
            )
            (data_dir / "ai" / "1.txt").write_text(
                question + "\n", encoding="utf-8"
            )
            sol_generator = {
                "provider": "openai",
                "requested_model": "gpt-5.6-sol",
                "model": "gpt-5.6-sol",
                "model_selection": "sol-medium",
                "reasoning_effort": "medium",
            }
            metadata = {
                "counts": {"total": 1, "human": 0, "ai": 1},
                "samples": [
                    {
                        "id": 1,
                        "file": "ai/1.txt",
                        "label": "ai",
                        "sample_type": "generated-question",
                        "target_response_words": 50,
                        "generator": sol_generator,
                    }
                ],
            }
            (data_dir / "meta.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            response = " ".join(f"safe{i}" for i in range(50))
            args = openai_repair_args(data_dir, count=1)
            with mock.patch.object(
                MODULE,
                "call_model",
                return_value=MODULE.ModelResult(
                    response, "gpt-5.6-luna", {}
                ),
            ) as call:
                MODULE.repair_dataset(args)

            sent_question = call.call_args.args[2]
            self.assertNotEqual(sent_question, question)
            self.assertIsNone(MODULE.BIOLOGY_PATTERN.search(sent_question))
            updated = json.loads(
                (data_dir / "meta.json").read_text(encoding="utf-8")
            )
            sample = updated["samples"][0]
            replacement = sample["prompt_replacement"]
            self.assertEqual(replacement["reason"], "biology-topic-prefilter")
            self.assertEqual(
                (data_dir / replacement["original_prompt_file"])
                .read_text(encoding="utf-8")
                .strip(),
                question,
            )
            self.assertEqual(
                (data_dir / replacement["replacement_prompt_file"])
                .read_text(encoding="utf-8")
                .strip(),
                sent_question,
            )
            self.assertEqual(sample["prompt_file"], replacement["replacement_prompt_file"])

    def test_openai_bio_policy_error_retries_with_safe_prompt(self):
        args = openai_repair_args(Path("/tmp/test-data"), count=1)
        question = (
            "How should a project team investigate an unexpected technical result?"
        )
        response = " ".join(f"fallback{i}" for i in range(50))
        results = [
            MODULE.GenerationError("policy block", code="bio_policy"),
            MODULE.ModelResult(response, "gpt-5.6-luna", {}),
        ]
        with mock.patch.object(MODULE, "call_model", side_effect=results) as call:
            generated, result, error = MODULE.generate_response(
                args, question, 50, 123
            )

        self.assertIsNone(error)
        self.assertEqual(generated, response)
        self.assertIsNotNone(result)
        self.assertEqual(
            result.usage["prompt_replacement"]["reason"], "openai-bio-policy"
        )
        self.assertEqual(call.call_args_list[0].args[2], question)
        self.assertNotEqual(call.call_args_list[1].args[2], question)

    def test_unusable_prompt_does_not_block_later_repair_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "data"
            (data_dir / "ai").mkdir(parents=True)
            (data_dir / "ai" / "1.txt").write_text("\n", encoding="utf-8")
            question = (
                "Why should one malformed sample not terminate a long-running "
                "dataset repair job?"
            )
            (data_dir / "ai" / "2.txt").write_text(
                question + "\n", encoding="utf-8"
            )
            samples = []
            for sample_id in (1, 2):
                samples.append(
                    {
                        "id": sample_id,
                        "file": f"ai/{sample_id}.txt",
                        "label": "ai",
                        "sample_type": "generated-question",
                        "target_response_words": 50,
                        "generator": {
                            "provider": "ollama",
                            "requested_model": "test-model",
                            "model": "test-model",
                        },
                    }
                )
            metadata = {
                "counts": {"total": 2, "human": 0, "ai": 2},
                "samples": samples,
            }
            (data_dir / "meta.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            answer = " ".join(f"continued{i}" for i in range(50))
            with mock.patch.object(
                MODULE,
                "call_model",
                return_value=MODULE.ModelResult(answer, "test-model", {}),
            ) as call:
                with self.assertRaisesRegex(SystemExit, "1 failures"):
                    MODULE.repair_dataset(ollama_args(data_dir, count=2))

            updated = json.loads(
                (data_dir / "meta.json").read_text(encoding="utf-8")
            )
            self.assertEqual(updated["samples"][0]["sample_type"], "generated-question")
            self.assertEqual(updated["samples"][1]["sample_type"], "generated-response")
            self.assertEqual(call.call_count, 1)
            failure = json.loads(
                (data_dir / "generation-failures.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertEqual(failure["sample_id"], 1)
            self.assertIn("is empty", failure["error"])


if __name__ == "__main__":
    unittest.main()
