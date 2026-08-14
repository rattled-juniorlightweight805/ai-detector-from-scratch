#!/usr/bin/env python3
"""Shared resource and latency benchmark for local AI-text classifiers."""

import argparse
import json
import shutil
import statistics
import subprocess
import threading
import time
from collections import namedtuple
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
DEFAULT_DATASET_REPO = "rasbt/human-vs-ai-50k"

from ai_detector import load_classifier, model_registry


GpuSample = namedtuple("GpuSample", "utilization_percent memory_used_mb")


class NvidiaSmiSampler:
    """Sample whole-device NVIDIA utilization while inference runs."""

    def __init__(self, *, device_index, interval):
        self.device_index = device_index
        self.interval = interval
        self.samples = []
        self._stop_event = threading.Event()
        self._thread = None

    def _query(self):
        command = [
            "nvidia-smi",
            f"--id={self.device_index}",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            first_line = completed.stdout.strip().splitlines()[0]
            utilization, memory_used = first_line.split(",", maxsplit=1)
            return GpuSample(
                utilization_percent=float(utilization.strip()),
                memory_used_mb=float(memory_used.strip()),
            )
        except (
            FileNotFoundError,
            IndexError,
            ValueError,
            subprocess.SubprocessError,
        ):
            return None

    def _sample_until_stopped(self):
        while not self._stop_event.is_set():
            sample = self._query()
            if sample is not None:
                self.samples.append(sample)
            self._stop_event.wait(self.interval)

    def start(self):
        self.samples.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._sample_until_stopped,
            name="nvidia-smi-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval * 2))
        return list(self.samples)


def parse_args(model_name):
    parser = argparse.ArgumentParser(
        description=(
            f"Benchmark {model_name} on the held-out test split. Model and "
            "dataset loading are excluded from the measurements."
        )
    )
    parser.add_argument(
        "--dataset-repo",
        default=DEFAULT_DATASET_REPO,
        help=f"Hugging Face dataset repository (default: {DEFAULT_DATASET_REPO})",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        help="Optional local DatasetDict directory instead of the Hub dataset",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        help="Override the classifier's default artifact path",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Inference device (default: auto)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference batch size (default: 8)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Measured full-test passes (default: 5)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Benchmark only the first N test texts for a quick check",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.1,
        help="Seconds between nvidia-smi samples (default: 0.1)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print only one JSON object for programmatic use",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.sample_interval <= 0:
        parser.error("--sample-interval must be positive")
    return args


def load_test_texts(args):
    if args.dataset_path is not None:
        from datasets import load_from_disk

        dataset_path = args.dataset_path.expanduser().resolve()
        if not dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        test_dataset = load_from_disk(str(dataset_path))["test"]
    else:
        from datasets import load_dataset

        test_dataset = load_dataset(args.dataset_repo, split="test")

    texts = list(test_dataset["text"])
    if args.limit is not None:
        texts = texts[: args.limit]
    if not texts:
        raise ValueError("The selected test split is empty")
    return texts


def torch_device_for(classifier):
    return getattr(classifier, "device", None)


def synchronize_device(device):
    if device is None:
        return
    import torch

    device_type = getattr(device, "type", str(device).split(":", maxsplit=1)[0])
    if device_type == "cuda":
        torch.cuda.synchronize(device)
    elif device_type == "mps":
        torch.mps.synchronize()


def reset_cuda_peak_memory(device):
    if device is None or getattr(device, "type", None) != "cuda":
        return
    import torch

    torch.cuda.reset_peak_memory_stats(device)


def cuda_peak_memory(device):
    if device is None or getattr(device, "type", None) != "cuda":
        return None, None
    import torch

    bytes_per_mb = 1024**2
    allocated = torch.cuda.max_memory_allocated(device) / bytes_per_mb
    reserved = torch.cuda.max_memory_reserved(device) / bytes_per_mb
    return allocated, reserved


def cuda_device_index(device):
    if device is None or getattr(device, "type", None) != "cuda":
        return None
    import torch

    if device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


def summarize_measurements(
    *,
    model_name,
    description,
    device_name,
    sample_count,
    batch_size,
    elapsed_times,
    gpu_samples,
    peak_allocated_mb,
    peak_reserved_mb,
):
    mean_seconds = statistics.fmean(elapsed_times)
    std_seconds = (
        statistics.stdev(elapsed_times) if len(elapsed_times) > 1 else 0.0
    )
    throughputs = [sample_count / elapsed for elapsed in elapsed_times]
    mean_throughput = statistics.fmean(throughputs)
    std_throughput = (
        statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0
    )
    utilizations = [sample.utilization_percent for sample in gpu_samples]
    memory_used = [sample.memory_used_mb for sample in gpu_samples]

    return {
        "model": model_name,
        "description": description,
        "device": device_name,
        "samples": sample_count,
        "batch_size": batch_size,
        "repeats": len(elapsed_times),
        "mean_latency_seconds": mean_seconds,
        "std_latency_seconds": std_seconds,
        "mean_latency_ms_per_text": mean_seconds / sample_count * 1_000,
        "std_latency_ms_per_text": std_seconds / sample_count * 1_000,
        "mean_texts_per_second": mean_throughput,
        "std_texts_per_second": std_throughput,
        "gpu_utilization_mean_percent": (
            statistics.fmean(utilizations) if utilizations else None
        ),
        "gpu_utilization_peak_percent": max(utilizations) if utilizations else None,
        "gpu_memory_used_peak_mb": max(memory_used) if memory_used else None,
        "gpu_memory_allocated_peak_mb": peak_allocated_mb,
        "gpu_memory_reserved_peak_mb": peak_reserved_mb,
    }


def benchmark(model_name, args):
    registry = model_registry()
    spec = registry[model_name]
    texts = load_test_texts(args)
    classifier = load_classifier(
        model_name,
        artifact_path=args.artifact,
        device=args.device,
    )
    device = torch_device_for(classifier)
    device_name = str(device) if device is not None else "cpu"

    warmup_texts = texts[: min(args.batch_size, len(texts))]
    classifier.score_many(warmup_texts, batch_size=args.batch_size)
    synchronize_device(device)

    elapsed_times = []
    gpu_samples = []
    peak_allocated_values = []
    peak_reserved_values = []
    gpu_index = cuda_device_index(device)

    for _ in range(args.repeats):
        reset_cuda_peak_memory(device)
        sampler = (
            NvidiaSmiSampler(
                device_index=gpu_index,
                interval=args.sample_interval,
            )
            if gpu_index is not None and shutil.which("nvidia-smi") is not None
            else None
        )
        if sampler is not None:
            sampler.start()

        synchronize_device(device)
        start_time = time.perf_counter()
        classifier.score_many(texts, batch_size=args.batch_size)
        synchronize_device(device)
        elapsed_times.append(time.perf_counter() - start_time)

        if sampler is not None:
            gpu_samples.extend(sampler.stop())
        allocated_mb, reserved_mb = cuda_peak_memory(device)
        if allocated_mb is not None:
            peak_allocated_values.append(allocated_mb)
        if reserved_mb is not None:
            peak_reserved_values.append(reserved_mb)

    return summarize_measurements(
        model_name=model_name,
        description=spec.description,
        device_name=device_name,
        sample_count=len(texts),
        batch_size=args.batch_size,
        elapsed_times=elapsed_times,
        gpu_samples=gpu_samples,
        peak_allocated_mb=(
            max(peak_allocated_values) if peak_allocated_values else None
        ),
        peak_reserved_mb=(
            max(peak_reserved_values) if peak_reserved_values else None
        ),
    )


def format_optional(value, *, suffix = "", digits = 1):
    if value is None:
        return "n/a"
    return f"{float(value):,.{digits}f}{suffix}"


def print_human_readable(result):
    print(f"Model: {result['model']} ({result['description']})")
    print(f"Device: {result['device']}")
    print(
        f"Samples: {int(result['samples']):,}; "
        f"batch size: {result['batch_size']}; repeats: {result['repeats']}"
    )
    print(
        "Latency, full test pass: "
        f"{float(result['mean_latency_seconds']):.4f} +/- "
        f"{float(result['std_latency_seconds']):.4f} s"
    )
    print(
        "Latency per text: "
        f"{float(result['mean_latency_ms_per_text']):.4f} +/- "
        f"{float(result['std_latency_ms_per_text']):.4f} ms"
    )
    print(
        "Throughput: "
        f"{float(result['mean_texts_per_second']):,.1f} +/- "
        f"{float(result['std_texts_per_second']):,.1f} texts/s"
    )
    print(
        "GPU utilization, mean / peak: "
        f"{format_optional(result['gpu_utilization_mean_percent'], suffix='%')} / "
        f"{format_optional(result['gpu_utilization_peak_percent'], suffix='%')}"
    )
    print(
        "GPU memory, process allocated / process reserved / device used: "
        f"{format_optional(result['gpu_memory_allocated_peak_mb'], suffix=' MB')} / "
        f"{format_optional(result['gpu_memory_reserved_peak_mb'], suffix=' MB')} / "
        f"{format_optional(result['gpu_memory_used_peak_mb'], suffix=' MB')}"
    )


def main_for_model(model_name):
    if model_name not in model_registry():
        raise ValueError(f"Unknown model: {model_name}")
    args = parse_args(model_name)
    result = benchmark(model_name, args)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print_human_readable(result)
