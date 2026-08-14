
import importlib.util
import json
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_DIR / "scripts" / "14_resource-latency"
MODULE_PATH = SCRIPT_DIR / "_benchmark_model.py"
PLOT_MODULE_PATH = SCRIPT_DIR / "plot_resource_latency.py"


def load_module():
    spec = importlib.util.spec_from_file_location("resource_benchmark", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_plot_module():
    spec = importlib.util.spec_from_file_location(
        "resource_benchmark_plot", PLOT_MODULE_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_summary_reports_latency_throughput_and_gpu_usage():
    module = load_module()
    result = module.summarize_measurements(
        model_name="distilbert",
        description="Fully fine-tuned DistilBERT",
        device_name="cuda",
        sample_count=100,
        batch_size=8,
        elapsed_times=[2.0, 2.0],
        gpu_samples=[
            module.GpuSample(50.0, 1_000.0),
            module.GpuSample(70.0, 1_100.0),
        ],
        peak_allocated_mb=900.0,
        peak_reserved_mb=950.0,
    )

    assert result["mean_latency_seconds"] == 2.0
    assert result["mean_latency_ms_per_text"] == 20.0
    assert result["mean_texts_per_second"] == 50.0
    assert result["gpu_utilization_mean_percent"] == 60.0
    assert result["gpu_utilization_peak_percent"] == 70.0
    assert result["gpu_memory_used_peak_mb"] == 1_100.0


def test_every_registered_model_has_an_entry_point():
    expected = {
        "benchmark_logreg.py": "logreg",
        "benchmark_distilbert.py": "distilbert",
        "benchmark_distilbert_lora.py": "distilbert-lora",
        "benchmark_distilbert_mica.py": "distilbert-mica",
        "benchmark_modernbert.py": "modernbert",
        "benchmark_gpt2_fixed.py": "gpt2-fixed",
        "benchmark_gpt2_variable.py": "gpt2-variable",
        "benchmark_qwen3_fixed.py": "qwen3-fixed",
        "benchmark_qwen3_variable.py": "qwen3-variable",
    }
    for filename, model_name in expected.items():
        source = (SCRIPT_DIR / filename).read_text(encoding="utf-8")
        assert f'main_for_model("{model_name}")' in source


def test_notebook_calls_every_entry_point_and_has_no_stored_outputs():
    notebook_path = SCRIPT_DIR / "notebooks" / "resource-latency.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    for script_path in SCRIPT_DIR.glob("benchmark_*.py"):
        assert script_path.name in source

    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        cell_source = "".join(cell.get("source", []))
        compile(cell_source, f"{notebook_path.name}:{cell['id']}", "exec")
        assert cell["execution_count"] is None
        assert cell["outputs"] == []


def test_plot_uses_requested_models_and_marks_cpu_memory_as_missing():
    module = load_plot_module()
    results = module.load_results(
        SCRIPT_DIR / "results" / "resource-latency-results.csv"
    )

    assert [result.model for result in results] == list(module.MODEL_ORDER)
    assert results[0].model == "logreg"
    assert results[0].gpu_memory_allocated_peak_mb is None
    assert all(
        result.gpu_memory_allocated_peak_mb is not None for result in results[1:]
    )


def test_resource_plot_is_horizontal_and_uses_log_throughput_axis():
    module = load_plot_module()
    results = module.load_results(
        SCRIPT_DIR / "results" / "resource-latency-results.csv"
    )
    figure = module.create_figure(results)

    assert figure.get_figwidth() > figure.get_figheight()
    assert len(figure.axes) == 2
    assert figure.axes[0].get_xscale() == "log"

    module.plt.close(figure)
