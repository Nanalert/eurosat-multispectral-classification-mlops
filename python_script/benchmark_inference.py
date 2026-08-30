"""Measure production model inference speed and save the result."""

import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from python_script.inference import load_model


IMAGE_PATH = (
    PROJECT_ROOT / "data" / "raw" / "EuroSAT_RGB" / "2750" / "River" / "River_100.jpg"
)
OUTPUT_PATH = PROJECT_ROOT / "reports" / "inference_benchmark.json"
DEVICE = "cpu"
WARMUP_RUNS = 3
MEASURED_RUNS = 20


def benchmark(
    image_path=IMAGE_PATH,
    output_path=OUTPUT_PATH,
    device=DEVICE,
    warmup_runs=WARMUP_RUNS,
    measured_runs=MEASURED_RUNS,
):
    predictor = load_model(device=device)
    for _ in range(warmup_runs):
        predictor.predict(image_path)

    durations_ms = []
    for _ in range(measured_runs):
        started = perf_counter()
        predictor.predict(image_path)
        durations_ms.append((perf_counter() - started) * 1000)

    result = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": predictor.model_version,
        "device": str(predictor.device),
        "image_path": str(Path(image_path).resolve()),
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "mean_ms": statistics.mean(durations_ms),
        "median_ms": statistics.median(durations_ms),
        "minimum_ms": min(durations_ms),
        "maximum_ms": max(durations_ms),
        "images_per_second": 1000 / statistics.mean(durations_ms),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    result = benchmark()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
